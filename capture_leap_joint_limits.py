#!/usr/bin/env python3
"""Measure preferred LEAP joint travel with a temporary single-joint hold.

Position the prompted joint with torque off. Terminal Enter holds just that
joint at its measured position, waits for stable feedback, captures the endpoint
and releases it. `joint NAME lower|upper` revisits an endpoint; `cancel` releases
an unfinished hold. `save` writes all 32 endpoints into a separate hand XML.
Progress resumes on rerun. No joint is automatically driven toward a stop.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import re
import signal
import sys
import tempfile
import threading
import time

import mujoco
import numpy as np

from ContactModelStudy.Hardware.Hand import HardwareHand, MOTORS, check_device_feedback
from ContactModelStudy.Hardware.Scene import JOINTS, ROOT
from capture_leap_joint_reference import ensure_port_free

SOURCE = ROOT / 'scenes/leap/hardware/calibrated_hand.xml'
OUTPUT = SOURCE.with_name('calibrated_hand_measured_limits.xml')
STATE = SOURCE.with_name('measured_joint_limits_capture.json')
ENDPOINTS = [(name, side) for name in JOINTS for side in ('lower', 'upper')]


class TransientReadError(OSError):
    """A timeout/corrupt reply, distinguishable from a device fault."""


class CaptureHand(HardwareHand):
    """Bounded serial-read retries for this interactive tool only; never retry writes."""
    def _require_success(self, comm_result, dxl_error=None, dxl_id=None, context=None):
        if (comm_result in (self.client.dxl.COMM_RX_TIMEOUT, self.client.dxl.COMM_RX_CORRUPT)
                and dxl_error in (None, 0)):
            raise TransientReadError(f'Dynamixel receive failure: {comm_result}, motor={dxl_id}, {context}')
        return super()._require_success(comm_result, dxl_error, dxl_id, context)

    def _retry_read(self, operation, description):
        for attempt in range(3):
            try:
                return operation()
            except TransientReadError:
                if attempt == 2:
                    raise
                # All serial work is synchronous on this thread; the failed
                # transaction has returned. Discard its late/incomplete reply.
                self.client.port_handler.clearPort()
                print(f'{description}: receive error; retry {attempt+1}/2', flush=True)
                time.sleep(.005)

    def register(self, motor, address, size):
        return self._retry_read(lambda: super(CaptureHand, self).register(motor, address, size),
                                f'Motor {motor} register {address}')

    def read(self):
        return self._retry_read(super().read, 'Joint feedback')


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as f:
        temporary = Path(f.name)
        f.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ranges_xml(source_text, ranges):
    """Override named joints only; preserve actuator ranges and all dynamics."""
    def replace(match):
        tag = match.group(0)
        name = re.search(r'\bname="([^"]+)"', tag)
        if not name or name[1] not in ranges:
            return tag
        low, high = ranges[name[1]]
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f'{name[1]}: lower must be strictly below upper')
        tag = re.sub(r'\s+(?:range|limited)="[^"]*"', '', tag)
        return tag[:-2].rstrip() + f' limited="true" range="{low:.12g} {high:.12g}" />'
    result = re.sub(r'<joint\b[^>]*?/>', replace, source_text)
    for name in ranges:
        if len(re.findall(r'<joint\b[^>]*\bname="' + re.escape(name) + r'"', result)) != 1:
            raise ValueError(f'Expected exactly one named joint: {name}')
    return result


class LimitCapture:
    def __init__(self, source=SOURCE, output=OUTPUT, state=STATE):
        self.source, self.output, self.path = map(Path, (source, output, state))
        paths = [p.resolve() for p in (self.source, self.output, self.path)]
        if len(set(paths)) != 3:
            raise ValueError('Source, output and capture record must be different files')
        self.original = self.source.read_bytes()
        if self.path.exists():
            self.record = json.loads(self.path.read_text())
            if (self.record.get('schema') != 1 or self.record['source_sha256'] != sha(self.original)
                    or self.record['joint_order'] != list(JOINTS)
                    or self.record['output_path'] != str(self.output.resolve())):
                raise ValueError('Capture provenance does not match source/output/joint order')
            if sha(self.output.read_bytes()) != self.record['output_sha256']:
                raise ValueError('Output XML changed externally; refusing to overwrite it')
        else:
            # Exclusive creation prevents overwriting an existing model without its record.
            with self.output.open('xb') as f:
                f.write(self.original)
            self.record = dict(schema=1, created_at=utc(), source_path=str(self.source.resolve()),
                source_sha256=sha(self.original), output_path=str(self.output.resolve()),
                output_sha256=sha(self.original), joint_order=list(JOINTS), captures={}, history=[],
                xml_status='uncaptured_copy', semantics='manually selected usable joint travel, radians',
                actuator_ranges='unchanged from source',
                torque_policy='only selected joint held temporarily at measured position on Enter')
            self.persist()
        self.selected = self.first_missing()

    def persist(self):
        self.record['updated_at'] = utc()
        atomic_write(self.path, (json.dumps(self.record, indent=2, allow_nan=False)+'\n').encode())

    def first_missing(self):
        return next(((j, s) for j, s in ENDPOINTS
                     if s not in self.record['captures'].get(j, {})), None)

    def capture(self, samples, max_velocity, max_spread, window, hold=None):
        if self.selected is None:
            raise ValueError('All endpoints captured; use save, or joint NAME lower|upper to redo')
        name, side = self.selected
        i = JOINTS.index(name)
        if len(samples) < 5 or samples[-1][0]-samples[0][0] < window:
            raise ValueError('Need a complete stable measurement window; hold still and retry')
        q = np.array([row[1] for row in samples])
        v = np.array([row[2] for row in samples])
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError('Nonfinite measurements')
        if np.max(np.abs(v[:, i])) > max_velocity or np.ptp(q[:, i]) > max_spread:
            raise ValueError(f'{name} is moving; hold it still and retry')
        value = float(np.median(q[:, i]))
        previous = self.record['captures'].get(name, {})
        other = previous.get('upper' if side == 'lower' else 'lower')
        if other and ((side == 'lower' and value >= other['value_rad'])
                      or (side == 'upper' and value <= other['value_rad'])):
            raise ValueError('Lower must be below upper in the displayed encoder coordinates')
        item = dict(value_rad=value, value_deg=float(np.rad2deg(value)), captured_at=utc(),
                    sample_span_s=samples[-1][0]-samples[0][0], sample_count=len(samples),
                    q_rad=q[:, i].tolist(), velocity_rad_s=v[:, i].tolist(),
                    monotonic_s=[r[0] for r in samples], median_full_hand_q_rad=np.median(q, axis=0).tolist())
        if hold is not None:
            item['hold'] = hold
        self.record['captures'].setdefault(name, {})[side] = item
        self.record['history'].append(dict(joint=name, endpoint=side, **item))
        self.record['xml_status'] = 'captures_pending_save'
        self.persist()
        self.selected = self.first_missing()
        return name, side, value

    def save(self):
        if self.first_missing() is not None:
            raise ValueError('Capture lower and upper for every joint before saving XML')
        if self.source.read_bytes() != self.original:
            raise ValueError('Source model changed during capture')
        if sha(self.output.read_bytes()) != self.record['output_sha256']:
            raise ValueError('Output XML changed externally; refusing to overwrite it')
        ranges = {j: [self.record['captures'][j][s]['value_rad'] for s in ('lower', 'upper')]
                  for j in JOINTS}
        payload = ranges_xml(self.original.decode(), ranges).encode()
        # Compile beside the destination so relative mesh assets resolve normally.
        with tempfile.NamedTemporaryFile(dir=self.output.parent, suffix='.xml', delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
        try:
            updated = mujoco.MjModel.from_xml_path(str(temporary))
            for j, bounds in ranges.items():
                np.testing.assert_allclose(updated.joint(j).range, bounds, atol=1e-11, rtol=1e-11)
            temporary.replace(self.output)
        finally:
            temporary.unlink(missing_ok=True)
        self.record.update(output_sha256=sha(payload), xml_status='all_16_joint_ranges_saved',
                           saved_at=utc(), ranges_rad=ranges)
        self.persist()

    def prompt(self):
        count = sum(len(x) for x in self.record['captures'].values())
        if self.selected is None:
            return f'{count}/32 captured. Type save to write XML; joint NAME lower|upper to redo.'
        name, side = self.selected
        return (f'{count}/32 captured. Position {name} at its {side.upper()} bound; Enter holds/measures/releases it. '
                '(help for commands)')


class EndpointHold:
    """One static servo target; collect settled feedback without manual steady holding."""
    def __init__(self, hand, session, window, max_velocity, max_spread,
                 settle_time=.25, timeout=10., max_error=.1):
        self.hand, self.session = hand, session
        self.window, self.max_velocity, self.max_spread = window, max_velocity, max_spread
        self.settle_time, self.timeout, self.max_error = settle_time, timeout, max_error
        self.motor = None
        self.history = deque()

    def start(self, q, now):
        if self.motor is not None:
            raise ValueError('Selected joint is settling; wait, or type cancel to release it')
        if self.session.selected is None:
            raise ValueError('All endpoints captured; type save or select a joint to redo')
        self.name, self.side = self.session.selected
        self.index = JOINTS.index(self.name)
        # MuJoCo indices and native motor IDs differ for MCP/rotation joints.
        mapping = np.asarray(self.hand.mapping.map_mujoco_to_leap(np.arange(16)))
        if mapping.shape != (16,) or not np.array_equal(np.sort(mapping), np.arange(16)):
            raise ValueError('Invalid MuJoCo-to-native motor mapping')
        self.motor = int(np.flatnonzero(mapping == self.index)[0])
        self.target = float(q[self.index])
        self.started = now
        self.history.clear()
        try:
            # Preloads only this motor before enabling it, with torque readback.
            # Dynamixel holds its stored goal; repeated target writes are unnecessary.
            self.hand.enable_motors(q, [self.motor])
        except BaseException:
            self.release()
            raise

    def release(self):
        if self.motor is not None:
            self.hand.disable_motors([self.motor])
            self.motor = None
        self.history.clear()

    def observe(self, q, v, now):
        if self.motor is None:
            return None
        if abs(q[self.index]-self.target) > self.max_error:
            self.release()
            return 'Released: selected joint moved too far from its held target. Reposition and Enter again.'
        if now-self.started > self.timeout:
            self.release()
            return 'Released: stable feedback not reached before timeout. Reposition and Enter again.'
        if now-self.started < self.settle_time:
            return None
        if abs(v[self.index]) > self.max_velocity:
            self.history.clear()
            return None
        self.history.append((now, q.copy(), v.copy()))
        while len(self.history) > 1 and self.history[1][0] <= now-self.window:
            self.history.popleft()
        if (len(self.history) < 5 or now-self.history[0][0] < self.window
                or np.ptp([r[1][self.index] for r in self.history]) > self.max_spread):
            return None
        metadata = dict(native_motor_id=self.motor, target_rad=self.target,
                        elapsed_s=now-self.started, settle_time_s=self.settle_time,
                        window_s=self.window, max_velocity_rad_s=self.max_velocity,
                        max_spread_rad=self.max_spread, max_hold_error_rad=self.max_error)
        try:
            name, side, value = self.session.capture(list(self.history), self.max_velocity,
                                                    self.max_spread, self.window, hold=metadata)
        except ValueError as exc:
            return f'Not captured: {exc}. Reposition and Enter again.'
        finally:
            self.release()
        return f'Captured {name} {side}: {value:.6f} rad ({np.rad2deg(value):.2f} deg). Released; torque OFF.'


def run(hand, model, config, session, stopped, rate, window, max_velocity, max_spread,
        settle_time=.25, hold_timeout=10., max_hold_error=.1):
    hold = EndpointHold(hand, session, window, max_velocity, max_spread,
                        settle_time, hold_timeout, max_hold_error)
    try:
        _run(hand, model, config, session, stopped, rate, hold)
    finally:
        hold.release()


def _run(hand, model, config, session, stopped, rate, hold):
    import mujoco.viewer
    # Leave gains unchanged. Start with torque OFF, before reading or opening the viewer.
    hand.connect_read_only()
    hand.disable_motors(MOTORS)
    q, v, current = hand.read()
    check_device_feedback(q, v, current, config)
    events = SimpleQueue()

    def terminal():
        while not stopped.is_set():
            try:
                events.put(input().strip())
            except EOFError:
                events.put('quit')
                return

    threading.Thread(target=terminal, daemon=True).start()
    data = mujoco.MjData(model)
    print('All torque OFF. Move joints gently by hand; do not force mechanical stops.', flush=True)
    print('Enter: only selected joint holds its current angle, settles, captures and releases. cancel: release now.', flush=True)
    print('Use the TERMINAL for commands. The viewer remains mouse-interactive.', flush=True)
    print(session.prompt(), flush=True)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .45, 130, -35
        last_health = 0.
        health_index = 0
        message = ''
        while viewer.is_running() and not stopped.is_set():
            start = time.monotonic()
            q, v, current = hand.read()
            now = time.monotonic()
            check_device_feedback(q, v, current, config)
            # One register transaction per health interval, rather than two.
            # Expected torque is ON only for the explicitly held native motor.
            if now-last_health >= .1:
                motor, stage = (health_index//3) % 16, health_index % 3
                if stage == 0:
                    if hand.register(motor, 64, 1) != int(motor == hold.motor):
                        raise RuntimeError(f'Motor {motor}: unexpected torque state')
                else:
                    hand.health_register(motor, stage-1)
                health_index += 1
                last_health = now
            try:
                command = events.get_nowait()
            except Empty:
                command = None
            if command is not None:
                try:
                    parts = command.split()
                    if command in ('quit', 'q', 'exit'):
                        stopped.set()
                        hold.release()
                    elif command == 'cancel':
                        hold.release()
                        message = 'Released; torque OFF. Position selected joint and Enter to retry.'
                    elif command == 'save':
                        if hold.motor is not None:
                            raise ValueError('Wait for capture to finish, or cancel before saving')
                        session.save()
                        model.jnt_range[:] = [session.record['ranges_rad'][j] for j in JOINTS]
                        message = f'SAVED {session.output}; torque remains OFF. Type quit to exit.'
                    elif len(parts) == 3 and parts[0] == 'joint' and tuple(parts[1:]) in ENDPOINTS:
                        hold.release()
                        session.selected = tuple(parts[1:])
                        message = 'Selected endpoint for capture/recapture.'
                    elif command == '':
                        # This cycle's freshly read pose is the preload target.
                        hold.start(q, time.monotonic())
                        message = f'Holding {hold.name} only. Release your grip; measuring after it settles...'
                    elif command == 'help':
                        message = ('Enter: hold, settle, capture, release | cancel: release now | joint NAME lower|upper: select/redo | '
                                   'status: show all captures | save: write completed XML | quit: exit')
                    elif command == 'status':
                        message = json.dumps({j: {s: x['value_rad'] for s, x in bounds.items()}
                                              for j, bounds in session.record['captures'].items()}, indent=2)
                    else:
                        raise ValueError('Unknown command; type help. Joint names: '+', '.join(JOINTS))
                except ValueError as exc:
                    message = str(exc)
                print(message+'\n'+session.prompt(), flush=True)
            result = hold.observe(q, v, now) if not stopped.is_set() else None
            if result:
                message = result
                print(message+'\n'+session.prompt(), flush=True)
            with viewer.lock():
                data.qpos[:], data.qvel[:], data.ctrl[:] = q, v, q
                data.act[:] = q
                mujoco.mj_forward(model, data)  # display only; never mj_step
            selected = session.selected[0] if session.selected else None
            angle = '' if selected is None else f'{selected}: {q[JOINTS.index(selected)]:+.5f} rad / {np.rad2deg(q[JOINTS.index(selected)]):+.2f} deg'
            torque = 'TORQUE OFF' if hold.motor is None else f'HOLDING {hold.name} only | cancel: release'
            viewer.set_texts((None, None, torque+' | TERMINAL commands\n'+session.prompt()+'\n'+angle+'\n'+message, None))
            viewer.sync()
            time.sleep(max(0., 1/rate-(time.monotonic()-start)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true', help='create isolated copy/record without USB access')
    parser.add_argument('--hardware-api', type=Path, default=Path.home()/'leap_agentic_sysid/third_party/LEAP_Hand_Hardware')
    parser.add_argument('--hardware-config', type=Path, default=ROOT/'configs/leap_hardware.json')
    parser.add_argument('--port')
    parser.add_argument('--rate', type=float, default=50.)
    parser.add_argument('--window', type=float, default=.5)
    parser.add_argument('--max-velocity', type=float, default=.08)
    parser.add_argument('--max-spread', type=float, default=.01)
    parser.add_argument('--settle-time', type=float, default=.25, help='minimum delay before measuring a held joint, seconds')
    parser.add_argument('--hold-timeout', type=float, default=10., help='release if stable capture cannot finish within this time, seconds')
    parser.add_argument('--max-hold-error', type=float, default=.1, help='release on excessive deviation from held target, radians')
    args = parser.parse_args(argv)
    for name in ('rate', 'window', 'max_velocity', 'max_spread', 'settle_time', 'hold_timeout', 'max_hold_error'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be finite and positive')
    if args.hold_timeout <= args.settle_time+args.window:
        parser.error('--hold-timeout must exceed --settle-time plus --window')
    session = LimitCapture()
    print(f'Model copy: {session.output}\nCapture record: {session.path}\n{session.prompt()}')
    if args.prepare_only:
        return 0
    if not sys.stdin.isatty():
        parser.error('Run interactively in a terminal; no hardware connected by this invocation')
    config = json.loads(args.hardware_config.read_text())
    ensure_port_free(args.port or config['port'])
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    if tuple(model.joint(i).name for i in range(model.njnt)) != JOINTS or (model.nq, model.nv, model.nu, model.na) != (16, 16, 16, 16):
        raise ValueError('Unexpected hand joint/actuator layout')
    session.record['torque_policy'] = 'only selected joint held temporarily at measured position on Enter'
    session.persist()
    hand = CaptureHand(args.hardware_api, config, args.port)
    stopped = threading.Event()
    handlers = {s: signal.signal(s, lambda *_: stopped.set()) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        run(hand, model, config, session, stopped, args.rate, args.window, args.max_velocity, args.max_spread,
            args.settle_time, args.hold_timeout, args.max_hold_error)
    finally:
        # Ignore a second interrupt while the synchronous torque-off readbacks finish.
        for s in handlers:
            signal.signal(s, signal.SIG_IGN)
        try:
            hand.close()
        finally:
            for s, handler in handlers.items():
                signal.signal(s, handler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
