#!/usr/bin/env python3
"""Control the USB-connected LEAP hand using MuJoCo's Control sliders.

The displayed joint positions are measured from hardware; no simulation is
stepped. Enter enables torque at the measured pose once it is inside model
limits. Space, Q, Escape, window close,
Ctrl-C, communication failure or a configured limit stops and disables torque.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from ContactModelStudy.Hardware.Hand import (
    HardwareHand, bounded_command, check_feedback, control_limits,
)
from ContactModelStudy.Hardware.Scene import JOINTS

ROOT = Path(__file__).resolve().parent


def run_viewer(hand, model, config, rate=50):
    """Use manual speed/tracking limits independently of MPPI ctrl_range."""
    lower, upper = control_limits(model)
    period = 1.0 / rate
    stopped = threading.Event()
    arm_requested = threading.Event()

    def key_pressed(key):
        if key in (32, 27, 256, ord('q'), ord('Q')):
            stopped.set()
        elif key in (13, 257, 335):
            arm_requested.set()

    hand.connect()
    q, v, current = hand.read()
    check_feedback(q, v, current, config, np.full(16, -np.inf), np.full(16, np.inf))
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.ctrl[:], data.act[:] = q, v, q, q
    mujoco.mj_forward(model, data)
    print('Initial measured joints (MuJoCo order, radians):', flush=True)
    for name, value in zip(JOINTS, q):
        print(f'  {name:8s} {value:+.4f}', flush=True)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_pressed,
                                      show_left_ui=False, show_right_ui=True) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance = 0.45
            viewer.cam.azimuth = 130
            viewer.cam.elevation = -35
        viewer.sync()
        print('VIEWER OPEN: torque OFF. Reposition any joints listed outside limits by hand, '
              'then press Enter to enable slider control.', flush=True)
        while viewer.is_running() and not stopped.is_set():
            started = time.monotonic()
            q, v, current = hand.read()
            check_feedback(q, v, current, config, np.full(16, -np.inf), np.full(16, np.inf))
            outside = np.flatnonzero((q < lower) | (q > upper))
            details = '\n'.join(f'{JOINTS[i]}: {q[i]:+.3f} outside [{lower[i]:+.3f}, {upper[i]:+.3f}]'
                                for i in outside)
            status = ('TORQUE OFF - gently reposition these joints:\n' + details
                      if outside.size else 'TORQUE OFF - pose is within limits. Press Enter to enable.')
            viewer.set_texts((None, None, status + '\nSpace / Q / Escape: stop', None))
            with viewer.lock():
                data.qpos[:], data.qvel[:], data.ctrl[:], data.act[:] = q, v, q, q
                mujoco.mj_forward(model, data)
            viewer.sync()
            if arm_requested.is_set():
                arm_requested.clear()
                if outside.size:
                    print('Cannot enable torque; reposition by hand first:\n' + details, flush=True)
                elif viewer.is_running() and not stopped.is_set():
                    # Re-read immediately before preloading and enabling torque.
                    q, v, current = hand.read()
                    check_feedback(q, v, current, config, lower, upper)
                    with viewer.lock():
                        data.ctrl[:] = q
                    hand.enable(q)
                    break
            time.sleep(max(0.0, period - (time.monotonic() - started)))
        else:
            return
        viewer.set_texts((None, None,
            'TORQUE ON - use Control sliders on the right.\nSpace / Q / Escape: STOP and disable torque.', None))
        command = q.copy()
        last_write = last_health = started_at = time.monotonic()
        print('READY: torque enabled at the measured pose. Expand Control on the right; '
              'move *_act sliders. The rendered hand follows measured feedback.', flush=True)
        print('STOP: Space / Q / Escape, close the window, or Ctrl-C. '
              'Stopping disables torque.', flush=True)
        while viewer.is_running() and not stopped.is_set():
            started = time.monotonic()
            q, v, current = hand.read()
            check_feedback(q, v, current, config, lower, upper)
            if np.max(np.abs(q - command)) > config['max_tracking_error']:
                raise RuntimeError('Hardware tracking error exceeded')
            with viewer.lock():
                data.qpos[:], data.qvel[:], data.act[:] = q, v, command
                data.time = started - started_at
                # Keep ctrl as the slider target; do not overwrite user input.
                mujoco.mj_forward(model, data)
            viewer.sync()
            if stopped.is_set() or not viewer.is_running():
                break
            with viewer.lock():
                target = data.ctrl.copy()
            now = time.monotonic()
            command = bounded_command(target, command, q, lower, upper,
                config['max_command_speed'] * min(max(now - last_write, 0.0), 2 * period),
                config['max_tracking_error'])
            hand.write(command)
            last_write = now
            if now - last_health >= 1.0:
                hand.health()
                last_health = time.monotonic()
            time.sleep(max(0.0, period - (time.monotonic() - started)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware-api', type=Path,
        default=Path.home() / 'leap_agentic_sysid/third_party/LEAP_Hand_Hardware')
    parser.add_argument('--hardware-config', type=Path, default=ROOT / 'configs/leap_hardware.json')
    parser.add_argument('--model', type=Path, default=ROOT / 'scenes/leap/hardware/calibrated_hand.xml')
    parser.add_argument('--port')
    parser.add_argument('--rate', type=float, default=50)
    args = parser.parse_args(argv)
    if not np.isfinite(args.rate) or not 0 < args.rate < 90:
        parser.error('--rate must be finite, positive and below 90 Hz')
    config = json.loads(args.hardware_config.expanduser().read_text())
    config = {**config, **config.get('manual_viewer_limits', {})}
    if config.get('reviewed_for_this_hand') is not True:
        parser.error('hardware config must have reviewed_for_this_hand=true')
    for name in ('max_command_speed', 'max_tracking_error', 'max_velocity',
                 'max_current_raw', 'max_temperature'):
        if not np.isfinite(config[name]) or config[name] <= 0:
            parser.error(f'{name} must be finite and positive')
    model = mujoco.MjModel.from_xml_path(str(args.model.expanduser()))
    if (model.nq, model.nv, model.nu, model.na) != (16, 16, 16, 16):
        parser.error('use the calibrated hand-only model with 16 joints/actuators/activations')
    for i, name in enumerate(JOINTS):
        if (model.joint(i).name != name or model.jnt_qposadr[i] != i
                or model.jnt_dofadr[i] != i or model.actuator_trnid[i, 0] != i):
            parser.error(f'unsupported joint/actuator ordering at {name}')
    control_limits(model)
    hand = HardwareHand(args.hardware_api, config, args.port)

    def interrupt(sig, frame):
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        run_viewer(hand, model, config, args.rate)
    except KeyboardInterrupt:
        print('Stopping joint test.', flush=True)
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            hand.close()
            print('STOPPED: torque disabled on all 16 motors; serial port closed.', flush=True)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
