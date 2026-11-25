#!/usr/bin/env python3
"""Replay recorded bridge-written M2 targets on an empty hand and measure error.

Default validates only. --hardware enables physical motion. This standalone
test bounds absolute targets to calibrated joint/actuator ranges and records
every adjustment; it changes no MPPI, bridge, calibration or motor settings.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import mujoco
import numpy as np

from ContactModelStudy.Hardware.Episodes import HardwareEpisodes
from ContactModelStudy.Hardware.Hand import control_limits, vector
from ContactModelStudy.Hardware.LCM import LCMFeedback, RobotPositionAction, RobotTargetState
from ContactModelStudy.Hardware.Scene import JOINTS

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT/'results/diagnostics/m2_tracking_baseline/leap_experiment_20261116_215909_720053/controller.json'
DEFAULT_CONFIG = ROOT/'configs/m2_tracking_diagnostic/hardware.json'
DEFAULT_MODEL = ROOT/'scenes/leap/hardware/calibrated_hand.xml'


def load_schedule(source, episode_number, model):
    document = json.loads(Path(source).read_text())
    episode = next((e for e in document['episodes'] if e['episode'] == episode_number-1), None)
    if episode is None or not episode.get('completed'):
        raise ValueError('Select a completed episode, numbered from one')
    ids = {s['action_timestamp_us'] for s in episode['steps'] if s.get('action_published')}
    if len(ids) < 2:
        raise ValueError('Need at least two published, timestamped actions')
    events = []
    telemetry = Path(source).with_name('bridge_telemetry.jsonl')
    with telemetry.open() as stream:
        for line in stream:
            row = json.loads(line)
            if (row.get('write_succeeded') and row.get('new_action')
                    and row.get('written_action_timestamp_us') in ids):
                events.append(row)
    if len(events) != len(ids) or len({r['written_action_timestamp_us'] for r in events}) != len(ids):
        raise ValueError('Every source action must have one exact successful bridge-write match')
    stamps = np.array([r['write_finished_us'] for r in events], dtype=np.int64)
    intervals = np.diff(stamps)*1e-6
    if np.any(intervals <= .005) or np.any(intervals >= .4):
        raise ValueError('Source intervals must be between 5 and 400 ms')
    requested = np.array([vector(r['written_target'], 'recorded target') for r in events])
    lower, upper = control_limits(model)
    applied = np.clip(requested, lower, upper)
    initial = np.clip(vector(events[0]['held_target_before_read'], 'initial held target'), lower, upper)
    adjustment = applied-requested
    return dict(source=str(Path(source).resolve()), source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                source_telemetry=str(telemetry.resolve()), episode=episode_number,
                source_finish_reason=episode['finish_reason'],
                initial_target=initial.tolist(), offsets_s=((stamps-stamps[0])*1e-6).tolist(),
                original_targets=requested.tolist(), applied_targets=applied.tolist(),
                position_limit_adjustments_rad=adjustment.tolist(),
                adjusted_joint_values=int(np.count_nonzero(abs(adjustment)>1e-8)),
                adjusted_commands=int(np.any(abs(adjustment)>1e-8, axis=1).sum()),
                max_adjustment_rad=float(np.max(abs(adjustment))),
                mean_command_interval_s=float(intervals.mean()),
                source_action_ids=[r['written_action_timestamp_us'] for r in events])


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=int(a.size), mae=float(abs(a).mean()), rmse=float(np.sqrt(np.mean(a*a))),
                p95_abs=float(np.percentile(abs(a), 95)), max_abs=float(abs(a).max())) if a.size else None


class Replay:
    def __init__(self, args, schedule, config, model, directory):
        self.args, self.schedule, self.config, self.model, self.directory = args, schedule, config, model, directory
        self.bridge = self.viewer = self.feedback = None
        self.commands = []
        self.report = dict(status='starting', started_at_utc=datetime.now(timezone.utc).isoformat(),
                           model=str(args.model.resolve()), hardware_config=config, schedule=schedule,
                           commands=self.commands, object_condition='empty_hand',
                           target_semantics='Absolute recorded SDK-written targets, bounded to calibrated ranges; no delta rebasing.',
                           control_semantics='Original write-to-write timing; no planner, simulation stepping or camera feedback.')
        latency = Path('/sys/bus/usb-serial/devices')/Path(config['port']).resolve().name/'latency_timer'
        self.report['usb_latency_timer_ms'] = int(latency.read_text()) if latency.exists() else None

    def check(self):
        if self.bridge.poll() is not None:
            raise RuntimeError(f'Bridge exited with {self.bridge.returncode}; see bridge.log')
        if (self.directory/'viewer_stop').exists() or (self.viewer is not None and self.viewer.poll() is not None):
            raise KeyboardInterrupt('Viewer stopped the replay')
        return self.feedback.buffer.snapshot_robot(.15)

    def publish(self, target, index=None):
        self.check()
        stamp = int(time.time()*1e6)
        self.feedback.connection.publish('LEAP_RESPONSE_ACTION', RobotPositionAction(stamp, target).encode())
        self.feedback.connection.publish('LEAP_RESPONSE_STATE_TARGET', RobotTargetState(stamp, target).encode())
        self.commands.append(dict(timestamp_us=stamp, index=index, target=np.asarray(target).tolist()))

    def hold(self, target, seconds):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            self.publish(target)
            time.sleep(min(.05, max(0, deadline-time.monotonic())))

    def run(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        motion = None
        bridge_log = (self.directory/'bridge.log').open('w')
        viewer_log = (self.directory/'viewer.log').open('w')
        self.feedback = LCMFeedback(self.model, robot_channel='LEAP_RESPONSE_STATE',
                                    pose_channel='LEAP_RESPONSE_UNUSED_POSE', action_channel='LEAP_RESPONSE_ACTION')
        try:
            self.bridge = subprocess.Popen([sys.executable, str(ROOT/'leap_hand_lcm_bridge.py'),
                '--hardware-config', str(self.args.hardware_config.resolve()), '--model', str(self.args.model.resolve()),
                '--state-channel', 'LEAP_RESPONSE_STATE', '--action-channel', 'LEAP_RESPONSE_ACTION',
                '--telemetry-log', str(self.directory/'bridge_telemetry.jsonl')],
                stdout=bridge_log, stderr=subprocess.STDOUT, env=env)
            deadline = time.monotonic()+30
            while True:
                try:
                    self.check(); break
                except RuntimeError:
                    if self.bridge.poll() is not None or time.monotonic() > deadline:
                        raise
                    time.sleep(.02)
            if self.args.viewer:
                self.viewer = subprocess.Popen([sys.executable, str(ROOT/'measure_leap_target_response.py'),
                    '--view-only', str(self.directory), '--model', str(self.args.model.resolve())],
                    stdout=viewer_log, stderr=subprocess.STDOUT, env=env)
            motion = HardwareEpisodes(self.feedback, self.config, .15, logging_enabled=False)
            motion.timeout = 15
            motion.reference = np.asarray(self.schedule['initial_target'])
            motion.prepare(0)
            self.hold(motion.reference, 1)
            (self.directory/'phase.json').write_text(json.dumps(dict(mode='Recorded M2 replay', joint='all joints')))
            self.report['status'] = 'replaying'
            self.report['replay_started_us'] = int(time.time()*1e6)
            start = time.monotonic()
            for index, (offset, target) in enumerate(zip(self.schedule['offsets_s'], self.schedule['applied_targets'])):
                deadline = start+offset
                while time.monotonic() < deadline:
                    self.check(); time.sleep(min(.001, max(0, deadline-time.monotonic())))
                lateness = time.monotonic()-deadline
                if lateness > .02:
                    raise RuntimeError(f'Replay scheduling overrun {lateness*1000:.1f} ms; stopping rather than bursting queued targets')
                self.publish(target, index)
                self.commands[-1]['scheduling_lateness_s'] = lateness
                if index % 10 == 0:
                    print(f'Replayed {index+1}/{len(self.schedule["offsets_s"])} targets', flush=True)
            self.report['sequence_finished_us'] = int(time.time()*1e6)
            self.hold(self.schedule['applied_targets'][-1], 2.)
            self.report['settled_tail_finished_us'] = int(time.time()*1e6)
            motion.return_to_zero(0)
            self.report['status'] = 'complete'
        except BaseException as exc:
            self.report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                               error=f'{type(exc).__name__}: {exc}')
            print(self.report['error'], flush=True)
            if motion is not None and self.bridge.poll() is None:
                try:
                    motion.return_to_zero(0)
                    self.report['error_return_to_zero'] = True
                except BaseException as reset:
                    self.report['reset_error'] = str(reset)
        finally:
            if self.bridge is not None:
                deadline = time.monotonic()+3
                while self.bridge.poll() is None and time.monotonic() < deadline:
                    self.feedback.publish_stop(); time.sleep(.05)
                if self.bridge.poll() is None:
                    self.bridge.terminate()
                try:
                    self.bridge.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.bridge.kill(); self.bridge.wait(timeout=2)
                self.report['bridge_exit_code'] = self.bridge.returncode
            if self.viewer is not None and self.viewer.poll() is None:
                self.viewer.terminate()
                try:
                    self.viewer.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.viewer.kill(); self.viewer.wait(timeout=2)
            self.feedback.close(); bridge_log.close(); viewer_log.close()
            self.report['torque_off_verified'] = 'Torque disabled and verified on all 16 motors.' in (self.directory/'bridge.log').read_text()
            self.report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            (self.directory/'experiment.json').write_text(json.dumps(self.report, indent=2)+'\n')
        return 0 if self.report['status'] == 'complete' and self.report['torque_off_verified'] else 1


def analyze(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    meta = json.loads((directory/'experiment.json').read_text())
    if 'sequence_finished_us' not in meta:
        return
    rows = []
    with (directory/'bridge_telemetry.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line)
            if r.get('event') == 'cycle' and 'q' in r:
                rows.append(r)
    start, end = meta['replay_started_us'], meta['sequence_finished_us']
    select = [r for r in rows if start <= r['read_started_us'] and r['read_finished_us'] <= end]
    tail = [r for r in rows if meta['settled_tail_finished_us']-500000 <= r['read_started_us'] <= meta['settled_tail_finished_us']]
    q = np.array([r['q'] for r in select]); u = np.array([r['held_target_before_read'] for r in select])
    t = np.array([(r['read_started_us']+r['read_finished_us'])/2-start for r in select])*1e-6
    error_deg = np.degrees(q-u)
    writes = {r['written_action_timestamp_us']: r for r in rows if r.get('new_action') and r.get('write_succeeded')}
    commands = [c for c in meta['commands'] if c['index'] is not None]
    missing = [c['index'] for c in commands if c['timestamp_us'] not in writes]
    command_index = {c['timestamp_us']: c['index'] for c in commands}
    original_error = np.array([np.degrees(np.asarray(r['q'])-meta['schedule']['original_targets'][command_index[r['held_action_timestamp_us']]])
                               for r in select if r.get('held_action_timestamp_us') in command_index])
    model = mujoco.MjModel.from_xml_path(meta['model']); data = mujoco.MjData(model)
    sites = [model.site(name).id for name in ('if_tip','mf_tip','rf_tip','th_tip')]
    tip_gap = []
    for measured, target in zip(q, u):
        data.qpos[:] = measured; mujoco.mj_kinematics(model, data); actual = data.site_xpos[sites].copy()
        data.qpos[:] = target; mujoco.mj_kinematics(model, data)
        tip_gap.append(np.linalg.norm(actual-data.site_xpos[sites], axis=1)*1000)
    tip_gap = np.asarray(tip_gap)
    report = dict(status=meta['status'], source_episode=meta['schedule']['episode'],
                  commands=len(commands), unmatched_command_ids=missing,
                  torque_off_verified=meta['torque_off_verified'],
                  position_adjustments={k: meta['schedule'][k] for k in ['adjusted_commands','adjusted_joint_values','max_adjustment_rad']},
                  aggregate_error_deg=stats(error_deg),
                  aggregate_error_to_original_unbounded_targets_deg=stats(original_error),
                  per_fingertip_fk_gap_mm={name:stats(tip_gap[:,i]) for i,name in enumerate(('index','middle','ring','thumb'))},
                  fraction_joint_samples_over_3deg=float(np.mean(abs(error_deg)>3)),
                  fraction_joint_samples_over_5deg=float(np.mean(abs(error_deg)>5)),
                  fraction_joint_samples_over_10deg=float(np.mean(abs(error_deg)>10)),
                  publish_to_write_ms=stats([(writes[c['timestamp_us']]['write_started_us']-c['timestamp_us'])*.001 for c in commands if c['timestamp_us'] in writes]),
                  scheduling_lateness_ms=stats([c['scheduling_lateness_s']*1000 for c in commands]),
                  per_joint={name: stats(error_deg[:,j]) for j,name in enumerate(JOINTS)},
                  settled_tail_per_joint_deg={name: stats([np.degrees(r['q'][j]-r['held_target_before_read'][j]) for r in tail]) for j,name in enumerate(JOINTS)},
                  interpretation='Encoder error relative to the previously SDK-written held target. Empty hand; position-limited replay of originally loaded commands. Not an isolated load-effect or manipulation-success comparison.')
    fig, axes = plt.subplots(4,4,figsize=(17,12),sharex=True,constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t,np.degrees(q[:,j]),label='Measured')
        ax.step(t,np.degrees(u[:,j]),where='post',label='Held target',alpha=.8)
        ax.set_title(JOINTS[j]); ax.grid(alpha=.25)
        if j//4 == 3: ax.set_xlabel('Replay time (s)')
        if j%4 == 0: ax.set_ylabel('Joint angle (deg)')
    axes.flat[0].legend(); fig.suptitle('Recorded M2 actions: empty-hand position-limited hardware replay')
    fig.savefig(directory/'joint_tracking.png',dpi=170); fig.savefig(directory/'joint_tracking.pdf');plt.close(fig)
    np.savez_compressed(directory/'joint_tracking.npz', time_s=t, measured_q=q, held_target=u, error_deg=error_deg)
    with (directory/'joint_errors.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['joint','mae','rmse','p95_abs','max_abs','settled_tail_mae_deg'])
        writer.writeheader()
        for name in JOINTS:
            s = report['per_joint'][name]
            writer.writerow(dict(joint=name,**{k:s[k] for k in ['mae','rmse','p95_abs','max_abs']},
                                 settled_tail_mae_deg=report['settled_tail_per_joint_deg'][name]['mae']))
    (directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['commands','unmatched_command_ids','aggregate_error_deg','fraction_joint_samples_over_5deg','torque_off_verified']},indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--episode', type=int, default=4, help='one-based recorded episode number')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--hardware-config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--hardware', action='store_true')
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--analyze', type=Path)
    args = p.parse_args(argv)
    if args.analyze:
        analyze(args.analyze);return 0
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if (model.nq,model.nv,model.nu)!=(16,16,16) or tuple(model.joint(i).name for i in range(16)) != JOINTS:
        p.error('Use the calibrated hand-only scene in the recorded joint order')
    schedule = load_schedule(args.source,args.episode,model)
    print(f"Episode {args.episode}: {len(schedule['offsets_s'])} exact SDK-written targets over {schedule['offsets_s'][-1]:.3f} s; "
          f"{schedule['adjusted_commands']} commands adjusted to calibrated absolute limits (max {schedule['max_adjustment_rad']:.4f} rad).",flush=True)
    if not args.hardware:
        print('Validation only; no USB connection, torque or motion.'); return 0
    config = json.loads(args.hardware_config.read_text())
    if config.get('reviewed_for_this_hand') is not True:
        p.error('Use a hardware configuration reviewed for this hand')
    owners = subprocess.run(['fuser',config['port']],capture_output=True,text=True)
    if owners.returncode != 1:
        p.error('Serial port is busy or ownership cannot be verified: '+owners.stdout+owners.stderr)
    directory = (args.output or ROOT/'results/diagnostics'/('m2_action_replay_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    directory.mkdir(parents=True,exist_ok=False)
    lock = ROOT/'.runtime/leap_target_response.lock';lock.parent.mkdir(exist_ok=True)
    def interrupt(*_): raise KeyboardInterrupt('Interrupted')
    signal.signal(signal.SIGTERM,interrupt)
    with lock.open('w') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result = Replay(args,schedule,config,model,directory).run()
    analyze(directory)
    print('Results:',directory,flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
