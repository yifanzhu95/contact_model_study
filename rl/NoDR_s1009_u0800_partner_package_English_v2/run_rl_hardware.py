#!/usr/bin/env python3
"""Supervised LEAP RL episodes over existing LCM state/pose/action protocols.

Use launch_rl_hardware.py for camera/tracker/viewer/bridge orchestration.
--check performs offline preflight only; --observe-only never publishes actions.
"""
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['XDG_CACHE_HOME'] = str(PACKAGE/'runtime/cache')
os.environ['MPLCONFIGDIR'] = str(PACKAGE/'runtime/cache/matplotlib')
os.environ['TMPDIR'] = str(PACKAGE/'runtime/tmp')
sys.path.insert(0, str(REPO))

import argparse
import hashlib
import json
import signal
import time

import numpy as np

from hardware_eval.core import (PERIOD, confined, world_registration, goal_for_episode, policy_state,
    errors, successful, feedback_model, actuator_bounds, command_target, wait_for_target, PolicyClock)
from hardware_eval.logging import RLLogger, utc
from ContactModelStudy.Hardware.Episodes import HardwareEpisodes
from ContactModelStudy.Hardware.Hand import delta_limits
from ContactModelStudy.Hardware.LCM import (LCMFeedback, DEFAULT_URL, DELTA_ACTION_CHANNEL,
    FeedbackTimingError, FeedbackSkewError, load_pose_transform)


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--eval-config', default=str(PACKAGE/'configs/hardware_eval.json'))
    initial, _ = bootstrap.parse_known_args(argv)
    cfg = json.loads(confined(initial.eval_config).read_text())
    thresholds = cfg.get('success_thresholds', {})
    if (set(thresholds) != {'pos','quat','vel'} or
            any(not isinstance(v, (int,float)) or not np.isfinite(v) or v <= 0 for v in thresholds.values())
            or thresholds['quat'] >= 1 or not np.isfinite(cfg['failure_z_world_m'])):
        raise ValueError('Invalid hardware success/drop thresholds')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-config', default=initial.eval_config)
    p.add_argument('--check', action='store_true', help='offline policy/interface/frame preflight; no LCM or hardware')
    p.add_argument('--hardware', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--task', choices=['cube_reorient'], default='cube_reorient')
    p.add_argument('--n-episodes', type=int, default=1)
    p.add_argument('--steps', type=int, default=cfg['steps'], help='truncate at <=500 controls; observation clock always uses 500')
    p.add_argument('--goal-sign', choices=['alternate', '-1', '1', '+1'], default=cfg['goal_sign'])
    p.add_argument('--frame-mode', choices=['palm_registered', 'identity'], default=cfg['frame_mode'])
    p.add_argument('--stop-on-success', action=argparse.BooleanOptionalAction, default=cfg['stop_on_success'],
                   help='default: freeze on hardware success; --no-stop-on-success evaluates the complete 500-control policy')
    p.add_argument('--backend', choices=['pytorch', 'torchscript'], default=cfg['backend'])
    p.add_argument('--torch-threads', type=int, default=cfg['torch_threads'])
    p.add_argument('--observe-only', action='store_true')
    p.add_argument('--logging', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--hardware-config', default=str(PACKAGE/'configs/leap_hardware_rl.json'))
    p.add_argument('--calibrated-hand', default=str(REPO/'scenes/leap/hardware/calibrated_hand.xml'))
    p.add_argument('--camera-config', default=str(REPO/'configs/leap_pose_transform.json'))
    p.add_argument('--lcm-url', default=DEFAULT_URL)
    p.add_argument('--state-channel', default='LEAP_STATE')
    p.add_argument('--pose-channel', default='POSE')
    p.add_argument('--action-channel', default=DELTA_ACTION_CHANNEL)
    p.add_argument('--feedback-timeout', type=float, default=cfg['feedback_timeout_s'])
    p.add_argument('--max-robot-age', type=float, default=cfg['max_robot_age_s'])
    p.add_argument('--max-pose-age', type=float, default=cfg['max_pose_age_s'])
    p.add_argument('--max-state-skew', type=float, default=cfg['max_state_skew_s'])
    p.add_argument('--target-ack-timeout', type=float, default=cfg['target_ack_timeout_s'])
    p.add_argument('--max-inference-time', type=float, default=cfg['max_inference_time_s'])
    p.add_argument('--results', default=str(PACKAGE/'runtime/results'/time.strftime('rl_%Y%m%d_%H%M%S')/'controller.json'))
    p.add_argument('--episode-status-file')
    p.add_argument('--goal-status-file')
    args = p.parse_args(argv)
    if args.n_episodes < 1 or not 1 <= args.steps <= 500 or args.torch_threads < 1:
        p.error('n-episodes/threads must be positive; steps must be 1..500')
    for key in ['feedback_timeout','max_robot_age','max_pose_age','max_state_skew','target_ack_timeout','max_inference_time']:
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f'{key} must be finite and positive')
    for key in ['results','episode_status_file','goal_status_file','eval_config']:
        if getattr(args, key) is not None:
            setattr(args, key, str(confined(getattr(args, key))))
    args.evaluation_config = cfg
    return args


def load_policy(args):
    import torch
    from portable_policy import Policy
    torch.set_num_threads(args.torch_threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    manifest = json.loads((PACKAGE/'SHA256SUMS.json').read_text())
    for r in manifest['files']:
        if hashlib.sha256(confined(r['file']).read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('Policy package checksum mismatch: ' + r['file'])
    policy = Policy(PACKAGE, backend=args.backend)
    case = json.loads((PACKAGE/'reference_cases.json').read_text())['cases'][0]
    for _ in range(4):
        result = policy.infer_state(case['state'])
    for key, saved in [('observation','observation'), ('raw_action','raw_action'), ('joint_target_rad','joint_target_rad')]:
        torch.testing.assert_close(result[key], torch.tensor(case[saved], dtype=torch.float32).reshape(1, -1),
                                   atol=2e-5 if key == 'raw_action' else 2e-6, rtol=2e-5)
    return policy, dict(backend=args.backend, device='cpu', torch_version=torch.__version__,
        actor_sha256=hashlib.sha256((PACKAGE/'policy_noDR_s1009_u0800.ts.pt').read_bytes()).hexdigest(),
        interface_sha256=hashlib.sha256((PACKAGE/'interface.json').read_bytes()).hexdigest(),
        eval_config_sha256=hashlib.sha256(confined(args.eval_config).read_bytes()).hexdigest(),
        receiver_code_sha256={name: hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()
            for name in ['run_rl_hardware.py','hardware_eval/core.py','hardware_eval/logging.py','launch_rl_hardware.py']},
        package_checksums_verified=True, archived_example_parity=True, control_period_s=PERIOD)


class RLEpisodes(HardwareEpisodes):
    def status(self, stage, episode, prompt, **details):
        super().status(stage, episode, prompt.replace('MPPI', 'RL policy'), **details)

    def wait_for_enter(self, episode, stage, prompt, target=None):
        # Retain commanded holding effort instead of latching measured q.
        if target is None:
            target = self.last_target
        super().wait_for_enter(episode, stage, prompt, target)

    def after_success(self, episode, final):
        self.wait_for_enter(episode, 'success_hold',
            f'Episode {episode+1}: success. Holding the last accepted target. Remove the cube, then press Enter to return to zero.',
            target=self.last_target)


def publish_goal(path, goal, thresholds, episode, records, step, total):
    if not path:
        return
    now = time.time()
    value = dict(episode=episode, updated_at=now, current_step=step, total_steps=total,
        target_pos=goal['world_position'].tolist(), target_quat=goal['world_quaternion'].tolist(),
        success_thresholds=thresholds,
        episode_windows=[dict(episode=e['episode'], started_at_utc=e['started_at_utc'],
            finished_at_utc=e.get('finished_at_utc'), outcome=e['finish_reason'],
            confirmed_through=None if e.get('finished_at_utc') else now) for e in records])
    p = confined(path); p.parent.mkdir(parents=True, exist_ok=True)
    RLLogger.json(p, value)


def fresh_state(feedback, limits):
    # Retry transient timestamp skew on the existing asynchronous receiver;
    # no new serial reads and no post-inference rebasing of the action.
    deadline = time.monotonic() + .05
    while True:
        try:
            return feedback.buffer.snapshot(**limits)
        except FeedbackSkewError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.002)


def main(argv=None):
    args = parse_args(argv)
    hardware = json.loads(Path(args.hardware_config).read_text())
    if hardware.get('delta_reference') != 'previous_target' and not args.observe_only:
        raise ValueError('RL requires a private hardware config with delta_reference=previous_target; use launch_rl_hardware.py')
    policy, identity = load_policy(args)  # finish dependency/contract checks before any LCM action
    T = world_registration(policy.interface, hardware, args.calibrated_hand, args.frame_mode)
    lower, upper = actuator_bounds(args.calibrated_hand, policy.interface)
    down, up = delta_limits(hardware)
    before, after, camera = load_pose_transform(args.camera_config)
    if np.any(np.asarray(hardware['robot_qref']) < lower) or np.any(np.asarray(hardware['robot_qref']) > upper):
        raise ValueError('Configured reference pose exceeds policy/calibrated actuator ranges')
    identity.update(T_training_world_from_current_world=T.tolist(), frame_mode=args.frame_mode,
        camera_config_sha256=hashlib.sha256(Path(args.camera_config).read_bytes()).hexdigest(),
        hardware_config_sha256=hashlib.sha256(Path(args.hardware_config).read_bytes()).hexdigest(),
        evaluation_definition=args.evaluation_config['success_definition'])
    from ContactModelStudy.Hardware.Scene import JOINTS
    if [j['joint_name'] for j in policy.interface['joints']] != list(JOINTS):
        raise ValueError('Policy joint ordering differs from the LCM bridge mapping')
    if args.check:
        print(json.dumps(dict(status='RL_PREFLIGHT_PASSED', policy=identity, hardware_access=False,
                              simulation_steps=0, goal=goal_for_episode(policy.interface,args.goal_sign,0,T)['world_position'].tolist()), indent=2))
        return 0
    thresholds = args.evaluation_config['success_thresholds']
    failure_z = float(args.evaluation_config['failure_z_world_m'])
    log = dict(mode='hardware_rl_observe' if args.observe_only else 'hardware_rl', status='starting', args=vars(args), argv=sys.argv[1:] if argv is None else list(argv),
        started_at_utc=utc(), episodes=[], policy=identity, hardware_config=hardware, pose_transform=camera,
        success_thresholds=thresholds, failure_z=failure_z, fixed_goal_quat=None,
        control_policy='previous_accepted_target_plus_0.064_tanh_actor', goal_policy='trained_fixed_canonical_x_signs',
        planner_config=identity, control_period=PERIOD,
        joint_order=[j['joint_name'] for j in policy.interface['joints']],
        object_velocity_convention='JSON q_dot uses MuJoCo body angular velocity; policy_state converts angular velocity to world/training axes')
    logger = RLLogger(confined(args.results)) if args.logging else None
    feedback, lifecycle, episode, goal = None, None, None, None
    started = time.monotonic(); episode_index = 0; exit_code = 0
    previous_sigterm = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit('Supervisor stop')))
    limits = dict(max_robot_age=args.max_robot_age, max_pose_age=args.max_pose_age, max_skew=args.max_state_skew)
    try:
        feedback = LCMFeedback(feedback_model(policy.interface), url=args.lcm_url,
            robot_channel=args.state_channel, pose_channel=args.pose_channel, action_channel=args.action_channel,
            before=before, after=after, target_feedback=not args.observe_only)
        feedback.wait(args.feedback_timeout, require_target=not args.observe_only, **limits)
        if not args.observe_only:
            lifecycle = RLEpisodes(feedback, hardware, args.max_robot_age, args.episode_status_file, args.logging)
        for episode_index in range(args.n_episodes):
            cycle_started = time.monotonic()
            goal = goal_for_episode(policy.interface, args.goal_sign, episode_index, T)
            publish_goal(args.goal_status_file, goal, thresholds, episode_index, log['episodes'], 0, args.steps)
            if lifecycle:
                lifecycle.before_episode(episode_index)
                q, v, metadata = lifecycle.wait_for_tracking(episode_index, args.feedback_timeout, **limits)
                previous, _ = feedback.buffer.snapshot_target(args.max_robot_age)
                # Preparation uses absolute waypoints; confirm the final reference has actually reached the bridge.
                previous, _ = wait_for_target(feedback, lifecycle.reference, 0, args.target_ack_timeout, args.max_robot_age)
                lifecycle.last_target = previous.copy()
            else:
                q, v, metadata = fresh_state(feedback, limits)
                previous = q[:16].copy()
            manipulation_started = time.monotonic()
            episode = dict(episode=episode_index, goal_sign=goal['sign'], goal=goal['world_quaternion'].tolist(),
                target_pos=goal['world_position'].tolist(), training_goal=goal['training_quaternion'].tolist(),
                training_target_pos=goal['training_position'].tolist(), q_start=q.tolist(), q_dot_start=v.tolist(),
                initial_accepted_target=previous.tolist(), started_at_utc=utc(), steps=[], n_steps=0,
                finish_reason='running', completed=False, successful_state_samples=0)
            log['episodes'].append(episode); log['status'] = 'running'
            clock = PolicyClock(); success_samples = []; last_display = 0.
            for step in range(args.steps):
                tick = time.monotonic()
                q, v, metadata = fresh_state(feedback, limits)
                if not args.observe_only:
                    previous, target_meta = feedback.buffer.snapshot_target(args.max_robot_age)
                    metadata.update(target_meta)
                error = errors(q, v, goal)
                passed = successful(error, thresholds)
                success_samples.append(passed)
                episode['successful_state_samples'] += int(passed)
                if passed:
                    episode.setdefault('first_arrival_step', step)
                if q[18] < failure_z:
                    episode['finish_reason'] = 'failure'; break
                if passed and args.stop_on_success:
                    episode['finish_reason'] = 'success'; break
                state = policy_state(q, v, previous, goal, step, T)
                row = dict(step=step, timestamp_utc=utc(), elapsed_s=tick-manipulation_started,
                    q=q.tolist(), q_dot=v.tolist(), previous_accepted_target=previous.tolist(), feedback=metadata,
                    goal_errors=error, quaternion_error_rad=float(2*np.arcsin(np.sqrt(error['quat']))),
                    policy_state=state, action_published=False, inference_time_s=0.)
                episode['steps'].append(row)
                inference_started = time.monotonic()
                try:
                    result = policy.infer_state(state)
                finally:
                    row['inference_time_s'] = time.monotonic()-inference_started
                if row['inference_time_s'] > args.max_inference_time:
                    raise FeedbackTimingError('RL inference exceeded its maximum time; holding last target')
                # Validate command-time freshness without changing the policy's previous-target base.
                feedback.buffer.snapshot(require_sync=False, **limits)
                proposed = result['joint_target_rad'][0].numpy().astype(float)
                desired = command_target(proposed, previous, lower, upper, down, up)
                row.update(observation=result['observation'][0].tolist(), raw_action=result['raw_action'][0].tolist(),
                    policy_target=proposed.tolist(), requested_target=desired.tolist(),
                    action_delta=(desired-previous).tolist(), clipped_joints=int(np.count_nonzero(abs(proposed-desired)>1e-7)))
                if not args.observe_only:
                    publication = feedback.publish_delta(desired-previous)
                    row.update(action_timestamp_us=publication, action_published_at_utc=utc(), action_published=True)
                    episode['n_steps'] += 1
                    # If acknowledgment fails, the error hold still retains the just-requested target.
                    lifecycle.last_target = desired.copy()
                    ack_started = time.monotonic()
                    accepted, ack = wait_for_target(feedback, desired, publication, args.target_ack_timeout, args.max_robot_age)
                    row.update(accepted_target=accepted.tolist(), target_ack=ack, target_ack_wait_s=time.monotonic()-ack_started)
                    lifecycle.last_target = accepted.copy()
                else:
                    row['accepted_target'] = None
                    episode['n_steps'] += 1
                waiting, overrun = clock.finish_period()
                row.update(pacing_wait_s=waiting, cycle_overrun=overrun, wall_time_s=time.monotonic()-tick)
                if time.monotonic()-last_display >= .1:
                    publish_goal(args.goal_status_file, goal, thresholds, episode_index, log['episodes'], episode['n_steps'], args.steps)
                    last_display = time.monotonic()
            else:
                episode['finish_reason'] = 'timeout'
            q, v, _ = fresh_state(feedback, limits)
            final_errors = errors(q, v, goal)
            final_passed = successful(final_errors, thresholds)
            if final_passed:
                episode.setdefault('first_arrival_step', episode['n_steps'])
            if episode['finish_reason'] == 'timeout' and final_passed:
                episode['finish_reason'] = 'success'
            episode.update(q_end=q.tolist(), q_dot_end=v.tolist(), goal_errors=final_errors,
                final_success=final_passed, success=episode['finish_reason']=='success',
                continuous_last150_success=(bool(all((success_samples+[final_passed])[-150:]))
                    if episode['n_steps'] == 500 else None),
                quaternion_error_rad=float(2*np.arcsin(np.sqrt(final_errors['quat']))), completed=True,
                manipulation_wall_time_s=time.monotonic()-manipulation_started, finished_at_utc=utc())
            publish_goal(args.goal_status_file, goal, thresholds, episode_index, log['episodes'], episode['n_steps'], args.steps)
            if logger: logger.checkpoint(log)
            print(f"Episode {episode_index+1}: {episode['finish_reason']}, {episode['n_steps']} controls; errors {final_errors}", flush=True)
            if lifecycle:
                if episode['success']: lifecycle.after_success(episode_index, episode_index==args.n_episodes-1)
                else: lifecycle.after_unsuccessful(episode_index, episode['finish_reason'])
                episode['q_after_reset'] = lifecycle.return_to_zero(episode_index).tolist()
            episode['cycle_wall_time_s'] = time.monotonic()-cycle_started
            if logger: logger.checkpoint(log)
        if lifecycle:
            lifecycle.status('complete', episode_index, 'All RL episodes complete; zero reached. Disabling torque.')
        log['status'] = 'complete'
    except KeyboardInterrupt:
        log.update(status='interrupted', error='KeyboardInterrupt')
        if episode and not episode['completed']:
            episode.update(finish_reason='interrupted', finished_at_utc=utc(), manipulation_wall_time_s=time.monotonic()-manipulation_started)
        if goal: publish_goal(args.goal_status_file, goal, thresholds, episode_index, log['episodes'], episode['n_steps'] if episode else 0, args.steps)
        if lifecycle: lifecycle.shutdown_on_interrupt(episode_index)
    except BaseException as exc:
        exit_code = 1
        log.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if episode and not episode['completed']:
            episode.update(finish_reason='error', finished_at_utc=utc(), manipulation_wall_time_s=time.monotonic()-manipulation_started)
        if goal: publish_goal(args.goal_status_file, goal, thresholds, episode_index, log['episodes'], episode['n_steps'] if episode else 0, args.steps)
        if isinstance(exc, FeedbackTimingError) and lifecycle:
            lifecycle.shutdown_after_feedback_error(episode_index, exc)
        else:
            print(log['error'], file=sys.stderr, flush=True)
    finally:
        # SIGTERM/viewer failure requests immediate torque cleanup; no reset motion.
        try:
            if feedback:
                try:
                    if not args.observe_only: feedback.publish_stop()
                finally: feedback.close()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
            log.update(finished_at_utc=utc(), wall_time_s=time.monotonic()-started)
            if logger: logger.close(log)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
