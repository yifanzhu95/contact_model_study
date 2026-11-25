#!/usr/bin/env python3
"""The study's MPPI workflow with LEAP joint states and FoundationPose over LCM.

Simulation is the default. --hardware subscribes to LEAP_STATE and POSE and
publishes joint increments on LEAP_ACTION_DELTA. The bridge adds each new delta
once to its latest measured pose and owns serial
communication and torque. --observe-only plans from LCM without actions.
Hardware episodes first move gradually to robot_qref and wait for Enter. On
success, the measured joint pose is held until Enter returns the hand to zero.
The final return to zero is followed by bridge shutdown and torque disable.
Ctrl-C also returns gradually to zero, then requests torque disable. A second
Ctrl-C or SIGTERM aborts the return immediately.
Camera/synchronization timing errors hold the last target until Enter, then
return gradually to zero and request torque disable using joint feedback only.
Hardware runs unpaced by default. --pace-control limits target updates to
1/control_timestep when planning is faster, without adding delay to overruns.

Examples::

    python run_leap_hardware.py --device cpu --n-samples 8 --steps 3
    python run_leap_hardware.py --hardware --observe-only
    python run_leap_hardware.py --hardware --goal-quat 0.707107 0 0 0.707107

Task costs, geometry selectors, contact presets and planner configuration come
from Drivers/run_episodes.py. Calibrated hand dynamics are applied to the
selected rollout and eval scenes. Camera intrinsics, extrinsics and small pose
corrections live together in configs/leap_pose_transform.json.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
import time
import tempfile

import mujoco
import numpy as np
import warp as wp

from ContactModelStudy.Drivers import run_episodes as reference
from ContactModelStudy.Hardware.Hand import clip_joint_target, control_limits, delta_limits, delta_reference, vector
from ContactModelStudy.Hardware.Calibration import verify_model_tag
from ContactModelStudy.Hardware.Episodes import HardwareEpisodes
from ContactModelStudy.Hardware.CubeOrientation import describe_start, terminal_lines
from ContactModelStudy.Hardware.Fingertips import FingertipPrediction, write_prediction
from ContactModelStudy.Hardware.PlanBlending import TemporalPlanBlend
from ContactModelStudy.Hardware.JointLimits import load_joint_limits, configure_limit_cost, clip_absolute_target
from ContactModelStudy.Hardware.LCM import DEFAULT_URL, DELTA_ACTION_CHANNEL, FeedbackSkewError, FeedbackTimingError, LCMFeedback, load_pose_transform
from ContactModelStudy.Hardware.Scene import JOINTS
from ContactModelStudy.Hardware.RunLogging import RunLogger, quaternion_angle, utc_now
from ContactModelStudy.Hardware.Workflow import build_tasks, build_rollout, make_evaluator, seed_evaluator, evaluator_activation
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI

ROOT = Path(__file__).resolve().parent


def model_limits(model):
    if (model.nq, model.nv, model.nu, model.na) != (23, 22, 16, 16):
        raise ValueError("Hand/object scene must have nq=23, nv=22, nu=16 and 16 servo activations")
    for i, name in enumerate(JOINTS):
        if (model.joint(i).name != name or model.jnt_qposadr[i] != i
                or model.jnt_dofadr[i] != i or model.actuator_trnid[i, 0] != i
                or model.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_FILTEREXACT
                or model.actuator_actadr[i] != i or model.actuator_dynprm[i, 0] <= 0):
            raise ValueError(f"Unsupported calibrated servo at {name}")
    if model.joint("obj_joint").qposadr[0] != 16:
        raise ValueError("Object free joint must follow the 16 hand joints")
    return control_limits(model)


def check_state(q, v):
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise RuntimeError("Non-finite measured state")


def planning_feedback(buffer, limits, record=None):
    """Briefly tolerate skew before planning; never use an invalid state.

    No plan exists yet, so wait for a coherent pair before starting its clock.
    A genuinely stale/invalid stream still fails immediately. A skew-only wait
    is bounded by the existing robot-age limit and each stream's remaining age.
    """
    started = time.monotonic()
    max_wait = limits['max_robot_age']
    retries = 0
    while True:
        try:
            return buffer.snapshot(**limits)
        except FeedbackSkewError as exc:
            retries += 1
            waited = time.monotonic()-started
            remaining = min(max_wait-waited,
                limits['max_robot_age']-exc.metadata['robot_age'],
                limits['max_pose_age']-exc.metadata['pose_age'])
            if record is not None:
                record['planning_feedback_sync'] = dict(retries=retries,wait_s=waited,
                    max_wait_s=max_wait,last_rejected_pair=exc.metadata)
            if remaining <= 0:
                raise FeedbackTimingError('Planning feedback did not synchronize before freshness deadline') from exc
            time.sleep(min(.002,remaining))
            if record is not None:
                record['planning_feedback_sync']['wait_s'] = time.monotonic()-started
        except RuntimeError as exc:
            if record is not None:
                record['planning_feedback_error'] = str(exc)
            raise


def command_feedback(buffer, limits, planned_metadata, iteration_started, max_plan_seconds, record=None):
    """Read the fresh command base within the ORIGINAL plan/state deadlines.

    Planning already required synchronized hand/object input. Rebasing its
    delta uses only the newest joints, so a camera-frame skew must not delay
    publication. All individual freshness/validity checks remain enforced,
    along with the original planned state's age and the plan's runtime limit.
    """
    def remaining_budget():
        now, monotonic = time.time(), time.monotonic()
        ages = dict(iteration_wall_time_s=monotonic-iteration_started,
                    robot_age_s=now-planned_metadata['robot_timestamp']*1e-6,
                    pose_age_s=now-planned_metadata['pose_timestamp']*1e-6,
                    max_plan_seconds=max_plan_seconds, max_robot_age=limits['max_robot_age'],
                    max_pose_age=limits['max_pose_age'])
        remaining = min(max_plan_seconds-ages['iteration_wall_time_s'],
                        limits['max_robot_age']-ages['robot_age_s'],
                        limits['max_pose_age']-ages['pose_age_s'])
        if remaining <= 0:
            if record is not None:
                record['rejection'] = ages
            raise FeedbackTimingError('Planned state is stale: no action published')
        return remaining
    remaining_budget()
    try:
        state = buffer.snapshot(**limits, require_sync=False)
    except RuntimeError as exc:
        if record is not None:
            record['command_feedback_error'] = str(exc)
        raise
    remaining_budget()  # lock acquisition/read must not consume the plan's deadline
    state[2]['synchronization_required'] = False
    return state


def wait_for_control_deadline(deadline):
    """Wait only for unused control-period time; never catch up after overruns."""
    started = time.monotonic()
    remaining = deadline - started
    if remaining <= 0:
        return 0.0
    while remaining > 0:
        time.sleep(remaining)
        remaining = deadline - time.monotonic()
    return time.monotonic() - started


def build_parser():
    parser = reference.build_parser()
    parser.description = __doc__
    # Reuse the task/episode/planner/buffer options verbatim. Hardware logs
    # have their own measured-state format, so don't advertise video/NPY flags.
    keep_output = {"results", "uncertainty", "debug"}
    for group in parser._action_groups:
        if group.title == "output":
            for action in list(group._group_actions):
                if action.dest not in keep_output:
                    group._group_actions.remove(action)
                    parser._remove_action(action)
                    for option in action.option_strings:
                        parser._option_string_actions.pop(option, None)
    parser.set_defaults(save_stride=1, action_rate_weight=0.0, results=str(ROOT / "results" /
                        f"leap_hardware_{time.strftime('%Y%m%d_%H%M%S')}" / 'controller.json'))
    parser.add_argument("--logging", action=argparse.BooleanOptionalAction, default=True,
                        help="save run parameters, per-step metrics and episode summaries (default: enabled)")
    parser.add_argument("--model", type=Path, default=None,
                        help="explicit scene override; otherwise use task/hand-acc/obj-acc selectors")
    parser.add_argument("--calibrated-hand", type=Path,
                        default=ROOT / "scenes/leap/hardware/calibrated_hand.xml")
    parser.add_argument('--joint-limit-weight', type=float, default=None,
                        help='hardware raw-target limit penalty; default: joint_limits.cost_weight in hardware config; 0 disables cost only')
    parser.add_argument("--contact-param", action="append", default=[], metavar="NAME=JSON",
                        help="override a backend contact/solver field after preset and task tuning")
    parser.add_argument('--cube-friction', nargs=3, type=float, default=None,
                        metavar=('SLIDING', 'TORSIONAL', 'ROLLING'),
                        help='override cube collision-geometry friction for this run; default retains scene values')
    for surface, description in (('fingertip', 'all four fingertip pads'),
                                 ('palm', 'palm collision surfaces'),
                                 ('finger', 'other finger/thumb collision surfaces, excluding fingertip pads')):
        parser.add_argument(f'--{surface}-friction', nargs=3, type=float, default=None,
                            metavar=('SLIDING', 'TORSIONAL', 'ROLLING'),
                            help=f'override friction for {description}; default retains scene values')
    parser.add_argument("--hardware", "--lcm", action="store_true")
    parser.add_argument('--pace-control', action=argparse.BooleanOptionalAction, default=False,
                        help='cap hardware target updates at 1/ctrl-time-step; slow plans publish immediately '
                             '(default: unpaced, as fast as possible)')
    parser.add_argument('--plan-blend-alpha', type=float, default=1.0,
                        help='new-plan weight in optional time-aligned two-plan delta blending, '
                             '0..1; 1 preserves the original execution path (default)')
    parser.add_argument('--plan-blend-metrics', action='store_true',
                        help='log within-plan and time-aligned between-plan delta differences, '
                             'also with alpha=1; blending runs log these automatically')
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs/leap_hardware.json")
    parser.add_argument("--episode-status-file", type=Path,
                        help="atomic episode-phase/prompt JSON for the experiment launcher")
    parser.add_argument("--goal-status-file", type=Path,
                        help="atomic current goal JSON for the synchronized viewer")
    parser.add_argument('--video-padding-seconds', type=float, default=0.,
                        help='stationary recording time before and after manipulation; launcher sets 1 with --record-video')
    parser.add_argument('--fingertip-path-file', type=Path,
                        help='publish the weighted-mean fingertip prediction for the MuJoCo viewer')
    parser.add_argument('--fingertip-path-hz', type=float, default=5.,
                        help='maximum background fingertip prediction rate (default: 5 Hz)')
    parser.add_argument('--fingertip-device', help='optional preview compute device; default matches planning')
    parser.add_argument("--pose-transform", "--camera-config", type=Path,
                        default=ROOT / "configs/leap_pose_transform.json")
    parser.add_argument('--camera-calibration',choices=['original','latest'],
                        help='use original calibration with -15 mm correction, or latest Method 5 fit; --camera-config describes the tracker publication')
    parser.add_argument("--lcm-url", default=DEFAULT_URL)
    parser.add_argument("--state-channel", default="LEAP_STATE")
    parser.add_argument("--pose-channel", default="POSE")
    parser.add_argument("--action-channel", default=DELTA_ACTION_CHANNEL)
    parser.add_argument("--feedback-timeout", type=float, default=10)
    parser.add_argument("--max-robot-age", type=float, default=0.15)
    parser.add_argument("--max-pose-age", type=float, default=0.3)
    parser.add_argument("--max-state-skew", type=float, default=0.15)
    parser.add_argument("--max-plan-seconds", type=float, default=0.2)
    parser.add_argument("--goal-quat", nargs=4, type=float, metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--goal-position", nargs=3, type=float,
                        help="override the task's object target position, in world meters")
    parser.add_argument("--measured-reference", action="store_true",
                        help="use starting measured hand/object positions for home/target costs")
    parser.add_argument('--object-joint-reference', action=argparse.BooleanOptionalAction, default=True,
                        help='load configs/leap_joint_references/<object>.json for w_joint when present')
    parser.add_argument('--joint-reference-file', type=Path,
                        help='explicit per-object joint-cost reference JSON; does not change initial pose')
    parser.add_argument("--device", default="cuda")
    return parser


def parseArgs(argv=None):
    parser = build_parser()
    args = reference.parseArgs(argv, parser=parser, results_prefix="leap_hardware")
    if args.cube_friction is not None:
        if args.task != 'cube_reorient':
            parser.error('--cube-friction requires --task cube_reorient')
    for surface in ('cube', 'fingertip', 'palm', 'finger'):
        friction = getattr(args, f'{surface}_friction')
        if friction is not None and (not np.isfinite(friction).all() or np.any(np.asarray(friction) < 0)):
            parser.error(f'--{surface}-friction values must be finite and nonnegative')
    args.contact_params = {}
    reserved = {"timestep", "gravity", "device", "substeps", "ctrl_time_step", "horizon", "time_horizon"}
    for item in args.contact_param:
        name, separator, raw = item.partition("=")
        if not separator or name in reserved:
            parser.error("--contact-param wants a contact/solver NAME=JSON; use schedule flags for timing")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if isinstance(value, float) and not np.isfinite(value):
            parser.error("--contact-param must be finite")
        args.contact_params[name] = value
    for name in ("steps", "n_episodes", "n_samples", "timestep", "n_iterations",
                 "noise_sigma", "temperature", "max_plan_seconds", "feedback_timeout",
                 "max_robot_age", "max_pose_age", "max_state_skew", 'fingertip_path_hz'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.delta is not None and (not np.isfinite(args.delta) or args.delta <= 0):
        parser.error("--delta must be finite and positive")
    if args.observe_only and not args.hardware:
        parser.error("--observe-only requires --hardware/--lcm")
    if args.pace_control and not args.hardware:
        parser.error('--pace-control requires --hardware/--lcm')
    if not np.isfinite(args.video_padding_seconds) or args.video_padding_seconds < 0:
        parser.error('--video-padding-seconds must be finite and nonnegative')
    if args.video_padding_seconds and (not args.hardware or args.observe_only or args.goal_status_file is None):
        parser.error('--video-padding-seconds requires commanding hardware and --goal-status-file')
    if not np.isfinite(args.plan_blend_alpha) or not 0 <= args.plan_blend_alpha <= 1:
        parser.error('--plan-blend-alpha must be finite and between 0 and 1')
    if args.plan_blend_alpha < 1 or args.plan_blend_metrics:
        if not args.hardware or args.observe_only:
            parser.error('plan blending/metrics require commanding --hardware (not --observe-only)')
        if args.plan_blend_metrics and not args.logging:
            parser.error('--plan-blend-metrics requires logging')
    if args.hardware:
        try:
            if args.camera_calibration is not None:
                from camera_hand_calibration.selection import selected_camera_config
                args.pose_transform=selected_camera_config(args.camera_calibration,args.pose_transform)
            config = json.loads(args.hardware_config.expanduser().read_text())
            measured_limits = load_joint_limits(config)
            if args.joint_limit_weight not in (None, 0) and measured_limits is None:
                raise ValueError('--joint-limit-weight requires joint_limits in hardware config')
            mode = 'ctrl_relative' if delta_reference(config) == 'previous_target' else 'pos_relative'
        except (OSError, TypeError, ValueError) as exc:
            parser.error(f'Invalid hardware configuration: {exc}')
        flags = sys.argv[1:] if argv is None else argv
        if any(flag == '--control-mode' or flag.startswith('--control-mode=') for flag in flags) and args.control_mode != mode:
            parser.error(f'Hardware delta_reference selects --control-mode {mode}; change the hardware config to switch')
        args.control_mode = mode
    elif args.joint_limit_weight is not None:
        parser.error('--joint-limit-weight requires --hardware/--lcm')
    if args.joint_limit_weight is not None and (not np.isfinite(args.joint_limit_weight) or args.joint_limit_weight < 0):
        parser.error('--joint-limit-weight must be finite and nonnegative')
    for name in ("goal_quat", "goal_position"):
        if getattr(args, name) is not None and not np.isfinite(getattr(args, name)).all():
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.cost_weights and not np.isfinite(list(args.cost_weights.values())).all():
        parser.error("--cost-weight must be finite")
    args.results = None if args.results is None else Path(args.results).expanduser()
    if args.joint_reference_file is not None and not args.object_joint_reference:
        parser.error('--joint-reference-file conflicts with --no-object-joint-reference')
    if not args.logging:
        args.results = None
    return args


def publish_goal(path, task, episode, current_step=0, total_steps=None, episode_records=None):
    """Publish the adopted cost target, including measured-position overrides."""
    if path is None:
        return
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(episode=episode, updated_at=time.time(),
                  target_pos=task.params["target_pos"].tolist(),
                  target_quat=task.getGoal().tolist(),
                  success_thresholds=dict(task.SUCCESS_THRESHOLDS))
    if total_steps is not None:
        record.update(current_step=current_step, total_steps=total_steps)
    if episode_records is not None:
        # Reuse the existing 10 Hz viewer telemetry, rather than adding recording
        # I/O to every control step. Recording padding is confirmed only while
        # the controller is actually holding; never authorize future footage.
        record['episode_windows'] = []
        for e in episode_records:
            end = e.get('recording_finished_at_utc', e.get('finished_at_utc'))
            window = dict(episode=e['episode'], started_at_utc=e['started_at_utc'],
                finished_at_utc=e.get('finished_at_utc'),
                confirmed_through=record['updated_at'] if not end else None, outcome=e['finish_reason'])
            for key in ('recording_started_at_utc','recording_finished_at_utc'):
                if key in e:window[key]=e[key]
            record['episode_windows'].append(window)
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(record, allow_nan=False) + '\n')
    pending.replace(path)


def close_video_window(path, task, episode, episode_records, outcome):
    """Close interrupted manipulation before any supervised hold or zero return."""
    if episode is None:
        return
    padded = 'recording_finished_at_utc' in episode
    if episode.get('completed') and (not padded or episode['recording_finished_at_utc'] is not None):
        return
    episode.setdefault('finished_at_utc', utc_now())
    if not episode.get('completed'):
        episode['finish_reason'] = outcome
    if padded:
        episode['recording_finished_at_utc'] = utc_now()
    try:
        publish_goal(path, task, episode['episode'], episode['n_steps'], episode_records=episode_records)
    except (OSError, ValueError) as exc:
        # Recorder telemetry must not prevent motor cleanup or mask the fault.
        print(f'Could not publish video episode end: {exc}', flush=True)


def video_hold(args, task, episode, records, hardware, target=None):
    """Outside the planning loop: hold while confirming recorder time at 10Hz."""
    next_update = 0.
    def update():
        nonlocal next_update
        now = time.monotonic()
        if now >= next_update:
            publish_goal(args.goal_status_file, task, episode['episode'], episode['n_steps'], args.steps, records)
            next_update = now + .1
    hardware.hold_for(args.video_padding_seconds, target=target, on_tick=update)


def capture_failed_plan(planner, rollout, q, v, u, activation, path=None):
    """Failure-only diagnostics; successful planning performs no extra copies."""
    def clean(value):
        if isinstance(value,np.ndarray):return clean(value.tolist())
        if isinstance(value,np.generic):return clean(value.item())
        if isinstance(value,dict):return {key:clean(item) for key,item in value.items()}
        if isinstance(value,(tuple,list)):return [clean(item) for item in value]
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    details={}
    arrays=dict(q=np.asarray(q),q_dot=np.asarray(v),u=np.asarray(u),activation=np.asarray(activation))
    try:
        costs=planner.costs_wp.numpy()
        arrays['costs']=costs
        finite=costs[np.isfinite(costs)]
        details.update(samples=int(costs.size),finite_costs=int(np.isfinite(costs).sum()),
            nan_costs=int(np.isnan(costs).sum()),positive_infinite_costs=int(np.isposinf(costs).sum()),
            negative_infinite_costs=int(np.isneginf(costs).sum()),
            minimum_finite_cost=float(finite.min()) if finite.size else None,
            maximum_finite_cost=float(finite.max()) if finite.size else None)
        start=getattr(planner,'last_plan_start',None)
        if start is not None:
            details['planner_start']=clean(start)
            arrays['planner_start_json']=np.asarray(json.dumps(clean(start),allow_nan=False))
        for field in ['V_wp','eps_wp']:
            if hasattr(planner,field):arrays[field]=getattr(planner,field).numpy()
        state=rollout.DeviceState()
        for field in ['qpos','qvel','site_xpos']:
            value=getattr(state,field,None)
            if value is not None:
                array=value.numpy();arrays['rollout_final_'+field]=array
                valid=np.isfinite(array).reshape(array.shape[0],-1).all(axis=1)
                details['finite_final_'+field+'_worlds']=int(valid.sum())
        if path is not None:
            np.savez(path,**arrays)
            details['snapshot_file']=str(path)
    except Exception as exc:
        details['diagnostic_error']=f'{type(exc).__name__}: {exc}'
    return details


def main(argv=None):
    args = parseArgs(argv)
    previous_target = args.hardware and args.control_mode == 'ctrl_relative'
    require_target = previous_target and not args.observe_only
    run_started = time.monotonic()
    log = dict(mode="lcm" if args.hardware else "simulation", status="starting",
               args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               argv=list(sys.argv[1:] if argv is None else argv), started_at_utc=utc_now(),
               episodes=[], mujoco_version=mujoco.__version__, warp_version=wp.__version__,
               python_version=sys.version)
    logger = RunLogger(args.results) if args.results is not None else None
    feedback = rollout = simulation = hardware_episodes = prediction = None
    episode = pending_step = None
    cycle_started = manipulation_started = None
    config = {}
    episode_index = 0
    def abort(sig, frame):
        # Launcher faults/window-close use SIGTERM, without a zero-return move.
        raise SystemExit(128 + sig)
    previous_sigterm = signal.signal(signal.SIGTERM, abort)
    try:
        with wp.ScopedDevice(args.device), tempfile.TemporaryDirectory(prefix="leap_mppi_") as directory:
            config = json.loads(args.hardware_config.expanduser().read_text())
            measured_limits = load_joint_limits(config) if args.hardware else None
            if args.hardware:
                target_down, target_up = delta_limits(config)
                log['control_policy'] = ('previous_target_plus_bounded_delta' if previous_target
                                         else 'raw_mppi_delta_final_hardware_target_clip')
            task, eval_task, scenes = build_tasks(args, directory,
                hardware_config=config if args.hardware else None)
            log.update(hardware_config=config, scenes=scenes, weights=task.params['cost_weights'],
                       success_thresholds=task.SUCCESS_THRESHOLDS, failure_z=task.FAILURE_Z,
                       metric_definitions=dict(pos='object position L2 error, meters',
                           quat='1 - dot(object quaternion, goal quaternion)^2',
                           quaternion_error_rad='orientation angle in radians',
                           vel='norm of object linear and angular velocity (6D)'),
                       execution=dict(controller='synchronous plan/publish loop',
                           hardware_loop_paced=args.hardware and args.pace_control,
                           bridge='separate unpaced process', nominal_control_frequency_hz=None,
                           preparation_command='absolute_position_waypoint' if args.hardware else None))
            log['action_rate_cost'] = dict(weight=args.action_rate_weight,
                definition='weight * sum_h ||target[h] - target[h-1]||^2',
                first_reference='previous applied/estimated target passed as Plan(u)',
                targets='absolute requested position targets before actuator filtering',
                timestep_normalized=False)
            if args.hardware:
                for current_task in (task, eval_task):
                    current_task.params["init_ctrl"] = vector(config["robot_qref"], "robot_qref").copy()
            model = eval_task.mjm
            lower, upper = model_limits(model)
            rollout = build_rollout(args, task, lower, upper)
            cfg = rollout.config
            log['rollout_config'] = asdict(cfg)
            period = cfg.control_timestep
            blending = (TemporalPlanBlend(args.plan_blend_alpha, period)
                        if args.plan_blend_alpha < 1 or args.plan_blend_metrics else None)
            log['temporal_plan_blending'] = dict(alpha=args.plan_blend_alpha,
                enabled=args.plan_blend_alpha < 1, metrics=blending is not None and args.logging,
                space='incremental_joint_delta', history_plans=1,
                alignment='elapsed_publication_time_zero_order_hold',
                expired_history='use_new_plan', warm_start_modified=False)
            planner_cfg = configure_limit_cost(reference.buildPlannerConfig(args), measured_limits,
                                               args.joint_limit_weight)
            if measured_limits is not None:
                print(f'Joint-limit target cost: weight={planner_cfg.joint_limit_weight:g}; '
                      f'measured bounds={measured_limits.model}', flush=True)
            log['joint_limits'] = (measured_limits.metadata() | dict(
                cost_weight=planner_cfg.joint_limit_weight,
                cost_definition='weight * sum_hj (max(lower_j-target_hj,0)^2 + max(target_hj-upper_j,0)^2)',
                cost_targets='raw requested absolute targets before filtering; includes final horizon target',
                hardware_clip_order=['ctrl_range', 'measured_joint_limits'])
                if measured_limits is not None else None)
            if args.device == "cpu":
                planner_cfg.use_graph = False
            planner = MPPI(rollout, task, planner_cfg)
            planner_cfg = planner.config
            log['planner_config'] = asdict(planner_cfg)
            q, v, u = eval_task.getInitialState()
            if args.hardware:
                before, after, transform_config = load_pose_transform(args.pose_transform)
                verify_model_tag(transform_config, model)
                log["pose_transform"] = transform_config
                feedback = LCMFeedback(model, args.lcm_url, args.state_channel, args.pose_channel,
                                       args.action_channel, before, after,
                                       **({'target_feedback': True} if require_target else {}))
                if not args.observe_only:
                    hardware_episodes = HardwareEpisodes(feedback, config, args.max_robot_age,
                                                         args.episode_status_file, args.logging)
            rollout.activation.assign(u.astype(np.float32))
            planner.Reset(u if args.control_mode == "absolute" else None)
            warmup_started = time.monotonic()
            try:
                planner.Plan(q, v, u=u)
            finally:
                log['warmup_wall_time_s'] = time.monotonic() - warmup_started
            if not planner.last_plan_ok:
                raise RuntimeError("MPPI warmup failed")
            if args.fingertip_path_file:
                try:
                    prediction = FingertipPrediction(args, task, lower, upper, args.fingertip_path_file)
                    prediction.start(planner.last_mean, q, v, u, u)
                    log['fingertip_prediction'] = dict(enabled=True, rate_hz=args.fingertip_path_hz,
                        device=str(prediction.device), source='mppi_weighted_mean_rollout')
                except Exception as exc:
                    prediction = None
                    error = f'{type(exc).__name__}: {exc}'
                    log['fingertip_prediction'] = dict(enabled=False, error=error)
                    write_prediction(args.fingertip_path_file, dict(status='error', error=error))
                    print(f'Fingertip preview unavailable: {error}', flush=True)
            limits = dict(max_robot_age=args.max_robot_age, max_pose_age=args.max_pose_age,
                          max_skew=args.max_state_skew)
            if args.hardware:
                q, v, _ = feedback.wait(args.feedback_timeout, require_target=require_target, **limits)
                u = (feedback.buffer.snapshot_target(args.max_robot_age)[0] if require_target
                     else q[:16].copy())
                simulation = None
            else:
                simulation = make_evaluator(args, eval_task)
            check_state(q, v)
            activation, activation_time = u.copy(), time.monotonic()
            tau = model.actuator_dynprm[:, 0]
            log.update(control_period=period, joint_order=list(JOINTS), hardware_config=config,
                       effective_ctrl_range=(dict(reference='previous_target' if previous_target else 'current_joint_position',
                           application='final_hardware_target',
                           lower=target_down.tolist(), upper=target_up.tolist())
                           if args.hardware else None),
                       model_ctrl_range=dict(lower=lower.tolist(), upper=upper.tolist()),
                       weights=task.params["cost_weights"], scenes=scenes,
                       joint_reference=task.jointReference().tolist(),
                       rollout_config=asdict(cfg), planner_config=asdict(planner_cfg),
                       rollout_model=args.rollout_model, task_contact_params=task.contactParams(args.rollout_model),
                       action_publishing=args.hardware and not args.observe_only)
            log.update(action_representation="joint_delta" if args.hardware else "absolute_target",
                       hardware_loop_paced=args.hardware and args.pace_control,
                       u_source='estimated_bridge_target' if args.hardware else 'simulation_control',
                       plan_base_source='bridge_target_feedback' if require_target else 'measured_position')
            log['execution']['nominal_control_frequency_hz'] = 1 / period
            log['execution']['control_frequency_limit_hz'] = 1 / period if args.hardware and args.pace_control else None
            fixed_goal = (np.asarray(args.goal_quat, dtype=float).copy() if args.goal_quat is not None
                          else eval_task.getGoal()) if args.hardware else None
            log.update(goal_policy='fixed' if args.hardware else 'simulation_sampler',
                       fixed_goal_quat=None if fixed_goal is None else fixed_goal.tolist())
            if logger is not None:
                logger.checkpoint(log)
            print(f"{log['mode']}: {args.task}, {args.rollout_model}, N={args.n_samples}, H={cfg.resolved_horizon}, "
                  f"period={period:.3f}s, device={args.device}", flush=True)
            if args.hardware:
                print(f"Hardware target updates: capped at {1/period:.3f} Hz; overruns publish immediately."
                      if args.pace_control else "Hardware target updates: unpaced, as fast as possible.", flush=True)
            for episode_index in range(args.n_episodes):
                if blending is not None:
                    blending.reset()
                if prediction is not None:
                    prediction.clear()
                cycle_started = time.monotonic()
                episode = None
                if simulation is not None:
                    q, v, u = eval_task.getInitialState()
                    seed_evaluator(simulation, q, v, u)
                    simulation.Step(round(args.settle / eval_task.timestep))
                    q, v = simulation.GetState()
                # Hardware keeps the task's default (or explicit) orientation
                # across episodes. The tracker supplies the starting pose.
                # Publish before the operator's start gate for placement.
                if fixed_goal is not None:
                    goal = fixed_goal.copy()
                elif args.goal_quat is not None:
                    goal = args.goal_quat
                else:
                    goal = eval_task.sampleNewGoal()
                for current_task in (task, eval_task):
                    if args.goal_position is not None:
                        current_task.setTargetPosition(args.goal_position)
                    current_task.setGoal(goal)
                publish_goal(args.goal_status_file, task, episode_index, 0, args.steps, log['episodes'])
                if feedback is not None:
                    if hardware_episodes is not None:
                        hardware_episodes.before_episode(episode_index)
                        # The operator may take arbitrarily long to place the
                        # object. Start from fresh synchronized feedback only.
                        q, v, _ = hardware_episodes.wait_for_tracking(episode_index, args.feedback_timeout, **limits)
                        u = (feedback.buffer.snapshot_target(args.max_robot_age)[0] if require_target
                             else (vector(config['robot_qref'], 'robot_qref') if args.action_rate_weight > 0
                                   else q[:16].copy()))
                        activation, activation_time = u.copy(), time.monotonic()
                    else:
                        q, v, _ = planning_feedback(feedback.buffer,limits)
                for current_task in (task, eval_task):
                    if args.measured_reference:
                        current_task.params["init_ctrl"] = q[:16].copy()
                        current_task.setJointReference(q[:16])
                        current_task.setTargetPosition(q[16:19])
                    if args.goal_position is not None:
                        current_task.setTargetPosition(args.goal_position)
                task.setGoal(goal)
                eval_task.setGoal(goal)
                next_progress_publish = time.monotonic() + .1
                planner.Reset(u if args.control_mode == "absolute" else None)
                manipulation_started = time.monotonic()
                next_control_deadline = manipulation_started + period
                episode = dict(episode=episode_index, goal=task.getGoal().tolist(),
                               target_pos=task.params["target_pos"].tolist(),
                               home=task.jointReference().tolist(), fallen_z=task.params["fallen_z"],
                               steps=[], finish_reason="running", completed=False, n_steps=0,
                               started_at_utc=utc_now())
                log["episodes"].append(episode)
                log['status'] = 'running'
                if args.video_padding_seconds:
                    episode.update(recording_started_at_utc=episode['started_at_utc'],
                                   recording_finished_at_utc=None, video_padding_seconds=args.video_padding_seconds)
                    hardware_episodes.status('video_lead_in', episode_index,
                        f'Episode {episode_index+1}: recording {args.video_padding_seconds:g}s before MPPI; holding reference.')
                    publish_goal(args.goal_status_file, task, episode_index, 0, args.steps, log['episodes'])
                    video_hold(args, task, episode, log['episodes'], hardware_episodes, target=hardware_episodes.reference)
                    manipulation_started = time.monotonic()
                    next_control_deadline = manipulation_started + period
                    episode['started_at_utc'] = utc_now()
                    hardware_episodes.status('running', episode_index, f'Episode {episode_index+1}: MPPI starting.')
                if args.task == 'cube_reorient':
                    # Padding may be a second long. Refresh from received LCM
                    # data after it, without a motor read or camera request.
                    if args.video_padding_seconds and feedback is not None:
                        q, v, _ = planning_feedback(feedback.buffer, limits)
                    episode['starting_orientation'] = describe_start(q[19:23], task.getGoal())
                    for line in terminal_lines(episode_index, episode['starting_orientation'], log['episodes']):
                        print(line, flush=True)
                    manipulation_started = time.monotonic()
                    next_control_deadline = manipulation_started + period
                    episode['started_at_utc'] = utc_now()
                publish_goal(args.goal_status_file, task, episode_index, 0, args.steps, log['episodes'])
                for step in range(args.steps):
                    started = time.monotonic()
                    planning_sync = {}
                    if feedback is not None:
                        try:
                            q, v, metadata = planning_feedback(feedback.buffer,limits,planning_sync)
                        except RuntimeError:
                            episode['feedback_failure'] = dict(step=step,**planning_sync)
                            raise
                        state_ready_started = time.monotonic()
                        now = time.monotonic()
                        activation = u + (activation - u) * np.exp(-(now - activation_time) / tau)
                        activation_time = now
                        if require_target:
                            u, target_metadata = feedback.buffer.snapshot_target(args.max_robot_age)
                            metadata.update(target_metadata, applied_target=u.tolist())
                        if args.observe_only:
                            u = q[:16].copy()
                            activation = u.copy()
                    else:
                        q, v = simulation.GetState()
                        activation, metadata = evaluator_activation(simulation, activation), {}
                    if feedback is not None:
                        check_state(q, v)
                    elif not np.isfinite(q).all() or not np.isfinite(v).all():
                        episode["finish_reason"] = "failure"
                        break
                    if q[18] < task.FAILURE_Z:
                        episode["finish_reason"] = "failure"
                        break
                    error = eval_task.goalErrors(q, v)
                    if all(error[key] < value for key, value in task.SUCCESS_THRESHOLDS.items()):
                        if args.stop_on_success or hardware_episodes is not None:
                            episode["finish_reason"] = "success"
                            if hardware_episodes is not None:
                                feedback.publish_hold()
                                episode["success_joint_pose"] = q[:16].tolist()
                            break
                        if not args.hardware:
                            goal = eval_task.sampleNewGoal()
                            task.setGoal(goal)
                            eval_task.setGoal(goal)
                            publish_goal(args.goal_status_file, task, episode_index, episode['n_steps'], args.steps, log['episodes'])
                            planner.Reset(u if args.control_mode == "absolute" else None)
                    rollout.activation.assign(activation.astype(np.float32))
                    if args.logging:
                        errors = eval_task.goalErrors(q, v)
                        pending_step = dict(step=step, timestamp_utc=utc_now(),
                            elapsed_s=started - manipulation_started, goal_errors=errors,
                            quaternion_error_rad=quaternion_angle(errors['quat']),
                            feedback=metadata, action_published=False)
                        pending_step.update(planning_sync)
                    plan_started = time.monotonic()
                    prediction_started = time.time()
                    if prediction is not None:
                        prediction_u, prediction_act = u.copy(), activation.copy()
                    plan_base = u.copy() if previous_target else q[:16].copy()
                    if pending_step is not None:
                        pending_step['plan_base'] = plan_base.tolist()
                        if args.action_rate_weight > 0:
                            pending_step['action_rate_reference'] = u.tolist()
                    try:
                        out = planner.Plan(q, v, u=u)
                    finally:
                        if pending_step is not None:
                            pending_step['planning_time'] = time.monotonic() - plan_started
                    proposed, sigma = out if isinstance(out, tuple) else (out, None)
                    planning_time = time.monotonic() - plan_started
                    if not planner.last_plan_ok:
                        diagnostic=capture_failed_plan(planner,rollout,q,v,u,activation,
                            args.results.parent/f'failed_plan_e{episode_index+1:02d}_s{step:05d}.npz'
                            if args.logging and args.results is not None else None)
                        if pending_step is not None:
                            pending_step.update(q=q.tolist(),q_dot=v.tolist(),u=u.tolist(),
                                activation=activation.tolist(),planner_failure=diagnostic)
                        detail=(f"finite rollout costs {diagnostic['finite_costs']}/{diagnostic['samples']}"
                                if 'finite_costs' in diagnostic else diagnostic.get('diagnostic_error','diagnostics unavailable'))
                        raise RuntimeError(f"MPPI failed: no action published ({detail})")
                    pacing_wait_s = 0.0
                    if feedback is not None and args.pace_control:
                        # Wait before rebasing, so the target uses fresh joints.
                        # command_feedback still checks the original plan/state
                        # ages after the wait; pacing never relaxes freshness.
                        pacing_wait_s = wait_for_control_deadline(next_control_deadline)
                    if pending_step is not None:
                        pending_step['pacing_wait_s'] = pacing_wait_s
                    blend_context = None
                    if feedback is not None:
                        command_q, command_v, command_metadata = command_feedback(
                            feedback.buffer, limits, metadata, state_ready_started, args.max_plan_seconds, pending_step)
                        check_state(command_q, command_v)
                        # The standard planner returns an absolute target. Send
                        # only its delta from the same base used for this plan.
                        delta = vector(proposed, "planned action") - plan_base
                        if blending is not None:
                            blend_started = time.monotonic()
                            delta, blend_context = blending.prepare(planner.last_mean, delta, blend_started)
                            blend_compute_s = time.monotonic() - blend_started
                        if require_target:
                            command_base, target_metadata = feedback.buffer.snapshot_target(args.max_robot_age)
                            metadata.update(command_target_state=target_metadata,
                                            command_target=command_base.tolist())
                        else:
                            command_base = command_q[:16]
                        metadata.update(command_state=command_metadata,
                                        command_qpos=command_q[:16].tolist())
                    # Estimate only the final target the bridge will apply.
                    # Hardware clipping stays in the bridge. Optional temporal
                    # blending modifies the transmitted delta, not the planner.
                    requested_target = command_base + delta if feedback is not None else proposed
                    delta_clipped_target = (clip_joint_target(requested_target, command_base, target_down, target_up)
                                            if feedback is not None else proposed)
                    command = (clip_absolute_target(delta_clipped_target, measured_limits)
                               if feedback is not None else proposed)
                    if pending_step is not None and measured_limits is not None:
                        pending_step['joint_limit_clipping'] = dict(
                            delta_clipped_target=np.asarray(delta_clipped_target).tolist(),
                            estimated_clipped_joints=int(np.count_nonzero(abs(command-delta_clipped_target)>1e-7)),
                            estimated_removed_l2_rad=float(np.linalg.norm(command-delta_clipped_target)))
                    if pending_step is not None and args.action_rate_weight > 0:
                        pending_step.update(
                            planned_action_rate_cost=float(args.action_rate_weight * np.sum((proposed-u)**2)),
                            estimated_action_rate_cost=float(args.action_rate_weight * np.sum((command-u)**2)))
                    action_published_at_utc = None
                    action_timestamp_us = None
                    if feedback is not None:
                        if not args.observe_only:
                            now = time.monotonic()
                            activation = u + (activation - u) * np.exp(-(now - activation_time) / tau)
                            action_timestamp_us = feedback.publish_delta(delta)
                            activation_time = time.monotonic()
                            if blending is not None:
                                blending.commit(blend_context, activation_time)
                            action_published_at_utc = utc_now() if args.logging else None
                            next_control_deadline = activation_time + period
                            u = command
                            if hardware_episodes is not None:
                                hardware_episodes.last_target = command.copy()
                        elif args.pace_control:
                            next_control_deadline = time.monotonic() + period
                    else:
                        simulation.SetControl(command)
                        simulation.Step(round(period / eval_task.timestep))
                        u = command
                    episode['n_steps'] += 1
                    if args.logging:
                        if blend_context is not None:
                            blend_record = blending.metrics(blend_context)
                            raw_requested = command_base + np.asarray(blend_record['unblended_delta'])
                            raw_clipped = clip_joint_target(raw_requested, command_base, target_down, target_up)
                            raw_clipped = clip_absolute_target(raw_clipped, measured_limits)
                            blend_record.update(compute_s=blend_compute_s,
                                estimated_unblended_clipped_joints=int(np.count_nonzero(abs(raw_requested-raw_clipped)>1e-7)),
                                estimated_blended_clipped_joints=int(np.count_nonzero(abs(requested_target-command)>1e-7)),
                                estimated_unblended_clip_removed_l2=float(np.linalg.norm(raw_requested-raw_clipped)),
                                estimated_blended_clip_removed_l2=float(np.linalg.norm(requested_target-command)))
                            pending_step['plan_blending'] = blend_record
                        pending_step.update(q=q.tolist(), q_dot=v.tolist(),
                            u=u.tolist(), planned_u=np.asarray(proposed).tolist(),
                            requested_target=np.asarray(requested_target).tolist(), planning_time=planning_time,
                            action_delta=delta.tolist() if feedback is not None else None,
                            goal=task.getGoal().tolist(), feedback=metadata,
                            action_published_at_utc=action_published_at_utc,
                            action_timestamp_us=action_timestamp_us,
                            uncertainty=None if sigma is None else sigma.tolist(),
                            wall_time_s=time.monotonic() - started,
                            action_published=feedback is not None and not args.observe_only)
                        episode["steps"].append(pending_step)
                        pending_step = None
                    if prediction is not None:
                        prediction.submit(planner.last_mean, q, v, prediction_u, prediction_act,
                                          prediction_started, episode_index, step)
                    # Display telemetry is limited to 10 Hz independently of control.
                    if args.goal_status_file is not None and time.monotonic() >= next_progress_publish:
                        publish_goal(args.goal_status_file, task, episode_index, episode['n_steps'], args.steps, log['episodes'])
                        next_progress_publish = time.monotonic() + .1
                if prediction is not None:
                    prediction.clear()
                if episode['finish_reason'] == 'running':
                    episode['finish_reason'] = 'timeout'
                if feedback is not None:
                    final_sync = {}
                    try:
                        q, v, _ = planning_feedback(feedback.buffer,limits,final_sync)
                    finally:
                        if final_sync:
                            episode['final_feedback_sync'] = final_sync
                else:
                    q, v = simulation.GetState()
                end_errors = eval_task.goalErrors(q, v)
                # Include the state reached by the last allowed action; there
                # may be no next planning iteration in which to detect success.
                if (hardware_episodes is not None and episode["finish_reason"] == "timeout"
                        and q[18] >= task.FAILURE_Z
                        and all(end_errors[key] < value for key, value in task.SUCCESS_THRESHOLDS.items())):
                    episode["finish_reason"] = "success"
                    episode["success_joint_pose"] = q[:16].tolist()
                    feedback.publish_hold()
                episode.update(q_end=q.tolist(), q_dot_end=v.tolist(), goal_errors=end_errors,
                               quaternion_error_rad=quaternion_angle(end_errors['quat']), completed=True,
                               manipulation_wall_time_s=time.monotonic() - manipulation_started,
                               finished_at_utc=utc_now())
                if episode['finish_reason'] == 'success':
                    episode['time_to_success_s'] = episode['manipulation_wall_time_s']
                publish_goal(args.goal_status_file, task, episode_index, episode['n_steps'], args.steps, log['episodes'])
                print(f"episode {episode_index}: {episode['finish_reason']}, "
                      f"{episode['n_steps']} steps, errors={episode['goal_errors']}", flush=True)
                if args.video_padding_seconds:
                    hardware_episodes.status('video_tail', episode_index,
                        f'Episode {episode_index+1}: {episode["finish_reason"]}; holding while recording {args.video_padding_seconds:g}s after MPPI.')
                    video_hold(args, task, episode, log['episodes'], hardware_episodes)
                    episode['recording_finished_at_utc'] = utc_now()
                    publish_goal(args.goal_status_file, task, episode_index, episode['n_steps'], args.steps, log['episodes'])
                if episode["finish_reason"] == "success" and hardware_episodes is not None:
                    log["status"] = "success_hold"
                    if logger is not None:
                        logger.checkpoint(log)
                    hardware_episodes.after_success(episode_index, final=episode_index == args.n_episodes - 1)
                elif hardware_episodes is not None:
                    if logger is not None:
                        logger.checkpoint(log)
                    hardware_episodes.after_unsuccessful(episode_index, episode["finish_reason"])
                if hardware_episodes is not None:
                    episode["q_after_reset"] = hardware_episodes.return_to_zero(episode_index).tolist()
                episode['cycle_wall_time_s'] = time.monotonic() - cycle_started
                log['wall_time_s'] = time.monotonic() - run_started
                if logger is not None:
                    logger.checkpoint(log)
            if hardware_episodes is not None:
                hardware_episodes.status("complete", args.n_episodes - 1,
                    "All episodes complete; zero pose reached. Disabling torque.")
                feedback.publish_stop()
            log["status"] = "complete"
    except FeedbackTimingError as exc:
        log.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        close_video_window(args.goal_status_file, task if episode is not None else None,
                           episode, log['episodes'], 'error')
        if episode is not None and not episode['completed']:
            episode['manipulation_wall_time_s'] = time.monotonic() - manipulation_started
        if pending_step is not None:
            pending_step['wall_time_s'] = time.monotonic() - started
        if hardware_episodes is not None:
            if prediction is not None:
                prediction.clear()
            try:
                q_zero = hardware_episodes.shutdown_after_feedback_error(episode_index, exc)
                log['feedback_error_return'] = dict(zero_reached=True, qpos=q_zero.tolist())
            except BaseException as reset_error:
                log['feedback_error_return'] = dict(zero_reached=False,
                    error=f"{type(reset_error).__name__}: {reset_error}")
                raise
        raise  # Preserve the aborted run and original failure after supervised reset.
    except KeyboardInterrupt:
        log.update(status="interrupted", error="KeyboardInterrupt")
        close_video_window(args.goal_status_file, task if episode is not None else None,
                           episode, log['episodes'], 'interrupted')
        if episode is not None and not episode['completed']:
            episode['manipulation_wall_time_s'] = time.monotonic() - manipulation_started
        if pending_step is not None:
            pending_step['wall_time_s'] = time.monotonic() - started
        if hardware_episodes is not None:
            print("Ctrl-C: MPPI stopped. Returning to zero before torque disable. "
                  "Press Ctrl-C again to stop immediately.", flush=True)
            try:
                q_zero = hardware_episodes.shutdown_on_interrupt(episode_index)
                log["interrupt_return"] = dict(zero_reached=True, qpos=q_zero.tolist())
            except BaseException as exc:
                log["interrupt_return"] = dict(zero_reached=False, error=f"{type(exc).__name__}: {exc}")
                raise
        elif not args.hardware:
            raise
        else:
            log["interrupt_return"] = dict(zero_reached=False, reason="No active hardware control")
    except BaseException as exc:
        log.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                   error=f"{type(exc).__name__}: {exc}")
        close_video_window(args.goal_status_file, task if episode is not None else None,
                           episode, log['episodes'], 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error')
        if episode is not None and not episode['completed']:
            episode['manipulation_wall_time_s'] = time.monotonic() - manipulation_started
        if pending_step is not None:
            pending_step['wall_time_s'] = time.monotonic() - started
        if (isinstance(exc, Exception) and hardware_episodes is not None
                and config.get('hold_on_non_motor_failure', False)):
            if prediction is not None:
                prediction.clear()
            try:
                q_zero = hardware_episodes.shutdown_after_error(episode_index, exc)
                log['software_error_return'] = dict(zero_reached=True, qpos=q_zero.tolist())
            except BaseException as recovery_error:
                log['software_error_return'] = dict(zero_reached=False,
                    error=f'{type(recovery_error).__name__}: {recovery_error}')
                raise
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if episode is not None:
            if not episode['completed']:
                episode['finish_reason'] = 'interrupted' if log['status'] == 'interrupted' else 'error'
                episode['error'] = log.get('error')
                episode.setdefault('manipulation_wall_time_s', time.monotonic() - manipulation_started)
                if pending_step is not None:
                    episode['last_known_goal_errors'] = pending_step['goal_errors']
            episode['cycle_wall_time_s'] = time.monotonic() - cycle_started
            if pending_step is not None:
                pending_step['error'] = log.get('error')
                pending_step.setdefault('wall_time_s', time.monotonic() - started)
                episode['failed_step'] = pending_step
        try:
            if feedback is not None:
                feedback.close()
        finally:
            if prediction is not None:
                prediction.close()
            if rollout is not None:
                rollout.Close()
            if simulation is not None:
                simulation.Close()
            log.update(wall_time_s=time.monotonic() - run_started, finished_at_utc=utc_now())
            if logger is not None:
                logger.close(log)
                print(f"Results: {args.results}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
