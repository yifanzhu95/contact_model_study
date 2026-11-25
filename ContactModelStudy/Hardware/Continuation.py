"""Resume a logged hardware batch without overwriting or recounting attempts."""
import copy
import hashlib
import json
from pathlib import Path
from ContactModelStudy.Hardware.TrackingSession import process_alive


GENERATED_VALUES = {'--episode-status-file', '--goal-status-file', '--results',
    '--camera-config', '--pose-transform', '--hardware-config', '--pose-channel',
    '--state-channel', '--action-channel', '--lcm-url', '--video-padding-seconds',
    '--device', '--fingertip-device', '--fingertip-path-file', '--n-episodes'}
GENERATED_FLAGS = {'--hardware', '--lcm', '--logging', '--no-logging'}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def controller_options(command):
    """Remove launcher transport/output options, preserving planner CLI exactly."""
    start = next((i+1 for i, value in enumerate(command)
                  if Path(value).name == 'run_leap_hardware.py'), None)
    if start is None:
        raise ValueError('Saved launch has no hardware controller command')
    result = []
    args = command[start:]
    i = 0
    while i < len(args):
        value = args[i]
        option = value.split('=', 1)[0]
        if option in GENERATED_VALUES:
            i += 1 if '=' in value else 2
        elif option in GENERATED_FLAGS:
            i += 1
        else:
            result.append(value)
            i += 1
    return result


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read continuation input {path}: {exc}') from exc


def progress(plan, current=None):
    """Count completed source episodes once, independently of crash summaries."""
    paths = [Path(p) for p in plan['source_runs']]
    if current is not None:
        paths.append(Path(current).resolve())
    completed, aborted = [], []
    for directory in dict.fromkeys(paths):
        controller = directory/'controller.json'
        doc = _read(controller) if controller.is_file() else {}
        failed_ids = set()
        for e in doc.get('episodes', []):
            local = e['episode'] + 1
            if e.get('completed'):
                if e.get('finish_reason') not in ('success', 'failure', 'timeout'):
                    raise ValueError(f'Invalid completed episode in {controller}')
                completed.append(dict(batch_episode=len(completed)+1,
                    run_directory=str(directory), source_episode=local,
                    outcome=e['finish_reason']))
            else:
                failed_ids.add(local)
        recording = directory/'video/recording.json'
        if recording.exists():
            # Older crashes sometimes lost their final controller checkpoint.
            failed_ids.update(e['episode'] for e in _read(recording).get('episodes', [])
                              if e.get('outcome') in ('error', 'interrupted'))
        aborted.extend(dict(run_directory=str(directory), source_episode=i)
                       for i in sorted(failed_ids))
    total = plan['target_episodes']
    if len(completed) > total:
        raise ValueError('Continuation contains more completed episodes than the original batch requested')
    return dict(target_episodes=total, completed_episodes=len(completed),
        remaining_episodes=total-len(completed), completed_episode_sources=completed,
        aborted_attempts=aborted)


def load(directory, root):
    directory = Path(directory).expanduser().resolve()
    launch = _read(directory/'launch.json')
    if launch.get('status') in ('starting', 'running') and process_alive(launch.get('launcher_process')):
        raise ValueError('Source launcher is still running; stop it before continuing')
    config = copy.deepcopy(launch['config'])
    if not config.get('logging', True) or config.get('observe_only', False):
        raise ValueError('--continue-run requires a logged hardware run, not observe-only')
    controller = _read(directory/'controller.json') if (directory/'controller.json').exists() else {}
    previous = launch.get('continuation')
    sources = list(previous['source_runs']) if previous else []
    if str(directory) in sources:
        raise ValueError('Continuation ancestry contains a cycle')
    sources.append(str(directory))
    command = launch.get('commands', {}).get('controller', [])
    requested = controller.get('args', {}).get('n_episodes')
    if requested is None:
        for i, value in enumerate(command):
            if value == '--n-episodes': requested = int(command[i+1])
            elif value.startswith('--n-episodes='): requested = int(value.split('=', 1)[1])
    total = previous['target_episodes'] if previous else requested
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise ValueError('Original episode count is unavailable; start a new explicit batch')
    plan = dict(schema_version=1, target_episodes=total, source_runs=sources,
        continued_from=str(directory),
        counting='Success, failure and timeout count as completed. Aborted attempts do not. New files keep local episode numbering; completed_episode_sources maps batch numbers.')
    plan.update(progress(plan))
    # Refuse silently changed physical/reference inputs; the user can start a
    # separate batch to intentionally change settings.
    def local(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else Path(root)/p
    snapshots = launch.get('input_snapshots', {})
    expected = controller.get('hardware_config', snapshots.get('hardware_config'))
    if expected is not None and _read(local(config['hardware_config'])) != expected:
        raise ValueError('Hardware configuration changed since the source run; start a new batch or restore it')
    if snapshots.get('camera_config') is not None and _read(local(config['camera_config'])) != snapshots['camera_config']:
        raise ValueError('Camera publication configuration changed since the source run')
    calibration = controller.get('args', {}).get('calibrated_hand')
    expected_hash = snapshots.get('calibrated_hand_sha256')
    if calibration and expected_hash and hashlib.sha256(local(calibration).read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Calibrated hand model changed since the source run')
    limits = controller.get('joint_limits') or snapshots.get('joint_limits') or {}
    if limits.get('model') and limits.get('model_sha256'):
        if hashlib.sha256(local(limits['model']).read_bytes()).hexdigest() != limits['model_sha256']:
            raise ValueError('Measured joint limits changed since the source run')
    selected = controller.get('pose_transform', {}).get('selected_calibration', {})
    if selected.get('source') and selected.get('source_sha256'):
        if hashlib.sha256(local(selected['source']).read_bytes()).hexdigest() != selected['source_sha256']:
            raise ValueError('Selected camera calibration changed since the source run')
    if selected:
        # "latest" may now name a different fit. Reuse the actual selection
        # file of this run while leaving the tracker's publication config alone.
        camera_paths = [command[i+1] for i, value in enumerate(command[:-1])
                        if value == '--camera-config']
        if not camera_paths:
            raise ValueError('Saved controller camera selection is unavailable')
        planning_camera = local(camera_paths[-1])
        camera_snapshot = snapshots.get('selected_camera_config')
        actual_camera = _read(planning_camera)
        if camera_snapshot is not None and actual_camera != camera_snapshot:
            raise ValueError('Saved camera selection file changed since the source run')
        config['continued_camera_config'] = str(planning_camera)
    args = controller.get('args', {})
    reference = args.get('joint_reference_file') or f"configs/leap_joint_references/{args.get('task', 'cube_reorient').split('_')[0]}.json"
    if (args.get('object_joint_reference', True) and not args.get('measured_reference', False)
            and controller.get('joint_reference') is not None):
        reference_file = local(reference)
        if not reference_file.is_file() or _read(reference_file).get('joint_reference') != controller['joint_reference']:
            raise ValueError('Object joint-cost reference changed since the source run')
    forwarded = controller_options(command)
    # Freeze every effective weight, including values that were implicit defaults.
    for name, value in controller.get('weights', {}).items():
        forwarded.extend(['--cost-weight', f'{name}={value}'])
    if controller.get('fixed_goal_quat') is not None:
        forwarded += ['--goal-quat', *map(str, controller['fixed_goal_quat'])]
    episodes = controller.get('episodes', [])
    if episodes and not args.get('measured_reference', False):
        forwarded += ['--goal-position', *map(str, episodes[0]['target_pos'])]
    config['controller_args'] = []
    for key in ('gpu_assignment', 'tracking_gpu_assignment'): config.pop(key, None)
    return plan, config, forwarded
