#!/usr/bin/env python3
"""Launch supervised RL evaluation with the existing vision/LCM supervisor.

    python rl/NoDR_s1009_u0800_partner_package_English_v2/launch_rl_hardware.py --n-episodes 12
    ... --dry-run
    ... --n-episodes 12 -- --goal-sign=-1 --no-stop-on-success

All new files stay inside this package. Original MPPI/tracking/bridge sources
and production camera/hardware configurations are reused without modification.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
RUNTIME = PACKAGE/'runtime'
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['TMPDIR'] = str(RUNTIME/'tmp')
os.environ['XDG_CACHE_HOME'] = str(RUNTIME/'cache')
os.environ['MPLCONFIGDIR'] = str(RUNTIME/'cache/matplotlib')
sys.path.insert(0, str(REPO))

import launch_leap_experiment as shared
from ContactModelStudy.Hardware.TrackingSession import process_alive

ORIGINAL_BUILD_COMMANDS = shared.build_commands


def inside(value):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(PACKAGE):
        raise ValueError(f'Keep RL outputs/configs inside {PACKAGE}: {path}')
    return path


def runner_module():
    # The runner imports NumPy/MuJoCo but imports PyTorch only in load_policy().
    spec = importlib.util.spec_from_file_location('leap_rl_runner', PACKAGE/'run_rl_hardware.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_rl_commands(config, directory, controller_args=()):
    forwarded = list(controller_args)
    reserved = {'--hardware-config', '--camera-config', '--lcm-url', '--state-channel', '--pose-channel',
        '--action-channel', '--results', '--episode-status-file', '--goal-status-file', '--logging',
        '--no-logging', '--observe-only', '--check', '--device', '--hardware'}
    for value in forwarded:
        if value.split('=', 1)[0] in reserved:
            raise ValueError(f'Configure shared setting {value} before -- in the launcher')
    args = runner_module().parse_args(forwarded)
    if config.get('fingertip_paths'):
        raise ValueError('RL has no MPPI horizon/fingertip paths; omit --fingertip-paths')
    shared_args = ['--task', 'cube_reorient', '--calibrated-hand', args.calibrated_hand,
        '--max-robot-age', str(args.max_robot_age), '--max-pose-age', str(args.max_pose_age),
        '--max-state-skew', str(args.max_state_skew)]
    base = copy.deepcopy(config)
    base['controller_args'] = []
    commands = ORIGINAL_BUILD_COMMANDS(base, directory, shared_args)
    control = shared.interpreter(config['interpreters']['control'])
    commands['controller'] = [control, '-B', '-u', str(PACKAGE/'run_rl_hardware.py'), *forwarded,
        '--hardware-config', str(shared.local_path(config['hardware_config'])),
        '--camera-config', str(shared.local_path(config['camera_config'])),
        '--lcm-url', config['lcm_url'], '--state-channel', config['state_channel'],
        '--pose-channel', config['pose_channel'], '--action-channel', config['action_channel'],
        '--results', str(directory/'controller.json'),
        '--episode-status-file', str(directory/'episode_status.json'),
        '--goal-status-file', str(directory/'goal.json'),
        '--logging' if config.get('logging', True) else '--no-logging']
    if config.get('observe_only'):
        commands['controller'].append('--observe-only')
        commands['bridge'].append('--state-only')
    # Actor runs on CPU; discard the original MPPI controller and its CUDA flags.
    return commands


def option_value(argv, option):
    for i, value in enumerate(argv):
        if value == option:
            if i+1 >= len(argv):
                raise ValueError(f'{option} requires a path')
            return argv[i+1]
        if value.startswith(option+'='):
            return value.split('=', 1)[1]
    return None


def borrow_live_tracking(config, source, restart=False):
    """Mirror a live registry privately; hold its existing lock read-only.

    No original registry is rewritten. With explicit restart/stop the mirrored
    owned services may be stopped; a later MPPI launch must restart its old,
    now-stale registry. Never run two hand experiments concurrently.
    """
    private = inside(config['tracking_dir'])
    if (private/'session.json').exists():
        own = json.loads((private/'session.json').read_text())
        borrowed = own.get('borrowed_readonly_from')
        if borrowed and process_alive(own.get('processes', {}).get('tracker')):
            lock = (Path(borrowed).parent/'session.lock').open('rb')
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                lock.close(); raise
            return lock
        return None
    if source is None: return None
    source = Path(source).expanduser().resolve()
    if source.is_dir(): source = source/'session.json'
    if not source.is_file(): return None
    saved = json.loads(source.read_text())
    if not process_alive(saved.get('processes', {}).get('tracker')):
        return None
    if not restart and not process_alive(saved.get('processes', {}).get('viewer')):
        raise ValueError('Saved tracker has no live receiver; use --restart-tracking to start vision inside the RL workspace')
    # Opening read-only and flocking preserves the existing file, while excluding
    # a simultaneous MPPI launch using the same shared tracking service.
    lock = (source.parent/'session.lock').open('rb')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        private.mkdir(parents=True, exist_ok=True)
        saved['borrowed_readonly_from'] = str(source)
        (private/'session.json').write_text(json.dumps(saved, indent=2)+'\n')
        print(f'Reusing live tracking from {source}; RL registry is {private}.', flush=True)
    except BaseException:
        lock.close(); raise
    return lock


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument('--config', default=str(REPO/'configs/leap_experiment.json'), help='source experiment configuration, read only')
    own.add_argument('--runtime-python', '--control-python', default=str(RUNTIME/'venv/bin/python'))
    own.add_argument('--reuse-tracking-from', default=str(REPO/'.runtime/leap_tracking/session.json'))
    own.add_argument('--check', action='store_true', help='offline preflight and exit, without starting services')
    # Keep policy flags after -- untouched for the controller.
    split = argv.index('--') if '--' in argv else len(argv)
    options, common = own.parse_known_args(argv[:split])
    forwarded = argv[split+1:] if split < len(argv) else []
    if '--help' in common or '-h' in common:
        print(__doc__)
        print('Additional RL options: --runtime-python PATH, --reuse-tracking-from REGISTRY, --check')
        print('Policy options after --: --goal-sign alternate|-1|1, --frame-mode palm_registered|identity,')
        print('  --no-stop-on-success, --steps 1..500, --backend pytorch|torchscript.\n')
        return shared.main(['--help'])
    config = json.loads(Path(options.config).expanduser().read_text())
    for key in ['--log-dir', '--run-dir', '--tracking-dir']:
        value = option_value(common, key)
        if value is not None: inside(value)
    if '--fingertip-paths' in common:
        raise ValueError('RL has no MPPI fingertip horizon')
    for name in ['tmp','configs','logs','cache']:
        (RUNTIME/name).mkdir(parents=True, exist_ok=True)
    invocation = RUNTIME/'configs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    invocation.mkdir()
    hardware_source = shared.local_path(config['hardware_config'])
    hardware = json.loads(hardware_source.read_text())
    hardware['delta_reference'] = 'previous_target'
    hardware_path = invocation/'hardware.json'
    hardware_path.write_text(json.dumps(hardware, indent=2)+'\n')
    config.update(hardware_config=str(hardware_path), controller_args=[], fingertip_paths=False,
        tracking_dir=str(RUNTIME/'tracking'))
    if option_value(common, '--tracking-dir'):
        config['tracking_dir'] = str(inside(option_value(common, '--tracking-dir')))
    # Preserve the venv executable symlink: resolving it would invoke base Python
    # and silently discard the package-local PyTorch environment.
    config.setdefault('interpreters', {})['control'] = str(Path(options.runtime_python).expanduser().absolute())
    if not Path(config['interpreters']['control']).is_file():
        raise ValueError('RL runtime missing. See HARDWARE_EVALUATION.txt for the package-local CPU environment setup.')
    if '--dry-run' in common:
        # Offline dry-run does not require nvidia-smi/device access.
        config.update(gpu=None, tracking_gpu=None, mppi_gpu=None)
    experiment = invocation/'experiment.json'
    experiment.write_text(json.dumps(config, indent=2)+'\n')
    if not any(option_value(common, flag) for flag in ['--run-dir','--log-dir']) and '--no-logging' not in common and config.get('logging', True):
        common += ['--log-dir', str(RUNTIME/'results')]
    camera_source = shared.local_path(option_value(common, '--camera-config') or config['camera_config'])
    check_args = [*forwarded, '--hardware-config', str(hardware_path), '--camera-config', str(camera_source)]
    # Resolve flags and world/actuator conventions before the supervisor can enable torque.
    if '--stop-tracking' not in common:
        with (invocation/'preflight.log').open('w') as stream:
            checked = subprocess.run([config['interpreters']['control'], '-B', str(PACKAGE/'run_rl_hardware.py'),
                                      *check_args, '--check'], cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        if checked.returncode:
            print((invocation/'preflight.log').read_text(), file=sys.stderr)
            raise RuntimeError(f'RL preflight failed before any hardware start; see {invocation/"preflight.log"}')
        print('RL CPU policy preflight passed. Original hardware/camera configurations unchanged.', flush=True)
    if options.check:
        print(f'Preflight report: {invocation/"preflight.log"}')
        return 0
    borrow_lock = None
    original_build = shared.build_commands
    old_temp = tempfile.tempdir
    try:
        if '--dry-run' not in common:
            borrow_lock = borrow_live_tracking(config, options.reuse_tracking_from,
                restart='--restart-tracking' in common or '--stop-tracking' in common)
        # Adapter scope is this process only; no shared source files are edited.
        shared.build_commands = build_rl_commands
        tempfile.tempdir = str(RUNTIME/'tmp')
        return shared.main(['--config', str(experiment), *common, '--', *forwarded])
    finally:
        shared.build_commands = original_build
        tempfile.tempdir = old_temp
        if borrow_lock is not None: borrow_lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
