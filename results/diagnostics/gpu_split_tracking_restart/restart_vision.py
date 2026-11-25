"""Operational restart of persistent vision only; hardware services are forbidden."""
from datetime import datetime
import json
from pathlib import Path
import signal
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import launch_leap_experiment as launcher
from ContactModelStudy.Hardware.GPU import select_gpu
from ContactModelStudy.Hardware.TrackingSession import TrackingSession, VISION_SERVICES, tracking_spec


def main():
    config = json.loads((ROOT/'configs/leap_experiment.json').read_text())
    stamp = datetime.now(ZoneInfo('America/Phoenix')).strftime('%Y%m%d_%H%M%S_%f')
    directory = Path(__file__).parent/f'vision_{stamp}'
    directory.mkdir()
    config.update(generate_mask=True, init_mask_path=str(directory/'mask.png'), mujoco_viewer=False)
    tracking_gpu = select_gpu('auto')
    mppi_gpu = select_gpu('auto', exclude_uuids=(tracking_gpu['uuid'],))
    config.update(tracking_gpu_assignment=tracking_gpu, gpu_assignment=mppi_gpu)
    print('Tracking GPU:', tracking_gpu['index'], 'MPPI GPU (not started):', mppi_gpu['index'], flush=True)
    print('Operational files:', directory, flush=True)
    commands = {key: value for key, value in launcher.build_commands(config, directory).items()
                if key in VISION_SERVICES}
    experiment = launcher.Experiment(directory, gpu_assignment=mppi_gpu, tracking_gpu_assignment=tracking_gpu)
    original_start = experiment.start
    def vision_start(name, command):
        assert name in VISION_SERVICES, 'Hardware/controller launch is forbidden'
        return original_start(name, command)
    experiment.start = vision_start
    session = TrackingSession(launcher.local_path(config['tracking_dir']))
    experiment.tracking_session = session
    selected = launcher.controller_selection(config['controller_args'])
    mesh, scale = launcher.tracking_mesh(config, selected)
    experiment.tracking_spec = tracking_spec({**config, 'camera_config': str(launcher.local_path(config['camera_config']))}, mesh, scale)
    state = {'stop': None}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda sig, frame: state.update(stop=signal.Signals(sig).name))
    report = dict(config=config, commands=commands, hardware_started=False, status='starting')
    success = False
    try:
        session.acquire()
        previous = session.load()
        if previous:
            (directory/'previous_session.json').write_text(json.dumps(previous, indent=2)+'\n')
            session.stop(config['shutdown_timeout'])
        result = launcher.prepare_tracking(experiment, commands, config, state)
        if result:
            raise RuntimeError(str(result))
        readiness = launcher.TrackingReadiness(config['tracking_ready_frames'], config['tracking_ready_max_age'])
        result = launcher.wait_stage(experiment, state, config['startup_timeout'],
            lambda: readiness.update(directory/'viewer_heartbeat.json'), 'waiting for fresh tracking')
        if result:
            raise RuntimeError(str(result))
        # Re-evaluate the remaining cards now that tracking is actually active.
        mppi_gpu = select_gpu('auto', exclude_uuids=(tracking_gpu['uuid'],))
        report['config']['gpu_assignment'] = mppi_gpu
        report.update(status='tracking_ready', mppi_selected_index=mppi_gpu['index'])
        success = True
        print('TRACKING READY on GPU', tracking_gpu['index'], '; MPPI selection with vision active:', mppi_gpu['index'], flush=True)
        print('Vision persists in', session.path, '; no hardware processes started.', flush=True)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        experiment.keep_tracking = success
        experiment.shutdown(config['shutdown_timeout'])
        if not success and session.record:
            session.stop(config['shutdown_timeout'])
        report['started_services'] = list(experiment.children)
        (directory/'restart.json').write_text(json.dumps(report, indent=2)+'\n')
        session.release()


if __name__ == '__main__':
    main()
