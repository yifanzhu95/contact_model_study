"""Persistent, owned vision processes shared by sequential hardware experiments."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time


VISION_SERVICES = frozenset(('container', 'sam', 'mask', 'tracker', 'viewer'))


def process_identity(pid):
    """Avoid signaling an unrelated process after PID reuse or a reboot."""
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory/'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        real_uid = next(line.split()[1] for line in (directory/'status').read_text().splitlines()
                        if line.startswith('Uid:'))
        return dict(pid=int(pid), start_ticks=int(fields[19]), pgid=int(fields[2]),
                    uid=int(real_uid),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def process_alive(identity):
    return bool(identity) and process_identity(identity['pid']) == identity


def tracking_spec(config, mesh, scale):
    camera = Path(config['camera_config']).expanduser().resolve()
    return dict(mesh=str(Path(mesh).resolve()), mesh_sha256=hashlib.sha256(Path(mesh).read_bytes()).hexdigest(),
                mesh_scale=scale, camera_config=str(camera), camera_sha256=hashlib.sha256(camera.read_bytes()).hexdigest(),
                camera_index=config['camera_index'], lcm_url=config['lcm_url'], pose_channel=config['pose_channel'],
                viewer_bind=config['viewer_bind'], viewer_endpoint=config['viewer_endpoint'],
                tracker_args=config.get('tracker_args', []), viewer_args=config.get('viewer_args', []),
                interpreters={key: config.get('interpreters', {}).get(key) for key in ('vision', 'viewer')},
                docker=config.get('docker', {}),
                cuda_visible_devices=(config.get('tracking_gpu_assignment') or
                                      config.get('gpu_assignment') or {}).get('cuda_visible_devices'))


class TrackingSession:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory/'session.json'
        self.lock = None
        self.record = None

    def acquire(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory/'session.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.release()
            raise RuntimeError('Another experiment is using this tracking session; stop that experiment first') from None
        return self

    def release(self):
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def load(self):
        if not self.path.exists():
            self.record = None
            return None
        try:
            self.record = json.loads(self.path.read_text())
            if self.record.get('version') != 1 or not isinstance(self.record.get('processes'), dict):
                raise ValueError('unsupported session record')
            return self.record
        except (OSError, ValueError, AttributeError) as exc:
            raise RuntimeError(f'Cannot read tracking session {self.path}: {exc}') from exc

    def save(self):
        pending = self.path.with_suffix('.tmp')
        pending.write_text(json.dumps(self.record, indent=2) + '\n')
        pending.chmod(0o600)
        pending.replace(self.path)

    def check_compatible(self, spec):
        previous = self.record['spec']
        changed = [key for key in spec if previous.get(key) != spec[key]]
        if changed:
            raise RuntimeError('Running tracking uses different settings (' + ', '.join(changed)
                               + '); pass --restart-tracking to apply the new settings')
        if not process_alive(self.record['processes'].get('tracker')):
            raise RuntimeError('The saved tracker is no longer running; pass --restart-tracking to recapture and track again')

    def register(self, experiment, commands, spec):
        processes = {name: process_identity(child.process.pid) for name, child in experiment.children.items()
                     if name in VISION_SERVICES and name not in ('mask', 'container')
                     and name not in getattr(experiment, 'completed_services', ())}
        if not processes.get('tracker'):
            raise RuntimeError('Tracker exited before its persistent session could be recorded')
        container = None
        if experiment.container is not None:
            executable, name = experiment.container
            result = subprocess.run([executable, 'inspect', '--format', '{{.Id}}', name],
                                    capture_output=True, text=True, timeout=5, check=True, start_new_session=True)
            identifier = result.stdout.strip()
            if not re.fullmatch('[0-9a-f]{64}', identifier):
                raise RuntimeError('Could not identify the owned vision container')
            container = [executable, identifier]
            # A replacement container reusing the name must never be stopped.
            experiment.container = tuple(container)
        tracker = commands['tracker']
        mask = (Path(tracker[tracker.index('--init_mask_path') + 1]) if '--init_mask_path' in tracker
                else experiment.directory/'mask.png')
        mask_snapshot = mask
        if mask.parent != experiment.directory:
            mask_snapshot = experiment.directory/('tracking_mask' + mask.suffix)
            shutil.copy2(mask, mask_snapshot)
        self.record = dict(version=1, directory=str(experiment.directory), spec=spec, processes=processes,
                           container=container, commands={name: argv for name, argv in commands.items()
                                                          if name in VISION_SERVICES},
                           mask_snapshot=str(mask_snapshot),
                           temporary_workspace=not experiment.logging_enabled, started_at=time.time())
        self.save()

    def replace_viewer(self, child):
        self.record['processes']['viewer'] = process_identity(child.process.pid)
        self.save()

    def finished(self, skip=()):
        for name in ('tracker', 'viewer'):
            if name not in skip and not process_alive(self.record['processes'].get(name)):
                return (f'{name} stopped in the persistent tracking session', 0 if name == 'viewer' else 1)
        return None

    def stop(self, grace):
        """Stop only identities recorded by this launcher, under its session lock."""
        if self.record is None:
            return
        identities = list(self.record['processes'].values())
        def send(sig):
            for identity in identities:
                if (process_alive(identity) and identity['uid'] == os.getuid()
                        and identity['pgid'] == identity['pid']):
                    try:
                        os.killpg(identity['pgid'], sig)
                    except ProcessLookupError:
                        pass
        send(signal.SIGINT)
        container_error = None
        if self.record.get('container'):
            executable, identifier = self.record['container']
            if not re.fullmatch('[0-9a-f]{64}', identifier):
                raise RuntimeError('Invalid owned container identity in tracking session')
            try:
                result = subprocess.run([executable, 'stop', '--time', '2', identifier],
                                        capture_output=True, text=True, timeout=10, start_new_session=True)
                if result.returncode and 'No such container' not in result.stderr:
                    container_error = result.stderr.strip()
            except (OSError, subprocess.TimeoutExpired) as exc:
                container_error = str(exc)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and any(process_alive(item) for item in identities):
            time.sleep(.05)
        send(signal.SIGTERM)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and any(process_alive(item) for item in identities):
            time.sleep(.05)
        send(signal.SIGKILL)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and any(process_alive(item) for item in identities):
            time.sleep(.05)
        if any(process_alive(item) for item in identities):
            raise RuntimeError('Owned vision processes did not stop; tracking will not be restarted')
        if container_error:
            raise RuntimeError(f'Could not stop the owned vision container: {container_error}')
        workspace = Path(self.record['directory'])
        # Only automatically generated, temporary operational workspaces are
        # disposable. Logged experiment folders are retained for analysis.
        if (self.record.get('temporary_workspace') and workspace.parent == Path(tempfile.gettempdir())
                and workspace.name.startswith('leap_experiment_')):
            shutil.rmtree(workspace, ignore_errors=True)
        self.path.unlink(missing_ok=True)
        self.record = None
