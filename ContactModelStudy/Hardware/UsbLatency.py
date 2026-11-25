"""Temporary FTDI latency setting, restored by a pipe-owning helper process.

No serial connection, motor command, periodic poll or control-loop hook.
The helper keeps the original sysfs file descriptor, so unplug/replacement
cannot make cleanup write to a different USB device.
"""
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time

MANAGED_ENV = 'LEAP_USB_LATENCY_MANAGED'


def _milliseconds(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 255:
        raise ValueError('USB latency must be an integer between 1 and 255 milliseconds')
    return value


def latency_path(port):
    device = Path(port).expanduser().resolve(strict=True)
    if device.parent != Path('/dev') or not re.fullmatch(r'ttyUSB\d+', device.name):
        raise ValueError(f'Expected an FTDI /dev/ttyUSB device, got {device}')
    path = Path('/sys/bus/usb-serial/devices') / device.name / 'latency_timer'
    if not path.is_file():
        raise FileNotFoundError(f'USB latency timer unavailable: {path}')
    return device, path


def _identity(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, ValueError, TypeError):
        return None


def _read_ms(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    return int(os.read(fd, 128).splitlines()[0])


def _write_ms(fd, value):
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, f'{value}\n'.encode())
    observed = _read_ms(fd)
    if observed != value:
        raise OSError(f'USB latency readback {observed} ms, expected {value} ms')


def _emit(record):
    try:
        print(json.dumps(record), flush=True)
        return True
    except (BrokenPipeError, OSError):
        return False


def guard(path, runtime_ms, restore_ms):
    """Private helper: EOF on stdin restores even after the owner is SIGKILLed."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    # Detach after sudo authentication, preserving the launcher's terminal
    # credential scope while protecting cleanup from its process-group kill.
    try:
        os.setsid()
    except PermissionError:
        pass  # already a session/process-group leader
    fd = None
    owned = False
    result = 0
    try:
        fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owned = True
        previous = _read_ms(fd)
        _write_ms(fd, runtime_ms)
        if _emit(dict(event='ready', previous_ms=previous, runtime_ms=runtime_ms,
                      restore_ms=restore_ms, path=str(path), guard_pid=os.getpid())):
            while sys.stdin.buffer.read(1):
                pass
    except BaseException as exc:
        result = 1
        _emit(dict(event='error', stage='start', message=f'{type(exc).__name__}: {exc}',
                   errno=getattr(exc, 'errno', None)))
    finally:
        if fd is not None:
            try:
                if owned:
                    _write_ms(fd, restore_ms)
                    _emit(dict(event='restored', restored_ms=restore_ms))
            except BaseException as exc:
                result = 2
                _emit(dict(event='error', stage='restore', message=f'{type(exc).__name__}: {exc}'))
            finally:
                os.close(fd)
    return result


class UsbLatencyTimer:
    def __init__(self, port, runtime_ms=1, restore_ms=16):
        self.port = str(port)
        self.runtime_ms = _milliseconds(runtime_ms)
        self.restore_ms = _milliseconds(restore_ms)
        self.process = None
        self.external = self.closed = False
        self.info = dict(port=self.port, runtime_ms=runtime_ms, restore_ms=restore_ms, active=False)

    @property
    def metadata(self):
        return dict(self.info)

    def inherited_environment(self):
        if not self.info['active']:
            raise RuntimeError('USB latency guard is not active')
        return json.dumps(dict(self.info, owner_pid=os.getpid(), owner_identity=_identity(os.getpid())))

    @staticmethod
    def _response(process, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError('USB latency helper did not respond')
        line = process.stdout.readline()
        if not line:
            raise OSError('USB latency helper exited before reporting readiness')
        return json.loads(line)

    def _spawn(self, privileged):
        command = [sys.executable, '-I', str(Path(__file__).resolve()), '--guard',
                   self.port, str(self.runtime_ms), str(self.restore_ms)]
        if privileged:
            command = ['sudo', '-n', '--', '/usr/bin/python3', *command[1:]]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, close_fds=True)
        try:
            response = self._response(process, 10)
            if response.get('event') != 'ready':
                raise OSError(response.get('errno') or errno.EIO, response.get('message', 'USB latency startup failed'))
        except BaseException:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Keep the guard alive to restore if an in-kernel call returns.
                pass
            process.stdout.close()
            process.stderr.close()
            raise
        self.process = process
        self.info.update(response, active=True, privileged=privileged)

    def start(self):
        if self.closed:
            raise RuntimeError('USB latency guard is closed; create a new hand connection')
        if self.info['active']:
            return self
        device, path = latency_path(self.port)
        self.info.update(device=str(device), path=str(path))
        inherited = os.environ.get(MANAGED_ENV)
        if inherited:
            owner = json.loads(inherited)
            if (owner.get('device') != str(device) or owner.get('runtime_ms') != self.runtime_ms
                    or owner.get('restore_ms') != self.restore_ms
                    or not owner.get('owner_identity')
                    or _identity(owner.get('owner_pid')) != owner['owner_identity']):
                raise RuntimeError('Inherited USB latency guard does not match this hand/session')
            if int(path.read_text().strip()) != self.runtime_ms:
                raise OSError('Inherited USB latency setting is no longer active')
            self.external = True
            self.info.update(owner, active=True, managed_by_launcher=True)
            return self
        deadline = time.monotonic() + 10
        while True:
            try:
                try:
                    self._spawn(False)
                except PermissionError:
                    try:
                        self._spawn(True)
                    except BlockingIOError:
                        raise
                    except Exception as exc:
                        raise PermissionError(f'Cannot set USB latency for {device}. Run `sudo -v` in the '
                            'same terminal before launching, or grant write access to the latency_timer. '
                            'No permanent USB configuration was changed.') from exc
                break
            except BlockingIOError as exc:
                # A recovery utility can see the old serial owner disappear
                # just before its launcher's timer cleanup finishes.
                if time.monotonic() >= deadline:
                    raise RuntimeError('USB latency timer is still owned by another hardware run; '
                                       'finish stopping that run before retrying') from exc
                time.sleep(.1)
        print(f'USB latency: {device}, {self.info["previous_ms"]} -> {self.runtime_ms} ms; '
              f'restore {self.restore_ms} ms at termination.', flush=True)
        return self

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.external:
            self.info['active'] = False
            return  # the launcher keeps 1ms until all motor cleanup finishes
        process = self.process
        if process is None:
            return
        try:
            process.stdin.close()
            process.wait(timeout=10)
            messages = [json.loads(line) for line in process.stdout if line.strip()]
            if process.returncode or not any(m.get('restored_ms') == self.restore_ms for m in messages):
                raise OSError(f'USB latency restoration unconfirmed: {messages}')
            self.info.update(active=False, restored_ms=self.restore_ms)
            print(f'USB latency restored: {self.info["device"]}, {self.restore_ms} ms (verified).', flush=True)
        except Exception as exc:
            self.info.update(active=False, error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            process.stdout.close()
            process.stderr.close()


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] != '--guard':
        raise SystemExit('Private USB latency helper; use the hardware runner')
    _, timer = latency_path(sys.argv[2])
    raise SystemExit(guard(timer, _milliseconds(int(sys.argv[3])), _milliseconds(int(sys.argv[4]))))
