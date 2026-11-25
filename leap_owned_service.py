#!/usr/bin/env python3
"""Start/stop one owned service inside its PID namespace (Python 3.8+).

The launcher invokes both modes in the same owned container. No CUDA libraries
are imported. An exec preserves PID/start time; stop verifies that identity
before each signal, so a reused PID is never signaled.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def identity(pid):
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory/'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        uid = next(line.split()[1] for line in (directory/'status').read_text().splitlines()
                   if line.startswith('Uid:'))
        return dict(pid=int(pid), start_ticks=int(fields[19]), uid=int(uid),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def run_service(path, command):
    if not command:
        raise ValueError('Missing service command')
    record = identity(os.getpid())
    if record is None:
        raise RuntimeError('Cannot identify owned service process')
    pending = path.with_suffix('.tmp')
    # Open with restricted permissions from creation, including in containers.
    fd = os.open(str(pending), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(record, stream)
    pending.replace(path)
    try:
        os.execvp(command[0], command)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def stop_service(path, timeout):
    record = json.loads(path.read_text())
    pid = record['pid']
    if (isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1
            or pid == os.getpid() or record['uid'] != os.getuid()):
        raise ValueError('Invalid owned service identity')
    def alive():
        return identity(pid) == record
    for sig, grace in ((signal.SIGINT, timeout), (signal.SIGTERM, 2.), (signal.SIGKILL, 2.)):
        if not alive():
            break
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while alive() and time.monotonic() < deadline:
            time.sleep(.02)
    if alive():
        raise RuntimeError('Owned service did not exit; GPU memory release is unconfirmed')
    path.unlink(missing_ok=True)
    print('Owned service exit verified; CUDA context can be released.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    start = modes.add_parser('run')
    start.add_argument('--pid-file', type=Path, required=True)
    start.add_argument('command', nargs=argparse.REMAINDER)
    stop = modes.add_parser('stop')
    stop.add_argument('--pid-file', type=Path, required=True)
    stop.add_argument('--timeout', type=float, default=10.)
    args = parser.parse_args(argv)
    if args.mode == 'run':
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        run_service(args.pid_file, command)
    else:
        import math
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error('--timeout must be finite and positive')
        stop_service(args.pid_file, args.timeout)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
