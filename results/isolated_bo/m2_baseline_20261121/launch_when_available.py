"""Queue the authorized BO on its fixed primary GPU without displacing jobs."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gpu_watchdog import atomic_json, query_gpus
from run import progress

OUT=Path(__file__).resolve().parent


def main():
    if sys.argv[1:] != ['--start']:
        raise SystemExit('Explicit start required: launch_when_available.py --start')
    manifest=json.loads((OUT/'manifest.json').read_text())
    running=True
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    with (OUT/'queued_launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'single_search.lock').open('a') as search_lock:
            fcntl.flock(search_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle_since=None
        atomic_json(OUT/'queued_launch.json',dict(status='queued',pid=os.getpid()))
        print(f'Queued launch PID {os.getpid()}; waiting for GPU 2, no device migration.',flush=True)
        while running:
            stamp=datetime.now(timezone.utc).isoformat()
            try:
                rows=query_gpus()
                selected={g['index']:g for g in rows}
                for index,uuid in zip(manifest['physical_gpu_indices'],manifest['gpus']):
                    if selected[index]['uuid'] != uuid:
                        raise RuntimeError('Fixed GPU UUID/index mapping changed; refusing migration')
                primary=selected[2]
                available=(primary['utilization']<15 and primary['memory_used_mib']<1000
                    and primary['headroom_mib']>=3072 and primary['temperature_c']<83)
                idle_since=(time.monotonic() if idle_since is None else idle_since) if available else None
                idle_for=0 if idle_since is None else time.monotonic()-idle_since
                state=dict(status='queued_waiting_for_primary_gpu',updated_at_utc=stamp,
                    pid=os.getpid(),primary_gpu=primary,idle_seconds=round(idle_for,1),
                    required_idle_seconds=30,**progress(manifest),
                    reason='GPU 2 must be free for 30 seconds; existing jobs are not interrupted')
                atomic_json(OUT/'queued_launch.json',state)
                atomic_json(OUT/'status.json',state)
                if available and idle_for>=30:
                    command=[manifest['python'],'-u',str(OUT/'run.py'),'--start']
                    print(f'{stamp}: GPU 2 available; starting authorized BO.',flush=True)
                    atomic_json(OUT/'queued_launch.json',dict(status='launching',pid=os.getpid(),
                        updated_at_utc=stamp,command=command))
                    # Replace the waiter so the detached PID becomes the supervisor.
                    os.execve(command[0],command,os.environ.copy())
                print(f'{stamp}: GPU 2 load={primary["utilization"]:.0f}% '
                      f'memory={primary["memory_used_mib"]:.0f}MiB; idle {idle_for:.0f}/30s',flush=True)
            except (OSError,subprocess.SubprocessError,ValueError,KeyError) as exc:
                idle_since=None
                state=dict(status='queued_telemetry_unavailable',pid=os.getpid(),updated_at_utc=stamp,error=str(exc))
                atomic_json(OUT/'queued_launch.json',state)
                atomic_json(OUT/'status.json',state)
                print(f'{stamp}: waiting: {exc}',flush=True)
            deadline=time.monotonic()+15
            while running and time.monotonic()<deadline:
                time.sleep(.5)
        atomic_json(OUT/'queued_launch.json',dict(status='cancelled',pid=os.getpid()))
        atomic_json(OUT/'status.json',dict(status='queued_launch_cancelled',**progress(manifest)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
