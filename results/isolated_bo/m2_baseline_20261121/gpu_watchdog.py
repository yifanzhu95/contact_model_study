"""Load watchdog for this isolated BO. Started only by run.py --start."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def query_gpus():
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=8)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, load, used, total, temperature = [s.strip() for s in line.split(',')]
        row = dict(index=int(index), uuid=uuid, utilization=float(load),
                   memory_used_mib=float(used), headroom_mib=float(total)-float(used),
                   temperature_c=float(temperature))
        import math
        if not all(math.isfinite(row[k]) for k in ('utilization', 'memory_used_mib',
                                                  'headroom_mib', 'temperature_c')):
            raise ValueError('Nonfinite GPU telemetry')
        rows.append(row)
    return rows


class LoadPolicy:
    """Time hysteresis avoids repeatedly destroying/rebuilding CUDA graphs."""
    def __init__(self, config):
        self.config = config
        self.enabled = False
        self.hot_since = self.cool_since = None

    def update(self, metrics, instant):
        observed = {r['index']:r for r in metrics}
        loads = [observed[i]['utilization'] for i in self.config['monitor_gpu_indices']]
        hot = any(v >= self.config['busy_percent'] for v in loads)
        cool = all(v < self.config['idle_percent'] for v in loads)
        self.hot_since = (instant if self.hot_since is None else self.hot_since) if hot else None
        self.cool_since = (instant if self.cool_since is None else self.cool_since) if cool else None
        reason = 'waiting for sustained idle load' if not self.enabled else 'two GPUs allowed'
        if self.hot_since is not None and instant-self.hot_since >= self.config['busy_seconds']:
            self.enabled = False
            reason = 'GPU 0 or 1 sustained increased load'
        elif self.cool_since is not None and instant-self.cool_since >= self.config['idle_seconds']:
            self.enabled = True
            reason = 'GPUs 0 and 1 sustained idle load'
        elif self.enabled:
            reason = 'load hysteresis retains second GPU'
        return dict(secondary_requested=self.enabled, reason=reason,
                    monitored_utilization_percent=dict(zip(map(str,self.config['monitor_gpu_indices']),loads)))


def process_identity(pid):
    try:
        # starttime survives PID reuse checks; strip comm which can contain spaces.
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, ValueError):
        return None


def cleanup_orphan(out, manifest, owner_pid=None, owner_identity=None):
    """Only stop this search's registered process group and private MPS servers."""
    try:
        owned = json.loads((out/'owned_processes.json').read_text())
    except (OSError, ValueError):
        return
    if owner_pid is not None and (owned.get('supervisor_pid') != owner_pid or
                                  owned.get('supervisor_identity') != owner_identity):
        return  # a newer supervisor owns this workspace now
    child = owned.get('search', {})
    pid = child.get('pid')
    if pid and child.get('identity') and process_identity(pid) == child['identity']:
        try:
            os.killpg(pid, signal.SIGINT)
            deadline = time.monotonic()+30
            while process_identity(pid) == child['identity'] and time.monotonic() < deadline:
                time.sleep(.5)
            if process_identity(pid) == child['identity']:
                os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    control = str(Path(manifest['mps_bin'])/'nvidia-cuda-mps-control')
    for gpu, pipe in owned.get('mps_pipes', {}).items():
        env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=pipe, CUDA_VISIBLE_DEVICES=gpu)
        try:
            subprocess.run([control], input='quit\n', text=True, env=env,
                           timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    out = Path(manifest['out'])
    parent_identity = process_identity(args.parent_pid)
    policy = LoadPolicy(manifest['gpu_watchdog'])
    running = True
    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    with (out/'gpu_metrics.jsonl').open('a',buffering=1) as log:
        while running:
            if parent_identity is None or process_identity(args.parent_pid) != parent_identity:
                cleanup_orphan(out,manifest,args.parent_pid,parent_identity)
                return 1
            stamp = time.time()
            try:
                metrics = query_gpus()
                by_index = {r['index']:r for r in metrics}
                for index,uuid in zip(manifest['physical_gpu_indices'],manifest['gpus']):
                    if by_index[index]['uuid'] != uuid:
                        raise ValueError('Selected GPU index/UUID mapping changed')
                state = dict(healthy=True, gpus=metrics, **policy.update(metrics,time.monotonic()))
            except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
                policy.enabled=False
                policy.cool_since=policy.hot_since=None
                state = dict(healthy=False,secondary_requested=False,reason=f'telemetry unavailable: {exc}',gpus=[])
            state.update(updated_at_unix=stamp, updated_at_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid())
            atomic_json(out/'watchdog_state.json',state)
            log.write(json.dumps(state)+'\n')
            deadline=time.monotonic()+manifest['gpu_watchdog']['poll_seconds']
            while running and time.monotonic()<deadline:
                time.sleep(.25)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
