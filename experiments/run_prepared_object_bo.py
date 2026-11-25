"""Run one explicitly selected prepared BO cell, or validate only with --check."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def gpu_metrics(manifest):
    output = subprocess.check_output(['nvidia-smi', '-i', ','.join(manifest['gpus']),
        '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = []
    for line in output.strip().splitlines():
        index, uuid, util, used, total, temp = [v.strip() for v in line.split(',')]
        rows.append(dict(index=int(index), uuid=uuid, utilization=float(util),
                         memory_used_mib=float(used), headroom_mib=float(total)-float(used),
                         temperature_c=float(temp)))
    if {r['uuid'] for r in rows} != set(manifest['gpus']):
        raise RuntimeError('GPU identities do not match the prepared configuration')
    return rows


def progress(manifest, cell):
    path = Path(manifest['results']) / cell / 'bo_state.json'
    if not path.exists():
        return dict(completed_trials=0, best_objective=None)
    state = json.loads(path.read_text())
    values = state['func_vals']
    return dict(completed_trials=len(values), best_objective=min(values) if values else None)


def export_winner(manifest, row, cell):
    folder = Path(manifest['results']) / cell
    summary = json.loads((folder / 'bo_summary.json').read_text())
    if summary['n_trials'] != row['n_calls']:
        raise RuntimeError('Search exited before the configured trial budget completed')
    best = summary['best']
    weights = {**{k: row[k] for k in ('w_velo', 'w_joint_velo', 'w_fallen_term')},
               **best['cost_weights']}
    save(folder / 'best_weights.json', dict(trial=best['trial'], objective=best['objective'],
                                          cost_weights=weights, settings=row))
    block = [f"# {row['label']}: shared M1-M4 weights; trial {best['trial']}.",
             '# "cost_weights": {']
    block.extend(f'#     "{name}": {float(value)!r},' for name, value in weights.items())
    block.append('# },')
    (folder / 'best_weights_commented.txt').write_text('\n'.join(block) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=Path(__file__).resolve().parent)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='Validate CSV/source hashes; run no episodes or GPU work')
    mode.add_argument('--cell', type=int, help='Start/resume only this cell (never the entire queue)')
    args = parser.parse_args()
    root = args.experiment.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    for relative, expected in manifest['source_snapshot_sha256'].items():
        actual = hashlib.sha256((Path(manifest['code']) / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Frozen source changed: {relative}')
    for obj in manifest['scenes'].values():
        for scene in obj.values():
            if hashlib.sha256(Path(scene['path']).read_bytes()).hexdigest() != scene['sha256']:
                raise RuntimeError(f"Prepared scene changed: {scene['path']}")
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', MUJOCO_GL='egl',
               WARP_CACHE_PATH=str(root/'warp_cache'), CUDA_CACHE_PATH=str(root/'cuda_cache'))
    command = [manifest['python'], '-u', str(Path(manifest['code'])/'experiments/run_bayes_opt.py'),
               manifest['csv'], '--outdir', manifest['results']]
    if args.check:
        # This branch does not query GPUs, create a planner pool, or launch MPS.
        return subprocess.run(command + ['--check'], cwd=manifest['code'], env=env).returncode
    if args.cell < 0 or args.cell >= len(manifest['searches']):
        parser.error(f"--cell must be between 0 and {len(manifest['searches'])-1}")
    with (root / 'single_search.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for gpu in gpu_metrics(manifest):
            if gpu['utilization'] >= 15 or gpu['memory_used_mib'] >= 1000:
                raise RuntimeError(f'Selected GPU is busy; no existing job was stopped: {gpu}')
        row = manifest['searches'][args.cell]
        cell = f"cell_{args.cell:04d}_{row['label']}"
        state = dict(status='starting', pid=os.getpid(), started_at_utc=timestamp(),
                     cell=cell, gpus=manifest['gpus'], **progress(manifest, cell))
        child, stopping, mps_started = None, False, False

        def stop(signum, frame):
            nonlocal stopping
            stopping = True
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        for key in list(env):
            if key.startswith('CUDA_MPS_'):
                env.pop(key)
        env.update(CUDA_MPS_PIPE_DIRECTORY=tempfile.mkdtemp(prefix='object-bo-mps-', dir='/tmp'),
                   CUDA_MPS_LOG_DIRECTORY=str(root/'mps_logs'))
        (root/'mps_logs').mkdir(exist_ok=True)
        control = str(Path(manifest['mps_bin'])/'nvidia-cuda-mps-control')
        mps_env = dict(env, CUDA_VISIBLE_DEVICES=','.join(manifest['gpus']))
        env['PATH'] = manifest['mps_bin'] + ':' + env['PATH']
        mps_env['PATH'] = env['PATH']
        save(root/'status.json', state)
        try:
            subprocess.run([control, '-d'], env=mps_env, check=True, timeout=20)
            mps_started = True
            for planners in (manifest['preferred_planners_per_gpu'], manifest['fallback_planners_per_gpu']):
                if stopping:
                    break
                logpath = root/f'{cell}_planners{planners}.log'
                cmd = command + ['--cell', str(args.cell), '--gpus', ','.join(manifest['gpus']),
                    '--planners-per-gpu', str(planners), '--workers', str(manifest['workers'])]
                state.update(status='running', command=cmd, planners_per_gpu=planners,
                             current_log=str(logpath), guard_reason=None)
                unsafe = telemetry_errors = 0
                with logpath.open('a') as log, (root/'gpu_metrics.jsonl').open('a', buffering=1) as metrics:
                    child = subprocess.Popen(cmd, cwd=manifest['code'], env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    state['child_pid'] = child.pid
                    while child.poll() is None:
                        try:
                            gpu = gpu_metrics(manifest)
                            telemetry_errors = 0
                            state['gpu_metrics'] = gpu
                            metrics.write(json.dumps(dict(time=timestamp(), cell=cell, gpu=gpu))+'\n')
                            unsafe = unsafe+1 if any(g['headroom_mib'] < 3072 or g['temperature_c'] >= 83 for g in gpu) else 0
                            if unsafe >= 2:
                                state['guard_reason'] = 'GPU headroom below 3 GiB or temperature at/above 83 C twice'
                        except (subprocess.SubprocessError, OSError, ValueError) as exc:
                            telemetry_errors += 1
                            if telemetry_errors >= 3:
                                state['guard_reason'] = f'GPU telemetry failed: {exc}'
                        state.update(updated_at_utc=timestamp(), **progress(manifest, cell))
                        save(root/'status.json', state)
                        if state['guard_reason']:
                            child.send_signal(signal.SIGINT)
                            break
                        time.sleep(5)
                    rc = child.wait(timeout=180)
                child = None
                state.update(last_returncode=rc, **progress(manifest, cell))
                if stopping:
                    break
                if rc == 0 and not state['guard_reason']:
                    export_winner(manifest, row, cell)
                    state['status'] = 'complete'
                    break
            else:
                raise RuntimeError('Both planner allocations stopped; checkpoints retained for resumption')
            if stopping:
                state['status'] = 'stopped'
        except BaseException as exc:
            state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc()
        finally:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                child.wait(timeout=180)
            if mps_started:
                try:
                    subprocess.run([control], input='quit\n', text=True, env=mps_env, check=True, timeout=30)
                except subprocess.SubprocessError as exc:
                    state['mps_cleanup_error'] = str(exc)
            state.update(finished_at_utc=timestamp(), **progress(manifest, cell))
            save(root/'status.json', state)
        return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
