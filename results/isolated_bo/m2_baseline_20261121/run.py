"""Validate a frozen setup, or explicitly start/resume its one 500-trial search."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

OUT=Path(__file__).resolve().parent


def now():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def progress(manifest):
    cell='cell_0000_'+manifest['row']['label']
    path=Path(manifest['results'])/cell/'bo_state.json'
    state=read_json(path,{})
    values=state.get('func_vals',[])
    partial=read_json(path.parent/'resume_progress.json',{})
    return dict(completed_trials=len(values),best_objective=min(values) if values else None,
        partial_trials=partial.get('partial_trials',0),saved_partial_episodes=partial.get('saved_partial_episodes',0))


def gpus(manifest):
    from gpu_watchdog import query_gpus
    rows=query_gpus()
    by_index={g['index']:g for g in rows}
    for index,uuid in zip(manifest['physical_gpu_indices'],manifest['gpus']):
        assert by_index[index]['uuid']==uuid, 'GPU index/UUID mapping changed'
    return rows


def read_json(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def enable_secondary(state, timestamp, stale_seconds=20):
    return bool(state and state.get('healthy') and
                timestamp-state.get('updated_at_unix',0)<stale_seconds and state.get('secondary_requested'))


class PrivateMPS:
    def __init__(self,manifest,env):
        self.manifest,self.env=manifest,env
        self.pipes={g:tempfile.mkdtemp(prefix=f'm2-baseline-gpu{i}-mps-',dir='/tmp')
                    for i,g in zip(manifest['physical_gpu_indices'],manifest['gpus'])}
        self.active=set()
        self.control=str(Path(manifest['mps_bin'])/'nvidia-cuda-mps-control')
    def environment(self,gpu):
        index=self.manifest['gpus'].index(gpu)
        log=OUT/'mps_logs'/f'gpu{self.manifest["physical_gpu_indices"][index]}'
        log.mkdir(parents=True,exist_ok=True)
        return dict(self.env,CUDA_VISIBLE_DEVICES=gpu,CUDA_MPS_PIPE_DIRECTORY=self.pipes[gpu],
                    CUDA_MPS_LOG_DIRECTORY=str(log))
    def start(self,gpu):
        if gpu not in self.active:
            subprocess.run([self.control,'-d'],env=self.environment(gpu),check=True,timeout=20)
            self.active.add(gpu)
    def stop(self,gpu):
        if gpu in self.active:
            subprocess.run([self.control],input='quit\n',text=True,env=self.environment(gpu),check=True,timeout=20)
            self.active.remove(gpu)


def export_best(manifest):
    cell=Path(manifest['results'])/('cell_0000_'+manifest['row']['label'])
    summary=json.loads((cell/'bo_summary.json').read_text())
    assert summary['n_trials']==500
    best=summary['best'];weights={**manifest['fixed_weights'],**best['cost_weights']}
    save(OUT/'best_parameters.json',dict(trial=best['trial'],objective=best['objective'],weights=weights,
        settings=best,baseline='baseline.json',publication='export only; active files untouched'))
    lines=['# M2 baseline BO: best completed 500-trial candidate; not activated.','# "cost_weights": {']
    lines.extend(f'#     "{k}": {v!r},' for k,v in weights.items());lines+=['# },']
    (OUT/'best_weights_commented.txt').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true',help='CPU validation only; no trials/GPU/MPS/hardware')
    mode.add_argument('--start',action='store_true',help='Explicitly start/resume this one search')
    args=parser.parse_args();manifest=json.loads((OUT/'manifest.json').read_text())
    code=Path(manifest['code'])
    for path,digest in manifest['source_snapshot_sha256'].items():assert sha(code/path)==digest,path
    for path,digest in manifest['assets_sha256'].items():assert sha(OUT/path)==digest,path
    for path,digest in manifest.get('supervisor_sha256',{}).items():assert sha(OUT/path)==digest,path
    assert sha(OUT/'baseline.json')==manifest['baseline_snapshot_sha256']
    assert sha(OUT/'initial_states.json')==manifest['initial_states_sha256']
    assert sha(OUT/'search.csv')==manifest['search_csv_sha256']
    assert sha(OUT/'seed_points.csv')==manifest['seed_points_sha256']
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MUJOCO_GL='egl',
        WARP_CACHE_PATH=str(OUT/'warp_cache'),CUDA_CACHE_PATH=str(OUT/'cuda_cache'))
    command=[manifest['python'],'-u',str(code/'run_search.py'),manifest['csv'],'--outdir',manifest['results']]
    if args.check:
        return subprocess.run(command+['--check'],cwd=code,env=env).returncode
    with (OUT/'single_search.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from gpu_watchdog import process_identity, cleanup_orphan
        previous=read_json(OUT/'owned_processes.json',{})
        old=previous.get('search',{})
        if old.get('pid') and old.get('identity') and process_identity(old['pid'])==old['identity']:
            raise RuntimeError('Previous search is still exiting; wait for watchdog cleanup before resuming')
        old_watch=previous.get('watchdog',{})
        if old_watch.get('pid') and old_watch.get('identity') and process_identity(old_watch['pid'])==old_watch['identity']:
            raise RuntimeError('Previous watchdog is still cleaning up; wait before resuming')
        if previous.get('mps_pipes'):
            cleanup_orphan(OUT,manifest,previous.get('supervisor_pid'),previous.get('supervisor_identity'))
        state=dict(status='starting',started_at_utc=now(),pid=os.getpid(),gpus=manifest['gpus'],
                   trial_budget=500,**progress(manifest))
        if state['completed_trials']>=500:
            export_best(manifest);return 0
        primary,secondary=manifest['gpus']
        available={g['uuid']:g for g in gpus(manifest)}
        gpu=available[primary]
        if gpu['utilization']>=15 or gpu['memory_used_mib']>=1000:
            raise RuntimeError(f'Primary GPU 2 is busy; no process stopped and no GPU migration: {gpu}')
        stopping=False;child=None;watchdog=None
        def stop(signum,frame):
            nonlocal stopping
            stopping=True
            if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
        for key in list(env):
            if key.startswith('CUDA_MPS_'):env.pop(key)
        env['PATH']=manifest['mps_bin']+':'+env['PATH']
        mps=PrivateMPS(manifest,env)
        owned=dict(mps_pipes=mps.pipes,supervisor_pid=os.getpid(),supervisor_identity=process_identity(os.getpid()))
        save(OUT/'owned_processes.json',owned)
        env.update(M2_BO_MPS_PIPES=json.dumps(mps.pipes),M2_BO_SECONDARY_GPU=secondary,
                   M2_BO_GPU_CONTROL=str(OUT/'gpu_control.json'),M2_BO_GPU_ACK=str(OUT/'gpu_ack.json'),
                   M2_BO_SUPERVISOR_PID=str(os.getpid()))
        watchdog_command=[manifest['python'],'-u',str(OUT/'gpu_watchdog.py'),
                          '--manifest',str(OUT/'manifest.json'),'--parent-pid',str(os.getpid())]
        def start_watchdog():
            (OUT/'watchdog_state.json').unlink(missing_ok=True)
            with (OUT/'watchdog.log').open('a') as log:
                process=subprocess.Popen(watchdog_command,env=env,cwd=OUT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            owned['watchdog']=dict(pid=process.pid,identity=process_identity(process.pid))
            save(OUT/'owned_processes.json',owned)
            return process
        try:
            watchdog=start_watchdog()
            mps.start(primary)
            for planners in (20,16):
                if stopping:break
                path=OUT/f'bo_planners{planners}.log'
                cmd=command+['--cell','0','--gpus',','.join(manifest['gpus']),
                             '--planners-per-gpu',str(planners),'--workers','48']
                save(OUT/'gpu_control.json',dict(secondary_enabled=False))
                (OUT/'gpu_ack.json').unlink(missing_ok=True)
                state.update(status='running',command=cmd,planners_per_gpu=planners,
                    workers=48,trials_in_flight=3,current_log=str(path),guard_reason=None,
                    primary_gpu_index=2,secondary_gpu_index=3,watchdog_pid=watchdog.pid)
                unsafe=0
                with path.open('a') as log:
                    child=subprocess.Popen(cmd,cwd=code,env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    owned['search']=dict(pid=child.pid,identity=process_identity(child.pid))
                    save(OUT/'owned_processes.json',owned)
                    state['child_pid']=child.pid
                    while child.poll() is None:
                        if watchdog.poll() is not None:
                            watchdog=start_watchdog()
                            state['watchdog_pid']=watchdog.pid
                        health=read_json(OUT/'watchdog_state.json',{})
                        desired=enable_secondary(health,time.time(),manifest['gpu_watchdog']['stale_seconds'])
                        rows={g['uuid']:g for g in health.get('gpus',[])}
                        secondary_metrics=rows.get(secondary,{})
                        primary_metrics=rows.get(primary,{})
                        if primary_metrics:
                            unsafe=unsafe+1 if (primary_metrics['headroom_mib']<3072 or primary_metrics['temperature_c']>=83) else 0
                            if unsafe>=2:state['guard_reason']='Primary GPU headroom below 3GiB or temperature >=83C twice'
                        if secondary_metrics and (secondary_metrics['headroom_mib']<3072 or secondary_metrics['temperature_c']>=83):
                            desired=False
                        current=read_json(OUT/'gpu_control.json',{}).get('secondary_enabled',False)
                        if desired and secondary not in mps.active:
                            # It is currently unclaimed: do not take a busy secondary.
                            desired=bool(secondary_metrics and secondary_metrics['utilization']<15 and
                                         secondary_metrics['memory_used_mib']<1000 and
                                         secondary_metrics['headroom_mib']>=3072 and secondary_metrics['temperature_c']<83)
                            if desired:
                                try:mps.start(secondary)
                                except (OSError,subprocess.SubprocessError) as exc:
                                    desired=False;state['secondary_start_error']=str(exc)
                        if desired!=current:
                            save(OUT/'gpu_control.json',dict(secondary_enabled=desired,updated_at_utc=now()))
                            with (OUT/'gpu_events.jsonl').open('a') as events:
                                events.write(json.dumps(dict(time=now(),secondary_enabled=desired,
                                    reason=health.get('reason','watchdog unavailable')))+'\n')
                        ack=read_json(OUT/'gpu_ack.json',{})
                        released=(ack.get('pool_pid')==child.pid and ack.get('secondary_released') is True)
                        if not desired and released and secondary in mps.active:
                            try:mps.stop(secondary)
                            except (OSError,subprocess.SubprocessError) as exc:
                                state['secondary_stop_error']=str(exc)
                        state.update(updated_at_utc=now(),watchdog_healthy=bool(health.get('healthy')),
                            secondary_gpu_state='enabled' if desired else ('released' if released else 'draining'),
                            watchdog_reason=health.get('reason'),**progress(manifest))
                        save(OUT/'status.json',state)
                        if state['guard_reason']:child.send_signal(signal.SIGINT);break
                        time.sleep(5)
                    rc=child.wait(timeout=180);child=None
                owned.pop('search',None);save(OUT/'owned_processes.json',owned)
                state.update(returncode=rc,**progress(manifest))
                if stopping:break
                if not rc and not state['guard_reason']:
                    if state['completed_trials']<500:
                        raise RuntimeError('Search ended early; partial episodes saved. Resume with --start.')
                    export_best(manifest);state['status']='complete';break
                text=path.read_text()[-65536:].lower()
                resource=state['guard_reason'] or 'out of memory' in text or 'cuda_error_out_of_memory' in text
                if planners==20 and resource:
                    state['fallback_reason']=state['guard_reason'] or '20-planner allocation failed';continue
                raise RuntimeError(f'BO stopped with code {rc}; checkpoints retained; inspect {path}')
            else:raise RuntimeError('Both planner allocations failed; checkpoints retained')
            if stopping:state['status']='stopped'
        except BaseException as exc:
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
        finally:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            for gpu in list(mps.active):
                try:mps.stop(gpu)
                except (OSError,subprocess.SubprocessError) as exc:state['mps_cleanup_error']=str(exc)
            if watchdog is not None and watchdog.poll() is None:
                watchdog.terminate()
                try:watchdog.wait(timeout=12)
                except subprocess.TimeoutExpired:watchdog.kill();watchdog.wait(timeout=5)
            owned.pop('search',None);owned.pop('watchdog',None);save(OUT/'owned_processes.json',owned)
            state.update(finished_at_utc=now(),**progress(manifest));save(OUT/'status.json',state)
    return 0


if __name__=='__main__':raise SystemExit(main())
