"""Simulation-only model sweep; independent workers and private MPS on selected GPUs."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parent
PYTHON='/home/chetan/miniconda3/envs/mj_env/bin/python'
MPS=Path('/home/chetan/contact_model_study/results/diagnostics/bo_gpu_concurrency_20261115/private_mps_tools/usr/bin')
GPUS=['GPU-6155fe7d-ffa5-baa3-50cc-c93841698927','GPU-1ada3dc2-9cb0-d2f6-2b12-80346a4659e0']
children={};stopping=False


def now():return datetime.now(timezone.utc).isoformat()


def save(state):
    p=ROOT/'status.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(ROOT/'status.json')


def stop(signum,frame):
    global stopping
    stopping=True
    for record in children.values():
        p=record['process']
        if p.poll() is None:p.send_signal(signal.SIGINT)


def metrics():
    text=subprocess.check_output(['nvidia-smi','-i',','.join(GPUS),
        '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    out=[]
    for line in text.strip().splitlines():
        i,u,used,total,t=map(float,line.split(','))
        out.append(dict(index=int(i),utilization=u,used_mib=used,headroom_mib=total-used,temperature_c=t))
    return out


def main():
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    for key in list(env):
        if key.startswith('CUDA_MPS_'):env.pop(key)
    pipe=tempfile.mkdtemp(prefix='cube-models-mps-',dir='/tmp');(ROOT/'mps_logs').mkdir(exist_ok=True)
    env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',PATH=str(MPS)+':'+env['PATH'],
        CUDA_MPS_PIPE_DIRECTORY=pipe,CUDA_MPS_LOG_DIRECTORY=str(ROOT/'mps_logs'),
        WARP_CACHE_PATH=str(ROOT.parent/'cube_joint_reference_20261116/warp_cache'),CUDA_CACHE_PATH=str(ROOT/'cuda_cache'))
    server_env=dict(env,CUDA_VISIBLE_DEVICES=','.join(GPUS));mps=False
    state=dict(status='starting',pid=os.getpid(),started_at_utc=now(),private_mps_pipe=pipe,
               max_planners_per_gpu=2,no_hardware=True)
    save(state)
    try:
        before=metrics()
        if any(g['utilization']>=30 or g['used_mib']>=4096 for g in before):raise RuntimeError(f'GPUs busy: {before}')
        with (ROOT/'mps_startup.log').open('a') as log:
            subprocess.run([str(MPS/'nvidia-cuda-mps-control'),'-d'],env=server_env,stdout=log,
                stderr=subprocess.STDOUT,timeout=20,check=True)
        mps=True;unsafe=errors=0
        with (ROOT/'gpu_metrics.jsonl').open('a',buffering=1) as logmetrics:
            while not stopping:
                manifest=json.loads((ROOT/'manifest.json').read_text());jobs=manifest['jobs']
                active=[x for x in children.values() if x['process'].poll() is None]
                for job in jobs:
                    key=job['weight_set']+'/'+job['model']
                    if key in children:continue
                    occupancy={g:sum(x['gpu']==g for x in active) for g in GPUS}
                    gpu=min(GPUS,key=lambda g:occupancy[g])
                    if occupancy[gpu]>=2:break
                    folder=Path(job['folder'])
                    with (folder/'worker.log').open('ab',buffering=0) as log:
                        p=subprocess.Popen([PYTHON,'-u',str(folder/'run.py'),'--reference','hardware'],
                            cwd=folder,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdin=subprocess.DEVNULL,
                            stdout=log,stderr=subprocess.STDOUT,close_fds=True)
                    record=dict(job,process=p,gpu=gpu,pid=p.pid);children[key]=record;active.append(record)
                rows=[];completed=0
                for key,x in children.items():
                    folder=Path(x['folder'])/'hardware';p=x['process']
                    records=[json.loads(path.read_text()) for path in folder.glob('episode_*.json')]
                    completed+=sum(e.get('status')=='complete' for e in records)
                    entry={k:v for k,v in x.items() if k!='process'};entry['returncode']=p.poll()
                    progress=folder/'status.json'
                    if progress.exists():entry['progress']=json.loads(progress.read_text())
                    rows.append(entry)
                state.update(status='running',workers=rows,completed_episodes=completed,
                             total_new_episodes=len(jobs)*12,updated_at_utc=now())
                try:
                    gpu=metrics();errors=0;state['gpu_metrics']=gpu
                    logmetrics.write(json.dumps(dict(at_utc=now(),gpu=gpu))+'\n')
                    unsafe=unsafe+1 if any(g['headroom_mib']<3072 or g['temperature_c']>=83 for g in gpu) else 0
                    if unsafe>=2:raise RuntimeError('GPU safety guard: headroom/temperature')
                except (subprocess.SubprocessError,OSError,ValueError) as exc:
                    errors+=1
                    if errors>=3:raise RuntimeError(f'GPU telemetry failed: {exc}')
                save(state)
                if len(children)==len(jobs) and all(x['process'].poll() is not None for x in children.values()):
                    state['status']='complete' if completed==len(jobs)*12 and all(x['process'].returncode==0 for x in children.values()) else 'completed_with_errors'
                    break
                time.sleep(5)
        if stopping:state['status']='stopped'
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc();stop(signal.SIGINT,None)
    finally:
        for x in children.values():
            p=x['process']
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:p.wait(timeout=180)
                except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30)
        if mps:
            with (ROOT/'mps_shutdown.log').open('a') as log:
                subprocess.run([str(MPS/'nvidia-cuda-mps-control')],input='quit\n',text=True,env=server_env,
                    stdout=log,stderr=subprocess.STDOUT,timeout=30,check=True)
        state['finished_at_utc']=now();save(state)


if __name__=='__main__':main()
