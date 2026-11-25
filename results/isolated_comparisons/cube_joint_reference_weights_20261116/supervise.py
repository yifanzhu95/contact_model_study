"""Run the four simulation-only workers with private MPS on GPUs 2 and 3."""
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
GPUS=['GPU-9d410eae-46e7-c13e-9c03-a44cbda70781','GPU-3beb4f90-65ae-0ef8-e126-97a81e20ab17']
children=[];stopping=False


def now():return datetime.now(timezone.utc).isoformat()


def save(value):
    p=ROOT/'status.tmp';p.write_text(json.dumps(value,indent=2)+'\n');p.replace(ROOT/'status.json')


def stop(signum,frame):
    global stopping
    stopping=True
    for p,info in children:
        if p.poll() is None:p.send_signal(signal.SIGINT)


def metrics():
    out=subprocess.check_output(['nvidia-smi','-i',','.join(GPUS),
        '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=[]
    for line in out.strip().splitlines():
        i,util,used,total,temp=map(float,line.split(','))
        rows.append(dict(index=int(i),utilization=util,used_mib=used,headroom_mib=total-used,temperature_c=temp))
    return rows


def main():
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    state=dict(status='starting',pid=os.getpid(),started_at_utc=now(),total_episodes=48,
               completed_episodes=0,gpu_indices=[2,3],private_mps=True,planners_per_gpu=2,no_hardware=True)
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    for k in list(env):
        if k.startswith('CUDA_MPS_'):env.pop(k)
    pipe=tempfile.mkdtemp(prefix='joint-reference-mps-',dir='/tmp')
    (ROOT/'mps_logs').mkdir(exist_ok=True)
    env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',PATH=str(MPS)+':'+env['PATH'],
        CUDA_MPS_PIPE_DIRECTORY=pipe,CUDA_MPS_LOG_DIRECTORY=str(ROOT/'mps_logs'),
        WARP_CACHE_PATH=str(ROOT.parent/'cube_joint_reference_20261116/warp_cache'),
        CUDA_CACHE_PATH=str(ROOT/'cuda_cache'))
    server_env=dict(env,CUDA_VISIBLE_DEVICES=','.join(GPUS));started=False
    state['private_mps_pipe']=pipe;save(state)
    try:
        before=metrics()
        if any(g['used_mib']>4096 or g['utilization']>=30 for g in before):
            raise RuntimeError(f'Selected GPUs became busy before launch: {before}')
        with (ROOT/'mps_startup.log').open('a') as log:
            subprocess.run([str(MPS/'nvidia-cuda-mps-control'),'-d'],env=server_env,
                           stdout=log,stderr=subprocess.STDOUT,timeout=20,check=True)
        started=True
        for label in ['trial314','original']:
            for variant,gpu in zip(['optimization','hardware'],GPUS):
                folder=ROOT/label;log=(folder/f'{variant}.log').open('ab',buffering=0)
                p=subprocess.Popen([PYTHON,'-u',str(folder/'run.py'),'--reference',variant],cwd=folder,
                    env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,close_fds=True)
                log.close();children.append((p,dict(weight_set=label,reference=variant,pid=p.pid,gpu=gpu)))
        state.update(status='running',workers=[info for _,info in children]);save(state)
        unsafe=errors=0
        with (ROOT/'gpu_metrics.jsonl').open('a',buffering=1) as metriclog:
            while any(p.poll() is None for p,_ in children):
                rows=[];count=0
                for p,info in children:
                    folder=ROOT/info['weight_set']/info['reference'];s=folder/'status.json'
                    record=dict(info,returncode=p.poll())
                    if s.exists():record['progress']=json.loads(s.read_text())
                    count+=sum(json.loads(e.read_text()).get('status')=='complete' for e in folder.glob('episode_*.json'))
                    rows.append(record)
                    if p.poll() not in [None,0]:raise RuntimeError(f'Worker failed: {record}')
                try:
                    gpu=metrics();errors=0
                    metriclog.write(json.dumps(dict(at_utc=now(),gpu=gpu))+'\n')
                    unsafe=unsafe+1 if any(g['headroom_mib']<3072 or g['temperature_c']>=83 for g in gpu) else 0
                    if unsafe>=2:raise RuntimeError('GPU safety guard: low memory headroom or high temperature')
                    state['gpu_metrics']=gpu
                except (subprocess.SubprocessError,OSError,ValueError) as exc:
                    errors+=1
                    if errors>=3:raise RuntimeError(f'GPU telemetry failed: {exc}')
                state.update(workers=rows,completed_episodes=count,updated_at_utc=now());save(state)
                if stopping:break
                time.sleep(5)
        for p,info in children:
            rc=p.wait(timeout=180)
            if rc!=0 and not stopping:raise RuntimeError(f'Worker exited {rc}: {info}')
        state.update(status='stopped' if stopping else 'complete',completed_episodes=sum(
            json.loads(e.read_text()).get('status')=='complete' for label in ['trial314','original']
            for variant in ['optimization','hardware'] for e in (ROOT/label/variant).glob('episode_*.json')))
        if not stopping:assert state['completed_episodes']==48
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc()
        stop(signal.SIGINT,None)
    finally:
        for p,info in children:
            if p.poll() is None:
                try:p.wait(timeout=180)
                except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30)
        if started:
            with (ROOT/'mps_shutdown.log').open('a') as log:
                subprocess.run([str(MPS/'nvidia-cuda-mps-control')],input='quit\n',text=True,
                    env=server_env,stdout=log,stderr=subprocess.STDOUT,timeout=30,check=True)
        state['finished_at_utc']=now();save(state)


if __name__=='__main__':main()
