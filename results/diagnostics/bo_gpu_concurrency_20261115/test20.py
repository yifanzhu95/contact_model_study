"""Isolated MPS20 capacity and full-length checks; never add observations to BO."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import benchmark as b


def now():
    return datetime.now(timezone.utc).isoformat()


def worker(opts):
    import numpy as np
    import run_bayes_opt as bo
    from ContactModelStudy.Drivers import EpisodePool as ep
    row={key:str(value) for key,value in b.MANIFEST['searches'][0].items()}
    spec=bo.parseCell(row,Path(b.MANIFEST['csv']).parent,opts.output/opts.phase)
    args=spec.args
    args.steps=opts.steps
    args.save_steps=False
    ep._plannerMain=b.warm_planner_main
    pool=ep.EpisodePool(args,48,[b.GPU],planners_per_gpu=20,rollout_models=spec.models)
    started=time.monotonic()
    pool.start()
    results=[]
    try:
        planners=set();workers=set();deadline=time.monotonic()+300
        while len(planners)<20 or len(workers)<48:
            if time.monotonic()>deadline:raise TimeoutError('Warmup readiness timeout')
            try:msg=pool._events.get(timeout=.5)
            except queue.Empty:
                pool._checkAlive()
                if pool.fatal:raise RuntimeError(pool.problems)
                continue
            if msg[0]=='planner_ready':planners.add(msg[1])
            if msg[0]=='worker_ready':workers.add(msg[1])
            pool._handle(msg)
            if pool.fatal:raise RuntimeError(pool.problems)
        warmup=time.monotonic()-started
        b.save(opts.output/'phase.json',dict(phase=opts.phase,state='measuring',started_at=now(),
              planners_ready=len(planners),workers_ready=len(workers),warmup_s=warmup))
        measured=time.monotonic()
        candidates=([10,10,10] if opts.phase=='prefix' else [0,10])
        for group,index in enumerate(candidates):
            path=b.CANDIDATE.parent.parent/f'trial_{index:04d}/trial.json'
            weights=json.loads(path.read_text())['cost_weights']
            for model in spec.models:
                for episode in range(4):
                    pool.submit(ep.EpisodeJob(episode,rollout_model=model,cost_weights=weights,
                                key=dict(group=group,candidate=index)))
        deadline=time.monotonic()+1500
        while pool.pending:
            results.extend(pool.poll(.2))
            if pool.fatal:raise RuntimeError(pool.problems)
            if time.monotonic()>deadline:raise TimeoutError('Measured phase timeout')
        wall=time.monotonic()-measured
    except BaseException:
        pool.interrupt('safety_test_failed')
        raise
    finally:
        pool.close()
    episodes=[]
    for r in results:
        s=r.summary
        if r.error:raise RuntimeError(r.error)
        if s['finish_reason'] in ['error','interrupted']:raise RuntimeError(s['finish_reason'])
        for key in ['q_end','q_dot_end']:
            if not np.all(np.isfinite(np.asarray(s[key],dtype=float))):
                raise RuntimeError('Non-finite '+key)
        episodes.append(dict(group=r.job.key['group'],candidate=r.job.key['candidate'],
            model=r.job.rollout_model,episode=r.job.episode,summary=s,
            goal_seed=r.episode.extra['goal_seed']))
    expected=len(candidates)*16
    assert len(episodes)==expected,(len(episodes),expected)
    plans=sum(e['summary']['n_steps'] for e in episodes)
    result=dict(phase=opts.phase,planners=20,workers=48,models=spec.models,candidates=candidates,
        steps_limit=opts.steps,n_episodes=len(episodes),total_plans=plans,warmup_s=warmup,
        measured_wall_s=wall,plans_per_s=plans/wall,errors=0,nonfinite_results=0,
        episodes=episodes,args=vars(args),bo_observation=False,hardware_used=False)
    # Path-valued CLI parameters are provenance only.
    result=json.loads(json.dumps(result,default=str))
    b.save(opts.output/(opts.phase+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','args']}),flush=True)


def main(opts):
    opts.output.mkdir(parents=True,exist_ok=False)
    before=b.gpu_info()
    if before['utilization']>10 or before['memory_used_mib']>1000:
        raise RuntimeError(f'GPU 1 not idle: {before}')
    private=b.OUT/'private_mps_tools/usr/bin'
    pipe=tempfile.mkdtemp(prefix='bo-mps-',dir='/tmp')
    logs=opts.output/'mps_logs';logs.mkdir()
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    for key in list(env):
        if key.startswith('CUDA_MPS_'):env.pop(key,None)
    env.update(PATH=str(private)+':'+env['PATH'],PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
        CUDA_VISIBLE_DEVICES='',CUDA_MPS_PIPE_DIRECTORY=pipe,CUDA_MPS_LOG_DIRECTORY=str(logs),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        MUJOCO_GL='egl',WARP_CACHE_PATH=str(b.EXPERIMENT/'warp_cache'),CUDA_CACHE_PATH=str(b.OUT/'cuda_cache'))
    server_env=dict(env,CUDA_VISIBLE_DEVICES=b.GPU)
    control=str(private/'nvidia-cuda-mps-control')
    status=dict(status='running',pid=os.getpid(),started_at=now(),gpu_index=1,gpu=b.GPU,
        private_mps_pipe=pipe,settings_unchanged=True,completed_phases=[])
    b.save(opts.output/'status.json',status)
    running=False;process=None;all_samples=[];results=[]
    try:
        r=subprocess.run([control,'-d'],env=server_env,text=True,capture_output=True,timeout=20)
        (opts.output/'mps_startup.log').write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('MPS daemon startup failed')
        running=True
        for phase,steps in [('prefix',64),('full',1000)]:
            status.update(phase=phase,updated_at=now());b.save(opts.output/'status.json',status)
            print('START',phase,flush=True)
            command=[b.MANIFEST['python'],'-u',str(Path(__file__).resolve()),'--worker',
                     '--phase',phase,'--steps',str(steps),'--output',str(opts.output)]
            samples=[]
            with (opts.output/(phase+'.log')).open('w') as log:
                process=subprocess.Popen(command,env=env,cwd=b.CODE,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+1900
                while process.poll() is None:
                    sample=dict(b.gpu_info(),timestamp=now(),phase=phase)
                    samples.append(sample);all_samples.append(sample)
                    b.save(opts.output/'live_gpu.json',sample)
                    if time.monotonic()>deadline:raise TimeoutError('Phase exceeded 1900 seconds')
                    time.sleep(1)
            if process.returncode:raise RuntimeError(f'{phase} exited {process.returncode}; inspect log')
            process=None
            result=json.loads((opts.output/(phase+'.json')).read_text())
            result.update(gpu_samples=samples,peak_gpu_memory_mib=max(s['memory_used_mib'] for s in samples),
                minimum_gpu_headroom_mib=min(s['memory_total_mib']-s['memory_used_mib'] for s in samples))
            b.save(opts.output/(phase+'.json'),result);results.append(result)
            status['completed_phases'].append(phase);b.save(opts.output/'status.json',status)
            print('FINISH',phase,'plans/s',result['plans_per_s'],'peak MiB',result['peak_gpu_memory_mib'],flush=True)
        headroom=min(s['memory_total_mib']-s['memory_used_mib'] for s in all_samples)
        checks=dict(no_execution_errors=all(r['errors']==0 for r in results),
            all_results_finite=all(r['nonfinite_results']==0 for r in results),
            full_length_evaluations_completed=results[-1]['n_episodes']==32,
            minimum_3_gib_vram_headroom=headroom>=3072,
            peak_temperature_below_80c=max(s['temperature_c'] for s in all_samples)<80)
        report=dict(passed=all(checks.values()),checks=checks,
            minimum_gpu_headroom_mib=headroom,peak_gpu_memory_mib=max(s['memory_used_mib'] for s in all_samples),
            prefix_plans_per_s=results[0]['plans_per_s'],full_plans_per_s=results[1]['plans_per_s'],
            full_wall_s=results[1]['measured_wall_s'],full_plans=results[1]['total_plans'],
            no_hardware=True,bo_results_modified=False,checked_at=now())
        b.save(opts.output/'safety.json',report)
        status.update(status='passed' if report['passed'] else 'failed',safety=report)
    except BaseException as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=120)
    finally:
        if running:
            r=subprocess.run([control],input='quit\n',env=server_env,text=True,capture_output=True,timeout=30)
            (opts.output/'mps_shutdown.log').write_text(r.stdout+r.stderr)
            if r.returncode:status.update(status='failed',cleanup_error=r.stderr)
        status.update(finished_at=now());b.save(opts.output/'status.json',status)
        print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true')
    p.add_argument('--phase');p.add_argument('--steps',type=int)
    p.add_argument('--output',type=Path,required=True)
    opts=p.parse_args();opts.output=opts.output.resolve()
    if opts.worker:worker(opts)
    else:main(opts)
