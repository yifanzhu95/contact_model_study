"""Bounded, isolated concurrency benchmark on GPU 1; leave the BO queue alone."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
import traceback
import tempfile

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
EXPERIMENT=ROOT/'results/isolated_bo/cube_friction_20261115'
MANIFEST=json.loads((EXPERIMENT/'manifest.json').read_text())
CODE=Path(MANIFEST['code'])
sys.path.insert(0,str(CODE))
sys.path.insert(0,str(CODE/'experiments'))
GPU='GPU-6155fe7d-ffa5-baa3-50cc-c93841698927'
CANDIDATE=Path(MANIFEST['results'])/'cell_0000_cube_mass070g_T0.1/trial_0010/trial.json'
WEIGHTS=json.loads(CANDIDATE.read_text())['cost_weights']
STEPS=64


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def warm_planner_main(args,gpu,planner_id,models,requests,replies,events,abort):
    """Warm each model before the existing pool emits planner_ready."""
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    from ContactModelStudy.Drivers import EpisodePool as ep
    original=ep._PlannerUnit.__init__
    def warmed(self,args,model):
        original(self,args,model)
        self.apply({},WEIGHTS)
        q,v,u=self.task.getInitialState()
        for _ in range(2):
            self.planner.Reset()
            self.planner.Plan(q,v,u=u)
    ep._PlannerUnit.__init__=warmed
    ep._plannerMain(args,gpu,planner_id,models,requests,replies,events,abort)


def run_case(opts):
    import numpy as np
    import run_bayes_opt as bo
    from ContactModelStudy.Drivers import EpisodePool as ep
    row={key:str(value) for key,value in MANIFEST['searches'][0].items()}
    spec=bo.parseCell(row,Path(MANIFEST['csv']).parent,OUT/opts.tag)
    args=spec.args
    args.steps=STEPS
    args.save_steps=False
    ep._plannerMain=warm_planner_main
    pool=ep.EpisodePool(args,16,[GPU],planners_per_gpu=opts.planners,rollout_models=spec.models)
    results=[]
    start=time.monotonic()
    pool.start()
    try:
        planners=set();workers=set()
        deadline=time.monotonic()+240
        while len(planners)<opts.planners or len(workers)<16:
            if time.monotonic()>deadline:raise TimeoutError('Warmup readiness timed out')
            try:msg=pool._events.get(timeout=.5)
            except queue.Empty:
                pool._checkAlive()
                if pool.fatal:raise RuntimeError(pool.problems)
                continue
            if msg[0]=='planner_ready':planners.add(msg[1])
            if msg[0]=='worker_ready':workers.add(msg[1])
            pool._handle(msg)
            if pool.fatal:raise RuntimeError(pool.problems)
        warmup=time.monotonic()-start
        measured=time.monotonic()
        for model in spec.models:
            for episode in range(4):
                pool.submit(ep.EpisodeJob(episode,rollout_model=model,cost_weights=WEIGHTS,key=opts.tag))
        deadline=time.monotonic()+360
        while pool.pending:
            results.extend(pool.poll(.1))
            if pool.fatal:raise RuntimeError(pool.problems)
            if time.monotonic()>deadline:raise TimeoutError('Measured benchmark timed out')
        wall=time.monotonic()-measured
    except BaseException:
        pool.interrupt('benchmark_failed')
        raise
    finally:
        pool.close()
    assert len(results)==16
    episodes=[]
    for r in sorted(results,key=lambda r:(r.job.rollout_model,r.job.episode)):
        s=r.summary
        if r.error:raise RuntimeError(r.error)
        episodes.append(dict(model=r.job.rollout_model,episode=r.job.episode,steps=s['n_steps'],
            outcome=s['finish_reason'],q_end=s['q_end'],q_dot_end=s['q_dot_end'],
            goal_values=s['goal_values'],noise_seed=s['noise_seed'],goal_seed=r.episode.extra['goal_seed'],
            mean_planning_s=s['plan_s_mean']))
    plans=sum(e['steps'] for e in episodes)
    result=dict(tag=opts.tag,mode=opts.mode,planners=opts.planners,workers=16,
        steps_per_episode=STEPS,episodes=episodes,total_plans=plans,
        warmup_s=warmup,measured_wall_s=wall,plans_per_s=plans/wall,
        errors=0,model_list=spec.models,gpu=GPU,physical_gpu_index=1,
        candidate=str(CANDIDATE),candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        metadata='Bounded performance benchmark only; episodes are not BO observations.')
    save(OUT/(opts.tag+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)


def gpu_info():
    text=subprocess.check_output(['nvidia-smi','-i',GPU,
        '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits'],text=True)
    values=[s.strip() for s in text.strip().split(',')]
    return dict(index=int(values[0]),utilization=float(values[1]),memory_used_mib=float(values[2]),
        memory_total_mib=float(values[3]),temperature_c=float(values[4]),power_w=float(values[5]))


def orchestrate(resume_mps=False, test_mps_planners=None):
    if test_mps_planners is not None:
        if not resume_mps or test_mps_planners<1:
            raise ValueError('--test-mps-planners requires --resume-mps and a positive count')
        assert not (OUT/f'mps_{test_mps_planners}.json').exists(), 'Preserve existing benchmark evidence'
    before=gpu_info()
    if before['utilization']>10 or before['memory_used_mib']>1000:
        raise RuntimeError(f'GPU 1 is not idle: {before}')
    if resume_mps:
        previous=json.loads((OUT/'summary.json').read_text())
        assert json.loads((OUT/'status.json').read_text())['status']=='complete'
        snapshot=OUT/('snapshot_'+datetime.now(timezone.utc).strftime('%H%M%S'))
        snapshot.mkdir()
        for name in ['summary.json','results.csv','status.json','mps_1.log','mps_startup.log','mps_shutdown.log']:
            if (OUT/name).exists(): (snapshot/name).write_bytes((OUT/name).read_bytes())
    else:
        assert not (OUT/'summary.json').exists(), 'Use a new output folder rather than overwrite benchmark evidence'
    # MPS uses Unix-domain sockets, which require a short path.
    own_pipe=Path(tempfile.mkdtemp(prefix='bo-mps-',dir='/tmp'));own_logs=OUT/'mps_logs'
    own_pipe.mkdir(exist_ok=True);own_logs.mkdir(exist_ok=True)
    os.chmod(own_pipe,0o700)
    base_env=os.environ.copy();base_env.pop('PYTHONPATH',None)
    private_bin=OUT/'private_mps_tools/usr/bin'
    if private_bin.exists():
        base_env['PATH']=str(private_bin)+':'+base_env['PATH']
    for key in list(base_env):
        if key.startswith('CUDA_MPS_'):base_env.pop(key,None)
    base_env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',
        WARP_CACHE_PATH=str(EXPERIMENT/'warp_cache'),CUDA_CACHE_PATH=str(OUT/'cuda_cache'))
    mps_env=dict(base_env,CUDA_VISIBLE_DEVICES=GPU,CUDA_MPS_PIPE_DIRECTORY=str(own_pipe),
        CUDA_MPS_LOG_DIRECTORY=str(own_logs))
    running_mps=False;case_process=None
    results=([json.loads((OUT/(r['tag']+'.json')).read_text()) for r in previous['results']]
             if resume_mps else [])
    skipped=[]
    status=dict(status='running',pid=os.getpid(),gpu=GPU,physical_gpu_index=1,
        started_at_utc=datetime.now(timezone.utc).isoformat(),completed_cases=[],before_gpu=before,
        main_optimization_unchanged=True,private_mps_pipe=str(own_pipe),resume_mps=resume_mps)
    def stop_mps():
        nonlocal running_mps
        if running_mps:
            r=subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,
                env=mps_env,capture_output=True,timeout=30)
            (OUT/'mps_shutdown.log').write_text(r.stdout+r.stderr)
            if r.returncode:raise RuntimeError('Private MPS shutdown failed: '+r.stderr)
            running_mps=False
    def start_mps():
        nonlocal running_mps
        if running_mps:return
        r=subprocess.run(['nvidia-cuda-mps-control','-d'],env=mps_env,capture_output=True,text=True,timeout=30)
        (OUT/'mps_startup.log').write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('Private MPS startup failed: '+r.stderr)
        running_mps=True
        time.sleep(1)
    def execute(mode,planners,tag):
        nonlocal case_process
        if mode=='mps':start_mps()
        else:stop_mps()
        env=dict(base_env)
        if mode=='mps':env.update(CUDA_MPS_PIPE_DIRECTORY=str(own_pipe),CUDA_MPS_LOG_DIRECTORY=str(own_logs))
        command=[MANIFEST['python'],'-u',str(Path(__file__).resolve()),'--case',
            '--mode',mode,'--planners',str(planners),'--tag',tag]
        status.update(current_case=tag,updated_at_utc=datetime.now(timezone.utc).isoformat())
        save(OUT/'status.json',status)
        print('START',tag,flush=True)
        samples=[]
        with (OUT/(tag+'.log')).open('w') as log:
            case_process=subprocess.Popen(command,env=env,cwd=CODE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            while case_process.poll() is None:
                samples.append(gpu_info())
                time.sleep(1)
            code=case_process.returncode
        case_process=None
        if code:raise RuntimeError(f'{tag} exited {code}; see {OUT/(tag+".log")}')
        r=json.loads((OUT/(tag+'.json')).read_text())
        r['gpu_samples']=samples
        r['peak_gpu_memory_mib']=max(s['memory_used_mib'] for s in samples)
        baseline=results[0] if results else r
        reference={(e['model'],e['episode']):e for e in baseline['episodes']}
        maximum=0.;outcomes_match=True;goals_match=True
        for e in r['episodes']:
            old=reference[e['model'],e['episode']]
            maximum=max(maximum,max(abs(x-y) for x,y in zip(e['q_end'],old['q_end'])))
            outcomes_match &= (e['outcome'],e['steps'])==(old['outcome'],old['steps'])
            goals_match &= (e['goal_values'],e['noise_seed'],e['goal_seed'])==(old['goal_values'],old['noise_seed'],old['goal_seed'])
        r.update(speedup_vs_first_baseline=r['plans_per_s']/baseline['plans_per_s'],
            max_abs_final_q_difference=maximum,outcomes_and_step_counts_match=outcomes_match,seeds_and_goals_match=goals_match)
        save(OUT/(tag+'.json'),r)
        results.append(r);status['completed_cases'].append(tag)
        save(OUT/'status.json',status)
        print('FINISH',tag,'wall',round(r['measured_wall_s'],2),'plans/s',round(r['plans_per_s'],2),
            'speedup',round(r['speedup_vs_first_baseline'],3),'q_difference',maximum,flush=True)
        return r
    try:
        if not resume_mps:
            for planners in [1,2,4]:execute('plain',planners,f'plain_{planners}')
        for planners in ([test_mps_planners] if test_mps_planners else [1,2,4]):
            try:execute('mps',planners,f'mps_{planners}')
            except Exception as exc:
                skipped.append(dict(mode='mps',planners=planners,error=str(exc)))
                print('MPS CASE FAILED',exc,flush=True)
                stop_mps()
                break
        eligible=[r for r in results if r['seeds_and_goals_match'] and r['outcomes_and_step_counts_match']]
        winner=max(eligible,key=lambda r:r['plans_per_s'])
        if test_mps_planners:
            if skipped:raise RuntimeError(skipped[-1]['error'])
            execute('mps',test_mps_planners,f'mps_{test_mps_planners}_repeat')
            execute('mps',4,f'mps_4_after_{test_mps_planners}')
        else:
            execute(winner['mode'],winner['planners'],'matched_repeat_best' if resume_mps else 'repeat_best')
            execute('plain',1,'matched_repeat_baseline' if resume_mps else 'repeat_baseline')
        summary=dict(results=[{k:v for k,v in r.items() if k not in ['episodes','gpu_samples']} for r in results],
            skipped=skipped,provisional_best=dict(mode=winner['mode'],planners=winner['planners']),
            physical_gpu_index=1,gpu=GPU,source_manifest=str(EXPERIMENT/'manifest.json'),
            main_optimization_settings_changed=False,benchmark_observations_added_to_bo=False,
            limitation='64-step prefixes of the same 16 episodes; initialization excluded; full-search scaling may differ.')
        save(OUT/'summary.json',summary)
        fields=list(summary['results'][0])
        with (OUT/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(summary['results'])
        status['status']='complete'
    except BaseException as exc:
        status.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        if case_process is not None and case_process.poll() is None:
            case_process.send_signal(__import__('signal').SIGINT)
            case_process.wait(timeout=90)
    finally:
        try:stop_mps()
        except Exception as exc:status['cleanup_error']=str(exc)
        status['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        save(OUT/'status.json',status)
        print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',action='store_true')
    parser.add_argument('--mode',choices=['plain','mps'],default='plain')
    parser.add_argument('--planners',type=int,default=1)
    parser.add_argument('--tag',default='test')
    parser.add_argument('--resume-mps',action='store_true')
    parser.add_argument('--test-mps-planners',type=int)
    args=parser.parse_args()
    if args.case:run_case(args)
    else:orchestrate(args.resume_mps,args.test_mps_planners)
