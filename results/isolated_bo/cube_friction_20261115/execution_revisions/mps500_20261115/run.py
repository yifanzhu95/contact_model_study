"""Resume only the authorized 70g/T0.1 search, with private MPS and checkpoints."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback

REV=Path(__file__).resolve().parent
ROOT=REV.parents[1]
CFG=json.loads((REV/'config.json').read_text())
ORIGINAL=json.loads(Path(CFG['source_manifest']).read_text())
CHILD=None
STOP=False


def now():return datetime.now(timezone.utc).isoformat()


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def stop(signum,frame):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:CHILD.send_signal(signal.SIGINT)


def progress():
    path=Path(CFG['results'])/CFG['cell']/'bo_state.json'
    try:
        data=json.loads(path.read_text());values=data.get('func_vals',[])
        return dict(completed_trials=len(values),best_objective=min(values) if values else None)
    except (OSError,ValueError):return {}


def gpu_stats():
    out=subprocess.check_output(['nvidia-smi','-i',','.join(CFG['gpus']),
        '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    result=[]
    for line in out.strip().splitlines():
        i,uuid,util,used,total,temp,power=[v.strip() for v in line.split(',')]
        result.append(dict(index=int(i),uuid=uuid,utilization=float(util),memory_used_mib=float(used),
            memory_total_mib=float(total),headroom_mib=float(total)-float(used),
            temperature_c=float(temp),power_w=float(power)))
    assert {r['index'] for r in result}=={2,3}
    return result


def verify_preserved_trials():
    hashes=json.loads((REV/'preserved_trial_hashes.json').read_text())
    for relative,digest in hashes.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative


def publish_weights():
    summary=json.loads((Path(CFG['results'])/CFG['cell']/'bo_summary.json').read_text())
    assert summary['n_trials']==CFG['n_calls']
    spec=importlib.util.spec_from_file_location('original_queue',ROOT/'run_queue.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.publish_weights(summary)


def main():
    global CHILD
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    for relative,digest in ORIGINAL['source_snapshot_sha256'].items():
        assert hashlib.sha256((Path(CFG['code'])/relative).read_bytes()).hexdigest()==digest,relative
    assert not Path('/proc/85385').exists(),'Original supervisor must have exited before resuming'
    verify_preserved_trials()
    safety=json.loads(Path(CFG['safety_report']).read_text()) if Path(CFG['safety_report']).exists() else {}
    safety_status=json.loads(Path(CFG['safety_status']).read_text())
    assert safety_status['status'] in ['passed','failed'],'Safety test is not finished'
    planners=CFG['preferred_planners_per_gpu'] if safety.get('passed') and safety_status['status']=='passed' else CFG['fallback_planners_per_gpu']
    for g in gpu_stats():assert g['memory_used_mib']<1000 and g['utilization']<15,g
    pipe=tempfile.mkdtemp(prefix='bo-mps-',dir='/tmp');logs=REV/'mps_logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    for key in list(env):
        if key.startswith('CUDA_MPS_'):env.pop(key,None)
    env.update(PATH=CFG['mps_bin']+':'+env['PATH'],PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
        CUDA_VISIBLE_DEVICES='',CUDA_MPS_PIPE_DIRECTORY=pipe,CUDA_MPS_LOG_DIRECTORY=str(logs),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        WARP_CACHE_PATH=str(ROOT/'warp_cache'),CUDA_CACHE_PATH=str(ROOT/'cuda_cache'),MUJOCO_GL='egl')
    server_env=dict(env,CUDA_VISIBLE_DEVICES=','.join(CFG['gpus']))
    control=str(Path(CFG['mps_bin'])/'nvidia-cuda-mps-control')
    state=dict(status='starting',pid=os.getpid(),started_at_utc=now(),current_cell=0,
        current_label='cube_mass070g_T0.1',total_cells=1,total_trials=500,
        config=str(REV/'config.json'),execution_revision=str(REV),gpus=CFG['gpus'],gpu_indices=[2,3],
        planners_per_gpu=planners,workers=48,trials_in_flight=3,initial_points=120,
        private_mps_pipe=pipe,safety_report=CFG['safety_report'],safety_test_passed=bool(safety.get('passed')) and safety_status['status']=='passed',
        original_queue_stopped=True,other_searches_preserved=True,completed_trials_at_resume=progress().get('completed_trials'),
        completed_cells=[],**progress())
    def checkpoint():
        state.update(updated_at_utc=now(),**progress())
        save(REV/'status.json',state);save(ROOT/'status.json',state)
    checkpoint()
    proc=Path('/proc')/str(os.getpid())
    identity=dict(pid=os.getpid(),start_ticks=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),command=[CFG['python'],'-u',str(Path(__file__).resolve())],
        execution_revision=str(REV),selected_gpu_indices=[2,3])
    save(REV/'pid.json',identity);save(ROOT/'pid.json',identity)
    running=False
    try:
        r=subprocess.run([control,'-d'],env=server_env,text=True,capture_output=True,timeout=20)
        (REV/'mps_startup.log').write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('Private MPS startup failed')
        running=True
        with (REV/'gpu_metrics.jsonl').open('a',buffering=1) as metrics:
            for attempt in range(2):
                if STOP:break
                command=[CFG['python'],'-u',str(Path(CFG['code'])/'experiments/run_bayes_opt.py'),
                    CFG['csv'],'--cell','0','--outdir',CFG['results'],'--gpus',','.join(CFG['gpus']),
                    '--planners-per-gpu',str(planners),'--workers',str(CFG['workers'])]
                logpath=REV/f'bo_planners{planners}_attempt{attempt}.log'
                state.update(status='running',command=command,current_log=str(logpath),
                    planners_per_gpu=planners,attempt=attempt)
                print(now(),'RESUME',planners,'planners per GPU',flush=True)
                guard=None;consecutive_unsafe=0;telemetry_errors=0
                with logpath.open('a') as log:
                    CHILD=subprocess.Popen(command,cwd=CFG['code'],env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT)
                    state['child_pid']=CHILD.pid;checkpoint()
                    while CHILD.poll() is None:
                        try:
                            gpu=gpu_stats();telemetry_errors=0
                            metrics.write(json.dumps(dict(time=now(),gpu=gpu,planners_per_gpu=planners))+'\n')
                            state['gpu_metrics']=gpu
                            unsafe=any(g['headroom_mib']<3072 or g['temperature_c']>=83 for g in gpu)
                            consecutive_unsafe=consecutive_unsafe+1 if unsafe else 0
                            if consecutive_unsafe>=2:
                                guard='Two consecutive samples below 3 GiB VRAM headroom or at/above 83 C'
                                CHILD.send_signal(signal.SIGINT)
                        except (subprocess.SubprocessError,ValueError,OSError) as exc:
                            telemetry_errors+=1;state['telemetry_error']=str(exc)
                            if telemetry_errors>=3:
                                guard='GPU telemetry failed three consecutive times';CHILD.send_signal(signal.SIGINT)
                        checkpoint()
                        if guard:break
                        time.sleep(5)
                    if CHILD.poll() is None:CHILD.wait(timeout=180)
                    rc=CHILD.returncode
                CHILD=None
                state.update(last_returncode=rc,guard_reason=guard);checkpoint()
                verify_preserved_trials()
                if STOP:state['status']='stopped';break
                done=progress().get('completed_trials',0)
                if rc==0 and done==500:
                    state.update(status='complete',completed_cells=[CFG['cell']])
                    try:publish_weights();state['comment_weights_published']=True
                    except Exception as exc:state['comment_update_error']=str(exc)
                    break
                if planners==20 and attempt==0:
                    state['fallback_reason']=guard or f'20-planner BO exited {rc} after {done} trials'
                    print(now(),'FALLBACK TO 16:',state['fallback_reason'],flush=True)
                    planners=16
                    time.sleep(3)
                    continue
                raise RuntimeError(f'BO stopped at {done}/500, exit {rc}, guard {guard}')
            if STOP:state['status']='stopped'
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}');traceback.print_exc()
        if CHILD is not None and CHILD.poll() is None:
            CHILD.send_signal(signal.SIGINT);CHILD.wait(timeout=180)
    finally:
        if running:
            r=subprocess.run([control],input='quit\n',env=server_env,text=True,capture_output=True,timeout=30)
            (REV/'mps_shutdown.log').write_text(r.stdout+r.stderr)
            if r.returncode:state['mps_cleanup_error']=r.stderr
        state['finished_at_utc']=now();checkpoint()
        print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
