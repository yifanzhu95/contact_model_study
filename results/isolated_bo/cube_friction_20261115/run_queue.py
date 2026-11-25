"""Run the isolated ten-cell BO queue, checkpoint status, and update comments only."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback

OUT = Path(__file__).resolve().parent
MANIFEST = json.loads((OUT/'manifest.json').read_text())
child = None
stopping = False


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def cell_name(index, row):
    return f'cell_{index:04d}_'+row['label']


def commented_region(text):
    matches=list(re.finditer(r'(?m)^(\s*)#\s*"cost_weights"\s*:\s*\{[^\n]*\n',text))
    if not matches:raise RuntimeError('No commented cost_weights block found')
    start=matches[-1]
    end=re.search(r'(?m)^\s*#\s*},[^\n]*(?:\n|$)',text[start.end():])
    if not end:raise RuntimeError('Commented cost_weights block has no closing line')
    # Restrict indentation to the header line; \s can otherwise include a preceding newline.
    indent=re.match(r'[ \t]*',text[start.start():].lstrip('\n')).group()
    return start.start(),start.end()+end.end(),indent


def render_update(text, weights, trial):
    begin,end,indent=commented_region(text)
    keys=['w_quat','w_pos_x','w_pos_y','w_pos_z','w_velo','w_contact','w_joint',
          'w_joint_velo','w_fallen','w_quat_term','w_pos_term','w_fallen_term']
    assert set(keys)==set(weights)
    block=indent+'# "cost_weights": { # BO: mass=0.07 kg, T=0.1, noise=0.05; trial '+str(trial)+'\n'
    for key in keys:block+=indent+'#     '+f'"{key}":'.ljust(19)+f'{weights[key]:.10g},\n'
    block+=indent+'# },\n'
    result=text[:begin]+block+text[end:]
    assert ast.dump(ast.parse(result))==ast.dump(ast.parse(text)), 'Executable code would change'
    assert result[:begin]==text[:begin] and result.endswith(text[end:])
    return result


def publish_weights(summary):
    path=Path(MANIFEST['target_file'])
    original=(OUT/'CubeReorient.before.py').read_text()
    current=path.read_text()
    a,b,_=commented_region(original);c,e,_=commented_region(current)
    if current[c:e]!=original[a:b]:
        # Idempotent queue resumption after a successful prior application.
        if (OUT/'applied_comment_weights.json').exists():return
        raise RuntimeError('The final commented block was edited during BO; results retained, update refused')
    frozen=ast.parse((Path(MANIFEST['code'])/'ContactModelStudy/Tasks/CubeReorient.py').read_text())
    candidates=[n for n in ast.walk(frozen) if isinstance(n,ast.Dict) and len(n.keys)==12
                and all(isinstance(k,ast.Constant) and str(k.value).startswith('w_') for k in n.keys)]
    assert len(candidates)==1
    weights=ast.literal_eval(candidates[0]);weights.update(summary['best']['cost_weights'])
    text=render_update(current,weights,summary['best']['trial'])
    before=hashlib.sha256(current.encode()).hexdigest()
    path.write_text(text)
    save(OUT/'applied_comment_weights.json',dict(applied_at_utc=utc(),target=str(path),
        before_sha256=before,after_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        active_code_ast_unchanged=True,trial=summary['best']['trial'],
        objective=summary['best']['objective'],weights=weights))


def stop(signum, frame):
    global stopping
    stopping=True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def progress(index,row):
    path=Path(MANIFEST['results'])/cell_name(index,row)/'bo_state.json'
    try:
        d=json.loads(path.read_text())
        vals=d.get('func_vals',[])
        return dict(completed_trials=len(vals),best_objective=min(vals) if vals else None)
    except (OSError,ValueError):return dict(completed_trials=0,best_objective=None)


def main():
    global child
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    for rel,digest in MANIFEST['source_snapshot_sha256'].items():
        assert hashlib.sha256((Path(MANIFEST['code'])/rel).read_bytes()).hexdigest()==digest
    env=os.environ.copy()
    env.pop('PYTHONPATH',None)
    env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        WARP_CACHE_PATH=str(OUT/'warp_cache'),CUDA_CACHE_PATH=str(OUT/'cuda_cache'),MUJOCO_GL='egl')
    state=dict(status='running',pid=os.getpid(),started_at_utc=utc(),completed_cells=[],
        total_cells=10,total_trials=7500,manifest=str(OUT/'manifest.json'))
    save(OUT/'status.json',state)
    try:
        for index,row in enumerate(MANIFEST['searches']):
            if stopping:break
            name=cell_name(index,row)
            log_path=OUT/(name+'.log')
            command=[MANIFEST['python'],str(Path(MANIFEST['code'])/'experiments/run_bayes_opt.py'),
                MANIFEST['csv'],'--cell',str(index),'--outdir',MANIFEST['results'],
                '--gpus',','.join(MANIFEST['gpus']),'--planners-per-gpu',str(MANIFEST['planners_per_gpu']),
                '--workers',str(MANIFEST['workers'])]
            state.update(current_cell=index,current_label=row['label'],current_log=str(log_path),
                command=command,updated_at_utc=utc(),**progress(index,row))
            save(OUT/'status.json',state)
            print(f'{utc()} START {name}',flush=True)
            with log_path.open('a') as log:
                child=subprocess.Popen(command,cwd=MANIFEST['code'],env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid
                while child.poll() is None:
                    state.update(updated_at_utc=utc(),**progress(index,row))
                    save(OUT/'status.json',state)
                    time.sleep(5)
                rc=child.returncode
            child=None
            state.update(last_returncode=rc,updated_at_utc=utc(),**progress(index,row))
            if stopping:
                state['status']='stopped';break
            if rc!=0:
                raise RuntimeError(f'{name} exited {rc}; inspect {log_path}. Resume preserves completed trials.')
            summary_path=Path(MANIFEST['results'])/name/'bo_summary.json'
            summary=json.loads(summary_path.read_text())
            if summary['n_trials']!=row['n_calls']:
                raise RuntimeError(f'{name}: only {summary["n_trials"]} trials, expected {row["n_calls"]}')
            state['completed_cells'].append(name)
            if index==0:
                try:publish_weights(summary)
                except Exception as exc:
                    state['comment_update_error']=str(exc)
                    print('COMMENT UPDATE FAILED:',exc,flush=True)
            save(OUT/'status.json',state)
            print(f'{utc()} FINISHED {name}; best objective={summary["best"]["objective"]}',flush=True)
        else:state['status']='complete'
        if stopping:state['status']='stopped'
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            child.wait()
    finally:
        state.update(updated_at_utc=utc(),finished_at_utc=utc())
        save(OUT/'status.json',state)
        print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
