"""Freeze the user-designated M2 baseline and prepare BO; never run a trial."""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'run_leap_hardware.py').is_file())
OUT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def patch(path,old,new,count=1):
    text=path.read_text();actual=text.count(old)
    if actual!=count:raise RuntimeError(f'Unexpected frozen source at {path}: wanted {count} replacements, found {actual}')
    path.write_text(text.replace(old,new))


def main():
    if (OUT/'manifest.json').exists():raise RuntimeError('Preparation already exists; refusing to replace the frozen setup')
    baseline_path=ROOT/'results/bo_validation/M2_baseline_latest.json'
    baseline=json.loads(baseline_path.read_text());batch=ROOT/baseline['results_directory']
    controller=json.loads((batch/'controller.json').read_text())
    assert sha(batch/'controller.json')==baseline['controller_sha256']
    assert controller['rollout_model']=='M2' and len(controller['episodes'])==12
    shutil.copy2(baseline_path,OUT/'baseline.json')
    code=OUT/'code';code.mkdir()
    shutil.copytree(ROOT/'ContactModelStudy',code/'ContactModelStudy',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (code/'experiments').mkdir()
    for name in ['run_bayes_opt.py','run_episode_batches.py']:
        shutil.copy2(ROOT/'experiments'/name,code/'experiments'/name)
    shutil.copy2(OUT/'runtime.py',code/'runtime.py')
    (code/'run_search.py').write_text(
        'from pathlib import Path\nimport sys\n'
        'sys.path.insert(0,str(Path(__file__).resolve().parent/"experiments"))\n'
        'import run_bayes_opt as bo\nfrom runtime import install\ninstall(bo)\n'
        'if __name__=="__main__": raise SystemExit(bo.main())\n')
    # Native source files remain untouched; only this private snapshot changes.
    planner=code/'ContactModelStudy/SamplingBasedPlanners/SamplingBasedPlannerBase.py'
    patch(planner,'    action_rate_weight: float = 0.0',
        '    plan_blend_alpha: float = 0.75\n    action_rate_weight: float = 0.0')
    patch(planner,'        self._rate_previous_wp = (',
        '        self._runtime_penalties_wp = wp.array([10.,100.],dtype=wp.float32,device=dev)\n        self._rate_previous_wp = (')
    patch(planner,'int(pos_relative), self.config.action_rate_weight]',
        'int(pos_relative), self._runtime_penalties_wp]')
    patch(planner,'int(pos_relative), self.config.joint_limit_weight]',
        'int(pos_relative), self._runtime_penalties_wp]')
    text=planner.read_text();start=text.index('def _action_rate_cost_kernel(');end=text.index('def _accumulate_kernel(',start)
    kernels=text[start:end]
    assert kernels.count('weight: float,')==2
    kernels=kernels.replace('weight: float,','weight: wp.array(dtype=float),')
    assert kernels.count('weight * value')==1 and kernels.count('weight*value')==1
    kernels=kernels.replace('weight * value','weight[0] * value').replace('weight*value','weight[1]*value')
    planner.write_text(text[:start]+kernels+text[end:])
    pool=code/'ContactModelStudy/Drivers/EpisodePool.py'
    patch(pool,'"params": self._params, "weights": self._weights}',
        '"params": self._params, "weights": self._weights, "activation": self.request_activation}')
    patch(pool,'self.last_action_uncertainty = rep["sigma"]',
        'self.last_action_uncertainty = rep["sigma"]\n        self.last_mean = rep["last_mean"]')
    patch(pool,'out = planner.Plan(req["q"], req["q_dot"], u=req["u"])',
        'unit.sim.activation.assign(np.asarray(req["activation"],dtype=np.float32))\n                out = planner.Plan(req["q"], req["q_dot"], u=req["u"])')
    patch(pool,'"state": planner.SaveState()})',
        '"state": planner.SaveState(), "last_mean": planner.last_mean})')
    patch(pool,'            except Exception as e:\n                replies[req["worker"]].put',
        '            except Exception as e:\n'
        '                if any(s in str(e).lower() for s in ("out of memory", "illegal memory access", "cuda_error")):\n'
        '                    events.put(("planner_failed",planner_id,str(e)))\n'
        '                    abort.set()\n                    return\n'
        '                replies[req["worker"]].put')
    bo=code/'experiments/run_bayes_opt.py'
    patch(bo,'KNOBS = ("temperature", "noise_sigma")',
        'KNOBS = ("temperature", "noise_sigma", "action_rate_weight", "joint_limit_weight", "plan_blend_alpha")')
    patch(bo,'    "per_model_temperature", "per_model_noise_sigma",',
        '    "action_rate_weight_range", "joint_limit_weight_range", "plan_blend_alpha_range",\n'
        '    "per_model_temperature", "per_model_noise_sigma",')
    patch(bo,'    if not 0 < lo < hi:\n        raise ValueError(f"{column} needs 0 < lo < hi, got {raw!r}")',
        '    allow_zero = column in ("action_rate_weight_range", "joint_limit_weight_range", "plan_blend_alpha_range")\n'
        '    if not (0 <= lo < hi if allow_zero else 0 < lo < hi):\n'
        '        raise ValueError(f"Invalid range for {column}: {raw!r}")')
    # 3 trials x 12 episodes would otherwise clamp the requested worker pool to
    # 36. Retain 48 workers as requested; idle workers are cheap CPU processes.
    patch(bo,'    workers = max(1, min(workers, in_flight_max * spec.episodes_per_trial))',
        '    workers = max(1, workers)  # preserve requested 48-worker allocation')
    patch(bo,'print(f"    {d.name:<20} [{d.lo:g}, {d.hi:g}]  log-uniform")',
        'print(f"    {d.name:<20} [{d.lo:g}, {d.hi:g}]  {d.skopt().prior}")')

    assets=OUT/'assets';assets.mkdir();scene_records={}
    for role,scene in controller['scenes'].items():
        xml=ET.fromstring(scene['scene_xml']);compiler=xml.find('compiler')
        for tag,directory in [('mesh',compiler.get('meshdir')),('texture',compiler.get('texturedir'))]:
            for element in xml.findall(f'./asset/{tag}'):
                if not element.get('file'):continue
                source=(Path(directory)/element.get('file')).resolve();digest=sha(source)
                destination=assets/(digest[:16]+source.suffix)
                if not destination.exists():shutil.copy2(source,destination)
                element.set('file',destination.name)
        compiler.set('meshdir',str(assets));compiler.set('texturedir',str(assets))
        target=code/'scenes/leap'/('env_leap_rollout_cube_high_high.xml' if role=='rollout' else 'env_leap_eval_cube.xml')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(ET.tostring(xml,encoding='unicode'))
        scene_records[role]=dict(path=str(target),sha256=sha(target),source_scene=scene['source_scene'],
                                 transformations='asset paths only; frozen calibrated high/high XML from designated baseline')
    initials=[];metrics=json.loads((batch/'analysis/report.json').read_text())['episodes']
    for e,row in zip(controller['episodes'],metrics):
        first=e['steps'][0]
        item=dict(source_run=e['source_run'],source_episode=e['source_episode'],rotation=row['rotation'],
            q=first['q'],q_dot=first['q_dot'],goal=e['goal'],target_pos=e['target_pos'],
            initial_position_error_mm=row['initial_position_error_mm'])
        item['sha256']=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest();initials.append(item)
    save(OUT/'initial_states.json',initials)
    fixed_weights={k:baseline['weights'][k] for k in ['w_velo','w_joint_velo','w_fallen_term','w_contact']}
    assert all(fixed_weights[k]==0 for k in ['w_velo','w_joint_velo','w_fallen_term'])
    ranges={k:(.01,100.) for k in ['w_quat','w_pos_x','w_pos_y','w_pos_z','w_joint']}
    ranges.update(w_fallen=(10.,500.),w_quat_term=(10.,1000.),w_pos_term=(10.,1000.))
    row=dict(label='M2_baseline_70g_T01_N512_H4',task='cube_reorient',rollout_models='M2',model_agg='mean',
        hand_acc='high',obj_acc='high',eval_sim='mujoco',n_episodes=12,steps=1000,seed=0,goal_difficulty=4,
        timestep=.004,eval_steps_per_rollout_step=1,ctrl_time_step=.064,time_horizon=.256,n_samples=512,n_iterations=1,
        temperature=.1,noise_sigma=.05,control_mode='pos_relative',warm_start='shift',resample_noise='false',center_noise='true',
        adaptive_temp='false',graph='true',plan_on_eval='false',stop_on_success='true',settle=0.,nconmax=100,njmax=300,
        save_steps='false',uncertainty='false',opt_weights=' '.join(f'{k}:{lo}:{hi}' for k,(lo,hi) in ranges.items()),
        action_rate_weight_range='0 50',joint_limit_weight_range='0 1000',plan_blend_alpha_range='0 1',
        n_calls=500,n_initial_points=120,acq_func='gp_hedge',bo_seed=0,w_success=1.,w_cost=.1,err_clip=250.,
        seed_defaults='true',trials_in_flight=3,**fixed_weights)
    row['seed_points']=str(OUT/'seed_points.csv')
    with (OUT/'seed_points.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['action_rate_weight','joint_limit_weight','plan_blend_alpha']);writer.writeheader()
        writer.writerows([dict(action_rate_weight=0.,joint_limit_weight=100.,plan_blend_alpha=.75),
            dict(action_rate_weight=10.,joint_limit_weight=0.,plan_blend_alpha=.75),
            dict(action_rate_weight=10.,joint_limit_weight=100.,plan_blend_alpha=1.)])
    with (OUT/'search.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    old=json.loads((ROOT/'results/isolated_bo/cube_friction_20261115/execution_revisions/mps500_20261115/config.json').read_text())
    manifest=dict(status='prepared_not_started',created_at_utc=datetime.now(timezone.utc).isoformat(),
        root=str(ROOT),out=str(OUT),code=str(code),csv=str(OUT/'search.csv'),results=str(OUT/'search_results'),
        python='/home/chetan/miniconda3/envs/mj_env/bin/python',baseline_record=str(baseline_path),
        baseline_record_sha256=sha(baseline_path),models=['M2'],row=row,searches=[row],
        weight_ranges={k:list(v) for k,v in ranges.items()},
        extra_ranges=dict(action_rate_weight=[0.,50.],joint_limit_weight=[0.,1000.],plan_blend_alpha=[0.,1.]),
        fixed_weights=fixed_weights,objective='-success_rate + 0.1*mean(min(sum(final_error/threshold),250)/250)',
        n_calls=500,n_initial_points=120,physical_gpu_indices=old['gpu_indices'],gpus=old['gpus'],
        preferred_planners_per_gpu=20,fallback_planners_per_gpu=16,workers=48,trials_in_flight=3,mps_bin=old['mps_bin'],
        scenes=scene_records,source_snapshot_sha256={str(p.relative_to(code)):sha(p) for p in code.rglob('*') if p.is_file()},
        assets_sha256={str(p.relative_to(OUT)):sha(p) for p in assets.iterdir()},
        initial_states_sha256=sha(OUT/'initial_states.json'),baseline_snapshot_sha256=sha(OUT/'baseline.json'),
        search_csv_sha256=sha(OUT/'search.csv'),seed_points_sha256=sha(OUT/'seed_points.csv'),
        episodes_per_trial=12,total_episodes_at_budget=6000,max_episode_simulation_s=64.,
        timing='Virtual 64ms deadline: CPU plant advances under held command before delta application. No wall-clock sleeps or GPU-time-dependent physics.',
        sensor_latency='No camera/USB latency or tracking noise is injected; initial states come from already calibrated hardware records.',
        evaluation='CPU MuJoCo calibrated high/high plant, fixed recorded initial states; no pre-episode settling or hardware.',
        starting_states='Twelve recorded baseline states; six signed ~90deg rotations twice each, including the recorded off-center placement.',
        completion_action='Export best weights/parameters and commented text only; never modify active task weights or hardware configuration.',
        notes=['No prior BO observations reused: dynamics, reference, clipping, horizon and new dimensions differ.',
            'Historical 20-planner memory test is not proof that 512-sample calibrated planners fit; validate on start and fall back to 16 if needed.',
            'Synthetic simulation success is not a hardware success prediction; no additional initial-pose correction is applied.'])
    save(OUT/'manifest.json',manifest)
    save(OUT/'status.json',dict(status='prepared_not_started',trials_started=0,episodes_run=0))
    print(json.dumps(dict(manifest=str(OUT/'manifest.json'),status=manifest['status'],dimensions=len(ranges)+3,trials=500),indent=2))


if __name__=='__main__':main()
