"""Paired physical-mismatch evaluation. Simulation only; no hardware/LCM."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parent
CONFIG=json.loads((OUT/'config.json').read_text())
sys.path.insert(0,str(OUT/'code'))
sys.path.insert(0,str(OUT/'code/experiments'))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('WARP_CACHE_PATH',str(OUT/'warp_cache'))
os.environ.setdefault('CUDA_CACHE_PATH',str(OUT/'cuda_cache'))
import numpy as np
import mujoco
import run_bayes_opt as bo
import runtime
runtime.install(bo)
from ContactModelStudy.Drivers import run_episodes as drv
from ContactModelStudy.Drivers.EpisodePool import _PlannerUnit,episodeSeeds
from ContactModelStudy.Simulators.Mujoco import Mujoco,MujocoConfig
from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder,StepSaving


def now():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenarios():
    """Seeded stratification covers each signed rotation once before repeats."""
    rng=np.random.default_rng(CONFIG['seed'])
    source=runtime.INITIAL
    rotations=list(dict.fromkeys(s['rotation'] for s in source))
    assert len(rotations)==6
    order=list(rng.permutation(rotations))+list(rng.choice(rotations,size=4,replace=False))
    used=set();cases=[]
    for i,rotation in enumerate(order):
        candidates=[j for j,s in enumerate(source) if s['rotation']==rotation and j not in used]
        index=int(rng.choice(candidates));used.add(index)
        physics=dict(mass_kg=float(rng.uniform(*CONFIG['mass_kg_range'])),
            sliding={k:float(rng.uniform(*bounds)) for k,bounds in CONFIG['sliding_friction_ranges'].items()})
        physics['effective_cube_contacts']={k:max(physics['sliding']['cube'],physics['sliding'][k])
            for k in ('fingertip','palm','finger')}
        _,seed=episodeSeeds(CONFIG['seed'],i)
        cases.append(dict(pair=i+1,source_index=index,rotation=rotation,noise_seed=seed,
            initial_state_sha256=source[index]['sha256'],physics=physics,
            execution_order=['baseline','trial358'] if i%2==0 else ['trial358','baseline']))
    return cases


def geom_group(name,body):
    if body=='obj':return 'cube'
    if body=='palm':return 'palm'
    if body.startswith(('if_','mf_','rf_','th_')):
        return 'fingertip' if name in ('if_tip','mf_tip','rf_tip','th_tip') else 'finger'
    return None


def write_scene(case):
    tree=ET.parse(OUT/'code/scenes/leap/env_leap_eval_cube.xml')
    counts=dict.fromkeys(CONFIG['sliding_friction_ranges'],0)
    obj=tree.find('.//body[@name="obj"]/geom[@name="obj"]')
    assert obj is not None
    obj.set('mass',format(case['physics']['mass_kg'],'.17g'))
    # XML recompilation recomputes the cube inertia consistently with mass.
    for body in tree.findall('.//worldbody//body'):
        for geom in body.findall('geom'):
            name=geom.get('name','')
            if 'visual' in name or name.endswith('_vis') or geom.get('class') in ('visual','tip_vis','thumb_tip_vis'):
                continue
            group=geom_group(name,body.get('name',''))
            if group is None:continue
            value=case['physics']['sliding'][group]
            geom.set('friction',f'{value:.17g} {CONFIG["torsional_friction"]:.17g} {CONFIG["rolling_friction"]:.17g}')
            counts[group]+=1
    assert all(n>0 for n in counts.values()),counts
    path=OUT/'scenes'/f'eval_pair_{case["pair"]:02d}.xml'
    path.parent.mkdir(exist_ok=True);tree.write(path,encoding='unicode')
    return path,counts


def prepare():
    for relative,digest in {**CONFIG['code_sha256'],**CONFIG['assets_sha256']}.items():
        assert sha(OUT/relative)==digest,relative
    # The original BO parser consumes CSV strings, not JSON numeric values.
    spec=bo.parseCell({k:str(v) for k,v in CONFIG['row'].items()},OUT,OUT/'unused_bo_cell')
    assert spec.args.n_samples==512 and spec.args.time_horizon==.256
    cases=scenarios()
    if (OUT/'scenarios.json').exists():
        previous=json.loads((OUT/'scenarios.json').read_text())
        assert [{k:v for k,v in c.items() if k not in ('eval_scene','eval_scene_sha256','geom_counts')} for c in previous]==cases
    for case in cases:
        path,counts=write_scene(case)
        case.update(eval_scene=str(path),eval_scene_sha256=sha(path),geom_counts=counts)
        model=mujoco.MjModel.from_xml_path(str(path))
        assert np.isclose(model.body('obj').mass[0],case['physics']['mass_kg'])
        np.testing.assert_allclose(model.body('obj').inertia,
            np.repeat(case['physics']['mass_kg']*.07**2/6,3),rtol=1e-12)
        assert model.na==16
        for gid in range(model.ngeom):
            if not(model.geom_contype[gid] or model.geom_conaffinity[gid]):continue
            group=geom_group(model.geom(gid).name,model.body(model.geom_bodyid[gid]).name)
            if group is not None:
                np.testing.assert_allclose(model.geom_friction[gid],
                    [case['physics']['sliding'][group],CONFIG['torsional_friction'],CONFIG['rolling_friction']])
    nominal=mujoco.MjModel.from_xml_path(str(OUT/'code/scenes/leap/env_leap_rollout_cube_high_high.xml'))
    assert np.isclose(nominal.body('obj').mass[0],.07)
    for group,geom in [('cube','obj'),('fingertip','mf_tip')]:
        assert np.isclose(nominal.geom(geom).friction[0],CONFIG['nominal_friction'][group])
    save(OUT/'scenarios.json',cases)
    return cases


class LocalPlanner:
    """Same calibrated state/activation and runtime weights as the BO pool."""
    def __init__(self,unit,params,seed):
        self.real=unit.planner;self.sim=unit.sim;self.task=unit.task
        self.config=replace(unit.base_config,**params)
        self.seed=seed;self.last_plan_seconds=None;self.durations=[]
    def __getattr__(self,name):return getattr(self.real,name)
    def Reset(self,mean=None):
        self.real.Reset(mean)
        self.real.LoadState({**self.real.SaveState(),'u_prev':None,'last_action_seq':None,
                            'noise_seed':self.seed,'resample_count':0})
    def Plan(self,q,v,u=None):
        import warp as wp
        self.sim.activation.assign(np.asarray(self.request_activation,dtype=np.float32))
        started=time.perf_counter();result=self.real.Plan(q,v,u=u);wp.synchronize()
        self.last_plan_seconds=time.perf_counter()-started
        self.durations.append(self.last_plan_seconds)
        return result


def episode(case,variant,args,unit):
    settings=CONFIG['variants'][variant]
    unit.apply(settings['params'],settings['weights'])
    planner=LocalPlanner(unit,settings['params'],case['noise_seed'])
    _,task=drv.buildTasks(args);task.setCostWeights(settings['weights'])
    task.getModelPath=lambda:case['eval_scene']
    sim=Mujoco(case['eval_scene'],MujocoConfig(timestep=.004,cone='pyramidal',solver='Newton',iterations=25,tolerance=1e-6))
    recorder=EpisodeRecorder(task,sim,planner,steps=StepSaving(enabled=True,stride=1,plans=True),
        scenario=case,variant=variant,nominal_planner_physics=dict(mass_kg=.07,sliding=CONFIG['nominal_friction']))
    started=time.perf_counter()
    try:
        summary=runtime.run_episode(case['source_index'],args,unit.task,task,sim,planner,recorder)
        if summary['finish_reason'] not in ('success','failure','timeout'):
            raise RuntimeError(f'Unexpected episode outcome: {summary}')
        summary.update(pair=case['pair'],variant=variant,paired_noise_seed=case['noise_seed'],
            physics=case['physics'],eval_scene_sha256=case['eval_scene_sha256'],
            wall_s=time.perf_counter()-started,all_plan_s=planner.durations,
            planner_nominal_mass_kg=.07,planner_nominal_friction=CONFIG['nominal_friction'])
        folder=OUT/variant;folder.mkdir(exist_ok=True)
        path=folder/f'pair_{case["pair"]:02d}.json'
        recorder.Save(path)
        save(folder/f'pair_{case["pair"]:02d}_metrics.json',summary)
        print(f'PAIR {case["pair"]:02d} {variant}: {summary["finish_reason"]}, '
              f'{summary["n_steps"]} steps, {summary["wall_s"]:.1f}s wall',flush=True)
        return summary
    finally:
        sim.Close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true')
    args_cli=parser.parse_args()
    cases=prepare()
    if args_cli.check:
        print('Validated 10 paired scenarios; masses, inertias, all friction groups and nominal planner checked. No dynamics run.')
        return 0
    import fcntl
    with (OUT/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import warp as wp
        row={k:str(v) for k,v in CONFIG['row'].items()};spec=bo.parseCell(row,OUT,OUT/'unused_bo_cell')
        args=spec.args;args.steps=CONFIG['max_steps'];args.n_episodes=10;args.save_steps=True
        save(OUT/'status.json',dict(status='starting',pid=os.getpid(),started_at_utc=now(),completed_episodes=0))
        unit=None
        try:
            with wp.ScopedDevice('cuda:0'):
                unit=_PlannerUnit(args,'M2')
                assert unit.sim.horizon==4
                # Capture once outside measured episodes. All episode seeds and
                # means are reset independently, so warmup consumes no test noise.
                warm=LocalPlanner(unit,CONFIG['variants']['baseline']['params'],0)
                initial=runtime.INITIAL[cases[0]['source_index']]
                warm.Reset();warm.request_activation=np.asarray(runtime.BASELINE['hardware_config']['robot_qref'])
                warm.Plan(np.array(initial['q']),np.array(initial['q_dot']),u=warm.request_activation)
                complete=[]
                for case in cases:
                    for variant in case['execution_order']:
                        path=OUT/variant/f'pair_{case["pair"]:02d}_metrics.json'
                        if path.exists():
                            summary=json.loads(path.read_text())
                            assert summary['eval_scene_sha256']==case['eval_scene_sha256']
                        else:
                            save(OUT/'status.json',dict(status='running',pid=os.getpid(),pair=case['pair'],
                                variant=variant,completed_episodes=len(complete),updated_at_utc=now()))
                            summary=episode(case,variant,args,unit)
                        complete.append(summary)
                save(OUT/'status.json',dict(status='complete',completed_episodes=len(complete),finished_at_utc=now()))
        except BaseException as exc:
            save(OUT/'status.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}',updated_at_utc=now()))
            raise
        finally:
            if unit is not None:unit.sim.Close()
    return 0


if __name__=='__main__':raise SystemExit(main())
