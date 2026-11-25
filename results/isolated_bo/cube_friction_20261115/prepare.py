"""Freeze the committed sim-to-sim workflow and build isolated friction/mass scenes."""
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CODE = OUT/'code'
TARGET = ROOT/'ContactModelStudy/Tasks/CubeReorient.py'
WEIGHT_SPACE = {'w_quat':[.01,100.], 'w_pos_x':[.01,100.], 'w_pos_y':[.01,100.],
    'w_pos_z':[.01,100.], 'w_contact':[.01,100.], 'w_joint':[.01,100.],
    'w_fallen':[10.,500.], 'w_quat_term':[10.,1000.], 'w_pos_term':[10.,1000.]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT/'manifest.json').exists():
        raise SystemExit('Already prepared; use existing manifest and queue, do not overwrite.')
    original_bytes = TARGET.read_bytes()
    ref = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    archive = subprocess.check_output(['git','archive',ref,'ContactModelStudy','experiments','scenes/leap'],cwd=ROOT)
    CODE.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(CODE,filter='data')
    # Only the isolated driver copy gains a scene-directory argument. The main
    # checkout, task costs, simulators and optimization algorithm are unedited.
    driver = CODE/'ContactModelStudy/Drivers/run_episodes.py'
    text = driver.read_text()
    anchor = '    g.add_argument("--obj-acc", default="high",'
    assert text.count(anchor)==1
    text = text.replace(anchor,'    g.add_argument("--scenes-dir", type=Path, default=None)\n'+anchor)
    anchor = 'goal_difficulty=args.goal_difficulty, cost_weights=args.cost_weights)'
    assert text.count(anchor)==1
    text = text.replace(anchor,'goal_difficulty=args.goal_difficulty, cost_weights=args.cost_weights, scenes_dir=args.scenes_dir)')
    ast.parse(text)
    driver.write_text(text)
    validations = {}
    tips = {'if_tip','mf_tip','rf_tip','th_tip'}
    for mass in [.07,.14]:
        name = f'mass_{round(mass*1000):03d}g'
        scene_root = OUT/'scenes'/name
        for role,filename in [('rollout','env_leap_rollout_cube_high_high.xml'),('eval','env_leap_eval_cube.xml')]:
            source = CODE/'scenes/leap'/filename
            original = mujoco.MjModel.from_xml_path(str(source))
            assert original.na==0, 'An uncalibrated scene must not contain filtered calibrated servos'
            assert np.isclose(original.body('obj').mass[0],.14)
            spec = mujoco.MjSpec.from_file(str(source))
            spec.meshdir = str((source.parent/spec.meshdir).resolve())
            spec.texturedir = str((source.parent/spec.texturedir).resolve())
            groups = dict(cube=[],fingertip=[],palm=[],finger=[])
            for body in spec.bodies:
                for geom in body.geoms:
                    if not (geom.contype or geom.conaffinity):continue
                    if body.name=='obj':
                        group='cube'; geom.mass=mass
                    elif body.name=='palm':group='palm'
                    elif body.name.startswith(('if_','mf_','rf_','th_')):
                        group='fingertip' if geom.name in tips else 'finger'
                    else:continue
                    geom.friction=[dict(cube=.3,fingertip=.5,palm=.4,finger=.4)[group],.005,.0001]
                    groups[group].append(geom.name)
            assert all(groups.values()),groups
            path = scene_root/'leap'/filename
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(spec.to_xml())
            compiled=mujoco.MjModel.from_xml_path(str(path))
            assert (compiled.nq,compiled.nv,compiled.nu,compiled.na)==(23,22,16,0)
            obj=compiled.body('obj').id
            assert np.isclose(compiled.body_mass[obj],mass)
            np.testing.assert_allclose(compiled.body_inertia[obj],original.body_inertia[obj]*(mass/.14),rtol=1e-6)
            for field in ['actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_dyntype',
                          'dof_damping','dof_armature','dof_frictionloss','jnt_range','body_pos','body_quat',
                          'geom_type','geom_size','geom_pos','geom_quat','geom_condim','geom_priority']:
                np.testing.assert_allclose(getattr(compiled,field),getattr(original,field),rtol=1e-5,atol=1e-7)
            mask=np.arange(compiled.nbody)!=obj
            np.testing.assert_allclose(compiled.body_mass[mask],original.body_mass[mask],rtol=1e-5)
            for group,names in groups.items():
                expected=[dict(cube=.3,fingertip=.5,palm=.4,finger=.4)[group],.005,.0001]
                for geom_name in names:np.testing.assert_allclose(compiled.geom(geom_name).friction,expected)
            validations[f'{name}_{role}']=dict(path=str(path),sha256=sha(path),mass_kg=mass,
                inertia=compiled.body_inertia[obj].tolist(),uncalibrated=True,servo_activations=compiled.na,
                groups=groups,hand_dynamics_and_geometry_match_committed_source=True)
    order=[(.07,.1),(.14,.1)]+[(m,t) for t in [.05,.075,.15,.2] for m in [.07,.14]]
    rows=[]
    for mass,temp in order:
        rows.append(dict(label=f'cube_mass{round(mass*1000):03d}g_T{temp:g}',task='cube_reorient',
            rollout_models='M1 M2 M3 M4',model_agg='worst',eval_sim='mujoco',hand_acc='high',obj_acc='high',
            scenes_dir=str(OUT/'scenes'/f'mass_{round(mass*1000):03d}g'),n_episodes=4,steps=1000,seed=64,
            goal_difficulty=4,timestep=.004,eval_steps_per_rollout_step=1,ctrl_time_step=.064,time_horizon=.352,
            n_samples=256,n_iterations=1,temperature=temp,noise_sigma=.05,control_mode='pos_relative',
            warm_start='none',resample_noise='false',center_noise='true',adaptive_temp='false',graph='true',
            plan_on_eval='false',stop_on_success='true',settle=1.,nconmax=50,njmax=300,save_steps='false',
            uncertainty='false',opt_weights=' '.join(f'{key}:{low}:{high}' for key,(low,high) in WEIGHT_SPACE.items()),
            n_calls=750,n_initial_points=120,acq_func='gp_hedge',bo_seed=0,w_success=1.,w_cost=.1,err_clip=250.,
            seed_defaults='true',trials_in_flight=1,w_velo=0.,w_joint_velo=0.,w_fallen_term=0.))
    csv_path=OUT/'searches.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'CubeReorient.before.py').write_bytes(original_bytes)
    config=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),source_git_commit=ref,
        python='/home/chetan/miniconda3/envs/mj_env/bin/python',root=str(ROOT),code=str(CODE),out=str(OUT),
        csv=str(csv_path),results=str(OUT/'search_results'),gpus=[
            'GPU-9d410eae-46e7-c13e-9c03-a44cbda70781','GPU-3beb4f90-65ae-0ef8-e126-97a81e20ab17'],
        physical_gpu_indices=[2,3],planners_per_gpu=1,workers=16,searches=rows,weight_space=WEIGHT_SPACE,
        target_file=str(TARGET),target_before_sha256=hashlib.sha256(original_bytes).hexdigest(),
        source_snapshot_sha256={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*.py')},
        scene_validation=validations,
        isolation='Committed code snapshot, original uncalibrated hand, frozen original task state/goal. Only isolated scene-directory CLI adapter added. Shared source scene files and active weights unchanged.',
        evaluation_choice='MuJoCo current-BO default; exact historical winning configuration unavailable.',
        objective='max_model(-success_rate + 0.1 * mean(min(sum(error/threshold),250)/250))',
        update_rule='After cell 0 (70g, T=0.1, noise=0.05) completes 750 trials, write its best weights only to the final commented cost_weights block in the original checkout; preserve executable AST.',
        budget=dict(searches=10,trials_per_search=750,episodes_per_trial=16,total_episode_evaluations=120000),
        no_hardware=True)
    (OUT/'manifest.json').write_text(json.dumps(config,indent=2)+'\n')
    assert TARGET.read_bytes()==original_bytes
    print(json.dumps(dict(manifest=str(OUT/'manifest.json'),searches=len(rows),scene_checks=list(validations)),indent=2))


if __name__=='__main__':main()
