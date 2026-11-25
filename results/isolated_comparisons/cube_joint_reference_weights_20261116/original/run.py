"""Paired closed-loop simulation: change only the w_joint reference, no hardware."""
from pathlib import Path
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import sys
import time
import traceback

OUT = Path(__file__).resolve().parent
CONFIG = json.loads((OUT/'config.json').read_text())
sys.path.insert(0, CONFIG['code'])
import mujoco
import numpy as np
import warp as wp
from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig
from ContactModelStudy.Tasks.TaskBase import TaskRole
from ContactModelStudy.Simulators.Mujoco import Mujoco, MujocoConfig
from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config


def save(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def now():return datetime.now(timezone.utc).isoformat()


class Task(CubeReorient):
    def __init__(self, role, ref):
        super().__init__(LeapReorientConfig(role=role,timestep=CONFIG['timestep'],
            hand_acc='high',obj_acc='high',cost_weights=CONFIG['weights']))
        self.params['target_pos']=np.asarray(CONFIG['target_pos'])
        self.params['fallen_z']=CONFIG['fallen_z']
        self.params['init_ctrl']=np.asarray(CONFIG['initial_hand'])
        self.setJointReference(ref)
        self.setGoal(CONFIG['target_quat'])

    def getModelPath(self):return CONFIG['scenes'][self.role.value]


def initial(case):
    quat=np.zeros(4);quat[0]=np.sqrt(.5)
    quat[1+case%6//2]=(-1 if case%2==0 else 1)*np.sqrt(.5)
    return np.r_[CONFIG['initial_hand'],CONFIG['target_pos'],quat],np.zeros(22),np.asarray(CONFIG['initial_hand'])


def measure(task,q,v):
    e=task.goalErrors(q,v)
    e['angle_deg']=float(np.rad2deg(2*np.arcsin(np.sqrt(np.clip(e['quat'],0,1)))))
    e['joint_raw']=float(np.sum((q[:16]-task.jointReference())**2))
    return e


def episode(case,variant,folder):
    seed=CONFIG['seeds'][case]
    ref=CONFIG['references'][variant]
    task=Task(TaskRole.ROLLOUT,ref);evaluation=Task(TaskRole.EVAL,ref)
    rollout=sim=planner=None
    started=time.perf_counter()
    result=dict(case=case,variant=variant,rotation=CONFIG['rotations'][case%6],
                repetition=case//6+1,seed=seed,status='running',started_at_utc=now())
    arrays=dict(q=[],q_dot=[],u=[],planning_s=[],pos_error=[],quat_error=[],angle_deg=[],joint_raw=[],target_delta=[])
    try:
        q0,v0,u0=initial(case)
        rollout=GetContactModelSim('M2',task.getModelPath(),N=CONFIG['n_samples'],task=task,
            timestep=CONFIG['timestep'],ctrl_time_step=CONFIG['ctrl_time_step'],
            time_horizon=CONFIG['time_horizon'],nconmax=CONFIG['nconmax'],njmax=CONFIG['njmax'],device='cuda:0')
        cfg=MPPI_Config(noise_sigma=CONFIG['noise_sigma'],temperature=CONFIG['temperature'],
            warm_start='shift',seed=seed,control_mode='pos_relative',delta_range=(None,None),
            n_iterations=1,action_rate_weight=0.,resample_noise=False,center_noise=True,
            adaptive_temp=False,use_graph=True)
        planner=MPPI(rollout,task,cfg)
        assert not planner._has_clip
        assert rollout.horizon==5 and rollout.config.resolved_substeps==16
        planner.Reset()
        # A separate disposable planning call captures kernels; Reset + restoring
        # the planner RNG state ensures warmup consumes no experimental noise.
        saved=planner.SaveState()
        warm=time.perf_counter();planner.Plan(q0,v0,u=u0);wp.synchronize()
        result['warmup_s']=time.perf_counter()-warm
        planner.LoadState(saved)
        sim=Mujoco(evaluation.getModelPath(),MujocoConfig(timestep=CONFIG['timestep']))
        assert sim.mjm.na==0
        assert np.isclose(sim.mjm.body('obj').mass[0],.07)
        mujoco.mj_resetData(sim.mjm,sim.mjd)
        sim.SetState(q0,v0);sim.SetControl(u0)
        result.update(initial_q=q0.tolist(),initial_v=v0.tolist(),initial_u=u0.tolist(),
            initial_state_sha256=hashlib.sha256(q0.tobytes()+v0.tobytes()+u0.tobytes()).hexdigest(),
            noise_seed=int(planner._noise_seed),initial_resample_count=int(planner._resample_count),
            joint_reference=list(ref),planner_config=asdict(cfg),rollout_config=asdict(rollout.config),
            initial_contact_count=int(sim.mjd.ncon),
            initial_min_contact_distance_m=float(min((c.dist for c in sim.mjd.contact),default=0)))
        last_u=u0.copy();limited_joints=limited_steps=0
        control_started=time.perf_counter();outcome='timeout'
        for step in range(CONFIG['max_steps']+1):
            q,v=sim.GetState();e=measure(evaluation,q,v)
            if not np.isfinite(q).all() or not np.isfinite(v).all():outcome='numerical_failure';break
            if evaluation.isFailure(sim):outcome='drop';break
            if evaluation.isSuccess(sim):outcome='success';break
            if step==CONFIG['max_steps']:break
            tick=time.perf_counter();u=planner.Plan(q,v,u=sim.GetControl());wp.synchronize();duration=time.perf_counter()-tick
            if not planner.last_plan_ok or not np.isfinite(u).all():
                outcome='planner_failure'
                costs=planner.costs_wp.numpy()
                result['finite_rollout_costs']=int(np.isfinite(costs).sum());break
            # Raw planner targets: no hardware ctrl_range or extra target/delta clip.
            arrays['q'].append(q);arrays['q_dot'].append(v);arrays['u'].append(u.copy())
            arrays['planning_s'].append(duration)
            for key in ['pos_error','quat_error','angle_deg','joint_raw']:
                arrays[key].append(e[{'pos_error':'pos','quat_error':'quat'}.get(key,key)])
            arrays['target_delta'].append(u-last_u)
            lohi=sim.mjm.actuator_ctrlrange
            outside=((u<lohi[:,0])|(u>lohi[:,1])) & sim.mjm.actuator_ctrllimited.astype(bool)
            limited_joints+=int(outside.sum());limited_steps+=int(outside.any())
            sim.SetControl(u);sim.Step(16);last_u=u.copy()
            if step%100==0:
                save(folder/'status.json',dict(status='running',case=case,rotation=result['rotation'],
                     step=step,completed_episodes=len(list(folder.glob('episode_*.json'))),updated_at_utc=now()))
        q,v=sim.GetState();final=measure(evaluation,q,v)
        plans=np.asarray(arrays['planning_s'])
        errors=np.asarray(arrays['angle_deg']+[final['angle_deg']])
        result.update(status='complete',outcome=outcome,success=outcome=='success',n_steps=len(plans),
            simulation_s=len(plans)*.064,control_wall_s=time.perf_counter()-control_started,
            final_q=q.tolist(),final_v=v.tolist(),final_errors=final,
            minimum_angle_deg=float(errors.min()),
            planning_mean_ms=float(plans.mean()*1000) if len(plans) else None,
            planning_p95_ms=float(np.percentile(plans,95)*1000) if len(plans) else None,
            targets_outside_native_actuator_range_steps=limited_steps,
            targets_outside_native_actuator_range_joints=limited_joints,
            hardware_clipped_steps=0,total_wall_s=time.perf_counter()-started,finished_at_utc=now())
        arrays.update(final_q=q,final_v=v)
    except BaseException as exc:
        result.update(status='error',outcome='error',success=False,error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc(),finished_at_utc=now())
        raise
    finally:
        np.savez_compressed(folder/f'episode_{case+1:02d}.npz',**{k:np.asarray(v) for k,v in arrays.items()})
        save(folder/f'episode_{case+1:02d}.json',result)
        if rollout is not None:rollout.Close()
        if sim is not None:sim.Close()
        del planner,rollout,sim
        gc.collect()
    print('EPISODE',variant,case+1,result['rotation'],result['outcome'],result['n_steps'],
          'final angle',round(result['final_errors']['angle_deg'],2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference',choices=['optimization','hardware'],required=True)
    args=parser.parse_args();variant=args.reference
    for rel,digest in CONFIG['source_sha256'].items():
        assert hashlib.sha256((Path(CONFIG['code'])/rel).read_bytes()).hexdigest()==digest,rel
    folder=OUT/variant;folder.mkdir(exist_ok=True)
    save(folder/'status.json',dict(status='starting',pid=os.getpid(),started_at_utc=now()))
    try:
        with wp.ScopedDevice('cuda:0'):
            for case in range(12):
                dest=folder/f'episode_{case+1:02d}.json'
                if dest.exists() and json.loads(dest.read_text()).get('status')=='complete':continue
                episode(case,variant,folder)
        save(folder/'status.json',dict(status='complete',completed_episodes=12,finished_at_utc=now()))
    except BaseException as exc:
        save(folder/'status.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at_utc=now()))
        raise


if __name__=='__main__':main()
