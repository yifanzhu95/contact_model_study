"""Offline hardware-weight screening. No LCM, serial, camera or motor operations."""
import argparse
from collections import deque
import ast
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
import traceback

import mujoco
import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import run_leap_hardware as runner
from ContactModelStudy.Hardware.Workflow import build_tasks,build_rollout
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI

OUT=Path(__file__).resolve().parent
MATRIX=json.loads((OUT/'matrix.json').read_text())
STATES=json.loads((OUT/'starting_states.json').read_text())
PROTECTED=json.loads((OUT/'protected_baseline.json').read_text())
body='\n'.join(line.split('#',1)[1] for line in PROTECTED['protected_text'].splitlines())
BASE=ast.literal_eval(body[body.index('{'):body.rindex('}')+1])

def weights(quat=25,pos=50,contact=25,joint=1,velocity=.01,tquat=150,tpos=300):
    return dict(w_quat=quat,w_pos_x=pos,w_pos_y=pos,w_pos_z=pos,w_velo=0.,
        w_contact=contact,w_joint=joint,w_joint_velo=velocity,w_fallen=0.,
        w_quat_term=tquat,w_pos_term=tpos,w_fallen_term=0.)

CANDIDATES=dict(protected=BASE,
    hardware_success=weights(quat=20,contact=10,joint=.5,velocity=0,tquat=100,tpos=100),
    balanced=weights(),grasp=weights(contact=75,tquat=100),
    endpoint=weights(contact=10,joint=.5,velocity=.005,tquat=300),
    posture=weights(joint=5),smooth=weights(velocity=.05),
    gentle_endpoint=weights(tquat=75,tpos=150),
    support={**weights(), 'w_fallen':25., 'w_fallen_term':50.},
    support_posture={**weights(joint=3,contact=40), 'w_fallen':25., 'w_fallen_term':50.},
    holding=weights(joint=5,velocity=.05),
    holding20=weights(joint=20,velocity=.05),
    holding50=weights(joint=50,velocity=.05),
    position600=weights(velocity=.05,tpos=600),
    smooth_floor={**weights(velocity=.05), 'w_fallen':5., 'w_fallen_term':10.},
    smooth_posture={**weights(joint=3,velocity=.05), 'w_fallen':5., 'w_fallen_term':10.},
    support_smooth={**weights(joint=3,contact=40,velocity=.1), 'w_fallen':25., 'w_fallen_term':50.},
    support_position={**weights(joint=3,contact=40,velocity=.05,tpos=600), 'w_fallen':25., 'w_fallen_term':50.})

def errors(q,v,target,goal):
    return dict(pos=float(np.linalg.norm(q[16:19]-target)),
        quat=float(max(0,1-np.dot(q[19:23],goal)**2)),vel=float(np.linalg.norm(v[16:22])))

def success(e):return e['pos']<.02 and e['quat']<.04 and e['vel']<10

def effective(costs,temp):
    costs=np.asarray(costs,dtype=float)
    good=np.isfinite(costs)
    if not good.any():return 0.
    a=np.zeros_like(costs)
    a[good]=np.exp(-(costs[good]-costs[good].min())/temp)
    a/=a.sum()
    return float(1/np.sum(a*a))

def temperature_for_ess(costs,fraction=.15):
    finite=costs[np.isfinite(costs)]
    if len(finite)==0:raise RuntimeError('No finite candidate costs')
    if np.ptp(finite)<1e-9:return .1
    lo,hi=1e-6,max(1.,float(np.std(finite))*100)
    for _ in range(45):
        mid=np.sqrt(lo*hi)
        if effective(costs,mid)<fraction*len(costs):lo=mid
        else:hi=mid
    return float(np.sqrt(lo*hi))

class Cell:
    def __init__(self,row,n,sigma):
        self.row=row
        self.directory=tempfile.TemporaryDirectory(prefix='leap_weight_cell_')
        self.args=runner.parseArgs(['--device','cuda:0','--task','cube_reorient',
            '--rollout-model',row['model'],'--hand-acc',row['hand'],'--obj-acc','high',
            '--n-samples',str(n),'--time-horizon',str(row['horizon']),
            '--ctrl-time-step','.064','--noise-sigma',str(sigma),'--seed','2026',
            '--action-rate-weight','0'])  # Historical study predates this cost.
        self.task,self.eval_task,_=build_tasks(self.args,self.directory.name)
        self.home=np.array(json.loads((ROOT/'configs/leap_hardware.json').read_text())['robot_qref'])
        for t in [self.task,self.eval_task]:t.params['init_ctrl']=self.home.copy()
        self.sim=build_rollout(self.args,self.task,*runner.model_limits(self.eval_task.mjm))
        self.planner=MPPI(self.sim,self.task,runner.reference.buildPlannerConfig(self.args))
        self.eval_model=mujoco.MjModel.from_xml_path(self.eval_task.getModelPath())
        self.eval_model.opt.timestep=.002
        self.data=mujoco.MjData(self.eval_model)
        self.original_friction=self.eval_model.geom_friction.copy()
        self.original_gain=self.eval_model.actuator_gainprm.copy()
        self.original_bias=self.eval_model.actuator_biasprm.copy()
        self.original_mass=self.eval_model.body_mass.copy()
        self.original_inertia=self.eval_model.body_inertia.copy()
        self.original_joint_ranges=self.eval_model.jnt_range.copy()
        self.original_ctrl_limited=self.eval_model.actuator_ctrllimited.copy()
        self.target=np.array(STATES[0]['target'])
        self.goal=np.array([1.,0.,0.,0.])
        self.task.setTargetPosition(self.target)
        self.task.setGoal(self.goal)

    def prepare(self,w):self.task.setCostWeights(w)

    def calibrate(self,w,indices,fraction):
        self.prepare(w)
        temps=[]
        timings=[]
        for i in indices:
            s=STATES[i];q,v=np.array(s['q']),np.array(s['v'])
            self.planner.Reset();self.planner.UpdateConfig(temperature=1.)
            self.planner._noise_seed=2026+i
            self.planner._resample_count=0
            self.sim.activation.assign(q[:16].astype(np.float32))
            start=time.perf_counter()
            self.planner.Plan(q,v,u=q[:16])
            timings.append(time.perf_counter()-start)
            temps.append(temperature_for_ess(self.planner.costs_wp.numpy(),fraction))
        return float(np.median(temps)),dict(calibrated_temperatures=temps,calibration_times_s=timings)

    def episode(self,w,temp,index,max_steps,period=.064,friction_scale=1.,gain_scale=1.,mass_scale=1.,sensor_stress=False,hardware_range_stress=False):
        episode_started=time.monotonic()
        self.prepare(w)
        self.planner.UpdateConfig(temperature=temp)
        self.planner.Reset()
        self.planner._noise_seed=2026+index
        self.planner._resample_count=0
        self.eval_model.geom_friction[:]=self.original_friction*friction_scale
        self.eval_model.actuator_gainprm[:]=self.original_gain
        self.eval_model.actuator_biasprm[:]=self.original_bias
        self.eval_model.actuator_gainprm[:,0]*=gain_scale
        self.eval_model.actuator_biasprm[:,1]*=gain_scale
        self.eval_model.jnt_range[:]=self.original_joint_ranges
        self.eval_model.actuator_ctrllimited[:]=self.original_ctrl_limited
        if hardware_range_stress:
            for name in ['if_mcp','mf_mcp','rf_mcp']:
                self.eval_model.jnt_range[self.eval_model.joint(name).id,0]=-.36
            self.eval_model.actuator_ctrllimited[:]=0
        self.eval_model.body_mass[:]=self.original_mass
        self.eval_model.body_inertia[:]=self.original_inertia
        obj=self.eval_model.body('obj').id
        self.eval_model.body_mass[obj]*=mass_scale
        self.eval_model.body_inertia[obj]*=mass_scale
        mujoco.mj_setConst(self.eval_model,self.data)
        mujoco.mj_resetData(self.eval_model,self.data)
        self.data.qpos[:]=STATES[index]['q']
        self.data.qvel[:]=STATES[index]['v']
        self.data.act[:]=self.home
        self.data.ctrl[:]=self.home
        mujoco.mj_forward(self.eval_model,self.data)
        initial=errors(self.data.qpos,self.data.qvel,self.target,self.goal)
        history=deque([(self.data.time,self.data.qpos.copy(),self.data.qvel.copy())],maxlen=128)
        sensor_rng=np.random.default_rng(9000+index)
        def integrate(n):
            for _ in range(n):
                mujoco.mj_step(self.eval_model,self.data)
                history.append((self.data.time,self.data.qpos.copy(),self.data.qvel.copy()))
        times=[];ess=[];deltas=[];changes=[];trajectory=[]
        previous=self.home.copy()
        reason='timeout';hit=False;held=False
        for step in range(max_steps):
            q,v=self.data.qpos.copy(),self.data.qvel.copy()
            e=errors(q,v,self.target,self.goal)
            if step%5==0:
                trajectory.append(dict(step=step,time_s=float(self.data.time),qpos=q.tolist(),errors=e))
            if not np.isfinite(q).all() or not np.isfinite(v).all() or q[18]<0:
                reason='failure';break
            if success(e):
                hit=True;reason='success';break
            if sensor_stress:
                hand=min(history,key=lambda h:abs(h[0]-(self.data.time-.02)))
                obj=min(history,key=lambda h:abs(h[0]-(self.data.time-.08)))
                q[:16],v[:16]=hand[1][:16],hand[2][:16]
                q[16:],v[16:]=obj[1][16:],obj[2][16:]
                q[:16]=np.round(q[:16]/.001534)*.001534
                q[16:19]+=sensor_rng.normal(0,.0015,3)
                r=sensor_rng.normal(0,np.radians(.5),3)
                angle=np.linalg.norm(r)
                perturb=np.r_[np.cos(angle/2),np.sin(angle/2)*r/max(angle,1e-12)]
                quat=np.empty(4)
                mujoco.mju_mulQuat(quat,q[19:23],perturb)
                q[19:23]=quat/np.linalg.norm(quat)
                v[16:19]+=sensor_rng.normal(0,.02,3)
                v[19:22]+=sensor_rng.normal(0,.15,3)
            self.sim.activation.assign(self.data.act.astype(np.float32))
            start=time.perf_counter()
            proposed=self.planner.Plan(q,v,u=previous)
            times.append(time.perf_counter()-start)
            if not self.planner.last_plan_ok:
                reason='planner_failed';break
            delta=np.asarray(proposed)-q[:16]
            deltas.append(float(np.max(np.abs(delta))))
            if step%10==0:
                a=self.planner.w_wp.numpy().astype(float)
                ess.append(float(1/(a@a)))
            # Model asynchronous hardware: the hand keeps the old target
            # during the first half-cycle, then the bridge adds delta to its
            # newly measured joints. No absolute-target clipping is introduced.
            nsteps=round(period/self.eval_model.opt.timestep)
            integrate(nsteps//2)
            command=self.data.qpos[:16]+np.clip(delta,-3.14,3.14)
            changes.append(float(np.linalg.norm(command-previous)))
            self.data.ctrl[:]=command
            previous=command.copy()
            integrate(nsteps-nsteps//2)
        final=errors(self.data.qpos,self.data.qvel,self.target,self.goal)
        if success(final):hit=True;reason='success'
        at_hit=final.copy()
        if hit:
            self.data.ctrl[:]=self.data.qpos[:16]
            for _ in range(round(.5/self.eval_model.opt.timestep)):
                mujoco.mj_step(self.eval_model,self.data)
            held=success(errors(self.data.qpos,self.data.qvel,self.target,self.goal))
        hold_error=errors(self.data.qpos,self.data.qvel,self.target,self.goal)
        score=(10*(reason in ['failure','planner_failed']) + min(final['quat']/.04,10)
            +min(final['pos']/.02,5)-12*hit-4*held
            +.05*(np.mean(changes) if changes else 0))
        return dict(wall_s=time.monotonic()-episode_started,state_index=index,source_state=STATES[index]['source'],finish=reason,
            initial_errors=initial,final_errors=at_hit,hold_errors=hold_error,success=hit,hold_success=held,
            score=float(score),steps=len(times),period=period,friction_scale=friction_scale,gain_scale=gain_scale,mass_scale=mass_scale,sensor_stress=sensor_stress,hardware_range_stress=hardware_range_stress,
            planning_median_s=float(np.median(times)) if times else None,
            planning_p95_s=float(np.percentile(times,95)) if times else None,
            planning_max_s=max(times) if times else None,
            ess_median=float(np.median(ess)) if ess else None,
            max_delta_rad=max(deltas) if deltas else 0.,
            command_change_mean=float(np.mean(changes)) if changes else 0.,trajectory=trajectory)

    def close(self):
        self.sim.Close()
        self.planner=None
        self.directory.cleanup()
        gc.collect()

def append(record,path):
    with path.open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
    print(json.dumps(record,allow_nan=False),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',default='2')
    p.add_argument('--candidates',default='hardware_success,balanced')
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--noise',type=float,default=.05)
    p.add_argument('--states',default='0,1')
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--ess-fraction',type=float,default=.15)
    p.add_argument('--temperature',type=float)
    p.add_argument('--temperatures')
    p.add_argument('--period',type=float,default=.064)
    p.add_argument('--friction-scale',type=float,default=1.)
    p.add_argument('--gain-scale',type=float,default=1.)
    p.add_argument('--mass-scale',type=float,default=1.)
    p.add_argument('--sensor-stress',action='store_true')
    p.add_argument('--hardware-range-stress',action='store_true')
    p.add_argument('--label',default='pilot')
    p.add_argument('--calibrate-only',action='store_true')
    p.add_argument('--timing-plans',type=int,default=0)
    a=p.parse_args()
    target=OUT/(a.label+'.jsonl')
    rows=[r for r in MATRIX if a.rows=='all' or r['index'] in [int(x) for x in a.rows.split(',')]]
    indices=[int(x) for x in a.states.split(',')]
    with wp.ScopedDevice('cuda:0'):
        for row in rows:
            cell=None
            try:
                cell=Cell(row,a.samples,a.noise)
                for name in a.candidates.split(','):
                    w=CANDIDATES[name]
                    temp,calibration=cell.calibrate(w,indices[:2],a.ess_fraction)
                    temperatures=([float(x) for x in a.temperatures.split(',')] if a.temperatures
                                  else [a.temperature if a.temperature is not None else temp])
                    for temp in temperatures:
                        record=dict(row=row,candidate=name,weights=w,n_samples=a.samples,noise=a.noise,
                            temperature=temp,ess_target_fraction=a.ess_fraction,calibration=calibration,
                            rollout_config=asdict(cell.sim.config),fallen_z=cell.task.params['fallen_z'],simulation_only=True)
                        if a.timing_plans:
                            timings=[]
                            state=STATES[indices[0]]
                            q,v=np.array(state['q']),np.array(state['v'])
                            for _ in range(a.timing_plans):
                                cell.planner.Reset()
                                cell.sim.activation.assign(q[:16].astype(np.float32))
                                start=time.perf_counter()
                                cell.planner.Plan(q,v,u=q[:16])
                                timings.append(time.perf_counter()-start)
                            record['steady_plan_seconds']=timings
                        record['episodes']=[] if a.calibrate_only else [cell.episode(w,temp,i,a.steps,
                            a.period,a.friction_scale,a.gain_scale,a.mass_scale,a.sensor_stress,a.hardware_range_stress) for i in indices]
                        record['mean_score']=float(np.mean([e['score'] for e in record['episodes']])) if record['episodes'] else None
                        append(record,target)
            except Exception as exc:
                traceback.print_exc()
                append(dict(row=row,error=f'{type(exc).__name__}: {exc}',simulation_only=True),target)
            finally:
                if cell is not None:cell.close()

if __name__=='__main__':main()
