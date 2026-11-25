"""Isolated baseline BO adapters. Imported in parent and spawned pool children.

No LCM, USB, camera, viewer or hardware runner is imported or contacted.
Only --start creates simulators/planners; --check constructs CPU models only.
"""
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

HERE=Path(__file__).resolve().parent
WORK=HERE.parent if HERE.name=='code' else HERE
BASELINE=json.loads((WORK/'baseline.json').read_text())
INITIAL=json.loads((WORK/'initial_states.json').read_text())
MANIFEST=None
EXTRA=('action_rate_weight','joint_limit_weight','plan_blend_alpha')


def final_target(base,delta,baseline=BASELINE):
    margins=baseline['hardware_config']['ctrl_range'];limits=baseline['joint_limits']
    stage1=np.clip(base+delta,base-np.asarray(margins['lower']),base+np.asarray(margins['upper']))
    return np.clip(stage1,limits['lower'],limits['upper']),stage1


def install(bo):
    """Extend only the frozen BO/driver/pool in this workspace."""
    from ContactModelStudy.Drivers import run_episodes as drv
    from ContactModelStudy.Drivers import EpisodePool as pool
    from ContactModelStudy.Hardware.Rollouts import CalibratedRollouts
    from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI

    original_parser=drv.build_parser
    def parser(*a,**kw):
        p=original_parser(*a,**kw)
        p.set_defaults(action_rate_weight=BASELINE['settings']['action_rate_weight'])
        p.add_argument('--joint-limit-weight',type=float,default=100.)
        p.add_argument('--plan-blend-alpha',type=float,default=.75)
        return p
    drv.build_parser=parser
    original_tasks=drv.buildTasks
    def tasks(*a,**kw):
        task,evaluation=original_tasks(*a,**kw)
        for t in (task,evaluation):
            t.setCostWeights({**BASELINE['weights'],**(t.config.cost_weights or {})})
            t.setJointReference(BASELINE['joint_reference'])
            t.params['target_pos']=np.array(BASELINE['target_pos'],dtype=float)
            t.params['fallen_z']=BASELINE['cost_drop_height']
            t.FAILURE_Z=BASELINE['failure_z']
            t.setGoal([1,0,0,0])
        return task,evaluation
    drv.buildTasks=tasks
    original_config=drv.buildPlannerConfig
    def planner_config(args):
        # Positive base gates allocate/capture both kernels exactly once.
        # The actual per-trial weights live in a 2-float GPU buffer.
        cfg=original_config(args)
        return replace(cfg,action_rate_weight=10.,joint_limit_weight=100.,
            joint_limit_lower=tuple(BASELINE['joint_limits']['lower']),
            joint_limit_upper=tuple(BASELINE['joint_limits']['upper']),plan_blend_alpha=.75)
    drv.buildPlannerConfig=planner_config

    def unit_init(self,args,model):
        self.task,_=drv.buildTasks(args)
        backend=GetContactModelSim(model,self.task.getModelPath(),N=args.n_samples,
            task=self.task,**drv.rolloutSimOverrides(args,self.task))
        self.sim=CalibratedRollouts(backend)
        self.planner=MPPI(self.sim,self.task,drv.buildPlannerConfig(args))
        self.base_config=self.planner.config;self.base_weights=dict(self.task.params['cost_weights'])
        self.goal=None;self.settings=((),())
    pool._PlannerUnit.__init__=unit_init
    original_apply=pool._PlannerUnit.apply
    def apply(self,params,weights):
        params=dict(params or {})
        penalties=[params.pop('action_rate_weight',10.),params.pop('joint_limit_weight',100.)]
        params.pop('plan_blend_alpha',None)  # Applied in the episode worker.
        if not np.isfinite(penalties).all() or min(penalties)<0:raise ValueError('Invalid runtime penalty weights')
        self.planner._runtime_penalties_wp.assign(np.asarray(penalties,dtype=np.float32))
        original_apply(self,params,weights)
    pool._PlannerUnit.apply=apply
    drv.runRecordedEpisode=run_episode

    # Uniform ranges permit exact zero penalties and alpha=0/1. Keep the nine
    # legacy cost dimensions log-uniform, at their previous bounds.
    original_space=bo.Dim.skopt
    def dimension(self):
        if self.name in EXTRA:
            from skopt.space import Real
            return Real(self.lo,self.hi,prior='uniform',name=self.name)
        return original_space(self)
    bo.Dim.skopt=dimension
    original_pinned=bo.CellSpec.pinned
    bo.CellSpec.pinned=lambda self,knob:float(getattr(self.args,knob)) if knob in EXTRA else original_pinned(self,knob)


def run_episode(index,args,task,eval_task,sim,planner,recorder):
    """Paced virtual-time planning/application, matching measured-position mode.

    Each plan sees the state at the start of its cycle. While the plan is
    pending, the plant advances 64ms under the previous held target. The new
    delta is then blended, rebased on the updated joints and clipped twice.
    GPU queue/compute wall time never changes virtual physics time.

    This does NOT reproduce camera/USB delays or tracking errors. Initial states
    are already in world coordinates from the saved Method-5 calibration.
    """
    import mujoco
    from ContactModelStudy.Simulators.Mujoco import MujocoConfig, Mujoco
    from ContactModelStudy.Hardware.PlanBlending import TemporalPlanBlend
    Mujoco.applyContactParams(sim.mjm,MujocoConfig(cone='pyramidal',solver='Newton',iterations=25,tolerance=1e-6))
    sim.mjm.opt.gravity[:]=BASELINE['hardware_config']['gravity_mujoco']
    initial=INITIAL[index]
    mujoco.mj_resetData(sim.mjm,sim.mjd)
    sim.SetState(np.array(initial['q']),np.array(initial['q_dot']))
    held=np.array(BASELINE['hardware_config']['robot_qref'],dtype=float)
    sim.SetControl(held);sim.mjd.act[:]=held;mujoco.mj_forward(sim.mjm,sim.mjd)
    for t in (task,eval_task):t.setGoal(initial['goal'])
    planner.Reset(None);recorder.recordGoal(initial['goal'],label=initial['rotation'])
    period=BASELINE['settings']['ctrl_time_step'];substeps=round(period/sim.timestep)
    blend=TemporalPlanBlend(planner.config.plan_blend_alpha,period)
    counts=dict(delta_clipped=0,absolute_clipped=0,applied_actions=0)
    end='timeout';virtual_time=0.;plans=0
    for step in range(args.steps):
        if eval_task.isFailure(sim):end='failure';break
        if eval_task.isSuccess(sim):recorder.recordSuccess();end='success';break
        q,v=sim.GetState()
        planner.request_activation=sim.mjd.act.copy()
        started=time.perf_counter()
        proposed=planner.Plan(q,v,u=held)
        plan_seconds=planner.last_plan_seconds or time.perf_counter()-started
        if isinstance(proposed,tuple):proposed=proposed[0]
        if not np.isfinite(proposed).all():raise RuntimeError('Planner returned a nonfinite command')
        plans+=1
        # Existing absolute target is held while planning/pacing happens.
        sim.Step(substeps);virtual_time+=period
        if eval_task.isFailure(sim):end='failure';break
        if eval_task.isSuccess(sim):recorder.recordSuccess();end='success';break
        application_q,_=sim.GetState()
        delta=np.asarray(proposed)-q[:16]
        delta,context=blend.prepare(planner.last_mean,delta,virtual_time)
        final,first=final_target(application_q[:16],delta)
        counts['delta_clipped']+=int(np.any(abs(first-(application_q[:16]+delta))>1e-7))
        counts['absolute_clipped']+=int(np.any(abs(final-first)>1e-7))
        recorder.recordStateAndAction(q,v,final,planning_time=plan_seconds)
        sim.SetControl(final);held=final.copy();blend.commit(context,virtual_time)
        counts['applied_actions']+=1
    recorder.episodeFinished(end,episode=index,settle_s=0.,simulation_s=virtual_time,
        source_run=initial['source_run'],source_episode=initial['source_episode'],
        rotation=initial['rotation'],initial_state_sha256=initial['sha256'],
        planner_steps=plans,final_target_clipping=counts,
        joint_cost_reference='P011',timing='virtual 64ms before applying each plan; no GPU wall-time pacing')
    return recorder.GenerateSummary(-1)
