"""Read-only saved-state contact cost audit; never uses LCM or hardware."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import mujoco
import numpy as np
import warp as wp

ROOT = Path('/home/chetan/contact_model_study')
sys.path.insert(0,str(ROOT))
import run_leap_hardware as runner
from ContactModelStudy.Hardware.Workflow import build_tasks,build_rollout
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI

source=ROOT/'results/planned/hardware/cube_reorient/hand-high_obj-high/M2/leap_experiment_20261112_221500_113029/controller.json'
saved=json.loads(source.read_text())
args=SimpleNamespace(**saved['args'])
args.hardware=False
args.device='cuda:0'
out=Path(sys.argv[1])
out.mkdir(parents=True,exist_ok=True)
report=dict(source_run=str(source.parent),simulation_only=True,physical_gpu=1,states=[])
sim=None
with wp.ScopedDevice(args.device),tempfile.TemporaryDirectory(prefix='leap_contact_audit_') as directory:
    task,evaluation,_=build_tasks(args,directory,hardware_config=saved['hardware_config'])
    episode=saved['episodes'][0]
    task.params['init_ctrl']=np.array(episode['home'])
    task.setTargetPosition(episode['target_pos'])
    task.setGoal(episode['goal'])
    task.setCostWeights(saved['weights'])
    sim=build_rollout(args,task,*runner.model_limits(evaluation.mjm))
    try:
        cfg=runner.reference.buildPlannerConfig(args)
        planner=MPPI(sim,task,cfg)
        model=sim.mjm
        tips=[model.site(name).id for name in ['if_tip','mf_tip','rf_tip','th_tip']]
        assert list(task.indices[5:9])==tips
        report.update(indices=task.indices.tolist(),site_names=[model.site(i).name for i in tips],
            weights=task.params['cost_weights'],temperature=args.temperature,noise_sigma=args.noise_sigma,
            n_samples=sim.N,horizon=sim.horizon,control_mode=args.control_mode,
            gravity=model.opt.gravity.tolist(),joint_stiffness=model.actuator_gainprm[:,0].tolist())
        captured=[]
        for _ in range(sim.horizon):
            captured.append(dict(q=wp.zeros_like(sim.DeviceState().qpos),
                                 sites=wp.zeros_like(sim.DeviceState().site_xpos),
                                 cost=wp.zeros(sim.N,dtype=wp.float32,device=args.device)))
        original_cost=task.calcCosts
        cursor=[0]
        def traced_cost(backend,terminal=False,out=None):
            value=original_cost(backend,terminal=terminal,out=out)
            item=captured[cursor[0]%sim.horizon]
            cursor[0]+=1
            wp.copy(item['q'],backend.DeviceState().qpos)
            wp.copy(item['sites'],backend.DeviceState().site_xpos)
            wp.copy(item['cost'],value)
            return value
        for index in [0,len(episode['steps'])//2,len(episode['steps'])-1]:
            row=episode['steps'][index]
            q,v,u=np.array(row['q']),np.array(row['q_dot']),np.array(row['u'])
            sim.activation.assign(u.astype(np.float32))
            sim.SetState(q,v)
            sim.SetControl(u)
            state=sim.DeviceState()
            gpu_points=state.site_xpos.numpy()[:,tips,:]
            current_q=state.qpos.numpy()
            expected=np.linalg.norm(gpu_points-current_q[:,None,16:19],axis=2).sum(axis=1)
            actual=task.calcCosts(sim).numpy()
            np.testing.assert_allclose(actual,expected,atol=2e-7)
            cpu=mujoco.MjData(model)
            cpu.qpos[:]=q
            mujoco.mj_kinematics(model,cpu)
            np.testing.assert_allclose(gpu_points[0],cpu.site_xpos[tips],atol=2e-6)
            before=dict(step=index,static_kernel_error=float(np.max(np.abs(actual-expected))),
                fingertip_world_m=cpu.site_xpos[tips].tolist(),object_origin_world_m=q[16:19].tolist(),
                tip_distances_m=np.linalg.norm(cpu.site_xpos[tips]-q[16:19],axis=1).tolist(),
                logged_max_delta_rad=float(np.max(np.abs(row['action_delta']))))
            planner.Reset()
            # Use the first episode's initial noise index to compare all cost
            # scales against exactly the same candidates, not fresh noise draws.
            planner._resample_count=1
            sim.activation.assign(u.astype(np.float32))
            task.calcCosts=traced_cost
            cursor[0]=0
            planner.Plan(q,v,u=u)
            task.calcCosts=original_cost
            costs=planner.costs_wp.numpy().astype(float)
            candidates=planner.V_wp.numpy().astype(float)
            weights=planner.w_wp.numpy().astype(float)
            expected_weights=np.exp(-(costs-costs.min())/args.temperature)
            expected_weights/=expected_weights.sum()
            np.testing.assert_allclose(weights,expected_weights,atol=2e-8,rtol=1e-5)
            expected_mean=np.einsum('n,nhj->hj',expected_weights,candidates)
            np.testing.assert_allclose(planner.last_mean,expected_mean,atol=1e-7)
            # These buffers were copied inside the ACTUAL rollout graph,
            # immediately after scoring each state (no second physics replay).
            independent=np.zeros(sim.N)
            trace=[]
            for t,item in enumerate(captured):
                positions=item['q'].numpy()
                points=item['sites'].numpy()[:,tips,:]
                term=np.linalg.norm(points-positions[:,None,16:19],axis=2).sum(axis=1)
                observed=item['cost'].numpy()
                if t<sim.horizon-1:
                    np.testing.assert_allclose(observed,term,atol=2e-7)
                    independent+=term
                else:
                    np.testing.assert_allclose(observed,0,atol=1e-9)
                trace.append(dict(step=t,contact_mean=float(term.mean()),contact_std=float(term.std()),
                    scored=bool(t<sim.horizon-1)))
            np.testing.assert_allclose(costs,independent,atol=3e-7)
            before.update(cost_min=float(costs.min()),cost_max=float(costs.max()),cost_std=float(costs.std()),
                effective_sample_size=float(1/np.sum(weights**2)),weight_min=float(weights.min()),
                weight_max=float(weights.max()),planned_max_delta_rad=float(np.max(np.abs(planner.last_mean[0]))),
                weighted_mean_error=float(np.max(np.abs(planner.last_mean-expected_mean))),
                accumulated_cost_error=float(np.max(np.abs(costs-independent))),trace=trace,scaling=[])
            for scale in [1,10,100,1000]:
                a=np.exp(-scale*(costs-costs.min())/args.temperature)
                a/=a.sum()
                mean=np.einsum('n,nhj->hj',a,candidates)
                before['scaling'].append(dict(contact_weight=scale,temperature=args.temperature,
                    effective_sample_size=float(1/np.sum(a*a)),max_delta_rad=float(np.max(np.abs(mean[0]))),
                    first_delta_rad=mean[0].tolist()))
            cpu.qpos[:]=q
            cpu.qvel[:]=0
            mujoco.mj_forward(model,cpu)
            before['hand_gravity_torques_Nm']=cpu.qfrc_bias[:16].tolist()
            before['gravity_only_servo_error_rad']=(cpu.qfrc_bias[:16]/model.actuator_gainprm[:,0]).tolist()
            before['planned_first_delta_rad']=planner.last_mean[0].tolist()
            report['states'].append(before)
        report['status']='passed'
    finally:
        sim.Close()
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
