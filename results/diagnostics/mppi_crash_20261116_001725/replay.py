"""Bounded offline replay of logged MPPI inputs; no LCM/camera/motor imports."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','3')
from pathlib import Path
import json
from types import SimpleNamespace
from datetime import datetime
import numpy as np
import warp as wp

from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig,TaskRole
from ContactModelStudy.Hardware.Workflow import SceneTaskMixin,build_rollout
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI,MPPI_Config

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
source=ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/random_tests/leap_experiment_20261116_001725_464670/controller.json'
record=json.loads(source.read_text());episode=record['episodes'][0]
path=OUT/'rollout_snapshot.xml';path.write_text(record['scenes']['rollout']['scene_xml'])
class Task(SceneTaskMixin,CubeReorient):pass
def stamp(s):return datetime.fromisoformat(s).timestamp()
summary=[]
with wp.ScopedDevice('cuda:0'):
 for iterations in [25,100]:
  args=SimpleNamespace(**record['args']);args.device='cuda:0';args.contact_params=dict(args.contact_params,iterations=iterations)
  task=Task(str(path),LeapReorientConfig(role=TaskRole.ROLLOUT,timestep=args.timestep,
      hand_acc=args.hand_acc,obj_acc=args.obj_acc,cost_weights=record['weights']))
  task.setJointReference(episode['home']);task.setTargetPosition(episode['target_pos']);task.params['fallen_z']=episode['fallen_z'];task.setGoal(episode['goal'])
  limits=task.mjm.actuator_ctrlrange
  rollout=build_rollout(args,task,limits[:,0],limits[:,1])
  planner=MPPI(rollout,task,MPPI_Config(**record['planner_config']))
  planner.Reset()
  tau=task.mjm.actuator_dynprm[:,0]
  steps=episode['steps'];previous_u=np.asarray(steps[0]['q'][:16]);activation=previous_u.copy();last_time=stamp(steps[0]['timestamp_utc'])
  failures=[];deviations=[]
  for index in range(len(steps)+1):
   approximate=index==len(steps)
   item=episode['failed_step'] if approximate else steps[index]
   q=np.asarray(steps[-1]['q'] if approximate else item['q'],float).copy()
   v=np.asarray(steps[-1]['q_dot'] if approximate else item['q_dot'],float).copy()
   if approximate and 'plan_base' in item:q[:16]=item['plan_base']
   now=stamp(item['timestamp_utc']);activation=previous_u+(activation-previous_u)*np.exp(-max(0,now-last_time)/tau)
   rollout.activation.assign(activation.astype(np.float32))
   output=planner.Plan(q,v,u=previous_u)
   if not planner.last_plan_ok:
    costs=planner.costs_wp.numpy();state=rollout.DeviceState()
    failure=dict(step=index,approximate_input=approximate,finite_costs=int(np.isfinite(costs).sum()),
        nan_costs=int(np.isnan(costs).sum()),infinite_costs=int(np.isinf(costs).sum()),
        finite_final_q_worlds=int(np.isfinite(state.qpos.numpy()).all(axis=1).sum()),
        finite_final_v_worlds=int(np.isfinite(state.qvel.numpy()).all(axis=1).sum()))
    failures.append(failure)
    saved=planner.last_plan_start
    np.savez(OUT/f'failure_{iterations}.npz',q=q,v=v,u=previous_u,activation=activation,
        U=saved['U'],noise_seed=saved['noise_seed'],resample_count=saved['resample_count'],costs=costs)
    break
   if not approximate:
    proposed=output[0] if isinstance(output,tuple) else output
    deviations.append(float(np.max(np.abs(proposed-np.asarray(item['planned_u'])))))
    published=stamp(item['action_published_at_utc'])
    activation=previous_u+(activation-previous_u)*np.exp(-max(0,published-now)/tau)
    previous_u=np.asarray(item['u']);last_time=published
  summary.append(dict(iterations=iterations,failures=failures,replayed_measured_steps=len(deviations),
    max_action_difference_from_hardware_log=max(deviations,default=None),
    median_action_difference_from_hardware_log=float(np.median(deviations)) if deviations else None,
    caveat='Activation initial value/timing reconstructed; failed-step object state/velocity were not saved and use last measured values. This is not an exact replay.'))
  (OUT/'replay_results.json').write_text(json.dumps(summary,indent=2)+'\n')
  print('REPLAY_RESULT',json.dumps(summary[-1]),flush=True)
  rollout.Close()
