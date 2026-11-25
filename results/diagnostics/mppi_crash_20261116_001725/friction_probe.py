"""Offline sensitivity check of the reproduced failure, not hardware sysid."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace
import numpy as np
import warp as wp
from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig,TaskRole
from ContactModelStudy.Hardware.Workflow import SceneTaskMixin,build_rollout
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI,MPPI_Config

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
d=json.loads((ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/random_tests/leap_experiment_20261116_001725_464670/controller.json').read_text());e=d['episodes'][0]
saved=np.load(OUT/'failure_25.npz')
class Task(SceneTaskMixin,CubeReorient):pass
results=[]
with wp.ScopedDevice('cuda:0'):
 for mu,tip in [(1e-4,.3),(1e-3,.3),(.01,.3),(.05,.3),(.1,.3),(.4,.5)]:
  root=ET.fromstring(d['scenes']['rollout']['scene_xml']);changed=[]
  for geom in root.findall('.//worldbody//geom'):
   friction=geom.get('friction')
   if friction and abs(float(friction.split()[0])-1e-4)<1e-8:
    values=friction.split();values[0]=str(mu);geom.set('friction',' '.join(values));changed.append(geom.get('name'))
   if geom.get('name') in ['if_tip','mf_tip','rf_tip','th_tip']:
    geom.set('friction',f'{tip} 0.005 0.0001')
  path=OUT/'probe.xml';path.write_text(ET.tostring(root,encoding='unicode'))
  args=SimpleNamespace(**d['args']);args.device='cuda:0'
  task=Task(str(path),LeapReorientConfig(role=TaskRole.ROLLOUT,timestep=args.timestep,
      hand_acc=args.hand_acc,obj_acc=args.obj_acc,cost_weights=d['weights']))
  task.setJointReference(e['home']);task.setTargetPosition(e['target_pos']);task.params['fallen_z']=e['fallen_z'];task.setGoal(e['goal'])
  limits=task.mjm.actuator_ctrlrange;rollout=build_rollout(args,task,limits[:,0],limits[:,1])
  cfg=dict(d['planner_config'],use_graph=False);planner=MPPI(rollout,task,MPPI_Config(**cfg))
  state=planner.SaveState();state.update(U=saved['U'],noise_seed=int(saved['noise_seed']),resample_count=int(saved['resample_count']),u_prev=saved['u'])
  planner.LoadState(state);rollout.activation.assign(saved['activation'].astype(np.float32))
  planner.Plan(saved['q'],saved['v'],u=saved['u'])
  costs=planner.costs_wp.numpy();q=rollout.DeviceState().qpos.numpy()
  results.append(dict(plastic_mu=mu,tip_mu=tip,changed_geoms=len(changed),ok=planner.last_plan_ok,
      finite_costs=int(np.isfinite(costs).sum()),nan_costs=int(np.isnan(costs).sum()),finite_world_states=int(np.isfinite(q).all(axis=1).sum())))
  (OUT/'friction_probe_results.json').write_text(json.dumps(results,indent=2)+'\n')
  print('PROBE_RESULT',json.dumps(results[-1]),flush=True);rollout.Close()
