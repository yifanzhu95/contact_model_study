"""Offline calibrated MPPI replay. No LCM instance, serial client, camera or viewer."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser()
parser.add_argument('--model',choices=['M2','M6'],required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--seconds',type=float,default=40.)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
def phase(name):
    path=args.out/'phase.json';pending=path.with_suffix('.tmp')
    pending.write_text(json.dumps(dict(phase=name,at=time.time())));pending.replace(path)
    print('PHASE',name,flush=True)
phase('imports')
import numpy as np
import warp as wp
import run_leap_hardware as runner
from ContactModelStudy.Hardware.Hand import HardwareHand

def forbidden(*a,**kw):
    raise AssertionError('Offline benchmark must not connect to hardware or LCM')
runner.LCMFeedback=forbidden
runner.HardwareEpisodes=forbidden
HardwareHand.__init__=forbidden
source=ROOT/'results/pilot/hardware/cube_reorient/hand-high_obj-high/M6/leap_experiment_20261113_230803_006211/controller.json'
saved=json.loads(source.read_text())
# Identical states for both backends, spanning the episode that slowed down.
recorded=saved['episodes'][2]['steps']
states=recorded[::4]
cli=['--hardware','--no-logging','--device','cuda:0','--task','cube_reorient','--rollout-model',args.model,
     '--hand-acc','high','--obj-acc','high','--n-samples','512','--n-iterations','1',
     '--ctrl-time-step','0.064','--time-horizon','.32' if args.model=='M2' else '.384',
     '--noise-sigma','.05','--temperature','.1','--action-rate-weight','0',
     '--no-object-joint-reference']
for key,value in saved['weights'].items():cli+=['--cost-weight',f'{key}={value}']
options=runner.parseArgs(cli)
config=saved['hardware_config']
wp.init()
assert len(wp.get_cuda_devices())==1
rollout=None
start_total=time.monotonic()
try:
 with wp.ScopedDevice('cuda:0'),tempfile.TemporaryDirectory(prefix='leap_offline_gpu_') as temporary:
    phase('build')
    task,evaluation,scenes=runner.build_tasks(options,temporary,hardware_config=config)
    for obj in (task,evaluation):
        obj.params['init_ctrl']=np.asarray(config['robot_qref'])
        obj.params['fallen_z']=saved['episodes'][2]['fallen_z']
        obj.setJointReference(saved['joint_reference'])
        obj.setTargetPosition(saved['episodes'][2]['target_pos'])
        obj.setGoal(saved['episodes'][2]['goal'])
    lower,upper=runner.model_limits(evaluation.mjm)
    rollout=runner.build_rollout(options,task,lower,upper)
    planner=runner.MPPI(rollout,task,runner.reference.buildPlannerConfig(options))
    phase('warmup')
    q=np.asarray(states[0]['q']);v=np.asarray(states[0]['q_dot']);u=q[:16].copy()
    rollout.activation.assign(u.astype(np.float32));planner.Reset()
    first=time.monotonic();planner.Plan(q,v,u=u);wp.synchronize();warmup=time.monotonic()-first
    assert planner.last_plan_ok,'Warmup solve failed'
    planner.Reset()
    tau=evaluation.mjm.actuator_dynprm[:,0]
    results=[];failures=0;last_update=time.monotonic();activation=u.copy()
    phase('measurement')
    start=time.monotonic();last_progress=start
    i=0
    while time.monotonic()-start<args.seconds:
        if i%len(states)==0:
            planner.Reset();u=np.asarray(states[0]['q'][:16]);activation=u.copy()
        row=states[i%len(states)]
        q=np.asarray(row['q']);v=np.asarray(row['q_dot'])
        # Use logged predecessor target/activation to keep replay inputs stable
        # across backends. Each cycle sweeps the same saved states.
        original_index=(i%len(states))*4
        previous=recorded[max(0,original_index-1)]
        u=np.asarray(previous['u'])
        elapsed=previous.get('wall_time_s',.064)
        activation=u+(activation-u)*np.exp(-elapsed/tau)
        rollout.activation.assign(activation.astype(np.float32))
        begin=time.perf_counter();stamp=time.time()
        proposed=planner.Plan(q,v,u=u)
        # Plan returns a host action and synchronizes already; include the
        # explicit final sync so no GPU work is charged to a subsequent call.
        wp.synchronize();duration=time.perf_counter()-begin
        if not planner.last_plan_ok:failures+=1
        results.append(dict(call=i,source_step=row['step'],source_elapsed_s=row['elapsed_s'],
                            timestamp=stamp,planning_s=duration,ok=bool(planner.last_plan_ok)))
        i+=1
        if time.monotonic()-last_progress>10:
            print('PROGRESS',args.model,i,'calls',round(time.monotonic()-start,1),'s',flush=True)
            last_progress=time.monotonic()
    measurement_s=time.monotonic()-start
    phase('finished')
    times=np.array([r['planning_s'] for r in results])
    report=dict(model=args.model,scope='Offline replay only; no hardware/LCM/camera instances',
      device=str(wp.get_device('cuda:0')),physical_gpu_uuid=str(wp.get_device('cuda:0').uuid),
      source_controller=str(source),state_selection='episode 3, every fourth logged state, repeated with mean reset at wrap',
      source_state_count=len(states),samples=options.n_samples,horizon_steps=rollout.config.resolved_horizon,
      control_period_s=rollout.config.control_timestep,rollout_config=asdict(rollout.config),
      planner_config=asdict(planner.config),weights=task.params['cost_weights'],
      warmup_s=warmup,measurement_s=measurement_s,calls=len(results),failed_plans=failures,
      planning_ms=dict(mean=float(times.mean()*1000),median=float(np.median(times)*1000),
       p95=float(np.percentile(times,95)*1000),max=float(times.max()*1000)),
      plan_rate_hz=len(results)/measurement_s,records=results)
    (args.out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('RESULT',json.dumps({k:v for k,v in report.items() if k not in ('records','weights','planner_config','rollout_config')}),flush=True)
finally:
 if rollout is not None:rollout.Close()
