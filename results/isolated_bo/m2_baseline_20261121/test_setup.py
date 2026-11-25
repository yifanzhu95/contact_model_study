"""CPU-only preparation validation. No real episodes, optimizer trials or GPU."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'code'));sys.path.insert(0,str(OUT/'code/experiments'))
import run_bayes_opt as bo
import runtime
runtime.install(bo)


def spec():
    rows=bo.batches.readRows(OUT/'search.csv')
    return bo.parseCell(rows[0],OUT,OUT/'check_only')


def test_search_seed_and_fixed_settings_match_baseline():
    s=spec();base=runtime.BASELINE
    assert s.models==['M2'] and s.n_calls==500 and s.episodes_per_trial==12
    assert len(s.dims)==11 and len(s.seeds)==4 and s.n_random==116
    assert s.defaultParams()=={**{k:base['weights'][k] for k in s.names if k.startswith('w_')},
        'action_rate_weight':10.,'joint_limit_weight':100.,'plan_blend_alpha':.75}
    assert all(s.base_weights[k]==0 for k in ['w_velo','w_joint_velo','w_fallen_term'])
    assert s.base_weights['w_contact']==80. and 'w_contact' not in s.names
    assert s.args.n_samples==512 and s.args.noise_sigma==.05 and s.args.temperature==.1
    assert s.args.time_horizon==.256 and s.args.ctrl_time_step==.064 and s.args.warm_start=='shift'
    for d in s.dims:
        assert d.skopt().prior==('uniform' if d.name in runtime.EXTRA else 'log-uniform')


def test_frozen_calibrated_models_reference_mass_friction_and_bounds():
    import mujoco
    from ContactModelStudy.Drivers import run_episodes as drv
    task,evaluation=drv.buildTasks(spec().args)
    for t in [task,evaluation]:
        assert Path(t.getModelPath()).resolve().is_relative_to(OUT/'code')
        model=mujoco.MjModel.from_xml_path(t.getModelPath())
        assert model.na==16 and model.nq==23
        np.testing.assert_allclose(t.jointReference(),runtime.BASELINE['joint_reference'])
        np.testing.assert_allclose(t.params['target_pos'],runtime.BASELINE['target_pos'])
        np.testing.assert_allclose(model.jnt_range[:16,0],runtime.BASELINE['joint_limits']['lower'])
        np.testing.assert_allclose(model.jnt_range[:16,1],runtime.BASELINE['joint_limits']['upper'])
        assert model.body('obj').mass[0]==pytest.approx(.07)
        np.testing.assert_allclose(model.geom('obj').friction,[.3,.005,.0001])
        for tip in ['if_tip','mf_tip','rf_tip','th_tip']:
            np.testing.assert_allclose(model.geom(tip).friction,[.5,.005,.0001])
    assert len(runtime.INITIAL)==12
    from collections import Counter
    assert set(Counter(s['rotation'] for s in runtime.INITIAL).values())=={2}


def test_final_target_clips_delta_then_absolute_without_modifying_inputs():
    base=np.full(16,.5);delta=np.full(16,2.);old=base.copy()
    final,stage1=runtime.final_target(base,delta)
    np.testing.assert_allclose(base,old);np.testing.assert_allclose(stage1,.7)
    np.testing.assert_allclose(final,np.clip(stage1,runtime.BASELINE['joint_limits']['lower'],runtime.BASELINE['joint_limits']['upper']))


def test_dynamic_penalties_read_new_values_without_rebuilding_kernels():
    import warp as wp
    from ContactModelStudy.SamplingBasedPlanners.SamplingBasedPlannerBase import _action_rate_cost_kernel,_joint_limit_cost_kernel
    device='cpu';seq=wp.zeros((2,1,2),dtype=float,device=device)
    target=np.array([[.4,.8],[.7,.1]],dtype=np.float32)
    target_wp=wp.array(target,dtype=float,device=device);previous=wp.zeros((2,2),dtype=float,device=device)
    applied=wp.zeros(2,dtype=float,device=device);cost=wp.zeros(2,dtype=float,device=device)
    bounds=np.array([[0,.5],[.2,.6]],dtype=np.float32);bounds_wp=wp.array(bounds,dtype=float,device=device)
    weights=wp.array(np.array([2,3],dtype=np.float32),dtype=float,device=device)
    def launch():
        wp.launch(_action_rate_cost_kernel,dim=2,inputs=[seq,target_wp,applied,0,1,weights],outputs=[previous,cost],device=device)
        wp.launch(_joint_limit_cost_kernel,dim=2,inputs=[seq,target_wp,bounds_wp,0,1,weights],outputs=[cost],device=device)
    launch()
    violation=np.maximum(bounds[:,0]-target,0)+np.maximum(target-bounds[:,1],0)
    np.testing.assert_allclose(cost.numpy(),2*np.sum(target**2,axis=1)+3*np.sum(violation**2,axis=1),rtol=1e-6)
    weights.assign(np.zeros(2,dtype=np.float32));cost.zero_();launch()
    np.testing.assert_array_equal(cost.numpy(),0.)


def test_virtual_application_uses_updated_joints_with_fake_plant(monkeypatch):
    import mujoco
    from ContactModelStudy.Simulators.Mujoco import Mujoco
    events=[];state={}
    monkeypatch.setattr(mujoco,'mj_resetData',lambda *a:None)
    monkeypatch.setattr(mujoco,'mj_forward',lambda *a:None)
    monkeypatch.setattr(Mujoco,'applyContactParams',lambda *a:None)
    class Sim:
        timestep=.004
        mjm=SimpleNamespace(opt=SimpleNamespace(gravity=np.zeros(3)))
        mjd=SimpleNamespace(act=np.zeros(16))
        def SetState(self,q,v):state.update(q=q.copy(),v=v.copy())
        def GetState(self):return state['q'].copy(),state['v'].copy()
        def SetControl(self,u):events.append(('write',u.copy()))
        def Step(self,n):
            assert n==16;events.append(('advance',n));state['q'][0]+=.1
    class Planner:
        config=SimpleNamespace(plan_blend_alpha=1.)
        last_plan_seconds=.01
        def Reset(self,*a):pass
        def Plan(self,q,v,u=None):
            events.append(('plan',q.copy()));self.last_mean=np.full((4,16),.1);return q[:16]+.1
    task=SimpleNamespace(setGoal=lambda *a:None,isFailure=lambda *a:False,isSuccess=lambda *a:False)
    recorder=SimpleNamespace(recordGoal=lambda *a,**k:None,recordStateAndAction=lambda *a,**k:None,
        episodeFinished=lambda *a,**k:state.update(summary=k),GenerateSummary=lambda *a:state['summary'])
    result=runtime.run_episode(0,SimpleNamespace(steps=1),task,task,Sim(),Planner(),recorder)
    assert [e[0] for e in events]==['write','plan','advance','write']
    initial=np.array(runtime.INITIAL[0]['q']);base=initial[:16].copy();base[0]+=.1
    expected,_=runtime.final_target(base,np.full(16,.1))
    np.testing.assert_allclose(events[-1][1],expected)
    assert result['simulation_s']==pytest.approx(.064)
    assert result['final_target_clipping']['applied_actions']==1


def test_no_trial_has_started():
    state=json.loads((OUT/'status.json').read_text())
    assert state['status']=='prepared_not_started' and state['trials_started']==0
    assert not list((OUT/'search_results').glob('**/trial.json'))
