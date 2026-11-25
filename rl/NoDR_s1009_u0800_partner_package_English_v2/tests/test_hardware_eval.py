"""Hardware-free frame/protocol/timing and complete fake-LCM episode tests."""
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

import launch_rl_hardware as launcher
import run_rl_hardware as runner
from hardware_eval import core
from ContactModelStudy.Hardware.LCM import (RobotDeltaAction, RobotTargetState, RobotState,
    FeedbackBuffer, FeedbackTimingError)
from leap_hand_lcm_bridge import LeapHandLCMBridge

PACKAGE = core.PACKAGE
INTERFACE = json.loads((PACKAGE/'interface.json').read_text())


def test_registered_world_maps_palm_and_goal_consistently():
    config = json.loads((PACKAGE/'configs/leap_hardware_rl.json').read_text())
    T = core.world_registration(INTERFACE, config, runner.REPO/'scenes/leap/hardware/calibrated_hand.xml', 'palm_registered')
    current = np.eye(4); current[:3,:3] = core.quat_matrix(config['palm_quat_wxyz']); current[:3,3] = [0,0,.01]
    expected = INTERFACE['static_training_scene_root_bodies'][0]
    np.testing.assert_allclose((T @ current)[:3,:3], core.quat_matrix(expected['quaternion_wxyz']), atol=1e-7)
    for episode, sign in [(0,-1),(1,1),(2,-1)]:
        goal = core.goal_for_episode(INTERFACE,'alternate',episode,T)
        assert goal['sign'] == sign
        np.testing.assert_allclose(T[:3,:3] @ goal['world_position']+T[:3,3], goal['training_position'])
        np.testing.assert_allclose(T[:3,:3] @ core.quat_matrix(goal['world_quaternion']),core.quat_matrix(goal['training_quaternion']), atol=1e-7)


def test_real_pose_packets_body_angular_rate_is_rotated_into_training_world():
    model = core.feedback_model(INTERFACE)
    buffer = FeedbackBuffer(model)
    from ContactModelStudy.Hardware.Pose import Pose
    R = Rotation.from_euler('z',90,degrees=True).as_matrix()
    omega_world = np.array([.2,0,0]);dt=.01; stamp=int(time.time()*1e6)
    for k, rot in enumerate([R,Rotation.from_rotvec(omega_world*dt).as_matrix()@R]):
        p = Pose(); p.timestamp = stamp+k*10000; T = np.eye(4);T[:3,:3]=rot;T[:3,3]=[.02,.035,.085]
        p.pose_matrix=T.ravel().tolist();buffer.on_pose('POSE',p.encode())
    buffer.on_robot('LEAP_STATE',RobotState(stamp+10000,np.zeros(16),np.zeros(16)).encode())
    q,v,_=buffer.snapshot()
    transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('y',12,degrees=True).as_matrix()
    state=core.policy_state(q,v,np.zeros(16),core.goal_for_episode(INTERFACE,'1',0,transform),0,transform)
    np.testing.assert_allclose(state['object_angular_velocity_world_rad_s'],transform[:3,:3]@omega_world,atol=2e-5)


def test_delta_reaches_bridge_once_and_uses_previous_target_despite_joint_drift():
    class Hand:
        def __init__(self):self.reads=0;self.writes=[]
        def connect(self):pass
        def enable(self,q):pass
        def read(self):
            self.reads+=1
            return np.full(16,.5 if self.reads==1 else .35),np.zeros(16),np.zeros(16)
        def write(self,q):
            self.writes.append(q.copy())
            if len(self.writes)==2: bridge.stop_requested=True
        def health(self,motor):pass
    class Connection:
        def __init__(self):self.publications=[];self.sent=False
        def subscribe(self,*a):return SimpleNamespace(set_queue_capacity=lambda *_:None)
        def publish(self,channel,data):self.publications.append((channel,data))
        def handle_timeout(self,*a):
            if not self.sent:
                self.sent=True;bridge.on_action('DELTA',RobotDeltaAction(int(time.time()*1e6),np.full(16,.02)).encode());return 1
            return 0
    hand=Hand();connection=Connection()
    model=SimpleNamespace(jnt_range=np.tile([-2,2],(16,1)),actuator_ctrlrange=np.tile([-2,2],(16,1)))
    bridge=LeapHandLCMBridge(hand,connection,model,dict(delta_reference='previous_target',ctrl_range=dict(lower=.2,upper=.2),max_current_raw=550))
    bridge.run()
    assert len(hand.writes)==2
    np.testing.assert_allclose(hand.writes,[np.full(16,.52)]*2,atol=1e-7)
    targets=[RobotTargetState.decode(p).action for channel,p in connection.publications if channel.endswith('_TARGET')]
    np.testing.assert_allclose(targets[-1],.52,atol=1e-7)


def test_clipping_matches_bridge_and_absolute_policy_limits():
    prev=np.full(16,.4); low=np.full(16,-1.);high=np.full(16,1.)
    out=core.command_target(np.full(16,.464),prev,low,high,np.full(16,.02),np.full(16,.02))
    np.testing.assert_allclose(out,.42)
    with pytest.raises(ValueError):core.command_target(np.full(16,np.nan),prev,low,high,prev,prev)
    with pytest.raises(RuntimeError):core.command_target(prev,np.full(16,2),low,high,prev,prev)


def test_clock_no_catchup_burst_after_overrun():
    now=[0.]
    clock=core.PolicyClock(monotonic=lambda:now[0],sleep=lambda dt:now.__setitem__(0,now[0]+dt))
    now[0]=.003;waiting,overrun=clock.finish_period()
    assert now[0]==pytest.approx(.064) and not overrun
    now[0]=.25; waiting,overrun=clock.finish_period()
    assert now[0]==pytest.approx(.314) and overrun and waiting==pytest.approx(.064)


def test_target_ack_requires_report_after_publication_and_matching_value():
    now=[0.]; states=iter([(np.zeros(16),dict(target_timestamp=9)),
        (np.ones(16),dict(target_timestamp=10)),(np.ones(16),dict(target_timestamp=11))])
    feedback=SimpleNamespace(buffer=SimpleNamespace(snapshot_target=lambda _:next(states)))
    actual,metadata=core.wait_for_target(feedback,np.ones(16),11,.1,.15,
        monotonic=lambda:now[0],sleep=lambda dt:now.__setitem__(0,now[0]+dt))
    assert metadata['target_timestamp']==11 and now[0]==pytest.approx(.002)


@pytest.fixture
def fake_run(monkeypatch,tmp_path):
    events=[];instances=[]
    config=json.loads((PACKAGE/'configs/leap_hardware_rl.json').read_text())
    class Buffer:
        def __init__(self):
            self.target=np.array(config['robot_qref']);self.q=np.r_[np.full(16,.2),[.03,.04,.09],[1.,0,0,0]]
            self.v=np.zeros(22);self.success=False
        def snapshot(self,**kwargs):
            return self.q.copy(),self.v.copy(),dict(robot_timestamp=int(time.time()*1e6),pose_timestamp=int(time.time()*1e6),robot_age=0.,pose_age=0.)
        def snapshot_robot(self,**kwargs):return self.q[:16].copy(),self.v[:16].copy()
        def snapshot_target(self,*a):return self.target.copy(),dict(target_timestamp=int(time.time()*1e6)+10,target_age=0.)
    class Feedback:
        def __init__(self,*a,**kw):self.buffer=Buffer();self.published=[];instances.append(self)
        def wait(self,*a,**kw):return self.buffer.snapshot()
        def publish_delta(self,d):
            self.published.append(d.copy());self.buffer.target+=d;events.append('delta');return int(time.time()*1e6)
        def publish_stop(self):events.append('stop')
        def publish_position(self,q):self.buffer.target=q.copy()
        def close(self):events.append('close')
    class Episodes:
        def __init__(self,feedback,config,*a):self.feedback=feedback;self.reference=np.array(config['robot_qref']);self.last_target=self.reference.copy()
        def before_episode(self,*a):events.append('prepare');events.append('enter')
        def wait_for_tracking(self,*a,**kw):return self.feedback.buffer.snapshot()
        def after_success(self,*a):events.append('success_hold_enter')
        def after_unsuccessful(self,*a):events.append('failure_hold_enter')
        def return_to_zero(self,*a):events.append('zero');return np.zeros(16)
        def shutdown_after_feedback_error(self,*a):events.extend(['feedback_hold_enter','zero','stop'])
        def shutdown_on_interrupt(self,*a):events.extend(['interrupt','zero','stop'])
        def status(self,*a):pass
    class Policy:
        interface=INTERFACE
        def __init__(self):self.calls=[];self.fail=None
        def infer_state(self,state):
            self.calls.append(copy.deepcopy(state))
            if self.fail and len(self.calls)==2:raise self.fail
            u=torch.tensor(state['previous_target_rad'],dtype=torch.float32).reshape(1,16)+.01
            return dict(observation=torch.zeros(1,73),raw_action=torch.zeros(1,16),joint_target_rad=u)
    policy=Policy()
    monkeypatch.setattr(runner,'load_policy',lambda _: (policy,dict(device='cpu',backend='fake',actor_sha256='test',interface_sha256='test')))
    monkeypatch.setattr(runner,'LCMFeedback',Feedback);monkeypatch.setattr(runner,'RLEpisodes',Episodes)
    monkeypatch.setattr(runner,'PolicyClock',lambda:SimpleNamespace(finish_period=lambda:(0.,False)))
    monkeypatch.setattr(mujoco,'mj_step',lambda *_:pytest.fail('Physics must never be stepped'))
    return SimpleNamespace(policy=policy,instances=instances,events=events,output=tmp_path/'controller.json')


def test_complete_fake_episode_accumulates_previous_accepted_targets(fake_run):
    f=fake_run
    assert runner.main(['--steps','3','--results',str(f.output)])==0
    assert len(f.policy.calls)==3 and f.events[:2]==['prepare','enter']
    states=f.policy.calls
    np.testing.assert_allclose(np.array(states[1]['previous_target_rad'])-states[0]['previous_target_rad'],.01,atol=1e-7)
    assert states[1]['joint_position_rad']==[.2]*16
    assert f.events[-4:]==['failure_hold_enter','zero','stop','close']
    d=json.loads(f.output.read_text());assert d['summary']['completed_episodes']==1
    assert d['episodes'][0]['n_steps']==3 and d['summary']['inference_time']['count']==3


@pytest.mark.parametrize('failure,code,expected',[(FeedbackTimingError('late pose'),1,'feedback_hold_enter'),
    (KeyboardInterrupt(),0,'interrupt'),(SystemExit('supervisor'),1,'stop')])
def test_faults_preserve_aborted_episode_and_cleanup(fake_run,failure,code,expected):
    f=fake_run;f.policy.fail=failure
    assert runner.main(['--steps','3','--results',str(f.output)])==code
    assert expected in f.events and f.events[-1]=='close'
    d=json.loads(f.output.read_text());assert d['summary']['aborted_episodes']==1
    assert d['summary']['completed_episodes']==0 and d['episodes'][0]['n_steps']==1
    if isinstance(failure,SystemExit):assert 'zero' not in f.events


def test_observe_only_never_publishes_any_hand_command(fake_run):
    f=fake_run
    assert runner.main(['--steps','2','--observe-only','--results',str(f.output)])==0
    assert f.events==['close']
    assert not f.instances[0].published


def test_success_freezes_before_an_extra_action_and_waits_before_zero(fake_run,monkeypatch):
    f=fake_run
    config=json.loads((PACKAGE/'configs/leap_hardware_rl.json').read_text())
    T=core.world_registration(INTERFACE,config,runner.REPO/'scenes/leap/hardware/calibrated_hand.xml','palm_registered')
    goal=core.goal_for_episode(INTERFACE,'alternate',0,T)
    original=runner.fresh_state
    def at_goal(feedback,limits):
        q,v,m=original(feedback,limits);q[16:19]=goal['world_position'];q[19:23]=goal['world_quaternion'];return q,v,m
    monkeypatch.setattr(runner,'fresh_state',at_goal)
    assert runner.main(['--steps','3','--results',str(f.output)])==0
    assert not f.policy.calls and not f.instances[0].published
    assert f.events.index('success_hold_enter') < f.events.index('zero')
    d=json.loads(f.output.read_text());assert d['summary']['successful_episodes']==1


def test_success_hold_retains_commanded_effort_instead_of_latching_measured_q():
    writes=[]
    feedback=SimpleNamespace(buffer=SimpleNamespace(snapshot_robot=lambda **_: (np.full(16,.2),np.zeros(16))),
        publish_position=lambda q:writes.append(q.copy()),publish_hold=lambda:pytest.fail('Must not latch measured joints'))
    config=json.loads((PACKAGE/'configs/leap_hardware_rl.json').read_text())
    episodes=runner.RLEpisodes(feedback,config,.15,logging_enabled=False)
    episodes.last_target=np.full(16,.4)
    replies=iter([False,True]);episodes.enter_pressed=lambda _:next(replies)
    episodes.after_success(0,True)
    assert len(writes)==2
    np.testing.assert_allclose(writes,[np.full(16,.4)]*2)


def test_shared_configuration_logs_accept_rl_results_and_append(fake_run):
    from ContactModelStudy.Hardware.ExperimentLogs import ConfigurationLogs
    f=fake_run;aggregate=ConfigurationLogs(f.output.parent/'aggregate')
    assert runner.main(['--steps','1','--results',str(f.output)])==0
    first=aggregate.record(f.output,{})
    assert first['runs']==1 and aggregate.record(f.output,{})['runs']==1
    assert runner.main(['--steps','1','--results',str(f.output)])==0
    second=aggregate.record(f.output,{})
    assert second['runs']==2 and first['configuration_id']==second['configuration_id']
    overwritten=aggregate.record(f.output,{},overwrite=True)
    assert overwritten['runs']==1 and overwritten['mode']=='overwrite'


def test_launcher_uses_rl_cpu_controller_and_state_only_observer_bridge(tmp_path):
    cfg=json.loads((runner.REPO/'configs/leap_experiment.json').read_text())
    cfg.update(controller_args=[],fingertip_paths=False,observe_only=True,hardware_config=str(PACKAGE/'configs/leap_hardware_rl.json'))
    cfg['init_mask_path']=str(tmp_path/'mask.png')
    cfg['interpreters']['control']=str(PACKAGE/'runtime/venv/bin/python')
    commands=launcher.build_rl_commands(cfg,tmp_path,['--n-episodes','2'])
    assert commands['controller'][0]==str(PACKAGE/'runtime/venv/bin/python')
    assert str(PACKAGE/'run_rl_hardware.py') in commands['controller']
    assert not any('run_leap_hardware.py' in arg for arg in commands['controller'])
    assert '--state-only' in commands['bridge'] and '--observe-only' in commands['controller']
    with pytest.raises(ValueError):launcher.build_rl_commands(cfg,tmp_path,['--action-channel','WRONG'])


def test_borrowing_tracking_never_rewrites_original_registry(tmp_path,monkeypatch):
    source=tmp_path/'shared';source.mkdir();(source/'session.lock').write_text('')
    original=json.dumps(dict(version=1,processes=dict(tracker=dict(pid=1),viewer=dict(pid=2))))
    (source/'session.json').write_text(original)
    monkeypatch.setattr(launcher,'process_alive',lambda p:bool(p))
    cfg=dict(tracking_dir=str(tmp_path/'private'))
    lock=launcher.borrow_live_tracking(cfg,source);assert lock is not None;lock.close()
    assert (source/'session.json').read_text()==original
    second=launcher.borrow_live_tracking(cfg,source);assert second is not None;second.close()


def test_outputs_cannot_escape_rl_package():
    with pytest.raises(ValueError):runner.parse_args(['--results','/tmp/rl-outside.json'])
    with pytest.raises(ValueError):launcher.inside('/tmp/rl-outside')
