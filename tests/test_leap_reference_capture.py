"""Independent joint-cost references and selective torque capture, without motors."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import mujoco
import numpy as np
import pytest
import warp as wp

import capture_leap_joint_reference as capture
from ContactModelStudy.Hardware import JointReferences as references
from ContactModelStudy.Hardware.Hand import HardwareHand
from ContactModelStudy.Hardware.Scene import JOINTS, ROOT
from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Tasks.BallReorient import BallReorient
from ContactModelStudy.Tasks.DuckReorient import DuckReorient


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_path(str(ROOT/'scenes/leap/hardware/calibrated_hand.xml'))


@pytest.fixture
def hand():
    events = []
    q = np.full(16,.5)
    return SimpleNamespace(
        events=events,read=lambda:(q.copy(),np.zeros(16),np.zeros(16)),
        connect=lambda **kw:events.append(('connect',kw)),
        enable_motors=lambda pose,ids:events.append(('enable',list(ids),pose.copy())),
        disable_motors=lambda ids:events.append(('disable',list(ids))),
        write_motors=lambda pose,ids:events.append(('write',list(ids),pose.copy())),
        health=lambda *a:None,close=lambda:events.append(('close',)))


def test_fingers_can_be_captured_in_any_order_and_only_complete_stable_pose_is_saved(model,hand,tmp_path):
    path = tmp_path/'cube.json'
    session = capture.FingerCapture(hand,model,'cube',path,settle=.5)
    q,velocity,_ = hand.read()
    for finger in ['thumb','index','ring','middle']:
        session.select(finger)
        assert session.capture(q,velocity)
        assert hand.events[-1][1] == capture.FINGERS[finger]
        assert not path.exists()
    session.hold()
    assert hand.events[-1][0:2] == ('write',list(range(16)))
    measured = q-.01
    for t in [0.,.25,.5]:
        session.observe(measured,velocity,t)
    record = json.loads(path.read_text())
    np.testing.assert_allclose(record['joint_reference'],measured)
    np.testing.assert_allclose(record['hold_targets'],q)
    assert record['capture_order'] == ['thumb','index','ring','middle']
    assert record['joint_order'] == list(JOINTS)
    assert record['sample_count'] == 3
    assert session.saved


def test_reposition_after_saving_releases_only_selected_finger_and_resaves_with_backup(model,hand,tmp_path):
    path = tmp_path/'cube.json'
    s = capture.FingerCapture(hand,model,'cube',path,settle=.1)
    q,v,_ = hand.read()
    for finger in capture.FINGERS:
        s.select(finger);s.capture(q,v)
    s.observe(q,v,0.);s.observe(q,v,.11)
    original = path.read_bytes()
    s.select('middle');s.release()
    assert set(s.locked)=={'index','ring','thumb'}
    assert [e for e in hand.events if e[0]=='disable']==[('disable',[4,5,6,7])]
    adjusted = q.copy();adjusted[4:8]=.65
    s.capture(adjusted,v)
    s.observe(adjusted,v,1.)
    assert path.read_bytes()==original  # unfinished measurement retains old file
    s.observe(adjusted,v,1.11)
    np.testing.assert_allclose(json.loads(path.read_text())['joint_reference'],adjusted)
    assert next((tmp_path/'history').iterdir()).read_bytes()==original


def test_capture_does_not_enable_a_moving_finger_or_an_out_of_range_target(model,hand,tmp_path):
    s = capture.FingerCapture(hand,model,'cube',tmp_path/'cube.json')
    q,v,_ = hand.read()
    assert not s.capture(q,v)
    s.select('middle')
    moving = v.copy();moving[5]=1.
    assert not s.capture(q,moving)
    invalid = q.copy();invalid[4]=10.
    assert not s.capture(invalid,v)
    # An unrelated FREE finger may be moving or outside planning ranges.
    q[0]=10.;v[0]=2.
    assert s.capture(q,v)
    assert [e[1] for e in hand.events if e[0]=='enable'] == [[4,5,6,7]]


def test_selecting_locked_finger_does_not_release_it_without_backspace(model,hand,tmp_path):
    s = capture.FingerCapture(hand,model,'cube',tmp_path/'cube.json')
    q,v,_ = hand.read()
    s.select('index');s.capture(q,v)
    s.select('index')
    assert not s.capture(q,v)
    assert not any(e[0]=='disable' for e in hand.events)
    s.release()
    assert hand.events[-1] == ('disable',[0,1,2,3])
    assert not s.locked
    s.observe(q,v,100.)
    assert not (tmp_path/'cube.json').exists()


def test_reference_files_are_object_specific_validated_and_backed_up(tmp_path,monkeypatch):
    monkeypatch.setattr(references,'DEFAULT_REFERENCE_DIR',tmp_path)
    assert references.load_reference('cube') is None
    references.save_reference('cube',np.full(16,.3),{})
    previous = (tmp_path/'cube.json').read_bytes()
    references.save_reference('cube',np.full(16,.4),{})
    np.testing.assert_allclose(references.load_reference('cube')['joint_reference'],.4)
    assert next((tmp_path/'history').iterdir()).read_bytes() == previous
    assert references.load_reference('ball') is None
    with pytest.raises(ValueError,match='schema/object'):
        references.load_reference('ball',tmp_path/'cube.json')
    record = json.loads((tmp_path/'cube.json').read_text())
    record['joint_order'] = record['joint_order'][::-1]
    (tmp_path/'bad.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='joint order'):
        references.load_reference('cube',tmp_path/'bad.json')


@pytest.mark.parametrize('cls',[CubeReorient,BallReorient,DuckReorient])
def test_cost_reference_changes_gpu_home_without_changing_initial_state_or_command(cls):
    task = cls()
    initial = task.getInitialState()
    before = task.goal.copy()
    goal_buffer = task._gpu_arrays('cpu')['goal']
    reference = np.linspace(.1,.8,16)
    task.setJointReference(reference)
    np.testing.assert_allclose(goal_buffer.numpy()[7:23],reference)
    np.testing.assert_array_equal(task.goal[:7],before[:7])
    for a,b in zip(initial,task.getInitialState()):np.testing.assert_array_equal(a,b)
    task.setJointReference(None)
    np.testing.assert_array_equal(task.goal,before)
    with pytest.raises(ValueError,match='16 finite'):
        task.setJointReference(np.full(16,np.nan))


def test_joint_cost_uses_new_reference_instead_of_init_ctrl(tmp_path):
    from ContactModelStudy.Hardware.Workflow import build_tasks,build_rollout
    from ContactModelStudy.Hardware.Hand import control_limits
    import run_leap_hardware as runner
    args = runner.parseArgs(['--device','cpu','--n-samples','1','--horizon','1',
                             '--no-object-joint-reference'])
    task,_,_ = build_tasks(args,tmp_path)
    task.setCostWeights({k:float(k=='w_joint') for k in task.params['cost_weights']})
    task.setJointReference(np.full(16,.5))
    q,v,_ = task.getInitialState();q[:16]=.5
    sim = build_rollout(args,task,*control_limits(task.mjm))
    sim.SetState(q,v)
    assert task.calcCosts(sim).numpy()[0] == pytest.approx(0,abs=1e-6)
    q[2]+=.2;sim.SetState(q,v)
    assert task.calcCosts(sim).numpy()[0] == pytest.approx(.04,abs=1e-6)


def test_workflow_automatically_loads_only_selected_object_and_can_be_disabled(tmp_path,monkeypatch):
    from ContactModelStudy.Hardware.Workflow import build_tasks
    import run_leap_hardware as runner
    monkeypatch.setattr(references,'DEFAULT_REFERENCE_DIR',tmp_path/'references')
    pose = np.linspace(.1,.8,16)
    references.save_reference('cube',pose,{})
    args = runner.parseArgs(['--task','cube_reorient'])
    task,evaluation,metadata = build_tasks(args,tmp_path)
    for current in [task,evaluation]:
        np.testing.assert_allclose(current.jointReference(),pose)
        np.testing.assert_array_equal(current.params['init_qpos'],CubeReorient().params['init_qpos'])
        current.params['init_ctrl'] = np.full(16,.7)
        current.setGoal([1,0,0,0])
        np.testing.assert_allclose(current.goal[7:23],pose)
    assert metadata['rollout']['joint_reference']['path'].endswith('cube.json')
    args = runner.parseArgs(['--no-object-joint-reference'])
    task,_,_ = build_tasks(args,tmp_path)
    np.testing.assert_array_equal(task.jointReference(),task.params['init_ctrl'])


def test_partial_motor_enable_preloads_mapped_targets_before_enabling_and_keeps_others_free():
    hand = HardwareHand.__new__(HardwareHand)
    events,torque = [],np.zeros(16,dtype=int)
    permutation = [1,0,2,3,5,4,6,7,9,8,10,11,12,13,14,15]
    hand.mapping = SimpleNamespace(map_mujoco_to_leap=lambda q:np.asarray(q)[permutation],
                                   apply_offset=lambda q:q+np.pi)
    def enable(ids,state,retries):
        events.append(('torque',list(ids),state));torque[ids]=state
    hand.client = SimpleNamespace(write_desired_pos=lambda ids,q:events.append(('target',ids,q.copy())),
                                   set_torque_enabled=enable)
    hand.register = lambda motor,*args:int(torque[motor])
    q = np.linspace(-.5,1.,16)
    hand.enable_motors(q,[4,5,6,7])
    assert events[0][0]=='target' and events[1]==('torque',[4,5,6,7],True)
    np.testing.assert_allclose(events[0][2],q[permutation][4:8]+np.pi)
    assert np.flatnonzero(torque).tolist() == [4,5,6,7]
    hand.disable_motors([4,5,6,7])
    assert not torque.any()


def test_capture_connection_disables_all_motors_before_inspecting_and_configuring():
    hand = HardwareHand.__new__(HardwareHand)
    hand.config = json.loads((ROOT/'configs/leap_hardware.json').read_text())
    torque,events = np.ones(16,dtype=int),[]
    def change(ids,state,retries):
        events.append(('torque',ids,state));torque[ids]=state
    hand.client = SimpleNamespace(connect=lambda:events.append(('connect',)),set_torque_enabled=change)
    fields = {0:'model_number',11:'mode',84:'p',82:'i',80:'d',102:'goal_current'}
    def register(motor,address,size):
        if address==64:return int(torque[motor])
        assert not torque.any()
        value = np.asarray(hand.config['expected_registers'][fields[address]])
        return int(np.broadcast_to(value,(16,))[motor])
    hand.register = register
    hand.health = lambda:events.append(('health',))
    hand.connect(torque_off=True)
    assert events[:2]==[('connect',),('torque',list(range(16)),False)]
    assert not torque.any()


@pytest.mark.parametrize('cancel', [False,True])
def test_viewer_requires_explicit_finger_and_enter_and_cleans_up(model,hand,tmp_path,monkeypatch,cancel):
    now = [0.]
    monkeypatch.setattr(capture.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(capture.time,'sleep',lambda dt:now.__setitem__(0,now[0]+max(dt,.02)))
    monkeypatch.setattr(capture,'ensure_port_free',lambda _:None)
    monkeypatch.setattr(capture,'HardwareHand',lambda *a:hand)
    class Viewer:
        cam = SimpleNamespace(lookat=np.zeros(3))
        opt = SimpleNamespace(geomgroup=np.ones(6,dtype=int))
        ticks = 0
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def lock(self):return nullcontext()
        def set_texts(self,*a):pass
        def is_running(self):return self.ticks<12
        def sync(self):
            if self.ticks==0:
                assert not any(e[0]=='enable' for e in hand.events)
            if self.ticks<4:
                self.callback([52,50,49,51][self.ticks]);self.callback(257)
                if cancel:self.callback(ord('Q'))
            self.ticks+=1
    def launch(*a,key_callback,**kw):
        viewer = Viewer();viewer.callback=key_callback;return viewer
    monkeypatch.setattr(capture.mujoco.viewer,'launch_passive',launch)
    output = tmp_path/'cube.json'
    assert capture.main(['--output',str(output),'--settle','.04','--status-file',str(tmp_path/'status.json')])==0
    assert hand.events[0]==('connect',{'torque_off':True})
    assert hand.events[-1]==('close',)
    enabled = [e[1] for e in hand.events if e[0]=='enable']
    if cancel:
        assert enabled==[] and not output.exists()
    else:
        assert enabled==[[12,13,14,15],[4,5,6,7],[0,1,2,3],[8,9,10,11]]
        assert output.exists()
