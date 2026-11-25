"""Coherent planning inputs and timely command rebasing preserve freshness limits."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import run_leap_hardware as runner
from ContactModelStudy.Hardware import LCM


class Clock:
    def __init__(self):
        self.elapsed = 0.
        self.on_sleep = lambda: None

    def time(self): return 1000. + self.elapsed
    def monotonic(self): return 100. + self.elapsed
    def sleep(self, seconds):
        self.elapsed += seconds
        self.on_sleep()


def test_post_plan_rebase_uses_fresh_joints_without_waiting_for_camera(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(LCM, 'time', clock)
    monkeypatch.setattr(runner, 'time', clock)
    model = mujoco.MjModel.from_xml_path(str(Path(runner.ROOT)/'scenes/leap/hardware/cube.xml'))
    buffer = LCM.FeedbackBuffer(model)
    def robot(value):
        buffer.on_robot('LEAP_STATE', LCM.RobotState(int(clock.time()*1e6), np.full(16,value), np.zeros(16)).encode())
    def pose():
        message = LCM.Pose()
        message.timestamp = int(clock.time()*1e6)
        message.pose_matrix = np.eye(4).reshape(16).tolist()
        buffer.on_pose('POSE', message.encode())
    # Reproduce the ages recorded before the failed 41 ms MPPI call.
    robot_stamp = int((clock.time()-.014758)*1e6)
    buffer.on_robot('LEAP_STATE', LCM.RobotState(robot_stamp, np.full(16,.2), np.zeros(16)).encode())
    message = LCM.Pose()
    message.timestamp = int((clock.time()-.133042)*1e6)
    message.pose_matrix = np.eye(4).reshape(16).tolist()
    buffer.on_pose('POSE', message.encode())
    limits = dict(max_robot_age=.15, max_pose_age=.3, max_skew=.15)
    _, _, planned = buffer.snapshot(**limits)
    iteration_started = clock.monotonic()
    clock.elapsed += .041271
    robot(.3)  # new hand feedback, while the tracker still has the old frame
    with pytest.raises(LCM.FeedbackSkewError) as rejected:
        buffer.snapshot(**limits)
    assert rejected.value.metadata['skew_s'] > .15
    assert rejected.value.metadata['pose_age'] < .3
    record = {}
    before = clock.elapsed
    q, _, actual = runner.command_feedback(buffer, limits, planned, iteration_started, .2, record)
    np.testing.assert_allclose(q[:16], .3)
    assert abs(actual['robot_timestamp']-actual['pose_timestamp'])*1e-6 > .15
    assert actual['synchronization_required'] is False
    assert clock.elapsed==before and 'command_feedback_sync' not in record
    # A future PLAN still must wait for a synchronized frame.
    attempts = []
    def next_frame():
        attempts.append(1)
        if len(attempts)==5:pose()
    clock.on_sleep = next_frame
    state = runner.planning_feedback(buffer,limits,record)
    assert abs(state[2]['robot_timestamp']-state[2]['pose_timestamp'])*1e-6 < .15
    assert record['planning_feedback_sync']['retries']==5
    assert record['planning_feedback_sync']['wait_s']==pytest.approx(.01)


def test_latest_crash_timing_no_longer_consumes_plan_age_waiting_for_new_pose(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(LCM,'time',clock)
    monkeypatch.setattr(runner,'time',clock)
    model = mujoco.MjModel.from_xml_path(str(Path(runner.ROOT)/'scenes/leap/hardware/cube.xml'))
    buffer = LCM.FeedbackBuffer(model)
    limits = dict(max_robot_age=.15,max_pose_age=.3,max_skew=.15)
    def robot():
        buffer.on_robot('LEAP_STATE',LCM.RobotState(int(clock.time()*1e6),np.ones(16),np.zeros(16)).encode())
    message = LCM.Pose()
    message.timestamp = int((clock.time()-.137798)*1e6)
    message.pose_matrix = np.eye(4).reshape(16).tolist()
    buffer.on_pose('POSE',message.encode())
    buffer.on_robot('LEAP_STATE',LCM.RobotState(int((clock.time()-.038789)*1e6),np.zeros(16),np.zeros(16)).encode())
    _,_,planned = buffer.snapshot(**limits)
    started = clock.monotonic()
    clock.elapsed += .067856
    robot()
    with pytest.raises(LCM.FeedbackSkewError):buffer.snapshot(**limits)
    q,_,metadata = runner.command_feedback(buffer,limits,planned,started,.2,{})
    np.testing.assert_allclose(q[:16],1)
    assert clock.elapsed==pytest.approx(.067856)
    assert metadata['pose_age']<.3
    # Next planning input waits for the next camera frame instead of accepting
    # an incoherent pair or aging an already-computed plan.
    attempts=[]
    def arrive():
        robot();attempts.append(1)
        if len(attempts)==20:
            message.timestamp=int(clock.time()*1e6)
            buffer.on_pose('POSE',message.encode())
    clock.on_sleep=arrive
    record={}
    _,_,coherent=runner.planning_feedback(buffer,limits,record)
    assert record['planning_feedback_sync']['wait_s']==pytest.approx(.04)
    assert abs(coherent['robot_timestamp']-coherent['pose_timestamp'])*1e-6<=.15


def test_preplan_skew_retry_is_bounded_and_never_accepts_invalid_feedback(monkeypatch):
    clock=Clock();monkeypatch.setattr(runner,'time',clock)
    limits=dict(max_robot_age=.15,max_pose_age=.3,max_skew=.15)
    def skew(**kwargs):
        raise LCM.FeedbackSkewError(dict(robot_timestamp=int(clock.time()*1e6),
            pose_timestamp=int((clock.time()-.16)*1e6),robot_age=0.,pose_age=.16),.15)
    record={}
    with pytest.raises(LCM.FeedbackTimingError,match='synchronize'):
        runner.planning_feedback(SimpleNamespace(snapshot=skew),limits,record)
    assert clock.elapsed<=.150001
    assert record['planning_feedback_sync']['retries']>0
    def invalid(**kwargs):raise RuntimeError('LCM feedback error: bad pose')
    before=clock.elapsed
    with pytest.raises(RuntimeError,match='bad pose'):
        runner.planning_feedback(SimpleNamespace(snapshot=invalid),limits,{})
    assert clock.elapsed==before


def test_lost_camera_still_expires_at_original_pose_age_limit(monkeypatch):
    clock=Clock()
    monkeypatch.setattr(LCM,'time',clock);monkeypatch.setattr(runner,'time',clock)
    model=mujoco.MjModel.from_xml_path(str(Path(runner.ROOT)/'scenes/leap/hardware/cube.xml'))
    buffer=LCM.FeedbackBuffer(model)
    def robot():
        buffer.on_robot('LEAP_STATE',LCM.RobotState(int(clock.time()*1e6),np.zeros(16),np.zeros(16)).encode())
    robot()
    message=LCM.Pose();message.timestamp=int((clock.time()-.2)*1e6)
    message.pose_matrix=np.eye(4).reshape(16).tolist();buffer.on_pose('POSE',message.encode())
    clock.on_sleep=robot
    with pytest.raises(RuntimeError):
        runner.planning_feedback(buffer,dict(max_robot_age=.15,max_pose_age=.3,max_skew=.15),{})
    assert .099 <= clock.elapsed <= .100002


@pytest.mark.parametrize('stream',['robot','object'])
def test_command_snapshot_still_rejects_individually_stale_streams(monkeypatch,stream):
    clock=Clock();monkeypatch.setattr(LCM,'time',clock)
    model=mujoco.MjModel.from_xml_path(str(Path(runner.ROOT)/'scenes/leap/hardware/cube.xml'))
    buffer=LCM.FeedbackBuffer(model)
    robot_age=.151 if stream=='robot' else 0
    pose_age=.301 if stream=='object' else 0
    buffer.on_robot('LEAP_STATE',LCM.RobotState(int((clock.time()-robot_age)*1e6),np.zeros(16),np.zeros(16)).encode())
    message=LCM.Pose();message.timestamp=int((clock.time()-pose_age)*1e6)
    message.pose_matrix=np.eye(4).reshape(16).tolist();buffer.on_pose('POSE',message.encode())
    with pytest.raises(RuntimeError,match=f'Stale {stream}') as error:
        buffer.snapshot(require_sync=False)
    assert isinstance(error.value, LCM.FeedbackTimingError) == (stream == 'object')


@pytest.mark.parametrize('expired', ['plan', 'robot', 'pose'])
def test_stale_original_plan_or_measurement_is_rejected_before_command_read(monkeypatch, expired):
    clock = Clock()
    monkeypatch.setattr(runner, 'time', clock)
    limits = dict(max_robot_age=.15, max_pose_age=.3, max_skew=.15)
    planned = dict(robot_timestamp=int((clock.time()-.13)*1e6), pose_timestamp=int((clock.time()-.13)*1e6))
    duration = .01 if expired=='plan' else .2
    if expired=='pose':
        limits['max_robot_age'] = .5
        planned['pose_timestamp'] = int((clock.time()-.29)*1e6)
        planned['robot_timestamp'] = int((clock.time()-.14)*1e6)
    clock.elapsed = .021 if expired=='robot' else .011
    def forbidden(**limits):
        raise AssertionError('An expired plan must not reach command snapshot')
    record = {}
    with pytest.raises(RuntimeError, match='Planned state is stale'):
        runner.command_feedback(SimpleNamespace(snapshot=forbidden), limits, planned, 100., duration, record)
    assert 'command_feedback_sync' not in record
    key, limit = {'plan':('iteration_wall_time_s','max_plan_seconds'),
                  'robot':('robot_age_s','max_robot_age'), 'pose':('pose_age_s','max_pose_age')}[expired]
    assert record['rejection'][key] >= record['rejection'][limit]


def test_invalid_feedback_is_not_retried(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(runner, 'time', clock)
    def invalid(**limits): raise RuntimeError('LCM feedback error: invalid pose')
    metadata = dict(robot_timestamp=int(clock.time()*1e6), pose_timestamp=int(clock.time()*1e6))
    record = {}
    with pytest.raises(RuntimeError, match='invalid pose'):
        runner.command_feedback(SimpleNamespace(snapshot=invalid),
            dict(max_robot_age=.15,max_pose_age=.3,max_skew=.15), metadata, clock.monotonic(), .2, record)
    assert clock.elapsed==0 and 'command_feedback_sync' not in record
    assert 'invalid pose' in record['command_feedback_error']


def test_feedback_read_cannot_overrun_the_original_plan_deadline(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(runner, 'time', clock)
    def delayed(**limits):
        clock.elapsed += .021
        return np.zeros(23), np.zeros(22), {}
    metadata = dict(robot_timestamp=int(clock.time()*1e6), pose_timestamp=int(clock.time()*1e6))
    with pytest.raises(RuntimeError, match='Planned state is stale'):
        runner.command_feedback(SimpleNamespace(snapshot=delayed),
            dict(max_robot_age=.15,max_pose_age=.3,max_skew=.15), metadata, clock.monotonic(), .02)


def test_pacing_wait_does_not_extend_original_state_freshness(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(runner, 'time', clock)
    started = clock.monotonic()
    # The joints used for planning are already 100 ms old. A 64 ms period must
    # not silently permit publishing a command from that expired planned state.
    metadata = dict(robot_timestamp=int((clock.time()-.1)*1e6), pose_timestamp=int(clock.time()*1e6))
    clock.elapsed += .02
    assert runner.wait_for_control_deadline(started+.064) == pytest.approx(.044)
    def forbidden(**limits):
        pytest.fail('Expired planned state must be rejected before reading command feedback')
    with pytest.raises(LCM.FeedbackTimingError, match='Planned state is stale'):
        runner.command_feedback(SimpleNamespace(snapshot=forbidden),
            dict(max_robot_age=.15, max_pose_age=.3, max_skew=.15), metadata, started, .2)
