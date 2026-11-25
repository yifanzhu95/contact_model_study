"""Reference motion and operator gates with a fake hand/clock, no USB or network."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ContactModelStudy.Hardware import Episodes
from ContactModelStudy.Hardware.LCM import FeedbackBuffer, RobotState
from ContactModelStudy.Hardware.Scene import DEFAULT_SCENE
import mujoco


class Clock:
    def __init__(self):
        self.now = 0.

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def phases(monkeypatch, tmp_path, moving=True, ctrl_range=None):
    clock = Clock()
    monkeypatch.setattr(Episodes, 'time', clock)
    measured = np.zeros(16)
    actions, holds = [], []
    def position(value):
        actions.append((clock.now, value - measured))
        if moving:
            measured[:] = value
    def snapshot_robot(**limits):
        return measured.copy(), np.zeros(16)
    feedback = SimpleNamespace(buffer=SimpleNamespace(snapshot_robot=snapshot_robot, error=None),
        publish_position=position, publish_delta=lambda *a: pytest.fail('Preparation must not publish MPPI deltas'),
        publish_hold=lambda: holds.append(clock.now))
    config = dict(robot_qref=np.linspace(-.6, .6, 16).tolist(),
                  ctrl_range=dict(lower=.1, upper=.1) if ctrl_range is None else ctrl_range,
                  episode_preparation=dict(speed_rad_s=.3, tolerance_rad=.001, command_hz=50, timeout_s=3))
    handler = Episodes.HardwareEpisodes(feedback, config, .15, tmp_path/'phase.json')
    return handler, measured, actions, holds, clock


def test_reference_motion_is_gradual_and_repeated_before_each_episode(monkeypatch, tmp_path):
    handler, measured, actions, holds, clock = phases(monkeypatch, tmp_path)
    gates = []
    def entered(timeout):
        gates.append(measured.copy())
        clock.sleep(timeout)
        return True
    monkeypatch.setattr(handler, 'enter_pressed', entered)
    for episode in range(2):
        measured[:] = 0
        handler.before_episode(episode)
        np.testing.assert_allclose(measured, handler.reference, atol=.001)
    assert len(gates) == 2 and not holds  # ready phase retains the reference target
    assert all(np.abs(delta).max() <= .1 for _, delta in actions)
    # With a responsive hand, each trajectory step advances <= speed / Hz.
    assert max(np.abs(delta).max() for _, delta in actions) <= .3 / 50 + 1e-12
    assert clock.now >= 4
    assert json.loads(handler.status_file.read_text())['stage'] == 'ready'


def test_preparation_timeout_does_not_start_an_episode(monkeypatch, tmp_path):
    handler, _, _, holds, _ = phases(monkeypatch, tmp_path, moving=False)
    handler.timeout = .1
    monkeypatch.setattr(handler, 'enter_pressed', lambda _: pytest.fail('Must not ask to start before reaching reference'))
    with pytest.raises(RuntimeError, match='preparation timeout'):
        handler.before_episode(0)
    assert not holds
    diagnostic = json.loads(handler.status_file.read_text())
    assert diagnostic['motion_state'] == 'timeout'
    assert diagnostic['worst_joint'] == 'if_mcp'
    assert diagnostic['max_error_rad'] == pytest.approx(.6)
    assert len(diagnostic['measured']) == len(diagnostic['error']) == 16
    trace = [json.loads(line) for line in (tmp_path/'preparation_trace.jsonl').read_text().splitlines()]
    assert trace[0]['motion_state'] == 'moving' and trace[-1]['motion_state'] == 'timeout'


def test_success_sends_only_hold_heartbeats_until_enter(monkeypatch, tmp_path):
    handler, _, actions, holds, clock = phases(monkeypatch, tmp_path)
    replies = iter([False, False, False, True])
    def entered(timeout):
        clock.sleep(timeout)
        return next(replies)
    monkeypatch.setattr(handler, 'enter_pressed', entered)
    handler.after_success(1, final=True)
    assert not actions and len(holds) == 5
    assert max(np.diff(holds)) <= handler.period + 1e-12
    status = json.loads(handler.status_file.read_text())
    assert status['stage'] == 'success_hold' and 'return to zero and disable torque' in status['prompt']


def test_video_padding_holds_for_one_second_and_checks_joint_freshness(monkeypatch,tmp_path):
    handler,measured,actions,holds,clock=phases(monkeypatch,tmp_path)
    updates=[]
    handler.hold_for(1.,target=handler.reference,on_tick=lambda:updates.append(clock.now))
    assert clock.now==pytest.approx(1.)
    assert len(actions)>=50 and not holds and len(updates)==len(actions)
    np.testing.assert_allclose(measured,handler.reference)
    actions.clear()
    handler.hold_for(1.)
    assert clock.now==pytest.approx(2.) and len(holds)>=50 and not actions
    def stale(**kwargs):raise RuntimeError('Stale robot feedback')
    handler.feedback.buffer.snapshot_robot=stale
    count=len(holds)
    with pytest.raises(RuntimeError,match='Stale robot'):handler.hold_for(1.)
    assert len(holds)==count  # Never sleep/continue a blind hold on a device fault.


def test_two_episodes_start_anywhere_hold_then_zero_before_next_reference(monkeypatch, tmp_path):
    handler, measured, actions, holds, clock = phases(monkeypatch, tmp_path)
    handler.timeout = 5
    measured[:] = np.linspace(.4, -.3, 16)  # no assumption of a zero start
    replies = []
    def entered(timeout):
        replies.append(json.loads(handler.status_file.read_text())['stage'])
        clock.sleep(timeout)
        return True
    monkeypatch.setattr(handler, 'enter_pressed', entered)
    for episode in range(2):
        handler.before_episode(episode)
        np.testing.assert_allclose(measured, handler.reference, atol=.001)
        measured[:] += .05  # stand in for completed manipulation
        handler.after_success(episode, final=episode == 1)
        # The success gate itself must not reset the hand before confirmation.
        np.testing.assert_allclose(measured, handler.reference + .05, atol=.001)
        q = handler.return_to_zero(episode)
        np.testing.assert_allclose(q, 0, atol=.001)
    assert replies == ['ready', 'success_hold', 'ready', 'success_hold']
    assert json.loads(handler.status_file.read_text())['stage'] == 'zero_reached'
    np.testing.assert_allclose(measured, 0, atol=.001)
    assert holds  # success still latches the measured pose


def test_zero_return_timeout_is_not_reported_as_completion(monkeypatch, tmp_path):
    handler, measured, _, _, _ = phases(monkeypatch, tmp_path, moving=False)
    measured[:] = .4
    handler.timeout = .1
    with pytest.raises(RuntimeError, match='zero pose.*timeout'):
        handler.return_to_zero(0)
    assert json.loads(handler.status_file.read_text())['stage'] == 'returning_to_zero'


def test_waiting_for_tracking_renews_hold_before_planning(monkeypatch, tmp_path):
    handler, _, actions, holds, _ = phases(monkeypatch, tmp_path)
    calls = []
    def snapshot(**limits):
        calls.append(limits)
        if len(calls) < 4:
            raise RuntimeError('Stale object')
        return 'fresh state'
    handler.feedback.buffer.snapshot = snapshot
    assert handler.wait_for_tracking(0, 1, max_pose_age=.3) == 'fresh state'
    assert len(actions) == 4 and not holds
    np.testing.assert_allclose(handler.reference, actions[0][1])
    assert json.loads(handler.status_file.read_text())['stage'] == 'running'


def test_reference_hold_does_not_require_object_tracking():
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    buffer = FeedbackBuffer(model)
    import time
    stamp = int(time.time() * 1e6)
    buffer.on_robot('LEAP_STATE', RobotState(stamp, np.ones(16), np.zeros(16)).encode())
    np.testing.assert_allclose(buffer.snapshot_robot()[0], 1)
    buffer.robot_received -= 1
    with pytest.raises(RuntimeError, match='Stale robot'):
        buffer.snapshot_robot()


def test_stationary_enter_gate_recovers_brief_joint_receive_gap_without_moving(monkeypatch,tmp_path):
    handler,measured,actions,holds,clock=phases(monkeypatch,tmp_path)
    measured[:]=handler.reference
    original=handler.feedback.buffer.snapshot_robot;calls=[0]
    def brief(**limits):
        calls[0]+=1
        if calls[0]<3:raise RuntimeError('Stale robot feedback (0.158s)')
        return original(**limits)
    handler.feedback.buffer.snapshot_robot=brief
    monkeypatch.setattr(handler,'enter_pressed',lambda timeout:True)
    handler.wait_for_enter(0,'ready','Ready',target=handler.reference)
    assert calls[0]==3 and clock.now==pytest.approx(.04)
    assert not holds and all(np.max(abs(delta))==0 for _,delta in actions)


def test_stationary_enter_gate_persistent_stale_state_still_aborts(monkeypatch,tmp_path):
    handler,_,actions,_,clock=phases(monkeypatch,tmp_path)
    def stale(**limits):raise RuntimeError('Stale robot feedback (0.2s)')
    handler.feedback.buffer.snapshot_robot=stale
    monkeypatch.setattr(handler,'enter_pressed',lambda timeout:pytest.fail('Started without fresh joints'))
    with pytest.raises(RuntimeError,match='Stale robot feedback'):
        handler.wait_for_enter(0,'ready','Ready',target=handler.reference)
    assert clock.now==pytest.approx(.1) and actions


def test_stationary_enter_gate_does_not_retry_device_or_decode_errors(monkeypatch,tmp_path):
    handler,_,actions,holds,clock=phases(monkeypatch,tmp_path)
    def invalid(**limits):raise RuntimeError('LCM feedback error: malformed robot state')
    handler.feedback.buffer.snapshot_robot=invalid
    with pytest.raises(RuntimeError,match='malformed'):
        handler.wait_for_enter(0,'ready','Ready',target=handler.reference)
    assert clock.now==0 and not actions and not holds


def test_measured_servo_offset_reaches_ready_and_keeps_reference_target(monkeypatch, tmp_path):
    handler, measured, actions, holds, clock = phases(monkeypatch, tmp_path)
    handler.tolerance = .06
    bias = np.zeros(16)
    bias[[0, 4, 8]] = [.0494, .0284, .0331]
    commands = []
    def servo(command):
        commands.append(command.copy())
        measured[:] = command - bias
    handler.feedback.publish_position = servo
    replies = iter([False, False, False, True])
    def entered(timeout):
        clock.sleep(timeout)
        return next(replies)
    monkeypatch.setattr(handler, 'enter_pressed', entered)
    handler.before_episode(0)
    assert clock.now >= 2 + handler.settle_time
    np.testing.assert_allclose(measured, handler.reference - bias, atol=1e-12)
    np.testing.assert_allclose(commands[-4:], np.tile(handler.reference, (4, 1)), atol=1e-12)
    assert not holds
    trace = [json.loads(line) for line in (tmp_path/'preparation_trace.jsonl').read_text().splitlines()]
    assert trace[-1]['motion_state'] == 'reached'
    assert trace[-1]['max_error_rad'] == pytest.approx(.0494)


def test_reference_must_settle_before_enter_gate(monkeypatch, tmp_path):
    handler, measured, actions, holds, clock = phases(monkeypatch, tmp_path)
    measured[:] = handler.reference
    handler.feedback.buffer.snapshot_robot = lambda **k: (measured.copy(),
        np.full(16, .2 if clock.now < 1 else 0))
    monkeypatch.setattr(handler, 'enter_pressed', lambda timeout: True)
    handler.before_episode(0)
    assert clock.now >= 1 + handler.settle_time
    assert not holds


def test_per_joint_tolerance_accepts_second_episode_offset_without_relaxing_other_joints(monkeypatch, tmp_path):
    handler, measured, actions, holds, clock = phases(monkeypatch, tmp_path, moving=False)
    handler.tolerance = .06
    handler.joint_tolerances = {'if_mcp': .08}
    measured[:] = handler.reference
    measured[0] -= .0617
    measured[4] -= .03
    monkeypatch.setattr(handler, 'enter_pressed', lambda timeout: True)
    handler.before_episode(1)
    assert json.loads(handler.status_file.read_text())['stage'] == 'ready'
    assert handler.position_tolerances()[0] == .08
    assert np.all(handler.position_tolerances()[1:] == .06)
    # The MCP allowance must not hide another joint missing its own bound.
    measured[4] = handler.reference[4] - .07
    handler.timeout = .5
    with pytest.raises(RuntimeError, match='mf_mcp.*0.0600'):
        handler.prepare(1)


def test_contact_during_preparation_and_ready_wait_does_not_timeout_or_start_automatically(monkeypatch, tmp_path):
    handler, measured, actions, holds, clock = phases(monkeypatch, tmp_path, moving=False)
    handler.timeout = 0
    handler.tolerance = .06
    handler.joint_tolerances = {'if_mcp': .08}
    stable = handler.reference.copy()
    stable[0] -= .0617
    def snapshot(**kwargs):
        # Contact holds fingers away for longer than the old 30-second deadline.
        return (handler.reference + .15 if clock.now < 35 else measured.copy()), np.zeros(16)
    measured[:] = stable
    handler.feedback.buffer.snapshot_robot = snapshot
    confirmations = []
    def entered(timeout):
        confirmations.append(clock.now)
        clock.sleep(timeout)
        # A second nudge after readiness must keep waiting for the operator.
        measured[:] = handler.reference + .2 if len(confirmations) < 3 else stable
        return len(confirmations) == 4
    monkeypatch.setattr(handler, 'enter_pressed', entered)
    handler.before_episode(1)
    assert len(confirmations) == 4 and confirmations[0] >= 35 + handler.settle_time
    assert any(np.max(np.abs(delta)) > .1 for _, delta in actions)
    assert not holds
    status = json.loads(handler.status_file.read_text())
    assert status['stage'] == 'ready' and 'press Enter to start MPPI' in status['prompt']


def test_ctrl_c_return_uses_joints_only_and_requests_torque_off_after_zero(monkeypatch, tmp_path):
    handler, measured, actions, holds, clock = phases(monkeypatch, tmp_path)
    measured[:] = .45
    stopped = []
    handler.feedback.publish_stop = lambda: stopped.append(measured.copy())
    np.testing.assert_allclose(handler.shutdown_on_interrupt(0), 0, atol=.001)
    assert len(stopped) == 1
    np.testing.assert_allclose(stopped[0], 0, atol=.001)
    assert clock.now >= .45 / handler.speed
    assert max(np.abs(delta).max() for _, delta in actions) <= .1
    assert json.loads(handler.status_file.read_text())['stage'] == 'interrupt_complete'
    assert handler.timeout == 3  # interrupt timeout does not alter episode preparation


@pytest.mark.parametrize('failure', ['stuck', 'stale', 'second_interrupt'])
def test_failed_or_interrupted_zero_return_still_requests_torque_off(monkeypatch, tmp_path, failure):
    handler, measured, actions, holds, clock = phases(monkeypatch, tmp_path, moving=False)
    handler.timeout = 0
    handler.interrupt_timeout = .1
    measured[:] = .4
    stops = []
    handler.feedback.publish_stop = lambda: stops.append(clock.now)
    if failure == 'stale':
        def stale(**kwargs):
            raise RuntimeError('Stale robot feedback')
        handler.feedback.buffer.snapshot_robot = stale
    elif failure == 'second_interrupt':
        def interrupt(position):
            raise KeyboardInterrupt
        handler.feedback.publish_position = interrupt
    with pytest.raises(KeyboardInterrupt if failure == 'second_interrupt' else RuntimeError):
        handler.shutdown_on_interrupt(1)
    assert len(stops) == 1
    assert handler.timeout == 0
    if failure == 'stuck':
        assert .1 <= clock.now <= .1 + handler.period
    assert json.loads(handler.status_file.read_text())['stage'] == 'interrupt_failed'


def test_logging_off_keeps_phase_ipc_but_omits_preparation_trace(monkeypatch, tmp_path):
    handler, measured, actions, holds, clock = phases(monkeypatch, tmp_path)
    handler.logging_enabled = False
    handler.prepare(0)
    assert json.loads(handler.status_file.read_text())['stage'] == 'preparing'
    assert not (tmp_path/'preparation_trace.jsonl').exists()


def test_feedback_error_keeps_last_target_until_enter_then_zero_and_torque_off(monkeypatch, tmp_path):
    handler, measured, actions, holds, clock = phases(monkeypatch, tmp_path)
    measured[:] = .2
    handler.last_target = np.full(16, .4)  # Do not latch the sagged measurement.
    stops, confirmations = [], []
    handler.feedback.publish_stop = lambda: stops.append(measured.copy())
    def entered(timeout):
        assert not stops
        np.testing.assert_allclose(measured, .4)
        assert json.loads(handler.status_file.read_text())['stage'] == 'feedback_hold'
        confirmations.append(clock.now)
        clock.sleep(timeout)
        return len(confirmations) == 60  # Longer than the bridge's 0.5 s watchdog.
    monkeypatch.setattr(handler, 'enter_pressed', entered)
    q = handler.shutdown_after_feedback_error(0, 'camera timestamps out of sync')
    assert len(confirmations) == 60 and confirmations[-1] > 1
    # Every command while waiting renews the same absolute target, without deltas.
    assert len(actions) > 60 and not holds
    np.testing.assert_allclose(q, 0, atol=.001)
    assert len(stops) == 1
    np.testing.assert_allclose(stops[0], 0, atol=.001)
    assert json.loads(handler.status_file.read_text())['stage'] == 'feedback_complete'
    assert handler.timeout == 3


@pytest.mark.parametrize('failure', ['stale_joints', 'eof', 'reset_timeout'])
def test_feedback_recovery_does_not_mask_robot_or_operator_failures(monkeypatch, tmp_path, failure):
    handler, measured, actions, holds, clock = phases(monkeypatch, tmp_path, moving=False)
    measured[:] = .4
    handler.last_target = measured.copy()
    handler.interrupt_timeout = .1
    stops = []
    handler.feedback.publish_stop = lambda: stops.append(clock.now)
    def entered(timeout):
        if failure == 'eof':
            raise RuntimeError('interactive stdin received EOF')
        return True
    monkeypatch.setattr(handler, 'enter_pressed', entered)
    if failure == 'stale_joints':
        def stale(**kwargs):
            raise RuntimeError('Stale robot feedback')
        handler.feedback.buffer.snapshot_robot = stale
    with pytest.raises(RuntimeError):
        handler.shutdown_after_feedback_error(0, 'camera delay')
    assert len(stops) == 1
    assert json.loads(handler.status_file.read_text())['stage'] == 'feedback_failed'


def test_ctrl_c_during_feedback_hold_confirms_zero_return(monkeypatch, tmp_path):
    handler, measured, actions, holds, clock = phases(monkeypatch, tmp_path)
    measured[:] = .2
    handler.last_target = measured.copy()
    stopped = []
    handler.feedback.publish_stop = lambda: stopped.append(measured.copy())
    def interrupt(timeout):
        raise KeyboardInterrupt
    monkeypatch.setattr(handler, 'enter_pressed', interrupt)
    handler.shutdown_after_feedback_error(0, 'camera delay')
    assert len(stopped) == 1
    np.testing.assert_allclose(stopped[0], 0, atol=.001)


@pytest.mark.parametrize('bound', [0, .001, .05])
def test_reference_and_zero_trajectories_ignore_mppi_delta_bounds(monkeypatch, tmp_path, bound):
    handler, measured, actions, _, clock = phases(monkeypatch, tmp_path,
        ctrl_range=dict(lower=bound, upper=bound))
    commands = []
    original = handler.feedback.publish_position
    def command(position):
        commands.append((clock.now, position.copy()))
        original(position)
    handler.feedback.publish_position = command
    handler.prepare(0)
    np.testing.assert_allclose(measured, handler.reference, atol=.001)
    np.testing.assert_allclose(commands[-1][1], handler.reference)
    assert clock.now >= .6 / handler.speed
    # Waypoints follow preparation speed even when MPPI deltas are frozen.
    assert max(np.max(np.abs(delta)) for _, delta in actions) <= .3 / 50 + 1e-12
    handler.return_to_zero(0)
    np.testing.assert_allclose(measured, 0, atol=.001)
    np.testing.assert_allclose(commands[-1][1], 0)


def test_reference_target_can_build_servo_error_despite_tiny_mppi_bound(monkeypatch, tmp_path):
    handler, measured, _, _, _ = phases(monkeypatch, tmp_path,
        ctrl_range=dict(lower=.001, upper=.001))
    handler.tolerance = .03
    errors = []
    def servo(target):
        error = target - measured
        errors.append(np.abs(error).max())
        # Emulate friction: tiny target errors cannot move the hand. Absolute
        # waypoints must advance until the servo has enough position error.
        active = np.abs(error) > .02
        measured[active] = target[active]
    handler.feedback.publish_position = servo
    handler.prepare(0)
    assert max(errors) > .02
    np.testing.assert_allclose(measured, handler.reference, atol=.03)


def test_ready_wait_keeps_exact_reference_after_a_nudge_independent_of_ctrl_range(monkeypatch, tmp_path):
    handler, measured, _, _, clock = phases(monkeypatch, tmp_path,
        ctrl_range=dict(lower=0, upper=0))
    measured[:] = handler.reference + .2
    positions = []
    handler.feedback.publish_position = lambda target: positions.append(target.copy())
    def entered(timeout):
        clock.sleep(timeout)
        return len(positions) >= 3
    monkeypatch.setattr(handler, 'enter_pressed', entered)
    handler.wait_for_enter(0, 'ready', 'ready', target=handler.reference)
    np.testing.assert_allclose(positions, np.tile(handler.reference, (3, 1)))
