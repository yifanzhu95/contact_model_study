"""Unthrottled delta application, duplicate handling and torque cleanup."""
from types import SimpleNamespace
import time
import signal
import sys

import mujoco
import numpy as np
import pytest

import leap_hand_lcm_bridge as bridge_module
from ContactModelStudy.Hardware.LCM import (
    DELTA_ACTION_CHANNEL, LCMFeedback, RobotAction, RobotDeltaAction, RobotPositionAction,
    RobotHoldAction, RobotStopAction, RobotState, RobotTargetState, FeedbackBuffer,
)
from ContactModelStudy.Hardware.Scene import DEFAULT_SCENE


class Connection:
    def __init__(self):
        self.messages = []
        self.handler = None
        self.pending = None
        self.schedule = {}
        self.step = 0
        self.unsubscribed = False

    def subscribe(self, channel, callback):
        assert channel == DELTA_ACTION_CHANNEL
        self.handler = callback
        return SimpleNamespace(set_queue_capacity=lambda n: None)

    def unsubscribe(self, subscription):
        self.unsubscribed = True

    def handle_timeout(self, milliseconds):
        assert milliseconds == 0  # never wait for a control-loop timer
        data = self.pending
        self.pending = None
        if data is None:
            data = self.schedule.pop(self.step, None)
        if data is None:
            return 0
        self.handler(DELTA_ACTION_CHANNEL, data)
        return 1

    def publish(self, channel, data):
        self.messages.append((channel, data))


class Hand:
    def __init__(self, connection, positions=(0.2, 0.2), velocity=0):
        self.events, self.writes = [], []
        self.connection, self.positions = connection, positions
        self.reads, self.velocity, self.current = 0, velocity, 0

    def connect(self):
        self.events.append('connect')

    def connect_read_only(self):
        self.events.append('connect read only')

    def read(self):
        self.events.append('read')
        self.reads += 1
        if self.reads > len(self.positions):
            raise KeyboardInterrupt
        self.connection.step = self.reads
        return (np.full(16, self.positions[self.reads - 1]),
                np.full(16, self.velocity), np.full(16, self.current))

    def enable(self, q):
        np.testing.assert_allclose(q, self.positions[0])
        self.events.append('enable measured')

    def write(self, command):
        self.writes.append(command.copy())
        self.events.append('write')

    def health(self, motor=None):
        self.events.append('health')

    def close(self):
        self.events.append('torque off')

    def close_read_only(self):
        self.events.append('serial closed read only')


def make_bridge(positions=(0.2, 0.2), velocity=0, state_only=False, reference='measured_position'):
    connection = Connection()
    hand = Hand(connection, positions, velocity)
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    # These old settings must have no effect on the manipulation path.
    config = dict(max_command_speed=1e-12, max_tracking_error=1e-12,
                  max_velocity=1e-12, max_current_raw=550, delta_reference=reference)
    return bridge_module.LeapHandLCMBridge(hand, connection, model, config, state_only=state_only), hand, connection


def run_until_interrupted(bridge):
    try:
        with pytest.raises(KeyboardInterrupt):
            bridge.run()
    finally:
        bridge.close()


def no_sleep(_):
    pytest.fail('Manipulation bridge must not sleep or pace commands')


def test_bridge_starts_measured_and_applies_full_delta_without_slew_limit(monkeypatch):
    bridge, hand, connection = make_bridge(velocity=100)
    connection.pending = RobotDeltaAction(int(time.time() * 1e6), np.full(16, 0.3)).encode()
    monkeypatch.setattr(bridge_module.time, 'sleep', no_sleep)
    run_until_interrupted(bridge)
    assert hand.events == ['connect', 'read', 'enable measured', 'read', 'write', 'read', 'torque off']
    np.testing.assert_allclose(hand.writes, 0.3)  # q=0.2 + clip(delta=0.3)=0.1
    channel, data = connection.messages[0]
    assert channel == 'LEAP_STATE'
    np.testing.assert_allclose(RobotState.decode(data).qpos, 0.2)
    np.testing.assert_allclose(RobotState.decode(data).qvel, 100)
    assert connection.unsubscribed


def test_new_delta_uses_bridge_pose_once_and_cached_writes_hold_absolute_target(monkeypatch):
    bridge, hand, connection = make_bridge(positions=(1.7, 1.7, 1.75, 1.76, 1.73), velocity=100)
    stamp = int(time.time() * 1e6)
    first = RobotDeltaAction(stamp, np.full(16, 1.0)).encode()
    second = RobotDeltaAction(stamp + 1, np.full(16, -1.0)).encode()
    connection.schedule = {2:first, 3:first, 4:second}
    monkeypatch.setattr(bridge_module.time, 'sleep', no_sleep)
    run_until_interrupted(bridge)
    # No physical/model-range clipping, tracking/velocity cap, or delta replay.
    np.testing.assert_allclose(hand.writes, np.array([1.8, 1.8, 1.66, 1.66])[:, None] * np.ones((1, 16)))


def test_previous_target_accumulates_once_and_zero_delta_preserves_hold_despite_sag(monkeypatch):
    bridge, hand, connection = make_bridge(positions=(.2, .2, .1, .05, 0, -.02), reference='previous_target')
    bridge.delta_lower[:] = bridge.delta_upper[:] = .05
    stamp = int(time.time()*1e6)
    connection.schedule = {
        2: RobotPositionAction(stamp, np.full(16, .8)).encode(),
        3: RobotDeltaAction(stamp+1, np.full(16, .03)).encode(),
        4: RobotDeltaAction(stamp+1, np.full(16, .03)).encode(),
        5: RobotDeltaAction(stamp+2, np.zeros(16)).encode(),
        6: RobotDeltaAction(stamp+3, np.full(16, -.2)).encode(),
    }
    monkeypatch.setattr(bridge_module.time, 'sleep', no_sleep)
    run_until_interrupted(bridge)
    expected = np.array([.8, .83, .83, .83, .78])[:, None] * np.ones((1, 16))
    np.testing.assert_allclose(hand.writes, expected)
    targets = [RobotTargetState.decode(raw).action for channel, raw in connection.messages if channel=='LEAP_STATE_TARGET']
    np.testing.assert_allclose(targets[0], .2)  # initial measured-pose preload
    np.testing.assert_allclose(targets[1:], expected, atol=1e-7)


def test_previous_target_feedback_stays_available_before_first_controller_command():
    bridge, hand, connection = make_bridge(positions=(.2, .15, .1), reference='previous_target')
    run_until_interrupted(bridge)
    assert not hand.writes
    targets = [RobotTargetState.decode(raw) for channel, raw in connection.messages if channel=='LEAP_STATE_TARGET']
    assert len(targets) == 3  # initial preload plus both feedback-only iterations
    for target in targets:
        np.testing.assert_allclose(target.action, .2)
    assert targets[-1].timestamp >= targets[0].timestamp


def test_previous_target_baseline_updates_after_explicit_hold_and_zero_reset(monkeypatch):
    bridge, hand, connection = make_bridge(positions=(.2, .2, .25, .3, .35, .4, .45, .5), reference='previous_target')
    stamp = int(time.time()*1e6)
    connection.schedule = {
        2: RobotPositionAction(stamp, np.full(16, .8)).encode(),
        3: RobotDeltaAction(stamp+1, np.full(16, .01)).encode(),
        4: RobotHoldAction(stamp+2).encode(),
        5: RobotHoldAction(stamp+3).encode(),
        6: RobotDeltaAction(stamp+4, np.full(16, .02)).encode(),
        7: RobotPositionAction(stamp+5, np.zeros(16)).encode(),
        8: RobotDeltaAction(stamp+6, np.full(16, .03)).encode(),
    }
    run_until_interrupted(bridge)
    np.testing.assert_allclose(hand.writes, np.array([.8, .81, .3, .3, .32, 0, .03])[:, None]*np.ones((1,16)))


def test_target_feedback_rejects_stale_bad_and_out_of_order_updates():
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    buffer = FeedbackBuffer(model)
    with pytest.raises(RuntimeError, match='Waiting for bridge target'):
        buffer.snapshot_target()
    stamp = int(time.time()*1e6)
    buffer.on_target('TARGET', RobotTargetState(stamp, np.full(16, .8)).encode())
    buffer.on_target('TARGET', RobotTargetState(stamp-1, np.zeros(16)).encode())
    target, metadata = buffer.snapshot_target()
    np.testing.assert_allclose(target, .8)
    assert metadata['target_timestamp'] == stamp
    buffer.on_target('TARGET', b'broken')
    with pytest.raises(RuntimeError, match='LCM target feedback error'):
        buffer.snapshot_target()
    buffer.on_target('TARGET', RobotTargetState(stamp+1, np.ones(16)).encode())
    target, _ = buffer.snapshot_target()
    np.testing.assert_allclose(target, 1)
    buffer.target_received -= 1
    with pytest.raises(RuntimeError, match='Stale bridge target'):
        buffer.snapshot_target()


def test_previous_target_feedback_subscription_uses_state_channel_suffix():
    import threading
    subscriptions = []
    queues = []
    client = SimpleNamespace(
        subscribe=lambda channel, callback: (subscriptions.append(channel) or
            SimpleNamespace(set_queue_capacity=lambda n: queues.append(n))),
        unsubscribe=lambda sub: None,
        handle_timeout=lambda ms: threading.Event().wait(ms/1000))
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    feedback = LCMFeedback(model, robot_channel='MY_STATE', connection=client, target_feedback=True)
    try:
        assert subscriptions == ['MY_STATE', 'POSE', 'MY_STATE_TARGET']
        assert queues == [1, 1, 1]
    finally:
        feedback.close()


def test_startup_wait_requires_target_feedback_when_requested(monkeypatch):
    from ContactModelStudy.Hardware import LCM as module
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock.now,
        sleep=lambda delay: setattr(clock, 'now', clock.now+delay)))
    feedback = LCMFeedback.__new__(LCMFeedback)
    def missing(max_age):
        raise RuntimeError('Waiting for bridge target feedback')
    feedback.buffer = SimpleNamespace(snapshot=lambda **limits: 'fresh state', snapshot_target=missing, error=None)
    assert feedback.wait(.03) == 'fresh state'
    with pytest.raises(RuntimeError, match='Waiting for bridge target feedback'):
        feedback.wait(.03, require_target=True)


def test_final_target_is_bounded_per_joint_around_latest_bridge_state(monkeypatch):
    bridge, hand, connection = make_bridge(positions=(.2, .7, .75))
    bridge.delta_lower = np.linspace(.01, .1, 16)
    bridge.delta_upper = np.linspace(.03, .2, 16)
    delta = np.where(np.arange(16) % 2, -2., 2.)
    connection.pending = RobotDeltaAction(int(time.time()*1e6), delta).encode()
    monkeypatch.setattr(bridge_module.time, 'sleep', no_sleep)
    run_until_interrupted(bridge)
    # Apply to .7, not the initial .2; the cached write stays fixed when q=.75.
    expected = np.clip(.7 + delta, .7 - bridge.delta_lower, .7 + bridge.delta_upper)
    np.testing.assert_allclose(hand.writes, np.tile(expected, (2, 1)))


@pytest.mark.parametrize('bound', [0, .001, .05])
def test_position_waypoints_ignore_ctrl_range_and_switch_back_to_clipped_mppi(monkeypatch, bound):
    bridge, hand, connection = make_bridge(positions=(.2, .2, .25, .3, .35, .4))
    bridge.delta_lower[:] = bridge.delta_upper[:] = bound
    stamp = int(time.time()*1e6)
    target = np.full(16, .8)
    connection.schedule = {
        2: RobotPositionAction(stamp, target).encode(),
        3: RobotPositionAction(stamp, target).encode(),  # duplicate holds exact target
        4: RobotDeltaAction(stamp+1, np.full(16, -.5)).encode(),
        5: RobotPositionAction(stamp+2, np.zeros(16)).encode(),
    }
    monkeypatch.setattr(bridge_module.time, 'sleep', no_sleep)
    run_until_interrupted(bridge)
    np.testing.assert_allclose(hand.writes,
        np.array([.8, .8, .3-bound, 0, 0])[:, None] * np.ones((1, 16)))


def test_position_waypoint_clears_success_hold_latch(monkeypatch):
    bridge, hand, connection = make_bridge(positions=(.2, .2, .25, .3, .35))
    stamp = int(time.time()*1e6)
    connection.schedule = {
        2: RobotHoldAction(stamp).encode(),
        3: RobotPositionAction(stamp+1, np.zeros(16)).encode(),
        4: RobotHoldAction(stamp+2).encode(),
    }
    monkeypatch.setattr(bridge_module.time, 'sleep', no_sleep)
    run_until_interrupted(bridge)
    np.testing.assert_allclose(hand.writes, np.array([.2, 0, .3, .3])[:, None] * np.ones((1, 16)))


def test_position_codec_and_publisher_are_explicitly_distinct_from_mppi_delta():
    output = []
    feedback = LCMFeedback.__new__(LCMFeedback)
    feedback.action_channel = DELTA_ACTION_CHANNEL
    feedback.connection = SimpleNamespace(publish=lambda channel, data: output.append((channel, data)))
    target = np.linspace(-.2, 1.4, 16)
    feedback.publish_position(target)
    channel, raw = output[0]
    assert channel == DELTA_ACTION_CHANNEL and len(raw) == 80
    decoded = RobotPositionAction.decode(raw)
    assert decoded.timestamp > 0
    np.testing.assert_allclose(decoded.action, target, atol=1e-7)
    with pytest.raises(ValueError):
        RobotDeltaAction.decode(raw)
    for invalid in (RobotAction(1, target).encode(), RobotDeltaAction(1, target).encode(), raw[:-1]):
        with pytest.raises(ValueError, match='LEAPPOS1'):
            RobotPositionAction.decode(invalid)
    with pytest.raises(ValueError, match='finite'):
        feedback.publish_position(np.full(16, np.nan))


def test_position_packet_watchdog_and_ordering_match_manipulation_commands():
    bridge, hand, _ = make_bridge()
    stamp = int(time.time()*1e6)
    bridge.on_action(DELTA_ACTION_CHANNEL, RobotPositionAction(stamp, np.full(16, .8)).encode())
    deadline = bridge.received
    bridge.on_action(DELTA_ACTION_CHANNEL, RobotPositionAction(stamp-1, np.zeros(16)).encode())
    bridge.on_action(DELTA_ACTION_CHANNEL, RobotPositionAction(stamp-1_000_000, np.zeros(16)).encode())
    assert bridge.received == deadline
    np.testing.assert_allclose(bridge.latest.action, .8)
    bridge.received = time.monotonic()-1
    try:
        with pytest.raises(RuntimeError, match='Action timeout'):
            bridge.run()
    finally:
        bridge.close()
    assert not hand.writes and hand.events[-1] == 'torque off'


def test_old_or_duplicate_packets_cannot_reapply_a_delta():
    bridge, _, _ = make_bridge()
    now = int(time.time() * 1e6)
    bridge.on_action(DELTA_ACTION_CHANNEL, RobotDeltaAction(now, np.full(16, 0.04)).encode())
    bridge.on_action(DELTA_ACTION_CHANNEL, RobotDeltaAction(now - 1, np.full(16, 0.09)).encode())
    bridge.on_action(DELTA_ACTION_CHANNEL, RobotDeltaAction(now, np.full(16, 0.09)).encode())
    np.testing.assert_allclose(bridge.latest.delta, 0.04)


def test_hold_heartbeats_keep_one_fixed_measured_pose_and_delta_resumes_motion(monkeypatch):
    bridge, hand, connection = make_bridge(positions=(.2, .2, .24, .26, .3, .31))
    stamp = int(time.time() * 1e6)
    connection.schedule = {
        2: RobotDeltaAction(stamp, np.full(16, .1)).encode(),
        3: RobotHoldAction(stamp + 1).encode(),
        4: RobotHoldAction(stamp + 2).encode(),
        5: RobotDeltaAction(stamp + 3, np.full(16, .05)).encode(),
        6: RobotHoldAction(stamp + 4).encode(),
    }
    monkeypatch.setattr(bridge_module.time, 'sleep', no_sleep)
    run_until_interrupted(bridge)
    np.testing.assert_allclose(hand.writes, np.array([.3, .24, .24, .35, .31])[:, None] * np.ones((1, 16)))


def test_hold_codec_and_publisher_use_action_channel():
    output = []
    feedback = LCMFeedback.__new__(LCMFeedback)
    feedback.action_channel = DELTA_ACTION_CHANNEL
    feedback.connection = SimpleNamespace(publish=lambda channel, data: output.append((channel, data)))
    feedback.publish_hold()
    channel, data = output[0]
    assert channel == DELTA_ACTION_CHANNEL
    assert len(data) == 16 and RobotHoldAction.decode(data).timestamp > 0
    with pytest.raises(ValueError, match='16 bytes'):
        RobotHoldAction.decode(RobotDeltaAction().encode())


def test_hold_heartbeat_timeout_retains_torque_cleanup():
    bridge, hand, _ = make_bridge()
    bridge.latest = RobotHoldAction(int(time.time() * 1e6))
    bridge.received = time.monotonic() - 1
    try:
        with pytest.raises(RuntimeError, match='Action timeout'):
            bridge.run()
    finally:
        bridge.close()
    assert not hand.writes and hand.events[-1] == 'torque off'


def test_explicit_stop_closes_bridge_without_another_motor_command():
    bridge, hand, connection = make_bridge(positions=(0, 0))
    connection.pending = RobotStopAction(int(time.time() * 1e6)).encode()
    try:
        bridge.run()
    finally:
        bridge.close()
    assert not hand.writes and hand.events[-1] == 'torque off'
    assert connection.unsubscribed


def test_stop_codec_and_publisher_use_action_channel():
    output = []
    feedback = LCMFeedback.__new__(LCMFeedback)
    feedback.action_channel = DELTA_ACTION_CHANNEL
    feedback.connection = SimpleNamespace(publish=lambda channel, data: output.append((channel, data)))
    feedback.publish_stop()
    channel, data = output[0]
    assert channel == DELTA_ACTION_CHANNEL
    assert len(data) == 16 and RobotStopAction.decode(data).timestamp > 0


def test_action_timeout_exits_and_cleanup_disables_torque():
    bridge, hand, _ = make_bridge()
    bridge.latest = RobotDeltaAction(int(time.time() * 1e6), np.full(16, 0.1))
    bridge.received = time.monotonic() - 1
    try:
        with pytest.raises(RuntimeError, match='Action timeout'):
            bridge.run()
    finally:
        bridge.close()
    assert not hand.writes and hand.events[-1] == 'torque off'


def test_old_action_is_not_applied():
    bridge, _, _ = make_bridge()
    bridge.on_action(DELTA_ACTION_CHANNEL, RobotDeltaAction(int((time.time() - 1) * 1e6)).encode())
    assert bridge.latest is None
    assert bridge.expired_actions == 1 and bridge.received == 0


def test_expired_queue_packet_cannot_extend_watchdog_or_replace_valid_action():
    bridge, _, _ = make_bridge()
    now = int(time.time() * 1e6)
    bridge.on_action(DELTA_ACTION_CHANNEL, RobotDeltaAction(now, np.full(16, .04)).encode())
    deadline = bridge.received
    bridge.on_action(DELTA_ACTION_CHANNEL, RobotDeltaAction(now - 1000000, np.full(16, .1)).encode())
    assert bridge.received == deadline
    np.testing.assert_allclose(bridge.latest.delta, .04)


def test_packet_transport_age_consumes_watchdog_budget(monkeypatch):
    bridge, _, _ = make_bridge()
    monkeypatch.setattr(bridge_module, 'time', SimpleNamespace(time=lambda: 10., monotonic=lambda: 30.))
    bridge.on_action(DELTA_ACTION_CHANNEL, RobotDeltaAction(9_600_000, np.zeros(16)).encode())
    assert bridge.received == pytest.approx(29.6)
    with pytest.raises(RuntimeError, match='Future-dated'):
        bridge.on_action(DELTA_ACTION_CHANNEL, RobotDeltaAction(11_000_000, np.zeros(16)).encode())


def test_newer_but_expired_packet_cannot_renew_watchdog(monkeypatch):
    bridge, _, _ = make_bridge()
    clock = SimpleNamespace(wall=10., mono=30.)
    monkeypatch.setattr(bridge_module, 'time', SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.mono))
    bridge.on_action(DELTA_ACTION_CHANNEL, RobotDeltaAction(9_800_000, np.full(16, .04)).encode())
    deadline = bridge.received
    clock.wall, clock.mono = 11., 31.
    bridge.on_action(DELTA_ACTION_CHANNEL, RobotDeltaAction(10_000_000, np.full(16, .1)).encode())
    assert bridge.expired_actions == 1 and bridge.received == deadline
    np.testing.assert_allclose(bridge.latest.delta, .04)


def test_serial_health_work_is_spread_without_blocking_feedback_or_delaying_writes(monkeypatch):
    bridge, hand, connection = make_bridge(positions=(.2,) * 200)
    clock = SimpleNamespace(now=0.)
    wall = time.time()
    monkeypatch.setattr(bridge_module, 'time', SimpleNamespace(
        time=lambda: wall + clock.now, monotonic=lambda: clock.now, sleep=no_sleep))
    original_read = hand.read
    checks = []
    def read():
        if clock.now >= 2.2:
            raise KeyboardInterrupt
        clock.now += .016
        connection.pending = RobotDeltaAction(int((wall + clock.now) * 1e6), np.full(16, .01)).encode()
        return original_read()
    def health(motor=None):
        assert motor is not None, 'A full health sweep would block the serial loop for 512 ms'
        checks.append(motor)
        clock.now += .032  # measured two-register cost, versus .512 s for all motors
    hand.read, hand.health = read, health
    run_until_interrupted(bridge)
    stamps = np.array([RobotState.decode(data).timestamp for _, data in connection.messages]) * 1e-6
    assert len(stamps) > 20 and len(hand.writes) == len(stamps)
    assert np.max(np.diff(stamps)) < .15
    assert checks[:16] == list(range(16))
    np.testing.assert_allclose(hand.writes, .21)


def test_delta_packet_is_versioned_and_legacy_absolute_packets_are_rejected():
    bridge, _, _ = make_bridge()
    legacy = RobotAction(int(time.time() * 1e6), np.full(16, 1.7)).encode()
    with pytest.raises(ValueError, match='not an absolute target'):
        bridge.on_action(DELTA_ACTION_CHANNEL, legacy)
    packet = RobotDeltaAction(123, np.arange(16) * 0.001)
    assert len(packet.encode()) == 80
    decoded = RobotDeltaAction.decode(packet.encode())
    assert decoded.timestamp == 123
    np.testing.assert_allclose(decoded.delta, packet.delta)


def test_controller_publisher_sends_delta_codec_without_a_network():
    output = []
    feedback = LCMFeedback.__new__(LCMFeedback)
    feedback.action_channel = DELTA_ACTION_CHANNEL
    feedback.connection = SimpleNamespace(publish=lambda channel, data: output.append((channel, data)))
    delta = np.linspace(-0.1, 0.1, 16)
    feedback.publish_delta(delta)
    channel, raw = output[0]
    assert channel == DELTA_ACTION_CHANNEL
    np.testing.assert_allclose(RobotDeltaAction.decode(raw).delta, delta)


def test_delta_bounds_do_not_limit_absolute_starting_pose(monkeypatch):
    bridge, hand, _ = make_bridge(positions=(0.2, 0.2))
    monkeypatch.setattr(bridge_module.time, 'sleep', no_sleep)
    run_until_interrupted(bridge)
    assert not hand.writes
    assert hand.events[-1] == 'torque off'


def test_device_overcurrent_still_stops_without_commanding_motion():
    bridge, hand, _ = make_bridge()
    hand.current = 551
    try:
        with pytest.raises(RuntimeError, match='current'):
            bridge.run()
    finally:
        bridge.close()
    assert 'enable measured' not in hand.events and not hand.writes


def test_state_only_bridge_has_no_action_subscription_or_motor_writes(monkeypatch):
    bridge, hand, connection = make_bridge(state_only=True)
    assert connection.handler is None and bridge.subscription is None
    bridge.latest = RobotDeltaAction(int(time.time()*1e6), np.full(16, 0.1))
    monkeypatch.setattr(bridge_module.time, 'sleep', no_sleep)
    run_until_interrupted(bridge)
    assert hand.events == ['connect read only', 'read', 'read', 'read', 'serial closed read only']
    assert not hand.writes and len(connection.messages) == 1


def test_sigterm_runs_torque_cleanup_without_second_signal_interrupt(monkeypatch):
    events = []
    old_term, old_int = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    monkeypatch.setitem(sys.modules, 'lcm', SimpleNamespace(LCM=lambda url: object()))
    monkeypatch.setattr(bridge_module, 'HardwareHand', lambda *a: object())
    def run():
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    def close():
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        events.append('torque cleanup')
    monkeypatch.setattr(bridge_module, 'LeapHandLCMBridge', lambda *a: SimpleNamespace(run=run, close=close))
    bridge_module.main([])
    assert events == ['torque cleanup']
    assert signal.getsignal(signal.SIGTERM) == old_term
    assert signal.getsignal(signal.SIGINT) == old_int


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_shutdown_signal_does_not_interrupt_serial_read_or_write_another_command(monkeypatch, signum):
    bridge, hand, connection = make_bridge(positions=(.2, .2))
    connection.pending = RobotDeltaAction(int(time.time()*1e6), np.full(16, .1)).encode()
    original_read = hand.read
    def read():
        if hand.reads == 1:
            # Model a signal while the SDK's synchronous transaction is active.
            signal.getsignal(signum)(signum, None)
            assert bridge.stop_requested
        return original_read()
    hand.read = read
    monkeypatch.setitem(sys.modules, 'lcm', SimpleNamespace(LCM=lambda url: connection))
    monkeypatch.setattr(bridge_module, 'HardwareHand', lambda *a: hand)
    monkeypatch.setattr(bridge_module, 'LeapHandLCMBridge', lambda *a: bridge)
    old_int, old_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    bridge_module.main([])
    assert not hand.writes and hand.events[-1] == 'torque off'
    assert signal.getsignal(signal.SIGINT) == old_int
    assert signal.getsignal(signal.SIGTERM) == old_term


def test_stop_requested_during_first_read_prevents_torque_enable(monkeypatch):
    bridge, hand, _ = make_bridge()
    read = hand.read
    def stopping_read():
        bridge.stop_requested = True
        return read()
    hand.read = stopping_read
    try:
        bridge.run()
    finally:
        bridge.close()
    assert 'enable measured' not in hand.events and not hand.writes


def test_torque_off_only_command_never_starts_bridge_or_enables_torque(monkeypatch):
    events = []
    monkeypatch.setattr(bridge_module, 'HardwareHand', lambda *a: SimpleNamespace(
        connect_read_only=lambda: events.append('connect without motor writes'),
        close=lambda: events.append('disable and verify torque')))
    monkeypatch.setattr(bridge_module, 'LeapHandLCMBridge', lambda *a: pytest.fail('Recovery must not start a bridge'))
    # Recovery also works without loading a scene or connecting to multicast.
    monkeypatch.setattr(bridge_module.mujoco.MjModel, 'from_xml_path', lambda *a: pytest.fail('Recovery needs no model'))
    bridge_module.main(['--disable-torque-only'])
    assert events == ['connect without motor writes', 'disable and verify torque']


def test_opt_in_telemetry_preserves_commands_and_links_reads_to_held_targets(tmp_path, monkeypatch):
    import json
    from ContactModelStudy.Hardware.BridgeTelemetry import BridgeTelemetry
    stamp = int(time.time()*1e6)
    first = RobotDeltaAction(stamp, np.ones(16)).encode()
    second = RobotDeltaAction(stamp+1, np.full(16, -.5)).encode()
    # Keep periodic health scheduling deterministic across both executions.
    monkeypatch.setattr(bridge_module, 'time', SimpleNamespace(time=time.time, monotonic=lambda: 1.))
    results = []
    for recording in [False, True]:
        bridge, hand, connection = make_bridge(positions=(.2, .2, .25, .3))
        connection.schedule = {2:first, 3:first, 4:second}
        if recording:
            bridge.telemetry = BridgeTelemetry(tmp_path/'bridge.jsonl')
        run_until_interrupted(bridge)
        results.append((hand.events, hand.writes, connection.messages))
    assert results[0][0] == results[1][0]  # no additional serial transactions
    np.testing.assert_array_equal(results[0][1], results[1][1])
    records = [json.loads(line) for line in (tmp_path/'bridge.jsonl').read_text().splitlines()]
    rows = [r for r in records if r.get('write_succeeded')]
    assert len(rows) == 3
    np.testing.assert_allclose([r['held_target_before_read'] for r in rows],
                              np.array([.2,.3,.3])[:,None]*np.ones((1,16)))
    np.testing.assert_allclose([r['written_target'] for r in rows], results[1][1])
    np.testing.assert_allclose([r['q'] for r in rows], np.array([.2,.25,.3])[:,None]*np.ones((1,16)))
    assert [r['held_action_timestamp_us'] for r in rows] == [0,stamp,stamp]
    assert [r['written_action_timestamp_us'] for r in rows] == [stamp,stamp,stamp+1]
    assert [r['new_action'] for r in rows] == [True,False,True]
    for row in rows:
        assert row['read_started_us'] <= row['read_finished_us'] <= row['state_timestamp_us']
        assert row['state_timestamp_us'] <= row['write_started_us'] <= row['write_finished_us']
    status = json.loads((tmp_path/'bridge.status.json').read_text())
    assert status['writer_finished'] and status['error'] is None and status['dropped'] == 0


def test_telemetry_retains_device_fault_and_torque_cleanup(tmp_path):
    import json
    from ContactModelStudy.Hardware.BridgeTelemetry import BridgeTelemetry
    bridge, hand, _ = make_bridge(positions=(.2,.2))
    read = hand.read
    def fault_on_loop_read():
        q,v,c = read()
        return q,v,np.full(16, 551.) if hand.reads>1 else c
    hand.read = fault_on_loop_read
    bridge.telemetry = BridgeTelemetry(tmp_path/'fault.jsonl')
    try:
        with pytest.raises(RuntimeError, match='current'):
            bridge.run()
    finally:
        bridge.close()
    assert not hand.writes and hand.events[-1] == 'torque off'
    rows = [json.loads(line) for line in (tmp_path/'fault.jsonl').read_text().splitlines()]
    fault = next(r for r in rows if r.get('error'))
    assert not fault['write_succeeded'] and fault['current_raw_native'][0] == 551


def test_delta_publisher_returns_exact_action_id_for_bridge_join():
    output=[]
    client=LCMFeedback.__new__(LCMFeedback)
    client.action_channel=DELTA_ACTION_CHANNEL
    client.connection=SimpleNamespace(publish=lambda channel,packet:output.append(packet))
    identity=client.publish_delta(np.full(16,.04))
    assert identity == RobotDeltaAction.decode(output[0]).timestamp and identity>0


def test_single_register_health_checks_preserve_full_coverage_and_bound_feedback_gaps(monkeypatch):
    bridge,hand,connection=make_bridge(positions=(.2,)*200)
    clock=SimpleNamespace(now=0.)
    wall=time.time()
    monkeypatch.setattr(bridge_module,'time',SimpleNamespace(
        time=lambda:wall+clock.now,monotonic=lambda:clock.now,sleep=no_sleep))
    read=hand.read;checks=[]
    def measured_read():
        if clock.now>=2.2:
            raise KeyboardInterrupt
        clock.now+=.016
        connection.pending=RobotDeltaAction(int((wall+clock.now)*1e6),np.full(16,.01)).encode()
        return read()
    def health_register(motor,index):
        checks.append((clock.now,motor,index))
        clock.now+=.016
    hand.read=measured_read;hand.health_register=health_register
    run_until_interrupted(bridge)
    stamps=np.array([RobotState.decode(raw).timestamp for _,raw in connection.messages])*1e-6
    assert len(hand.writes)==len(stamps)
    assert np.max(np.diff(stamps))<=.03201
    assert [(m,i) for _,m,i in checks[:32]]==[(m,i) for m in range(16) for i in range(2)]
    assert 1.0<=checks[32][0]-checks[0][0]<=1.04  # same ~1 s monitoring cadence
    np.testing.assert_allclose(hand.writes,.21)
