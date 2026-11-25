"""Low-level serial lifecycle and command bounds, without a physical hand."""
from types import SimpleNamespace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ContactModelStudy.Hardware import Hand as runner


def test_delta_reference_defaults_to_old_mode_and_accepts_both_choices():
    assert runner.delta_reference({}) == 'measured_position'
    for mode in ('measured_position', 'previous_target'):
        assert runner.delta_reference(dict(delta_reference=mode)) == mode


@pytest.mark.parametrize('value', [None, 'invalid', True, 1, []])
def test_invalid_delta_reference_is_rejected(value):
    with pytest.raises(ValueError, match='delta_reference'):
        runner.delta_reference(dict(delta_reference=value))


def test_delta_limits_accept_scalar_and_per_joint_magnitudes():
    down, up = runner.delta_limits({"ctrl_range": {"lower": 0.05, "upper": 0.2}})
    np.testing.assert_allclose(down, 0.05)
    np.testing.assert_allclose(up, 0.2)
    per_joint = np.linspace(0.02, 0.1, 16)
    down, up = runner.delta_limits({"ctrl_range": {"lower": per_joint.tolist(), "upper": None}})
    np.testing.assert_allclose(down, per_joint)
    np.testing.assert_allclose(up, 0.1)
    down, up = runner.delta_limits({})
    np.testing.assert_allclose(down, 0.1)
    np.testing.assert_allclose(up, 0.1)
    model = SimpleNamespace(jnt_range=np.tile([-1.0, 1.0], (16, 1)),
                            actuator_ctrlrange=np.tile([-0.8, 0.8], (16, 1)))
    low, high = runner.control_limits(model)
    np.testing.assert_allclose(low, -0.8)
    np.testing.assert_allclose(high, 0.8)


@pytest.mark.parametrize("bounds", [
    {"lower": [0, 1]}, {"upper": float('nan')}, {"lower": float('inf')},
    {"lower": True}, {"lower": -0.1}, {"upper": -1}, {"min": 0}, [0, 1]])
def test_invalid_delta_magnitudes_are_rejected(bounds):
    with pytest.raises(ValueError, match="ctrl_range"):
        runner.delta_limits({"ctrl_range": bounds})


def test_final_target_is_clipped_around_current_position():
    current = np.full(16, 1.7)
    down, up = runner.delta_limits({"ctrl_range": {"lower": 0.1, "upper": 0.1}})
    np.testing.assert_allclose(runner.clip_joint_target(np.full(16, 5.0), current, down, up), 1.8)
    np.testing.assert_allclose(runner.clip_joint_target(np.full(16, -5.0), current, down, up), 1.6)
    np.testing.assert_allclose(runner.clip_joint_target(np.full(16, 1.74), current, down, up), 1.74)
    # The bound follows a new measured q; it is not a fixed absolute interval.
    np.testing.assert_allclose(runner.clip_joint_target(np.full(16, 5.0), current + 0.5, down, up), 2.3)


def test_rate_limit_cannot_push_a_command_outside_the_current_position_delta():
    with pytest.raises(RuntimeError, match="No command"):
        runner.bounded_command(np.full(16, 1.0), np.full(16, 1.0), np.zeros(16),
            np.full(16, -2.0), np.full(16, 2.0), 0.01, 2.0, np.full(16, 0.1), np.full(16, 0.1))


def test_command_intersects_rate_position_and_tracking_bounds():
    zeros = np.zeros(16)
    actual = runner.bounded_command(np.ones(16), zeros, zeros,
                                    np.full(16, -0.5), np.full(16, 0.5), 0.02, 0.35)
    np.testing.assert_allclose(actual, 0.02)
    with pytest.raises(ValueError, match="finite"):
        runner.bounded_command(np.full(16, np.nan), zeros, zeros, -1, 1, 0.02, 0.35)
    with pytest.raises(RuntimeError, match="No command"):
        runner.bounded_command(zeros, zeros, np.ones(16), -0.5, 0.5, 0.02, 0.35)


@pytest.mark.parametrize('motor,current', [(0,551),(5,-551),(15,700)])
def test_current_fault_reports_signed_native_motor_reading_without_changing_threshold(motor,current):
    from ContactModelStudy.Hardware.Hand import check_device_feedback
    currents = np.zeros(16)
    currents[motor] = current
    with pytest.raises(RuntimeError) as caught:
        check_device_feedback(np.zeros(16),np.zeros(16),currents,{'max_current_raw':550})
    message = str(caught.value)
    assert f'motor {motor}={current} raw' in message
    assert 'allowed +/-550 raw' in message
    assert f'currents_by_motor_id_raw={currents.tolist()}' in message
    # Exact equality remains accepted on both sides, as before.
    currents[motor] = np.sign(current)*550
    check_device_feedback(np.zeros(16),np.zeros(16),currents,{'max_current_raw':550})


def test_upstream_missing_feedback_is_fatal():
    hand = runner.HardwareHand.__new__(runner.HardwareHand)
    hand.client = SimpleNamespace(
        read_pos_vel_cur=lambda: (np.zeros(16), np.zeros(16), np.zeros(16)),
        _pos_vel_cur_reader=SimpleNamespace(address=126, size=10,
                                           operation=SimpleNamespace(isAvailable=lambda *a: False)))
    with pytest.raises(OSError, match="Fresh feedback missing"):
        hand.read()


def test_upstream_joint_order_and_offset_round_trip_without_serial():
    path = (Path.home() / "leap_agentic_sysid/third_party/LEAP_Hand_Hardware/python/"
            "leap_hand_utils/mj_leap_utils.py")
    if not path.is_file():
        pytest.skip("Upstream LEAP checkout is not available")
    spec = importlib.util.spec_from_file_location("hardware_mapping_contract", path)
    mapping = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapping)
    hand = runner.HardwareHand.__new__(runner.HardwareHand)
    hand.mapping = mapping
    q, v, current = np.linspace(-0.1, 1.4, 16), np.linspace(-0.7, 0.8, 16), np.arange(16)
    permutation = [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]
    native = q[permutation] + np.pi
    writes = []
    hand.client = SimpleNamespace(
        read_pos_vel_cur=lambda: (native, v[permutation], current),
        _pos_vel_cur_reader=SimpleNamespace(address=126, size=10,
            operation=SimpleNamespace(isAvailable=lambda *a: True)),
        write_desired_pos=lambda motors, target: writes.append((motors, target)))
    actual_q, actual_v, actual_current = hand.read()
    np.testing.assert_allclose(actual_q, q)
    np.testing.assert_allclose(actual_v, v)
    np.testing.assert_array_equal(actual_current, current)
    hand.write(actual_q)
    assert writes[0][0] == list(range(16))
    np.testing.assert_allclose(writes[0][1], native)


def test_enable_preloads_measured_pose_before_torque():
    events = []
    hand = runner.HardwareHand.__new__(runner.HardwareHand)
    hand.write = lambda q: events.append(("write", q.copy()))
    hand.client = SimpleNamespace(set_torque_enabled=lambda *a, **k: events.append(("torque", True)))
    hand.register = lambda *a: 1
    q = np.full(16, 0.3)
    hand.enable(q)
    assert [event[0] for event in events] == ["write", "torque"]
    np.testing.assert_array_equal(events[0][1], q)


def test_read_only_connection_and_close_never_write_torque_or_registers():
    events = []
    hand = runner.HardwareHand.__new__(runner.HardwareHand)
    class Client:
        is_connected = True
        OPEN_CLIENTS = set()
        port_handler = SimpleNamespace(closePort=lambda: events.append('port closed'))
        connect = staticmethod(lambda: events.append('connected'))
        set_torque_enabled = staticmethod(lambda *a, **k: pytest.fail('read-only torque write'))
    hand.client = Client()
    hand.client.OPEN_CLIENTS.add(hand.client)
    hand.register = lambda *a: pytest.fail('read-only close must not touch registers')
    hand.connect_read_only()
    hand.close_read_only()
    assert events == ['connected', 'port closed']
    assert not hand.client.OPEN_CLIENTS


def test_live_health_checks_read_only_one_motor_but_startup_checks_all():
    hand = runner.HardwareHand.__new__(runner.HardwareHand)
    hand.config = dict(max_temperature=50)
    calls = []
    def register(motor, address, size):
        calls.append((motor, address, size))
        return 25 if address == 146 else 0
    hand.register = register
    hand.health(7)
    assert calls == [(7, 70, 1), (7, 146, 1)]
    calls.clear()
    hand.health()
    assert len(calls) == 32 and {motor for motor, _, _ in calls} == set(range(16))


@pytest.mark.parametrize('address,value', [(70, 1), (146, 50)])
def test_single_motor_health_fault_still_stops(address, value):
    hand = runner.HardwareHand.__new__(runner.HardwareHand)
    hand.config = dict(max_temperature=50)
    hand.register = lambda motor, field, size: value if field == address else 0
    with pytest.raises(RuntimeError, match='Motor 7'):
        hand.health(7)


def test_cleanup_attempts_every_motor_even_after_one_failure():
    hand = runner.HardwareHand.__new__(runner.HardwareHand)
    attempted = []

    def torque(motors, enabled, retries):
        attempted.extend(motors)
        assert not enabled and retries == 0
        if motors == [0]:
            raise OSError("unreachable motor")

    class Client:
        is_connected = True
        OPEN_CLIENTS = set()
        set_torque_enabled = staticmethod(torque)
        port_handler = SimpleNamespace(closePort=lambda: attempted.append("port closed"))

    hand.client = Client()
    hand.client.OPEN_CLIENTS.add(hand.client)
    hand.register = lambda *a: 0
    with pytest.raises(OSError, match="unreachable motor"):
        hand.close()
    assert attempted == [*range(16), "port closed"]
    assert not hand.client.OPEN_CLIENTS


def test_cleanup_recovers_abandoned_sdk_busy_flag_before_disabling_every_motor(capsys):
    events = []
    hand = runner.HardwareHand.__new__(runner.HardwareHand)
    port = SimpleNamespace(is_using=True, clearPort=lambda: events.append('flush'),
                           closePort=lambda: events.append('close'))
    def torque(motors, enabled, retries):
        assert not port.is_using and not enabled and retries == 0
        events.append(('off', motors[0]))
    class Client:
        is_connected = True
        OPEN_CLIENTS = set()
        port_handler = port
        set_torque_enabled = staticmethod(torque)
    hand.client = Client()
    hand.client.OPEN_CLIENTS.add(hand.client)
    def register(motor, address, size):
        assert address == 64 and size == 1
        events.append(('verify', motor))
        return 0
    hand.register = register
    hand.close()
    assert events[0] == 'flush' and events[-1] == 'close'
    assert events[1:-1] == [event for motor in range(16) for event in [('off', motor), ('verify', motor)]]
    assert 'Torque disabled and verified on all 16 motors' in capsys.readouterr().out
    assert not hand.client.OPEN_CLIENTS


def startup_hand():
    """An in-memory register table: no serial port and no torque writes."""
    hand = runner.HardwareHand.__new__(runner.HardwareHand)
    hand.config = dict(max_temperature=50, expected_registers=dict(
        model_number=1240, mode=5, p=[450, 600, 600, 600]*3 + [600]*4,
        i=0, d=[150, 200, 200, 200]*3 + [200]*4, goal_current=550))
    fields = {'model_number': 0, 'mode': 11, 'p': 84, 'i': 82, 'd': 80, 'goal_current': 102}
    registers = {(motor, address): int(np.broadcast_to(hand.config['expected_registers'][name], (16,))[motor])
                 for name, address in fields.items() for motor in runner.MOTORS}
    registers.update({(motor, address): value for motor in runner.MOTORS
                      for address, value in [(64, 0), (70, 0), (146, 25)]})
    events = []
    def read(motor, address, size):
        events.append(('read', motor, address))
        return registers[motor, address]
    def write(port, motor, address, value):
        events.append(('write', motor, address, value))
        registers[motor, address] = value
        return 0, 0
    hand.register = read
    hand.client = SimpleNamespace(connect=lambda: events.append(('connect',)),
        dxl=SimpleNamespace(COMM_SUCCESS=0), port_handler=object(),
        packet_handler=SimpleNamespace(write2ByteTxRx=write))
    return hand, registers, events


def test_startup_initializes_mismatched_gains_and_current_with_verified_readback():
    hand, registers, events = startup_hand()
    for motor in runner.MOTORS:
        registers[motor, 84] = 800
        registers[motor, 80] = 0
    registers[9, 102] = 100
    hand.connect()
    writes = [event for event in events if event[0] == 'write']
    assert len(writes) == 33
    assert {event[2] for event in writes} == {84, 80, 102}
    for motor in runner.MOTORS:
        assert registers[motor, 64] == 0
        assert registers[motor, 84] == hand.config['expected_registers']['p'][motor]
        assert registers[motor, 80] == hand.config['expected_registers']['d'][motor]
    assert registers[9, 102] == 550
    first_write = next(i for i, event in enumerate(events) if event[0] == 'write')
    for motor in runner.MOTORS:
        # Every motor's identity, mode, torque and health checked before any write.
        for address in (0, 11, 64, 70, 146):
            assert ('read', motor, address) in events[:first_write]
    for index, event in enumerate(events):
        if event[0] == 'write':
            assert events[index+1] == ('read', event[1], event[2])


def test_startup_leaves_matching_registers_unchanged():
    hand, _, events = startup_hand()
    hand.connect()
    assert not any(event[0] == 'write' for event in events)


def test_temporary_usb_latency_is_set_before_serial_and_restored_on_startup_fault():
    hand,registers,events=startup_hand()
    hand.usb_latency=SimpleNamespace(start=lambda:events.append(('latency',1)),
                                     close=lambda:events.append(('latency',16)))
    registers[15,11]=3
    with pytest.raises(RuntimeError,match='mode=3'):
        hand.connect()
    assert events[:2]==[('latency',1),('connect',)]
    assert events[-1]==('latency',16)
    assert not any(event[0]=='write' for event in events)


def test_usb_latency_restores_after_serial_connection_failure():
    hand,_,events=startup_hand()
    hand.usb_latency=SimpleNamespace(start=lambda:events.append(('latency',1)),
                                     close=lambda:events.append(('latency',16)))
    def fail():raise OSError('serial busy')
    hand.client.connect=fail
    with pytest.raises(OSError,match='serial busy'):hand.connect()
    assert events==[('latency',1),('latency',16)]


def test_read_only_close_restores_latency_after_port_closure_without_motor_writes():
    events=[]
    hand=runner.HardwareHand.__new__(runner.HardwareHand)
    hand.usb_latency=SimpleNamespace(start=lambda:events.append('latency 1'),
                                     close=lambda:events.append('latency 16'))
    class Client:
        is_connected=True
        OPEN_CLIENTS=set()
        connect=staticmethod(lambda:events.append('serial open'))
        port_handler=SimpleNamespace(closePort=lambda:events.append('serial close'))
    hand.client=Client();hand.client.OPEN_CLIENTS.add(hand.client)
    hand.connect_read_only();hand.close_read_only()
    assert events==['latency 1','serial open','serial close','latency 16']


@pytest.mark.parametrize('address,value,message', [
    (0, 100, 'model_number=100, expected 1240'),
    (11, 3, 'mode=3, expected 5'), (64, 1, 'start with torque disabled'),
    (70, 1, 'hardware error'), (146, 50, 'temperature limit')])
def test_invalid_motor_or_health_blocks_all_initialization(address, value, message):
    hand, registers, events = startup_hand()
    registers[0, 84] = 800  # earlier motor needs a gain update
    registers[15, address] = value  # later motor must block that update
    with pytest.raises(RuntimeError, match=message):
        hand.connect()
    assert not any(event[0] == 'write' for event in events)


def test_startup_rejects_failed_register_write():
    hand, registers, _ = startup_hand()
    registers[0, 84] = 800
    hand.client.packet_handler.write2ByteTxRx = lambda *args: (1, 0)
    with pytest.raises(OSError, match='initialize p'):
        hand.connect()


def test_startup_rejects_acknowledged_write_with_wrong_readback():
    hand, registers, _ = startup_hand()
    registers[0, 84] = 800
    hand.client.packet_handler.write2ByteTxRx = lambda *args: (0, 0)
    with pytest.raises(RuntimeError, match='p readback=800, expected 450'):
        hand.connect()


@pytest.mark.parametrize('field,value', [('p', [1, 2]), ('p', float('nan')), ('p', True),
    ('i', -1), ('d', 1.5), ('goal_current', 65536), ('mode', 256)])
def test_invalid_register_configuration_is_rejected_before_connecting(field, value):
    hand, _, events = startup_hand()
    hand.config['expected_registers'][field] = value
    with pytest.raises(ValueError, match=f'expected_registers.{field}'):
        hand.connect()
    assert events == []
