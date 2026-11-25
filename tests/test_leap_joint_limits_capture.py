"""Joint-limit capture and XML isolation checks; no serial access or simulation."""
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import pytest

import capture_leap_joint_limits as capture
from ContactModelStudy.Hardware.Scene import CALIBRATION_FIELDS, JOINTS


@pytest.fixture
def session(tmp_path):
    # Absolute asset path lets copies compile in a disposable directory.
    source = tmp_path/'source.xml'
    source.write_text(capture.SOURCE.read_text().replace('meshdir="assets"',
                      f'meshdir="{capture.SOURCE.parent / "assets"}"'))
    return capture.LimitCapture(source, tmp_path/'measured.xml', tmp_path/'capture.json')


def samples(value, index=0, velocity=0.):
    q = np.zeros(16)
    q[index] = value
    v = np.zeros(16)
    v[index] = velocity
    return [(t, q.copy(), v.copy()) for t in np.linspace(10., 10.6, 7)]


def test_prepare_is_exact_copy_and_cannot_overwrite_source_or_unknown_output(session):
    assert session.source.read_bytes() == session.output.read_bytes()
    with pytest.raises(ValueError, match='different files'):
        capture.LimitCapture(session.source, session.source, session.path)
    with pytest.raises(FileExistsError):
        capture.LimitCapture(session.source, session.output, session.path.with_name('other.json'))
    with pytest.raises(ValueError, match='every joint'):
        session.save()


def test_stability_order_recapture_and_resume_preserve_raw_data(session):
    for rows, pattern in [(samples(-.4, velocity=.2), 'moving'),
                          (samples(-.4)[:3], 'complete stable')]:
        with pytest.raises(ValueError, match=pattern):
            session.capture(rows, .08, .01, .5)
    name, side, value = session.capture(samples(-.4), .08, .01, .5)
    assert (name, side, value) == ('if_mcp', 'lower', -.4)
    assert session.selected == ('if_mcp', 'upper')
    with pytest.raises(ValueError, match='Lower must'):
        session.capture(samples(-.5), .08, .01, .5)
    session.capture(samples(2.), .08, .01, .5)
    session.selected = ('if_mcp', 'lower')
    session.capture(samples(-.42), .08, .01, .5)
    resumed = capture.LimitCapture(session.source, session.output, session.path)
    assert resumed.selected == ('if_rot', 'lower')
    assert len(resumed.record['history']) == 3
    assert resumed.record['captures']['if_mcp']['lower']['q_rad'] == [-.42]*7
    assert resumed.output.read_bytes() == resumed.source.read_bytes()  # only save commits XML


def test_final_xml_changes_only_explicit_joint_limits_and_keeps_all_dynamics(session):
    original = session.source.read_bytes()
    base = mujoco.MjModel.from_xml_path(str(session.source))
    expected = base.jnt_range.copy()
    expected[:, 0] -= .1
    expected[:, 1] += .1
    for i, name in enumerate(JOINTS):
        for k, side in enumerate(('lower', 'upper')):
            session.selected = (name, side)
            session.capture(samples(expected[i, k], i), .08, .01, .5)
    session.save()
    updated = mujoco.MjModel.from_xml_path(str(session.output))
    assert session.source.read_bytes() == original
    np.testing.assert_allclose(updated.jnt_range, expected)
    assert updated.jnt_limited.all()
    for field in CALIBRATION_FIELDS:
        if field != 'jnt_range':
            np.testing.assert_array_equal(getattr(updated, field), getattr(base, field), err_msg=field)
    source_tree, new_tree = ET.fromstring(original), ET.fromstring(session.output.read_bytes())
    for before, after in zip(source_tree.iter(), new_tree.iter(), strict=True):
        attrs = dict(after.attrib)
        if before.tag == 'joint' and before.get('name') in JOINTS:
            attrs.pop('range')
            attrs.pop('limited')
        assert attrs == before.attrib
    assert session.record['xml_status'] == 'all_16_joint_ranges_saved'
    assert capture.LimitCapture(session.source, session.output, session.path).selected is None


def test_source_and_output_external_edits_are_detected(session):
    session.output.write_text(session.output.read_text()+'\n')
    with pytest.raises(ValueError, match='changed externally'):
        capture.LimitCapture(session.source, session.output, session.path)


def test_viewer_reads_feedback_with_torque_off_and_never_sends_targets(session, monkeypatch):
    from contextlib import nullcontext
    events = []
    # No write()/enable() methods exist on this fake: any such call fails the test.
    hand = SimpleNamespace(connect_read_only=lambda: events.append('connect'),
        disable_motors=lambda ids: events.append(('disable', list(ids))),
        read=lambda: (np.zeros(16), np.zeros(16), np.zeros(16)),
        register=lambda *args: 0, health_register=lambda *args: None)
    class Viewer:
        def __init__(self):
            self.remaining = 2
            self.cam = SimpleNamespace(lookat=np.zeros(3))
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def lock(self): return nullcontext()
        def is_running(self):
            self.remaining -= 1
            return self.remaining >= 0
        def sync(self): pass
        def set_texts(self, *args): pass
    monkeypatch.setattr(mujoco.viewer, 'launch_passive', lambda *a, **kw: Viewer())
    monkeypatch.setattr(capture.threading, 'Thread', lambda **kw: SimpleNamespace(start=lambda: None))
    model = mujoco.MjModel.from_xml_path(str(session.source))
    capture.run(hand, model, {'max_current_raw': 550}, session, threading.Event(), 1000., .5, .08, .01)
    assert events == ['connect', ('disable', list(range(16)))]


def test_main_disables_torque_on_error(session, monkeypatch):
    events = []
    monkeypatch.setattr(capture, 'LimitCapture', lambda: session)
    monkeypatch.setattr(capture, 'OUTPUT', session.output)
    monkeypatch.setattr(capture.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(capture, 'ensure_port_free', lambda *args: None)
    monkeypatch.setattr(capture, 'CaptureHand', lambda *args: SimpleNamespace(close=lambda: events.append('closed')))
    def failing_run(*args):
        raise OSError('lost feedback')
    monkeypatch.setattr(capture, 'run', failing_run)
    with pytest.raises(OSError, match='lost feedback'):
        capture.main([])
    assert events == ['closed']


def test_prepare_only_never_creates_a_hardware_client(session, monkeypatch):
    monkeypatch.setattr(capture, 'LimitCapture', lambda: session)
    monkeypatch.setattr(capture, 'CaptureHand', lambda *args: pytest.fail('Unexpected hardware access'))
    assert capture.main(['--prepare-only']) == 0


def holding_hand():
    events = []
    permutation = np.array([1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15])
    hand = SimpleNamespace(events=events,
        mapping=SimpleNamespace(map_mujoco_to_leap=lambda q: np.asarray(q)[permutation]),
        enable_motors=lambda q, motors: events.append(('enable', list(motors), q.copy())),
        disable_motors=lambda motors: events.append(('disable', list(motors))))
    return hand, permutation


@pytest.mark.parametrize('index', range(16))
def test_enter_holds_only_correct_native_motor_then_saves_settled_pose_and_releases(session, index):
    hand, permutation = holding_hand()
    session.selected = (JOINTS[index], 'lower')
    hold = capture.EndpointHold(hand, session, .5, .08, .01)
    q = np.zeros(16)
    q[index] = -.4
    # No pre-capture stability history is required.
    hold.start(q, 10.)
    native = int(np.flatnonzero(permutation == index)[0])
    assert hand.events[0][:2] == ('enable', [native])
    np.testing.assert_array_equal(hand.events[0][2], q)
    settled = q.copy()
    settled[index] -= .003
    assert hold.observe(settled, np.zeros(16), 10.1) is None
    assert not session.record['captures']
    result = None
    for t in np.linspace(10.3, 10.9, 7):
        result = hold.observe(settled, np.zeros(16), t) or result
    assert 'Captured' in result
    assert hold.motor is None
    assert hand.events[-1] == ('disable', [native])
    endpoint = session.record['captures'][JOINTS[index]]['lower']
    assert endpoint['value_rad'] == settled[index]
    assert endpoint['hold']['target_rad'] == -.4
    assert endpoint['hold']['native_motor_id'] == native


@pytest.mark.parametrize('reason', ['cancel', 'timeout', 'drift', 'invalid_order'])
def test_hold_releases_without_saving_on_cancel_timeout_drift_or_bad_bound(session, reason):
    hand, _ = holding_hand()
    hold = capture.EndpointHold(hand, session, .5, .08, .01)
    q, v = np.zeros(16), np.zeros(16)
    if reason == 'invalid_order':
        session.capture(samples(-.4), .08, .01, .5)
        # Now selected upper at -.5 would be below captured lower -.4.
        q[0] = -.5
    hold.start(q, 10.)
    count = len(session.record['history'])
    if reason == 'cancel':
        hold.release()
    elif reason == 'timeout':
        assert 'timeout' in hold.observe(q, v, 21.)
    elif reason == 'drift':
        q[0] += .2
        assert 'too far' in hold.observe(q, v, 10.3)
    else:
        result = None
        for t in np.linspace(10.3, 10.9, 7):
            result = hold.observe(q, v, t) or result
        assert 'Not captured' in result
    assert hold.motor is None
    assert hand.events[-1] == ('disable', [1])  # if_mcp is native motor 1
    assert len(session.record['history']) == count


def test_hold_cannot_switch_or_restart_while_active_and_cleans_up_partial_enable(session):
    hand, _ = holding_hand()
    hold = capture.EndpointHold(hand, session, .5, .08, .01)
    hold.start(np.zeros(16), 10.)
    with pytest.raises(ValueError, match='settling'):
        hold.start(np.zeros(16), 11.)
    hold.release()
    def failed_enable(*args):
        raise OSError('enable readback failed')
    hand.enable_motors = failed_enable
    with pytest.raises(OSError, match='readback failed'):
        hold.start(np.zeros(16), 12.)
    assert hold.motor is None
    assert hand.events[-1] == ('disable', [1])


def test_viewer_failure_releases_active_hold(session, monkeypatch):
    hand, _ = holding_hand()
    def failing_loop(hand, model, config, session, stopped, rate, hold):
        hold.start(np.zeros(16), 10.)
        raise OSError('feedback missing')
    monkeypatch.setattr(capture, '_run', failing_loop)
    with pytest.raises(OSError, match='feedback missing'):
        capture.run(hand, None, {}, session, threading.Event(), 50., .5, .08, .01)
    assert hand.events[-1] == ('disable', [1])


@pytest.mark.parametrize('replies, expected_calls, expected_exception', [
    ([(0, -3001, 0), (1, 0, 0)], 2, None),
    ([(0, -3002, 0), (1, 0, 0)], 2, None),
    ([(0, -3001, 0)]*3, 3, capture.TransientReadError),
    ([(0, 0, 4)], 1, OSError),  # device error must not be retried
])
def test_register_read_retries_only_transient_receive_errors(monkeypatch, replies, expected_calls, expected_exception):
    calls, clears = [], []
    def read(*args):
        calls.append(args)
        return replies[len(calls)-1]
    hand = capture.CaptureHand.__new__(capture.CaptureHand)
    hand.client = SimpleNamespace(
        dxl=SimpleNamespace(COMM_RX_TIMEOUT=-3001, COMM_RX_CORRUPT=-3002, COMM_SUCCESS=0),
        packet_handler=SimpleNamespace(read1ByteTxRx=read),
        port_handler=SimpleNamespace(clearPort=lambda: clears.append(True)))
    monkeypatch.setattr(capture.time, 'sleep', lambda _: None)
    if expected_exception:
        with pytest.raises(expected_exception):
            hand.register(4, 64, 1)
    else:
        assert hand.register(4, 64, 1) == 1
    assert len(calls) == expected_calls
    assert len(clears) == expected_calls-1


def test_bulk_feedback_retries_a_fresh_read_without_returning_failed_cached_values(monkeypatch):
    calls = []
    fresh = (np.ones(16), np.zeros(16), np.zeros(16))
    def read(self):
        calls.append(True)
        if len(calls) == 1:
            raise capture.TransientReadError('timeout')
        return fresh
    hand = capture.CaptureHand.__new__(capture.CaptureHand)
    hand.client = SimpleNamespace(port_handler=SimpleNamespace(clearPort=lambda: None))
    monkeypatch.setattr(capture.HardwareHand, 'read', read)
    monkeypatch.setattr(capture.time, 'sleep', lambda _: None)
    assert hand.read() is fresh
    assert len(calls) == 2
