"""Hardware slider control and cleanup without a GUI or physical motors."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import run_leap_joint_viewer as joint_viewer


def setup_viewer(monkeypatch, failure=None, stop_key=None):
    model = mujoco.MjModel.from_xml_path(str(joint_viewer.ROOT / 'scenes/leap/hardware/calibrated_hand.xml'))
    config = json.loads((joint_viewer.ROOT / 'configs/leap_hardware.json').read_text())
    config.update(config['manual_viewer_limits'])
    config['max_command_speed'] = 1.0
    low, high = joint_viewer.control_limits(model)
    measured = (low + high) / 2
    events, writes, displayed = [], [], []
    clock = [0.0]
    monkeypatch.setattr(joint_viewer.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(joint_viewer.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0] + dt))

    def read():
        if failure == 'read' and 'enabled' in events:
            raise OSError('USB disconnected')
        return measured.copy(), np.zeros(16), np.zeros(16)

    def enable(q):
        np.testing.assert_allclose(q, measured)
        events.append('enabled')

    hand = SimpleNamespace(connect=lambda: events.append('connected'), read=read,
        enable=enable, write=lambda target: writes.append(target.copy()),
        health=lambda: None, close=lambda: events.append('closed'))

    class Viewer:
        running = True
        syncs = 0
        cam = SimpleNamespace(lookat=np.zeros(3), distance=0.0, azimuth=0.0, elevation=0.0)
        def __enter__(self): return self
        def __exit__(self, *args): events.append('viewer closed')
        def lock(self): return nullcontext()
        def is_running(self): return self.running
        def set_texts(self, *args): pass
        def sync(self):
            self.syncs += 1
            displayed.append(self.data.qpos.copy())
            if self.syncs == 1:
                np.testing.assert_allclose(self.data.ctrl, measured)
                self.callback(257)  # operator explicitly arms after viewing the pose
            else:
                self.data.ctrl[:] = measured + 0.7
                clock[0] += 0.02
            if self.syncs == 5:
                if stop_key is not None:
                    self.callback(stop_key)
                else:
                    self.running = False

    def launch(m, data, key_callback, **kwargs):
        if failure == 'viewer':
            raise RuntimeError('No display')
        assert kwargs['show_right_ui']
        viewer = Viewer()
        viewer.data, viewer.callback = data, key_callback
        return viewer

    monkeypatch.setattr(joint_viewer.mujoco.viewer, 'launch_passive', launch)
    monkeypatch.setattr(joint_viewer, 'HardwareHand', lambda *a: hand)
    return hand, model, config, measured, events, writes, displayed


@pytest.mark.parametrize('stop_key', [None, 32, ord('Q'), 256])
def test_sliders_send_bounded_targets_and_render_only_measured_state(monkeypatch, stop_key):
    hand, model, config, measured, events, writes, displayed = setup_viewer(monkeypatch, stop_key=stop_key)
    joint_viewer.run_viewer(hand, model, config)
    assert events[:2] == ['connected', 'enabled']
    assert len(writes) == 2
    # Manual speed limits advance the absolute target independently of MPPI bounds.
    np.testing.assert_allclose(writes, measured + np.array([.02, .04])[:, None])
    for q in displayed:
        np.testing.assert_allclose(q, measured)
    assert events[-1] == 'viewer closed'


@pytest.mark.parametrize('failure', [None, 'read', 'viewer'])
def test_main_disables_torque_and_closes_after_normal_or_failed_test(monkeypatch, failure):
    _, _, _, _, events, _, _ = setup_viewer(monkeypatch, failure=failure)
    if failure is None:
        assert joint_viewer.main([]) == 0
    else:
        with pytest.raises((OSError, RuntimeError)):
            joint_viewer.main([])
    assert events[-1] == 'closed'
    if failure == 'viewer':
        assert 'enabled' not in events


def test_out_of_range_start_is_displayed_with_torque_off_until_repositioned(monkeypatch):
    hand, model, config, measured, events, writes, _ = setup_viewer(monkeypatch)
    measured[0] = model.jnt_range[0, 0] - 0.3
    joint_viewer.run_viewer(hand, model, config)
    assert 'enabled' not in events
    assert not writes


@pytest.mark.parametrize('bound', [0, .001, .05])
def test_manual_slider_motion_ignores_mppi_ctrl_range(monkeypatch, bound):
    hand, model, config, measured, _, writes, _ = setup_viewer(monkeypatch)
    config['ctrl_range'] = dict(lower=bound, upper=bound)
    joint_viewer.run_viewer(hand, model, config)
    np.testing.assert_allclose(writes, measured + np.array([.02, .04])[:, None])
