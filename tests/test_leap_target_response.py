"""Validate the hardware-response experiment without device access."""
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import measure_leap_target_response as response
from ContactModelStudy.Hardware.TargetResponseAnalysis import harmonic_lag


def test_identical_absolute_targets_are_small_and_return_to_reference():
    reference=np.arange(16)*.01
    for joint in response.MOVING_JOINTS:
        targets=np.array(response.sequence(reference,joint,.08,4))
        np.testing.assert_allclose(targets[:,joint]-reference[joint],np.tile([.08,0,-.08,0],4),atol=1e-12)
        np.testing.assert_allclose(targets[-1],reference)
        assert np.max(abs(np.diff(targets,axis=0)))<=.08000001
        np.testing.assert_array_equal(np.delete(targets,joint,axis=1),np.tile(np.delete(reference,joint),(16,1)))


@pytest.mark.parametrize('amplitude',[0,-.01,.2,float('nan')])
def test_invalid_motion_amplitude_is_rejected(amplitude):
    with pytest.raises(ValueError):response.sequence(np.zeros(16),2,amplitude,4)


def test_rate_uses_within_episode_publications_not_operator_waits(tmp_path):
    path=tmp_path/'controller.json'
    path.write_text(json.dumps(dict(episodes=[dict(steps=[dict(action_timestamp_us=t) for t in times])
        for times in ([1000000,1070000,1140000],[90000000,90070000]) ])))
    assert response.measured_period(path)==pytest.approx(.07)


def test_out_of_range_targets_rejected_before_hardware():
    model=mujoco.MjModel.from_xml_path(str(response.DEFAULT_MODEL))
    reference=np.zeros(16);reference[2]=model.jnt_range[2,1]
    with pytest.raises(ValueError,match='outside'):response.validate_targets(model,reference,.08,4)


def test_default_mode_never_starts_hardware(monkeypatch,capsys):
    monkeypatch.setattr(response.Experiment,'run',lambda *_:pytest.fail('No hardware in default mode'))
    assert response.main([])==0
    assert 'USB not opened' in capsys.readouterr().out


def test_harmonic_lag_distinguishes_bias_gain_and_delay():
    period=.07;t=np.arange(0,3,.016);omega=2*np.pi/(4*period)
    target=.08*np.sin(omega*t)+.9
    measured=.06*np.sin(omega*(t-.035))+.87
    result=harmonic_lag(t,target,measured,period)
    assert result['lag_ms']==pytest.approx(35,abs=1e-8)
    assert result['amplitude_ratio']==pytest.approx(.75)


def test_sequence_validation_uses_calibrated_joint_order():
    model=mujoco.MjModel.from_xml_path(str(response.DEFAULT_MODEL))
    config=json.loads(response.DEFAULT_CONFIG.read_text())
    lower,upper=response.control_limits(model)
    reference=np.clip(config['robot_qref'],lower,upper)
    response.validate_targets(model,reference,.08,4)


def test_loaded_sequence_never_opens_more_than_starting_grasp():
    reference=np.full(16,.5)
    for joint in response.MOVING_JOINTS:
        targets=np.array(response.sequence(reference,joint,.03,4,closing_only=True))
        assert np.all(targets>=reference) and np.max(targets-reference)<=.0300001
        np.testing.assert_allclose(targets[:,joint]-reference[joint],np.tile([.03,.015,0,.015],4))


def test_loaded_operator_gate_keeps_the_same_target_and_restores_failure_status(tmp_path,monkeypatch):
    test=response.Experiment.__new__(response.Experiment)
    test.directory=tmp_path;test.phase=None;test.last_target=np.full(16,.5)
    test.report=dict(status='failed');calls=[]
    def publish(target,kind):
        calls.append((target.copy(),kind));(tmp_path/'object_removed').touch()
    test.publish=publish
    test.check=lambda **kw:calls.append(kw)
    monkeypatch.setattr(response.time,'sleep',lambda _:None)
    test.object_gate('object_removed','Remove cube',ignore_viewer=True)
    assert test.report['status']=='failed'
    assert calls[0]==dict(ignore_viewer=True)
    np.testing.assert_array_equal(calls[1][0],np.full(16,.5))
    assert calls[1][1]=='operator_hold' and 'object_removed_acknowledged_us' in test.report


def test_closing_sine_is_smooth_bounded_and_ends_at_reference():
    ref=np.full(16,.5)
    values=np.array(response.sequence(ref,2,.08,3,closing_only=True,waveform='sine',sine_samples=16))
    assert values.shape==(48,16)
    assert np.min(values-ref)>=0 and np.max(values-ref)<=.08000001
    assert np.max(abs(np.diff(values[:,2])))<.016
    np.testing.assert_array_equal(values[-1],ref)
    np.testing.assert_allclose(values[7,2],.58)
    np.testing.assert_array_equal(np.delete(values,2,axis=1),np.tile(np.delete(ref,2),(48,1)))


def test_sine_lag_fit_uses_sine_period_instead_of_four_sample_step_period():
    period=.07;t=np.arange(0,4,.016);omega=2*np.pi/(16*period)
    result=harmonic_lag(t,.04*np.sin(omega*t),.03*np.sin(omega*(t-.06)),period,samples_per_cycle=16)
    assert result['lag_ms']==pytest.approx(60,abs=1e-8)
    assert result['amplitude_ratio']==pytest.approx(.75)


def test_screen_rejects_unscreened_motion_before_usb(tmp_path):
    path=tmp_path/'screen.json';ref=np.zeros(16)
    record=dict(hardware_candidate=dict(approved=True,waveform='sine',amplitude_rad=.08,
        sine_samples=16,cycles=3,closing_only=True),reference=ref.tolist(),command_period_s=.07)
    path.write_text(json.dumps(record))
    args=SimpleNamespace(waveform='sine',amplitude=.08,sine_samples=16,cycles=3,loaded_object=True)
    response.validate_screen(path,args,ref,.07)
    args.amplitude=.1
    with pytest.raises(ValueError,match='do not match'):response.validate_screen(path,args,ref,.07)


def test_invalid_sine_sampling_rejected_before_usb():
    with pytest.raises(ValueError,match='at least 8'):
        response.sequence(np.zeros(16),2,.08,3,closing_only=True,waveform='sine',sine_samples=4)


def test_common_cube_settling_is_not_counted_as_sine_excitation():
    from screen_leap_sine_response import excitation_against_control
    control=dict(trace=[[0,0,0,.08,1,0,0,0],[1,.01,0,.08,1,0,0,0]])
    same=dict(trace=[list(x) for x in control['trace']])
    assert excitation_against_control(control,same)['pass_excitation'] is False
    same['trace'][1][1]+=.003
    result=excitation_against_control(control,same)
    assert result['pass_excitation'] and result['max_induced_translation_mm']==pytest.approx(3.)


def test_excitation_check_ignores_quaternion_sign():
    from screen_leap_sine_response import excitation_against_control
    control=dict(trace=[[0,0,0,.08,1,0,0,0],[1,0,0,.08,1,0,0,0]])
    same=dict(trace=[[0,0,0,.08,-1,0,0,0],[1,0,0,.08,-1,0,0,0]])
    assert excitation_against_control(control,same)['max_induced_rotation_deg']==pytest.approx(0.)
