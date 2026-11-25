"""Measured-range cost and serial clipping; fake feedback only, no hardware."""
from dataclasses import replace
import json
import time

import mujoco
import numpy as np
import pytest
import warp as wp

import run_leap_hardware as hardware
from ContactModelStudy.Hardware import JointLimits as limits_module
from ContactModelStudy.Hardware.LCM import RobotDeltaAction, RobotPositionAction
from ContactModelStudy.Hardware.Scene import JOINTS, ROOT
from ContactModelStudy.Hardware.Workflow import build_tasks
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config
from test_action_rate import Servo, StateCost, score
from test_leap_lcm_bridge import make_bridge, run_until_interrupted


@pytest.fixture
def limits():
    config = json.loads((ROOT/'configs/leap_hardware.json').read_text())
    value = limits_module.load_joint_limits(config)
    assert value is not None
    return value


def test_measured_limits_are_saved_and_not_actuator_intersection(limits):
    model = mujoco.MjModel.from_xml_path(str(limits.model))
    np.testing.assert_allclose(limits.lower, model.jnt_range[:, 0])
    np.testing.assert_allclose(limits.upper, model.jnt_range[:, 1])
    i = JOINTS.index('if_rot')
    assert limits.lower[i] < model.actuator_ctrlrange[i, 0]  # use joint bounds, not intersection
    assert limits_module.load_joint_limits({}) is None


def test_loader_rejects_pending_capture_or_modified_xml(tmp_path, limits):
    xml = tmp_path/'measured.xml'
    xml.write_bytes(limits.model.read_bytes())
    record = json.loads(limits.model.with_name('measured_joint_limits_capture.json').read_text())
    path = tmp_path/'measured_joint_limits_capture.json'
    config = {'joint_limits': {'model': str(xml), 'cost_weight': 12.}}
    record['xml_status'] = 'captures_pending_save'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='completed, saved'):
        limits_module.load_joint_limits(config)
    record['xml_status'] = 'all_16_joint_ranges_saved'
    path.write_text(json.dumps(record))
    assert limits_module.load_joint_limits(config).weight == 12.
    xml.write_text(xml.read_text()+'\n')
    with pytest.raises(ValueError, match='completed, saved'):
        limits_module.load_joint_limits(config)


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_invalid_weights_fail_before_any_device_is_opened(limits, weight):
    with pytest.raises(ValueError, match='joint_limit_weight'):
        limits_module.configure_limit_cost(MPPI_Config(), limits, weight)
    with pytest.raises(SystemExit):
        hardware.parseArgs(['--hardware', '--joint-limit-weight', str(weight)])


def test_cost_can_be_disabled_without_disabling_clip(limits):
    cfg = limits_module.configure_limit_cost(MPPI_Config(), limits)
    assert cfg.joint_limit_weight == 100.
    zero = limits_module.configure_limit_cost(cfg, limits, 0.)
    assert zero.joint_limit_weight == 0
    raw = limits.lower-.1
    np.testing.assert_allclose(limits_module.clip_absolute_target(raw, limits), limits.lower)


@pytest.mark.parametrize('horizon', [1, 3])
@pytest.mark.parametrize('mode', ['absolute', 'ctrl_relative', 'pos_relative'])
def test_raw_target_penalty_matches_independent_horizon_calculation(mode, horizon):
    tape = np.array([[[.2, -.5], [-.3, .1], [.8, .6]],
                     [[.8, .7], [.5, -.6], [-.4, .9]]])
    tape = tape[:, :horizon]
    q, previous = np.array([.6, -.2]), np.array([.3, .1])
    lower, upper, weight = np.array([.1, -.3]), np.array([.7, .5]), 3.
    expected = []
    for trajectory in tape:
        position, target = q.copy(), previous.copy()
        cost = 0.
        for action in trajectory:
            target = action if mode == 'absolute' else action+(position if mode == 'pos_relative' else target)
            cost += weight*np.sum(np.maximum(lower-target, 0)**2+np.maximum(target-upper, 0)**2)
            position += .1*target
        expected.append(cost+(horizon-1)*2+7)  # unchanged task costs, including terminal
    with wp.ScopedDevice('cpu'):
        planner = MPPI(Servo(horizon=horizon), StateCost(), MPPI_Config(control_mode=mode, use_graph=False,
            joint_limit_weight=weight, joint_limit_lower=tuple(lower), joint_limit_upper=tuple(upper)))
        np.testing.assert_allclose(score(planner, tape, q, previous), expected, rtol=1e-6)


def test_weight_zero_is_identical_and_penalty_changes_weighted_action():
    with wp.ScopedDevice('cpu'):
        configs = [MPPI_Config(use_graph=False, control_mode='absolute', seed=81, noise_sigma=.5),
                   MPPI_Config(use_graph=False, control_mode='absolute', seed=81, noise_sigma=.5,
                               joint_limit_weight=0, joint_limit_lower=(.5, .5), joint_limit_upper=(1., 1.)),
                   MPPI_Config(use_graph=False, control_mode='absolute', seed=81, noise_sigma=.5,
                               joint_limit_weight=100., joint_limit_lower=(.5, .5), joint_limit_upper=(1., 1.))]
        actions = []
        for cfg in configs:
            planner = MPPI(Servo(n=256), StateCost(), cfg)
            actions.append(planner.Plan(np.zeros(2), np.zeros(2)))
        np.testing.assert_array_equal(actions[0], actions[1])
        assert np.linalg.norm(actions[2]-.5) < np.linalg.norm(actions[0]-.5)


@pytest.mark.parametrize('reference', ['measured_position', 'previous_target'])
def test_bridge_absolute_clip_follows_delta_clip_and_cached_targets_do_not_accumulate(limits, reference):
    bridge, hand, connection = make_bridge(positions=(.4, .4, .8), reference=reference)
    bridge.joint_limits = limits
    bridge.delta_lower[:] = bridge.delta_upper[:] = .2
    connection.pending = RobotDeltaAction(int(time.time()*1e6), np.full(16, 10.)).encode()
    run_until_interrupted(bridge)
    expected = np.clip(np.full(16, .6), limits.lower, limits.upper)
    np.testing.assert_allclose(hand.writes, np.array([expected, expected]))
    assert np.all(np.asarray(hand.writes) >= limits.lower)
    assert np.all(np.asarray(hand.writes) <= limits.upper)


def test_joint_bounds_take_priority_if_measured_pose_is_already_outside(limits):
    bridge, hand, connection = make_bridge(positions=(-1., -1.))
    bridge.joint_limits = limits
    bridge.delta_lower[:] = bridge.delta_upper[:] = .2
    connection.pending = RobotDeltaAction(int(time.time()*1e6), np.zeros(16)).encode()
    run_until_interrupted(bridge)
    expected = np.clip(np.full(16, -1.), limits.lower, limits.upper)
    np.testing.assert_allclose(hand.writes[0], expected)
    assert hand.writes[0][0]-(-1.) > .2  # absolute-priority order is explicit


def test_zero_preparation_and_reset_bypass_only_manipulation_bounds(limits):
    bridge, hand, connection = make_bridge(positions=(.4, .4))
    bridge.joint_limits = limits
    connection.pending = RobotPositionAction(int(time.time()*1e6), np.zeros(16)).encode()
    run_until_interrupted(bridge)
    np.testing.assert_array_equal(hand.writes[0], np.zeros(16))


def test_hardware_scenes_receive_only_measured_joint_ranges(tmp_path, limits):
    args = hardware.parseArgs(['--hardware', '--device', 'cpu'])
    config = json.loads(args.hardware_config.read_text())
    rollout, evaluation, _ = build_tasks(args, tmp_path, hardware_config=config)
    original = mujoco.MjModel.from_xml_path(str(args.calibrated_hand))
    for task in (rollout, evaluation):
        np.testing.assert_allclose(task.mjm.jnt_range[:16], np.column_stack((limits.lower, limits.upper)))
        np.testing.assert_array_equal(task.mjm.actuator_ctrlrange, original.actuator_ctrlrange)
        np.testing.assert_array_equal(task.mjm.actuator_gainprm, original.actuator_gainprm)


@pytest.mark.skipif(not wp.is_cuda_available(), reason='CUDA graph test requires a GPU')
def test_joint_limit_penalty_is_correct_in_cuda_graph():
    # Tiny synthetic plant, not a hand simulation or hardware run.
    cfg = MPPI_Config(use_graph=True, joint_limit_weight=5., joint_limit_lower=(.1, -.1),
                      joint_limit_upper=(.3, .4), control_mode='pos_relative', seed=71)
    with wp.ScopedDevice('cuda:0'):
        graph = MPPI(Servo(device='cuda:0'), StateCost(), cfg)
        eager = MPPI(Servo(device='cuda:0'), StateCost(), replace(cfg, use_graph=False))
        for q in (np.array([.2, -.3]), np.array([.4, .7])):
            np.testing.assert_allclose(graph.Plan(q, np.zeros(2)), eager.Plan(q, np.zeros(2)), atol=1e-6)
        assert graph._rollout_graph is not None
