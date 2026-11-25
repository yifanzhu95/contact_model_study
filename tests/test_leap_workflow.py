"""Hardware and simulation share planner defaults, task tuning and geometry selection."""
from dataclasses import asdict
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import warp as wp

import run_leap_hardware as hardware
from ContactModelStudy.Drivers import run_episodes as reference
from ContactModelStudy.Hardware import Workflow
from ContactModelStudy.Hardware.Camera import DEFAULT_CAMERA_CONFIG, load_camera_config
from ContactModelStudy.Hardware.Calibration import tag_site_in_base
from ContactModelStudy.Hardware.Rollouts import CalibratedRollouts
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI
from ContactModelStudy.Hardware.Scene import DEFAULT_SCENE
from ContactModelStudy.Hardware.Hand import control_limits, delta_limits
from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujocoConfig
from ContactModelStudy.Utils.ContactModelPresets import getContactModel


@pytest.fixture(autouse=True)
def isolated_object_reference_files(tmp_path,monkeypatch):
    # Local captures must not silently change the legacy-equivalence fixtures.
    from ContactModelStudy.Hardware import JointReferences
    monkeypatch.setattr(JointReferences,'DEFAULT_REFERENCE_DIR',tmp_path/'joint_references')


@pytest.mark.parametrize("extra", [[], ["--plan-on-eval"], ["--control-mode", "absolute", "--warm-start", "hold",
    "--noise-sigma", "0.03", "--temperature", "2", "--convergence-ratio", "1.5",
    "--max-iterations", "6", "--resample-noise", "--no-center-noise", "--delta", "0.1",
    "--uncertainty", "--no-graph"], ["--action-rate-weight", "2.5"]])
def test_planner_configuration_is_identical_to_reference(extra):
    a, b = hardware.parseArgs(extra), reference.parseArgs(extra)
    actual, original = asdict(reference.buildPlannerConfig(a)), asdict(reference.buildPlannerConfig(b))
    assert actual == original
    for field in ("task", "hand_acc", "obj_acc", "rollout_model", "timestep", "substeps",
                  "ctrl_time_step", "horizon", "time_horizon", "goal_difficulty"):
        assert getattr(a, field) == getattr(b, field)


GEOMETRY_OPTIONS = [
    (task, hand, obj)
    for task, hand in product(reference.TASKS, ("low", "med", "high"))
    for obj in (("high", "foam4", "foam16a", "foam16b", "foam64") if task == "duck_reorient" else ("high",))
]


@pytest.mark.parametrize("task,hand,obj", GEOMETRY_OPTIONS)
def test_geometry_and_object_specific_costs_survive_calibration(tmp_path, task, hand, obj):
    args = hardware.parseArgs(["--task", task, "--hand-acc", hand, "--obj-acc", obj,
                               "--cost-weight", "w_quat=12.5"])
    rollout, evaluation, metadata = Workflow.build_tasks(args, tmp_path)
    original, original_eval = reference.buildTasks(args)
    assert rollout.params["cost_weights"] == original.params["cost_weights"]
    for contact in ("M1", "M2", "M3", "M4", "M5", "M6"):
        assert rollout.contactParams(contact) == original.contactParams(contact)
    assert evaluation.params["cost_weights"] == original_eval.params["cost_weights"]
    np.testing.assert_array_equal(rollout.getInitialState()[0], original.getInitialState()[0])
    assert metadata["rollout"]["source_scene"] == original.getModelPath()
    # The selected collision geometry is retained, including FOAM sphere counts.
    assert rollout.mjm.ngeom == original.mjm.ngeom
    for field in ("geom_type", "geom_size", "geom_pos", "geom_contype", "geom_conaffinity", "geom_friction"):
        np.testing.assert_allclose(getattr(rollout.mjm, field), getattr(original.mjm, field), atol=1e-6)
    calibrated = mujoco.MjModel.from_xml_path(str(args.calibrated_hand))
    for field in ("actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "dof_armature",
                  "dof_damping", "dof_frictionloss"):
        np.testing.assert_allclose(getattr(rollout.mjm, field)[:16], getattr(calibrated, field), rtol=1e-12)
    for field in ("jnt_pos", "jnt_axis", "jnt_range"):
        np.testing.assert_allclose(getattr(rollout.mjm, field)[:16], getattr(calibrated, field), rtol=1e-12)
    for name in ("palm", "if_px", "th_ds"):
        np.testing.assert_allclose(rollout.mjm.body(name).mass, calibrated.body(name).mass)
    for field in ("mass", "inertia", "ipos", "iquat"):
        np.testing.assert_allclose(getattr(rollout.mjm.body("obj"), field),
                                   getattr(original.mjm.body("obj"), field), rtol=1e-5, atol=1e-8)
    hardware.model_limits(rollout.mjm)
    hardware.model_limits(evaluation.mjm)
    for selected in (rollout, evaluation):
        np.testing.assert_allclose(tag_site_in_base(selected.mjm), tag_site_in_base(calibrated), atol=1e-12)


@pytest.mark.parametrize("task_name", reference.TASKS)
def test_goal_position_override_survives_resampling_without_changing_default_task_goal(tmp_path, task_name):
    args = hardware.parseArgs(["--task", task_name])
    task, evaluation, _ = Workflow.build_tasks(args, tmp_path)
    original, _ = reference.buildTasks(args)
    for goal in ([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]):
        task.setGoal(goal)
        original.setGoal(goal)
        np.testing.assert_allclose(task.goal, original.goal)
    for current in (task, evaluation):
        current.setTargetPosition([0.012, 0.023, 0.087])
        # Warm a device goal so we check in-place updates used by captured graphs.
        device_goal = current._gpu_arrays("cpu")["goal"]
        for goal in ([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]):
            current.setGoal(goal)
            np.testing.assert_allclose(current.goal[:3], [0.012, 0.023, 0.087])
            np.testing.assert_allclose(device_goal.numpy(), current.goal)


def test_unavailable_geometry_reports_the_selected_options(tmp_path):
    args = hardware.parseArgs(["--task", "ball_reorient", "--obj-acc", "foam4"])
    with pytest.raises(ValueError, match="ball_reorient, hand=high, object=foam4"):
        Workflow.build_tasks(args, tmp_path)


@pytest.mark.parametrize('hand', ['low', 'med', 'high'])
def test_cube_friction_override_reaches_scenes_and_physics_without_changing_sources(tmp_path, hand):
    friction = [.8, .012, .0004]
    args = hardware.parseArgs(['--device', 'cpu', '--rollout-model', 'M1', '--hand-acc', hand,
        '--n-samples', '2', '--horizon', '2', '--cube-friction', *map(str, friction)])
    originals = reference.buildTasks(args)
    source_text = {Path(t.getModelPath()): Path(t.getModelPath()).read_bytes() for t in originals}
    rollout_task, eval_task, metadata = Workflow.build_tasks(args, tmp_path)
    for name, task, original in zip(('rollout', 'eval'), (rollout_task, eval_task), originals):
        model = task.mjm
        collision = ((model.geom_bodyid == model.body('obj').id)
                     & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        assert collision.any()
        np.testing.assert_allclose(model.geom_friction[collision], np.tile(friction, (collision.sum(), 1)))
        # Hand and noncolliding object visuals retain their original friction.
        np.testing.assert_allclose(model.geom_friction[~collision], original.mjm.geom_friction[~collision])
        compiled = mujoco.MjModel.from_xml_string(metadata[name]['scene_xml'])
        np.testing.assert_array_equal(compiled.geom_friction, model.geom_friction)
    backend = Workflow.build_rollout(args, rollout_task, *hardware.model_limits(eval_task.mjm))
    try:
        np.testing.assert_allclose(backend.mjm.geom('obj').friction, friction)
    finally:
        backend.Close()
    for path, content in source_text.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize('flags', [
    ['--cube-friction', '-.1', '.005', '.0001'],
    ['--cube-friction', 'nan', '.005', '.0001'],
    ['--cube-friction', '.8', 'inf', '.0001'],
    ['--task', 'ball_reorient', '--cube-friction', '.8', '.005', '.0001'],
])
def test_invalid_cube_friction_fails_before_startup(flags):
    with pytest.raises(SystemExit):
        hardware.parseArgs(flags)


@pytest.mark.parametrize('hand', ['low', 'med', 'high'])
@pytest.mark.parametrize('surface', ['fingertip', 'palm', 'finger', 'all'])
def test_hand_friction_overrides_are_independent_for_every_geometry_accuracy(tmp_path, hand, surface):
    values = dict(fingertip=[.31, .006, .0002], palm=[.42, .007, .0003], finger=[.53, .008, .0004])
    groups = list(values) if surface == 'all' else [surface]
    flags = ['--device', 'cpu', '--hand-acc', hand, '--rollout-model', 'M1', '--n-samples', '2', '--horizon', '2']
    for group in groups:
        flags += [f'--{group}-friction', *map(str, values[group])]
    args = hardware.parseArgs(flags)
    original = reference.buildTasks(args)
    source_text = {Path(t.getModelPath()): Path(t.getModelPath()).read_bytes() for t in original}
    rollout_task, evaluation, _ = Workflow.build_tasks(args, tmp_path)
    for task, baseline in zip((rollout_task, evaluation), original):
        model = task.mjm
        masks = {key: np.zeros(model.ngeom, dtype=bool) for key in values}
        for i in range(model.ngeom):
            if not (model.geom_contype[i] or model.geom_conaffinity[i]):
                continue
            body, name = model.body(int(model.geom_bodyid[i])).name, model.geom(i).name
            if body == 'palm':
                masks['palm'][i] = True
            elif body.startswith(('if_', 'mf_', 'rf_', 'th_')):
                tip = (name in ('if_tip', 'mf_tip', 'rf_tip', 'th_tip')
                       or (not name and body in ('if_ds', 'mf_ds', 'rf_ds', 'th_ds')
                           and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE))
                masks['fingertip' if tip else 'finger'][i] = True
        assert masks['fingertip'].sum() == 4
        changed = np.zeros(model.ngeom, dtype=bool)
        for group in groups:
            mask = masks[group]
            assert mask.any()
            np.testing.assert_allclose(model.geom_friction[mask], np.tile(values[group], (mask.sum(), 1)))
            changed |= mask
        np.testing.assert_array_equal(model.geom_friction[~changed], baseline.mjm.geom_friction[~changed])
        np.testing.assert_array_equal(model.geom_condim, baseline.mjm.geom_condim)
        np.testing.assert_array_equal(model.geom_priority, baseline.mjm.geom_priority)
        assert model.ngeom == baseline.mjm.ngeom
    if surface == 'all':
        backend = Workflow.build_rollout(args, rollout_task, *hardware.model_limits(evaluation.mjm))
        try:
            np.testing.assert_array_equal(backend.mjm.geom_friction, rollout_task.mjm.geom_friction)
        finally:
            backend.Close()
    for path, data in source_text.items():
        assert path.read_bytes() == data


@pytest.mark.parametrize('surface', ['fingertip', 'palm', 'finger'])
@pytest.mark.parametrize('value', ['-.1', 'nan', 'inf'])
def test_invalid_hand_friction_fails_before_startup(surface, value):
    with pytest.raises(SystemExit):
        hardware.parseArgs([f'--{surface}-friction', value, '.005', '.0001'])


def test_cpu_contact_preset_and_explicit_overrides_reach_physics(tmp_path):
    args = hardware.parseArgs(["--device", "cpu", "--rollout-model", "M1", "--n-samples", "2",
        "--timestep", "0.001", "--eval-steps-per-rollout-step", "4", "--substeps", "3",
        "--horizon", "2", "--contact-param", "iterations=123"])
    task, evaluation, _ = Workflow.build_tasks(args, tmp_path)
    low, high = hardware.model_limits(task.mjm)
    rollout = Workflow.build_rollout(args, task, low, high)
    expected = getContactModel("M1").makeConfig(task,
        **reference.rolloutSimOverrides(args, task), iterations=123, device="cpu",
        gravity=tuple(task.mjm.opt.gravity))
    assert asdict(rollout.config) == asdict(expected)
    assert rollout.mjm.opt.iterations == 123
    assert rollout.mjm.opt.timestep == 0.004 and evaluation.timestep == 0.001
    assert rollout.config.control_timestep == 0.012
    np.testing.assert_allclose(rollout.mjm.geom_solref[:, 0], 0.008)
    np.testing.assert_allclose(rollout.mjm.geom_solimp[:, 0], 0.9999)


@pytest.mark.parametrize("task_name,contact", product(reference.TASKS, ("M1", "M2")))
def test_cpu_incremental_rollouts_execute_for_every_object_and_cpu_contact_model(tmp_path, task_name, contact):
    args = hardware.parseArgs(["--hardware", "--device", "cpu", "--task", task_name,
        "--rollout-model", contact, "--n-samples", "4", "--horizon", "2", "--substeps", "1"])
    task, evaluation, _ = Workflow.build_tasks(args, tmp_path)
    rollout = Workflow.build_rollout(args, task, *hardware.model_limits(evaluation.mjm))
    config = reference.buildPlannerConfig(args)
    config.use_graph = False
    q, v, u = evaluation.getInitialState()
    with wp.ScopedDevice("cpu"):
        planner = MPPI(rollout, task, config)
        rollout.activation.assign(u.astype(np.float32))
        action = planner.Plan(q, v, u=u)
    assert planner.last_plan_ok and np.isfinite(action).all()
    assert planner.config.delta_range == (None, None)
    assert np.isfinite(rollout.DeviceState().qpos.numpy()).all()


def test_delta_bounds_do_not_replace_absolute_scene_or_bridge_limits(tmp_path):
    import leap_hand_lcm_bridge as bridge_module
    args = hardware.parseArgs(["--hardware", "--device", "cpu", "--n-samples", "2"])
    config = {"ctrl_range": {"lower": 0.1, "upper": np.linspace(0.4, 0.6, 16).tolist()}}
    task, evaluation, _ = Workflow.build_tasks(args, tmp_path, hardware_config=config)
    original = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    lo, hi = control_limits(original)
    for current in (task, evaluation):
        np.testing.assert_allclose(current.mjm.actuator_ctrlrange, original.actuator_ctrlrange)
    rollout = Workflow.build_rollout(args, task, lo, hi)
    np.testing.assert_allclose(rollout.mjm.actuator_ctrlrange, np.column_stack((lo, hi)))
    connection = SimpleNamespace(subscribe=lambda *a: SimpleNamespace(set_queue_capacity=lambda n: None))
    bridge = bridge_module.LeapHandLCMBridge(None, connection, original, config)
    np.testing.assert_array_equal(bridge.lower, lo)
    np.testing.assert_array_equal(bridge.upper, hi)
    down, up = delta_limits(config)
    np.testing.assert_array_equal(bridge.delta_lower, down)
    np.testing.assert_array_equal(bridge.delta_upper, up)


@pytest.mark.parametrize('extra', [[], ['--delta', '.04']])
def test_hardware_ctrl_range_does_not_change_standard_planner_samples(tmp_path, extra):
    args = hardware.parseArgs(["--hardware", "--device", "cpu", "--n-samples", "8",
                               "--horizon", "2", "--substeps", "2", "--noise-sigma", ".2", *extra])
    samples = []
    for index, bound in enumerate([.001, .5]):
        directory = tmp_path/str(index)
        directory.mkdir()
        task, _, _ = Workflow.build_tasks(args, directory,
            hardware_config=dict(ctrl_range=dict(lower=bound, upper=bound)))
        rollout = Workflow.build_rollout(args, task, *hardware.model_limits(task.mjm))
        planner_cfg = reference.buildPlannerConfig(args)
        planner_cfg.use_graph = False
        with wp.ScopedDevice('cpu'):
            planner = hardware.MPPI(rollout, task, planner_cfg)
            assert type(planner) is MPPI
            planner.Reset()
            planner._sampleNoise()
            planner._buildSamples()
        samples.append(planner.V_wp.numpy())
        assert planner.config.delta_range == ((None, None) if not extra else (-.04, .04))
    np.testing.assert_array_equal(samples[0], samples[1])
    assert np.max(np.abs(samples[0])) > .001
    if extra:
        assert np.max(np.abs(samples[0])) <= .040001
    else:
        assert np.max(np.abs(samples[0])) > .12


@pytest.mark.parametrize('reference_kind,mode', [('previous_target', 'ctrl_relative'), ('measured_position', 'pos_relative')])
def test_hardware_config_selects_matching_relative_planner_mode(tmp_path, reference_kind, mode):
    import json
    path = tmp_path/'hardware.json'
    path.write_text(json.dumps(dict(delta_reference=reference_kind)))
    flags = ['--hardware', '--hardware-config', str(path)]
    assert hardware.parseArgs(flags).control_mode == mode
    assert hardware.parseArgs(flags + ['--control-mode', mode]).control_mode == mode
    for other in {'absolute', 'ctrl_relative', 'pos_relative'} - {mode}:
        with pytest.raises(SystemExit):
            hardware.parseArgs(flags + ['--control-mode', other])


@pytest.mark.parametrize("model", ["M1", "M2", "M3", "M4", "M5", "M6"])
@pytest.mark.parametrize("task_name", reference.TASKS)
def test_every_cuda_backend_uses_the_shared_factory(tmp_path, monkeypatch, model, task_name):
    args = hardware.parseArgs(["--rollout-model", model, "--task", task_name])
    task, _, _ = Workflow.build_tasks(args, tmp_path)
    calls = []
    def factory(name, path, **kwargs):
        calls.append((name, path, kwargs))
        return "backend"
    monkeypatch.setattr(Workflow, "GetContactModelSim", factory)
    monkeypatch.setattr(Workflow, "CalibratedRollouts", lambda backend: backend)
    assert Workflow.build_rollout(args, task, *hardware.model_limits(task.mjm)) == "backend"
    name, path, kwargs = calls[0]
    assert name == model and path == task.getModelPath() and kwargs["task"] is task
    assert kwargs["N"] == args.n_samples
    if model == "M3":
        cfg = getContactModel(model).makeConfig(task)
        expected = getContactModel(model).makeConfig(reference.buildTasks(args)[0])
        assert asdict(cfg) == asdict(expected)


def test_camera_intrinsics_and_extrinsics_are_loaded_together(tmp_path):
    import json
    config = load_camera_config()
    assert config["camera_intrinsics"][0, 0] > 0
    assert config["image_size"] == (640, 480)
    data = json.loads(DEFAULT_CAMERA_CONFIG.read_text())
    data.pop('T_base_from_tag', None)
    data.pop('T_tag_from_camera', None)
    data['camera_to_published'] = config['camera_to_published'].tolist()
    data["camera_intrinsics"][0][0] = 700
    data["camera_to_published"][0][3] += 0.01
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(data))
    updated = load_camera_config(path)
    assert updated["camera_intrinsics"][0, 0] == 700
    assert updated["camera_to_published"][0, 3] == pytest.approx(config["camera_to_published"][0, 3] + 0.01)
    override = np.eye(3)
    override[0, 0] = override[1, 1] = 500
    np.testing.assert_array_equal(load_camera_config(path, intrinsics=override)["camera_intrinsics"], override)
    data["camera_intrinsics"][1][1] = -1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="camera_intrinsics"):
        load_camera_config(path)


def test_external_servo_filter_resets_and_integrates_without_backend_activation():
    # Kamino has target buffers rather than MjData.act. Exercise its adapter
    # contract with Warp CPU buffers without importing Newton or requiring CUDA.
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    cfg = VectorizedMujocoConfig(device="cpu", timestep=0.002, horizon=2)
    control = wp.zeros((1, 16), dtype=wp.float32, device="cpu")
    targets = []
    def set_control(value):
        control.assign(value.numpy() if isinstance(value, wp.array) else np.asarray(value).reshape(1, 16))
    backend = SimpleNamespace(model_path=DEFAULT_SCENE, config=cfg, N=1, mjm=model,
        nq=23, nv=22, nu=16, BroadcastState=lambda *a: None, ClearControlSequence=lambda: None,
        SetControl=set_control, DeviceState=lambda: SimpleNamespace(ctrl=control),
        Step_GPU=lambda n: targets.append(control.numpy().copy()))
    with wp.ScopedDevice("cpu"):
        sim = CalibratedRollouts(backend)
        sim.activation.assign(np.full(16, 0.2, dtype=np.float32))
        q = wp.zeros(23, dtype=wp.float32, device="cpu")
        sim.BroadcastState(q)
        sim.SetControl(np.full((1, 16), 0.3, dtype=np.float32))
        sim.Step_GPU(2)
        np.testing.assert_allclose(targets[0], 0.2)
        expected = 0.3 - 0.1 * np.exp(-0.002 / model.actuator_dynprm[:, 0])
        np.testing.assert_allclose(targets[1][0], expected, rtol=1e-6)
        sim.BroadcastState(q)
        sim.SetControl(np.full((1, 16), 0.3, dtype=np.float32))
        sim.Step_GPU(2)
        np.testing.assert_array_equal(targets[0], targets[2])


@pytest.mark.skipif(not hasattr(mujoco.MjModel, "dof_dampingpoly"), reason="MuJoCo before polynomial damping")
def test_calibration_preserves_polynomial_joint_damping(tmp_path):
    from pathlib import Path
    hand = DEFAULT_SCENE.parent / 'calibrated_hand.xml'
    spec = mujoco.MjSpec.from_file(str(hand))
    spec.meshdir = str(hand.parent / 'assets')
    spec.joint('if_mcp').damping = [.03, .002, .0001]
    calibration = tmp_path / 'polynomial_hand.xml'
    calibration.write_text(spec.to_xml())
    scene = tmp_path / 'scene.xml'
    Workflow.calibrate_scene(Path(__file__).resolve().parents[1] / 'scenes/leap/env_leap_eval_cube.xml',
                             calibration, scene)
    calibrated = mujoco.MjModel.from_xml_path(str(calibration))
    model = mujoco.MjModel.from_xml_path(str(scene))
    np.testing.assert_allclose(model.dof_damping[:16], calibrated.dof_damping, rtol=1e-12)
    np.testing.assert_allclose(model.dof_dampingpoly[:16], calibrated.dof_dampingpoly, rtol=1e-12)
