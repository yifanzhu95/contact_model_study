"""LCM interoperability, complete measured states and calibrated cube rollouts."""
import importlib.util
import json
from pathlib import Path
import struct
import time
from types import SimpleNamespace
import zipfile

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import warp as wp

import run_leap_hardware as runner
from ContactModelStudy.Hardware.LCM import FeedbackBuffer, Pose, RobotAction, RobotState, load_pose_transform
from ContactModelStudy.Hardware.Scene import check_calibration, prepare_scene
from ContactModelStudy.Hardware.Rollouts import CpuRollouts
from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujocoConfig


def test_failed_plan_snapshot_retains_invalid_costs_without_invalid_json(tmp_path):
    array=lambda value: SimpleNamespace(numpy=lambda:np.asarray(value))
    planner=SimpleNamespace(costs_wp=array([np.nan,np.inf,np.nan]),
        last_plan_start=dict(U=np.zeros((2,16)),noise_seed=0,resample_count=0,u_prev=np.zeros(16)),
        V_wp=array(np.zeros((3,2,16))),eps_wp=array(np.zeros((3,2,16))))
    state=SimpleNamespace(qpos=array(np.full((3,23),np.nan)),qvel=array(np.zeros((3,22))),
        site_xpos=array(np.zeros((3,4,3))))
    path=tmp_path/'failed.npz'
    details=runner.capture_failed_plan(planner,SimpleNamespace(DeviceState=lambda:state),
        np.zeros(23),np.zeros(22),np.zeros(16),np.zeros(16),path)
    assert details['finite_costs']==0 and details['nan_costs']==2 and details['positive_infinite_costs']==1
    assert details['finite_final_qpos_worlds']==0 and details['finite_final_qvel_worlds']==3
    json.dumps(details,allow_nan=False)
    with np.load(path,allow_pickle=False) as saved:
        assert np.isnan(saved['costs'][0])
        assert saved['q'].shape==(23,) and saved['V_wp'].shape==(3,2,16)
        assert json.loads(str(saved['planner_start_json']))['noise_seed']==0


def test_failed_plan_diagnostic_error_never_masks_original_failure():
    details=runner.capture_failed_plan(SimpleNamespace(),None,np.zeros(23),np.zeros(22),np.zeros(16),np.zeros(16))
    assert 'diagnostic_error' in details


@pytest.fixture(autouse=True)
def operator_phases(monkeypatch):
    # These planner tests isolate episode phases; their motion, Enter gate and
    # heartbeat behavior are exercised with measured feedback in test_leap_episodes.
    def phases(feedback, *args):
        feedback.publish_stop = lambda: None
        return SimpleNamespace(before_episode=lambda episode: None,
            wait_for_tracking=lambda episode, timeout, **limits: feedback.buffer.snapshot(**limits),
            after_success=lambda episode, final: None, after_unsuccessful=lambda *a: None,
            shutdown_after_feedback_error=lambda *a: np.zeros(16),
            return_to_zero=lambda episode: np.zeros(16), status=lambda *a: None)
    monkeypatch.setattr(runner, "HardwareEpisodes", phases)


@pytest.fixture
def scene(tmp_path):
    bodies = "".join(
        f'<body name="body{i}" pos="{i * 0.1} 0 0"><joint name="{name}" type="hinge" '
        'range="-3 3" armature="0.01"/><geom type="sphere" size="0.01" mass="0.1" '
        'contype="0" conaffinity="0"/>' +
        (f'<site name="{("if", "mf", "rf", "th")[i // 4]}_tip"/>' if i % 4 == 3 else '') +
        '</body>' for i, name in enumerate(runner.JOINTS))
    actuators = "".join(
        f'<general name="{name}_act" joint="{name}" dyntype="filterexact" '
        'dynprm="0.01" gainprm="2" biastype="affine" biasprm="0 -2 -0.1" '
        'ctrllimited="true" ctrlrange="-3 3" forcelimited="true" forcerange="-1 1"/>'
        for name in runner.JOINTS)
    source = tmp_path / "source.xml"
    source.write_text(f'<mujoco><compiler angle="radian"/>'
        f'<option timestep="0.002" gravity="0 0 0"/><worldbody>{bodies}</worldbody>'
        f'<actuator>{actuators}</actuator></mujoco>')
    model = mujoco.MjModel.from_xml_path(str(source))
    binary = tmp_path / "model.mjb"
    mujoco.mj_saveModel(model, str(binary))
    archive_path = tmp_path / "model.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(binary, "calibrated.mjb")
        archive.writestr("optimized_parameters.json", '{"metadata_only": true}')
    path = prepare_scene(archive_path, source, tmp_path / "export/cube.xml")
    return path


def legacy_hardware_config():
    # These fixtures exercise delta clipping independently of local measured
    # captures. The two-stage path is covered explicitly below.
    config = json.loads((runner.ROOT/'configs/leap_hardware.json').read_text())
    config.pop('joint_limits', None)
    config.pop('hold_on_non_motor_failure', None)
    return config


def argv(scene, output):
    camera = output.with_name('test_pose_transform.json')
    camera.write_text(json.dumps(dict(T_model_from_published=np.eye(4).tolist(),
        translation_adjust_m=[0, 0, 0], rotation_adjust_deg=[0, 0, 0],
        T_tracked_from_object=np.eye(4).tolist())))
    config = legacy_hardware_config()
    config['delta_reference'] = 'measured_position'
    config['ctrl_range'] = dict(lower=.1, upper=.1)
    hardware_config = output.with_name('test_hardware_config.json')
    hardware_config.write_text(json.dumps(config))
    return ["--model", str(scene), "--device", "cpu", "--n-samples", "8", "--horizon", "2",
            "--ctrl-time-step", "0.012", "--steps", "2", "--settle", "0", "--results", str(output),
            "--camera-config", str(camera), '--hardware-config', str(hardware_config),
            '--no-object-joint-reference']


def pose_bytes(matrix, stamp=None):
    pose = Pose()
    pose.timestamp = int(time.time() * 1e6) if stamp is None else stamp
    pose.pose_matrix = np.asarray(matrix).reshape(16).tolist()
    return pose.encode()


def feedback(model, stamp=None, before=None):
    buffer = FeedbackBuffer(model, before)
    stamp = int(time.time() * 1e6) if stamp is None else stamp
    buffer.on_robot("LEAP_STATE", RobotState(stamp, np.linspace(0, 0.3, 16), np.ones(16)).encode())
    buffer.on_pose("POSE", pose_bytes(np.eye(4), stamp))
    return buffer


def test_export_preserves_calibration_and_adds_free_cube(scene):
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert (model.nq, model.nv, model.nu, model.na) == (23, 22, 16, 16)
    hand = mujoco.MjModel.from_xml_path(str(scene.parent / "calibrated_hand.xml"))
    original = mujoco.MjModel.from_xml_path(str(scene.parent.parent / "source.xml"))
    check_calibration(hand, original)
    assert np.allclose(model.actuator_dynprm[:, 0], 0.01)
    assert model.joint("obj_joint").qposadr[0] == 16


def test_wire_formats_match_original_bridge_and_tracker():
    q, v = np.arange(16, dtype=np.float32), np.arange(16, dtype=np.float32) * 2
    raw = struct.pack("<q", 123) + q.tobytes() + v.tobytes()
    state = RobotState.decode(raw)
    assert state.encode() == raw
    np.testing.assert_array_equal(state.qvel, v)
    action = struct.pack("<q", 456) + q.tobytes()
    assert RobotAction.decode(action).encode() == action
    original = Path(__file__).resolve().parents[1] / "iris-foundationpose-plusplus/src/lcm_types/pose_types/Pose.py"
    spec = importlib.util.spec_from_file_location("tracker_pose", original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pose = module.Pose()
    pose.timestamp, pose.pose_matrix = 123, np.eye(4).reshape(16).tolist()
    assert Pose.decode(pose.encode()).timestamp == 123
    assert Pose.decode(pose.encode()).pose_matrix == tuple(pose.pose_matrix)


def test_assembled_state_contains_measured_joints_object_pose_and_velocities(scene):
    model = mujoco.MjModel.from_xml_path(str(scene))
    now = int(time.time() * 1e6)
    buffer = feedback(model, now - 50000)
    matrix = np.eye(4)
    matrix[:3, 3] = [0.01, -0.02, 0.03]
    matrix[:3, :3] = Rotation.from_euler("z", 0.1).as_matrix()
    buffer.on_pose("POSE", pose_bytes(matrix, now))
    q, v, metadata = buffer.snapshot()
    assert q.shape == (23,) and v.shape == (22,)
    np.testing.assert_allclose(q[:16], np.linspace(0, 0.3, 16), atol=1e-7)
    np.testing.assert_allclose(q[16:19], [0.01, -0.02, 0.03])
    np.testing.assert_allclose(q[19:23], [np.cos(0.05), 0, 0, np.sin(0.05)], atol=1e-7)
    np.testing.assert_allclose(v[16:19], [0.2, -0.4, 0.6], atol=1e-6)
    np.testing.assert_allclose(v[19:22], [0, 0, 2], atol=1e-6)
    assert metadata["pose_timestamp"] == now


def test_pose_adjustments_are_easy_to_change(scene, tmp_path):
    path = tmp_path / "transform.json"
    path.write_text(json.dumps(dict(T_model_from_published=np.eye(4).tolist(),
        translation_adjust_m=[0.01, 0.02, -0.03], rotation_adjust_deg=[0, 0, 90],
        T_tracked_from_cube=np.eye(4).tolist())))
    before, after, _ = load_pose_transform(path)
    model = mujoco.MjModel.from_xml_path(str(scene))
    buffer = FeedbackBuffer(model, before, after)
    stamp = int(time.time() * 1e6)
    buffer.on_robot("LEAP_STATE", RobotState(stamp).encode())
    buffer.on_pose("POSE", pose_bytes(np.eye(4), stamp))
    q, _, _ = buffer.snapshot()
    np.testing.assert_allclose(q[16:19], [0.01, 0.02, -0.03])
    np.testing.assert_allclose(q[19:23], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])


def test_stale_and_malformed_feedback_is_rejected(scene):
    model = mujoco.MjModel.from_xml_path(str(scene))
    old = feedback(model, int((time.time() - 1) * 1e6))
    with pytest.raises(RuntimeError, match="Stale"):
        old.snapshot()
    old.on_robot("LEAP_STATE", b"broken")
    with pytest.raises(RuntimeError, match="LCM feedback error"):
        old.snapshot()
    with pytest.raises(ValueError, match="72 bytes"):
        RobotAction.decode(bytes(80))


def test_live_pose_rotation_drift_is_normalized_without_changing_translation(scene):
    from ContactModelStudy.Hardware.LCM import rigid_transform, tracked_transform
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler('xyz', [.4, -.7, .2]).as_matrix() @ np.diag([1.00004, 1.00001, .99976])
    matrix[:3, 3] = [.02, .035, .085]
    with pytest.raises(ValueError, match='rigid'):
        rigid_transform(matrix)  # calibration validation remains strict
    normalized = tracked_transform(matrix)
    np.testing.assert_array_equal(normalized[:3, 3], matrix[:3, 3])
    np.testing.assert_allclose(normalized[:3, :3].T @ normalized[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(normalized[:3, :3]) == pytest.approx(1)
    model = mujoco.MjModel.from_xml_path(str(scene))
    buffer = feedback(model)
    buffer.on_pose('POSE', pose_bytes(matrix))
    q, _, _ = buffer.snapshot()
    assert buffer.pose_error is None
    np.testing.assert_allclose(q[16:19], matrix[:3, 3], atol=1e-8)
    assert np.linalg.norm(q[19:23]) == pytest.approx(1)


@pytest.mark.parametrize('kind', ['nan', 'scale', 'reflection', 'homogeneous'])
def test_invalid_live_pose_is_rejected_without_poisoning_joint_feedback_and_recovers(scene, kind):
    model = mujoco.MjModel.from_xml_path(str(scene))
    buffer = feedback(model)
    previous, received = buffer.pose, buffer.pose_received
    matrix = np.eye(4)
    if kind == 'nan': matrix[0, 0] = np.nan
    elif kind == 'scale': matrix[0, 0] = 1.02
    elif kind == 'reflection': matrix[0, 0] = -1
    else: matrix[3, 0] = .1
    buffer.on_pose('POSE', pose_bytes(matrix))
    assert buffer.pose is previous and buffer.pose_received == received
    assert buffer.pose_error is not None and buffer.error is None
    assert buffer.rejected_poses == 1
    buffer.snapshot_robot()  # reference/reset do not require an object pose
    with pytest.raises(RuntimeError, match='LCM feedback error'):
        buffer.snapshot()
    buffer.on_pose('POSE', pose_bytes(np.eye(4)))
    _, _, metadata = buffer.snapshot()
    assert buffer.pose_error is None and metadata['rejected_poses'] == 1


def test_bad_robot_packet_blocks_joints_until_new_valid_robot_feedback(scene):
    buffer = feedback(mujoco.MjModel.from_xml_path(str(scene)))
    buffer.on_robot('LEAP_STATE', b'bad')
    with pytest.raises(RuntimeError, match='LCM feedback error'):
        buffer.snapshot_robot()
    buffer.on_robot('LEAP_STATE', RobotState(int(time.time()*1e6), np.zeros(16), np.zeros(16)).encode())
    buffer.snapshot_robot()
    assert buffer.robot_error is None


def test_simulation_runs_cube_mppi_without_lcm(scene, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "LCMFeedback", lambda *a: pytest.fail("simulation connected to LCM"))
    output = tmp_path / "result.json"
    goal = tmp_path / 'goal.json'
    assert runner.main(argv(scene, output) + ["--n-episodes", "2", '--goal-status-file', str(goal)]) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "complete" and not result["action_publishing"]
    assert len(result["episodes"]) == 2
    for episode in result["episodes"]:
        assert len(episode["steps"]) == 2
        assert len(episode["q_end"]) == 23 and len(episode["q_dot_end"]) == 22
        assert all(not step["action_published"] for step in episode["steps"])
    progress = json.loads(goal.read_text())
    assert (progress['episode'], progress['current_step'], progress['total_steps']) == (1, 2, 2)


@pytest.mark.parametrize("observe", [False, True])
def test_live_loop_uses_lcm_state_and_respects_observe_only(scene, tmp_path, monkeypatch, observe):
    events = []
    q = np.r_[np.full(16, 0.2), [0.02, 0.035, 0.085, 1, 0, 0, 0]]
    def snapshot(**limits):
        stamp = int(time.time() * 1e6)
        return q.copy(), np.zeros(22), dict(robot_timestamp=stamp, pose_timestamp=stamp)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot),
        wait=lambda *a, **k: snapshot(), publish_delta=lambda u: events.append(np.array(u)),
        close=lambda: events.append("closed"))
    monkeypatch.setattr(runner, "LCMFeedback", lambda *a: connection)
    output = tmp_path / "live.json"
    extra = ["--observe-only"] if observe else []
    assert runner.main(argv(scene, output) + ["--hardware", *extra, "--max-plan-seconds", "10",
        '--goal-quat', '0', '0', '0', '1']) == 0
    result = json.loads(output.read_text())
    assert events[-1] == "closed"
    assert len(events) == (1 if observe else 3)
    assert all(step["q"] == q.tolist() for step in result["episodes"][0]["steps"])


@pytest.mark.parametrize("success_after_action", [False, True])
@pytest.mark.parametrize('padding', [False, True])
def test_hardware_success_holds_and_gates_each_episode_even_without_stop_flag(
        scene, tmp_path, monkeypatch, success_after_action, padding):
    events, plans = [], []
    if padding:
        from datetime import datetime,timezone
        offset=[0.];real=time.monotonic
        monkeypatch.setattr(runner,'time',SimpleNamespace(monotonic=lambda:real()+offset[0],
            time=time.time,sleep=time.sleep,strftime=time.strftime))
        monkeypatch.setattr(runner,'utc_now',lambda:datetime.fromtimestamp(time.time()+offset[0],timezone.utc).isoformat())
    output = tmp_path / 'success.json'
    q = np.r_[np.full(16, .2), [.02, .035, .085, 1, 0, 0, 0]]
    def snapshot(**limits):
        stamp = int(time.time() * 1e6)
        return q.copy(), np.zeros(22), dict(robot_timestamp=stamp, pose_timestamp=stamp)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot),
        wait=lambda *a, **k: snapshot(), close=lambda: events.append('closed'),
        publish_hold=lambda: events.append('hold'),
        publish_stop=lambda: events.append('torque off'),
        publish_delta=lambda delta: (events.append('delta'), q.__setitem__(slice(19, 23), [1, 0, 0, 0])))
    monkeypatch.setattr(runner, 'LCMFeedback', lambda *a: connection)
    def before(episode):
        events.append(('before', episode))
        q[19:23] = [0, 0, 0, 1] if success_after_action else [1, 0, 0, 0]
    def after(episode, final):
        events.append(('after', episode, final))
        deadline = time.monotonic() + 5
        while True:
            saved = json.loads(output.read_text()) if output.exists() else {}
            if saved.get('status') == 'success_hold' and len(saved['episodes']) == episode + 1:
                break
            assert time.monotonic() < deadline, 'Background checkpoint did not finish during the hold'
            time.sleep(.005)  # stand in for operator wait; hardware heartbeats stay in its phase loop
        assert saved['status'] == 'success_hold'
        assert saved['episodes'][-1]['finish_reason'] == 'success'
    def zero(episode):
        events.append(('zero', episode))
        return np.zeros(16)
    def hold_for(seconds,target=None,on_tick=None):
        events.append('video lead' if target is not None else 'video tail')
        on_tick();offset[0]+=seconds;on_tick()
    monkeypatch.setattr(runner, 'HardwareEpisodes', lambda feedback, *a: SimpleNamespace(
        before_episode=before, wait_for_tracking=lambda *a, **k: snapshot(), after_success=after,
        return_to_zero=zero, status=lambda *a: None,reference=np.zeros(16),hold_for=hold_for))
    def plan(self, state, *a, **k):
        plans.append(state.copy())
        self.last_plan_ok = True
        return state[:16].copy()
    monkeypatch.setattr(runner.MPPI, 'Plan', plan)
    extra=['--video-padding-seconds','1','--goal-status-file',str(tmp_path/'goal.json')] if padding else []
    runner.main(argv(scene, output) + extra + ['--hardware', '--n-episodes', '2', '--steps', '1',
        '--goal-quat', '1', '0', '0', '0', '--goal-position', '.02', '.035', '.085',
        '--no-stop-on-success', '--max-plan-seconds', '10'])
    expected=[]
    for i in range(2):
        expected += [('before',i), *(['video lead'] if padding else []),
                     *(['delta'] if success_after_action else []), 'hold',
                     *(['video tail'] if padding else []), ('after',i,i==1),('zero',i)]
    expected += ['torque off','closed']
    assert events == expected
    assert len(plans) == (3 if success_after_action else 1)  # warmup + unsucceeded episodes only
    saved = json.loads(output.read_text())
    assert saved['status'] == 'complete'
    assert all('starting_orientation' in e for e in saved['episodes'])
    if padding:
        for episode in saved['episodes']:
            assert episode['manipulation_wall_time_s']<1  # Padding is outside MPPI metrics.
            lead=datetime.fromisoformat(episode['started_at_utc'])-datetime.fromisoformat(episode['recording_started_at_utc'])
            tail=datetime.fromisoformat(episode['recording_finished_at_utc'])-datetime.fromisoformat(episode['finished_at_utc'])
            assert lead.total_seconds()>=1 and tail.total_seconds()>=1
    assert all(episode['home'] == saved['hardware_config']['robot_qref'] for episode in saved['episodes'])


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize('measured_bounds', [False, True])
def test_configured_hardware_bounds_limit_live_commands_only(scene, tmp_path, monkeypatch, live, measured_bounds):
    config = json.loads((runner.ROOT / "configs/leap_hardware.json").read_text())
    if not measured_bounds:
        config.pop('joint_limits', None)
    config["ctrl_range"] = dict(lower=0.1, upper=0.1)
    config['delta_reference'] = 'measured_position'
    config.update(max_command_speed=1e-12, max_tracking_error=1e-12, max_velocity=1e-12)
    config_path = tmp_path / "hardware.json"
    config_path.write_text(json.dumps(config))
    published = []
    q = np.r_[np.full(16, 0.2), [0.02, 0.035, 0.085, 1, 0, 0, 0]]
    def snapshot(**limits):
        stamp = int(time.time() * 1e6)
        return q.copy(), np.zeros(22), dict(robot_timestamp=stamp, pose_timestamp=stamp)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot), wait=lambda *a, **k: snapshot(),
        publish_delta=lambda u: published.append(u.copy()), close=lambda: None)
    monkeypatch.setattr(runner, "LCMFeedback", lambda *a: connection)
    def plan(self, *a, **k):
        self.last_plan_ok = True
        return np.full(16, 2.0)
    monkeypatch.setattr(runner.MPPI, "Plan", plan)
    result_path = tmp_path / "bounded.json"
    extra = ["--hardware", "--max-plan-seconds", "10"] if live else []
    assert runner.main(argv(scene, result_path) + ["--hardware-config", str(config_path), *extra,
        '--goal-quat', '0', '0', '0', '1']) == 0
    result = json.loads(result_path.read_text())
    if live:
        assert published
        np.testing.assert_allclose(published, 1.8)  # raw delta crosses LCM; only the final target is bounded
        from ContactModelStudy.Hardware.JointLimits import load_joint_limits, clip_absolute_target
        measured = load_joint_limits(config)
        expected = clip_absolute_target(np.full(16, .3), measured)
        np.testing.assert_allclose(result["episodes"][0]["steps"][0]["u"], expected)
        if measured_bounds:
            assert result['planner_config']['joint_limit_weight'] == measured.weight
            np.testing.assert_allclose(result['planner_config']['joint_limit_lower'], measured.lower)
            clipping = result['episodes'][0]['steps'][0]['joint_limit_clipping']
            np.testing.assert_allclose(clipping['delta_clipped_target'], .3)
            assert clipping['estimated_clipped_joints'] == int(np.count_nonzero(abs(expected-.3)>1e-7))
        np.testing.assert_allclose(result["effective_ctrl_range"]["lower"], 0.1)
        np.testing.assert_allclose(result["effective_ctrl_range"]["upper"], 0.1)
        assert result["effective_ctrl_range"]["reference"] == "current_joint_position"
        assert result['effective_ctrl_range']['application'] == 'final_hardware_target'
        assert result['planner_config']['delta_range'] == [None, None]
        np.testing.assert_allclose(result['episodes'][0]['steps'][0]['requested_target'], 2)
    else:
        assert not published and result["effective_ctrl_range"] is None
        assert max(result["model_ctrl_range"]["upper"]) > 0.1


def test_ctrl_c_stops_planning_returns_zero_then_stops_bridge_and_closes_lcm(scene, tmp_path, monkeypatch):
    events = []
    q = np.r_[np.full(16, 0.2), [0.02, 0.035, 0.085, 1, 0, 0, 0]]
    def snapshot(**limits):
        stamp = int(time.time() * 1e6)
        return q, np.zeros(22), dict(robot_timestamp=stamp, pose_timestamp=stamp)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot),
        wait=lambda *a, **k: snapshot(), publish_delta=lambda u: events.append("action"),
        close=lambda: events.append("closed"))
    monkeypatch.setattr(runner, "LCMFeedback", lambda *a: connection)
    def shutdown(episode):
        events.extend(['zero', 'torque off'])
        return np.zeros(16)
    monkeypatch.setattr(runner, 'HardwareEpisodes', lambda *a: SimpleNamespace(
        before_episode=lambda episode: None, wait_for_tracking=lambda *a, **k: snapshot(),
        shutdown_on_interrupt=shutdown))
    original, calls = runner.MPPI.Plan, []
    def interrupted(self, *a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return original(self, *a, **k)
    monkeypatch.setattr(runner.MPPI, "Plan", interrupted)
    output = tmp_path / "partial.json"
    assert runner.main(argv(scene, output) + ["--hardware", "--max-plan-seconds", "10",
        '--goal-quat', '0', '0', '0', '1']) == 0
    log = json.loads(output.read_text())
    assert log["status"] == "interrupted" and len(log["episodes"][0]["steps"]) == 1
    assert events == ["action", "zero", "torque off", "closed"]
    assert log['interrupt_return']['zero_reached']


@pytest.mark.parametrize('timing_error', [True, False])
def test_feedback_timing_fault_uses_supervised_reset_but_robot_fault_does_not(
        scene, tmp_path, monkeypatch, timing_error):
    from ContactModelStudy.Hardware.LCM import FeedbackTimingError
    events, reads = [], []
    q = np.r_[np.full(16, .2), [.02, .035, .085, 1, 0, 0, 0]]
    error = FeedbackTimingError('camera synchronization deadline') if timing_error else RuntimeError('Stale robot feedback')
    def snapshot(**limits):
        reads.append(1)
        if len(reads) == 5:
            raise error
        stamp = int(time.time()*1e6)
        return q.copy(), np.zeros(22), dict(robot_timestamp=stamp, pose_timestamp=stamp)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot), wait=lambda *a, **k: snapshot(),
        publish_delta=lambda delta: events.append('action'), close=lambda: events.append('closed'))
    monkeypatch.setattr(runner, 'LCMFeedback', lambda *a: connection)
    def reset(episode, exc):
        assert exc is error
        np.testing.assert_allclose(phases.last_target, .25)
        events.extend(['operator hold', 'enter', 'zero', 'torque off'])
        return np.zeros(16)
    phases = SimpleNamespace(before_episode=lambda episode: None,
        wait_for_tracking=lambda *a, **k: snapshot(), shutdown_after_feedback_error=reset)
    monkeypatch.setattr(runner, 'HardwareEpisodes', lambda *a: phases)
    def plan(self, *a, **k):
        self.last_plan_ok = True
        return q[:16]+.05
    monkeypatch.setattr(runner.MPPI, 'Plan', plan)
    output = tmp_path/'feedback-fault.json'
    with pytest.raises(type(error), match=str(error)):
        runner.main(argv(scene, output)+['--hardware', '--goal-quat', '0', '0', '0', '1'])
    log = json.loads(output.read_text())
    assert log['status'] == 'failed'
    assert not log['episodes'][0]['completed'] and log['episodes'][0]['n_steps'] == 1
    assert log['episodes'][0]['finish_reason'] == 'error'
    if timing_error:
        assert events == ['action', 'operator hold', 'enter', 'zero', 'torque off', 'closed']
        assert log['feedback_error_return']['zero_reached']
    else:
        assert events == ['action', 'closed']
        assert 'feedback_error_return' not in log


def test_control_pacing_is_opt_in_and_hardware_only():
    assert runner.parseArgs(['--device', 'cpu']).pace_control is False
    assert runner.parseArgs(['--hardware', '--device', 'cpu', '--pace-control']).pace_control is True
    assert runner.parseArgs(['--hardware', '--device', 'cpu', '--pace-control', '--no-pace-control']).pace_control is False
    with pytest.raises(SystemExit):
        runner.parseArgs(['--device', 'cpu', '--pace-control'])


@pytest.mark.parametrize('alpha,metrics', [(1.,False),(1.,True),(.6,False)])
def test_temporal_blend_changes_only_published_delta_and_keeps_bridge_clipping(
        scene, tmp_path, monkeypatch, alpha, metrics):
    class Clock:
        elapsed = 0.
        strftime = staticmethod(time.strftime)
        def monotonic(self): return self.elapsed
        def time(self): return 1000.+self.elapsed
    clock = Clock()
    monkeypatch.setattr(runner, 'time', clock)
    calls, publications, means = [], [], []
    def snapshot(**limits):
        stamp = int(clock.time()*1e6)
        q = np.r_[np.full(16,.2+clock.elapsed),[.02,.035,.085,1,0,0,0]]
        return q, np.zeros(22), dict(robot_timestamp=stamp, pose_timestamp=stamp)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot),
        wait=lambda *a,**k:snapshot(), publish_delta=lambda d:publications.append(d.copy()), close=lambda:None)
    monkeypatch.setattr(runner,'LCMFeedback',lambda *a:connection)
    def plan(self,q,*a,**k):
        calls.append(1)
        self.last_plan_ok = True
        self.last_mean = np.repeat(np.array([.3,.4] if len(calls)<=2 else [-.3,-.2])[:,None],16,axis=1)
        means.append(self.last_mean.copy())
        if len(calls)>1: clock.elapsed += .074
        return q[:16] + self.last_mean[0]
    monkeypatch.setattr(runner.MPPI,'Plan',plan)
    # Alpha=1 without metrics must never access the optional implementation.
    if alpha == 1 and not metrics:
        monkeypatch.setattr(runner,'TemporalPlanBlend',lambda *a:pytest.fail('disabled blend constructed'))
    output=tmp_path/'blend.json'
    flags=['--plan-blend-metrics'] if metrics else []
    assert runner.main(argv(scene,output)+['--hardware','--ctrl-time-step','.064',
        '--plan-blend-alpha',str(alpha),'--goal-quat','0','0','0','1',*flags]) == 0
    assert len(calls)==3  # Warmup and exactly two MPPI plans; no extra rollouts.
    np.testing.assert_allclose(publications[0],.3)
    np.testing.assert_allclose(publications[1],-.3 if alpha==1 else .6*(-.3)+.4*.4)
    np.testing.assert_array_equal(means[-1],np.repeat(np.array([-.3,-.2])[:,None],16,axis=1))
    saved=json.loads(output.read_text()); steps=saved['episodes'][0]['steps']
    for s,published in zip(steps,publications):
        np.testing.assert_allclose(s['action_delta'],published)
        base=np.asarray(s['feedback']['command_qpos'])
        np.testing.assert_allclose(s['u'],base+np.clip(published,-.1,.1))
    if alpha<1 or metrics:
        info=steps[1]['plan_blending']
        assert info['old_slot']==1 and info['status']=='overlap'
        assert info['estimated_unblended_clipped_joints']==16
        assert info['estimated_blended_clipped_joints']==(16 if alpha==1 else 0)
        np.testing.assert_allclose(info['aligned_plan_diff_l2'],[4*.7])
    else:
        assert 'plan_blending' not in steps[0]


@pytest.mark.parametrize('paced', [False, True])
def test_control_pacing_caps_publications_without_catchup_and_rebases_after_wait(
        scene, tmp_path, monkeypatch, paced):
    class Clock:
        elapsed = 0.
        sleeps = []
        strftime = staticmethod(time.strftime)
        def monotonic(self): return self.elapsed
        def time(self): return 1000.+self.elapsed
        def sleep(self, delay):
            self.sleeps.append(delay)
            self.elapsed += delay
    clock = Clock()
    monkeypatch.setattr(runner, 'time', clock)
    publications, calls = [], []
    def snapshot(**limits):
        q = np.r_[np.full(16, .2+clock.elapsed), [.02, .035, .085, 1, 0, 0, 0]]
        stamp = int(clock.time()*1e6)
        return q, np.zeros(22), dict(robot_timestamp=stamp, pose_timestamp=stamp)
    def publish(delta):
        publications.append((clock.elapsed, delta.copy()))
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot),
        wait=lambda *a, **k: snapshot(), publish_delta=publish, close=lambda: None)
    monkeypatch.setattr(runner, 'LCMFeedback', lambda *a: connection)
    def plan(self, q, *a, **k):
        self.last_plan_ok = True
        calls.append(1)
        if len(calls)>1:  # Warmup is outside the manipulation loop.
            clock.elapsed += [.020, .090, .010][len(calls)-2]
        return q[:16]+.05
    monkeypatch.setattr(runner.MPPI, 'Plan', plan)
    output = tmp_path/'pacing.json'
    flags = ['--pace-control'] if paced else []
    assert runner.main(argv(scene, output)+['--hardware', '--steps', '3', '--ctrl-time-step', '.064',
        '--goal-quat', '0', '0', '0', '1', *flags]) == 0
    expected = [.064, .154, .218] if paced else [.020, .110, .120]
    np.testing.assert_allclose([stamp for stamp, _ in publications], expected)
    if paced:
        # The 90 ms overrun receives no extra sleep; the next fast plan starts
        # a new 64 ms interval instead of catching up to an old schedule.
        np.testing.assert_allclose(clock.sleeps, [.044, .054])
        assert np.min(np.diff([stamp for stamp, _ in publications])) >= .064-1e-9
    else:
        assert not clock.sleeps
    data = json.loads(output.read_text())
    assert data['hardware_loop_paced'] is paced
    assert data['execution']['hardware_loop_paced'] is paced
    assert data['execution']['control_frequency_limit_hz'] == (15.625 if paced else None)
    steps = data['episodes'][0]['steps']
    np.testing.assert_allclose([s['planning_time'] for s in steps], [.020, .090, .010])
    np.testing.assert_allclose([s['pacing_wait_s'] for s in steps], [.044, 0, .054] if paced else [0, 0, 0])
    for (stamp, delta), step in zip(publications, steps):
        np.testing.assert_allclose(delta, .05)
        # In particular, the measured command base comes after any pacing wait.
        np.testing.assert_allclose(step['feedback']['command_qpos'], .2+stamp)
        assert step['action_published_at_utc'] is not None
    import csv
    rows = list(csv.DictReader((output.parent/'steps.csv').open()))
    assert float(rows[0]['pacing_wait_s']) == pytest.approx(.044 if paced else 0)
    assert rows[0]['action_published_at_utc']


@pytest.mark.parametrize('bound', [.001, .05, .1])
def test_controller_publishes_delta_without_pacing_or_velocity_caps(scene, tmp_path, monkeypatch, bound):
    config = legacy_hardware_config()
    config.update(max_command_speed=1e-12, max_tracking_error=1e-12, max_velocity=1e-12,
                  ctrl_range=dict(lower=bound, upper=bound), delta_reference='measured_position')
    path = tmp_path / "hardware.json"
    path.write_text(json.dumps(config))
    measured = np.r_[np.full(16, 0.2), [0.02, 0.035, 0.085, 1, 0, 0, 0]]
    proposed_delta = np.linspace(-0.2, 0.2, 16)
    published, calls = [], []
    def snapshot(**limits):
        stamp = int(time.time() * 1e6)
        return measured.copy(), np.r_[np.full(16, 100), np.zeros(6)], dict(robot_timestamp=stamp, pose_timestamp=stamp)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot), wait=lambda *a, **k: snapshot(),
        publish_delta=lambda u: published.append(u.copy()), close=lambda: None)
    monkeypatch.setattr(runner, "LCMFeedback", lambda *a: connection)
    def plan(self, q, *a, **k):
        self.last_plan_ok = True
        calls.append(q.copy())
        command = q[:16] + proposed_delta
        if len(calls) > 1:  # first call compiles/warms the planner before LCM
            measured[:16] += 0.04
        return command
    monkeypatch.setattr(runner.MPPI, "Plan", plan)
    monkeypatch.setattr(runner.time, "sleep", lambda _: pytest.fail("Controller must not pace hardware actions"))
    output = tmp_path / "rebase.json"
    runner.main(argv(scene, output) + ["--hardware", "--hardware-config", str(path),
                "--goal-quat", "0", "0", "0", "1", "--max-plan-seconds", "10"])
    np.testing.assert_allclose(published[0], proposed_delta)
    np.testing.assert_allclose(published[1], proposed_delta)
    steps = json.loads(output.read_text())["episodes"][0]["steps"]
    np.testing.assert_allclose(steps[0]["q"][:16], 0.2)
    np.testing.assert_allclose(steps[0]["feedback"]["command_qpos"], 0.24)
    np.testing.assert_allclose(steps[0]['requested_target'], .24 + proposed_delta)
    np.testing.assert_allclose(steps[1]['requested_target'], .28 + proposed_delta)
    np.testing.assert_allclose(steps[0]["u"], np.clip(.24 + proposed_delta, .24-bound, .24+bound))
    np.testing.assert_allclose(steps[1]["u"], np.clip(.28 + proposed_delta, .28-bound, .28+bound))
    assert json.loads(output.read_text())['planner_config']['delta_range'] == [None, None]
    assert not json.loads(output.read_text())["hardware_loop_paced"]


def test_previous_target_mode_plans_from_bridge_target_and_recovers_dropped_commands(scene, tmp_path, monkeypatch):
    config = legacy_hardware_config()
    config.update(delta_reference='previous_target', ctrl_range=dict(lower=.01, upper=.01))
    path = tmp_path/'previous_target.json'
    path.write_text(json.dumps(config))
    q = np.r_[np.full(16, .7), [.02, .035, .085, 0, 0, 0, 1]]
    applied = np.ones(16)
    plans, published = [], []
    def snapshot(**limits):
        stamp = int(time.time()*1e6)
        return q.copy(), np.zeros(22), dict(robot_timestamp=stamp, pose_timestamp=stamp)
    def target(max_age):
        return applied.copy(), dict(target_timestamp=int(time.time()*1e6), target_age=0)
    def publish(delta):
        published.append(delta.copy())
        if len(published) > 1:  # first packet is dropped before the bridge applies it
            applied[:] += np.clip(delta, -.01, .01)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot, snapshot_target=target),
        wait=lambda *a, **k: snapshot(), publish_delta=publish, close=lambda: None)
    def connect(*a, **kw):
        assert kw['target_feedback'] is True
        return connection
    monkeypatch.setattr(runner, 'LCMFeedback', connect)
    def plan(self, state, velocity, u=None):
        assert self.config.control_mode == 'ctrl_relative'
        plans.append(u.copy())
        self.last_plan_ok = True
        return u + .02
    monkeypatch.setattr(runner.MPPI, 'Plan', plan)
    output = tmp_path/'target_relative.json'
    assert runner.main(argv(scene, output)+['--hardware', '--hardware-config', str(path),
        '--steps', '3', '--max-plan-seconds', '10']) == 0
    np.testing.assert_allclose(plans[1:], np.array([1, 1, 1.01])[:,None]*np.ones((1,16)))
    np.testing.assert_allclose(published, .02)
    saved = json.loads(output.read_text())
    assert saved['args']['control_mode'] == saved['planner_config']['control_mode'] == 'ctrl_relative'
    assert saved['effective_ctrl_range']['reference'] == 'previous_target'
    assert saved['control_policy'] == 'previous_target_plus_bounded_delta'
    steps = saved['episodes'][0]['steps']
    np.testing.assert_allclose([row['u'] for row in steps], np.array([1.01, 1.01, 1.02])[:,None]*np.ones((1,16)))
    np.testing.assert_allclose([row['plan_base'] for row in steps], plans[1:])
    np.testing.assert_allclose([row['q'][:16] for row in steps], .7)


def test_fingertip_preview_receives_weighted_mean_and_plan_start_state_without_extra_plans(scene, tmp_path, monkeypatch):
    q = np.r_[np.full(16, .2), [.02, .035, .085, 0, 0, 0, 1]]
    events, submissions, means = [], [], []
    def snapshot(**limits):
        stamp = int(time.time()*1e6)
        return q.copy(), np.zeros(22), dict(robot_timestamp=stamp, pose_timestamp=stamp)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot), wait=lambda *a, **k: snapshot(),
        publish_delta=lambda delta: events.append('action'), close=lambda: events.append('feedback closed'))
    monkeypatch.setattr(runner, 'LCMFeedback', lambda *a: connection)
    def plan(self, state, velocity, u=None):
        self.last_mean = np.full((self.horizon,16), .03)
        means.append(self.last_mean.copy())
        self.last_plan_ok = True
        return state[:16]+.03
    monkeypatch.setattr(runner.MPPI, 'Plan', plan)
    class Preview:
        device = 'cpu'
        def __init__(self, args, task, lower, upper, path):
            assert path == tmp_path/'fingertips.json'
        def start(self, mean, *state):
            np.testing.assert_array_equal(mean, means[0])
            events.append('preview warmed')
        def clear(self): events.append('clear')
        def submit(self, mean, positions, velocity, control, activation, stamp, episode, step):
            assert events[-1]=='action'  # publishing the hardware action is never delayed by prediction
            submissions.append((mean.copy(), positions.copy(), stamp, episode, step))
        def close(self): events.append('preview closed')
    monkeypatch.setattr(runner, 'FingertipPrediction', Preview)
    output = tmp_path/'preview_run.json'
    assert runner.main(argv(scene, output)+['--hardware', '--fingertip-path-file', str(tmp_path/'fingertips.json'),
        '--max-plan-seconds', '10']) == 0
    assert len(means)==3 and len(submissions)==2  # warmup + two ordinary MPPI plans
    for index, (mean, positions, stamp, episode, step) in enumerate(submissions):
        np.testing.assert_array_equal(mean, means[index+1])
        np.testing.assert_array_equal(positions, q)
        assert stamp>0 and episode==0 and step==index
    assert events[-2:] == ['feedback closed', 'preview closed']


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_object_frame_composition_handles_half_turns_and_old_packets(scene, axis):
    model = mujoco.MjModel.from_xml_path(str(scene))
    before, after = np.eye(4), np.eye(4)
    before[:3, :3] = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    before[:3, 3] = [0.01, -0.02, 0.03]
    after[:3, 3] = [0.003, 0.004, -0.005]
    tracked = np.eye(4)
    tracked[:3, :3] = Rotation.from_euler(axis, 180, degrees=True).as_matrix()
    tracked[:3, 3] = [0.02, 0.03, 0.08]
    buffer = FeedbackBuffer(model, before, after)
    now = int(time.time() * 1e6)
    q_hand, v_hand = np.linspace(-0.2, 1.2, 16), np.linspace(-1, 1, 16)
    buffer.on_robot("LEAP_STATE", RobotState(now, q_hand, v_hand).encode())
    buffer.on_pose("POSE", pose_bytes(tracked, now))
    buffer.on_pose("POSE", pose_bytes(np.eye(4), now - 1000))
    buffer.on_robot("LEAP_STATE", RobotState(now - 1000, np.zeros(16), np.zeros(16)).encode())
    q, v, _ = buffer.snapshot()
    expected = before @ tracked @ after
    np.testing.assert_allclose(q[:16], q_hand, atol=1e-7)
    np.testing.assert_allclose(v[:16], v_hand, atol=1e-7)
    np.testing.assert_allclose(q[16:19], expected[:3, 3], atol=1e-7)
    actual_rotation = Rotation.from_quat(q[[20, 21, 22, 19]]).as_matrix()
    np.testing.assert_allclose(actual_rotation, expected[:3, :3], atol=1e-7)


def test_servo_state_resets_between_cube_rollouts(scene):
    cfg = VectorizedMujocoConfig(device="cpu", horizon=2, substeps=2)
    sim = CpuRollouts(scene, cfg, 2, np.full(16, -3), np.full(16, 3))
    sim.activation.assign(np.full(16, 0.2, dtype=np.float32))
    q = wp.array(sim.mjm.qpos0.astype(np.float32), device="cpu")
    sim.BroadcastState(q)
    sim.SetControl(np.full(16, 0.3))
    sim.Step_GPU(2)
    first = sim.GetState()[0]
    sim.BroadcastState(q)
    assert np.allclose(sim.data[0].act, 0.2)
    sim.SetControl(np.full(16, 0.3))
    sim.Step_GPU(2)
    np.testing.assert_array_equal(first, sim.GetState()[0])
    assert sim.DeviceState().site_xpos.dtype == wp.vec3


def test_fault_sigterm_does_not_request_a_zero_return(scene, tmp_path, monkeypatch):
    import signal
    events = []
    q = np.r_[np.full(16, .2), [.02, .035, .085, 1, 0, 0, 0]]
    def snapshot(**limits):
        stamp = int(time.time()*1e6)
        return q.copy(), np.zeros(22), dict(robot_timestamp=stamp, pose_timestamp=stamp)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot), wait=lambda *a, **k: snapshot(),
                                 close=lambda: events.append('closed'))
    monkeypatch.setattr(runner, 'LCMFeedback', lambda *a: connection)
    calls = []
    def plan(self, *args, **kwargs):
        self.last_plan_ok = True
        calls.append(1)
        if len(calls) > 1:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return q[:16].copy()
    monkeypatch.setattr(runner.MPPI, 'Plan', plan)
    monkeypatch.setattr(runner, 'HardwareEpisodes', lambda *a: SimpleNamespace(
        before_episode=lambda episode: None, wait_for_tracking=lambda *a, **k: snapshot(),
        shutdown_on_interrupt=lambda episode: pytest.fail('Fault must not command reset motion')))
    output = tmp_path/'fault.json'
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as error:
        runner.main(argv(scene, output)+['--hardware', '--goal-quat', '0', '0', '0', '1'])
    assert error.value.code == 128+signal.SIGTERM
    assert events == ['closed']
    assert signal.getsignal(signal.SIGTERM) == previous


def test_planner_exception_uses_supervised_software_hold_when_enabled(scene,tmp_path,monkeypatch):
    output=tmp_path/'software-fault.json'
    flags=argv(scene,output)
    config_path=Path(flags[flags.index('--hardware-config')+1])
    config=json.loads(config_path.read_text());config['hold_on_non_motor_failure']=True
    config_path.write_text(json.dumps(config))
    q=np.r_[np.full(16,.2),[.02,.035,.085,1,0,0,0]]
    events=[]
    def snapshot(**limits):
        stamp=int(time.time()*1e6)
        return q.copy(),np.zeros(22),dict(robot_timestamp=stamp,pose_timestamp=stamp)
    feedback=SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot),wait=lambda *a,**k:snapshot(),
                             close=lambda:events.append('closed'))
    monkeypatch.setattr(runner,'LCMFeedback',lambda *a:feedback)
    phases=SimpleNamespace(before_episode=lambda e:None,wait_for_tracking=lambda *a,**k:snapshot(),
        shutdown_after_error=lambda e,error:events.append(('recovery',str(error))) or np.zeros(16))
    monkeypatch.setattr(runner,'HardwareEpisodes',lambda *a:phases)
    calls=[]
    def plan(self,*args,**kwargs):
        self.last_plan_ok=True;calls.append(1)
        if len(calls)>1:raise RuntimeError('synthetic planner failure')
        return q[:16]
    monkeypatch.setattr(runner.MPPI,'Plan',plan)
    with pytest.raises(RuntimeError,match='synthetic planner failure'):
        runner.main(flags+['--hardware','--goal-quat','0','0','0','1'])
    assert events==[('recovery','synthetic planner failure'),'closed']
    record=json.loads(output.read_text())
    assert record['software_error_return']['zero_reached']
    assert record['status']=='failed' and record['summary']['aborted_episodes']==1


def test_goal_published_before_enter_and_updated_for_measured_target(scene, tmp_path, monkeypatch):
    path = tmp_path/'goal.json'
    q = np.r_[np.full(16, .2), [.03, .045, .11, 0, 0, 0, 1]]
    gates, deltas = [], []
    def snapshot(**limits):
        stamp = int(time.time()*1e6)
        return q.copy(), np.zeros(22), dict(robot_timestamp=stamp, pose_timestamp=stamp)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot), wait=lambda *a, **k: snapshot(),
        publish_delta=lambda delta: deltas.append(delta.copy()), publish_stop=lambda: None, close=lambda: None)
    monkeypatch.setattr(runner, 'LCMFeedback', lambda *a: connection)
    def gate(episode):
        gates.append(json.loads(path.read_text()))
    monkeypatch.setattr(runner, 'HardwareEpisodes', lambda *a: SimpleNamespace(
        before_episode=gate, wait_for_tracking=lambda *a, **k: snapshot(), after_unsuccessful=lambda *a: None,
        return_to_zero=lambda episode: np.zeros(16), status=lambda *a: None))
    def plan(self, state, *a, **k):
        self.last_plan_ok = True
        return state[:16].copy()
    monkeypatch.setattr(runner.MPPI, 'Plan', plan)
    output = tmp_path/'live_goal.json'
    runner.main(argv(scene, output)+['--hardware', '--measured-reference', '--steps', '1',
        '--goal-quat', '1', '0', '0', '0', '--goal-status-file', str(path), '--max-plan-seconds', '10'])
    assert len(gates) == 1
    assert (gates[0]['current_step'], gates[0]['total_steps']) == (0, 1)
    np.testing.assert_allclose(gates[0]['target_quat'], [1, 0, 0, 0])
    saved = json.loads(path.read_text())
    assert (saved['current_step'], saved['total_steps']) == (1, 1)
    np.testing.assert_allclose(saved['target_pos'], q[16:19])
    log = json.loads(output.read_text())
    np.testing.assert_allclose(saved['target_pos'], log['episodes'][0]['target_pos'])
    np.testing.assert_allclose(saved['target_quat'], log['episodes'][0]['goal'])


@pytest.mark.parametrize('override', [None, [0, 0, 1, 0]])
def test_hardware_keeps_one_goal_across_episodes_with_different_measured_starts(scene, tmp_path, monkeypatch, override):
    output, goal_file = tmp_path/'fixed.json', tmp_path/'goal.json'
    q = np.r_[np.full(16, .2), [.02, .035, .085, 0, 0, 0, 1]]
    starts = [[0, 0, 0, 1], [0, 1, 0, 0]]
    expected = [1, 0, 0, 0] if override is None else override
    gates, plans = [], []
    def snapshot(**limits):
        stamp = int(time.time()*1e6)
        return q.copy(), np.zeros(22), dict(robot_timestamp=stamp, pose_timestamp=stamp)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot), wait=lambda *a, **k: snapshot(),
        publish_delta=lambda delta: None, publish_stop=lambda: None, close=lambda: None)
    monkeypatch.setattr(runner, 'LCMFeedback', lambda *a: connection)
    def gate(episode):
        gates.append(json.loads(goal_file.read_text()))
        q[19:23] = starts[episode]  # operator changes placement before Enter
    monkeypatch.setattr(runner, 'HardwareEpisodes', lambda *a: SimpleNamespace(
        before_episode=gate, wait_for_tracking=lambda *a, **k: snapshot(), after_unsuccessful=lambda *a: None,
        return_to_zero=lambda episode: np.zeros(16), status=lambda *a: None))
    monkeypatch.setattr(runner.reference.TASKS['cube_reorient'], 'sampleNewGoal',
                        lambda *a: pytest.fail('Hardware must not sample a new target orientation'))
    def plan(self, state, *a, **k):
        self.last_plan_ok = True
        plans.append(state.copy())
        return state[:16].copy()
    monkeypatch.setattr(runner.MPPI, 'Plan', plan)
    extra = [] if override is None else ['--goal-quat', *map(str, override)]
    assert runner.main(argv(scene, output)+['--hardware', '--n-episodes', '2', '--steps', '1',
        '--goal-status-file', str(goal_file), '--max-plan-seconds', '10', *extra]) == 0
    saved = json.loads(output.read_text())
    assert saved['goal_policy'] == 'fixed'
    np.testing.assert_allclose(saved['fixed_goal_quat'], expected)
    assert len(gates) == 2 and len(saved['episodes']) == 2
    for index, episode in enumerate(saved['episodes']):
        np.testing.assert_allclose(episode['goal'], expected)
        np.testing.assert_allclose(gates[index]['target_quat'], expected)
        np.testing.assert_allclose(episode['steps'][0]['q'][19:23], starts[index])
        np.testing.assert_allclose(plans[index+1][19:23], starts[index])
    np.testing.assert_allclose(json.loads(goal_file.read_text())['target_quat'], expected)


@pytest.mark.parametrize('action_rate_weight', [0.,1.])
def test_logging_exports_metrics_parameters_and_actual_timings(scene, tmp_path, monkeypatch, action_rate_weight):
    import csv
    output = tmp_path/'logged'/'controller.json'
    # Build args before making the nested output directory.
    output.parent.mkdir()
    command = argv(scene, output) + ['--n-episodes', '2', '--logging',
                                    '--action-rate-weight',str(action_rate_weight)]
    assert runner.main(command) == 0
    data = json.loads(output.read_text())
    summary = json.loads((output.parent/'summary.json').read_text())
    assert data['argv'] == command
    assert data['wall_time_s'] > 0 and data['warmup_wall_time_s'] > 0
    assert data['execution']['hardware_loop_paced'] is False
    assert data['planner_config']['temperature'] == .1
    assert data['planner_config']['action_rate_weight'] == action_rate_weight
    assert data['action_rate_cost']['weight'] == action_rate_weight
    assert data['scenes']['rollout']['task_config']['hand_acc'] == 'high'
    assert '<mujoco' in data['scenes']['rollout']['scene_xml']
    assert summary['requested_episodes'] == summary['completed_episodes'] == 2
    assert summary['aborted_episodes'] == 0
    assert summary['mppi_time']['count'] == 4
    with (output.parent/'steps.csv').open() as stream:
        steps = list(csv.DictReader(stream))
    with (output.parent/'episodes.csv').open() as stream:
        episodes = list(csv.DictReader(stream))
    assert len(steps) == 4 and len(episodes) == 2
    for row in steps:
        assert float(row['position_error_m']) >= 0
        assert 0 <= float(row['quaternion_error_rad']) <= np.pi
        assert float(row['wall_time_s']) >= float(row['planning_time_s']) > 0
    for episode in data['episodes']:
        for step in episode['steps']:
            if action_rate_weight:
                previous = np.array(step['action_rate_reference'])
                assert step['planned_action_rate_cost'] == pytest.approx(
                    action_rate_weight*np.sum((np.array(step['planned_u'])-previous)**2))
                assert step['estimated_action_rate_cost'] == pytest.approx(
                    action_rate_weight*np.sum((np.array(step['u'])-previous)**2))
            else:
                assert 'action_rate_reference' not in step
        assert episode['completed']
        assert episode['cycle_wall_time_s'] >= episode['manipulation_wall_time_s'] > 0
        assert episode['actual_control_frequency_hz'] == pytest.approx(2/episode['manipulation_wall_time_s'])
        assert episode['mppi_time']['count'] == 2


def test_no_logging_does_not_start_writer_or_save_metrics(scene, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'RunLogger', lambda *a: pytest.fail('Disabled logging created a writer'))
    output = tmp_path/'disabled.json'
    assert runner.main(argv(scene, output)+['--no-logging']) == 0
    assert not output.exists()
    assert not (tmp_path/'steps.csv').exists()
    assert not (tmp_path/'episodes.csv').exists()
    assert not (tmp_path/'summary.json').exists()


def test_rejected_plan_logs_failed_attempt_and_exact_freshness_values(scene, tmp_path, monkeypatch):
    import csv
    q = np.r_[np.full(16, .2), [.02, .035, .085, 1, 0, 0, 0]]
    def snapshot(**limits):
        return q.copy(), np.zeros(22), dict(robot_timestamp=int((time.time()-.14)*1e6),
            pose_timestamp=int(time.time()*1e6), robot_age=.14, pose_age=0)
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot), wait=lambda *a, **k: snapshot(),
        close=lambda: None, publish_delta=lambda *a: pytest.fail('Rejected plan published an action'))
    monkeypatch.setattr(runner, 'LCMFeedback', lambda *a: connection)
    def plan(self, state, *a, **k):
        self.last_plan_ok = True
        time.sleep(.025)
        return state[:16].copy()
    monkeypatch.setattr(runner.MPPI, 'Plan', plan)
    output = tmp_path/'stale.json'
    with pytest.raises(RuntimeError, match='Planned state is stale'):
        runner.main(argv(scene, output)+['--hardware', '--goal-quat', '0', '0', '0', '1'])
    data = json.loads(output.read_text())
    episode = data['episodes'][0]
    assert episode['finish_reason'] == 'error' and not episode['completed']
    assert episode['n_steps'] == 0
    failure = episode['failed_step']
    assert failure['planning_time'] >= .025
    assert failure['rejection']['robot_age_s'] > failure['rejection']['max_robot_age']
    assert data['summary']['aborted_episodes'] == 1
    with (tmp_path/'steps.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]['action_published'] == 'False'
    assert 'Planned state is stale' in rows[0]['error']


def test_transient_skew_waits_before_planning_and_does_not_delay_command_rebase(scene, tmp_path, monkeypatch):
    from ContactModelStudy.Hardware.LCM import FeedbackSkewError
    q = np.r_[np.full(16,.2), [.02,.035,.085,0,0,0,1]]
    plans, actions = [], []
    pending_skew = [0]
    command_context = [False]
    def snapshot(**limits):
        stamp = int(time.time()*1e6)
        command_context[0] = not limits.get('require_sync',True)
        if pending_skew[0] and limits.get('require_sync',True):
            pending_skew[0] -= 1
            raise FeedbackSkewError(dict(robot_timestamp=stamp,pose_timestamp=stamp-160000,
                robot_age=0.,pose_age=.16),limits['max_skew'])
        return q.copy(), np.zeros(22), dict(robot_timestamp=stamp,pose_timestamp=stamp)
    def plan(self, state, *a, **kw):
        plans.append(1)
        self.last_plan_ok = True
        if len(plans)>1:
            pending_skew[0] = 2
        return state[:16]+.02
    def publish(delta):
        assert command_context[0]
        actions.append(delta.copy())
    connection = SimpleNamespace(buffer=SimpleNamespace(snapshot=snapshot),wait=lambda *a,**k:snapshot(),
        close=lambda:None,publish_delta=publish)
    monkeypatch.setattr(runner,'LCMFeedback',lambda *a:connection)
    monkeypatch.setattr(runner.MPPI,'Plan',plan)
    output = tmp_path/'sync_recovery.json'
    assert runner.main(argv(scene,output)+['--hardware','--steps','2'])==0
    assert len(plans)==3 and len(actions)==2  # warmup + two plans, two publications
    np.testing.assert_allclose(actions[0],.02)
    saved = json.loads(output.read_text())
    first,step = saved['episodes'][0]['steps']
    assert 'planning_feedback_sync' not in first
    assert step['action_published'] and step['planning_feedback_sync']['retries']==2
    assert step['planning_feedback_sync']['last_rejected_pair']['skew_s']==pytest.approx(.16)
