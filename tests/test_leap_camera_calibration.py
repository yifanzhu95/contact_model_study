"""RealSense profile selection and the tag->base object-pose chain, without USB."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import calibrate_leap_camera as calibrator
from ContactModelStudy.Hardware.Calibration import (
    average_tag_transforms, tag_site_in_base, verify_model_tag, write_camera_calibration,
)
from ContactModelStudy.Hardware.Camera import (
    load_camera_config, select_camera_device, verify_camera_profile,
)
from ContactModelStudy.Hardware.LCM import FeedbackBuffer, Pose, RobotState, load_pose_transform
from ContactModelStudy.Utils.Quaternions import matToQuat

ROOT = Path(__file__).resolve().parents[1]
HAND = ROOT / 'scenes/leap/hardware/calibrated_hand.xml'


def metadata():
    K = [[600, 0, 320], [0, 601, 240], [0, 0, 1]]
    return dict(camera_intrinsics=K, camera_serial='synthetic', image_size=[640, 480], camera_fps=60,
        realsense=dict(color_intrinsics=deepcopy(K), distortion_coefficients=[0]*5))


@pytest.fixture
def camera_path(tmp_path):
    path = tmp_path / 'camera.json'
    config = dict(camera_intrinsics=metadata()['camera_intrinsics'], image_size=[640, 480],
        camera_to_published=np.eye(4).tolist(), T_model_from_published=np.eye(4).tolist(),
        translation_adjust_m=[0, 0, 0], rotation_adjust_deg=[0, 0, 0],
        T_tracked_from_object=np.eye(4).tolist())
    path.write_text(json.dumps(config))
    return path


def test_tag_site_uses_world_pose_including_calibrated_palm_mount():
    model = mujoco.MjModel.from_xml_path(str(HAND))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    index = model.site('tag').id
    expected = np.eye(4)
    expected[:3, :3], expected[:3, 3] = data.site_xmat[index].reshape(3, 3), data.site_xpos[index]
    np.testing.assert_allclose(tag_site_in_base(model), expected)
    assert not np.allclose(expected[:3, 3], model.site_pos[index])


def test_moving_tag_mount_is_rejected():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/>'
        '<geom type="sphere" size=".01"/><site name="tag"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError, match='fixed to the hand base'):
        tag_site_in_base(model)
    with pytest.raises(ValueError, match='no'):
        tag_site_in_base(model, 'missing')


def test_controller_rejects_a_different_tag_mount():
    model = mujoco.MjModel.from_xml_path(str(HAND))
    config = dict(T_base_from_tag=tag_site_in_base(model).tolist())
    verify_model_tag(config, model)
    config['T_base_from_tag'][0][3] += 0.01
    with pytest.raises(ValueError, match='does not match'):
        verify_model_tag(config, model)


@pytest.mark.parametrize("offset", [[0, 0, 0], [0, 0, -0.015]])
def test_full_tracker_to_lcm_path_matches_pose_to_mujoco_frame(camera_path, offset):
    # The same composition as the user's reference, with nontrivial rotations
    # and translations so inversion/order/double-transform mistakes are visible.
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'scenes/leap/hardware/cube.xml'))
    camera_from_tag, camera_from_object = np.eye(4), np.eye(4)
    camera_from_tag[:3, :3] = Rotation.from_euler('xyz', [0.1, -0.2, 0.3]).as_matrix()
    camera_from_tag[:3, 3] = [0.02, -0.03, 0.6]
    camera_from_object[:3, :3] = Rotation.from_euler('xyz', [1, -0.4, 0.7]).as_matrix()
    camera_from_object[:3, 3] = [0.01, 0.02, 0.5]
    write_camera_calibration(camera_path, metadata(), np.linalg.inv(camera_from_tag), HAND, tag_id=7)
    saved = json.loads(camera_path.read_text())
    saved["object_translation_adjust_camera_m"] = offset
    camera_path.write_text(json.dumps(saved))
    config = load_camera_config(camera_path, require_extrinsics=True)
    source = ROOT / 'iris-foundationpose-plusplus/src/online_tracking2ctrl.py'
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'calculate_object_in_base')
    namespace = {'np': np, 'args': SimpleNamespace(camera_config_data=config)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    original = camera_from_object.copy()
    published = namespace['calculate_object_in_base'](camera_from_object)
    corrected = camera_from_object.copy()
    corrected[:3, 3] += offset
    expected = tag_site_in_base(model) @ np.linalg.inv(camera_from_tag) @ corrected
    np.testing.assert_array_equal(camera_from_object, original)
    np.testing.assert_allclose(published, expected, atol=1e-12)
    before, after, _ = load_pose_transform(camera_path)
    buffer = FeedbackBuffer(model, before, after)
    stamp = int(time.time() * 1e6)
    pose = Pose()
    pose.timestamp, pose.pose_matrix = stamp, published.reshape(16).tolist()
    buffer.on_robot('LEAP_STATE', RobotState(stamp, np.zeros(16), np.zeros(16)).encode())
    buffer.on_pose('POSE', pose.encode())
    q, _, _ = buffer.snapshot()
    np.testing.assert_allclose(q[16:19], expected[:3, 3], atol=1e-7)
    assert abs(np.dot(q[19:23], matToQuat(expected[:3, :3]))) > 1 - 1e-7


def test_tag_fields_are_authoritative_and_legacy_override_still_works(camera_path):
    camera_from_tag = np.eye(4)
    camera_from_tag[:3, 3] = [0.01, 0.02, 0.6]
    saved = write_camera_calibration(camera_path, metadata(), np.linalg.inv(camera_from_tag), HAND, tag_id=7)
    saved['camera_to_published'] = np.eye(4).tolist()  # stale cached matrix must not be used
    camera_path.write_text(json.dumps(saved))
    loaded = load_camera_config(camera_path)
    np.testing.assert_allclose(loaded['camera_to_published'],
        np.asarray(saved['T_base_from_tag']) @ saved['T_tag_from_camera'])
    np.testing.assert_allclose(load_camera_config(camera_path, extrinsics=np.eye(4))['camera_to_published'], np.eye(4))


def test_pending_calibration_blocks_tracking_and_controller_but_allows_preview(camera_path):
    write_camera_calibration(camera_path, metadata(), mark_pending=True)
    assert load_camera_config(camera_path)['camera_fps'] == 60
    with pytest.raises(ValueError, match='pending'):
        load_camera_config(camera_path, require_extrinsics=True)
    with pytest.raises(ValueError, match='pending'):
        load_pose_transform(camera_path)


def test_half_written_tag_transform_is_rejected(camera_path):
    data = json.loads(camera_path.read_text())
    data['T_base_from_tag'] = np.eye(4).tolist()
    camera_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='requires both'):
        load_camera_config(camera_path)


def test_rotation_average_handles_half_turn_wraparound_and_rejects_motion():
    samples = []
    for angle in (179.9, 180, -179.9):
        value = np.eye(4)
        value[:3, :3] = Rotation.from_euler('z', angle, degrees=True).as_matrix()
        samples.append(value)
    mean, stats = average_tag_transforms(samples)
    np.testing.assert_allclose(mean[:3, :3], np.diag([-1, -1, 1]), atol=1e-12)
    assert stats['samples'] == 3 and stats['max_rotation_deviation_deg'] < 0.101
    samples[0][:3, 3] = [0.02, 0, 0]
    with pytest.raises(ValueError, match='not stable'):
        average_tag_transforms(samples)


def test_calibrated_serial_wins_over_usb_enumeration_order():
    devices = [{'serial':'other'}, {'serial':'synthetic'}]
    assert select_camera_device(metadata(), devices, 0)['serial'] == 'synthetic'
    with pytest.raises(ValueError, match='not connected'):
        select_camera_device(metadata(), devices[:1])


@pytest.mark.parametrize('mismatch', ['camera_serial', 'camera_fps', 'image_size', 'intrinsics', 'distortion'])
def test_runtime_profile_rejects_wrong_camera_or_pixel_grid(mismatch):
    actual, configured = metadata(), metadata()
    if mismatch == 'camera_serial':
        actual[mismatch] = 'other'
    elif mismatch == 'camera_fps':
        actual[mismatch] = 30
    elif mismatch == 'image_size':
        actual[mismatch] = [320, 240]
    elif mismatch == 'intrinsics':
        actual['camera_intrinsics'][0][0] = 610
    else:
        actual['realsense']['distortion_coefficients'][0] = 0.1
    with pytest.raises(ValueError):
        verify_camera_profile(configured, actual)


def test_intentional_calibrated_K_override_does_not_change_factory_provenance():
    actual, configured = metadata(), metadata()
    configured['camera_intrinsics'][0][0] = 610
    verify_camera_profile(configured, actual)


def test_real_camera_depth_is_aligned_and_scaled_in_meters(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyrealsense2', SimpleNamespace())
    source = ROOT / 'iris-foundationpose-plusplus/src/utils/camera.py'
    spec = importlib.util.spec_from_file_location('calibration_depth_contract', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw_depth = np.array([[100, 1000], [0, 2000]], dtype=np.uint16)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    frames = SimpleNamespace(get_depth_frame=lambda: SimpleNamespace(get_data=lambda: raw_depth),
        get_color_frame=lambda: SimpleNamespace(get_data=lambda: rgb, get_timestamp=lambda: 123))
    camera = module.Camera.__new__(module.Camera)
    camera.pipeline = SimpleNamespace(wait_for_frames=lambda: frames)
    camera.align = SimpleNamespace(process=lambda f: f)
    camera.depth_scale = 0.002
    color, depth, stamp = camera.getFrame()
    np.testing.assert_allclose(depth, [[0.2, 2.0], [0, 4.0]])
    assert color.shape == rgb.shape and stamp == 123


@pytest.mark.parametrize('intrinsics_only', [False, True])
def test_calibration_capture_saves_and_releases_camera(camera_path, tmp_path, monkeypatch, intrinsics_only):
    closed = []
    camera = SimpleNamespace(calibration=metadata(), pipeline=SimpleNamespace(stop=lambda: closed.append(True)),
        getFrame=lambda: (np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640)), 123))
    monkeypatch.setattr(calibrator, 'open_camera', lambda *args: camera)
    detection = SimpleNamespace(tag_id=7, hamming=0, pose_R=np.eye(3), pose_t=np.array([[0], [0], [0.5]]),
        corners=np.array([[310,230], [330,230], [330,250], [310,250]]))
    monkeypatch.setitem(sys.modules, 'pupil_apriltags', SimpleNamespace(
        Detector=lambda **kwargs: SimpleNamespace(detect=lambda *a, **k: [detection])))
    record = tmp_path / 'measurement.json'
    extra = ['--intrinsics-only'] if intrinsics_only else ['--once', '--samples', '2', '--tag-id', '7']
    assert calibrator.main(['--camera-config', str(camera_path), '--record', str(record),
        '--no-viewer', *extra]) == 0
    saved = json.loads(camera_path.read_text())
    assert closed == [True] and saved['camera_serial'] == 'synthetic'
    if intrinsics_only:
        assert saved['extrinsics_status'] == 'pending_tag_measurement'
    else:
        assert saved['extrinsics_status'] == 'calibrated'
        np.testing.assert_allclose(saved['T_tag_from_camera'], np.array([
            [1,0,0,0], [0,1,0,0], [0,0,1,-0.5], [0,0,0,1]]))
        assert json.loads(record.read_text())['before']['camera_to_published'] == np.eye(4).tolist()


@pytest.mark.parametrize("offset", [[0, 0], [0, 0, float("nan")]])
def test_invalid_camera_pose_offset_is_rejected(camera_path, offset):
    saved = json.loads(camera_path.read_text())
    saved["object_translation_adjust_camera_m"] = offset
    camera_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="object_translation_adjust_camera_m"):
        load_camera_config(camera_path)


def test_registration_frame_preserves_exact_mask_capture_rgb_depth_and_timestamp(tmp_path):
    from ContactModelStudy.Hardware.Camera import save_registration_frame, load_registration_frame
    color = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    depth = np.linspace(.2, .8, 6 * 8, dtype=np.float32).reshape(6, 8)
    path = tmp_path / 'frame.npz'
    save_registration_frame(path, color, depth, 1234.5, 'test-camera')
    rgb, measured_depth, timestamp = load_registration_frame(path, (8, 6), 'test-camera')
    np.testing.assert_array_equal(rgb, color)
    np.testing.assert_array_equal(measured_depth, depth)
    assert timestamp == 1234.5
    with pytest.raises(ValueError, match='serial'):
        load_registration_frame(path, (8, 6), 'another-camera')
    with pytest.raises(ValueError, match='image size'):
        load_registration_frame(path, (16, 12), 'test-camera')
