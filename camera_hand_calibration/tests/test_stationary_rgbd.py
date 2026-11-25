"""Hardware-free checks for the independent stationary fit and frame mapping."""
import inspect
import numpy as np
from scipy.spatial.transform import Rotation

from stationary_rgbd import solve, apply, sample_depth, depth_bias_alternative
from workspace import read_json


def synthetic_case():
    config = read_json('configs/stationary_rgbd.json')
    points = np.asarray([r['body_xyz_m'] for r in config['landmarks']])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', [3., -4., 88.], degrees=True).as_matrix()
    T[:3, 3] = [.02, -.09, .65]
    K = np.array([[605., 0., 320.], [0., 605., 240.], [0., 0., 1.]])
    camera = apply(T, points)
    uv = (camera @ K.T)[:, :2] / camera[:, 2, None]
    x, y = np.meshgrid(np.linspace(-.09, -.025, 6), np.linspace(-.08, .02, 6))
    flat = np.c_[x.ravel(), y.ravel(), np.full(x.size, -.0347)]
    region = dict(points=apply(T, flat), normal_body=[0, 0, -1], plane_offset_m=.0347)
    return config, points, T, K, uv, [region]


def test_scratch_rgbd_recovers_known_pose_without_prior():
    config, points, T, K, uv, regions = synthetic_case()
    actual, fit, branches = solve(points, uv, K, regions, config)
    assert fit.success and len(branches) >= 2
    np.testing.assert_allclose(actual, T, atol=1e-6)
    assert 'prior' not in inspect.signature(solve).parameters


def test_world_transform_direction_and_object_offset_order():
    _, points, camera_from_body, _, _, _ = synthetic_case()
    world_from_body = np.eye(4)
    world_from_body[:3, :3] = Rotation.from_euler('y', 30, degrees=True).as_matrix()
    world_from_body[:3, 3] = [0, 0, .01]
    world_from_camera = world_from_body @ np.linalg.inv(camera_from_body)
    np.testing.assert_allclose(apply(world_from_camera, apply(camera_from_body, points)),
                               apply(world_from_body, points), atol=1e-12)
    offset = np.array([0, 0, -.015])
    np.testing.assert_allclose(apply(world_from_camera, points + offset) - apply(world_from_camera, points),
                               np.tile(world_from_camera[:3, :3] @ offset, (len(points), 1)), atol=1e-12)


def test_validation_features_do_not_enter_training_geometry():
    config = read_json('configs/stationary_rgbd.json')
    assert all(r['id'].startswith(('if_', 'rf_')) for r in config['landmarks'] if r['split'] == 'train')
    assert all(r['id'].startswith('mf_') for r in config['landmarks'] if r['split'] == 'validation')
    assert not set(config['train_frame_indices']) & set(config['validation_frame_indices'])


def test_depth_zero_and_unstable_pixels_are_excluded():
    config = dict(depth_grid_stride_px=1, max_depth_temporal_std_m=.002)
    depth = np.full((3, 12, 12), .6)
    depth[:, 3, 3] = 0
    depth[:, 4, 4] = [.6, .7, .8]
    r = dict(id='patch', polygon_px=[[1, 1], [10, 1], [10, 10], [1, 10]],
             normal_body=[0, 0, -1], plane_offset_m=.03)
    result = sample_depth(depth, [r], np.array([[600, 0, 6], [0, 600, 6], [0, 0, 1]]), [0, 1, 2], config)[0]
    assert [3, 3] not in result['pixels'].tolist()
    assert [4, 4] not in result['pixels'].tolist()
    assert np.isfinite(result['points']).all()


def test_depth_bias_alternative_recovers_known_bias_without_extrinsic_change():
    config, _, T, K, _, regions = synthetic_case()
    original = regions[0]['points']
    measured = original * (1 + .02 / original[:, 2])[:, None]
    patch = dict(regions[0], points=measured, id='training_patch', split='train')
    heldout = dict(patch, id='heldout_patch', split='validation')
    unchanged = T.copy()
    result = depth_bias_alternative(T, config, [patch], [heldout], K)
    np.testing.assert_array_equal(T, unchanged)
    assert abs(result['fitted_depth_bias_m'] + .02) < 1e-6
    assert result['metrics']['depth'][0]['signed_plane_error_mm']['rms'] < .001
