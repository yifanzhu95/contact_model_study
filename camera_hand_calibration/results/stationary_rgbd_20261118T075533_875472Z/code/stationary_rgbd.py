#!/usr/bin/env python3
"""Stationary, marker-free RGB-D calibration candidate. Never touches motors.

Natural features are explicitly annotated in a confined configuration. The fit
has no current-transform argument: the prior is read ONLY after fitting, for
comparison. Fixed-palm geometry avoids assuming any finger joint positions.
"""
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(BASE / 'cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(BASE / 'cache/xdg')

import argparse
import csv
import json
import shutil
import time
import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from geometry import matrix, parameters, residual_stats
from workspace import local, read_json, write_json, digest, identifier, utc, environment, verify_model


def load_session(config):
    folder = local(config['session'])
    paths = sorted(folder.glob('frame_*.npz'))
    rgbs, depths, metas = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as frame:
            meta = json.loads(str(frame['metadata_json']))
            rgbs.append(frame['color_rgb'])
            depths.append(frame['depth_z16'].astype(float) * meta['depth_scale_m'])
            metas.append(meta)
    if not paths:
        raise ValueError('No RGB-D frames')
    K = np.asarray(metas[0]['camera_intrinsics'], float)
    for meta in metas:
        if (not np.allclose(K, meta['camera_intrinsics']) or
                np.any(np.abs(meta['distortion_coefficients']) > 1e-10) or
                meta['camera_serial'] != metas[0]['camera_serial']):
            raise ValueError('Requires one fixed, undistorted camera profile')
    return paths, np.asarray(rgbs), np.asarray(depths), metas, K


def sample_depth(depths, regions, K, indices, config):
    """Use non-boundary patches, median depth, and a sparse spatial grid."""
    data = depths[indices]
    valid = data > 0
    masked = np.where(valid, data, np.nan)
    depth = np.nanmedian(masked, axis=0)
    scatter = np.nanstd(masked, axis=0)
    y, x = np.indices(depth.shape)
    stride = config['depth_grid_stride_px']
    grid = (x % stride == 0) & (y % stride == 0)
    result = []
    for region in regions:
        mask = np.zeros(depth.shape, np.uint8)
        cv2.fillPoly(mask, [np.asarray(region['polygon_px'], np.int32)], 1)
        choose = ((mask > 0) & grid & np.isfinite(depth) &
                  (valid.sum(axis=0) >= max(1, int(len(indices) * .8))) &
                  (scatter < config['max_depth_temporal_std_m']))
        uv = np.c_[x[choose], y[choose]]
        z = depth[choose]
        if len(z) < 6:
            raise ValueError(f'Insufficient depth for {region["id"]}')
        points = np.c_[(uv[:, 0] - K[0, 2]) * z / K[0, 0],
                       (uv[:, 1] - K[1, 2]) * z / K[1, 1], z]
        result.append(dict(**region, points=points, pixels=uv))
    return result


def residual(x, body_points, pixels, K, regions, config):
    T = matrix(x)  # camera from body
    camera = body_points @ T[:3, :3].T + T[:3, 3]
    if np.any(camera[:, 2] <= .01):
        # Do not silently accept a behind-camera PnP branch.
        return np.full(2 * len(pixels) + sum(len(r['points']) for r in regions), 1e6)
    uv = (camera @ K.T)[:, :2] / camera[:, 2, None]
    values = [((uv - pixels) / config['pixel_sigma']).ravel()]
    for r in regions:
        body = (r['points'] - T[:3, 3]) @ T[:3, :3]
        values.append((body @ r['normal_body'] - r['plane_offset_m']) / config['depth_sigma_m'])
    return np.concatenate(values)


def solve(body, pixels, K, regions, config):
    """Two planar PnP branches plus SQPnP; no previous extrinsics."""
    starts = []
    for flag in [cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP]:
        answer = cv2.solvePnPGeneric(np.asarray(body, float), np.asarray(pixels, float),
                                     K, np.zeros(5), flags=flag)
        if answer[0]:
            starts.extend(np.r_[r.ravel(), t.ravel()] for r, t in zip(answer[1], answer[2]))
    solutions = []
    for x in starts:
        fit = least_squares(residual, x, args=(body, pixels, K, regions, config),
                            loss='soft_l1', f_scale=1., x_scale='jac', max_nfev=500)
        T = matrix(fit.x)
        if np.min((body @ T[:3, :3].T + T[:3, 3])[:, 2]) <= .01:
            continue
        solutions.append((fit.cost, fit, T))
    if not solutions:
        raise ValueError('No valid positive-depth fit')
    solutions.sort(key=lambda a: a[0])
    return solutions[0][2], solutions[0][1], [dict(cost=float(c), T_camera_from_body=T.tolist())
                                            for c, _, T in solutions]


def evaluate(T, landmarks, K, regions):
    body = np.asarray([r['body_xyz_m'] for r in landmarks])
    pixels = np.asarray([r['pixel'] for r in landmarks])
    cam = body @ T[:3, :3].T + T[:3, 3]
    uv = (cam @ K.T)[:, :2] / cam[:, 2, None]
    corner_rows = [dict(id=r['id'], split=r['split'], pixel=r['pixel'],
                       predicted_pixel=p.tolist(), error_px=float(np.linalg.norm(p - r['pixel'])))
                   for r, p in zip(landmarks, uv)]
    depth_rows = []
    for r in regions:
        b = (r['points'] - T[:3, 3]) @ T[:3, :3]
        error = (b @ r['normal_body'] - r['plane_offset_m']) * 1000
        depth_rows.append(dict(id=r['id'], split=r['split'], signed_plane_error_mm=residual_stats(error)))
    return dict(corners=corner_rows, depth=depth_rows,
                train_pixel_error=residual_stats([r['error_px'] for r in corner_rows if r['split'] == 'train']),
                heldout_pixel_error=residual_stats([r['error_px'] for r in corner_rows if r['split'] == 'validation']))


def frozen_geometry():
    manifest = verify_model()
    binary = local('model/hand.mjb')
    if digest(binary) != manifest['binary_sha256']:
        raise ValueError('Frozen binary model changed')
    m = mujoco.MjModel.from_binary_path(str(binary))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    b, g = m.body('palm').id, m.geom('palm_visual').id
    # Palm must remain fixed to world, irrespective of all joint coordinates.
    cursor = b
    while cursor:
        if m.body_jntnum[cursor]:
            raise ValueError('Palm is articulated; camera-only fitting would be invalid')
        cursor = int(m.body_parentid[cursor])
    T_world_from_body = np.eye(4)
    T_world_from_body[:3, :3] = d.xmat[b].reshape(3, 3)
    T_world_from_body[:3, 3] = d.xpos[b]
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    vertices = (world - d.xpos[b]) @ d.xmat[b].reshape(3, 3)
    faces = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
    return T_world_from_body, vertices, faces, manifest


def bootstrap(T, body, pixels, K, regions, config):
    """Annotation sensitivity, not an absolute calibration confidence interval."""
    rng = np.random.default_rng(config['seed'])
    transforms = []
    for _ in range(config['bootstrap_repetitions']):
        uv = pixels + rng.normal(0, config['pixel_sigma'], pixels.shape)
        perturbed = []
        for r in regions:
            points = r['points'].copy()
            # Correlated depth bias per patch is deliberately retained.
            points *= (1 + rng.normal(0, config['depth_sigma_m']) / points[:, 2])[:, None]
            perturbed.append(dict(r, points=points))
        result = least_squares(residual, parameters(T), args=(body, uv, K, perturbed, config),
                               loss='soft_l1', max_nfev=120, x_scale='jac')
        transforms.append(matrix(result.x))
    return np.asarray(transforms)


def write_csv(path, rows):
    with local(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path):
    start = time.monotonic()
    config = read_json(config_path)
    out = local('results') / ('stationary_rgbd_' + identifier())
    out.mkdir()
    write_json(out / 'config.json', config)
    paths, rgbs, depths, metas, K = load_session(config)
    train_indices, valid_indices = config['train_frame_indices'], config['validation_frame_indices']
    if set(train_indices) & set(valid_indices):
        raise ValueError('Train and validation frames overlap')
    T_world_from_body, vertices, faces, manifest = frozen_geometry()
    for point in config['landmarks']:
        distance = np.linalg.norm(vertices - point['body_xyz_m'], axis=1).min()
        if distance > .0001:
            raise ValueError(f'Annotated CAD corner is not a mesh vertex: {point["id"]}, {distance}')
    regions = sample_depth(depths, config['depth_regions'], K, train_indices, config)
    training = [r for r in config['landmarks'] if r['split'] == 'train']
    body = np.asarray([r['body_xyz_m'] for r in training], float)
    pixels = np.asarray([r['pixel'] for r in training], float)
    train_regions = [r for r in regions if r['split'] == 'train']
    T_camera_from_body, fit, branches = solve(body, pixels, K, train_regions, config)
    candidate = T_world_from_body @ np.linalg.inv(T_camera_from_body)
    print('Independent RGB-D fit complete; now loading frozen prior for comparison.', flush=True)
    # The prior is deliberately inaccessible to solve(), residual(), and solvePnP.
    prior_record = read_json('configs/current_transform.json')
    prior = np.asarray(prior_record['T_world_from_camera'])
    T_prior_camera_from_body = np.linalg.inv(prior) @ T_world_from_body
    validation_regions = sample_depth(depths, config['depth_regions'], K, valid_indices, config)
    metrics = dict(current=evaluate(T_prior_camera_from_body, config['landmarks'], K, validation_regions),
                   scratch=evaluate(T_camera_from_body, config['landmarks'], K, validation_regions))
    trials = bootstrap(T_camera_from_body, body, pixels, K, train_regions, config)
    world_trials = np.asarray([T_world_from_body @ np.linalg.inv(t) for t in trials])
    center_error = np.linalg.norm(world_trials[:, :3, 3] - candidate[:3, 3], axis=1) * 1000
    angles = Rotation.from_matrix(trials[:, :3, :3] @ T_camera_from_body[:3, :3].T).magnitude() * 180 / np.pi
    jac = fit.jac @ np.diag([1, 1, 1, .1, .1, .1])
    sv = np.linalg.svd(jac, compute_uv=False)
    comparison = dict(created_at_utc=utc(), status='candidate_unvalidated_single_pose',
        T_world_from_camera_current=prior.tolist(), T_world_from_camera_scratch=candidate.tolist(),
        camera_center_difference_world_mm=((candidate[:3, 3] - prior[:3, 3]) * 1000).tolist(),
        camera_center_distance_mm=float(np.linalg.norm(candidate[:3, 3] - prior[:3, 3]) * 1000),
        relative_rotation_deg=float(Rotation.from_matrix(candidate[:3, :3] @ prior[:3, :3].T).magnitude() * 180 / np.pi),
        metrics=metrics, pnp_branches=branches,
        optimizer=dict(success=bool(fit.success), cost=float(fit.cost), evaluations=fit.nfev,
                       scaled_jacobian_rank=int(np.linalg.matrix_rank(jac)),
                       scaled_jacobian_condition=float(sv[0] / sv[-1])),
        annotation_depth_sensitivity=dict(repetitions=len(trials), pixel_sigma=config['pixel_sigma'],
            per_patch_depth_bias_sigma_m=config['depth_sigma_m'],
            camera_center_variation_mm=residual_stats(center_error), rotation_variation_deg=residual_stats(angles),
            warning='Model/feature identity, mounting, sensor systematics are not represented. Not metrological confidence.'),
        training_features=len(training), heldout_features=len(config['landmarks']) - len(training),
        independent_physical_poses=1, hand_access=False, joint_reads=False, motor_commands=False,
        tag_detection=False, production_modified=False, object_only_offset_applied_to_hand=False,
        intrinsics_fixed=K.tolist(), environment=environment(), model_manifest=manifest,
        config_sha256=digest(local(config_path)),
        frames=[dict(path=str(p.relative_to(BASE)), sha256=digest(p)) for p in paths],
        limitations=config['limitations'], wall_time_s=time.monotonic() - start)
    write_json(out / 'candidate.json', dict(T_world_from_camera=candidate.tolist(),
        convention='OpenCV color optical camera -> MuJoCo frozen model world, metres',
        status=comparison['status'], production_exported=False,
        object_only_offset_excluded=True))
    write_json(out / 'comparison.json', comparison)
    np.savez_compressed(out / 'fit_data.npz', K=K, body_points=body, image_points=pixels,
                        T_world_from_body=T_world_from_body, bootstrap_T_world_from_camera=world_trials)
    cv2.imwrite(str(out / 'median_rgb.png'), cv2.cvtColor(np.median(rgbs, axis=0).astype(np.uint8), cv2.COLOR_RGB2BGR))
    for name in ['stationary_rgbd.py', 'workspace.py', 'geometry.py']:
        target = out / 'code' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(BASE / name, target)
    write_json(out / 'current_transform_used_for_comparison_only.json', prior_record)
    print(json.dumps(dict(directory=str(out.relative_to(BASE)),
                         camera_center_distance_mm=comparison['camera_center_distance_mm'],
                         relative_rotation_deg=comparison['relative_rotation_deg'], metrics=metrics), indent=2), flush=True)
    return out, comparison, rgbs, depths, K, vertices, faces, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/stationary_rgbd.json')
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
