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
import warnings
import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # Pixels with no valid depth are explicitly excluded below.
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


def camera_cloud(depths, K, polygon):
    depth = np.median(depths, axis=0)
    mask = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 1)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    y, x = np.where((mask > 0) & (depth > .1) & (depth < 1.5))
    z = depth[y, x]
    return np.c_[(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z], np.c_[x, y]


def apply(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def add_object_comparison(out, comparison, depths, K, config):
    points, pixels = camera_cloud(depths, K, config['object_surface_polygon_px'])
    old = np.asarray(comparison['T_world_from_camera_current'])
    new = np.asarray(comparison['T_world_from_camera_scratch'])
    # Existing pipeline applies this correction to the camera-frame object pose
    # translation, BEFORE camera_to_published and published_to_world.
    offset = np.asarray(read_json('configs/source_camera.json').get('object_translation_adjust_camera_m', [0, 0, 0]))
    old_points, new_points = apply(old, points), apply(new, points)
    current_pipeline = apply(old, points + offset)
    trials = np.load(out / 'fit_data.npz')['bootstrap_T_world_from_camera']
    centroid = points.mean(axis=0)
    jitter = np.linalg.norm(np.array([apply(t, centroid) for t in trials]) - apply(new, centroid), axis=1) * 1000
    result = dict(description='Centroid of the observed visible cube surface; NOT the physical cube centre or ground truth.',
        samples=len(points), camera_surface_centroid_m=centroid.tolist(),
        current_geometric_world_surface_centroid_m=old_points.mean(axis=0).tolist(),
        current_object_pipeline_world_surface_centroid_m=current_pipeline.mean(axis=0).tolist(),
        scratch_world_surface_centroid_m=new_points.mean(axis=0).tolist(),
        scratch_minus_current_geometric_mm=((new_points - old_points).mean(axis=0) * 1000).tolist(),
        scratch_minus_current_object_pipeline_mm=((new_points - current_pipeline).mean(axis=0) * 1000).tolist(),
        current_object_correction_camera_m=offset.tolist(),
        scratch_includes_no_extra_object_correction=True,
        bootstrap_surface_centroid_variation_mm=residual_stats(jitter),
        warning='Keep geometric extrinsics and object-only correction separate. Do not stack the existing -15 mm offset onto a new depth-based candidate without revalidation.')
    np.savez_compressed(out / 'object_surface_clouds.npz', pixels=pixels, camera=points,
                        current_geometric=old_points, current_object_pipeline=current_pipeline, scratch=new_points)
    comparison['object_surface_comparison'] = result


def mesh_wire_image(rgb, vertices, faces, T_camera_from_body, K, color):
    cam = apply(T_camera_from_body, vertices)
    uv = (cam @ K.T)[:, :2] / cam[:, 2, None]
    mask = np.zeros(rgb.shape[:2], np.uint8)
    for i in np.argsort(cam[faces, 2].mean(axis=1))[::-1]:
        if cam[faces[i], 2].min() > 0:
            cv2.fillConvexPoly(mask, np.rint(uv[faces[i]]).astype(np.int32), 255)
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 1)
    return image


def render_report(out, comparison, rgbs, depths, K, vertices, faces, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rgb = np.median(rgbs, axis=0).astype(np.uint8)
    T_world_from_body = np.load(out / 'fit_data.npz')['T_world_from_body']
    modes = ['current', 'scratch']
    colors = [(230, 80, 20), (30, 190, 50)]  # BGR
    rows = []
    for mode, color in zip(modes, colors):
        T = np.linalg.inv(comparison[f'T_world_from_camera_{mode}']) @ T_world_from_body
        image = mesh_wire_image(rgb, vertices, faces, T, K, color)
        for r in comparison['metrics'][mode]['corners']:
            actual = tuple(np.rint(r['pixel']).astype(int))
            pred = tuple(np.rint(r['predicted_pixel']).astype(int))
            cv2.circle(image, actual, 2, (0, 220, 255), -1)
            cv2.line(image, actual, pred, color, 1)
            cv2.drawMarker(image, pred, color, cv2.MARKER_CROSS, 6, 1)
        cv2.putText(image, f'{mode}: CAD outline; yellow = observed corners', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1)
        cv2.imwrite(str(out / f'overlay_{mode}.png'), image)
        rows.append(cv2.resize(image[105:188, 270:410], (700, 415), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(out / 'overlay_comparison.png'), np.concatenate(rows, axis=1))
    annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for r in config['depth_regions']:
        color = (0, 200, 0) if r['split'] == 'train' else (255, 150, 0)
        cv2.polylines(annotated, [np.asarray(r['polygon_px'], np.int32)], True, color, 1)
    for i, r in enumerate(config['landmarks']):
        point = tuple(np.rint(r['pixel']).astype(int))
        cv2.circle(annotated, point, 2, (0, 200, 0) if r['split'] == 'train' else (255, 150, 0), -1)
        cv2.putText(annotated, str(i), (point[0] + 2, point[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, .25, (0, 255, 255), 1)
    cv2.putText(annotated, 'Green: train; blue: held out. Tag/fingers not used.', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 255, 255), 1)
    cv2.imwrite(str(out / 'observations.png'), annotated)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for i, mode in enumerate(modes):
        metrics = comparison['metrics'][mode]
        axes[0].bar(np.arange(2) + i * .32, [metrics['train_pixel_error']['rms'], metrics['heldout_pixel_error']['rms']], .32, label=mode)
        vals = [r['signed_plane_error_mm']['rms'] for r in metrics['depth']]
        axes[1].bar(np.arange(len(vals)) + i * .32, vals, .32, label=mode)
    axes[0].set(xticks=np.arange(2) + .16, xticklabels=['Training corners', 'Held-out housing'], ylabel='Reprojection RMS (pixels)')
    axes[1].set(xticks=np.arange(len(config['depth_regions'])) + .16,
                xticklabels=[r['id'] for r in config['depth_regions']], ylabel='Point-to-CAD-plane RMS (mm)')
    axes[1].tick_params(axis='x', rotation=25)
    for ax in axes:
        ax.legend(); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Stationary natural-feature / depth agreement — candidate, not ground truth')
    fig.savefig(out / 'residuals.png', dpi=170); fig.savefig(out / 'residuals.pdf'); plt.close(fig)
    cloud = np.load(out / 'object_surface_clouds.npz')
    fig = plt.figure(figsize=(10, 7)); ax = fig.add_subplot(projection='3d')
    verts_world = apply(T_world_from_body, vertices[::30])
    ax.scatter(*(verts_world * 1000).T, s=.4, c='gray', label='Fixed palm CAD')
    for name, label in [('current_geometric', 'Current geometric transform'),
                         ('current_object_pipeline', 'Current + existing object correction'), ('scratch', 'Scratch RGB-D candidate')]:
        pts = cloud[name][::10] * 1000
        ax.scatter(*pts.T, s=2, label=label)
    ax.set(xlabel='World X (mm)', ylabel='World Y (mm)', zlabel='World Z (mm)',
           title='Same observed cube surface, transformed three ways\nCube is on the table, not on the palm')
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]); ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / 'object_world_comparison.png', dpi=170); fig.savefig(out / 'object_world_comparison.pdf'); plt.close(fig)
    text = ['# Stationary RGB-D calibration comparison', '',
        '**Status: candidate only; no production parameters changed.**', '',
        'The scratch fit uses manually identified natural corners on the three rigid palm servo housings, sparse aligned depth, and the fixed palm CAD. No tag detector, finger geometry, joint reads, serial access, or motor commands. The current transform is loaded only after the independent fit.', '',
        '| Metric | Current | Scratch RGB-D |', '|---|---:|---:|']
    for key, label in [('train_pixel_error', 'Training corner RMS (px)'), ('heldout_pixel_error', 'Held-out middle-housing RMS (px)')]:
        text.append(f'| {label} | {comparison["metrics"]["current"][key]["rms"]:.3f} | {comparison["metrics"]["scratch"][key]["rms"]:.3f} |')
    for r in comparison['metrics']['current']['depth']:
        s = next(a for a in comparison['metrics']['scratch']['depth'] if a['id'] == r['id'])
        text.append(f'| {r["id"]}, {r["split"]}, depth RMS (mm) | {r["signed_plane_error_mm"]["rms"]:.3f} | {s["signed_plane_error_mm"]["rms"]:.3f} |')
    obj = comparison['object_surface_comparison']
    text += ['', f'Camera-centre difference: {comparison["camera_center_distance_mm"]:.2f} mm. Relative rotation: {comparison["relative_rotation_deg"]:.3f} degrees.',
        'Camera-centre change and rotation are coupled; this is a difference between estimates, not a measured correction to the real camera.', '',
        f'Same visible cube-surface centroid, scratch minus current **effective object pipeline**, world XYZ mm: {np.round(obj["scratch_minus_current_object_pipeline_mm"], 3).tolist()}.',
        'The current effective pipeline already shifts the tracked object -15 mm along camera Z. That correction is excluded from hand calibration and explicitly included in the object comparison. The new candidate has no additional object offset.', '',
        f'At the observed cube surface, annotation/depth perturbations give a P95 position variation of **{obj["bootstrap_surface_centroid_variation_mm"]["p95"]:.1f} mm**. The difference between object placements therefore does not establish that the new placement is more accurate.', '',
        '## Depth-bias alternative', '',
        f'With current extrinsics unchanged, a fitted additive raw-depth bias of {comparison["depth_bias_only_alternative"]["fitted_depth_bias_m"] * 1000:.2f} mm explains much of the depth discrepancy. This is fitted on training patches only and evaluated on the held-out patches. Image reprojection remains unchanged. This alternative is logged, not applied.', '',
        'Depth agreement alone cannot distinguish camera pose error, sensor depth bias, or imperfect CAD surface locations in this fixed view.', '',
        '## Later recording', '',
        f'A later camera-only capture supplies {comparison["later_static_recording_check"]["frames"]} additional saved frames. The candidate was not refitted; depth RMS values by frame and patch are in later_capture_depth_by_frame.csv. This remains the same physical view, not an independent camera pose.', '',
        '## Limits', '', '* ' + '\n* '.join(config['limitations']),
        '* Spatial hold-outs and repeated static frames support consistency checks, not independent multi-view validation.',
        '* RealSense depth bias, uncertain feature identities, and CAD-to-physical housing differences can masquerade as camera extrinsic error. No absolute ground-truth accuracy claim.',
        f'* Perturbing annotations by {config["pixel_sigma"]} px and correlated patch depths by {config["depth_sigma_m"] * 1000:.1f} mm gives a P95 rotation variation of {comparison["annotation_depth_sensitivity"]["rotation_variation_deg"]["p95"]:.2f} degrees.',
        '* Do not promote this candidate automatically. The current calibration and object correction remain untouched.', '',
        '## Reproduce', '', '```bash',
        '/home/chetan/miniconda3/envs/mj_env/bin/python -B camera_hand_calibration/stationary_rgbd.py --config ' + str(local(out / 'config.json').relative_to(BASE)), '```', '',
        'All numerical data are in comparison.json, candidate.json and fit_data.npz; image/depth source hashes are retained. The stationary image annotations are explicit in config.json.']
    (out / 'report.md').write_text('\n'.join(text) + '\n')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Stationary RGB-D comparison</title>'
        '<style>body{font:16px system-ui;max-width:1300px;margin:30px auto}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>Stationary RGB-D calibration candidate</h1><p>No production changes or robot access. '
        '<a href="report.md">Report</a> · <a href="comparison.json">Numbers</a> · <a href="candidate.json">Candidate transform</a></p>'
        '<h2>Observations</h2><img src="observations.png"><h2>Current vs scratch CAD projections</h2>'
        '<img src="overlay_comparison.png"><h2>Residuals</h2><img src="residuals.png">'
        '<h2>Object in MuJoCo world</h2><img src="object_world_comparison.png">'
        '<p>Single-pose, manually identified features: an experimental candidate, not an independently verified replacement.</p>')


def temporal_checks(out, comparison, config, reference_rgb, K, T_camera_from_body, T_prior_camera_from_body):
    """Fixed candidate, no refitting, on each frame of a later capture."""
    validation_config = dict(config, session=config['later_validation_session'])
    paths, rgbs, depths, metas, K2 = load_session(validation_config)
    if not np.allclose(K, K2):
        raise ValueError('Validation intrinsics differ')
    # Image registration detects camera/hand drift; it is not an annotation update.
    gray0 = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)[118:185, 278:400].astype(np.float32)
    rows = []
    for i, (path, rgb, depth, meta) in enumerate(zip(paths, rgbs, depths, metas)):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[118:185, 278:400].astype(np.float32)
        shift, response = cv2.phaseCorrelate(gray0, gray)
        regions = sample_depth(depth[None], config['depth_regions'], K, [0], config)
        for mode, T in [('current', T_prior_camera_from_body), ('scratch', T_camera_from_body)]:
            result = evaluate(T, config['landmarks'], K, regions)
            for region in result['depth']:
                rows.append(dict(frame=i, timestamp_epoch_s=meta['timestamp_epoch_s'], mode=mode,
                    region=region['id'], split=region['split'],
                    signed_depth_mean_mm=region['signed_plane_error_mm']['mean'],
                    depth_rms_mm=region['signed_plane_error_mm']['rms'],
                    phase_shift_x_px=shift[0], phase_shift_y_px=shift[1], phase_response=response))
    write_csv(out / 'later_capture_depth_by_frame.csv', rows)
    shifts = [np.hypot(r['phase_shift_x_px'], r['phase_shift_y_px']) for r in rows[::2 * len(config['depth_regions'])]]
    comparison['later_static_recording_check'] = dict(session=config['later_validation_session'],
        frames=len(paths), no_refitting=True, independent_physical_pose=False,
        appearance_registration_shift_px=residual_stats(shifts),
        frames_manifest=[dict(path=str(p.relative_to(BASE)), sha256=digest(p)) for p in paths],
        depth_rms_by_mode_region={mode: {region['id']: residual_stats([r['depth_rms_mm'] for r in rows
            if r['mode'] == mode and r['region'] == region['id']]) for region in config['depth_regions']}
            for mode in ['current', 'scratch']})
    cv2.imwrite(str(out / 'later_capture_rgb.png'), cv2.cvtColor(rgbs[0], cv2.COLOR_RGB2BGR))


def depth_bias_alternative(T_prior_camera_from_body, config, regions, validation_regions, K):
    """Compare an alternative explanation; never updates camera/depth settings."""
    train = [r for r in regions if r['split'] == 'train']
    def shifted(values, bias):
        return [dict(r, points=r['points'] * (1 + bias / r['points'][:, 2])[:, None]) for r in values]
    def score(bias):
        rows = evaluate(T_prior_camera_from_body, config['landmarks'], K, shifted(train, bias))['depth']
        return sum(r['signed_plane_error_mm']['rms'] ** 2 * r['signed_plane_error_mm']['n'] for r in rows)
    answer = minimize_scalar(score, bounds=(-.05, .05), method='bounded', options=dict(xatol=1e-10))
    result = evaluate(T_prior_camera_from_body, config['landmarks'], K, shifted(validation_regions, answer.x))
    return dict(explanation='Current extrinsics held fixed; fit only one additive aligned-depth Z bias from training patches, before deprojection.',
        fitted_depth_bias_m=float(answer.x), metrics=result,
        warning='A diagnostic alternative, not a sensor calibration or proposed production change. Depth bias and extrinsics cannot be reliably separated from this one stationary view.')


def source_hashes():
    """Read-only guard: all generated files remain in this workspace."""
    paths = [BASE.parent / 'configs/leap_pose_transform.json',
             BASE.parent / 'scenes/leap/hardware/calibrated_hand.xml',
             BASE.parent / 'configs/leap_hardware.json']
    return {str(p): digest(p) for p in paths}


def run(config_path):
    start = time.monotonic()
    sources_before = source_hashes()
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
    comparison['depth_bias_only_alternative'] = depth_bias_alternative(
        T_prior_camera_from_body, config, regions, validation_regions, K)
    write_json(out / 'candidate.json', dict(T_world_from_camera=candidate.tolist(),
        convention='OpenCV color optical camera -> MuJoCo frozen model world, metres',
        status=comparison['status'], production_exported=False,
        object_only_offset_excluded=True))
    np.savez_compressed(out / 'fit_data.npz', K=K, body_points=body, image_points=pixels,
                        T_world_from_body=T_world_from_body, bootstrap_T_world_from_camera=world_trials)
    cv2.imwrite(str(out / 'median_rgb.png'), cv2.cvtColor(np.median(rgbs, axis=0).astype(np.uint8), cv2.COLOR_RGB2BGR))
    temporal_checks(out, comparison, config, np.median(rgbs, axis=0).astype(np.uint8), K,
                    T_camera_from_body, T_prior_camera_from_body)
    add_object_comparison(out, comparison, depths, K, config)
    render_report(out, comparison, rgbs, depths, K, vertices, faces, config)
    comparison['source_hashes_before'] = sources_before
    comparison['source_hashes_after'] = source_hashes()
    comparison['production_inputs_unchanged'] = sources_before == comparison['source_hashes_after']
    comparison['wall_time_s'] = time.monotonic() - start
    write_json(out / 'comparison.json', comparison)
    for name in ['stationary_rgbd.py', 'workspace.py', 'geometry.py']:
        target = out / 'code' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(BASE / name, target)
    write_json(out / 'current_transform_used_for_comparison_only.json', prior_record)
    write_json('results/stationary_rgbd_latest.json', dict(directory=str(out.relative_to(BASE)),
        status=comparison['status'], absolute_accuracy_established=False,
        production_inputs_unchanged=comparison['production_inputs_unchanged']))
    print(json.dumps(dict(directory=str(out.relative_to(BASE)),
                         camera_center_distance_mm=comparison['camera_center_distance_mm'],
                         relative_rotation_deg=comparison['relative_rotation_deg'], metrics=metrics), indent=2), flush=True)
    return out, comparison, rgbs, depths, K, vertices, faces, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/stationary_rgbd.json')
    args = parser.parse_args()
    try:
        run(args.config)
    except BaseException:
        import traceback
        log = local('logs') / ('stationary_rgbd_error_' + identifier() + '.log')
        log.write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
