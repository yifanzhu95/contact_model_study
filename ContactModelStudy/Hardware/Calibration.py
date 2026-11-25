"""Store RealSense intrinsics and camera/tag/base calibration in shared JSON."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .Camera import rigid_transform


def tag_site_in_base(model, site_name="tag"):
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if index < 0:
        raise ValueError(f"Calibration model has no {site_name!r} site")
    body = model.site_bodyid[index]
    while body:
        if model.body_jntnum[body] or model.body_mocapid[body] >= 0:
            raise ValueError("Calibration tag site must be fixed to the hand base")
        body = model.body_parentid[body]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    matrix = np.eye(4)
    matrix[:3, :3] = data.site_xmat[index].reshape(3, 3)
    matrix[:3, 3] = data.site_xpos[index]
    return matrix


def verify_model_tag(config, model):
    if 'T_base_from_tag' in config:
        site_name = config.get('tag_calibration', {}).get('site_name', 'tag')
        actual = tag_site_in_base(model, site_name)
        if not np.allclose(actual, config['T_base_from_tag'], atol=1e-6, rtol=0):
            raise ValueError('Camera tag calibration does not match the selected model tag mount; recalibrate the camera')


def average_tag_transforms(samples, max_translation_m=0.003, max_rotation_deg=2.0):
    if not samples:
        raise ValueError("No tag poses to average")
    values = np.stack([rigid_transform(value, "tag pose") for value in samples])
    rotation = values[:, :3, :3].mean(axis=0)
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = values[:, :3, 3].mean(axis=0)
    translation_error = np.linalg.norm(values[:, :3, 3] - matrix[:3, 3], axis=1).max()
    relative = values[:, :3, :3] @ rotation.T
    angles = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    rotation_error = angles.max()
    if translation_error > max_translation_m or rotation_error > max_rotation_deg:
        raise ValueError(f"Tag pose not stable: {translation_error * 1000:.1f} mm, {rotation_error:.2f} deg")
    return matrix, dict(samples=len(samples), max_translation_deviation_m=float(translation_error),
                        max_rotation_deviation_deg=float(rotation_error))


def write_camera_calibration(path, metadata, tag_from_camera=None, model_path=None,
                             tag_id=None, tag_size_m=0.04, tag_family="tag36h11",
                             site_name="tag", stability=None, mark_pending=False):
    """An atomic snapshot; intrinsic-only updates preserve existing transforms."""
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    config.update(metadata)
    now = datetime.now(timezone.utc).isoformat()
    config["intrinsics_measured_at_utc"] = now
    if mark_pending:
        config["extrinsics_status"] = "pending_tag_measurement"
    if tag_from_camera is not None:
        tag_from_camera = rigid_transform(tag_from_camera, "T_tag_from_camera")
        model_path = Path(model_path).expanduser().resolve()
        model = mujoco.MjModel.from_xml_path(str(model_path))
        base_from_tag = tag_site_in_base(model, site_name)
        config.update(T_tag_from_camera=tag_from_camera.tolist(),
                      T_base_from_tag=base_from_tag.tolist(),
                      camera_to_published=(base_from_tag @ tag_from_camera).tolist(),
                      published_pose_frame="mujoco_world", extrinsics_status="calibrated")
        config["tag_calibration"] = dict(tag_id=tag_id, tag_family=tag_family, tag_size_m=tag_size_m,
            model_path=str(model_path), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            site_name=site_name, measured_at_utc=now, stability=stability or {})
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return config
