"""Shared camera intrinsics/extrinsics, with no vision or controller dependencies."""
import json
from pathlib import Path

import numpy as np

DEFAULT_CAMERA_CONFIG = Path(__file__).resolve().parents[2] / "configs/leap_pose_transform.json"


def save_registration_frame(path, color_rgb, depth, timestamp, camera_serial=""):
    """Keep the RGB-D frame paired with a SAM mask, rather than a later capture."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.tmp.npz')
    np.savez_compressed(pending, color_rgb=color_rgb, depth=depth,
                        timestamp=float(timestamp), camera_serial=str(camera_serial or ""))
    pending.replace(path)


def load_registration_frame(path, image_size, camera_serial=None):
    with np.load(path, allow_pickle=False) as frame:
        color, depth = frame['color_rgb'], frame['depth']
        stamp, serial = float(frame['timestamp']), str(frame['camera_serial'].item())
    width, height = image_size
    if color.dtype != np.uint8 or color.shape != (height, width, 3):
        raise ValueError('Registration RGB frame must match the calibrated RGB8 image size')
    if (depth.shape != (height, width) or not np.isfinite(depth).all()
            or (depth < 0).any() or not (depth > 0).any()):
        raise ValueError('Registration depth must be finite, nonnegative aligned depth in meters')
    if not np.isfinite(stamp) or stamp <= 0:
        raise ValueError('Registration timestamp must be epoch seconds')
    if camera_serial and serial != camera_serial:
        raise ValueError('Registration frame camera serial differs from the calibrated camera')
    return color, depth, stamp


def rigid_transform(value, name):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return matrix


def load_camera_config(path=DEFAULT_CAMERA_CONFIG, intrinsics=None, extrinsics=None, require_extrinsics=False):
    config = json.loads(Path(path).expanduser().read_text())
    K = np.asarray(config["camera_intrinsics"] if intrinsics is None else intrinsics, dtype=float)
    if require_extrinsics and config.get("extrinsics_status") == "pending_tag_measurement":
        raise ValueError("Camera extrinsics are pending: save the AprilTag calibration before tracking")
    if extrinsics is not None:
        T = rigid_transform(extrinsics, "camera-to-published override")
    elif "T_tag_from_camera" in config or "T_base_from_tag" in config:
        if "T_tag_from_camera" not in config or "T_base_from_tag" not in config:
            raise ValueError("Tag calibration requires both T_base_from_tag and T_tag_from_camera")
        # tag_calib.py outputs T_tag_from_camera, already inverted from pose_R/t.
        # The tag site supplies T_base_from_tag, exactly as pose_to_mujoco_frame.
        T = (rigid_transform(config["T_base_from_tag"], "T_base_from_tag")
             @ rigid_transform(config["T_tag_from_camera"], "T_tag_from_camera"))
    else:
        T = rigid_transform(config["camera_to_published"], "camera_to_published")
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError("camera_intrinsics must be a finite 3x3 K with positive focal lengths")
    size = config.get("image_size", [640, 480])
    if len(size) != 2 or any(isinstance(n, bool) or int(n) != n or n <= 0 for n in size):
        raise ValueError("image_size must contain positive integer width and height")
    fps = config.get("camera_fps", 60)
    if isinstance(fps, bool) or int(fps) != fps or fps <= 0:
        raise ValueError("camera_fps must be a positive integer")
    offset = np.asarray(config.get("object_translation_adjust_camera_m", [0, 0, 0]), dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("object_translation_adjust_camera_m must contain three finite camera-frame meters")
    return {**config, "camera_intrinsics": K, "camera_to_published": T, "camera_fps": int(fps),
            "object_translation_adjust_camera_m": offset,
            "image_size": tuple(map(int, size))}


def select_camera_device(calibration, devices, camera_index=0):
    """Pin mask capture and tracking to the calibrated serial, not USB order."""
    serial = calibration.get("camera_serial")
    if serial:
        matches = [device for device in devices if device['serial'] == serial]
        if not matches:
            raise ValueError(f"Calibrated RealSense {serial} is not connected")
        return matches[0]
    if not 0 <= camera_index < len(devices):
        raise ValueError(f"camera_index {camera_index} is not available")
    return devices[camera_index]


def verify_camera_profile(calibration, actual):
    """Check factory/profile provenance while allowing intentional K overrides."""
    if calibration.get("camera_serial") not in (None, actual["camera_serial"]):
        raise ValueError("RealSense serial does not match camera calibration")
    for field in ("image_size", "camera_fps"):
        if field in calibration and not np.array_equal(calibration[field], actual[field]):
            raise ValueError(f"RealSense {field} does not match camera calibration")
    recorded = calibration.get("realsense", {}).get("color_intrinsics")
    if recorded is not None and not np.allclose(recorded, actual["camera_intrinsics"], atol=1e-3, rtol=0):
        raise ValueError("RealSense factory intrinsics changed; refresh camera calibration")
    if np.any(np.abs(actual["realsense"]["distortion_coefficients"]) > 1e-10):
        raise ValueError("This pinhole tracking pipeline requires rectified color/depth frames for nonzero distortion")
