#!/usr/bin/env python3
"""Read RealSense intrinsics and calibrate camera -> tag -> MuJoCo base.

Uses the calibrated hand model's fixed 'tag' site, as pose_to_mujoco_frame
does in comfree_dex/test_leap_hardware.py. No hand motor is accessed.
S saves a stable AprilTag measurement; Q/Escape closes the camera. --once
saves automatically after enough stable frames. Tag size is the BLACK square.
"""
from collections import deque
from datetime import datetime, timezone
import argparse
import importlib.util
import json
from pathlib import Path
import time

import cv2
import mujoco
import numpy as np

from ContactModelStudy.Hardware.Camera import (
    DEFAULT_CAMERA_CONFIG, load_camera_config, select_camera_device,
)
from ContactModelStudy.Hardware.Calibration import (
    average_tag_transforms, tag_site_in_base, write_camera_calibration,
)

ROOT = Path(__file__).resolve().parent


def open_camera(args, calibration):
    source = ROOT / "iris-foundationpose-plusplus/src/utils/camera.py"
    spec = importlib.util.spec_from_file_location("leap_calibration_camera", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    devices = module.discover_camera_devices()
    selection = {**calibration, **({"camera_serial": args.serial} if args.serial else {})}
    device = select_camera_device(selection, devices, args.camera_index)
    width, height = calibration["image_size"]
    return module.Camera(device_id=device["serial"], width=width, height=height,
                         fps=args.fps or calibration["camera_fps"])


def detection_transform(detection):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = detection.pose_R, np.asarray(detection.pose_t).reshape(3)
    return matrix  # pupil_apriltags returns T_camera_from_tag, not its inverse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--model", type=Path, default=ROOT / "scenes/leap/hardware/calibrated_hand.xml")
    parser.add_argument("--site", default="tag")
    parser.add_argument("--serial")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--intrinsics-only", action="store_true")
    parser.add_argument("--tag-id", type=int, help="otherwise lock to the sole visible tag")
    parser.add_argument("--tag-family", default="tag36h11")
    parser.add_argument("--tag-size", type=float, default=0.04, help="black-square edge length in meters")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--max-translation-mm", type=float, default=3.0)
    parser.add_argument("--max-rotation-deg", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="automatically save stable tag calibration and exit")
    parser.add_argument("--no-viewer", action="store_true", help="use with --once or --intrinsics-only")
    parser.add_argument("--timeout", type=float, default=0, help="seconds; 0 waits for the operator")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    if args.samples < 2 or not np.isfinite([args.tag_size, args.max_translation_mm, args.max_rotation_deg]).all():
        parser.error("samples must be >=2 and tag size/stability limits must be finite")
    if min(args.tag_size, args.max_translation_mm, args.max_rotation_deg) <= 0:
        parser.error("tag size and stability limits must be positive")
    if not np.isfinite(args.timeout) or args.timeout < 0 or (args.fps is not None and args.fps <= 0):
        parser.error("timeout must be nonnegative; fps must be positive")
    if args.no_viewer and not (args.once or args.intrinsics_only):
        parser.error("--no-viewer requires --once or --intrinsics-only")
    calibration = load_camera_config(args.camera_config)
    before = json.loads(args.camera_config.expanduser().read_text())
    if not args.intrinsics_only:
        model = mujoco.MjModel.from_xml_path(str(args.model.expanduser()))
        base_from_tag = tag_site_in_base(model, args.site)
        from pupil_apriltags import Detector
        detector = Detector(families=args.tag_family, nthreads=2, quad_decimate=1.0,
                            refine_edges=1, decode_sharpening=0.25)
    record_path = args.record or ROOT / "results" / (
        "camera_calibration_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".json")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(before=before, status="starting")
    camera = None
    window = "LEAP camera calibration - S: save, Q / Esc: close"
    selected_id = args.tag_id
    saved = False
    try:
        camera = open_camera(args, calibration)
        metadata = camera.calibration
        mark_pending = (not args.intrinsics_only or before.get("extrinsics_status") != "calibrated"
                        or before.get("camera_serial") != metadata["camera_serial"])
        write_camera_calibration(args.camera_config, metadata, mark_pending=mark_pending)
        record.update(camera=metadata, status="intrinsics_saved")
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print("Measured RealSense intrinsics:", json.dumps(metadata, indent=2), flush=True)
        if args.intrinsics_only:
            return 0
        K = np.asarray(metadata["camera_intrinsics"])
        camera_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        print(f"Tag: {args.tag_family}, black-square edge {args.tag_size * 1000:g} mm; "
              f"site={args.site}. Expose the fixed tag to the camera. S saves; Q exits.", flush=True)
        print("T_base_from_tag (calibrated model site):\n", base_from_tag, flush=True)
        samples = deque(maxlen=args.samples)
        started = last_print = time.monotonic()
        if not args.no_viewer:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            frame = camera.getFrame()
            if frame is None:
                raise RuntimeError("RealSense returned no color/depth frame")
            color, _, _ = frame
            detections = detector.detect(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY), estimate_tag_pose=True,
                                         camera_params=camera_params, tag_size=args.tag_size)
            candidates = [d for d in detections if d.hamming == 0
                          and (selected_id is None or d.tag_id == selected_id)]
            mean = stability = None
            status = "NO TAG: expose the full black square and white border"
            display = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            if len(candidates) == 1:
                detection = candidates[0]
                selected_id = int(detection.tag_id)
                camera_from_tag = detection_transform(detection)
                if np.isfinite(camera_from_tag).all() and camera_from_tag[2, 3] > 0:
                    samples.append(camera_from_tag)
                    status = f"Tag {selected_id}, samples {len(samples)}/{args.samples}"
                    cv2.polylines(display, [np.rint(detection.corners).astype(np.int32)], True, (0, 255, 0), 2)
                    rotation, _ = cv2.Rodrigues(camera_from_tag[:3, :3])
                    cv2.drawFrameAxes(display, K, np.zeros(5), rotation, camera_from_tag[:3, 3], args.tag_size / 2)
                    if len(samples) == args.samples:
                        try:
                            mean, stability = average_tag_transforms(list(samples),
                                args.max_translation_mm / 1000, args.max_rotation_deg)
                            status += " STABLE - S to save"
                        except ValueError as exc:
                            status = str(exc)
                else:
                    samples.clear()
            else:
                samples.clear()
                if len(candidates) > 1:
                    status = "Multiple tags: restart with --tag-id"
            now = time.monotonic()
            if now - last_print >= 1:
                print(status, flush=True)
                if mean is not None:
                    print("T_tag_from_camera:\n", np.linalg.inv(mean), flush=True)
                last_print = now
            key = -1
            if not args.no_viewer:
                cv2.putText(display, f"{args.tag_family}, black edge {args.tag_size * 1000:g} mm", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(display, status, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if (args.once or key in (ord('s'), ord('S'))) and mean is not None:
                after = write_camera_calibration(args.camera_config, metadata, np.linalg.inv(mean), args.model,
                    selected_id, args.tag_size, args.tag_family, args.site, stability)
                record.update(after=after, status="calibrated")
                cv2.imwrite(str(record_path.with_suffix('.png')), display)
                saved = True
                print("SAVED:", args.camera_config, "\nT_base_from_camera:\n",
                      np.asarray(after['camera_to_published']), flush=True)
                if args.once:
                    break
            if args.timeout and now - started >= args.timeout:
                if saved:
                    break
                raise TimeoutError("No stable tag calibration saved before timeout")
        return 0 if saved else 1
    finally:
        try:
            if camera is not None:
                camera.pipeline.stop()
            if not args.no_viewer and not args.intrinsics_only:
                cv2.destroyAllWindows()
        finally:
            record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            print("Calibration record:", record_path, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
