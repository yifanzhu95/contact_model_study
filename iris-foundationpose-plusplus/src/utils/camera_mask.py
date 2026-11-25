# src/utils/camera_mask.py
import os, sys, time, json, argparse, requests, traceback
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # avoid Qt/X11 even if OpenCV tries

try:
    sys.stdout.reconfigure(line_buffering=True)  # py3.7+
except Exception:
    pass

DEBUG = False
def log(*a):
    if DEBUG:
        print(*a, flush=True)

def fatal(msg, e=None, tb=False):
    print(f"[FATAL] {msg}", flush=True)
    if e is not None:
        print(f"        {type(e).__name__}: {e}", flush=True)
    if tb:
        traceback.print_exc()
    sys.exit(1)

# --- make imports work like your other scripts ---
here = os.path.dirname(os.path.abspath(__file__))         # .../src/utils
src_path = os.path.abspath(os.path.join(here, ".."))      # .../src
root_path = os.path.abspath(os.path.join(src_path, "..")) # project root
for p in [src_path, root_path]:
    if p not in sys.path:
        sys.path.append(p)

# Camera import (robust)
try:
    from utils.camera import Camera, discover_camera_devices
    CAM_SRC = "utils.camera"
except Exception as e1:
    try:
        from camera import Camera, discover_camera_devices  # fallback
        CAM_SRC = "camera"
    except Exception as e2:
        fatal("Could not import Camera/ discover_camera_devices", e2, tb=True)

# Publisher import (optional)
try:
    from utils.publisher import Publisher
    PUB_SRC = "utils.publisher"
except Exception as e:
    Publisher = None
    PUB_SRC = "N/A"
    print(f"[WARN] Publisher not available: {e}", flush=True)

import numpy as np
import cv2  # no HighGUI usage here (no windows)

def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def post_mask(frame_rgb, bbox_xywh, output_mask_path, web_api_url, snapshot_path=None):
    ensure_dir(output_mask_path)
    tmp_path = os.path.abspath(snapshot_path or "./tmp/temp_frame.png")
    output_mask_path = os.path.abspath(output_mask_path)
    ensure_dir(tmp_path)
    try:
        if not cv2.imwrite(tmp_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)):
            raise OSError("could not save captured frame")
    except Exception as e:
        fatal("Failed to write temp_frame.png", e, tb=True)
    payload = {"frame_path": tmp_path, "bbox_xywh": bbox_xywh, "output_mask_path": output_mask_path}
    log(f"[API] POST {web_api_url} with {payload}")
    try:
        r = requests.post(web_api_url, json=payload, timeout=30)
        if r.status_code == 200:
            print(f"[OK] Mask saved at {output_mask_path}", flush=True)
            return True
        else:
            print(f"[ERR] API {r.status_code}: {r.text[:400]}", flush=True)
    except Exception as e:
        print(f"[EXC] POST failed: {e}", flush=True)
    return False

def load_bbox_from_file(bbox_path, fallback):
    try:
        with open(bbox_path, "r") as f:
            v = json.load(f)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            x, y, w, h = map(int, v)
            return [max(0,x), max(0,y), max(1,w), max(1,h)]
    except Exception as e:
        log(f"[BBox] failed to load {bbox_path}: {e}")
    return fallback

def main():
    global DEBUG
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox_xywh", type=json.loads, default="[420,200,180,200]")
    ap.add_argument("--output_mask_path", default="./tmp/mask.jpg")
    ap.add_argument("--web_api_url", default="http://localhost:9069/hq_sam")
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--trigger_path", default="./tmp/snap.trigger",
                    help="touch this file to send current frame+bbox to mask API")
    ap.add_argument("--bbox_path", default="./tmp/bbox.json",
                    help="write [x,y,w,h] here to update bbox live")
    ap.add_argument("--overlay_thickness", type=int, default=2)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--once", action="store_true", help="capture, request a mask, validate it and exit")
    ap.add_argument("--preview", action="store_true",
                    help="with --once, publish the live bounding box and wait for --trigger_path before capture")
    ap.add_argument("--viewer-bind", default="tcp://127.0.0.1:2223")
    ap.add_argument("--reuse-mask", action="store_true", help="validate the existing mask after placement; do not call SAM")
    ap.add_argument("--snapshot-path", default=None, help="absolute shared frame path for the SAM server")
    ap.add_argument("--registration-frame-path", default=None)
    ap.add_argument("--review-mask", action="store_true", help="show the generated mask and wait for acceptance")
    ap.add_argument("--mask-confirm-trigger", default=None)
    ap.add_argument("--camera-config", default=None)
    args = ap.parse_args()
    if (args.preview or args.reuse_mask) and not args.once:
        ap.error("--preview and --reuse-mask require --once")
    if args.review_mask and (not args.preview or not args.mask_confirm_trigger):
        ap.error("--review-mask requires --preview and --mask-confirm-trigger")
    DEBUG = args.debug

    print(f"[BOOT] headless camera_mask starting", flush=True)
    print(f"       CAM_SRC={CAM_SRC}, PUB_SRC={PUB_SRC}", flush=True)

    # discover + open camera
    try:
        devs = discover_camera_devices()
    except Exception as e:
        fatal("discover_camera_devices() crashed", e, tb=True)
    if not devs:
        fatal("No cameras found (discover_camera_devices returned empty).")
    print(f"[INFO] Found {len(devs)} device(s). Using index={args.camera_index}", flush=True)
    try:
        width, height, fps = 640, 480, 60
        calibration = None
        if args.camera_config:
            sys.path.insert(0, os.path.dirname(root_path))
            from ContactModelStudy.Hardware.Camera import load_camera_config, select_camera_device
            calibration = load_camera_config(args.camera_config)
            width, height = calibration["image_size"]
            fps = calibration["camera_fps"]
            cam_serial = select_camera_device(calibration, devs, args.camera_index)["serial"]
        else:
            cam_serial = devs[args.camera_index]["serial"]
        log(f"[INFO] Opening Camera serial={cam_serial}")
        cam = Camera(device_id=cam_serial, width=width, height=height, fps=fps, calibration=calibration)
    except Exception as e:
        fatal("Camera() failed to initialize RealSense pipeline", e, tb=True)

    # publisher
    pub = None
    if Publisher is not None and (not args.once or args.preview):
        try:
            pub = Publisher(endpoint=args.viewer_bind)
            print("[INFO] Publisher initialized.", flush=True)
        except Exception as e:
            if args.preview:
                cam.pipeline.stop()
                raise RuntimeError("Could not start the object-placement preview publisher") from e
            print(f"[WARN] Publisher init failed: {e}", flush=True)
            pub = None
    else:
        print("[INFO] Running without Publisher (still headless).", flush=True)

    # warm up frames
    print("[INFO] Grabbing warm-up frames...", flush=True)
    color_rgb = None
    try:
        for i in range(10):
            color_rgb, depth, ts = cam.getFrame()
            if color_rgb is None:
                time.sleep(0.02)
    except Exception as e:
        fatal("getFrame() crashed during warmup", e, tb=True)
    if color_rgb is None:
        fatal("Failed to grab any frame from camera.")

    if args.once:
        try:
            bbox = list(map(int, args.bbox_xywh))
            height, width = color_rgb.shape[:2]
            if (len(bbox) != 4 or bbox[0] < 0 or bbox[1] < 0 or min(bbox[2:]) <= 0
                    or bbox[0] + bbox[2] > width or bbox[1] + bbox[3] > height):
                raise ValueError("bbox_xywh must fit inside the captured image")
            while True:
                if args.preview:
                    if pub is None:
                        raise RuntimeError("Object-placement preview needs the image publisher")
                    print("[READY] Keep the cube inside the box and still; Space/Enter captures its mask.", flush=True)
                    while not os.path.isfile(args.trigger_path):
                        color_rgb, depth, ts = cam.getFrame()
                        vis_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                        x, y, w, h = bbox
                        cv2.rectangle(vis_bgr, (x, y), (x+w, y+h), (0, 255, 255), args.overlay_thickness)
                        cv2.putText(vis_bgr, "Place cube in box | Space / Enter: capture mask", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                        pub.publish(np.eye(4, dtype=np.float32), vis_bgr, time.time())
                captured_at = time.time()
                if not args.reuse_mask and not post_mask(color_rgb, bbox, args.output_mask_path, args.web_api_url, args.snapshot_path):
                    raise RuntimeError("SAM-HQ mask request failed")
                mask = cv2.imread(args.output_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != (height, width) or not (mask > 127).any() or (mask > 127).all():
                    raise RuntimeError("SAM-HQ returned a missing, empty or invalid mask")
                if args.registration_frame_path:
                    sys.path.insert(0, os.path.dirname(root_path))
                    from ContactModelStudy.Hardware.Camera import save_registration_frame
                    save_registration_frame(args.registration_frame_path, color_rgb, depth, captured_at, cam_serial)
                if not args.review_mask:
                    break
                confirm, retry = args.mask_confirm_trigger, args.mask_confirm_trigger + '.retry'
                for flag in (confirm, retry):
                    if os.path.exists(flag):
                        os.unlink(flag)
                overlay = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                selected = mask > 127
                overlay[selected] = (.6 * overlay[selected] + .4 * np.array([0, 255, 0])).astype(np.uint8)
                cv2.imwrite(args.output_mask_path + '.overlay.png', overlay)
                marker = np.eye(4, dtype=np.float32)
                marker[2, 3] = -1  # review snapshot, never a tracking pose
                print('[REVIEW] Green must cover the cube. Enter accepts; R recaptures. Keep cube still.', flush=True)
                while not os.path.isfile(confirm) and not os.path.isfile(retry):
                    pub.publish(marker, overlay, captured_at)
                    time.sleep(.03)
                if os.path.isfile(confirm) and not os.path.isfile(retry):
                    break
                for flag in (confirm, retry, args.trigger_path):
                    if os.path.exists(flag):
                        os.unlink(flag)
            print("[OK] Mask ready; releasing camera before tracking.", flush=True)
            return
        finally:
            try:
                cam.pipeline.stop()
            finally:
                if pub is not None:
                    pub.close()

    # initial bbox + file watchers
    bbox = list(map(int, args.bbox_xywh))
    last_trigger_mtime = None
    last_bbox_mtime = None

    print("\n[READY] Headless controls:", flush=True)
    print(f"  • SNAPSHOT to API: touch {args.trigger_path}", flush=True)
    print(f"  • UPDATE BBOX   : echo \"[x,y,w,h]\" > {args.bbox_path}", flush=True)
    print(f"  • Current bbox  : {bbox}", flush=True)

    frames = 0
    try:
        while True:
            try:
                color_rgb, depth, ts = cam.getFrame()
            except Exception as e:
                fatal("getFrame() crashed inside loop", e, tb=True)
            if color_rgb is None:
                log("[WARN] Empty frame, continuing")
                time.sleep(0.01)
                continue

            # watch bbox file
            if os.path.exists(args.bbox_path):
                m = os.path.getmtime(args.bbox_path)
                if last_bbox_mtime is None or m > last_bbox_mtime:
                    new_bbox = load_bbox_from_file(args.bbox_path, bbox)
                    if new_bbox != bbox:
                        bbox = new_bbox
                        print(f"[BBox] updated -> {bbox}", flush=True)
                    last_bbox_mtime = m

            # overlay bbox (so viewer sees it) — safe even without Publisher
            x, y, w, h = bbox
            H, W = color_rgb.shape[:2]
            x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
            w = max(1, min(w, W-x)); h = max(1, min(h, H-y))
            bbox = [x, y, w, h]
            vis_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(vis_bgr, (x, y), (x+w, y+h), (0,255,255), args.overlay_thickness)
            cv2.putText(vis_bgr, f"bbox={bbox}", (10, H-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

            # publish (pose = identity)
            if pub is not None:
                try:
                    pose_I = np.eye(4, dtype=np.float32)
                    pub.publish(pose_I, vis_bgr, ts)
                except Exception as e:
                    print(f"[WARN] Publish error (disabling publisher): {e}", flush=True)
                    pub = None
            
            last_frame_rgb = color_rgb
            last_bbox = bbox[:]

            # file trigger for snapshot
            if os.path.exists(args.trigger_path):
                m = os.path.getmtime(args.trigger_path)
                if last_trigger_mtime is None or m > last_trigger_mtime:
                    print(f"[Snap] trigger detected -> sending to API...", flush=True)
                    post_mask(color_rgb, bbox, args.output_mask_path, args.web_api_url)
                    last_trigger_mtime = m

            frames += 1
            if DEBUG and frames % 60 == 0:
                print(f"[LOOP] frames={frames}", flush=True)
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl-C received.", flush=True)
        try:
            if last_frame_rgb is not None:
                print("[EXIT] Capturing final snapshot before quitting...", flush=True)
                post_mask(last_frame_rgb, last_bbox, args.output_mask_path, args.web_api_url)
        except Exception as e:
            print(f"[EXIT] Snapshot on exit failed: {e}", flush=True)

    finally:
        try:
            cam.pipeline.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()


# import json
# import argparse
# import requests

# from camera import Camera, discover_camera_devices
# from typing import List
# import cv2
# import numpy as np


# def generate_mask(
#         frame: np.ndarray,
#         bbox_xywh: List,
#         output_mask_path: str,
#         web_api_url: str,
# ):
#     # save frame to a temporary file, rgb to bgr
#     frame_path = "tmp/temp_frame.png"
#     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(frame_path, frame_bgr)
#     data = {
#         'frame_path': frame_path,
#         'bbox_xywh': bbox_xywh,
#         "output_mask_path": output_mask_path,
#     }

#     try:
#         response = requests.post(web_api_url, json=data)

#         if response.status_code == 200:
#             print(f"The mask has been saved at {output_mask_path}")
#         else:
#             print(f"Error: {response.status_code}, {response.json()}")

#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--bbox_xywh", type=json.loads, default="[240, 80, 160, 240]")
#     parser.add_argument("--output_mask_path", default="./tmp/mask.jpg")
#     parser.add_argument("--web_api_url", type=str, default="http://localhost:9002/hq_sam")
#     parser.add_argument("--camera_index", type=int, default=0)
#     args = parser.parse_args()

#     available_devices = discover_camera_devices()
#     if not available_devices:
#         print("No cameras found. Exiting.")
#         exit(1)

#     camera_device_id = available_devices[args.camera_index]['serial']
#     d435i = Camera(device_id=camera_device_id)

#     for i in range(30):
#         color, depth, timestamp = d435i.getFrame()

#     generate_mask(
#         color,
#         args.bbox_xywh,
#         args.output_mask_path,
#         args.web_api_url,   
#     )
