#!/usr/bin/env python3
# Remeber to run this on the local machine where you want to visualize the data. The serever is headless.

import os
import sys
pybind_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(f'{pybind_dir}')
sys.path.insert(0, os.path.dirname(pybind_dir))
from ContactModelStudy.Hardware.Camera import DEFAULT_CAMERA_CONFIG, load_camera_config

import cv2
import time
import zmq
import trimesh
import copy
import threading
import numpy as np
import argparse
import json
from pathlib import Path

class Subscriber:
    def __init__(self, port=2223, host='127.0.0.1', endpoint=None):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.connect(endpoint or f"tcp://{host}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self._cfg_img_height = 480
        self._cfg_img_width = 640

    def receive(self, timeout_ms=20):
        if not self._socket.poll(timeout_ms, zmq.POLLIN):
            return None
        recv_msg = self._socket.recv_multipart(0)
        if len(recv_msg) != 3:
            raise ValueError('Expected pose, JPEG frame and epoch timestamp')

        # decode pose
        pose = np.frombuffer(recv_msg[0],dtype=np.float32).reshape(4,4)

        # decode color
        color = cv2.imdecode(np.frombuffer(recv_msg[1], dtype=np.uint8), cv2.IMREAD_COLOR)   
        if color is None or not np.isfinite(pose).all():
            raise ValueError('Invalid supervision frame or pose')


        timestamp_recv = np.frombuffer(recv_msg[2],dtype=np.float64).item()

        # print("Decode time: ", (time.time() - decode_time)*1000, 'ms')
        
        return pose, color, timestamp_recv

    def close(self):
        self._socket.close(linger=0)
        self._context.term()
    



def project_3d_to_2d(pt,K,ob_in_cam):
    pt = pt.reshape(4,1)
    projected = K @ ((ob_in_cam@pt)[:3,:])
    projected = projected.reshape(-1)
    projected = projected/(projected[2])
    return projected.reshape(-1)[:2].round().astype(int)

def draw_ground_axis(color, K=np.eye(3)):
    T_hand_in_cam = np.linalg.inv(T_cam_in_ground)

    transparency=0
    thickness=3
    scale = 0.05

    xx = np.array([1,0,0,1]).astype(float)
    yy = np.array([0,1,0,1]).astype(float)
    zz = np.array([0,0,1,1]).astype(float)
    xx[:3] = xx[:3]*scale
    yy[:3] = yy[:3]*scale
    zz[:3] = zz[:3]*scale
    origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, T_hand_in_cam))
    xx = tuple(project_3d_to_2d(xx, K, T_hand_in_cam))
    yy = tuple(project_3d_to_2d(yy, K, T_hand_in_cam))
    zz = tuple(project_3d_to_2d(zz, K, T_hand_in_cam))
    arrow_len = 0
    tmp = color.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, xx, color=(0,0,255), thickness=thickness,line_type=cv2.FILLED, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, yy, color=(0,255,0), thickness=thickness,line_type=cv2.FILLED, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, zz, color=(255,0,0), thickness=thickness,line_type=cv2.FILLED, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    tmp = tmp.astype(np.uint8)
    return tmp

def draw_xyz_axis(color, ob_in_cam, scale=0.1, K=np.eye(3), thickness=3, transparency=0.3,is_input_rgb=False):
  '''
  @color: BGR
  '''

  if is_input_rgb:
    color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
  xx = np.array([1,0,0,1]).astype(float)
  yy = np.array([0,1,0,1]).astype(float)
  zz = np.array([0,0,1,1]).astype(float)
  xx[:3] = xx[:3]*scale
  yy[:3] = yy[:3]*scale
  zz[:3] = zz[:3]*scale
  origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, ob_in_cam))
  xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
  yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
  zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
  line_type = cv2.FILLED
  arrow_len = 0

  tmp = color
  arrows = np.zeros_like(tmp)
  # Draw the arrows on the blank image
  arrows = cv2.arrowedLine(arrows, origin, xx, color=(0, 0, 255), thickness=thickness, line_type=line_type, tipLength=arrow_len)
  arrows = cv2.arrowedLine(arrows, origin, yy, color=(0, 255, 0), thickness=thickness, line_type=line_type, tipLength=arrow_len)
  arrows = cv2.arrowedLine(arrows, origin, zz, color=(255, 0, 0), thickness=thickness, line_type=line_type, tipLength=arrow_len)
  mask = np.any(arrows > 0, axis=-1)
  tmp[mask] = tmp[mask] * transparency + arrows[mask] * (1 - transparency)

  tmp = tmp.astype(np.uint8)
  if is_input_rgb:
    tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)

  return tmp


def visFn(color, mask):
    vis_blended = color

    mask_rows, mask_cols = np.nonzero(mask)
    y_min, y_max = mask_rows.min(), mask_rows.max()
    x_min, x_max = mask_cols.min(), mask_cols.max()

    vis_blended = cv2.rectangle(vis_blended, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return vis_blended




def main(argv=None):
    global T_cam_in_ground
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_index', type=int, default=0, help='Camera device index (0, 1, etc.)')
    parser.add_argument('--camera-config', default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument('--cam_K', type=json.loads, default=None,
                        help='override camera intrinsics from the shared config')
    parser.add_argument('--draw-axes', action='store_true', help='project object axes using configured intrinsics')
    parser.add_argument('--endpoint', default=None)
    parser.add_argument('--mesh', help='tracked mesh, in object coordinates, for bounding-box overlay')
    parser.add_argument('--mesh-scale', type=float, default=1.0)
    parser.add_argument('--ready-file', type=Path,
                        help='write heartbeat after displaying each received frame (used by launcher)')
    parser.add_argument('--capture-trigger', type=Path,
                        help='Space/Enter requests the initial mask during object-placement preview')
    parser.add_argument('--mask-confirm-trigger', type=Path,
                        help='Enter accepts the green SAM mask; R requests a new capture')
    parser.add_argument('--reuse-mask', action='store_true', help='placement confirmation reuses the existing mask')
    args = parser.parse_args(argv)
    calibration = load_camera_config(args.camera_config, intrinsics=args.cam_K)
    K = calibration['camera_intrinsics']
    T_cam_in_ground = calibration['camera_to_published']

    receiver = Subscriber(port=2223 + args.camera_index, endpoint=args.endpoint)
    bounds = None
    if args.mesh:
        # Only geometry is needed for the overlay. Scene.bounds also handles
        # textured OBJ groups without loading/copying materials on the host.
        bounds = trimesh.load(args.mesh, process=False, skip_materials=True).bounds * args.mesh_scale
    window = 'LEAP tracking supervision - Q / Esc to stop'
    # dyn_odom_sub = dynOdomSubscriber(3443, 'dyn_odom_sub')

    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, *calibration['image_size'])
        last_frame, timestamp_recv = None, None
        stage = 'placement'
        while True:
            message = receiver.receive()
            if message is not None:
                pose, color, timestamp_recv = message
                # The placement stream deliberately sends identity at zero depth;
                # tracked object poses have positive depth in the camera frame.
                stage = 'tracking' if pose[2, 3] > 0 else ('mask_review' if pose[2, 3] < 0 else 'placement')
                if args.draw_axes and stage == 'tracking':
                    color = draw_xyz_axis(color, pose, K=K)
                if bounds is not None and stage == 'tracking':
                    corners = np.array([[bounds[(i >> 0) & 1, 0], bounds[(i >> 1) & 1, 1],
                                         bounds[(i >> 2) & 1, 2], 1] for i in range(8)])
                    if np.all((pose @ corners.T)[2] > 0):
                        pixels = [tuple(project_3d_to_2d(point, K, pose)) for point in corners]
                        for i in range(8):
                            for bit in (1, 2, 4):
                                if i < (i ^ bit):
                                    cv2.line(color, pixels[i], pixels[i ^ bit], (0, 255, 0), 2)
                last_frame = color
            if last_frame is not None:
                display = last_frame.copy()
                age = time.time() - timestamp_recv
                action = 'start tracking' if args.reuse_mask else 'capture mask'
                prompt = (f'Place object in box | Space / Enter: {action}'
                          if stage == 'placement' else 'Tracking | Q / Esc: stop experiment')
                if stage == 'mask_review':
                    prompt = 'Check green mask covers cube | Enter: accept | R: retry | Keep cube still'
                if stage == 'placement' and args.capture_trigger is not None and args.capture_trigger.exists():
                    prompt = ('Starting tracking with existing mask | Q / Esc: stop' if args.reuse_mask
                              else 'Generating mask / starting tracking | Q / Esc: stop')
                age_label = 'snapshot age' if stage == 'mask_review' else 'age'
                cv2.putText(display, f'{prompt} | {age_label} {age:.2f}s',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0) if age < 1 or stage == 'mask_review' else (0, 0, 255), 1)
                cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), ord('Q'), 27):
                break
            if (key in (32, 10, 13) and stage == 'placement' and last_frame is not None
                    and args.capture_trigger is not None and time.time() - timestamp_recv < 1):
                args.capture_trigger.parent.mkdir(parents=True, exist_ok=True)
                args.capture_trigger.write_text(json.dumps(dict(frame_timestamp=timestamp_recv)))
            if stage == 'mask_review' and args.mask_confirm_trigger is not None:
                flag = (args.mask_confirm_trigger if key in (10, 13)
                        else Path(str(args.mask_confirm_trigger) + '.retry') if key in (ord('r'), ord('R')) else None)
                if flag is not None:
                    flag.parent.mkdir(parents=True, exist_ok=True)
                    flag.write_text(json.dumps(dict(frame_timestamp=timestamp_recv)))
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            if message is not None and args.ready_file is not None:
                args.ready_file.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.ready_file.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(displayed_at=time.time(), frame_timestamp=timestamp_recv,
                                                     stage=stage)))
                temporary.replace(args.ready_file)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
