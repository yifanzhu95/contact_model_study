import argparse
import os
import torch
import json
import cv2
import sys
import numpy as np
import multiprocessing as mp
import time
from typing import List
import imageio.v2 as imageio  # Suppress DeprecationWarning
import trimesh
from scipy.spatial.transform import Rotation
from VOT import Cutie, Tracker_2D  
from utils.kalman_filter_6d import KalmanFilter6D
from utils.publisher import Publisher


from utils.camera import discover_camera_devices, Camera


src_path = os.path.join(os.path.dirname(__file__), "..")
foundationpose_path = os.path.join(src_path, "FoundationPose")
if src_path not in sys.path:
    sys.path.append(src_path)
if foundationpose_path not in sys.path:
    sys.path.append(foundationpose_path)


def get_sorted_frame_list(dir: str) -> List:
    files = os.listdir(dir)
    if not files:
        return []
    files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    if not files:
        return []
    if files[0].count('.') == 1:
        files.sort(key=lambda x: int(x.split('.')[0]))
    elif files[0].count('.') == 2:
        files.sort(key=lambda x: int(x.split('.')[0] + x.split('.')[1]))
    return files


def adjust_pose_to_image_point(
        ob_in_cam: torch.Tensor,
        K: torch.Tensor,
        x: float = -1.,
        y: float = -1.,
) -> torch.Tensor:
    """
    Adjusts the 6D pose(s) so that the projection matches the given 2D coordinate (x, y).

    Parameters:
    - ob_in_cam: Original 6D pose(s) as [4,4] or [B,4,4] tensor.
    - K: Camera intrinsic matrix (3x3 tensor).
    - x, y: Desired 2D coordinates on the image plane.

    Returns:
    - ob_in_cam_new: Adjusted pose(s) in same shape as input (tensor).
    """
    device = ob_in_cam.device
    dtype = ob_in_cam.dtype

    is_batched = ob_in_cam.ndim == 3
    if not is_batched:
        ob_in_cam = ob_in_cam.unsqueeze(0)  # [1, 4, 4]

    B = ob_in_cam.shape[0]
    ob_in_cam_new = torch.eye(4, device=device, dtype=dtype).repeat(B, 1, 1)

    for i in range(B):
        R = ob_in_cam[i, :3, :3]
        t = ob_in_cam[i, :3, 3]

        tx, ty = get_pose_xy_from_image_point(ob_in_cam[i], K, x, y)
        t_new = torch.tensor([tx, ty, t[2]], device=device, dtype=dtype)

        ob_in_cam_new[i, :3, :3] = R
        ob_in_cam_new[i, :3, 3] = t_new

    return ob_in_cam_new if is_batched else ob_in_cam_new[0]


def get_pose_xy_from_image_point(
        ob_in_cam: torch.Tensor, 
        K: torch.Tensor, 
        x: float = -1., 
        y: float = -1.,
) -> tuple:
    """
    Computes new (tx, ty) in camera space such that the projection matches image point (x, y).

    Parameters:
    - ob_in_cam: 4x4 pose tensor.
    - K: 3x3 intrinsic matrix tensor.
    - x, y: Desired image coordinates.

    Returns:
    - tx, ty: New x/y in camera coordinate system.
    """

    is_batched = ob_in_cam.ndim == 3
    if is_batched:
        ob_in_cam_new = ob_in_cam[0].cpu()  # [1, 4, 4]
    else:
        ob_in_cam_new = ob_in_cam.cpu()

    if x == -1. or y == -1.:
        return x, y
    
    t = ob_in_cam_new[:3, 3]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    tz = t[2]

    tx = (x - cx) * tz / fx
    ty = (y - cy) * tz / fy

    return tx, ty


def project_3d_to_2d(point_3d_homogeneous, K, ob_in_cam):
    # Transform point to camera frame
    point_cam = ob_in_cam @ point_3d_homogeneous

    # Perspective division to get normalized image coordinates
    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]

    # Apply camera intrinsics
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]

    return (int(u), int(v))


def get_mat_from_6d_pose_arr(pose_arr):
    # 提取位移 (xyz)
    xyz = pose_arr[:3]
    
    # 提取欧拉角
    euler_angles = pose_arr[3:]
    
    # 从欧拉角生成旋转矩阵
    rotation = Rotation.from_euler('xyz', euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()
    
    # 创建 4x4 变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz
    
    return transformation_matrix


def get_6d_pose_arr_from_mat(pose):
    if torch.is_tensor(pose):
        is_batched = pose.ndim == 3
        if is_batched:
            pose_np = pose[0].cpu().numpy()
        else:
            pose_np = pose.cpu().numpy()
    else:
        pose_np = pose

    xyz = pose_np[:3, 3]
    rotation_matrix = pose_np[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    return np.r_[xyz, euler_angles]


def pose_track(
    mesh_path: str,
    init_mask_path: str,
    cam_K: np.ndarray,
    mask_visualization_path: str,
    bbox_visualization_path: str,
    pose_visualization_path: str,
    est_refine_iter: int,
    track_refine_iter: int,
    activate_2d_tracker: bool = False,
    activate_kalman_filter: bool = False,
    camera_index: int = 0,
    
):



    #################################################
    # Initialize RealSense (camera-only mode)
    #################################################
    available_devices = discover_camera_devices()
    if not available_devices:
        print("No cameras found. Exiting.")
        exit(1)

    camera_device_id = available_devices[camera_index]['serial']
    d435i = Camera(device_id=camera_device_id)


    # initialize publisher and create a for loop to publish pose and frame
    pub = Publisher()

    # while True:
    #     # Read the frame from RealSense
    #     color, depth, color_timestamp = d435i.getFrame()
    #     color_pose=np.eye(4, dtype=np.float32)
    #     pub.publish(color_pose, color, color_timestamp)




    #################################################
    # Read the initial mask
    #################################################
    init_mask = cv2.imread(init_mask_path, cv2.IMREAD_GRAYSCALE)
    if init_mask is None:
        print(f"Failed to read mask file {init_mask_path}.")
        return
    init_mask = init_mask.astype(bool)



    #################################################
    # Load the mesh
    #################################################
    from FoundationPose.estimater import trimesh_add_pure_colored_texture
    
    mesh_file = os.path.join(mesh_path)
    if not os.path.exists(mesh_file):
        print(f"Mesh file not found.")
        return
    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    # Convert units to meters
    mesh.apply_scale(args.apply_scale)
    if args.force_apply_color:
        mesh = trimesh_add_pure_colored_texture(mesh, color=np.array(args.apply_color), resolution=10)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    #################################################
    # Instantiate the 6D pose estimator
    #################################################

    from FoundationPose.estimater import (
        ScorePredictor,
        PoseRefinePredictor,
        dr,
        FoundationPose,
        logging,
        draw_posed_3d_box,
        draw_xyz_axis,
    )
    import logging

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
    )
    logging.info("Estimator initialization done")

    #################################################
    # Instantiate the 2D tracker
    #################################################

    if activate_2d_tracker:     # Default using Cutie as a 2D tracker
        tracker_2D = Cutie()
    else:
        tracker_2D = Tracker_2D()


    #################################################
    # 6D pose tracking
    #################################################

    if activate_kalman_filter:
        kf = KalmanFilter6D(args.kf_measurement_noise_scale)

    # pose_seq = [None] * total_frames  # Initialize as None
    kf_mean, kf_covariance = None, None

    # Forward processing from initial frame
    frame_idx = 0
    try:
        while True:
            #################################################
            # Read the frame from RealSense
            #################################################
            color, depth, color_timestamp = d435i.getFrame()


            #################################################
            # 6D pose tracking
            #################################################

            if frame_idx == 0:
                mask = init_mask.astype(np.uint8) * 255
                pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
                if activate_kalman_filter:
                    kf_mean, kf_covariance = kf.initiate(get_6d_pose_arr_from_mat(pose))

                mask_visualization_color_filename = None
                bbox_visualization_color_filename = None
                frame_color_filename = f"{frame_idx:06d}.jpg"
                
                if mask_visualization_path is not None:
                    os.makedirs(mask_visualization_path, exist_ok=True)
                    mask_visualization_color_filename = os.path.join(mask_visualization_path, frame_color_filename)
                if bbox_visualization_path is not None:
                    os.makedirs(bbox_visualization_path, exist_ok=True)
                    bbox_visualization_color_filename = os.path.join(bbox_visualization_path, frame_color_filename)

                if activate_2d_tracker:
                    tracker_2D.initialize(
                        color, 
                        init_info={"mask": init_mask}, 
                        mask_visualization_path=mask_visualization_color_filename, 
                        bbox_visualization_path=bbox_visualization_color_filename
                    )

                input("Press Enter to start tracking...")

            else:
                start = time.time()

                mask_visualization_color_filename = None
                bbox_visualization_color_filename = None
                frame_color_filename = f"{frame_idx:06d}.jpg"
                
                if mask_visualization_path is not None:
                    os.makedirs(mask_visualization_path, exist_ok=True)
                    mask_visualization_color_filename = os.path.join(mask_visualization_path, frame_color_filename)
                if bbox_visualization_path is not None:
                    os.makedirs(bbox_visualization_path, exist_ok=True)
                    bbox_visualization_color_filename = os.path.join(bbox_visualization_path, frame_color_filename)

                if activate_2d_tracker:
                    bbox_2d = tracker_2D.track(
                        color,
                        mask_visualization_path=mask_visualization_color_filename,
                        bbox_visualization_path=bbox_visualization_color_filename
                    )
                # TODO: get occluded mask
                # adjusted_last_pose = adjust_pose_to_image_point(ob_in_cam=pose, K=cam_K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2)
                if activate_2d_tracker:
                    if not activate_kalman_filter:
                        est.pose_last = adjust_pose_to_image_point(ob_in_cam=est.pose_last, K=cam_K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2)
                    else:
                        # using kf to estimate the 6d estimation of the last pose
                        kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance, get_6d_pose_arr_from_mat(est.pose_last))
                        measurement_xy = np.array(get_pose_xy_from_image_point(ob_in_cam=est.pose_last, K=cam_K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2))
                        kf_mean, kf_covariance = kf.update_from_xy(kf_mean, kf_covariance, measurement_xy)
                        est.pose_last = torch.from_numpy(get_mat_from_6d_pose_arr(kf_mean[:6])).unsqueeze(0).to(est.pose_last.device)

                pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=track_refine_iter)
                if activate_2d_tracker and activate_kalman_filter:
                    # use kf to predict from last pose, and update kf status
                    kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)     # kf is alway one step behind
                calculation_time = time.time() - start
                fps = 1 / calculation_time
                print(f"FPS: {fps:.2f} ({(time.time() - start)*1000:.2f} ms)")
                # Open the text file in write mode ('w' will overwrite the file if it exists)
                with open('output.txt', 'w') as f:
                    f.write(f"Cal time = {calculation_time * 1000:.2f} ms, FPS: {fps:.2f}\n")
            center_pose = pose @ np.linalg.inv(to_origin)
            vis_color = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis_color = draw_xyz_axis(
                vis_color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=cam_K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )

            # Publish pose and frame (if publisher available)
            try:
                if pub is not None:
                    # ensure pose is numpy float32 4x4
                    pose_np = np.array(pose).astype(np.float32).reshape(4, 4)
                    # convert RGB -> BGR for cv2 encoding (publisher encodes with cv2)
                    try:
                        color_to_send = cv2.cvtColor(vis_color, cv2.COLOR_RGB2BGR)
                    except Exception:
                        color_to_send = color
                    pub.publish(pose_np, color_to_send, color_timestamp)
            except Exception as e:
                print('Publish error, disabling publisher:', e)
                pub = None

            # pose_seq[frame_idx] = pose.reshape(4, 4)

            frame_idx += 1

            if pose_visualization_path is not None:
                # depth_normalized = cv2.normalize(np.clip(depth, 0, 900), None, 0, 255, cv2.NORM_MINMAX)
                # depth_8bit = depth_normalized.astype(np.uint8)
                # depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

                center_pose = pose @ np.linalg.inv(to_origin)
                vis_color = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis_color = draw_xyz_axis(
                    vis_color,
                    ob_in_cam=center_pose,
                    scale=0.1,
                    K=cam_K,
                    thickness=3,
                    transparency=0,
                    is_input_rgb=True,
                )
    
                
                if not os.path.exists(pose_visualization_path):
                    os.makedirs(pose_visualization_path, exist_ok=True)
                
                pose_visualization_color_filename = os.path.join(pose_visualization_path, frame_color_filename)
                imageio.imwrite(
                    pose_visualization_color_filename, vis_color
                )

    finally:
        torch.cuda.empty_cache()
        # Stop RealSense pipeline if used
        d435i.pipeline.stop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mesh_path", type=str, default="/workspace/yanwenhao/detection/test_case2/mesh/1x4.stl")
    parser.add_argument("--init_mask_path", type=str, default="/FoundationPose-plus-plus/tmp/mask.jpg")
    parser.add_argument("--cam_K", type=json.loads, default="[[607.229248046875, 0, 328.500396728516], [0, 606.978149414062, 243.475112915039], [0.0, 0.0, 1.0]]", help="Camera intrinsic parameters")
    parser.add_argument("--pose_visualization_path", type=str, default=None, help="Path to save mask visualization images")
    parser.add_argument("--est_refine_iter", type=int, default=10, help="FoundationPose initial refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--track_refine_iter", type=int, default=5, help="FoundationPose tracking refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--activate_2d_tracker", action='store_true', help="activate 2d tracker")
    parser.add_argument("--activate_kalman_filter", action='store_true', help="activate kalman_filter")
    parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05, help="The scale of measurement noise relative to prediction in kalman filter, greater value means more filtering. Only effective if activate_kalman_filter")
    parser.add_argument("--apply_scale", type=float, default=1.0, help="Mesh scale factor in meters (1.0 means no scaling), commonly use 0.01")
    parser.add_argument("--force_apply_color", action='store_true', help="force a color for colorless mesh")
    parser.add_argument("--apply_color", type=json.loads, default="[0, 159, 237]", help="RGB color to apply, in format 'r,g,b'. Only effective if force_apply_color")
    parser.add_argument("--camera_index", type=int, help="Index of the camera to use among discovered devices", default=0)
    args = parser.parse_args()

    pose_track(
        args.mesh_path,
        args.init_mask_path,
        np.array(args.cam_K),
        None,        # args.mask_visualization_path,
        None,        # args.bbox_visualization_path,
        None,        # args.pose_visualization_path,
        args.est_refine_iter,
        args.track_refine_iter,
        args.activate_2d_tracker,
        args.activate_kalman_filter,
        args.camera_index,
    )

    torch.cuda.empty_cache()
