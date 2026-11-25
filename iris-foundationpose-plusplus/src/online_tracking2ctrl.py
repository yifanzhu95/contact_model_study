import argparse
import os
import torch
import json
import cv2
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ContactModelStudy.Hardware.Camera import DEFAULT_CAMERA_CONFIG, load_camera_config
from ContactModelStudy.Hardware.VisionRuntime import startup_stage, install_bounded_mesh_diameter, normalize_tracker_pose
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
from utils.lcm_publisher import LCMPublisher


from utils.camera import discover_camera_devices, Camera


src_path = os.path.join(os.path.dirname(__file__), "..")
foundationpose_path = os.environ.get("FOUNDATIONPOSE_ROOT", os.path.join(src_path, "FoundationPose"))
if src_path not in sys.path:
    sys.path.append(src_path)
if os.environ.get("FOUNDATIONPOSE_ROOT"):
    # A prepared Docker image can supply weights and compiled extensions while
    # this checkout supplies camera calibration and LCM/preview integration.
    sys.path.insert(0, os.path.dirname(os.path.abspath(foundationpose_path)))
    sys.path.insert(0, os.path.abspath(foundationpose_path))
elif foundationpose_path not in sys.path:
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


def calculate_object_in_base(T_object_in_cam):
    """Apply a camera-axis translation tweak before the calibrated base transform."""
    T = args.camera_config_data["camera_to_published"]

    # 5. Multiply each matrix one by one sequentially
    if hasattr(T_object_in_cam, 'detach'):
        T_object_in_cam = T_object_in_cam.detach().cpu().numpy()
    camera_pose = np.array(T_object_in_cam, dtype=float, copy=True).reshape(4, 4)
    # This correction affects published control/inspection poses only. Keep raw
    # camera poses for pixel overlays and tracking, without shifting the tag mount.
    camera_pose[:3, 3] += args.camera_config_data.get("object_translation_adjust_camera_m", [0, 0, 0])
    step2 = T @ camera_pose

    # 6. Apply the exact output cast structure from track_one
    return step2.reshape(4, 4)


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
    # FoundationPose builds float32 CUDA poses and uses K without casting in
    # crop projection. JSON calibration is float64; normalize at this boundary.
    cam_K = np.asarray(cam_K, dtype=np.float32)

    #################################################
    # Initialize RealSense (camera-only mode)
    #################################################
    available_devices = discover_camera_devices()
    if not available_devices:
        print("No cameras found. Exiting.")
        exit(1)

    from ContactModelStudy.Hardware.Camera import select_camera_device
    camera_device_id = select_camera_device(args.camera_config_data, available_devices, camera_index)['serial']
    width, height = args.camera_config_data["image_size"]
    d435i = Camera(device_id=camera_device_id, width=width, height=height,
                  fps=args.camera_config_data["camera_fps"], calibration=args.camera_config_data)


    # initialize publisher and create a for loop to publish pose and frame
    pub = Publisher(endpoint=args.viewer_bind)
    # lcm_pub = LCMPublisher(channel_name="POSE")
    lcm_pub = LCMPublisher(channel_name=args.pose_channel, provider=args.lcm_url)

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
    init_mask = init_mask > 127  # JPEG background artifacts must not become foreground
    if init_mask.shape != (height, width):
        raise ValueError('Initial mask size differs from the calibrated camera frame')
    if not init_mask.any() or init_mask.all():
        raise ValueError('Initial object mask must contain foreground and background')



    #################################################
    # Load the mesh
    #################################################
    with startup_stage('FoundationPose imports'):
        import FoundationPose.estimater as estimator_module
        install_bounded_mesh_diameter(estimator_module)
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
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)

    with startup_stage('score network'):
        scorer = ScorePredictor()
    with startup_stage('refinement network'):
        refiner = PoseRefinePredictor()
    with startup_stage('CUDA rasterizer'):
        glctx = dr.RasterizeCudaContext()
    with startup_stage('object geometry and pose candidates'):
        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
        )
    logging.info("Estimator initialization done")
    centered_from_object=est.get_tf_to_centered_mesh().detach().cpu().numpy()

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
            if frame_idx == 0 and args.registration_frame_path:
                from ContactModelStudy.Hardware.Camera import load_registration_frame
                color, depth, frame_epoch_seconds = load_registration_frame(
                    args.registration_frame_path, args.camera_config_data['image_size'],
                    args.camera_config_data.get('camera_serial'))
                print('Registering from the exact RGB-D frame paired with the accepted mask.', flush=True)
            else:
                color, depth, color_timestamp = d435i.getFrame()
                # Timestamp before inference; initial registration uses the
                # original mask capture timestamp, never a fabricated fresh one.
                frame_epoch_seconds = time.time()


            #################################################
            # 6D pose tracking
            #################################################

            if frame_idx == 0:
                mask = init_mask.astype(np.uint8) * 255
                with startup_stage('initial object registration'):
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

                # input("Press Enter to start tracking...")

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
                if args.verbose or frame_idx % 30 == 0:
                    print(f"FPS: {fps:.2f} ({(time.time() - start)*1000:.2f} ms)", flush=True)
                # Open the text file in write mode ('w' will overwrite the file if it exists)
                if args.verbose:
                    with open('output.txt', 'w') as f:
                        f.write(f"Cal time = {calculation_time * 1000:.2f} ms, FPS: {fps:.2f}\n")
            # Prevent float32 matrix-composition drift from accumulating over a
            # persistent tracking session. Both publications and the next-frame
            # estimator cache use this same corrected rigid pose.
            pose,cached_pose=normalize_tracker_pose(pose,centered_from_object)
            est.pose_last=torch.as_tensor(cached_pose,dtype=est.pose_last.dtype,
                device=est.pose_last.device).reshape(est.pose_last.shape)
            # center_pose = pose @ np.linalg.inv(to_origin)
            center_pose = pose
            
            # Debug: Print pose values to verify they're correct
            if args.verbose and frame_idx % 30 == 0:
                print(f"\n=== Pose Debug (frame {frame_idx}) ===")
                print(f"center_pose type: {type(center_pose)}")
                print(f"center_pose shape: {center_pose.shape if hasattr(center_pose, 'shape') else 'N/A'}")
                print(f"center_pose:\n{center_pose}")
                print("=" * 40 + "\n")
            
            # vis_color = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose @ np.linalg.inv(to_origin), bbox=bbox)
            # vis_color = draw_xyz_axis(
            #     vis_color,
            #     ob_in_cam=center_pose,
            #     scale=0.1,
            #     K=cam_K,
            #     thickness=3,
            #     transparency=0,
            #     is_input_rgb=True,
            # )

            # Publish pose via LCM
            try:
                # print("center_pose:\n", center_pose)
                # lcm_pub.publish_pose(center_pose, timestamp=color_timestamp)
                pose_transformed = calculate_object_in_base(center_pose)
                lcm_pub.publish_pose(pose_transformed, timestamp=frame_epoch_seconds)
            except Exception as e:
                print(f'LCM publish error: {e}')

            # The viewer needs the object pose in CAMERA coordinates for K-based
            # overlays, while LCM above carries the transformed base-frame pose.
            pose_np = (center_pose.detach().cpu().numpy() if hasattr(center_pose, 'detach')
                       else np.asarray(center_pose))
            pub.publish(pose_np, cv2.cvtColor(color, cv2.COLOR_RGB2BGR), frame_epoch_seconds)

            # pose_seq[frame_idx] = pose.reshape(4, 4)

            frame_idx += 1

            if pose_visualization_path is not None:
                # depth_normalized = cv2.normalize(np.clip(depth, 0, 900), None, 0, 255, cv2.NORM_MINMAX)
                # depth_8bit = depth_normalized.astype(np.uint8)
                # depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

                # center_pose = pose @ np.linalg.inv(to_origin)
                center_pose = pose
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
        # Close LCM publisher
        if 'lcm_pub' in locals():
            lcm_pub.close()
        pub.close()
        # Stop RealSense pipeline if used
        d435i.pipeline.stop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--mesh_path", type=str, default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../scenes/leap/hardware/cube_tracking.obj")))
    parser.add_argument("--init_mask_path", type=str, default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../tmp/mask.jpg")))
    parser.add_argument("--registration-frame-path", default=None,
                        help="RGB-D NPZ captured with the initial mask; used only for registration")
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG),
                        help="shared camera intrinsics, extrinsics and image size JSON")
    parser.add_argument("--cam_K", type=json.loads, default=None, help="override camera_intrinsics from config")
    parser.add_argument("--pose_visualization_path", type=str, default=None, help="Path to save mask visualization images")
    parser.add_argument("--est_refine_iter", type=int, default=10, help="FoundationPose initial refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--track_refine_iter", type=int, default=5, help="FoundationPose tracking refine iterations, see https://github.com/NVlabs/FoundationPose")
    parser.add_argument("--activate_2d_tracker", action='store_true', help="activate 2d tracker")
    parser.add_argument("--activate_kalman_filter", action='store_true', help="activate kalman_filter")
    parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05, help="The scale of measurement noise relative to prediction in kalman filter, greater value means more filtering. Only effective if activate_kalman_filter")
    parser.add_argument("--apply_scale", type=float, default=1.0, help="Mesh scale factor in meters (1.0 means no scaling), commonly use 0.01")
    parser.add_argument("--force_apply_color", action='store_true', help="force a color for colorless mesh")
    parser.add_argument("--apply_color", type=json.loads, default="[149, 206, 50]", help="RGB color to apply, in format 'r,g,b'. Only effective if force_apply_color") #[149, 206, 50] for lime green
    parser.add_argument("--camera_index", type=int, help="Index of the camera to use among discovered devices", default=0)
    parser.add_argument("--pose-channel", default="POSE")
    parser.add_argument("--lcm-url", default="udpm://239.255.76.67:7667?ttl=1")
    parser.add_argument("--viewer-bind", default="tcp://127.0.0.1:2223",
                        help="ZMQ pose/image endpoint for the supervision receiver")
    parser.add_argument("--verbose", action="store_true", help="detailed per-frame FoundationPose diagnostics")
    parser.add_argument("--camera-to-published", type=json.loads, default=None,
                        help="4x4 camera-to-published-frame matrix as JSON; default existing calibration")
    args = parser.parse_args()
    args.camera_config_data = load_camera_config(args.camera_config, args.cam_K, args.camera_to_published,
                                               require_extrinsics=True)

    pose_track(
        args.mesh_path,
        args.init_mask_path,
        args.camera_config_data["camera_intrinsics"],
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
