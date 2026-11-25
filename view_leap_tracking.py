#!/usr/bin/env python3
"""Display measured LEAP joints and a tracked object in calibrated MuJoCo world.

Subscribes to LEAP_STATE and POSE only. No simulation stepping, planner, serial
port, motor command or torque control. Q/Escape/window close exits the viewer.
Run leap_hand_lcm_bridge.py --state-only for a read-only joint-state source.
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
import shlex

import mujoco
import mujoco.viewer
import numpy as np

from ContactModelStudy.Drivers.run_episodes import TASKS
from ContactModelStudy.Hardware.Calibration import verify_model_tag
from ContactModelStudy.Hardware.Camera import DEFAULT_CAMERA_CONFIG, load_camera_config
from ContactModelStudy.Hardware.LCM import DEFAULT_URL, LCMFeedback, load_pose_transform
from ContactModelStudy.Hardware.Workflow import calibrate_scene
from ContactModelStudy.Tasks.LeapReorient import FINGERTIP_SITES, LeapReorientConfig
from ContactModelStudy.Tasks.TaskBase import TaskRole

ROOT = Path(__file__).resolve().parent
CALIBRATED_CAMERA = 'inspection_realsense'
FINGERTIP_COLORS = ((1., .2, .2, .95), (.2, 1., .2, .95),
                   (.1, .8, 1., .95), (1., .85, .1, .95))


def draw_fingertip_paths(scene, path, max_age=.5, now=None):
    """Draw weighted-plan world-space predictions as decorations only."""
    scene.ngeom = 0
    if path is None:
        return None, 'MPPI fingertip paths disabled'
    try:
        record = json.loads(Path(path).read_text())
        if not isinstance(record, dict):
            raise ValueError('Invalid prediction record')
        if record.get('status') == 'error':
            return None, 'MPPI fingertip prediction unavailable: ' + str(record.get('error', 'unknown error'))[:160]
        if record.get('source') != 'mppi_weighted_mean_rollout' or record.get('status') != 'ok':
            raise ValueError('Unknown prediction source')
        age = (time.time() if now is None else now) - float(record['planned_at'])
        points, times = np.asarray(record['points'], dtype=float), np.asarray(record['time_offsets_s'], dtype=float)
        if (list(record['site_names']) != list(FINGERTIP_SITES)
                or points.ndim != 3 or points.shape[1:] != (4, 3) or len(points) < 2
                or times.shape != (len(points),) or not np.isfinite(points).all()
                or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0)
                or not np.isfinite(age) or age < -.1 or age > max_age):
            raise ValueError('Stale or invalid prediction')
    except (OSError, ValueError, KeyError, TypeError):
        return None, 'MPPI fingertip paths: waiting for a fresh weighted plan'
    # Preserve the full horizon even when a long path must be thinned to fit
    # the viewer's decoration capacity. Each finger also has two end markers.
    capacity = scene.maxgeom//4 - 1
    if capacity < 2:
        return None, 'MPPI fingertip paths: insufficient viewer geometry capacity'
    indices = np.unique(np.linspace(0, len(points)-1, min(len(points), capacity)).astype(int))
    displayed = points[indices]
    for finger, color in enumerate(FINGERTIP_COLORS):
        rgba = np.asarray(color, dtype=np.float32)
        for start, end in zip(displayed[:-1, finger], displayed[1:, finger]):
            if np.linalg.norm(end-start) < 1e-8:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), start, np.eye(3).ravel(), rgba)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3., start, end)
            scene.ngeom += 1
        for point, radius in ((displayed[0, finger], .0015), (displayed[-1, finger], .003)):
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, radius), point, np.eye(3).ravel(), rgba)
            scene.ngeom += 1
    info = dict(plan_age_s=age, horizon_s=float(times[-1]), step=record.get('step'),
                episode=record.get('episode'), rollout_model=record.get('rollout_model'),
                computation_time_s=record.get('computation_time_s'), displayed_points=len(displayed))
    status = ('WEIGHTED MPPI FINGERTIPS\nIndex red | Middle green | Ring cyan | Thumb yellow\n'
              f'Horizon {times[-1]:.3f}s | Plan age {age*1000:.0f}ms')
    return info, status


def add_calibrated_camera(scene, calibration, before):
    """Place a render-only camera at the calibrated color camera viewpoint."""
    tree = ET.parse(scene)
    # OpenCV: right/down/forward. MuJoCo camera: right/up/backward.
    selected=calibration.get('selected_calibration')
    transform = (np.asarray(selected['T_world_from_camera']) if selected is not None
                 else before @ np.asarray(calibration['camera_to_published']))
    rotation = transform[:3, :3] @ np.diag([1., -1., -1.])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    K = np.asarray(calibration['camera_intrinsics'])
    width, height = calibration['image_size']
    def numbers(values):
        return ' '.join(format(float(value), '.17g') for value in values)
    ET.SubElement(tree.getroot().find('worldbody'), 'camera', name=CALIBRATED_CAMERA,
        pos=numbers(transform[:3, 3]), quat=numbers(quat), ipd='0',
        resolution=f'{width} {height}', sensorsize=numbers([width * 1e-6, height * 1e-6]),
        focalpixel=numbers([K[0, 0], K[1, 1]]),
        # MuJoCo frustum offsets have the opposite sign to OpenCV's
        # image principal-point displacement; retain the off-center K.
        principalpixel=numbers([width / 2 - K[0, 2], height / 2 - K[1, 2]]))
    tree.write(scene, encoding='unicode')


def initialize_interactive_camera(model, data, camera, focus):
    """Start a free camera at the color camera pose, orbiting near the hand.

    MuJoCo's free camera uses world-up rather than the sensor's roll/principal
    point. Set it once so subsequent mouse orbit/pan/zoom remains untouched.
    """
    camera_id = model.camera(CALIBRATED_CAMERA).id
    position = data.cam_xpos[camera_id].copy()
    forward = -data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
    distance = max(float(np.dot(np.asarray(focus) - position, forward)), 0.1)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = position + distance * forward
    camera.distance = distance
    camera.azimuth = np.rad2deg(np.arctan2(forward[1], forward[0]))
    camera.elevation = np.rad2deg(np.arcsin(np.clip(forward[2], -1, 1)))
    model.vis.global_.fovy = model.cam_fovy[camera_id]
    model.vis.global_.ipd = 0


def corner_preview_position(model, geom, quat, camera, viewport, fraction=0.18):
    """Fit the goal's bounding sphere in a padded upper-right screen region."""
    near = camera.frustum_near
    bottom, top = camera.frustum_bottom, camera.frustum_top
    aspect = max(viewport[0], 1) / max(viewport[1], 1)
    half_width = camera.frustum_width or (top - bottom) * aspect / 2
    left = camera.frustum_center - half_width
    span_x, span_y = 2 * half_width / near, (top - bottom) / near
    radius = max(float(model.geom_rbound[geom]), 1e-5)
    # Use depth - radius for the perspective bound, so even the nearest
    # vertex stays inside the requested footprint when the target rotates.
    depth = radius + 2 * radius / (fraction * min(span_x, span_y))
    x = (left + 0.86 * 2 * half_width) / near * depth
    y = (bottom + 0.86 * (top - bottom)) / near * depth
    forward = np.asarray(camera.forward, dtype=float)
    forward /= np.linalg.norm(forward)
    up = np.asarray(camera.up, dtype=float)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    center = camera.pos + forward * depth + right * x + up * y
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    # Mesh compilation can move the geom's center relative to its raw origin.
    return center - rotation.reshape(3, 3) @ model.geom_pos[geom]


def set_object_visual_mesh(scene, mesh, scale=1.0):
    """Keep the tracked mesh's raw origin, axes, metric scale and MTL texture."""
    scene, mesh = Path(scene).resolve(), Path(mesh).expanduser().resolve()
    if not mesh.is_file() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Tracked mesh must exist and mesh scale must be finite and positive')
    tree = ET.parse(scene)
    root = tree.getroot()
    visual = root.find(".//geom[@name='obj_visual']")
    if visual is None:
        raise ValueError('Inspection scene has no obj_visual geometry')
    asset = root.find('asset')
    ET.SubElement(asset, 'mesh', name='inspection_object_mesh', file=str(mesh),
                  scale=' '.join([str(scale)]*3))
    visual.set('mesh', 'inspection_object_mesh')
    visual.set('pos', '0 0 0')
    visual.set('quat', '1 0 0 0')
    visual.set('rgba', '1 1 1 1')
    visual.attrib.pop('material', None)
    texture = None
    if mesh.suffix.lower() == '.obj':
        for line in mesh.read_text().splitlines():
            if line.startswith('mtllib '):
                material = mesh.parent / shlex.split(line)[1]
                for field in material.read_text().splitlines():
                    if field.startswith('map_Kd '):
                        texture = material.parent / shlex.split(field)[-1]
                        break
                if texture is not None:
                    break
    if texture is not None:
        # MuJoCo consumes PNG textures; preserve the OBJ's actual JPEG pixels.
        if texture.suffix.lower() != '.png':
            import cv2
            image = cv2.imread(str(texture))
            if image is None:
                raise ValueError(f'Could not load tracked mesh texture: {texture}')
            texture = scene.parent / 'inspection_object_texture.png'
            if not cv2.imwrite(str(texture), image):
                raise OSError('Could not convert object texture to PNG')
        ET.SubElement(asset, 'texture', name='inspection_object_texture', type='2d', file=str(texture))
        ET.SubElement(asset, 'material', name='inspection_object_material',
                      texture='inspection_object_texture', texuniform='false')
        visual.set('material', 'inspection_object_material')
    tree.write(scene, encoding='unicode')


def update_display(model, data, buffer, object_geoms, opacity, limits,
                   robot_geoms=None, robot_opacity=None):
    """Copy measured states without stepping; hide geometry with stale feedback."""
    with buffer.lock:
        if buffer.robot is not None:
            data.qpos[:16] = buffer.robot.qpos
            data.qvel[:16] = buffer.robot.qvel
            data.act[:] = buffer.robot.qpos
            data.ctrl[:] = buffer.robot.qpos
    if robot_geoms is not None:
        try:
            buffer.snapshot_robot(max_robot_age=limits.get('max_robot_age', 0.15))
            model.geom_rgba[robot_geoms, 3] = robot_opacity
        except RuntimeError:
            model.geom_rgba[robot_geoms, 3] = 0
    try:
        q, v, metadata = buffer.snapshot(**limits)
        data.qpos[:], data.qvel[:] = q, v
        model.geom_rgba[object_geoms, 3] = opacity
        status = (f'OBJECT IN MUJOCO BASE: {q[16]:+.3f}, {q[17]:+.3f}, {q[18]:+.3f} m\n'
                  f'Pose age {metadata["pose_age"]:.2f}s | joint age {metadata["robot_age"]:.2f}s')
        valid = True
    except RuntimeError as exc:
        model.geom_rgba[object_geoms, 3] = 0
        status, metadata, valid = str(exc), {}, False
    mujoco.mj_forward(model, data)
    return valid, status, metadata


def add_goal_preview(scene):
    """Copy the tracked visual into a separate, nonphysical mocap body."""
    tree = ET.parse(scene)
    root = tree.getroot()
    visual = root.find(".//geom[@name='obj_visual']")
    if visual is None:
        raise ValueError('Goal preview requires the object visual geometry')
    body = ET.SubElement(root.find('worldbody'), 'body', name='inspection_goal', mocap='true')
    preview = copy.deepcopy(visual)
    preview.set('name', 'inspection_goal_visual')
    for name, value in dict(contype='0', conaffinity='0', density='0', group='2').items():
        preview.set(name, value)
    preview.attrib.pop('mass', None)
    body.append(preview)
    tree.write(scene, encoding='unicode')


def update_goal_preview(model, data, path, offset, opacity, camera=None, viewport=None):
    """Read the controller's adopted target; never infer or sample a goal here."""
    geom = model.geom('inspection_goal_visual').id
    try:
        if path is None:
            raise ValueError('No controller goal file')
        record = json.loads(Path(path).read_text())
        position = np.asarray(record['target_pos'], dtype=float)
        quat = np.asarray(record['target_quat'], dtype=float)
        if (position.shape != (3,) or quat.shape != (4,) or not np.isfinite(position).all()
                or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8):
            raise ValueError('Invalid controller goal')
        quat = quat / np.linalg.norm(quat)
    except (OSError, ValueError, TypeError, KeyError):
        model.geom_rgba[geom, 3] = 0
        return None, 'GOAL PREVIEW: waiting for controller target'
    mid = model.body('inspection_goal').mocapid[0]
    in_corner = offset is None and camera is not None
    display_position = (corner_preview_position(model, geom, quat, camera, viewport) if in_corner
                        else position + np.asarray([0.18, 0, 0.08] if offset is None else offset))
    data.mocap_pos[mid] = display_position
    data.mocap_quat[mid] = quat
    model.geom_rgba[geom, 3] = opacity
    target = dict(target_pos=position.tolist(), target_quat=quat.tolist(),
                  display_offset_m=(display_position - position).tolist(),
                  display_position_m=display_position.tolist(),
                  preview_placement='upper_right' if in_corner else 'world_offset', episode=record.get('episode'))
    if 'success_thresholds' in record:
        target['success_thresholds'] = record['success_thresholds']
    placement = 'upper-right corner; display only' if in_corner else 'offset for display only'
    status = (f'GOAL PREVIEW ({placement})\n'
              + 'Target base XYZ: ' + ', '.join(f'{value:+.3f}' for value in position) + ' m\n'
              + 'Target WXYZ: ' + ', '.join(f'{value:+.3f}' for value in quat))
    current, total = record.get('current_step'), record.get('total_steps')
    if type(current) is int and type(total) is int and 0 <= current <= total and total > 0:
        target.update(current_step=current, total_steps=total)
        status += f'\nSteps: {current} / {total}'
    return target, status


def goal_error_display(qpos, target, thresholds, valid=True):
    """Compare live base-frame object pose with the physical goal, not its preview."""
    try:
        limits = target.get('success_thresholds', thresholds) if target is not None else thresholds
        pos_limit, quat_limit = float(limits['pos']), float(limits['quat'])
        if not (np.isfinite(pos_limit) and pos_limit > 0 and np.isfinite(quat_limit) and 0 < quat_limit <= 1):
            raise ValueError('Invalid success thresholds')
        angle_limit = float(np.degrees(2*np.arcsin(np.sqrt(quat_limit))))
    except (TypeError, KeyError, ValueError):
        return None, 'GOAL ERRORS: waiting for success thresholds'
    unavailable = (f'Position error: -- / {pos_limit*1000:.1f} mm\n'
                   f'Orientation error: -- / {angle_limit:.1f} deg (waiting for live pose/goal)')
    if not valid or target is None:
        return None, unavailable
    try:
        position = np.asarray(qpos[16:19], dtype=float)
        quat = np.asarray(qpos[19:23], dtype=float)
        goal_pos = np.asarray(target['target_pos'], dtype=float)
        goal_quat = np.asarray(target['target_quat'], dtype=float)
        if (position.shape != (3,) or goal_pos.shape != (3,) or quat.shape != (4,) or goal_quat.shape != (4,)
                or not all(np.isfinite(value).all() for value in (position, goal_pos, quat, goal_quat))
                or np.linalg.norm(quat) < 1e-8 or np.linalg.norm(goal_quat) < 1e-8):
            raise ValueError('Invalid pose')
        dot = float(np.clip(abs(np.dot(quat/np.linalg.norm(quat), goal_quat/np.linalg.norm(goal_quat))), 0, 1))
        pos_error = float(np.linalg.norm(position-goal_pos))
        quat_error = 1-dot*dot
        angle = float(np.degrees(2*np.arccos(dot)))
    except (TypeError, KeyError, ValueError):
        return None, unavailable
    position_met, orientation_met = pos_error < pos_limit, quat_error < quat_limit
    metrics = dict(position_m=pos_error, quaternion_error=quat_error, orientation_deg=angle,
                   position_threshold_m=pos_limit, orientation_threshold_deg=angle_limit,
                   position_within_threshold=position_met, orientation_within_threshold=orientation_met)
    text = (f'Position error: {pos_error*1000:.1f} / {pos_limit*1000:.1f} mm '
            f'[{"within" if position_met else "outside"}]\n'
            f'Orientation error: {angle:.1f} / {angle_limit:.1f} deg '
            f'[{"within" if orientation_met else "outside"}]')
    return metrics, text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=sorted(TASKS), default='cube_reorient')
    parser.add_argument('--camera-config', type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument('--calibrated-hand', type=Path, default=ROOT / 'scenes/leap/hardware/calibrated_hand.xml')
    parser.add_argument('--model', type=Path, help='same custom scene override used by the controller')
    parser.add_argument('--lcm-url', default=DEFAULT_URL)
    parser.add_argument('--state-channel', default='LEAP_STATE')
    parser.add_argument('--pose-channel', default='POSE')
    parser.add_argument('--status-file', type=Path)
    parser.add_argument('--goal-file', type=Path, help='current goal JSON written by the controller')
    parser.add_argument('--fingertip-path-file', type=Path, help='background weighted-mean fingertip prediction')
    parser.add_argument('--goal-offset', nargs=3, type=float, default=None,
                        metavar=('X', 'Y', 'Z'), help='optional world-offset preview instead of the default screen corner')
    parser.add_argument('--max-robot-age', type=float, default=0.15)
    parser.add_argument('--max-pose-age', type=float, default=0.3)
    parser.add_argument('--max-state-skew', type=float, default=0.15)
    parser.add_argument('--mesh', type=Path, help='use the tracker OBJ and its texture as object visual geometry')
    parser.add_argument('--mesh-scale', type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.goal_offset is not None and not np.isfinite(args.goal_offset).all():
        parser.error('--goal-offset must be finite')
    for name in ('max_robot_age', 'max_pose_age', 'max_state_skew'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and positive')
    stopped = threading.Event()

    def key(keycode):
        if keycode in (27, 256, ord('q'), ord('Q')):
            stopped.set()

    task = TASKS[args.task](LeapReorientConfig(role=TaskRole.EVAL))
    before, after, calibration = load_pose_transform(args.camera_config)
    camera_calibration = load_camera_config(args.camera_config, require_extrinsics=True)
    feedback = None
    viewer_threads = []
    try:
        with tempfile.TemporaryDirectory(prefix='leap_tracking_viewer_') as directory:
            path = calibrate_scene(args.model or task.getModelPath(), args.calibrated_hand,
                                   Path(directory) / 'scene.xml')
            if args.mesh:
                set_object_visual_mesh(path, args.mesh, args.mesh_scale)
            add_goal_preview(path)
            add_calibrated_camera(path, camera_calibration, before)
            model = mujoco.MjModel.from_xml_path(path)
            verify_model_tag(calibration, model)
            data = mujoco.MjData(model)
            data.qpos[:] = task.getInitialState()[0]
            # Hide the reorientation goal marker; this is a measured-pose display.
            goal = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'goal')
            if goal >= 0:
                model.geom_rgba[model.geom_bodyid == goal, 3] = 0
            # The physical table is not the simulation's assumed floor height.
            # Hide its plane so measured objects below that plane remain visible.
            model.geom_rgba[model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE, 3] = 0
            obj = model.body('obj').id
            preview = model.body('inspection_goal').id
            preview_geom = model.geom('inspection_goal_visual').id
            preview_opacity = model.geom_rgba[preview_geom, 3].copy()
            model.geom_rgba[preview_geom, 3] = 0
            object_geoms = np.flatnonzero(model.geom_bodyid == obj)
            robot_geoms = np.flatnonzero((model.geom_bodyid != 0) & (model.geom_bodyid != obj)
                                        & (model.geom_bodyid != goal) & (model.geom_bodyid != preview))
            robot_opacity = model.geom_rgba[robot_geoms, 3].copy()
            model.geom_rgba[robot_geoms, 3] = 0
            opacity = model.geom_rgba[object_geoms, 3].copy()
            model.geom_rgba[object_geoms, 3] = 0
            feedback = LCMFeedback(model, args.lcm_url, args.state_channel, args.pose_channel,
                                   before=before, after=after)
            limits = dict(max_robot_age=args.max_robot_age, max_pose_age=args.max_pose_age,
                          max_skew=args.max_state_skew)
            mujoco.mj_forward(model, data)
            camera_scene = mujoco.MjvScene(model, maxgeom=1)
            existing_threads = set(threading.enumerate())
            with mujoco.viewer.launch_passive(model, data, key_callback=key,
                    show_left_ui=False, show_right_ui=False) as viewer:
                viewer_threads = [thread for thread in threading.enumerate()
                                  if thread not in existing_threads]
                with viewer.lock():
                    initialize_interactive_camera(model, data, viewer.cam, task.params['target_pos'])
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
                print('VIEWER READY: read-only LEAP joints and tracked object in MuJoCo base frame.', flush=True)
                print('Mouse: left-drag orbit; right-drag pan; scroll zoom.', flush=True)
                print('Waiting for LEAP_STATE and POSE. Q/Escape or close window exits.', flush=True)
                last_print = 0.0
                while viewer.is_running() and not stopped.is_set():
                    with viewer.lock():
                        mujoco.mjv_updateCamera(model, data, viewer.cam, camera_scene)
                        render_camera = mujoco.mjv_averageCamera(camera_scene.camera[0], camera_scene.camera[1])
                        viewport = viewer.viewport
                        goal_target, goal_status = update_goal_preview(model, data, args.goal_file,
                            args.goal_offset, preview_opacity, render_camera, (viewport.width, viewport.height))
                        valid, status, metadata = update_display(model, data, feedback.buffer,
                                                                 object_geoms, opacity, limits,
                                                                 robot_geoms, robot_opacity)
                        if valid:
                            fingertip_info, fingertip_status = draw_fingertip_paths(viewer.user_scn, args.fingertip_path_file)
                        else:
                            viewer.user_scn.ngeom = 0
                            fingertip_info, fingertip_status = None, 'MPPI fingertip paths: waiting for live feedback'
                    goal_errors, error_status = goal_error_display(data.qpos, goal_target, task.SUCCESS_THRESHOLDS, valid)
                    viewer.set_texts([(None, None, 'READ ONLY: measured joints + tracked object\n' + status + '\n' + error_status, None),
                                      (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, goal_status, None),
                                      (None, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, fingertip_status, None)])
                    viewer.sync()
                    now = time.time()
                    if args.status_file:
                        args.status_file.parent.mkdir(parents=True, exist_ok=True)
                        record = dict(displayed_at=now, tracking_valid=valid, status=status,
                                      qpos=data.qpos.tolist(), feedback=metadata, goal=goal_target,
                                      goal_errors=goal_errors,
                                      fingertip_prediction=fingertip_info)
                        temporary = args.status_file.with_suffix('.tmp')
                        temporary.write_text(json.dumps(record, allow_nan=False) + '\n')
                        temporary.replace(args.status_file)
                    if now - last_print > 2:
                        print(status, flush=True)
                        last_print = now
                    time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        # Handle.close() requests exit but does not join MuJoCo's daemon render
        # thread. Let it destroy its GL context before Python terminates GLFW.
        for thread in viewer_threads:
            thread.join(timeout=5)
        if feedback is not None:
            feedback.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
