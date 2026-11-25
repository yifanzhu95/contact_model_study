"""Measured-state rendering must not simulate motion or issue commands."""
import threading
from types import SimpleNamespace
from pathlib import Path

import mujoco
import numpy as np

from ContactModelStudy.Hardware.Scene import DEFAULT_SCENE
from ContactModelStudy.Hardware.Workflow import calibrate_scene
from view_leap_tracking import update_display, set_object_visual_mesh


def test_goal_errors_use_measured_pose_and_physical_target_not_corner_preview():
    from view_leap_tracking import goal_error_display
    q = np.r_[np.zeros(16), [.03, 0, 0], [np.sqrt(.5), 0, 0, np.sqrt(.5)]]
    target = dict(target_pos=[0, 0, 0], target_quat=[1, 0, 0, 0],
                  display_position_m=[10, 10, 10], success_thresholds=dict(pos=.02, quat=.04))
    errors, text = goal_error_display(q, target, dict(pos=.1, quat=1))
    assert np.isclose(errors['position_m'], .03)
    assert np.isclose(errors['orientation_deg'], 90)
    assert np.isclose(errors['orientation_threshold_deg'], 23.073918)
    assert not errors['position_within_threshold'] and not errors['orientation_within_threshold']
    assert '30.0 / 20.0 mm [outside]' in text
    assert '90.0 / 23.1 deg [outside]' in text
    # Equivalent quaternion signs must not show a 360-degree error.
    q[16:19] = [.001, 0, 0]
    q[19:23] = [-1, 0, 0, 0]
    errors, text = goal_error_display(q, target, dict(pos=.02, quat=.04))
    assert errors['position_within_threshold'] and errors['orientation_within_threshold']
    assert errors['orientation_deg'] == 0


def test_goal_errors_hide_stale_values_and_support_older_goal_files():
    from view_leap_tracking import goal_error_display
    q = np.r_[np.zeros(16), [.02, 0, 0], [1, 0, 0, 0]]
    target = dict(target_pos=[0, 0, 0], target_quat=[1, 0, 0, 0])
    errors, text = goal_error_display(q, target, dict(pos=.02, quat=.04))
    assert not errors['position_within_threshold']  # Strict inequality, like the controller.
    errors, text = goal_error_display(q, target, dict(pos=.02, quat=.04), valid=False)
    assert errors is None and '-- / 20.0 mm' in text and '-- / 23.1 deg' in text
    assert goal_error_display(q, None, dict(pos=.02, quat=.04))[0] is None
    q[19:23] = 0
    assert goal_error_display(q, target, dict(pos=.02, quat=.04))[0] is None


def test_controller_publishes_actual_success_thresholds_for_viewer(tmp_path):
    import json
    from run_leap_hardware import publish_goal
    task = SimpleNamespace(params={'target_pos': np.array([.02, .035, .08])},
                           getGoal=lambda: np.array([1, 0, 0, 0]),
                           SUCCESS_THRESHOLDS=dict(pos=.012, quat=.01, vel=3))
    path = tmp_path/'goal.json'
    publish_goal(path, task, 2, 17, 1000)
    record = json.loads(path.read_text())
    assert record['success_thresholds'] == task.SUCCESS_THRESHOLDS
    assert record['current_step'] == 17 and record['total_steps'] == 1000


def test_video_windows_use_actual_episode_times_and_confirm_only_elapsed_time(tmp_path):
    import json
    from run_leap_hardware import publish_goal,close_video_window
    task=SimpleNamespace(params={'target_pos':np.array([.02,.035,.08])},
        getGoal=lambda:np.array([1,0,0,0]),SUCCESS_THRESHOLDS=dict(pos=.02,quat=.04,vel=10))
    path=tmp_path/'goal.json'
    previous=dict(episode=0,started_at_utc='2026-11-16T05:00:00+00:00',
        finished_at_utc='2026-11-16T05:00:01+00:00',finish_reason='success',completed=True,n_steps=4)
    active=dict(episode=1,started_at_utc='2026-11-16T05:01:00+00:00',
        finish_reason='running',completed=False,n_steps=3)
    records=[previous,active]
    publish_goal(path,task,1,3,1000,records)
    message=json.loads(path.read_text())
    assert message['episode_windows'][0]['finished_at_utc']==previous['finished_at_utc']
    assert message['episode_windows'][0]['confirmed_through'] is None
    assert message['episode_windows'][1]['confirmed_through']==message['updated_at']
    assert message['episode_windows'][1]['finished_at_utc'] is None
    # The exception handler closes video before entering any user hold/reset.
    close_video_window(path,task,active,records,'interrupted')
    ended=json.loads(path.read_text())['episode_windows'][1]
    assert ended['finished_at_utc']==active['finished_at_utc']
    assert ended['outcome']=='interrupted' and ended['confirmed_through'] is None
    assert not active['completed']


def test_padded_windows_confirm_hold_time_and_close_immediately_on_interrupt(tmp_path):
    import json
    from run_leap_hardware import publish_goal,close_video_window
    task=SimpleNamespace(params={'target_pos':np.array([.02,.035,.08])},
        getGoal=lambda:np.array([1,0,0,0]),SUCCESS_THRESHOLDS=dict(pos=.02,quat=.04,vel=10))
    episode=dict(episode=0,n_steps=3,completed=True,finish_reason='success',
        started_at_utc='2026-11-21T10:00:01+00:00',finished_at_utc='2026-11-21T10:00:03+00:00',
        recording_started_at_utc='2026-11-21T10:00:00+00:00',recording_finished_at_utc=None)
    path=tmp_path/'goal.json'
    publish_goal(path,task,0,3,1000,[episode])
    record=json.loads(path.read_text());window=record['episode_windows'][0]
    assert window['confirmed_through']==record['updated_at']  # Tail still recording.
    assert window['finished_at_utc']==episode['finished_at_utc']
    close_video_window(path,task,episode,[episode],'interrupted')
    window=json.loads(path.read_text())['episode_windows'][0]
    assert window['recording_finished_at_utc'] is not None and window['confirmed_through'] is None
    assert episode['completed'] and episode['finish_reason']=='success'


def test_viewer_renders_measured_base_pose_and_hides_stale_object():
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    data = mujoco.MjData(model)
    geoms = np.flatnonzero(model.geom_bodyid == model.body('obj').id)
    opacity = model.geom_rgba[geoms, 3].copy()
    q = np.r_[np.full(16, 0.3), [0.15, -0.09, 0.04, 0.7071067812, 0, 0, 0.7071067812]]
    robot = SimpleNamespace(qpos=q[:16], qvel=np.zeros(16))
    buffer = SimpleNamespace(lock=threading.Lock(), robot=robot,
        snapshot=lambda **kwargs: (q.copy(), np.zeros(22), dict(pose_age=0.1, robot_age=0.01)))
    valid, _, _ = update_display(model, data, buffer, geoms, opacity, {})
    assert valid
    np.testing.assert_allclose(data.qpos, q)
    np.testing.assert_allclose(model.geom_rgba[geoms, 3], opacity)
    def stale(**kwargs):
        raise RuntimeError('Stale object feedback')
    buffer.snapshot = stale
    robot.qpos = np.full(16, 0.4)
    valid, status, _ = update_display(model, data, buffer, geoms, opacity, {})
    assert not valid and 'Stale object' in status
    np.testing.assert_allclose(data.qpos[:16], 0.4)
    np.testing.assert_allclose(model.geom_rgba[geoms, 3], 0)


def test_scan_visual_preserves_raw_mesh_frame_texture_and_physics(tmp_path):
    from scipy.spatial import cKDTree
    root = Path(__file__).resolve().parents[1]
    mesh = root / 'iris-foundationpose-plusplus/meshes/cube_70_mm_2_adjusted/cube_70_mm_2.obj'
    scene = tmp_path / 'scene.xml'
    calibrate_scene(root / 'scenes/leap/env_leap_eval_cube.xml', DEFAULT_SCENE.parent / 'calibrated_hand.xml', scene)
    original = mujoco.MjModel.from_xml_path(str(scene))
    set_object_visual_mesh(scene, mesh, 1.0)
    model = mujoco.MjModel.from_xml_path(str(scene))
    geom = model.geom('obj_visual').id
    index = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[index], model.mesh_vertnum[index]
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
    actual = model.mesh_vert[start:start+count] @ rotation.reshape(3, 3).T + model.geom_pos[geom]
    expected = np.array([[float(x) for x in line.split()[1:4]] for line in mesh.read_text().splitlines()
                         if line.startswith('v ')])
    assert cKDTree(expected).query(actual)[0].max() < 1e-7
    assert cKDTree(actual).query(expected)[0].max() < 1e-7
    assert (tmp_path / 'inspection_object_texture.png').is_file()
    for name in ('mass', 'inertia', 'ipos', 'iquat'):
        np.testing.assert_allclose(getattr(model.body('obj'), name), getattr(original.body('obj'), name))


def test_viewer_hides_hand_until_measured_states_and_when_they_become_stale():
    import time
    from ContactModelStudy.Hardware.LCM import FeedbackBuffer, RobotState
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    data = mujoco.MjData(model)
    object_geoms = np.flatnonzero(model.geom_bodyid == model.body('obj').id)
    robot_geoms = np.flatnonzero((model.geom_bodyid != 0) & (model.geom_bodyid != model.body('obj').id))
    opacity, robot_opacity = model.geom_rgba[object_geoms, 3].copy(), model.geom_rgba[robot_geoms, 3].copy()
    buffer = FeedbackBuffer(model)
    update_display(model, data, buffer, object_geoms, opacity, {}, robot_geoms, robot_opacity)
    np.testing.assert_array_equal(model.geom_rgba[robot_geoms, 3], 0)
    buffer.on_robot('STATE', RobotState(int(time.time()*1e6), np.full(16, .3), np.zeros(16)).encode())
    update_display(model, data, buffer, object_geoms, opacity, {}, robot_geoms, robot_opacity)
    np.testing.assert_allclose(data.qpos[:16], .3)
    np.testing.assert_array_equal(model.geom_rgba[robot_geoms, 3], robot_opacity)
    buffer.robot_received -= 1
    update_display(model, data, buffer, object_geoms, opacity, {}, robot_geoms, robot_opacity)
    np.testing.assert_array_equal(model.geom_rgba[robot_geoms, 3], 0)


def test_goal_preview_uses_controller_target_and_keeps_live_object_unchanged(tmp_path):
    import json
    from view_leap_tracking import add_goal_preview, update_goal_preview
    root = Path(__file__).resolve().parents[1]
    scene = tmp_path/'scene.xml'
    calibrate_scene(root/'scenes/leap/env_leap_eval_cube.xml', DEFAULT_SCENE.parent/'calibrated_hand.xml', scene)
    original = mujoco.MjModel.from_xml_path(str(scene))
    mesh = root/'iris-foundationpose-plusplus/meshes/cube_70_mm_2_adjusted/cube_70_mm_2.obj'
    set_object_visual_mesh(scene, mesh)
    add_goal_preview(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    q = data.qpos.copy()
    assert (model.nq, model.nv, model.nu) == (original.nq, original.nv, original.nu)
    visual, preview = model.geom('obj_visual').id, model.geom('inspection_goal_visual').id
    assert model.geom_dataid[visual] == model.geom_dataid[preview]
    assert model.geom_matid[visual] == model.geom_matid[preview]
    assert model.geom_contype[preview] == model.geom_conaffinity[preview] == 0
    mid = model.body('inspection_goal').mocapid[0]
    path = tmp_path/'goal.json'
    offset = np.array([.18, 0, .08])
    target, status = update_goal_preview(model, data, path, offset, 1)
    assert target is None and model.geom_rgba[preview, 3] == 0
    for episode, position, quat in [(0, [.02, .035, .085], [0, 0, 2, 0]),
                                     (1, [.03, .04, .11], [1, 0, 0, 0])]:
        path.write_text(json.dumps(dict(episode=episode, target_pos=position, target_quat=quat,
                                       current_step=episode*37, total_steps=1000,
                                       success_thresholds=dict(pos=.012, quat=.01))))
        target, status = update_goal_preview(model, data, path, offset, 1)
        np.testing.assert_allclose(data.mocap_pos[mid], np.asarray(position)+offset)
        np.testing.assert_allclose(data.mocap_quat[mid], np.asarray(quat)/np.linalg.norm(quat))
        np.testing.assert_array_equal(data.qpos, q)
        assert model.geom_rgba[preview, 3] == 1 and target['episode'] == episode
        assert 'offset for display only' in status
        assert f'Steps: {episode*37} / 1000' in status
        assert target['current_step'] == episode*37 and target['total_steps'] == 1000
        assert target['success_thresholds'] == dict(pos=.012, quat=.01)
    path.write_text('{broken')
    assert update_goal_preview(model, data, path, offset, 1)[0] is None
    assert model.geom_rgba[preview, 3] == 0


def test_calibrated_view_matches_realsense_extrinsics_and_off_center_intrinsics(tmp_path):
    from scipy.spatial.transform import Rotation
    from ContactModelStudy.Hardware.Camera import load_camera_config
    from view_leap_tracking import add_calibrated_camera, CALIBRATED_CAMERA
    root = Path(__file__).resolve().parents[1]
    calibration = load_camera_config(root/'configs/leap_pose_transform.json')
    before = np.eye(4)
    before[:3, :3] = Rotation.from_euler('xyz', [7, -4, 11], degrees=True).as_matrix()
    before[:3, 3] = [.01, -.02, .03]
    scene = tmp_path/'camera.xml'
    scene.write_text('<mujoco><worldbody/></mujoco>')
    add_calibrated_camera(scene, calibration, before)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = model.camera(CALIBRATED_CAMERA).id
    scn = mujoco.MjvScene(model, maxgeom=1)
    mujoco.mjv_updateCamera(model, data, camera, scn)
    gl = mujoco.mjv_averageCamera(scn.camera[0], scn.camera[1])
    transform = before @ calibration['camera_to_published']
    np.testing.assert_allclose(gl.pos, transform[:3, 3], atol=1e-6)
    np.testing.assert_allclose(gl.forward, transform[:3, 2], atol=1e-6)
    np.testing.assert_allclose(gl.up, -transform[:3, 1], atol=1e-6)
    width, height = calibration['image_size']
    K = calibration['camera_intrinsics']
    for cv_point in ([0, 0, .6], [.1, .03, .7], [-.08, -.07, .8]):
        cv_point = np.asarray(cv_point)
        world = transform[:3, :3] @ cv_point + transform[:3, 3]
        ray = world - gl.pos
        depth = np.dot(ray, gl.forward)
        right = np.cross(gl.forward, gl.up)
        x, y = np.dot(ray, right)*gl.frustum_near/depth, np.dot(ray, gl.up)*gl.frustum_near/depth
        pixel = [width*(x - gl.frustum_center + gl.frustum_width)/(2*gl.frustum_width),
                 height*(gl.frustum_top-y)/(gl.frustum_top-gl.frustum_bottom)]
        projected = K @ cv_point
        np.testing.assert_allclose(pixel, projected[:2]/projected[2], atol=1e-3)


def test_corner_preview_stays_in_upper_right_as_camera_and_viewport_change(tmp_path):
    import json
    from view_leap_tracking import add_goal_preview, corner_preview_position, update_goal_preview
    root = Path(__file__).resolve().parents[1]
    scene = tmp_path/'scene.xml'
    calibrate_scene(root/'scenes/leap/env_leap_eval_cube.xml', DEFAULT_SCENE.parent/'calibrated_hand.xml', scene)
    add_goal_preview(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    visual = model.geom('inspection_goal_visual').id
    mid = model.body('inspection_goal').mocapid[0]
    goal = tmp_path/'goal.json'
    goal.write_text(json.dumps(dict(episode=0, target_pos=[.02, .035, .085], target_quat=[0, 0, 1, 0])))
    qpos = data.qpos.copy()
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, .1]
    cam.distance = .8
    scn = mujoco.MjvScene(model, maxgeom=1)
    for azimuth, elevation, viewport in [(130, -35, (1200, 900)), (20, -60, (640, 480)), (230, -15, (500, 900))]:
        cam.azimuth, cam.elevation = azimuth, elevation
        mujoco.mj_forward(model, data)
        mujoco.mjv_updateCamera(model, data, cam, scn)
        gl = mujoco.mjv_averageCamera(scn.camera[0], scn.camera[1])
        target, _ = update_goal_preview(model, data, goal, None, 1, gl, viewport)
        mujoco.mj_forward(model, data)
        # The visual center must sit at normalized screen coordinates (.86, .86).
        ray = data.geom_xpos[visual]-gl.pos
        depth = np.dot(ray, gl.forward)
        near_x = np.dot(ray, np.cross(gl.forward, gl.up))*gl.frustum_near/depth
        near_y = np.dot(ray, gl.up)*gl.frustum_near/depth
        half_width = gl.frustum_width or (gl.frustum_top-gl.frustum_bottom)*viewport[0]/viewport[1]/2
        np.testing.assert_allclose((near_x-gl.frustum_center+half_width)/(2*half_width), .86, atol=1e-5)
        np.testing.assert_allclose((near_y-gl.frustum_bottom)/(gl.frustum_top-gl.frustum_bottom), .86, atol=1e-5)
        np.testing.assert_array_equal(data.qpos, qpos)
        assert target['preview_placement'] == 'upper_right'
        np.testing.assert_allclose(target['target_pos'], [.02, .035, .085])


def test_interactive_view_starts_at_camera_pose_and_allows_orbit_pan_zoom(tmp_path):
    from ContactModelStudy.Hardware.Camera import load_camera_config
    from view_leap_tracking import add_calibrated_camera, initialize_interactive_camera, CALIBRATED_CAMERA
    root = Path(__file__).resolve().parents[1]
    scene = tmp_path/'camera.xml'
    scene.write_text('<mujoco><worldbody><geom type="sphere" size=".01"/></worldbody></mujoco>')
    add_calibrated_camera(scene, load_camera_config(root/'configs/leap_pose_transform.json'), np.eye(4))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    initialize_interactive_camera(model, data, cam, [.02, .035, .085])
    camera_id = model.camera(CALIBRATED_CAMERA).id
    assert cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert cam.fixedcamid == -1 and cam.trackbodyid == -1
    scn = mujoco.MjvScene(model, maxgeom=1)
    mujoco.mjv_updateCamera(model, data, cam, scn)
    gl = mujoco.mjv_averageCamera(scn.camera[0], scn.camera[1])
    np.testing.assert_allclose(gl.pos, data.cam_xpos[camera_id], atol=1e-6)
    np.testing.assert_allclose(gl.forward, -data.cam_xmat[camera_id].reshape(3, 3)[:, 2], atol=1e-6)
    np.testing.assert_allclose(model.vis.global_.fovy, model.cam_fovy[camera_id], rtol=1e-6)
    assert model.vis.global_.fovy > 0
    initial = cam.azimuth, cam.elevation
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, .12, -.05, cam)
    assert (cam.azimuth, cam.elevation) != initial
    focus = cam.lookat.copy()
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_H, .12, .05, cam)
    assert not np.allclose(cam.lookat, focus)
    distance = cam.distance
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, .1, cam)
    assert cam.distance != distance
    changed = cam.lookat.copy(), cam.azimuth, cam.elevation, cam.distance
    for _ in range(3):
        mujoco.mj_forward(model, data)
        mujoco.mjv_updateCamera(model, data, cam, scn)
    np.testing.assert_array_equal(cam.lookat, changed[0])
    assert (cam.azimuth, cam.elevation, cam.distance) == changed[1:]


def test_weighted_fingertip_paths_are_colored_decorations_and_stale_paths_clear(tmp_path):
    import json
    from view_leap_tracking import draw_fingertip_paths, FINGERTIP_COLORS, FINGERTIP_SITES
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    scene = mujoco.MjvScene(model, maxgeom=100)
    points = np.arange(4*4*3).reshape(4,4,3)*.001
    record = dict(source='mppi_weighted_mean_rollout', status='ok', planned_at=100,
                  points=points.tolist(), time_offsets_s=[0,.064,.128,.192], site_names=list(FINGERTIP_SITES))
    path = tmp_path/'paths.json'
    path.write_text(json.dumps(record))
    info, status = draw_fingertip_paths(scene, path, now=100.1)
    assert info['horizon_s']==.192 and scene.ngeom==20
    assert 'Index red' in status and '0.192' in status
    for finger, color in enumerate(FINGERTIP_COLORS):
        np.testing.assert_allclose(scene.geoms[finger*5+3].pos, points[0,finger])
        np.testing.assert_allclose(scene.geoms[finger*5+4].pos, points[-1,finger])
        np.testing.assert_allclose(scene.geoms[finger*5].rgba, color, atol=1e-7)
    assert draw_fingertip_paths(scene, path, now=101)[0] is None and scene.ngeom==0
    record['points'][0][0][0] = float('nan')
    path.write_text(json.dumps(record))
    assert draw_fingertip_paths(scene, path, now=100.1)[0] is None and scene.ngeom==0
    path.write_text('[]')
    assert draw_fingertip_paths(scene, path, now=100.1)[0] is None and scene.ngeom==0


def test_long_fingertip_paths_fit_scene_capacity_and_keep_endpoints(tmp_path):
    import json
    from view_leap_tracking import draw_fingertip_paths, FINGERTIP_SITES
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
    scene = mujoco.MjvScene(model, maxgeom=40)
    points = np.arange(301*4*3).reshape(301,4,3)*.0001
    path = tmp_path/'paths.json'
    path.write_text(json.dumps(dict(source='mppi_weighted_mean_rollout', status='ok', planned_at=100,
        points=points.tolist(), time_offsets_s=(np.arange(301)*.064).tolist(), site_names=list(FINGERTIP_SITES))))
    info, _ = draw_fingertip_paths(scene, path, now=100.1)
    assert info is not None and scene.ngeom<=40
    per_finger = info['displayed_points']+1
    for finger in range(4):
        np.testing.assert_allclose(scene.geoms[(finger+1)*per_finger-1].pos, points[-1,finger])
