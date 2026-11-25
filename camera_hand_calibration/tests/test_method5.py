import numpy as np
import pytest
from geometry import matrix
from method5_alignment import ReferenceEditor,fit_calibration
from workspace import read_json,local


def config():return read_json('method5/config.json')


def test_numpad_edits_only_reference_with_world_axis_rotations_and_undo():
    editor=ReferenceEditor(config());source=matrix([.1,.2,.3,.02,.03,.08]);original=source.copy();editor.seed(source)
    assert editor.key(ord('4')) is None
    editor.key(326);assert editor.pose[0,3]==pytest.approx(source[0,3]+.001)
    editor.key(329);np.testing.assert_allclose(editor.pose,original)
    editor.key(325);editor.key(328)
    np.testing.assert_allclose(editor.pose[:3,:3],matrix([0,np.radians(1),0,0,0,0])[:3,:3]@source[:3,:3])
    editor.key(320);assert editor.step_index==2
    np.testing.assert_array_equal(source,original)


def test_full_six_dof_calibration_maps_same_camera_pose_to_manual_reference():
    truth=matrix([.12,-.31,.14,.44,.03,.52]);camera=matrix([.3,.1,-.2,.01,-.04,.65]);reference=truth@camera
    initial=matrix([.14,-.3,.15,.435,.034,.518]);poses=np.tile(camera,(40,1,1))
    fitted,metrics=fit_calibration(initial,poses,reference,config())
    np.testing.assert_allclose(fitted,truth,atol=1e-8)
    assert metrics['after']['position_rms_mm']<1e-5 and metrics['after']['orientation_rms_deg']<1e-5
    assert not metrics['hardware_commands_generated'] and not metrics['object_only_correction_applied']


def test_unstable_tracking_and_large_alignment_changes_are_rejected():
    camera=matrix([0,0,0,0,0,.65]);poses=np.tile(camera,(40,1,1));cfg=config()
    with pytest.raises(ValueError,match='more fresh'):fit_calibration(np.eye(4),poses[:2],camera,cfg)
    poses[:20,0,3]+=.02
    with pytest.raises(ValueError,match='unstable'):fit_calibration(np.eye(4),poses,camera,cfg)
    poses[:20,0,3]-=.02;reference=camera.copy();reference[0,3]=.2
    with pytest.raises(ValueError,match='large'):fit_calibration(np.eye(4),poses,reference,cfg)


def test_reference_model_is_separate_textured_mocap_without_new_robot_dofs():
    import mujoco
    from method5_viewer import editor_model
    out=local('cache/method5_test_scene');out.mkdir(exist_ok=True)
    m=editor_model(read_json('configs/live_tracking.json'),out)
    obj=m.geom('obj_visual').id;ref=m.geom('method5_reference_visual').id
    assert m.nq==23 and m.nv==22 and m.nu==16
    assert m.geom_type[ref]==mujoco.mjtGeom.mjGEOM_MESH and m.geom_dataid[ref]==m.geom_dataid[obj]
    assert m.body('method5_reference').mocapid[0]>=0 and m.geom_contype[ref]==0


def test_mouse_camera_controls_match_installed_mujoco_api():
    import mujoco
    from method5_viewer import move_camera
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.distance=1.
    move_camera(m,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05,cam)
    assert cam.distance!=1.
    angle=cam.azimuth
    move_camera(m,mujoco.mjtMouse.mjMOUSE_ROTATE_H,.05,0,cam)
    assert cam.azimuth!=angle
    before=cam.lookat.copy()
    move_camera(m,mujoco.mjtMouse.mjMOUSE_MOVE_H,.05,.02,cam)
    assert np.linalg.norm(cam.lookat-before)>0


def test_rotation_at_object_anchor_can_move_camera_origin_more_than_five_cm():
    from scipy.spatial.transform import Rotation
    camera=matrix([0,0,0,0,0,.6]);reference=camera.copy();reference[0,3]+=.004
    reference[:3,:3]=Rotation.from_euler('y',6,degrees=True).as_matrix()
    fitted,metrics=fit_calibration(np.eye(4),np.tile(camera,(40,1,1)),reference,config())
    np.testing.assert_allclose(fitted@camera,reference,atol=1e-8)
    assert metrics['camera_origin_displacement_m']>.05
    assert metrics['anchor_displacement_m']==pytest.approx(.004)


def test_anchor_correction_limit_still_rejects_large_cube_displacement():
    camera=matrix([0,0,0,0,0,.6]);reference=camera.copy();reference[0,3]+=.06
    with pytest.raises(ValueError,match='cube 60.0/50 mm'):
        fit_calibration(np.eye(4),np.tile(camera,(40,1,1)),reference,config())
