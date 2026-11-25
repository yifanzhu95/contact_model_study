import numpy as np
import pytest
from method4_reference import (level_pose,palm_stopped_cube,refine_translation,recover_raw_camera_pose,cube_orientation_check)
from articulated_motion import model
from geometry import matrix


def test_level_support_and_two_palm_stops_define_cube_position():
    m=model();q=level_pose(m,inward_yaw=0.);center,geometry=palm_stopped_cube(m,q)
    np.testing.assert_allclose(q[[1,5,9]],0.,atol=1e-12)
    assert geometry['max_support_tilt_deg']<.01
    assert geometry['support_height_spread_mm']<.1
    assert max(geometry['corner_to_finite_palm_face_error_mm'])<.01
    assert abs(center[2]-geometry['bottom_height_m']-.035)<1e-10
    corners=np.array(geometry['lower_rear_corners_world_m'])
    np.testing.assert_allclose(corners@geometry['palm_stop_normal_world'],geometry['palm_stop_plane_offset_m'],atol=1e-10)


def test_translation_anchor_rejects_outliers_and_preserves_rotation():
    rng=np.random.default_rng(4);truth=matrix([.2,-.1,.1,.4,.02,.5]);initial=truth.copy();initial[:3,3]+=[.004,-.002,.003]
    poses=np.tile(matrix([.1,.2,-.1,.02,-.04,.65]),(60,1,1));poses[:,:3,3]+=rng.normal(0,.0002,(60,3))
    mesh_center=np.array([-.0005,-.0004,-.0005]);camera_centers=np.einsum('nij,j->ni',poses[:,:3,:3],mesh_center)+poses[:,:3,3]
    known=camera_centers@truth[:3,:3].T+truth[:3,3];poses[:5,:3,3]+=.03
    fitted,stats=refine_translation(initial,poses,known,mesh_center)
    np.testing.assert_allclose(fitted,truth,atol=1e-10);np.testing.assert_array_equal(fitted[:3,:3],initial[:3,:3])
    assert stats['inlier_samples']==55 and not stats['object_only_correction_used']
    with pytest.raises(ValueError,match='Not enough'):refine_translation(initial,poses[:2],known[:2],mesh_center)


def test_tracking_offset_is_removed_before_method4_calibration():
    raw=matrix([.2,-.1,.3,.02,.01,.64]);publication=matrix([.1,.2,.3,.4,.05,.5]);offset=np.array([0,0,-.015]);corrected=raw.copy();corrected[:3,3]+=offset
    np.testing.assert_allclose(recover_raw_camera_pose(publication@corrected,publication,offset),raw,atol=1e-12)


def test_resting_face_check_detects_tilt_without_updating_rotation():
    initial=matrix([.2,-.1,.3,.4,.05,.5]);raw=np.linalg.inv(initial)@matrix([0,0,0,.03,.04,.08])
    assert cube_orientation_check(initial,[raw])['resting_face_normal_max_deg']<1e-5
    tilted=np.linalg.inv(initial)@matrix([.1,0,0,.03,.04,.08]);result=cube_orientation_check(initial,[tilted])
    assert result['resting_face_normal_max_deg']>5 and not result['rotation_updated']


def test_capture_requires_a_stable_cube_window():
    from method4_capture import object_stability
    poses=[dict(raw_camera_object_pose=matrix([0,0,0,.01,.02,.65]).tolist()) for _ in range(60)]
    assert object_stability(poses)['position_std_m']<1e-12
    poses[-1]['raw_camera_object_pose']=matrix([.2,0,0,.03,.02,.65]).tolist()
    assert object_stability(poses)['position_std_m']>.0015
    assert object_stability(poses)['orientation_rms_deg']>1.


def test_close_resting_fingers_are_separated_before_normal_motion():
    from method4_reference import approach_path
    m=model();start=np.array([-.35895157,.14419413,-.2393012,.80227184,-.35434961,-.0306797,-.34207773,.86823297,
        -.34054375,-.15493226,-.42031074,-.20248556,-.36048555,-.18407774,.78693199,.32213593])
    waypoints,screening=approach_path(m,start,level_pose(m,0.))
    assert screening['entry_retreat'] and screening['normal_path']['passed']
    distances=np.array(screening['retreat_clearance_mm'])
    assert distances.min()>2.5 and distances[-1]>=4 and np.diff(distances).min()>=-.02


def test_cube_reference_follows_measured_link_tilt_instead_of_ground():
    from method4_reference import link_plane_stopped_cube
    m=model();q=level_pose(m,0.);q[[0,4,8]]+=.02
    center,g=link_plane_stopped_cube(m,q)
    assert g['support_tilt_from_world_vertical_deg']>1.
    n=np.array(g['support_plane_normal_world']);origin=np.array(g['support_plane_point_world'])
    assert abs((center-origin)@n-.035)<1e-10
    corners=np.array(g['lower_rear_corners_world_m'])
    np.testing.assert_allclose((corners-origin)@n,0.,atol=1e-10)
    np.testing.assert_allclose(corners@g['palm_stop_normal_world'],g['palm_stop_plane_offset_m'],atol=1e-10)
    initial=matrix([.1,-.2,.3,.4,.05,.5]);expected=np.array(g['expected_cube_axes_world']);raw=np.eye(4);raw[:3,:3]=initial[:3,:3].T@expected;raw[:3,3]=[0,0,.65]
    assert cube_orientation_check(initial,[raw],[expected])['resting_face_normal_max_deg']<1e-5
    assert cube_orientation_check(initial,[raw])['resting_face_normal_max_deg']>1.
