"""Switch the same published pose without restarting tracking or touching joints."""
import numpy as np
from compare_tracking import object_world_pose,Selection,SelectedBuffer,CALIBRATION_KEYS,CORRECTION_KEY,build_model
from geometry import matrix
from workspace import read_json


def test_original_matches_existing_pipeline_including_offset_and_object_axes():
    raw=matrix([.1,-.2,.3,.01,-.03,.65]);P=matrix([.3,.1,-.2,.4,.05,.5]);before=matrix([0,.1,0,.02,0,.01]);after=matrix([0,0,.2,0,0,0])
    offset=np.array([0,0,-.015]);corrected=raw.copy();corrected[:3,3]+=offset;published=P@corrected
    original=object_world_pose(published,P,before@P,after,offset,True)
    np.testing.assert_allclose(original,before@published@after,atol=1e-12)
    alternative=matrix([.1,.4,.2,.3,.02,.55])
    np.testing.assert_allclose(object_world_pose(published,P,alternative,after,offset,False),alternative@raw@after,atol=1e-12)


def test_selection_changes_pose_from_same_packet_and_preserves_hand_state():
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    cfg=dict(read_json('configs/live_tracking.json'),initial_calibration=0,object_correction_default=True);selection=Selection(cfg);selection.key(296);assert selection.current()[0]==1
    selection.key(297);assert selection.current()[0]==2
    q=np.r_[np.arange(16)*.01,[.04,.03,.6],[1.,0,0,0]];velocity=np.zeros(22)
    raw=SimpleNamespace(snapshot=lambda **kwargs:(q.copy(),velocity.copy(),dict(pose_timestamp=123)))
    camera=dict(camera_to_published=np.eye(4),object_translation_adjust_camera_m=[0,0,-.015]);buffer=SelectedBuffer(raw,selection,camera,np.eye(4))
    a,_,_=buffer.snapshot();selection.key(295);b,_,_=buffer.snapshot()
    np.testing.assert_array_equal(a[:16],b[:16]);np.testing.assert_array_equal(b[:16],q[:16]);assert not np.allclose(a[16:],b[16:])
    selection.key(CORRECTION_KEY);assert selection.current()[2] is False


def test_native_shortcuts_do_not_change_calibration_or_correction():
    selection=Selection(dict(read_json('configs/live_tracking.json'),initial_calibration=0,object_correction_default=True))
    for key in [ord('1'),ord('2'),ord('3'),9,258,ord('O'),ord('Q'),256,290,291,292,293,294]:
        selection.key(key)
        assert selection.current()[0]==0 and selection.current()[2] is True


def test_compiled_object_is_the_textured_mesh_not_a_box():
    import mujoco
    from workspace import local
    model=build_model(read_json('configs/live_tracking.json'),local('cache/live_viewer'))
    geom=model.geom('obj_visual').id;mesh=model.geom_dataid[geom]
    assert model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_MESH
    assert mesh>=0 and model.mesh_vertnum[mesh]>1000 and model.mesh_texcoordnum[mesh]>1000
    assert model.geom_matid[geom]>=0


def test_method4_shortcut_starts_without_object_only_offset():
    cfg=dict(read_json('configs/live_tracking.json'),initial_calibration=0,object_correction_default=True)
    cfg['calibrations']=cfg['calibrations'][:3]+[cfg['calibrations'][0]]
    selection=Selection(cfg);selection.key(299)
    assert selection.current()[0]==3 and selection.current()[2] is False
