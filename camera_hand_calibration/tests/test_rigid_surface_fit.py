"""Geometric validation and tag exclusion, without devices."""
import numpy as np
from rigid_surface_fit import mask_paper,fit_surface,evaluate,verify_visible_surface
from geometry import matrix
from articulated_fit import transform_points


def test_paper_pixels_cannot_affect_detection_input():
    rng=np.random.default_rng(81);a=rng.integers(0,256,(480,640,3),dtype=np.uint8);b=a.copy()
    b[75:151,320:391]=255-a[75:151,320:391]
    np.testing.assert_array_equal(mask_paper(a,[320,75,391,151]),mask_paper(b,[320,75,391,151]))
    assert not np.array_equal(a,b)  # masking must not mutate the saved raw RGB


def test_known_surface_pose_is_recovered_and_wrong_pose_fails_validation():
    X=matrix([.17,-.24,.12,.01,-.02,.65]);K=np.array([[605.,0,317],[0,605,253],[0,0,1.]])
    rng=np.random.default_rng(1);obs=[]
    for finger,split in [('if','train'),('rf','train'),('mf','validation')]:
        point=np.r_[rng.normal(0,.01,2),-.0001]
        for i in range(8):
            F=matrix(np.r_[rng.normal(0,.45,3),rng.normal(0,.03,3)])
            cam=transform_points(X,transform_points(F,point));uv=(K@cam)[:2]/cam[2]
            obs.append(dict(finger=finger,split=split,pose=f'{finger}_{i}',F=F.tolist(),uv=uv.tolist(),xyz=cam.tolist()))
    initial=X.copy();initial[:3,3]+=[.02,-.015,.01];cfg=dict(face_z_m=-.0001,pixel_sigma=.7,depth_sigma_m=.002)
    recovered,diag=fit_surface(obs,K,initial,cfg)
    np.testing.assert_allclose(recovered,X,atol=1e-7)
    assert evaluate(recovered,obs,K,cfg)['metrics']['validation']['pixel']['rms']<1e-6
    assert evaluate(initial,obs,K,cfg)['metrics']['validation']['pixel']['rms']>1.


def test_opposite_housing_face_is_rejected():
    import mujoco
    import pytest
    from articulated_motion import model
    m=model();d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);b=m.body('if_px').id
    F=np.eye(4);F[:3,:3]=d.xmat[b].reshape(3,3);F[:3,3]=d.xpos[b]
    camera_body=np.eye(4);camera_body[2,3]=-.6;X=np.linalg.inv(F@camera_body)
    cfg=dict(fingers={'if':dict(body='if_px')},face_z_m=-.02308408388820924)
    obs=[dict(finger='if',F=F.tolist(),q=d.qpos.tolist())]
    assert verify_visible_surface(obs,X,cfg)[0]['visible_extreme_z_m']<-.02
    with pytest.raises(ValueError,match='camera-facing'):
        verify_visible_surface(obs,X,dict(cfg,face_z_m=-.0000839))
