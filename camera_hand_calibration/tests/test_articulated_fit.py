"""Synthetic frame/observability tests; no hardware access or old extrinsics."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from articulated_fit import matrix,parameters,transform_points,solve_hand_eye,robust_rigid,bundle_adjust


def test_hand_eye_recovers_camera_without_any_initial_extrinsic():
    X=matrix([.2,-.4,.3,.02,-.04,.65]);rng=np.random.default_rng(18);pairs=[]
    for _ in range(12):
        B=matrix(np.r_[rng.normal(0,.4,3),rng.normal(0,.04,3)])
        pairs.append(dict(A=X@B@np.linalg.inv(X),B=B,n=20,rms=.001))
    fit,diagnostics=solve_hand_eye(pairs)
    np.testing.assert_allclose(fit,X,atol=1e-7)
    assert diagnostics['translation_rank']==3


def test_single_joint_axis_is_rejected_as_unobservable():
    X=matrix([.2,-.4,.3,.02,-.04,.65]);pairs=[]
    for angle in [.2,.4,.6,.8]:
        B=matrix([angle,0,0,0,0,0]);pairs.append(dict(A=X@B@np.linalg.inv(X),B=B))
    with pytest.raises(ValueError,match='independent rotation axes'):solve_hand_eye(pairs)


def test_rigid_patch_fit_rejects_bad_depth_points():
    rng=np.random.default_rng(3);points=rng.normal(0,.01,(20,3));T=matrix([.2,-.1,.1,.02,-.01,.03])
    moved=transform_points(T,points);moved[:3]+=.04
    fit,keep,error=robust_rigid(points,moved,rng)
    assert keep.sum()==17 and error<1e-9
    np.testing.assert_allclose(fit,T,atol=1e-8)


def synthetic_groups():
    rng=np.random.default_rng(4);X=matrix([.1,.05,.15,.01,-.02,.65]);K=np.array([[605,0,320],[0,605,240],[0,0,1.]])
    groups=[]
    for name in ['if','rf']:
        local=rng.uniform(-.02,.02,(6,3));poses=[]
        for i in range(8):
            F=matrix(np.r_[rng.normal(0,.4,3),rng.normal(0,.035,3)])
            camera=transform_points(X,transform_points(F,local));uv=(camera@K.T)[:,:2]/camera[:,2,None]
            poses.append(dict(id=name+str(i),F=F,uv=uv,xyz=camera))
        groups.append(dict(finger=name,split='train',K=K,poses=poses))
    return X,groups


def test_bundle_adjustment_uses_unknown_local_features_and_recovers_metric_pose():
    X,groups=synthetic_groups();initial=X.copy();initial[:3,3]+=[.015,-.01,.02]
    fit,diag=bundle_adjust(groups,initial,dict(pixel_sigma=.8,depth_sigma_m=.002),'rgbd')
    np.testing.assert_allclose(fit,X,atol=2e-5)
    assert diag['training_reprojection_px']['rms']<.001
