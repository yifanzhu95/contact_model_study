"""Natural-landmark pose fitting; fixed intrinsics, explicit world/camera frames."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from workspace import transform


def matrix(parameters):
    result=np.eye(4);result[:3,:3]=Rotation.from_rotvec(parameters[:3]).as_matrix();result[:3,3]=parameters[3:]
    return result


def parameters(value):
    value=transform(value)
    return np.r_[Rotation.from_matrix(value[:3,:3]).as_rotvec(),value[:3,3]]


def camera_points(points,T_world_from_camera):
    T=np.linalg.inv(transform(T_world_from_camera))
    return np.asarray(points)@T[:3,:3].T+T[:3,3]


def project(points,T_world_from_camera,K):
    q=camera_points(points,T_world_from_camera)
    pixels=q@np.asarray(K).T
    return pixels[:,:2]/pixels[:,2,None],q[:,2]


def fit_pose(world_points,image_points,K,initial=None,threshold=3.,scale=2.,seed=0):
    import cv2
    points,pixels,K=np.asarray(world_points,float),np.asarray(image_points,float),np.asarray(K,float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0 or not np.allclose(K[2],[0,0,1]):
        raise ValueError('Invalid pinhole camera intrinsics')
    if points.shape!=(len(points),3) or pixels.shape!=(len(points),2) or len(points)<6:
        raise ValueError('Need at least six valid 3D/2D correspondences')
    if not np.isfinite(points).all() or not np.isfinite(pixels).all():raise ValueError('Non-finite correspondences')
    spread=np.linalg.svd(points-points.mean(axis=0),compute_uv=False)
    if spread[1]<1e-5:raise ValueError('Landmarks are collinear or coincide')
    if initial is None:
        cv2.setRNGSeed(seed)
        ok,r,t,inliers=cv2.solvePnPRansac(points,pixels,K,np.zeros(5),iterationsCount=1000,
            reprojectionError=threshold,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers)<6:raise ValueError('PnP did not find six consistent correspondences')
        x=np.r_[r.ravel(),t.ravel()]
    else:
        x=parameters(np.linalg.inv(transform(initial)))
    def residual(x,indices):
        T=matrix(x);q=points[indices]@T[:3,:3].T+T[:3,3]
        z=np.maximum(q[:,2],1e-6);uv=(q@K.T)[:,:2]/z[:,None]
        return (uv-pixels[indices]).ravel()
    indices=np.arange(len(points))
    result=least_squares(residual,x,args=(indices,),loss='soft_l1',f_scale=scale,max_nfev=500,x_scale='jac')
    errors=np.linalg.norm(residual(result.x,indices).reshape(-1,2),axis=1)
    inliers=errors<threshold
    if inliers.sum()<6:raise ValueError('Fit has fewer than six inliers')
    result=least_squares(residual,result.x,args=(np.flatnonzero(inliers),),loss='soft_l1',f_scale=scale,max_nfev=500,x_scale='jac')
    T=np.linalg.inv(matrix(result.x));_,z=project(points,T,K)
    if np.any(z[inliers]<=0):raise ValueError('Fitted visible landmarks are behind the camera')
    # Scale translations to 10 cm so the rank/conditioning comparison has
    # meaningful rotational/translational units rather than arbitrary meters.
    jac=result.jac@np.diag([1,1,1,.1,.1,.1]);sv=np.linalg.svd(jac,compute_uv=False)
    return T,dict(success=bool(result.success),message=result.message,inliers=inliers.tolist(),
        jacobian_rank=int(np.linalg.matrix_rank(jac)),scaled_jacobian_condition=float(sv[0]/max(sv[-1],1e-15)),
        planar_geometry=bool(spread[2]<spread[0]*1e-3),
        warning='Planar landmarks may admit ambiguous poses; validate on other visible surfaces/poses.' if spread[2]<spread[0]*1e-3 else None)


def point_in_world(model,body,xyz,q=None):
    import mujoco
    bid=model.body(body).id;cursor=bid;articulated=False
    while cursor:
        articulated|=bool(model.body_jntnum[cursor]);cursor=int(model.body_parentid[cursor])
    if articulated and q is None:raise ValueError(f'{body} requires measured joints; nominal/default angles are not accepted')
    data=mujoco.MjData(model)
    if q is not None:
        q=np.asarray(q,float)
        if q.shape!=(model.nq,) or not np.isfinite(q).all():raise ValueError('Measured q does not match frozen hand model')
        data.qpos[:]=q
    mujoco.mj_kinematics(model,data)
    return data.xpos[bid]+data.xmat[bid].reshape(3,3)@np.asarray(xyz,float)


def residual_stats(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return dict(n=int(a.size),mean=float(a.mean()),median=float(np.median(a)),
                rms=float(np.sqrt(np.mean(a*a))),mae=float(np.mean(abs(a))),
                p95=float(np.percentile(a,95)),p95_abs=float(np.percentile(abs(a),95)),max=float(a.max())) if a.size else None
