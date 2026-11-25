#!/usr/bin/env python3
"""Marker-free eye-to-hand calibration from natural features and measured FK.

Camera relative motions A and FK relative motions B satisfy A X = X B,
where X maps MuJoCo world to camera. The solver has no old-calibration input.
The old transform is loaded only after fitting for independent comparison.
Reference: https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html
"""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
os.environ['XDG_CACHE_HOME']=str(ROOT/'cache/xdg')

import argparse
import csv
import json
import time
import warnings
from itertools import combinations

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix

from workspace import local,read_json,write_json,digest,identifier,utc
from geometry import matrix,parameters,residual_stats
from articulated_motion import model

cv2.setNumThreads(1)


def without_paper(rgb,config):
    """Keep the manual-calibration paper outside every feature pyramid."""
    result=rgb.copy();x0,y0,x1,y1=config.get('exclude_rectangle_xyxy',[320,75,391,151])
    result[y0:y1,x0:x1]=50
    return result


def transform_points(T,p):return np.asarray(p)@T[:3,:3].T+T[:3,3]


def kabsch(a,b):
    ac,bc=a.mean(axis=0),b.mean(axis=0)
    u,s,vt=np.linalg.svd((a-ac).T@(b-bc))
    if s[1]<1e-9:raise ValueError('Feature cloud is collinear')
    R=vt.T@np.diag([1,1,np.linalg.det(vt.T@u.T)])@u.T
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=bc-R@ac
    return T


def robust_rigid(a,b,rng):
    best=None
    for _ in range(120):
        index=rng.choice(len(a),3,replace=False)
        try:T=kabsch(a[index],b[index])
        except ValueError:continue
        error=np.linalg.norm(transform_points(T,a)-b,axis=1)
        keep=error<.004
        if keep.sum()>=5 and (best is None or keep.sum()>best.sum()):best=keep
    if best is None:raise ValueError('No consistent rigid patch motion')
    for _ in range(3):
        T=kabsch(a[best],b[best]);error=np.linalg.norm(transform_points(T,a)-b,axis=1)
        keep=error<.004
        if keep.sum()<5:break
        best=keep
    return T,best,float(np.sqrt(np.mean(error[best]**2)))


def solve_hand_eye(pairs):
    if len(pairs)<3:raise ValueError('Need at least three usable relative motions')
    a=np.array([Rotation.from_matrix(p['A'][:3,:3]).as_rotvec() for p in pairs])
    b=np.array([Rotation.from_matrix(p['B'][:3,:3]).as_rotvec() for p in pairs])
    axes=np.linalg.svd(b,compute_uv=False)
    if axes[1]<.03*axes[0]:raise ValueError('Joint motions do not excite two independent rotation axes')
    weights=np.array([p.get('n',10)/max(.001,p.get('rms',.001)) for p in pairs]);weights/=weights.max()
    u,_,vt=np.linalg.svd((b*weights[:,None]).T@a)
    R=vt.T@np.diag([1,1,np.linalg.det(vt.T@u.T)])@u.T
    lhs=np.concatenate([np.sqrt(w)*(p['A'][:3,:3]-np.eye(3)) for w,p in zip(weights,pairs)])
    rhs=np.concatenate([np.sqrt(w)*(R@p['B'][:3,3]-p['A'][:3,3]) for w,p in zip(weights,pairs)])
    t,_,rank,sv=np.linalg.lstsq(lhs,rhs,rcond=None)
    if rank<3:raise ValueError('Translation is not observable from these motions')
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
    def residual(x):
        X=matrix(x);values=[]
        for w,p in zip(weights,pairs):
            L=p['A']@X;Q=X@p['B']
            values.extend(np.sqrt(w)*Rotation.from_matrix(L[:3,:3]@Q[:3,:3].T).as_rotvec()/.05)
            values.extend(np.sqrt(w)*(L[:3,3]-Q[:3,3])/.004)
        return values
    fit=least_squares(residual,parameters(T),loss='soft_l1',f_scale=1.,max_nfev=300,x_scale='jac')
    X=matrix(fit.x)
    return X,dict(relative_motions=len(pairs),axis_singular_values=axes.tolist(),translation_rank=int(rank),
        translation_condition=float(sv[0]/sv[-1]),cost=float(fit.cost),success=bool(fit.success))


def depth_point(depth,uv,K,threshold):
    u,v=np.rint(uv).astype(int)
    if not 3<=u<depth.shape[1]-3 or not 3<=v<depth.shape[0]-3:return None
    patch=depth[v-2:v+3,u-2:u+3];good=patch[(patch>.1)&(patch<1.5)]
    if len(good)<15 or np.std(good)>threshold:return None
    z=float(np.median(good));return np.array([(uv[0]-K[0,2])*z/K[0,0],(uv[1]-K[1,2])*z/K[1,1],z])


def load_metadata(folder):
    rows=[]
    for p in sorted(folder.glob('frame_*.npz')):
        with np.load(p,allow_pickle=False) as f:meta=json.loads(str(f['metadata_json']))
        rows.append(dict(path=p,meta=meta))
    return rows


def track_finger(folder,rows,finger,config,out):
    spec=config['fingers'][finger]
    selected=[(i,r) for i,r in enumerate(rows) if r['meta']['phase']=='capture' and r['meta']['pose_id']==finger+'_neutral' and r['meta']['joints']['usable']]
    if not selected:raise ValueError('Missing reference capture for '+finger)
    reference=[r for _,r in selected]
    reference_index=selected[-1][0]
    images=[];depths=[]
    for r in reference:
        with np.load(r['path']) as f:
            images.append(f['color_rgb']);depths.append(f['depth_z16']*r['meta']['depth_scale_m'])
    rgb=without_paper(np.median(images,axis=0).astype(np.uint8),config);dep=np.median(depths,axis=0)
    K=np.array(reference[0]['meta']['camera_intrinsics'])
    prev=cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(3,3),.6)
    mask=np.zeros(prev.shape,np.uint8);x0,y0,x1,y1=spec['roi_xyxy'];mask[y0:y1,x0:x1]=255
    features=cv2.goodFeaturesToTrack(prev,maxCorners=config['feature_count'],qualityLevel=.015,minDistance=2,mask=mask,blockSize=3)
    if features is None or len(features)<5:raise ValueError('Insufficient natural features in '+finger)
    n=len(features);valid=np.ones(n,bool);points=features[:,0].copy();ids=np.arange(n)
    ref_uv=points.copy();ref_xyz=np.full((n,3),np.nan)
    for i,p in enumerate(points):
        xyz=depth_point(dep,p,K,config['depth_patch_std_max_m'])
        if xyz is not None:ref_xyz[i]=xyz
    accum={};last_id=finger+'_return'
    for row in rows[reference_index+1:]:
        meta=row['meta']
        if meta['pose_id']==finger+'_park':break
        with np.load(row['path']) as f:rgb1=f['color_rgb'];depth=f['depth_z16']*meta['depth_scale_m']
        gray=cv2.GaussianBlur(cv2.cvtColor(without_paper(rgb1,config),cv2.COLOR_RGB2GRAY),(3,3),.6)
        active=np.flatnonzero(valid)
        if len(active)<5:break
        old=points[active,None,:].astype(np.float32)
        new,ok,error=cv2.calcOpticalFlowPyrLK(prev,gray,old,None,winSize=(21,21),maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01))
        back,ok2,_=cv2.calcOpticalFlowPyrLK(gray,prev,new,None,winSize=(21,21),maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01))
        keep=(ok[:,0]>0)&(ok2[:,0]>0)&(np.linalg.norm(back[:,0]-old[:,0],axis=1)<config['flow_fb_max_px'])&(error[:,0]<35)
        valid[active[~keep]]=False;points[active[keep]]=new[keep,0];prev=gray
        if meta['phase']!='capture' or not meta['joints']['usable']:continue
        label=meta['pose_id']
        if label.endswith('_neutral'):continue
        if not label.startswith(finger+'_'):continue
        uv=np.full((n,2),np.nan);xyz=np.full((n,3),np.nan)
        for j in np.flatnonzero(valid):
            uv[j]=points[j];p=depth_point(depth,points[j],K,config['depth_patch_std_max_m'])
            if p is not None:xyz[j]=p
        accum.setdefault(label,[]).append(dict(uv=uv,xyz=xyz,q=meta['joints']['q'],frame=str(row['path'].relative_to(ROOT))))
    poses=[dict(id=finger+'_neutral',uv=ref_uv,xyz=ref_xyz,q=np.median([r['meta']['joints']['q'] for r in reference],axis=0),
                frames=[str(r['path'].relative_to(ROOT)) for r in reference])]
    for label,values in accum.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            uv=np.nanmedian([r['uv'] for r in values],axis=0);xyz=np.nanmedian([r['xyz'] for r in values],axis=0)
        poses.append(dict(id=label,uv=uv,xyz=xyz,q=np.median([r['q'] for r in values],axis=0),frames=[r['frame'] for r in values]))
    m=model();d=mujoco.MjData(m);body=m.body(spec['body']).id
    for p in poses:
        d.qpos[:]=p['q'];mujoco.mj_kinematics(m,d)
        T=np.eye(4);T[:3,:3]=d.xmat[body].reshape(3,3);T[:3,3]=d.xpos[body];p['F']=T
    image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for i,p in enumerate(ref_uv):cv2.circle(image,tuple(np.rint(p).astype(int)),2,(0,255,0) if np.isfinite(ref_xyz[i]).all() else (0,0,255),1)
    cv2.rectangle(image,(x0,y0),(x1,y1),(0,255,255),1);cv2.imwrite(str(out/(finger+'_reference.png')),image)
    np.savez_compressed(out/(finger+'_tracks.npz'),uv=np.array([p['uv'] for p in poses]),xyz=np.array([p['xyz'] for p in poses]),
        q=np.array([p['q'] for p in poses]),F=np.array([p['F'] for p in poses]),ids=np.array([p['id'] for p in poses]),K=K)
    print(f'{finger}: {n} natural features, {len(poses)} settled poses; valid 3D points '+str([int(np.isfinite(p['xyz']).all(axis=1).sum()) for p in poses]),flush=True)
    return dict(finger=finger,body=spec['body'],split=spec['split'],poses=poses,K=K)


def motion_pairs(groups,seed):
    rng=np.random.default_rng(seed);result=[]
    for group in groups:
        if group['split']!='train':continue
        for a,b in combinations(group['poses'],2):
            valid=np.isfinite(a['xyz']).all(axis=1)&np.isfinite(b['xyz']).all(axis=1)
            if valid.sum()<5:continue
            B=b['F']@np.linalg.inv(a['F'])
            angle=Rotation.from_matrix(B[:3,:3]).magnitude()
            if not .18<angle<1.5:continue
            try:A,keep,rms=robust_rigid(a['xyz'][valid],b['xyz'][valid],rng)
            except ValueError:continue
            observed=Rotation.from_matrix(A[:3,:3]).magnitude()
            if abs(observed-angle)>.12:continue
            result.append(dict(A=A,B=B,n=int(keep.sum()),rms=rms,finger=group['finger'],first=a['id'],second=b['id']))
    return result


def reanchor_tracks(groups,rows,config,out):
    """Independent static-frame template check prevents cumulative LK drift.

    Crops on rigid housings are occluded by distal links during deep curls;
    these predeclared phases are excluded without consulting any calibration.
    """
    for group in groups:
        ref=group['poses'][0];reference_image=None
        for p in group['poses']:
            matches=[r for r in rows if r['meta']['pose_id']==p['id'] and r['meta']['phase']=='capture' and r['meta']['joints']['usable']]
            colors=[];depths=[]
            for r in matches:
                with np.load(r['path']) as f:
                    colors.append(f['color_rgb']);depths.append(f['depth_z16']*r['meta']['depth_scale_m'])
            if not colors:continue
            gray=cv2.GaussianBlur(cv2.cvtColor(without_paper(np.median(colors,axis=0).astype(np.uint8),config),cv2.COLOR_RGB2GRAY),(3,3),.6)
            dep=np.median(depths,axis=0)
            if reference_image is None:
                reference_image=gray
                continue
            keep=np.isfinite(p['uv']).all(axis=1)
            indices=np.flatnonzero(keep)
            if len(indices):
                a=ref['uv'][indices,None].astype(np.float32);guess=p['uv'][indices,None].astype(np.float32)
                b,ok,err=cv2.calcOpticalFlowPyrLK(reference_image,gray,a,guess,winSize=(11,11),maxLevel=2,
                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,50,.005))
                c,ok2,_=cv2.calcOpticalFlowPyrLK(gray,reference_image,b,a.copy(),winSize=(11,11),maxLevel=2,
                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,50,.005))
                valid=(ok[:,0]>0)&(ok2[:,0]>0)&(np.linalg.norm(c[:,0]-a[:,0],axis=1)<.7)&(err[:,0]<20)
                p['uv'][:]=np.nan;p['xyz'][:]=np.nan
                for j,uv in zip(indices[valid],b[valid,0]):
                    p['uv'][j]=uv
                    xyz=depth_point(dep,uv,group['K'],config['depth_patch_std_max_m'])
                    if xyz is not None:p['xyz'][j]=xyz
        before=len(group['poses'])
        excluded=config.get('exclude_pose_suffixes',[])
        group['poses']=[p for p in group['poses'] if not any(p['id'].endswith(suffix) for suffix in excluded)]
        print(group['finger'],'reanchored',before,'->',len(group['poses']),'poses, valid pixels',
              [int(np.isfinite(p['uv']).all(axis=1).sum()) for p in group['poses']],flush=True)
        np.savez_compressed(out/(group['finger']+'_reanchored.npz'),uv=np.array([p['uv'] for p in group['poses']]),
            xyz=np.array([p['xyz'] for p in group['poses']]),q=np.array([p['q'] for p in group['poses']]),
            F=np.array([p['F'] for p in group['poses']]),ids=np.array([p['id'] for p in group['poses']]),K=group['K'])


def bundle_adjust(groups,X,config,mode='rgbd'):
    """Shared extrinsics and unknown rigid natural-feature body coordinates.

    Unknown body coordinates avoid pretending a fabric/texture corner is an
    exact CAD vertex. FK, not commanded targets, supplies every measured pose.
    """
    local_points=[];observations=[];keys=[]
    for group in groups:
        if group['split']!='train':continue
        poses=group['poses'];n=len(poses[0]['uv'])
        for j in range(n):
            usable=[p for p in poses if np.isfinite(p['uv'][j]).all()]
            depths=[p for p in usable if np.isfinite(p['xyz'][j]).all()]
            if len(usable)<4 or not depths:continue
            p=depths[0];point=transform_points(np.linalg.inv(p['F']),transform_points(np.linalg.inv(X),p['xyz'][j]))
            index=len(local_points);local_points.append(point);keys.append((group['finger'],j))
            for p in usable:
                observations.append(dict(point=index,F=p['F'],uv=p['uv'][j],z=p['xyz'][j,2],pose=p['id'],finger=group['finger']))
    local_points=np.array(local_points);features=len(local_points)
    if features<5:raise ValueError('Not enough persistent natural features')
    index=np.array([r['point'] for r in observations]);F=np.array([r['F'] for r in observations]);uv=np.array([r['uv'] for r in observations]);z=np.array([r['z'] for r in observations]);good=np.isfinite(z)&(mode!='rgb_only')
    K=groups[0]['K'];bias=mode=='rgbd_bias'
    x0=np.r_[parameters(X),local_points.ravel(),[0.] if bias else []]
    def residual(x):
        T=matrix(x[:6]);P=x[6:6+features*3].reshape(-1,3)
        world=np.einsum('nij,nj->ni',F[:,:3,:3],P[index])+F[:,:3,3]
        cam=transform_points(T,world);pred=(cam@K.T)[:,:2]/np.maximum(cam[:,2,None],.01)
        result=np.zeros((len(uv),3));result[:,:2]=(pred-uv)/config['pixel_sigma']
        result[good,2]=(cam[good,2]-z[good]-(x[-1] if bias else 0))/config['depth_sigma_m']
        return result.ravel()
    sparse=lil_matrix((len(observations)*3,len(x0)),dtype=np.int8)
    for i,j in enumerate(index):
        sparse[i*3:i*3+3,:6]=1;sparse[i*3:i*3+3,6+j*3:6+j*3+3]=1
        if bias:sparse[i*3+2,-1]=1
    lower=np.full(len(x0),-np.inf);upper=np.full(len(x0),np.inf)
    if bias:lower[-1]=-.05;upper[-1]=.05
    fit=least_squares(residual,x0,jac_sparsity=sparse.tocsr(),bounds=(lower,upper),loss='soft_l1',f_scale=1.,
        max_nfev=800,x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-7)
    T=matrix(fit.x[:6]);P=fit.x[6:6+features*3].reshape(-1,3)
    res=residual(fit.x).reshape(-1,3)
    return T,dict(mode=mode,success=bool(fit.success),message=fit.message,cost=float(fit.cost),evaluations=fit.nfev,
        features=features,observations=len(observations),estimated_additive_depth_bias_m=float(fit.x[-1]) if bias else 0.,
        training_reprojection_px=residual_stats(np.linalg.norm(res[:,:2],axis=1)*config['pixel_sigma']),
        training_depth_mm=residual_stats(res[good,2]*config['depth_sigma_m']*1000),
        feature_body_coordinates={f'{f}_{j}':p.tolist() for (f,j),p in zip(keys,P)})


def motion_validation(groups,X,bias=0.):
    """One depth anchor per natural feature; all other poses predict forward.

    Each candidate gets the same reference observation, not new fits to the
    validation motions. This scores motion consistency, not independent absolute
    point accuracy or fingertip metrology.
    """
    rows=[]
    for group in groups:
        poses=group['poses'];ref=poses[0];K=group['K']
        for j in range(len(ref['uv'])):
            if not np.isfinite(ref['xyz'][j]).all():continue
            anchor=ref['xyz'][j].copy();anchor*=1+bias/anchor[2]
            body=transform_points(np.linalg.inv(ref['F']),transform_points(np.linalg.inv(X),anchor))
            for p in poses[1:]:
                if not np.isfinite(p['uv'][j]).all():continue
                prediction=transform_points(X,transform_points(p['F'],body))
                pixel=(K@prediction)[:2]/prediction[2]
                row=dict(finger=group['finger'],split=group['split'],pose_id=p['id'],feature=j,
                    pixel_error=float(np.linalg.norm(pixel-p['uv'][j])),predicted_u=float(pixel[0]),predicted_v=float(pixel[1]),
                    observed_u=float(p['uv'][j,0]),observed_v=float(p['uv'][j,1]),depth_error_mm=None,point_error_mm=None)
                if np.isfinite(p['xyz'][j]).all():
                    actual=p['xyz'][j]*(1+bias/p['xyz'][j,2])
                    row.update(depth_error_mm=float((prediction[2]-actual[2])*1000),point_error_mm=float(np.linalg.norm(prediction-actual)*1000))
                rows.append(row)
    metrics={}
    for key in ['train','validation']:
        chosen=[r for r in rows if r['split']==key]
        metrics[key]=dict(pixel_error=residual_stats([r['pixel_error'] for r in chosen]),
            depth_error_mm=residual_stats([r['depth_error_mm'] for r in chosen if r['depth_error_mm'] is not None]),
            point_error_mm=residual_stats([r['point_error_mm'] for r in chosen if r['point_error_mm'] is not None]))
    return metrics,rows


def run(config_path,reuse_tracks=None):
    config=read_json(config_path);folder=local(config['session'])
    session=read_json(folder/'session.json')
    if session['status']!='complete' or not session.get('torque_disabled_verified'):raise ValueError('Need complete, torque-clean sweep')
    out=local('results')/('articulated_fit_'+identifier());out.mkdir();write_json(out/'config.json',config)
    if reuse_tracks:
        source=local(reuse_tracks);saved=read_json(source/'config.json')
        if saved['session']!=config['session'] or saved['fingers']!=config['fingers']:raise ValueError('Cached tracks use another scene/ROI/body')
        groups=[]
        import shutil
        for f,spec in config['fingers'].items():
            with np.load(source/(f+'_tracks.npz')) as d:
                poses=[dict(id=str(d['ids'][i]),uv=d['uv'][i].copy(),xyz=d['xyz'][i].copy(),q=d['q'][i].copy(),F=d['F'][i].copy()) for i in range(len(d['ids']))]
                groups.append(dict(finger=f,body=spec['body'],split=spec['split'],poses=poses,K=d['K'].copy()))
            shutil.copy2(source/(f+'_tracks.npz'),out/(f+'_tracks.npz'));shutil.copy2(source/(f+'_reference.png'),out/(f+'_reference.png'))
    else:
        rows=load_metadata(folder);groups=[track_finger(folder,rows,f,config,out) for f in config['fingers']]
    if config.get('reanchor'):
        rows=load_metadata(folder)
        reanchor_tracks(groups,rows,config,out)
    pairs=motion_pairs(groups,config['seed'])
    X,diag=solve_hand_eye(pairs)
    write_json(out/'initialization.json',dict(T_camera_from_world=X.tolist(),diagnostics=diag,
        relative_motions=[dict(p, A=p['A'].tolist(),B=p['B'].tolist()) for p in pairs]))
    print('SCRATCH HAND EYE',json.dumps(dict(T_world_from_camera=np.linalg.inv(X).tolist(),diagnostics=diag)),flush=True)
    results={}
    for mode in ['rgbd','rgbd_bias','rgb_only']:
        T,metrics=bundle_adjust(groups,X,config,mode)
        validation,details=motion_validation(groups,T,metrics['estimated_additive_depth_bias_m'])
        results[mode]=dict(T_world_from_camera=np.linalg.inv(T).tolist(),optimization=metrics,motion_validation=validation)
        print('REFINED',mode,json.dumps(results[mode]),flush=True)
        with (out/(mode+'_motion_residuals.csv')).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    # Prior only enters here, after ALL independent variants have been fitted.
    current=np.linalg.inv(read_json('configs/current_transform.json')['T_world_from_camera'])
    validation,details=motion_validation(groups,current)
    results['current']=dict(T_world_from_camera=np.linalg.inv(current).tolist(),motion_validation=validation)
    write_json(out/'comparison.json',dict(created_at_utc=utc(),results=results,status='candidate_pending_validation'))
    write_json('results/articulated_latest.json',dict(directory=str(out.relative_to(ROOT))))
    return out,config,groups,X


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/articulated_fit.json');p.add_argument('--reuse-tracks');args=p.parse_args()
    run(args.config,args.reuse_tracks)


if __name__=='__main__':main()
