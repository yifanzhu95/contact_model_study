#!/usr/bin/env python3
"""Independent marker-free fit of rigid housing features to measured FK.

Small white housing patches supply repeatable image centroids, not fiducial
identities or known 3D corners. Their two tangential body coordinates are fitted;
the CAD front face supplies the normal coordinate. Palm-paper pixels are masked
BEFORE detection, including all previews. No old extrinsics initialize the fit.
"""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import argparse
import json
import shutil
import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from articulated_fit import load_metadata, transform_points
from articulated_motion import model
from geometry import matrix, parameters, residual_stats
from workspace import local,read_json,write_json,identifier,digest,utc


def mask_paper(rgb,rectangle):
    result=rgb.copy();x0,y0,x1,y1=rectangle;result[y0:y1,x0:x1]=50
    return result


def fresh_palm_fit(config,out):
    """Fit fresh fixed-housing observations without historical extrinsics."""
    from stationary_rgbd import frozen_geometry,sample_depth,solve
    cfg=read_json(config['palm_config']);rows=load_metadata(local(config['session']))
    chosen=[r for r in rows if r['meta']['phase']=='capture' and r['meta']['pose_id']=='park' and r['meta']['joints']['usable']]
    colors=[];depths=[]
    x0,y0,x1,y1=config['exclude_rectangle_xyxy']
    for row in chosen:
        with np.load(row['path']) as f:
            colors.append(mask_paper(f['color_rgb'],config['exclude_rectangle_xyxy']))
            depth=f['depth_z16']*row['meta']['depth_scale_m'];depth[y0:y1,x0:x1]=0.;depths.append(depth)
    colors=np.array(colors);depths=np.array(depths);K=np.array(chosen[0]['meta']['camera_intrinsics'])
    regions=sample_depth(depths,cfg['depth_regions'],K,np.arange(len(depths)),cfg)
    tr=[p for p in cfg['landmarks'] if p['split']=='train']
    W,vertices,_,manifest=frozen_geometry()
    # Verify every natural corner is an actual mesh vertex; this specifically
    # guards the earlier accidental selection of an inner notch vertex.
    for p in cfg['landmarks']:
        if np.min(np.linalg.norm(vertices-p['body_xyz_m'],axis=1))>1e-5:raise ValueError('Corner not on frozen mesh: '+p['id'])
    body=np.array([p['body_xyz_m'] for p in tr]);uv=np.array([p['pixel'] for p in tr])
    C,fit,starts=solve(body,uv,K,[r for r in regions if r['split']=='train'],cfg)
    T=W@np.linalg.inv(C)
    write_json(out/'palm_candidate.json',dict(T_world_from_camera=T.tolist(),status='Unaccepted independent candidate',
        frames=[str(r['path'].relative_to(ROOT)) for r in chosen],palm_config=cfg,positive_depth_starts=starts,model_sha256=manifest['binary_sha256']))
    return np.linalg.inv(T),dict(config=cfg,regions=regions,K=K,W=W,rgb=np.median(colors,axis=0).astype(np.uint8),q=np.median([r['meta']['joints']['q'] for r in chosen],axis=0))


def extract(config,out):
    rows=load_metadata(local(config['session']));m=model();d=mujoco.MjData(m)
    observations=[];tiles=[]
    for finger,spec in config['fingers'].items():
        for suffix in spec['poses']:
            label=finger+'_'+suffix
            chosen=[r for r in rows if r['meta']['phase']=='capture' and r['meta']['pose_id']==label and r['meta']['joints']['usable']]
            if not chosen:continue
            colors=[];depths=[]
            for row in chosen:
                with np.load(row['path']) as f:
                    colors.append(mask_paper(f['color_rgb'],config['exclude_rectangle_xyxy']))
                    depths.append(f['depth_z16']*row['meta']['depth_scale_m'])
            rgb=np.median(colors,axis=0).astype(np.uint8);dep=np.median(depths,axis=0)
            K=np.array(chosen[0]['meta']['camera_intrinsics']);x0,y0,x1,y1=spec['roi_xyxy']
            bright=(rgb.min(axis=2)>config['white_threshold']).astype(np.uint8)
            roi=np.zeros_like(bright);roi[y0:y1,x0:x1]=1;bright*=roi
            n,labels,stats,centers=cv2.connectedComponentsWithStats(bright)
            candidates=[i for i in range(1,n) if config['area_px'][0]<=stats[i,cv2.CC_STAT_AREA]<=config['area_px'][1]
                and .35<stats[i,cv2.CC_STAT_WIDTH]/stats[i,cv2.CC_STAT_HEIGHT]<2.5]
            if not candidates:raise ValueError('No small bright housing feature: '+label)
            seed=spec.get('pose_seed_pixels',{}).get(suffix,spec['seed_pixel'])
            index=min(candidates,key=lambda i:np.linalg.norm(centers[i]-seed))
            uv=centers[index];u,v=np.rint(uv).astype(int)
            patch=dep[v-2:v+3,u-2:u+3];z=patch[(patch>.1)&(patch<1.5)]
            if len(z)<8 or np.std(z)>.004:raise ValueError(f'Unreliable depth on {label}: uv={uv}, valid={len(z)}, std={np.std(z)}')
            depth_count=len(z);depth_std=float(np.std(z))
            z=float(np.median(z));xyz=np.r_[(uv-K[:2,2])/np.diag(K)[:2]*z,z]
            d.qpos[:]=np.median([r['meta']['joints']['q'] for r in chosen],axis=0);mujoco.mj_kinematics(m,d)
            b=m.body(spec['body']).id;F=np.eye(4);F[:3,:3]=d.xmat[b].reshape(3,3);F[:3,3]=d.xpos[b]
            observations.append(dict(finger=finger,pose=label,split=spec['split'],uv=uv.tolist(),xyz=xyz.tolist(),F=F.tolist(),
                q=d.qpos.tolist(),area=int(stats[index,cv2.CC_STAT_AREA]),depth_valid_px=depth_count,depth_std_m=depth_std,
                frames=[str(r['path'].relative_to(ROOT)) for r in chosen]))
            cv2.circle(rgb,tuple(np.rint(uv).astype(int)),5,(255,0,0),1)
            tile=cv2.resize(rgb[175:235,x0:x1],(180,180));cv2.putText(tile,label,(2,15),0,.4,(255,0,0),1);tiles.append(tile)
    while len(tiles)%5:tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out/'detections.png'),cv2.cvtColor(np.concatenate([np.concatenate(tiles[i:i+5],axis=1) for i in range(0,len(tiles),5)]),cv2.COLOR_RGB2BGR))
    write_json(out/'observations.json',dict(K=K.tolist(),observations=observations,excluded_paper_rectangle=config['exclude_rectangle_xyxy']))
    return observations,K


def fit_surface(observations,K,initial,config,bias=False,holdout=False):
    rows=[r for r in observations if (holdout or r['split']=='train')]
    fingers=sorted(set(r['finger'] for r in rows));index=np.array([fingers.index(r['finger']) for r in rows])
    F=np.array([r['F'] for r in rows]);uv=np.array([r['uv'] for r in rows]);xyz=np.array([r['xyz'] for r in rows]);zplane=config['face_z_m']
    points=[]
    for finger in fingers:
        row=next(r for r in rows if r['finger']==finger)
        p=transform_points(np.linalg.inv(row['F']),transform_points(np.linalg.inv(initial),row['xyz']))
        points.extend(p[:2])
    x0=np.r_[parameters(initial),points,[0.] if bias else []]
    def residual(x):
        X=matrix(x[:6]);P=np.c_[x[6:6+2*len(fingers)].reshape(-1,2),np.full(len(fingers),zplane)]
        world=np.einsum('nij,nj->ni',F[:,:3,:3],P[index])+F[:,:3,3];cam=transform_points(X,world)
        pred=(cam@K.T)[:,:2]/np.maximum(cam[:,2,None],.01)
        return np.c_[(pred-uv)/config['pixel_sigma'],(cam[:,2]-xyz[:,2]-(x[-1] if bias else 0))/config['depth_sigma_m']].ravel()
    lower=np.full(len(x0),-np.inf);upper=-lower
    if bias:lower[-1]=-.05;upper[-1]=.05
    fit=least_squares(residual,x0,bounds=(lower,upper),loss='soft_l1',max_nfev=2000,x_scale='jac')
    X=matrix(fit.x[:6]);P=np.c_[fit.x[6:6+2*len(fingers)].reshape(-1,2),np.full(len(fingers),zplane)]
    return X,dict(cost=float(fit.cost),success=bool(fit.success),optimality=float(fit.optimality),
        points={f:p.tolist() for f,p in zip(fingers,P)},depth_bias_m=float(fit.x[-1]) if bias else 0.,
        jacobian_singular_values=np.linalg.svd(fit.jac,compute_uv=False).tolist())


def evaluate(X,observations,K,config,bias=0.):
    """Held-out finger centroid is anchored at neutral once, on its CAD plane.

    Its tangential coordinates are estimated from neutral RGB alone. All
    non-neutral poses (including their depth) remain held out.
    """
    rows=[]
    for finger in sorted(set(r['finger'] for r in observations)):
        group=[r for r in observations if r['finger']==finger];ref=group[0];F=np.array(ref['F'])
        ray=np.linalg.solve(K,np.r_[ref['uv'],1.]);C=np.linalg.inv(F)@np.linalg.inv(X)
        origin=C[:3,3];direction=C[:3,:3]@ray;s=(config['face_z_m']-origin[2])/direction[2]
        point=origin+s*direction
        for r in group:
            cam=transform_points(X,transform_points(np.array(r['F']),point));uv=(K@cam)[:2]/cam[2]
            rows.append(dict(finger=finger,pose=r['pose'],split=r['split'],neutral_anchor=r is ref,
                pixel_error_px=float(np.linalg.norm(uv-r['uv'])),depth_error_mm=float((cam[2]-r['xyz'][2]-bias)*1000),
                predicted_pixel=uv.tolist(),observed_pixel=r['uv']))
    metrics={}
    for split in ['train','validation']:
        subset=[r for r in rows if r['split']==split and not r['neutral_anchor']]
        metrics[split]=dict(pixel=residual_stats([r['pixel_error_px'] for r in subset]),depth_mm=residual_stats([r['depth_error_mm'] for r in subset]))
    return dict(metrics=metrics,rows=rows)


def report(results,palm,obs,config,out):
    """Accept only a candidate consistent with both fixed and moving geometry."""
    import html
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from stationary_rgbd import evaluate as palm_evaluate
    names=list(results);limits=config['acceptance_limits']
    for name,r in results.items():
        X=np.linalg.inv(r['T_world_from_camera'])
        r['palm_validation']=palm_evaluate(X@palm['W'],palm['config']['landmarks'],palm['K'],palm['regions'])
        r['heldout_checks']=dict(palm_image_rms_px=r['palm_validation']['heldout_pixel_error']['rms'],
            moving_image_rms_px=r['validation']['metrics']['validation']['pixel']['rms'],
            moving_depth_rms_mm=r['validation']['metrics']['validation']['depth_mm']['rms'],
            fixed_housing_depth_rms_mm=next(x['signed_plane_error_mm']['rms'] for x in r['palm_validation']['depth'] if x['id']=='mf_front'),
            fixed_palm_depth_rms_mm=next(x['signed_plane_error_mm']['rms'] for x in r['palm_validation']['depth'] if x['id']=='palm_flat_right'))
        r['passes_cross_checks']=all(v <= (limits['heldout_pixel_rms_px'] if 'px' in k else limits['heldout_depth_rms_mm']) for k,v in r['heldout_checks'].items())
    # These thresholds are diagnostic acceptance requirements, not calibrated
    # confidence bounds. Publish no winner when the independent checks conflict.
    accepted=[n for n in names if n!='current' and results[n]['passes_cross_checks']]
    status='No new calibration accepted: independent fixed-palm and moving-finger checks conflict.' if not accepted else 'Candidates pass internal checks; absolute external validation is still required.'
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    x=np.arange(len(names));width=.25
    for ax,keys,limit in [(axes[0],['palm_image_rms_px','moving_image_rms_px'],limits['heldout_pixel_rms_px']),
                          (axes[1],['fixed_housing_depth_rms_mm','fixed_palm_depth_rms_mm','moving_depth_rms_mm'],limits['heldout_depth_rms_mm'])]:
        for j,key in enumerate(keys):ax.bar(x+(j-(len(keys)-1)/2)*width,[results[n]['heldout_checks'][key] for n in names],width,label=key.replace('_rms','').replace('_',' '))
        ax.axhline(limit,color='black',ls='--',label='required cross-check limit');ax.set_xticks(x,names,rotation=15);ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('RMS image residual (pixels)');axes[1].set_ylabel('RMS depth / plane residual (mm)')
    fig.savefig(out/'cross_checks.png',dpi=170);fig.savefig(out/'cross_checks.pdf');plt.close(fig)
    # A simple geometry overlay, using only housing feature locations, keeps
    # the paper region hidden even in review artifacts.
    colors=[(255,80,80),(80,255,80),(80,180,255),(255,200,50)]
    tiles=[]
    for name,color in zip(names,colors):
        rgb=palm['rgb'].copy();r=results[name]
        for c in r['palm_validation']['corners']:
            actual=tuple(np.rint(c['pixel']).astype(int));pred=tuple(np.rint(c['predicted_pixel']).astype(int))
            cv2.circle(rgb,actual,2,(255,255,255),1);cv2.line(rgb,actual,pred,color,1);cv2.drawMarker(rgb,pred,color,cv2.MARKER_CROSS,5,1)
        rgb=mask_paper(rgb,config['exclude_rectangle_xyxy']);rgb=cv2.resize(rgb[150:190,270:410],(560,160))
        cv2.putText(rgb,name,(3,17),0,.5,color,1);tiles.append(rgb)
    cv2.imwrite(str(out/'palm_cross_check.png'),cv2.cvtColor(np.concatenate(tiles),cv2.COLOR_RGB2BGR))
    header='| Candidate | Fixed housing image (px) | Moving middle finger image (px) | Fixed housing depth (mm) | Palm depth (mm) | Moving depth (mm) |'
    lines=['# Independent articulated camera calibration', '', status, '',
        'The paper tag is left in place and masked from RGB and depth. It is not detected, measured, or used as a feature. Current calibration is read only after all new fits finish. Production calibration remains unchanged.', '',
        'Fixed-housing image checks use the middle palm housing, excluded from training. Moving-finger checks use the middle finger, excluded from fitting; its neutral RGB observation anchors an unknown tangential feature position once. Only its other three poses are scored. These are internal geometric consistency tests, not absolute accuracy estimates.', '',header,'|---|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        c=r['heldout_checks'];lines.append(f"| {name} | {c['palm_image_rms_px']:.2f} | {c['moving_image_rms_px']:.2f} | {c['fixed_housing_depth_rms_mm']:.2f} | {c['fixed_palm_depth_rms_mm']:.2f} | {c['moving_depth_rms_mm']:.2f} |")
    lines+=['','The surface-only candidate explains the selected moving features, but its fixed-palm checks must also pass. The depth-bias fit is a nuisance diagnostic and does not alter factory depth scale or camera settings. Each model/annotation assumption can bias an apparently good local fit.', '',
        'An audit of the earlier stationary analysis found that three intended outer lower-right housing corners were assigned inner-notch CAD coordinates, 4.752 mm away. This experiment uses newly annotated pixels and corrected outer vertices. Earlier artifacts remain intact; their reported residuals are not proof of absolute calibration accuracy.', '',
        'Likely contributors requiring separation: natural-feature/CAD correspondence, CAD-to-real rigid housing placement, encoder zero / kinematic geometry, aligned-depth bias and missing depth on small surfaces. The present dataset does not establish which is responsible. No joint offsets or hardware gains were fitted/applied to compensate.', '',
        'Reproduce the complete offline comparison:', '', '```bash','python camera_hand_calibration/rigid_surface_fit.py --config configs/rigid_surface_fit.json','```','',
        'Raw data and motion safety/torque status: `'+config['session']+'`. All output and provenance stay in this workspace. No additional hardware motion is part of this command.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'index.html').write_text('<meta charset="utf-8"><style>body{font:16px system-ui;max-width:1100px;margin:2em auto}img{max-width:100%}pre{white-space:pre-wrap}</style><pre>'+html.escape('\n'.join(lines))+'</pre><img src="cross_checks.png"><img src="detections.png"><img src="palm_cross_check.png">')
    return status


def run(config_path):
    cfg=read_json(config_path);out=local('results/rigid_surface_'+identifier());out.mkdir()
    write_json(out/'config.json',cfg);shutil.copy2(__file__,out/'rigid_surface_fit.py')
    initial,palm=fresh_palm_fit(cfg,out)
    obs,K=extract(cfg,out)
    results={}
    for name,bias in [('surface',False),('surface_bias',True)]:
        X,fit=fit_surface(obs,K,initial,cfg,bias=bias)
        results[name]=dict(T_world_from_camera=np.linalg.inv(X).tolist(),fit=fit,validation=evaluate(X,obs,K,cfg,fit['depth_bias_m']))
        print(name,'training optimizer cost',fit['cost'],flush=True)
    results['palm']=dict(T_world_from_camera=np.linalg.inv(initial).tolist(),validation=evaluate(initial,obs,K,cfg))
    # Historical transform is opened only after both independent fits finish.
    old=np.array(read_json('configs/current_transform.json')['T_world_from_camera'])
    results['current']=dict(T_world_from_camera=old.tolist(),validation=evaluate(np.linalg.inv(old),obs,K,cfg))
    for r in results.values():
        T=np.array(r['T_world_from_camera']);r['difference_from_current']=dict(rotation_deg=float(Rotation.from_matrix(T[:3,:3]@old[:3,:3].T).magnitude()*180/np.pi),
            camera_origin_mm=float(np.linalg.norm(T[:3,3]-old[:3,3])*1000))
    status=report(results,palm,obs,cfg,out)
    write_json(out/'comparison.json',dict(created_utc=utc(),results=results,status=status,production_configuration_modified=False,
        config_sha256=digest(local(config_path)),paper_tag_used=False,absolute_accuracy_established=False))
    write_json('results/rigid_surface_latest.json',dict(directory=str(out.relative_to(ROOT))))
    print('Results:',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/rigid_surface_fit.json');args=parser.parse_args();run(args.config)
