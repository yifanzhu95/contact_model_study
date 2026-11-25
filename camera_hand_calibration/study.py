"""Frozen-CAD landmarks, offline annotation, fitting, held-out reports."""
import hashlib
from pathlib import Path
import re
import shutil

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from workspace import ROOT,local,read_json,write_json,identifier,utc,digest,environment,verify_model,transform
from capture import load_frame
from geometry import project,point_in_world,fit_pose,residual_stats


def model():
    verify_model()
    return mujoco.MjModel.from_xml_path(str(local('model/hand.xml')))


def visual_vertices(m,body):
    data=mujoco.MjData(m);mujoco.mj_kinematics(m,data);bid=m.body(body).id;points=[]
    for g in range(m.ngeom):
        if m.geom_bodyid[g]!=bid or m.geom_group[g]!=1 or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
        world=m.mesh_vert[start:start+n]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        points.append((world-data.xpos[bid])@data.xmat[bid].reshape(3,3))
    if not points:raise ValueError(f'No visual mesh geometry on body {body}')
    return np.unique(np.round(np.concatenate(points),10),axis=0)


def catalog(body,count):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not 6<=count<=50:raise ValueError('Choose 6..50 candidate landmarks')
    m=model();points=visual_vertices(m,body)
    if len(points)<count:raise ValueError('Not enough distinct mesh vertices')
    selected=[int(np.argmin(points.sum(axis=1)))];distance=np.full(len(points),np.inf)
    for _ in range(count-1):
        distance=np.minimum(distance,np.linalg.norm(points-points[selected[-1]],axis=1))
        selected.append(int(np.argmax(distance)))
    values=[dict(id=f'{body}_v{i:02d}',body=body,xyz_body_m=points[index].tolist(),
        kind='actual_visual_mesh_vertex',needs_visual_identification=True) for i,index in enumerate(selected)]
    path=local(f'configs/landmarks_{body}.json')
    if path.exists():raise ValueError('Landmark catalog already exists; preserve annotations and use a new name/body')
    write_json(path,dict(model_sha256=digest(local('model/hand.xml')),landmarks=values,
        warning='Candidates are actual CAD vertices, not automatically identified real features. Use only clearly corresponding visible points.'))
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    for ax,(a,b) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.scatter(points[::max(1,len(points)//3000),a]*1000,points[::max(1,len(points)//3000),b]*1000,s=1,c='gray')
        for i,index in enumerate(selected):
            ax.scatter(points[index,a]*1000,points[index,b]*1000,c='red',s=14)
            ax.annotate(f'v{i:02d}',points[index,[a,b]]*1000,fontsize=8)
        ax.set(xlabel='XYZ'[a]+' body mm',ylabel='XYZ'[b]+' body mm',aspect='equal');ax.grid(alpha=.2)
    fig.suptitle(f'Natural CAD landmark candidates: {body} (inspect before annotating)')
    figure=local(f'results/landmarks_{body}.png');fig.savefig(figure,dpi=170);plt.close(fig)
    return dict(catalog=str(path.relative_to(ROOT)),figure=str(figure.relative_to(ROOT)),count=count)


def measured_q(meta):
    joint=meta.get('joints')
    return np.asarray(joint['q']) if joint and joint['usable'] else None


def overlay(frame_path,catalog_path,transform_path):
    rgb,_,meta=load_frame(frame_path);m=model();catalog=read_json(catalog_path)
    q=measured_q(meta)
    world=np.array([point_in_world(m,p['body'],p['xyz_body_m'],q) for p in catalog['landmarks']])
    uv,z=project(world,read_json(transform_path)['T_world_from_camera'],meta['camera_intrinsics'])
    image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);visible=0
    for p,pixel,depth in zip(catalog['landmarks'],uv,z):
        if depth<=0 or not np.isfinite(pixel).all():continue
        u,v=np.rint(pixel).astype(int)
        if 0<=u<image.shape[1] and 0<=v<image.shape[0]:
            visible+=1;cv2.circle(image,(u,v),4,(255,0,255),1)
            cv2.putText(image,p['id'],(u+3,v),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,0,255),1)
    cv2.putText(image,'Projected CAD candidates only - NOT measured correspondences',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,255,255),1)
    out=local('results')/('projection_'+identifier());out.mkdir()
    cv2.imwrite(str(out/'overlay.png'),image)
    record=dict(frame=str(local(frame_path).relative_to(ROOT)),frame_sha256=digest(local(frame_path)),
        catalog=catalog_path,transform=transform_path,in_image_candidates=visible,
        hardware_access=False,calibration_accuracy_verified=False,
        note='Fixed palm points need no joint reads; any articulated landmark requires measured stable joints.')
    write_json(out/'report.json',record)
    return dict(directory=str(out.relative_to(ROOT)),**record)


def annotate(frame_path,catalog_path,split,output):
    rgb,depth,meta=load_frame(frame_path);definitions=read_json(catalog_path);m=model()
    if definitions['model_sha256']!=digest(local('model/hand.xml')):raise ValueError('Catalog/model hash mismatch')
    landmarks=definitions['landmarks'];q=measured_q(meta)
    world=np.array([point_in_world(m,p['body'],p['xyz_body_m'],q) for p in landmarks])
    prior=read_json('configs/current_transform.json')['T_world_from_camera']
    predicted,z=project(world,prior,meta['camera_intrinsics']);width=rgb.shape[1]
    chosen=[0];clicks={};window='Natural landmarks | reference right, click real feature left | [ ] select | U undo | S save | Esc cancel'
    def click(event,x,y,flags,param):
        if event!=cv2.EVENT_LBUTTONDOWN:return
        if x>=width:
            valid=np.flatnonzero(z>0)
            if len(valid):chosen[0]=int(valid[np.argmin(np.linalg.norm(predicted[valid]-[x-width,y],axis=1))])
        else:clicks[chosen[0]]=[int(x),int(y)]
    cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.setMouseCallback(window,click)
    try:
        while True:
            left=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);right=np.full_like(left,245)
            for i,pixel in enumerate(predicted):
                if z[i]<=0 or not np.isfinite(pixel).all():continue
                xy=tuple(np.round(pixel).astype(int));color=(0,0,220) if i==chosen[0] else (140,70,0)
                cv2.circle(right,xy,4,color,-1);cv2.putText(right,landmarks[i]['id'],xy,cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
            for i,xy in clicks.items():
                cv2.circle(left,tuple(xy),4,(0,200,0),-1);cv2.putText(left,landmarks[i]['id'],tuple(xy),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,200,0),1)
            cv2.putText(left,f'Selected {landmarks[chosen[0]]["id"]} | {split} | {meta["pose_id"]}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
            cv2.imshow(window,np.concatenate([left,right],axis=1));key=cv2.waitKey(30)&255
            if key in (27,ord('q')):return dict(saved=False)
            if key==ord('['):chosen[0]=(chosen[0]-1)%len(landmarks)
            if key==ord(']'):chosen[0]=(chosen[0]+1)%len(landmarks)
            if key==ord('u'):clicks.pop(chosen[0],None)
            if key==ord('s'):break
    finally:cv2.destroyAllWindows()
    existing=read_json(output) if local(output).exists() else dict(schema_version=1,observations=[])
    for i,pixel in clicks.items():
        u,v=pixel;patch=depth[max(0,v-2):v+3,max(0,u-2):u+3];valid=patch[patch>0]
        item=dict(frame=str(local(frame_path).relative_to(ROOT)),frame_sha256=digest(local(frame_path)),
            model_sha256=digest(local('model/hand.xml')),landmark=landmarks[i],world_point_m=world[i].tolist(),
            pixel=pixel,pose_id=meta['pose_id'],split=split,annotated_at_utc=utc(),
            depth_m=float(np.median(valid)) if valid.size else None,
            depth_patch_std_m=float(np.std(valid)) if valid.size else None,
            depth_reliable=bool(valid.size>=20 and np.std(valid)<.003))
        existing['observations']=[r for r in existing['observations'] if (r['frame'],r['landmark']['id'])!=(item['frame'],item['landmark']['id'])]
        existing['observations'].append(item)
    write_json(output,existing)
    return dict(saved=True,points_added=len(clicks),output=str(local(output).relative_to(ROOT)))


def checked_observations(path):
    record=read_json(path);rows=record['observations'];cache={};m=model();world=[];pixels=[];K=None;serial=None
    seen=set()
    for r in rows:
        key=(r['frame'],r['landmark']['id'])
        if key in seen:raise ValueError('Duplicate frame/landmark observation')
        seen.add(key)
        if r['frame'] not in cache:
            if digest(local(r['frame']))!=r['frame_sha256']:raise ValueError('Annotated frame changed')
            cache[r['frame']]=load_frame(r['frame'])
        _,_,meta=cache[r['frame']]
        if r['pose_id']!=meta['pose_id']:raise ValueError('Observation pose ID differs from captured frame')
        if meta.get('split') in ('train','validation') and r['split']!=meta['split']:
            raise ValueError('Annotation split differs from the reserved capture split')
        uv=np.asarray(r['pixel'],float);height,width=cache[r['frame']][0].shape[:2]
        if uv.shape!=(2,) or not np.isfinite(uv).all() or not(0<=uv[0]<width and 0<=uv[1]<height):
            raise ValueError('Annotated image point is outside the recorded frame')
        if r['model_sha256']!=digest(local('model/hand.xml')):raise ValueError('Model changed since annotation')
        if K is None:K=np.asarray(meta['camera_intrinsics']);serial=meta['camera_serial']
        if not np.allclose(K,meta['camera_intrinsics'],rtol=0,atol=1e-6) or serial!=meta['camera_serial']:
            raise ValueError('One fit must use one camera profile/serial; do not pool changed intrinsics')
        p=r['landmark'];point=point_in_world(m,p['body'],p['xyz_body_m'],measured_q(meta))
        if not np.allclose(point,r['world_point_m'],atol=1e-8):raise ValueError('Landmark world coordinates do not match frozen model/measurements')
        world.append(point);pixels.append(r['pixel'])
    if not rows:raise ValueError('No landmark observations')
    train={r['pose_id'] for r in rows if r['split']=='train'};valid={r['pose_id'] for r in rows if r['split']=='validation'}
    if train&valid:raise ValueError('Train/validation share pose IDs; reserve independent poses rather than adjacent frames')
    return rows,np.asarray(world),np.asarray(pixels),K,serial


def fit(observations,initial,name):
    rows,world,pixels,K,serial=checked_observations(observations)
    train=np.array([r['split']=='train' for r in rows]);cfg=read_json('configs/experiment.json')['fit']
    prior=read_json('configs/current_transform.json')['T_world_from_camera']
    fitted,diagnostics=fit_pose(world[train],pixels[train],K,initial=prior if initial=='current' else None,
        threshold=cfg['ransac_px'],scale=cfg['robust_scale_px'],seed=cfg['seed'])
    name=name or identifier()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise ValueError('Result name must contain only letters, digits, _ or -')
    directory=local('results')/name
    if directory.exists():raise ValueError('Result already exists; choose a new name')
    directory.mkdir()
    (directory/'code').mkdir()
    for source in ROOT.glob('*.py'):shutil.copy2(source,directory/'code'/source.name)
    snapshot=read_json(observations);copied={}
    (directory/'frames').mkdir()
    for observation in snapshot['observations']:
        source=observation['frame']
        if source not in copied:
            target=directory/'frames'/(observation['frame_sha256'][:12]+'_'+Path(source).name)
            shutil.copy2(local(source),target)
            copied[source]=str(target.relative_to(ROOT))
        observation['frame']=copied[source]
    snapshot['source_frame_copies']=copied
    write_json(directory/'observations.json',snapshot)
    write_json(directory/'candidate_transform.json',dict(T_world_from_camera=fitted.tolist(),
        convention='OpenCV color optical frame -> frozen MuJoCo world, meters',
        camera_serial=serial,camera_intrinsics=K.tolist(),object_only_offset_applied=False))
    write_json(directory/'fit.json',dict(created_at_utc=utc(),initial=initial,diagnostics=diagnostics,
        synthetic_test_only=bool(read_json(observations).get('synthetic_test_only',False)),
        prior_transform=prior,config=read_json('configs/experiment.json'),environment=environment(),
        code_sha256={p.name:digest(p) for p in ROOT.glob('*.py')},
        model_manifest=read_json('model/manifest.json'),source_observations_sha256=digest(local(observations))))
    return report(directory)


def report(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=local(directory);recipe=read_json(directory/'fit.json')
    rows,world,pixels,K,serial=checked_observations(directory/'observations.json')
    candidate=read_json(directory/'candidate_transform.json')['T_world_from_camera']
    prior=recipe['prior_transform'];before,bz=project(world,prior,K);after,az=project(world,candidate,K)
    eb=np.linalg.norm(before-pixels,axis=1);ea=np.linalg.norm(after-pixels,axis=1)
    summary={}
    for split in ['train','validation']:
        mask=np.array([r['split']==split for r in rows]);depth=np.array([r['depth_m'] or np.nan for r in rows])
        reliable=np.array([r['depth_reliable'] for r in rows])&mask
        summary[split]=dict(poses=sorted({r['pose_id'] for r in rows if r['split']==split}),
            before_pixel_error=residual_stats(eb[mask]),after_pixel_error=residual_stats(ea[mask]),
            before_depth_error_mm=residual_stats((bz[reliable]-depth[reliable])*1000),
            after_depth_error_mm=residual_stats((az[reliable]-depth[reliable])*1000))
    delta=transform(candidate)@np.linalg.inv(transform(prior))
    validation=summary['validation']['after_pixel_error']
    supported=(validation is not None and validation['n']>=6 and validation['p95']<=recipe['config']['fit']['heldout_p95_limit_px']
        and len(summary['train']['poses'])>=2 and recipe['diagnostics']['jacobian_rank']==6
        and recipe['diagnostics']['scaled_jacobian_condition']<=recipe['config']['fit'].get('max_scaled_jacobian_condition',1e6))
    result=dict(status=('synthetic_test_only' if recipe.get('synthetic_test_only') else
                       ('candidate_passes_image_checks' if supported else 'candidate_unvalidated')),
        synthetic_test_only=recipe.get('synthetic_test_only',False),
        absolute_accuracy_established=False,camera_serial=serial,metrics=summary,
        delta_translation_mm=(delta[:3,3]*1000).tolist(),delta_rotation_deg=float(Rotation.from_matrix(delta[:3,:3]).magnitude()*180/np.pi),
        model_dependence='Agreement is relative to frozen CAD and measured joints; mounting/geometry/joint-zero errors may be confounded.',
        production_configuration_modified=False)
    write_json(directory/'report.json',result)
    for frame in sorted({r['frame'] for r in rows}):
        rgb,_,_=load_frame(frame);image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        for i,r in enumerate(rows):
            if r['frame']!=frame:continue
            if bz[i]>0:cv2.circle(image,tuple(np.rint(before[i]).astype(int)),5,(0,80,255),1)
            if az[i]>0:cv2.circle(image,tuple(np.rint(after[i]).astype(int)),4,(0,220,0),1)
            cv2.drawMarker(image,tuple(map(int,r['pixel'])),(255,255,0),cv2.MARKER_CROSS,8,1)
        label=Path(frame).parent.name+'_'+Path(frame).stem
        cv2.putText(image,'Before orange | after green | observed cyan',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
        cv2.imwrite(str(directory/(label+'_overlay.png')),image)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for split,color in [('train','#377eb8'),('validation','#ff7f00')]:
        mask=np.array([r['split']==split for r in rows])
        if mask.any():
            axes[0].scatter(eb[mask],ea[mask],label=split,c=color)
            x=np.sort(ea[mask]);axes[1].plot(x,np.arange(1,len(x)+1)/len(x),label=split,c=color)
    axes[0].set(xlabel='Before pixel error',ylabel='After pixel error');axes[1].set(xlabel='After pixel error',ylabel='Cumulative fraction')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    for extension in ['png','pdf']:fig.savefig(directory/f'residuals.{extension}',dpi=170)
    plt.close(fig)
    (directory/'index.html').write_text('<meta charset="utf-8"><title>Hand camera calibration</title><pre>'+__import__('html').escape(__import__('json').dumps(result,indent=2))+'</pre><img width="100%" src="residuals.png">'+''.join(f'<img width="640" src="{p.name}">' for p in directory.glob('*overlay.png')))
    return dict(directory=str(directory.relative_to(ROOT)),**result)
