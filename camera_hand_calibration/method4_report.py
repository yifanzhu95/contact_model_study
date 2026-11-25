#!/usr/bin/env python3
"""Reproducible comparison at the method-4 known placement (not a hold-out)."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import argparse
import html
import numpy as np
from workspace import local,read_json,write_json,digest
from method4_reference import cube_orientation_check


def report(run_dir):
    run=local(run_dir);candidate=read_json(run/'candidate.json');session=read_json(run/'session.json');live=read_json('configs/live_tracking.json')
    if not session['torque_disabled_verified']:raise ValueError('Report requires completed torque cleanup')
    samples=session['samples'];poses=np.array([r['raw_camera_object_pose'] for r in samples]);center=np.array(candidate['mesh_center_object_m'])
    raw_centers=np.einsum('nij,j->ni',poses[:,:3,:3],center)+poses[:,:3,3]
    references=np.array([r['center_world_m'] for r in candidate['references']]);expected=np.array([r['geometry']['expected_cube_axes_world'] for r in candidate['references']])
    transforms=[('Method 1, raw',np.array(read_json('configs/current_transform.json')['T_world_from_camera']),False)]
    transforms.append(('Method 1, -15 mm',transforms[0][1],True))
    transforms.extend([('Method 2, raw',np.array(read_json(live['calibrations'][1]['file'])['T_world_from_camera']),False),
        ('Method 3, raw',np.array(session['initial_transform']['T_world_from_camera']),False),('Method 4, raw',np.array(candidate['T_world_from_camera']),False)])
    offset=np.array(read_json('configs/source_camera.json')['object_translation_adjust_camera_m']);rows=[];error_series=[]
    from stationary_rgbd import frozen_geometry
    W,*_=frozen_geometry();landmarks=read_json('configs/articulated_palm_fit.json')['landmarks']
    heldout=[r for r in landmarks if r['split']=='validation'];body=np.array([r['body_xyz_m'] for r in heldout]);pixels=np.array([r['pixel'] for r in heldout])
    K=np.array(read_json('configs/source_camera.json')['camera_intrinsics'])
    for name,T,correction in transforms:
        predicted=(raw_centers+(offset if correction else 0))@T[:3,:3].T+T[:3,3];error=predicted-references
        orientation=cube_orientation_check(T,poses,expected)
        camera_from_palm=np.linalg.inv(T)@W;palm_camera=body@camera_from_palm[:3,:3].T+camera_from_palm[:3,3]
        palm_pixels=(palm_camera@K.T)[:,:2]/palm_camera[:,2,None]
        rows.append(dict(name=name,position_rms_mm=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))*1000),
            mean_error_world_xyz_mm=(error.mean(0)*1000).tolist(),position_scatter_world_xyz_mm=(error.std(0)*1000).tolist(),
            resting_face_normal_rms_deg=orientation['resting_face_normal_rms_deg'],cube_axes_alignment_rms_deg=orientation['cube_axes_alignment_rms_deg'],
            heldout_rigid_palm_reprojection_rms_px=float(np.sqrt(np.mean(np.sum((palm_pixels-pixels)**2,axis=1))))))
        error_series.append(np.linalg.norm(error,axis=1)*1000)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained');names=[r['name'].replace(', ','\n') for r in rows]
    axes[0].bar(names,[r['position_rms_mm'] for r in rows]);axes[0].set_ylabel('Position RMS at reference (mm)')
    axes[1].bar(names,[r['resting_face_normal_rms_deg'] for r in rows]);axes[1].set_ylabel('Resting-face normal RMS (degrees)')
    axes[2].bar(names,[r['heldout_rigid_palm_reprojection_rms_px'] for r in rows]);axes[2].set_ylabel('Independent rigid-palm image RMS (pixels)')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',labelsize=8)
    fig.savefig(run/'method_comparison.png',dpi=170);fig.savefig(run/'method_comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');time=np.array([r['timestamp'] for r in samples]);time-=time[0]
    for row,series in zip(rows,error_series):ax.plot(time,series,label=row['name'])
    ax.set(xlabel='Capture time (s)',ylabel='Position error at reference (mm)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(run/'reference_error_timeseries.png',dpi=170);plt.close(fig)
    flow=None
    if local('cache/method4_postcapture_rgb.png').exists():
        import cv2
        reference=np.load(local('cache/new_static_frame.npz'))['rgb'];current=cv2.cvtColor(cv2.imread(str(local('cache/method4_postcapture_rgb.png'))),cv2.COLOR_BGR2RGB)
        reference[75:151,320:391]=50;current[75:151,320:391]=50
        a=cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(current,cv2.COLOR_RGB2GRAY);mask=np.zeros(a.shape,np.uint8)
        for x0,x1 in [(281,314),(325,358),(370,404)]:mask[156:188,x0:x1]=255
        points=cv2.goodFeaturesToTrack(a,maxCorners=60,qualityLevel=.03,minDistance=3,mask=mask,blockSize=3)
        new,ok,err=cv2.calcOpticalFlowPyrLK(a,b,points,None,winSize=(13,13),maxLevel=2)
        back,ok2,_=cv2.calcOpticalFlowPyrLK(b,a,new,None,winSize=(13,13),maxLevel=2)
        keep=(ok[:,0]>0)&(ok2[:,0]>0)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<.6)&(err[:,0]<25)
        if keep.sum()>=5:flow=dict(matched_features=int(keep.sum()),median_displacement_px=np.median(new[keep,0]-points[keep,0],axis=0).tolist(),paper_tag_masked=True)
    info=dict(samples=len(samples),capture_span_s=float(time[-1]),maximum_abs_joint_skew_ms=float(max(abs(r['joint_skew_ms']) for r in samples)),
        torque_disabled_verified=True,rows=rows,translation_fit=candidate['translation_fit'],rotation_verification=candidate['rotation_verification'],
        support_plane_tilt_deg=candidate['support_plane_tilt_from_world_vertical_deg'],
        placement_is_training_reference=True,independent_accuracy_established=False,palm_image_motion_check=flow,
        production_camera_unchanged=digest(ROOT.parent/'configs/leap_pose_transform.json')==read_json('model/manifest.json')['camera_source_sha256'])
    write_json(run/'comparison.json',info)
    palm3=next(r['heldout_rigid_palm_reprojection_rms_px'] for r in rows if r['name']=='Method 3, raw')
    palm4=rows[-1]['heldout_rigid_palm_reprojection_rms_px']
    conflict=palm4>palm3+1.
    candidate['production_ready']=False
    candidate['independent_cross_checks']=dict(method3_rigid_palm_rms_px=palm3,method4_rigid_palm_rms_px=palm4,
        worsens_rigid_palm=conflict,palm_motion_check=flow)
    write_json(run/'candidate.json',candidate)
    movement=(f" A masked natural-feature image check found {np.linalg.norm(flow['median_displacement_px']):.2f} px median palm displacement after the experiment." if flow else '')
    conflict_text=(f'The independent rigid-palm check changes from {palm3:.2f} px for method 3 to {palm4:.2f} px for method 4.'+movement+
        (' Method 4 may be absorbing cube-tracker bias, physical placement error or contact-model geometry error rather than correcting the true camera extrinsic.' if conflict else '')+
        ' It remains an experimental object-reference candidate, not a globally validated camera calibration.')
    lines=['# Method 4: known-cube translation anchor','',
        f"Captured {len(samples)} unique stable frames over {time[-1]:.2f} s. Cube removal was confirmed before all 16 motors were verified torque-off. Refinement ran after torque cleanup.",'',
        'The cube rests on the measured finger-link plane, not an assumed horizontal/table plane. Lateral centering follows the middle finger; both lower rear corners touch the modeled fixed palm housing faces. No paper-tag observation is used.','',
        '| Calibration | Position RMS at reference (mm) | Resting-face normal RMS (deg) | Cube-axis alignment RMS (deg) | Rigid palm image RMS (px) |','|---|---:|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['name']} | {row['position_rms_mm']:.3f} | {row['resting_face_normal_rms_deg']:.3f} | {row['cube_axes_alignment_rms_deg']:.3f} | {row['heldout_rigid_palm_reprojection_rms_px']:.3f} |")
    stats=candidate['translation_fit'];lines+=['',
        'Method 4 preserves method 3 rotation exactly. Translation change in world XYZ (mm): '+', '.join(f'{v*1000:+.3f}' for v in stats['translation_change_world_m'])+'.',
        'The same change in camera XYZ (mm): '+', '.join(f'{v*1000:+.3f}' for v in stats['translation_change_camera_m'])+'.','',
        'All raw camera poses exclude the old -15 mm object-only correction. Only the explicitly labeled method-1 comparison row includes it. Method 4 needs no additional object-only offset.','',
        'The low method-4 residual is at the placement used to fit its translation. It is not an independent accuracy estimate. Manual centering/contact, nominal cube size, CAD surface differences and tracker bias can be absorbed into translation. Additional placements are needed to establish workspace accuracy. Rotation is verified modulo cube symmetry, not printed face identity.','',
        conflict_text,'',
        'The acquisition-time horizontal-plane calculation is retained separately for provenance; `candidate.json` is the final measured-link-plane result. Production calibration files remain unchanged.','',
        'Reproduce after torque is off:','', '```bash',f'python camera_hand_calibration/method4_refine.py --run-dir {run_dir}',f'python camera_hand_calibration/method4_report.py --run-dir {run_dir}','```']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    (run/'index.html').write_text('<meta charset="utf-8"><style>body{font:16px system-ui;max-width:1100px;margin:2em auto}pre{white-space:pre-wrap}img{max-width:100%}</style><pre>'+html.escape('\n'.join(lines))+'</pre><img src="method_comparison.png"><img src="reference_error_timeseries.png">')
    __import__('shutil').copy2(__file__,run/'code/method4_report.py')
    print('Report:',run/'report.md')
    for row in rows:print(row['name'],f"{row['position_rms_mm']:.3f} mm",f"{row['resting_face_normal_rms_deg']:.3f} deg")
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);a=p.parse_args();report(a.run_dir)
