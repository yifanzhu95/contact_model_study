#!/usr/bin/env python3
"""Recompute method 4 from measured link-plane contacts after torque is off."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import argparse
import shutil
import numpy as np
import trimesh
from workspace import local,read_json,write_json,digest,utc
from articulated_motion import model
from method4_reference import link_plane_stopped_cube,refine_translation,cube_orientation_check


def refine(run_dir):
    run=local(run_dir);session=read_json(run/'session.json')
    if not session.get('torque_disabled_verified'):raise ValueError('Refinement requires verified torque off after cube removal')
    if not (run/'object_removed.trigger').exists():raise ValueError('Operator cube-removal acknowledgement is missing')
    cfg=session['config'];m=model();samples=session['samples'];initial=np.array(session['initial_transform']['T_world_from_camera'])
    references=[link_plane_stopped_cube(m,np.array(s['measured_q']),cfg['cube_edge_m'],cfg['pad_surface_offset_m']) for s in samples]
    centers=np.array([x[0] for x in references]);rotations=np.array([x[1]['expected_cube_axes_world'] for x in references])
    live=read_json('configs/live_tracking.json');mesh_center=trimesh.load(live['mesh'],process=False).bounds.mean(0)
    poses=np.array([s['raw_camera_object_pose'] for s in samples]);T,stats=refine_translation(initial,poses,centers,mesh_center,cfg['max_translation_change_m'],cfg['min_stable_frames'])
    orientation=cube_orientation_check(initial,poses,rotations)
    previous=run/'candidate.json'
    if previous.exists() and not (run/'candidate_horizontal_assumption.json').exists():shutil.copy2(previous,run/'candidate_horizontal_assumption.json')
    write_json(run/'candidate.json',dict(T_world_from_camera=T.tolist(),method=4,parent_method=3,
        status='Translation anchor using measured tilted finger links; independent placement validation required.',
        translation_fit=stats,rotation_verification=orientation,object_only_offset_applied=False,
        torque_disabled_verified=True,reference_geometry='measured finger-link plane; no horizontal/table assumption',
        reference_center_world_m_mean=centers.mean(0).tolist(),
        support_plane_tilt_from_world_vertical_deg=float(np.mean([r[1]['support_tilt_from_world_vertical_deg'] for r in references])),
        references=[dict(timestamp=s['timestamp'],center_world_m=c.tolist(),geometry=g) for s,(c,g) in zip(samples,references)],
        mesh_center_object_m=mesh_center.tolist(),production_configuration_modified=False,created_at=utc()))
    shutil.copy2(__file__,run/'code/method4_refine.py');shutil.copy2(ROOT/'method4_reference.py',run/'code/method4_reference_link_plane.py')
    write_json('method4/latest_capture.json',dict(directory=str(run.relative_to(ROOT)),reference='measured finger-link plane',status='refined_torque_off'))
    print('Method 4 reference follows measured finger links.');print('Translation world (mm):',np.array(stats['translation_change_world_m'])*1000)
    print('Translation camera (mm):',np.array(stats['translation_change_camera_m'])*1000);print('Orientation:',orientation)
    return T,stats,orientation


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);a=p.parse_args();refine(a.run_dir)
