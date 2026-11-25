#!/usr/bin/env python3
"""Known-cube reference geometry and translation-only refinement for method 4.

This module is hardware-free. Motion/acquisition must use an exclusive hand
connection, a fresh collision screen, and operator placement/removal stages.
Nothing here modifies the production transform or object correction.
"""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
sys.path.insert(0,str(ROOT))
import argparse
import json
import mujoco
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from articulated_motion import model,collision_pairs,clearance,screen
from workspace import local,read_json,write_json,digest,identifier
from ContactModelStudy.Hardware.LCM import rigid_transform


def pad_reference(m,data,finger):
    """Middle of the distal pad footprint, on its modeled upper face.

    The pad is a 3D mesh, not the fingertip site (which is below the contact
    surface and at the finger's end). Physical fabric thickness/compression
    remains a reference uncertainty and must not be silently treated as zero.
    """
    geom=m.geom(finger+'_tip').id;body=int(m.geom_bodyid[geom]);mid=int(m.geom_dataid[geom])
    vertices=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
    world=vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
    R=data.xmat[body].reshape(3,3);body_vertices=(world-data.xpos[body])@R
    low,high=body_vertices.min(0),body_vertices.max(0);point=(low+high)*.5;point[0]=low[0]
    normal=-R[:,0]
    return dict(finger=finger,body=m.body(body).name,point_body_m=point.tolist(),
        surface_center_world_m=(R@point+data.xpos[body]).tolist(),normal_world=normal.tolist(),
        tilt_from_vertical_deg=float(np.degrees(np.arccos(np.clip(normal[2],-1,1)))),
        footprint_world_bounds_m=[world.min(0).tolist(),world.max(0).tolist()])


def level_pose(m,inward_yaw=.15):
    q=np.r_[[.52,inward_yaw,0,0],[.52,0,0,0],[.52,-inward_yaw,0,0],[0,0,.55,.35]]
    d=mujoco.MjData(m)
    for j,finger in [(0,'if'),(4,'mf'),(8,'rf')]:
        def vertical_component(angle):
            q[j]=angle;d.qpos[:]=q;mujoco.mj_fwdPosition(m,d)
            return float((d.xmat[m.body(finger+'_ds').id].reshape(3,3)@np.array([0.,-1.,0.]))[2])
        q[j]=brentq(vertical_component,0.,1.)
    if np.any(q<m.jnt_range[:,0]) or np.any(q>m.jnt_range[:,1]):raise ValueError('Reference pose outside joint limits')
    return q


def approach_path(m,start,goal,margin=.004):
    """Spread close resting fingers monotonically before the normal approach.

    An empty torque-off hand can already be inside the conservative 4 mm
    planning margin. An entry retreat is permitted only above 2.5 mm, with
    sampled clearance never decreasing, and every later segment above 4 mm.
    The live measured abort threshold stays at 2 mm throughout.
    """
    start=np.asarray(start,float);goal=np.asarray(goal,float);pairs=collision_pairs(m);d=mujoco.MjData(m)
    park=np.array(read_json('configs/articulated_sweep.json')['waypoints'][0]['target'])
    distance,_=clearance(m,d,start,pairs)
    if distance>=margin:
        check=screen(m,[start,park,goal],pairs,margin)
        if check['passed']:return [park,goal],dict(normal_path=check,entry_retreat=False)
    if distance<.0025:raise ValueError('Initial modeled clearance below 2.5 mm; manual separation is required before torque enable')
    for spread in [0.,.10,.15,.20]:
        retreat=start.copy();retreat[[1,5,9]]=[-spread,0,spread]
        samples=np.linspace(start,retreat,80)
        distances=np.array([clearance(m,d,q,pairs)[0] for q in samples])
        if distances[-1]<margin or np.min(np.diff(distances))<-.00002 or distances.min()<distance-.00002:continue
        rest=screen(m,[retreat,park,goal],pairs,margin)
        if rest['passed']:
            return [retreat,park,goal],dict(entry_retreat=True,initial_clearance_mm=distance*1000,
                retreat_clearance_mm=(distances*1000).tolist(),normal_path=rest,
                rationale='Only outward separation first; clearance is sampled monotonically nondecreasing. Subsequent motion satisfies the full margin.')
    raise ValueError('No screened clearance-increasing entry path found')


def expected_cube_center(m,q,edge_m=.07,pad_surface_offset_m=0.):
    d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_fwdPosition(m,d)
    pads=[pad_reference(m,d,f) for f in ['if','mf','rf']]
    center=np.array(pads[1]['surface_center_world_m']);center[2]+=edge_m*.5+pad_surface_offset_m
    heights=np.array([p['surface_center_world_m'][2] for p in pads])
    return center,dict(pads=pads,pad_height_spread_mm=float(np.ptp(heights)*1000),
        max_pad_tilt_deg=max(p['tilt_from_vertical_deg'] for p in pads),
        definition='Cube geometric center above the middle distal pad footprint center, with bottom face horizontal.',
        cube_edge_m=edge_m,pad_surface_offset_m=pad_surface_offset_m)


def proximal_support(m,data,finger):
    """Upper planar face of the proximal housing, where the cube rests."""
    geom=m.geom(finger+'_px_visual').id;body=int(m.geom_bodyid[geom]);mid=int(m.geom_dataid[geom])
    vertices=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
    world=vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
    R=data.xmat[body].reshape(3,3);body_vertices=(world-data.xpos[body])@R
    low,high=body_vertices.min(0),body_vertices.max(0);point=(low+high)*.5;point[2]=low[2]
    center=R@point+data.xpos[body];normal=-R[:,2]
    return dict(finger=finger,point_body_m=point.tolist(),surface_center_world_m=center.tolist(),
        normal_world=normal.tolist(),tilt_from_vertical_deg=float(np.degrees(np.arccos(np.clip(normal[2],-1,1)))))


def palm_stopped_cube(m,q,edge_m=.07,surface_offset_m=0.):
    """Cube lower rear corners against fixed motor faces; bottom on fingers.

    The selected palm plane is the exposed front of the fixed gray index/ring
    motor housings. Verify each corner lies on the actual finite mesh face.
    """
    from trimesh.triangles import closest_point
    d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_fwdPosition(m,d)
    support=[proximal_support(m,d,f) for f in ['if','mf','rf']]
    centers=np.array([s['surface_center_world_m'] for s in support]);bottom=float(centers[:,2].max()+surface_offset_m)
    middle_y=float(centers[1,1]);body=m.body('palm').id;R=d.xmat[body].reshape(3,3);p=d.xpos[body]
    normal=-R[:,2];offset=.0282+float(normal@p)
    if abs(normal[0])<.1:raise ValueError('Palm face does not constrain the longitudinal cube stop')
    rear_x=(offset-normal[1]*middle_y-normal[2]*bottom)/normal[0]
    corners=np.array([[rear_x,middle_y-edge_m/2,bottom],[rear_x,middle_y+edge_m/2,bottom]])
    g=m.geom('palm_visual').id;mid=m.geom_dataid[g]
    v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
    v=(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-p)@R
    faces=m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid]+m.mesh_facenum[mid]];tri=v[faces]
    tri=tri[np.all(abs(tri[:,:,2]+.0282)<3e-6,axis=1)]
    if not len(tri):raise ValueError('Known palm stop plane is absent from the frozen visual mesh')
    errors=[]
    for corner in (corners-p)@R:
        closest=closest_point(tri,np.tile(corner,(len(tri),1)));errors.append(float(np.linalg.norm(closest-corner,axis=1).min()))
    if max(errors)>.0005:raise ValueError('Cube corners miss the finite palm housing faces; placement stop is not geometrically valid')
    center=np.array([rear_x+edge_m/2,middle_y,bottom+edge_m/2])
    return center,dict(reference='Both palm-side bottom corners touch the fixed gray index/ring motor-housing front faces; cube centered over middle proximal finger; bottom face on upper finger surfaces.',
        support_surfaces=support,bottom_height_m=bottom,palm_stop_normal_world=normal.tolist(),palm_stop_plane_offset_m=offset,
        lower_rear_corners_world_m=corners.tolist(),corner_to_finite_palm_face_error_mm=(np.array(errors)*1000).tolist(),
        support_height_spread_mm=float(np.ptp(centers[:,2])*1000),max_support_tilt_deg=max(s['tilt_from_vertical_deg'] for s in support),
        cube_edge_m=edge_m,surface_offset_m=surface_offset_m)


def cube_orientation_check(initial,raw_camera_object_poses):
    """Verify horizontal resting face and aligned rear edge, modulo cube symmetry.

    Two bottom rear corners against the same palm plane specify geometric yaw;
    without labeled face identities the 24 equivalent cube orientations remain.
    This is a verification only: method 3's rotation is never changed here.
    """
    world=np.array([np.asarray(initial)[:3,:3]@np.asarray(p)[:3,:3] for p in raw_camera_object_poses])
    rotations=Rotation.from_matrix(world);sym=Rotation.create_group('O')
    angles=np.array([min((sym.inv()*r).magnitude()) for r in rotations])
    vertical=np.degrees(np.arccos(np.clip(abs(world[:,2,:]).max(axis=1),-1,1)))
    return dict(resting_face_normal_rms_deg=float(np.sqrt(np.mean(vertical**2))),resting_face_normal_max_deg=float(vertical.max()),
        cube_axes_alignment_rms_deg=float(np.degrees(np.sqrt(np.mean(angles**2)))),
        rotation_updated=False,face_identity_verified=False,
        note='Axes are checked modulo 24 cube symmetries; printed face identity is not inferred. Mesh axes are assumed to represent physical cube faces.')


def recover_raw_camera_pose(published,camera_to_published,object_offset_camera_m):
    raw=np.linalg.inv(rigid_transform(camera_to_published))@rigid_transform(published)
    raw[:3,3]-=np.asarray(object_offset_camera_m,float)
    return raw


def refine_translation(initial,raw_camera_object_poses,known_world_centers,mesh_center_object_m,
                       max_translation_change_m=.03,min_samples=30):
    T3=rigid_transform(initial).copy();poses=np.array([rigid_transform(p) for p in raw_camera_object_poses])
    centers=np.asarray(known_world_centers,float);mesh_center=np.asarray(mesh_center_object_m,float)
    if centers.shape!=(len(poses),3) or not np.isfinite(centers).all() or mesh_center.shape!=(3,):raise ValueError('Invalid known center observations')
    if len(poses)<min_samples:raise ValueError('Not enough distinct stable tracked frames')
    camera_centers=np.einsum('nij,j->ni',poses[:,:3,:3],mesh_center)+poses[:,:3,3]
    predicted=camera_centers@T3[:3,:3].T+T3[:3,3]
    corrections=centers-predicted;middle=np.median(corrections,axis=0)
    distances=np.linalg.norm(corrections-middle,axis=1);median=np.median(distances);mad=np.median(abs(distances-median))
    keep=distances<=max(.003,float(median+3*1.4826*mad))
    if keep.sum()<min_samples:raise ValueError('Too few stable samples after outlier rejection')
    delta=corrections[keep].mean(0)
    if np.linalg.norm(delta)>max_translation_change_m:raise ValueError('Translation exceeds configured calibration envelope; inspect the reference/tracking')
    T4=T3.copy();T4[:3,3]+=delta
    before=np.linalg.norm(corrections[keep],axis=1);after=np.linalg.norm(corrections[keep]-delta,axis=1)
    return T4,dict(total_samples=len(poses),inlier_samples=int(keep.sum()),inliers=keep.tolist(),
        translation_change_world_m=delta.tolist(),translation_change_camera_m=(T3[:3,:3].T@delta).tolist(),
        before_reference_rms_mm=float(np.sqrt(np.mean(before**2))*1000),
        after_reference_rms_mm=float(np.sqrt(np.mean(after**2))*1000),
        residual_scatter_mm=(np.std(corrections[keep]-delta,axis=0)*1000).tolist(),
        rotation_changed=False,object_only_correction_used=False,
        limitation='Single-pose anchor residual is a training residual, not independent accuracy. Placement, contact geometry and tracker bias can be absorbed into translation.')


def plan(config_path):
    config=read_json(config_path);m=model();q=level_pose(m,config['inward_yaw_rad']);pairs=collision_pairs(m)
    d=mujoco.MjData(m);minimum,nearest=clearance(m,d,q,pairs)
    if minimum<config['collision_margin_m']:raise ValueError('Flat pose fails modeled collision clearance')
    center,geometry=palm_stopped_cube(m,q,config['cube_edge_m'],config['pad_surface_offset_m'])
    # Actual hardware start must be read and screened immediately before moving.
    previous=np.array(read_json('configs/articulated_sweep.json')['waypoints'][0]['target'])
    path=screen(m,[previous,q],pairs,config['collision_margin_m'])
    if not path['passed']:raise ValueError('Approach from calibration park fails collision screen')
    result=dict(status='offline_plan_ready_awaiting_empty_hand_confirmation',q_level=q.tolist(),q_level_deg=np.degrees(q).tolist(),
        expected_cube_center_world_m=center.tolist(),geometry=geometry,approach_from_park=path,
        minimum_clearance_mm=float(minimum*1000),nearest_collision_pair=nearest,
        hardware_motion_executed=False,requires_fresh_measured_pose_path_screen=True,
        operator_stages=['Confirm hand empty and clear','Gradually move and settle at level pose',
            'Prompt for cube placement at physical reference','Confirm placement; settle under load and average fresh poses',
            'Prompt for cube removal; keep holding','Confirm removal','Disable and verify torque off','Refine method 3 and save isolated method 4 candidate'])
    out=local('method4/results/plan_'+identifier());out.mkdir(parents=True)
    write_json(out/'plan.json',result);write_json('method4/latest_plan.json',dict(directory=str(out.relative_to(ROOT))))
    print(json.dumps(result,indent=2));return result


def render_plan(plan_data,out):
    """Save a proposed placement image; never touches the real hand."""
    import cv2
    import trimesh
    cfg=read_json('configs/live_tracking.json');m=mujoco.MjModel.from_binary_path(str(local('cache/live_viewer/scene.mjb')))
    m.vis.global_.offwidth=800;m.vis.global_.offheight=600
    d=mujoco.MjData(m);d.qpos[:16]=plan_data['q_level'];center=trimesh.load(cfg['mesh'],process=False).bounds.mean(0)
    d.qpos[16:19]=np.array(plan_data['expected_cube_center_world_m'])-center;d.qpos[19:23]=[1,0,0,0];mujoco.mj_forward(m,d)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.lookat[:]=[.045,.035,.035]
    camera.distance=.36;camera.azimuth=115;camera.elevation=-25
    with mujoco.Renderer(m,height=600,width=800) as renderer:
        renderer.update_scene(d,camera=camera)
        for point in plan_data['geometry']['lower_rear_corners_world_m']:
            scn=renderer.scene;mujoco.mjv_initGeom(scn.geoms[scn.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.002),
                np.array(point),np.eye(3).ravel(),np.array([1.,.15,.1,1.]));scn.ngeom+=1
        image=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
    cv2.putText(image,'METHOD 4 PLAN ONLY - actual hand has not moved',(15,25),0,.6,(0,255,255),1)
    cv2.putText(image,'Red dots: lower rear cube corners against fixed palm housings',(15,575),0,.55,(0,255,255),1)
    cv2.imwrite(str(local(out)/'placement_preview.png'),image)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='method4/config.json');p.add_argument('--plan',action='store_true',required=True);p.add_argument('--preview',action='store_true');a=p.parse_args();result=plan(a.config)
    if a.preview:render_plan(result,read_json('method4/latest_plan.json')['directory'])
