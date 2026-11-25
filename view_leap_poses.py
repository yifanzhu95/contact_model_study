#!/usr/bin/env python3
"""Interactive static LEAP pose browser. No serial, LCM, torque or physics stepping."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import queue
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT=Path(__file__).resolve().parent


def make_gallery(joint_order):
    spec=mujoco.MjSpec()
    spec.modelname='LEAP pose catalogue (static, no hardware)'
    spec.visual.global_.offwidth=1800
    spec.visual.global_.offheight=1100
    spec.worldbody.add_light(pos=[0,0,2],dir=[0,0,-1],diffuse=[.9,.9,.9])
    spec.worldbody.add_light(pos=[1,-1,1],dir=[-1,1,-1],diffuse=[.7,.7,.7])
    paths=[ROOT/'scenes/leap/hardware/calibrated_hand.xml',ROOT/'scenes/leap/leap_right_hand_high.xml']
    centers=[]
    for kind,path in enumerate(paths):
        for slot in range(9):
            center=[(slot%3-1)*.30,(1-slot//3)*.28,0]
            if kind==0:centers.append(center)
            child=mujoco.MjSpec.from_file(str(path))
            child.meshdir=str((path.parent/child.meshdir).resolve())
            for element in list(child.excludes)+list(child.actuators):child.delete(element)
            for geom in child.geoms:
                geom.contype=geom.conaffinity=0
                # Keep visual meshes and fingertip pad meshes, hide proxy boxes.
                geom.group=1 if geom.group==1 or geom.name in ['if_tip','mf_tip','rf_tip','th_tip'] else 3
            frame=spec.worldbody.add_frame(name=f'frame_{kind}_{slot}',pos=center)
            spec.attach(child,prefix=f'h{kind}_{slot}_',frame=frame)
    model=spec.compile();data=mujoco.MjData(model)
    slots={}
    for kind in range(2):
        for slot in range(9):
            prefix=f'h{kind}_{slot}_'
            slots[kind,slot]=dict(qpos=[model.jnt_qposadr[model.joint(prefix+n).id] for n in joint_order],
                geoms=[i for i in range(model.ngeom) if (model.geom(i).name or '').startswith(prefix)])
    return model,data,slots,np.asarray(centers)


def apply_page(model,data,slots,poses,page,selected,kind,detail,base_rgba):
    model.geom_rgba[:]=base_rgba
    for (which,slot),info in slots.items():
        index=page*9+slot
        visible=which==kind and index<len(poses) and (not detail or index==selected)
        if not visible:model.geom_rgba[info['geoms'],3]=0
        if index<len(poses):data.qpos[info['qpos']]=poses[index]['q']
    mujoco.mj_forward(model,data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,default=ROOT/'results/pose_catalog_20261116/catalog.json')
    parser.add_argument('--check',action='store_true',help='Compile and apply every pose, then exit without a window.')
    parser.add_argument('--object',choices=['all','cube'],default='cube',help='Show only Cube-related poses by default.')
    parser.add_argument('--screenshot',type=Path,help='Save the first gallery page without opening a window.')
    args=parser.parse_args()
    catalog=json.loads(args.catalog.read_text());poses=catalog['poses']
    if args.object=='cube':
        from build_leap_pose_catalog import cube_poses
        poses=cube_poses(poses)
    if not poses:raise ValueError('Empty pose catalogue')
    model,data,slots,centers=make_gallery(catalog['joint_order'])
    base_rgba=model.geom_rgba.copy()
    if args.check:
        for index in range(len(poses)):
            for kind in range(2):
                apply_page(model,data,slots,poses,index//9,index,kind,False,base_rgba)
                if not np.isfinite(data.xpos).all():raise ValueError(f"Invalid FK: {poses[index]['id']}")
        assert model.nu==0
        print(f'Validated {len(poses)} poses on both models; {model.nq} joints; {model.nu} actuators; static forward kinematics only.')
        return
    selected=page=kind=0;detail=False
    events=queue.SimpleQueue()
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=[0,0,.06];camera.distance=1.18;camera.azimuth=110;camera.elevation=-48
    if args.screenshot:
        apply_page(model,data,slots,poses,page,selected,kind,detail,base_rgba)
        import PIL.Image
        with mujoco.Renderer(model,height=1100,width=1800) as renderer:
            option=mujoco.MjvOption();option.geomgroup[3]=0
            renderer.update_scene(data,camera,scene_option=option)
            PIL.Image.fromarray(renderer.render()).save(args.screenshot)
        print(args.screenshot);return
    print(f'Loaded {len(poses)} distinct poses. No hardware communication. No physics stepping.',flush=True)
    def fit(viewer):
        viewer.cam.lookat[:]=centers[selected%9]+[0,0,.07] if detail else [0,0,.06]
        viewer.cam.distance=.42 if detail else 1.18
    with mujoco.viewer.launch_passive(model,data,key_callback=events.put,show_left_ui=False,show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:]=camera.lookat;viewer.cam.distance=camera.distance
            viewer.cam.azimuth=camera.azimuth;viewer.cam.elevation=camera.elevation
            viewer.opt.geomgroup[3]=0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]=False
        running=True;dirty=True
        while viewer.is_running() and running:
            refit=False
            while not events.empty():
                key=events.get();previous=selected
                if key in [81,256]:running=False
                elif key in [78,262,264,32]:selected=(selected+1)%len(poses)
                elif key in [80,263,265]:selected=(selected-1)%len(poses)
                elif key==267:selected=min(len(poses)-1,(page+1)*9)
                elif key==266:selected=max(0,(page-1)*9)
                elif 49<=key<=57:selected=min(len(poses)-1,page*9+key-49)
                elif key in [71,257]:detail=not detail;refit=True
                elif key==70:refit=True
                elif key==77:kind=1-kind
                elif key==83:print(json.dumps(poses[selected],indent=2),flush=True)
                if selected!=previous and detail:refit=True
                page=selected//9;dirty=True
            if dirty:
                with viewer.lock():
                    apply_page(model,data,slots,poses,page,selected,kind,detail,base_rgba)
                    if refit:fit(viewer)
                    viewer.user_scn.ngeom=0
                    for slot in range(9):
                        index=page*9+slot
                        if index>=len(poses) or (detail and index!=selected):continue
                        geom=viewer.user_scn.geoms[viewer.user_scn.ngeom]
                        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LABEL,np.zeros(3),
                            centers[slot]+[0,.12,.16],np.eye(3).ravel(),np.array([1.,.8,.2,1.]))
                        geom.label=('> ' if index==selected else '')+f'{slot+1}: {poses[index]["id"]}'
                        viewer.user_scn.ngeom+=1
                pose=poses[selected];source=pose['sources'][0]
                rows=['Index ','Middle','Ring  ','Thumb ']
                values='\n'.join(f'{name}: '+', '.join(f'{v: .4f}' for v in pose['q'][i*4:i*4+4]) for i,name in enumerate(rows))
                status=f"{pose['id']} | {pose['name']}\n{source['status']}\n{values}"
                limits=pose.get('outside_calibrated_joint_limits',[])
                if limits:status+='\nOutside calibrated limits: '+', '.join(limits)
                origin=Path(source['path'])
                try:origin=origin.relative_to(ROOT)
                except ValueError:pass
                viewer.set_texts([(None,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    'STATIC POSE BROWSER - no hardware\n'+status,None),
                    (None,mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                     f"Page {page+1}/{(len(poses)+8)//9} | {'Detail' if detail else '9-pose gallery'}\n"+
                     ('Calibrated hand (30 deg mount)' if kind==0 else 'Original high-accuracy hand (27 deg mount)'),None),
                    (None,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     'N/P or arrows: next/previous | PgUp/PgDn: page | 1-9: select\nG/Enter: gallery/detail | M: model | F: fit | S: print | Q: close\nMouse: orbit / pan / zoom',None),
                    (None,mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,f'{origin}\n{source["role"]} | {len(pose["sources"])} sources (full list in catalog.json)',None)])
                dirty=False
            viewer.sync();time.sleep(.025)
    print('Pose browser closed.',flush=True)


if __name__=='__main__':main()
