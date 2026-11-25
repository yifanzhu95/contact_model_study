#!/usr/bin/env python3
"""Read-only live comparison of camera calibrations; never publishes actions."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
sys.path.insert(0,str(ROOT.parent))
import argparse
import json
import threading
import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from workspace import local,read_json,write_json,digest
from ContactModelStudy.Hardware.Camera import load_camera_config
from ContactModelStudy.Hardware.LCM import LCMFeedback,load_pose_transform,rigid_transform
from ContactModelStudy.Utils.Quaternions import matToQuat
from view_leap_tracking import (add_calibrated_camera,initialize_interactive_camera,
    set_object_visual_mesh,update_display)

# MuJoCo 3.14 uses F1--F5 for its own panels/full-screen. F6--F9 are
# unassigned there; numbers, Tab, O, Q and Escape must keep native behavior.
CALIBRATION_KEYS={295:0,296:1,297:2,299:3}  # GLFW F6, F7, F8, F10 (method 4 when present)
CORRECTION_KEY=298  # GLFW F9


def object_world_pose(published,camera_to_published,world_from_camera,after,offset,correction=True):
    """Undo tracker publication, then apply the selected extrinsic exactly once."""
    camera= np.linalg.inv(camera_to_published)@rigid_transform(published)
    # The live tracker already applied the camera-axis offset. Remove it for
    # raw calibration comparison; retain it uniformly when O is enabled.
    if not correction:camera[:3,3]-=np.asarray(offset)
    return world_from_camera@camera@after


class Selection:
    def __init__(self,config):
        self.lock=threading.Lock();self.index=int(config.get('initial_calibration',0));self.correction=config.get('object_correction_default',True);self.config=config
        self.modes=config['calibrations']
        self.transforms=[rigid_transform(read_json(x['file'])['T_world_from_camera']) for x in self.modes]
    def key(self,key):
        with self.lock:
            if key in CALIBRATION_KEYS and CALIBRATION_KEYS[key]<len(self.modes):
                self.index=CALIBRATION_KEYS[key]
                if self.index==3:self.correction=False  # method 4 is fitted without object-only offset
            elif key==CORRECTION_KEY:self.correction=not self.correction
    def current(self):
        with self.lock:return self.index,self.transforms[self.index].copy(),self.correction


class SelectedBuffer:
    def __init__(self,buffer,selection,camera,after):
        self.raw=buffer;self.selection=selection;self.camera=camera;self.after=after
    def __getattr__(self,name):return getattr(self.raw,name)
    def snapshot(self,**limits):
        q,v,meta=self.raw.snapshot(**limits)
        published=np.eye(4);published[:3,3]=q[16:19]
        published[:3,:3]=Rotation.from_quat(q[[20,21,22,19]]).as_matrix()
        index,T,correction=self.selection.current()
        world=object_world_pose(published,self.camera['camera_to_published'],T,self.after,
                               self.camera['object_translation_adjust_camera_m'],correction)
        q=q.copy();q[16:19]=world[:3,3];q[19:23]=matToQuat(world[:3,:3])
        # No velocity is inferred across a calibration toggle. This process
        # displays measured positions only and never advances dynamics.
        v=v.copy();v[16:]=0
        return q,v,dict(meta,calibration_index=index+1,calibration_name=self.selection.modes[index]['name'],object_correction=correction)


def build_model(config,run):
    import xml.etree.ElementTree as ET
    from stationary_rgbd import frozen_geometry
    from workspace import verify_model
    manifest=verify_model();cache=local('cache/live_viewer');cache.mkdir(exist_ok=True)
    camera_path=local(config['camera_config']);camera=load_camera_config(camera_path,require_extrinsics=True)
    before,_,_=load_pose_transform(camera_path)
    mesh=Path(config['mesh']).expanduser().resolve()
    # Reuse the hardware experiment's original mesh and materials read-only.
    # Only generated scene/cache files are written to this workspace.
    signature=dict(model=manifest['binary_sha256'],camera=digest(camera_path),mesh=digest(mesh),
        materials={p.name:digest(p) for p in mesh.parent.iterdir() if p.suffix.lower() in ['.mtl','.jpg','.jpeg','.png']},
        mesh_scale=config['mesh_scale'],scene_version=2)
    binary=cache/'scene.mjb';stamp=cache/'signature.json'
    if binary.exists() and stamp.exists() and read_json(stamp)==signature:
        return mujoco.MjModel.from_binary_path(str(binary))
    print('Compiling isolated hand + tracked cube scene...',flush=True)
    tree=ET.parse(local('model/hand.xml'));root=tree.getroot()
    compiler=root.find('compiler');compiler.set('meshdir',str(local('model/assets')));compiler.set('texturedir',str(local('model/assets')))
    world=root.find('worldbody');ET.SubElement(world,'light',pos='0 0 1')
    body=ET.SubElement(world,'body',name='obj',pos='0.02 0.035 0.085');ET.SubElement(body,'freejoint',name='obj_joint')
    ET.SubElement(body,'geom',name='obj_visual',type='mesh',mass='0.07',contype='0',conaffinity='0',rgba='1 1 1 1')
    path=cache/'scene.xml';tree.write(path,encoding='unicode')
    set_object_visual_mesh(path,mesh,config['mesh_scale']);add_calibrated_camera(path,camera,before)
    m=mujoco.MjModel.from_xml_path(str(path));mujoco.mj_saveModel(m,str(binary));write_json(stamp,signature)
    if m.geom_type[m.geom('obj_visual').id]!=mujoco.mjtGeom.mjGEOM_MESH:
        raise RuntimeError('Tracked object must render as a mesh, not a textured primitive')
    return m


def view(config_path,run_dir):
    cfg=read_json(config_path);run=local(run_dir);run.mkdir(exist_ok=True,parents=True)
    selection=Selection(cfg);camera=load_camera_config(local(cfg['camera_config']),require_extrinsics=True)
    _,after,_=load_pose_transform(local(cfg['camera_config']))
    model=build_model(cfg,run);data=mujoco.MjData(model)
    object_geoms=np.flatnonzero(model.geom_bodyid==model.body('obj').id)
    robot_geoms=np.flatnonzero((model.geom_bodyid!=0)&(model.geom_bodyid!=model.body('obj').id))
    oo=model.geom_rgba[object_geoms,3].copy();ro=model.geom_rgba[robot_geoms,3].copy()
    model.geom_rgba[np.r_[object_geoms,robot_geoms],3]=0
    feedback=LCMFeedback(model,cfg['lcm_url'],cfg['state_channel'],cfg['pose_channel'])
    buffer=SelectedBuffer(feedback.buffer,selection,camera,after);stop=threading.Event();threads=[]
    def key(code):selection.key(code)
    try:
        mujoco.mj_forward(model,data);existing=set(threading.enumerate())
        with mujoco.viewer.launch_passive(model,data,key_callback=key,show_left_ui=False,show_right_ui=False) as viewer:
            threads=[t for t in threading.enumerate() if t not in existing]
            with viewer.lock():
                initialize_interactive_camera(model,data,viewer.cam,[.02,.035,.085]);viewer.opt.frame=mujoco.mjtFrame.mjFRAME_WORLD
            print('COMPARISON VIEWER READY: F6 original | F7 camera+depth | F8 camera+depth+poses | F9 object correction'+
                  (' | F10 method 4' if len(selection.modes)>3 else ''),flush=True)
            while viewer.is_running() and not stop.is_set():
                with viewer.lock():
                    valid,status,meta=update_display(model,data,buffer,object_geoms,oo,
                        dict(max_robot_age=.3,max_pose_age=.5,max_skew=.3),robot_geoms,ro)
                index,_,correction=selection.current();mode=selection.modes[index]
                fourth=' | F10 Method 4' if len(selection.modes)>3 else ''
                overlay=(f'CALIBRATION {index+1}/{len(selection.modes)}: {mode["name"]}\nF6 Original | F7 Camera+depth | F8 Camera+depth+poses{fourth}\n'
                    f'F9: camera-axis object correction {"ON (-15 mm)" if correction else "OFF"}\n'
                    'READ ONLY: no motion / no torque enable\n'+status)
                viewer.set_texts([(None,None,overlay,None),(None,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    mode.get('note','Candidates are provisional.')+' Mouse orbit/pan/zoom remains interactive.\nSwitching changes this display only; tracker and production settings remain unchanged.',None)])
                viewer.sync()
                write_json(run/'mujoco_status.json',dict(displayed_at=time.time(),tracking_valid=valid,status=status,
                    calibration=index+1,calibration_name=mode['name'],object_correction=correction,qpos=data.qpos.tolist(),feedback=meta))
                time.sleep(1/60)
    except KeyboardInterrupt:pass
    finally:
        for t in threads:t.join(timeout=5)
        feedback.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/live_tracking.json');p.add_argument('--run-dir',required=True);a=p.parse_args();view(a.config,a.run_dir)
