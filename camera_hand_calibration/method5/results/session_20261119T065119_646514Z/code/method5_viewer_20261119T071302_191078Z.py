#!/usr/bin/env python3
"""Dedicated MuJoCo/numpad reference editor; no motor command interface."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
sys.path.insert(0,str(ROOT.parent))
import argparse
import copy
import json
import shutil
import time
import xml.etree.ElementTree as ET
from collections import deque
import glfw
import mujoco
import numpy as np
import zmq
from scipy.spatial.transform import Rotation
from workspace import local,read_json,write_json,digest,identifier,utc
from method5_alignment import ReferenceEditor,fit_calibration
from method4_reference import recover_raw_camera_pose
from compare_tracking import build_model,SelectedBuffer
from view_leap_tracking import initialize_interactive_camera,update_display
from ContactModelStudy.Hardware.Camera import load_camera_config
from ContactModelStudy.Hardware.LCM import LCMFeedback,load_pose_transform,tracked_transform
from ContactModelStudy.Utils.Quaternions import matToQuat


class ActiveCalibration:
    def __init__(self,T):
        self.initial=T.copy();self.fitted=None;self.use_fit=True;self.modes=[dict(name='Method 3 baseline')]
    def current(self):return 0,(self.fitted if self.fitted is not None and self.use_fit else self.initial).copy(),False


def editor_model(live,run):
    build_model(live,run)
    path=local('cache/live_viewer/scene.xml');tree=ET.parse(path);root=tree.getroot()
    body=ET.SubElement(root.find('worldbody'),'body',name='method5_reference',mocap='true')
    geom=copy.deepcopy(root.find(".//geom[@name='obj_visual']"));geom.set('name','method5_reference_visual')
    for key in ['mass','density']:geom.attrib.pop(key,None)
    geom.set('contype','0');geom.set('conaffinity','0');body.append(geom)
    out=run/'editor_scene.xml';tree.write(out,encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(out))


def move_camera(model,action,dx,dy,cam):
    """MuJoCo 3.14's camera API takes no scene argument."""
    mujoco.mjv_moveCamera(model,action,dx,dy,cam)


def view(config_path,run_dir,resume_held_pose=False):
    cfg=read_json(config_path);run=local(run_dir);live=read_json(cfg['tracking_config']);camera=load_camera_config(local(live['camera_config']),require_extrinsics=True)
    (run/'code').mkdir(exist_ok=True)
    for script in ['method5_viewer.py','method5_alignment.py']:
        old=run/'code'/script
        if old.exists() and digest(old)!=digest(ROOT/script):
            shutil.copy2(old,run/'code'/(Path(script).stem+'_'+identifier()+'.py'))
        shutil.copy2(ROOT/script,run/'code'/script)
    write_json(run/'editor_inputs.json',dict(configuration=cfg,tracking_configuration=live,
        camera_configuration=read_json(local(live['camera_config'])),
        source_sha256={script:digest(ROOT/script) for script in ['method5_viewer.py','method5_alignment.py']},created_at=utc()))
    _,after,_=load_pose_transform(local(live['camera_config']));initial=np.array(read_json(cfg['initial_transform'])['T_world_from_camera'])
    active=ActiveCalibration(initial);editor=ReferenceEditor(cfg);model=editor_model(live,run);data=mujoco.MjData(model)
    obj=model.body('obj').id;ref=model.body('method5_reference').id;refmid=int(model.body_mocapid[ref]);refgeom=model.geom('method5_reference_visual').id
    object_geoms=np.flatnonzero(model.geom_bodyid==obj);robot_geoms=np.flatnonzero((model.geom_bodyid!=0)&(model.geom_bodyid!=obj)&(model.geom_bodyid!=ref))
    model.geom_rgba[object_geoms,:3]=[.4,.8,1.];model.geom_rgba[object_geoms,3]=cfg['tracked_opacity'];opacity=model.geom_rgba[object_geoms,3].copy();robot_opacity=model.geom_rgba[robot_geoms,3].copy()
    model.geom_rgba[refgeom,:]=[1.,.85,.45,0.];model.geom_rgba[np.r_[robot_geoms,object_geoms],3]=0
    feedback=LCMFeedback(model,live['lcm_url'],live['state_channel'],live['pose_channel']);buffer=SelectedBuffer(feedback.buffer,active,camera,after)
    context=zmq.Context();sub=context.socket(zmq.SUB);sub.setsockopt(zmq.SUBSCRIBE,b'');sub.setsockopt(zmq.LINGER,0);sub.setsockopt(zmq.RCVHWM,1);sub.connect(live['viewer_endpoint'])
    history=deque(maxlen=300);latest_world=None;latest_jpeg=None;seed_timestamp=0.;fit_requested=False;message='Place cube on hand, then NumPad 3 copies its tracked pose into the reference.';last_status=0.;last_time=0.;fit_count=0
    fit_count=max([int(p.name.split('_')[-1]) for p in run.glob('fit_[0-9][0-9][0-9]')]+[0])
    resumed_at=time.time() if resume_held_pose else 0.
    if resume_held_pose:
        hold=read_json(run/'status.json')
        if not hold['torque_enabled'] or time.time()-hold['updated_at']>1. or (run/'object_removed.trigger').exists():
            raise RuntimeError('Cannot resume editor without a live holding hand and no removal confirmation')
        write_json(run/f'editor_resume_{identifier()}.json',dict(resumed_at=resumed_at,hold_status=hold,reason='Operator requested viewer reopen; existing physical hold and tracker preserved'))
    def shutdown_pending():
        path=run/'shutdown.request'
        return path.exists() and read_json(path).get('requested_at',float('inf'))>resumed_at
    if not glfw.init():raise RuntimeError('GLFW desktop initialization failed')
    window=glfw.create_window(1400,950,'Method 5 - reference cube editor (numpad)',None,None)
    if not window:raise RuntimeError('Could not open method-5 viewer')
    glfw.make_context_current(window);glfw.swap_interval(1)
    render=mujoco.MjrContext(model,mujoco.mjtFontScale.mjFONTSCALE_150);scene=mujoco.MjvScene(model,maxgeom=5000);opt=mujoco.MjvOption();pert=mujoco.MjvPerturb();cam=mujoco.MjvCamera()
    opt.frame=mujoco.mjtFrame.mjFRAME_WORLD
    # Start at method 3's camera pose, leaving mouse orbit/pan/zoom fully free.
    cid=model.camera('inspection_realsense').id;model.cam_pos[cid]=initial[:3,3];quat=np.empty(4);mujoco.mju_mat2Quat(quat,(initial[:3,:3]@np.diag([1,-1,-1])).ravel());model.cam_quat[cid]=quat
    mujoco.mj_forward(model,data);initialize_interactive_camera(model,data,cam,[.02,.035,.085])
    mouse=[0.,0.];buttons=[False,False,False]
    def mouse_button(win,button,action,mods):
        if button in [0,1,2]:buttons[button]=action==glfw.PRESS
        mouse[:]=glfw.get_cursor_pos(win)
    def cursor(win,x,y):
        dx,dy=x-mouse[0],y-mouse[1];mouse[:]=[x,y]
        if not any(buttons):return
        shift=glfw.get_key(win,glfw.KEY_LEFT_SHIFT)==glfw.PRESS or glfw.get_key(win,glfw.KEY_RIGHT_SHIFT)==glfw.PRESS
        action=(mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V) if buttons[1] else (
            (mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V) if buttons[0] else mujoco.mjtMouse.mjMOUSE_ZOOM)
        height=max(glfw.get_window_size(win)[1],1);move_camera(model,action,dx/height,dy/height,cam)
    def scroll(win,x,y):move_camera(model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*y,cam)
    def key(win,code,scancode,action,mods):
        nonlocal fit_requested,message,seed_timestamp
        if action not in [glfw.PRESS,glfw.REPEAT]:return
        if action==glfw.REPEAT and code not in ReferenceEditor.AXES:return
        with (run/'key_events.jsonl').open('a') as stream:stream.write(json.dumps(dict(time=time.time(),key=code,mods=mods))+'\n')
        if code==glfw.KEY_KP_3:
            if latest_world is None:message='Waiting for fresh tracked pose; keep cube visible.'
            else:
                editor.seed(latest_world);seed_timestamp=history[-1]['timestamp'] if history else time.time();message='Reference copied. Use numpad to align it with the real cube.'
        elif code==glfw.KEY_KP_ENTER:
            if shutdown_pending():message='Shutdown pending: remove cube, then NumPad / confirms removal.'
            else:fit_requested=True
        elif code==glfw.KEY_F11:active.use_fit=not active.use_fit
        elif code==glfw.KEY_F12:
            write_json(run/'shutdown.request',dict(requested_at=time.time()));message='Remove the cube. Then NumPad / confirms removal and disables torque.'
        elif code==glfw.KEY_KP_DIVIDE and shutdown_pending():
            write_json(run/'object_removed.trigger',dict(confirmed_at=time.time(),source='Operator pressed NumPad / after removal request.'));message='Removal confirmed; waiting for verified torque off.'
        else:editor.key(code)
    glfw.set_key_callback(window,key);glfw.set_mouse_button_callback(window,mouse_button);glfw.set_cursor_pos_callback(window,cursor);glfw.set_scroll_callback(window,scroll)
    print('METHOD 5 EDITOR READY. NumPad3 seed; NumPad5 translate/rotate; 4/6 X, 2/8 Y, 1/7 Z; 0 step; 9 undo; NumPadEnter fit. F12 shutdown then NumPad/ removal confirmation.',flush=True)
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events();now=time.time()
            while sub.poll(0):
                parts=sub.recv_multipart()
                try:
                    raw=tracked_transform(np.frombuffer(parts[0],np.float32).reshape(4,4));stamp=float(np.frombuffer(parts[2],np.float64)[0])
                    if raw[2,3]<=0 or not -.1<now-stamp<.5:continue
                    if not history or stamp>history[-1]['timestamp']:
                        history.append(dict(timestamp=stamp,pose=raw@after));latest_jpeg=parts[1]
                except (ValueError,IndexError):continue
            valid,status,meta=update_display(model,data,buffer,object_geoms,opacity,dict(max_robot_age=.3,max_pose_age=.5,max_skew=.3),robot_geoms,robot_opacity)
            latest_world=None
            if valid:
                latest_world=np.eye(4);latest_world[:3,3]=data.qpos[16:19];latest_world[:3,:3]=Rotation.from_quat(data.qpos[[20,21,22,19]]).as_matrix()
            if fit_requested:
                fit_requested=False
                try:
                    if editor.pose is None:raise ValueError('First press NumPad3 to initialize the reference on the hand')
                    if not valid:raise ValueError('Waiting for fresh hand/object feedback before fitting')
                    hold_status=read_json(run/'status.json')
                    allowed_stages=('holding_preparation','awaiting_object_removal') if resume_held_pose else ('holding_preparation',)
                    if (hold_status['stage'] not in allowed_stages or not hold_status['torque_enabled']
                            or now-hold_status['updated_at']>1.):
                        raise ValueError('Calibration fitting requires the live preparation hold with torque enabled')
                    samples=[r for r in history if r['timestamp']>=seed_timestamp and now-cfg['fit_window_s']<=r['timestamp']<=now]
                    if not samples or samples[-1]['timestamp']-samples[0]['timestamp']<cfg['min_sample_span_s']:raise ValueError('Wait for stable tracked frames after initializing the reference')
                    current=active.current()[1];T,metrics=fit_calibration(current,[r['pose'] for r in samples],editor.pose,cfg);fit_count+=1
                    out=run/f'fit_{fit_count:03d}';out.mkdir()
                    record=dict(method=5,T_world_from_camera=T.tolist(),reference_world_object_pose=editor.pose.tolist(),metrics=metrics,
                        parent_transform=current.tolist(),initial_method3_transform=initial.tolist(),object_only_offset_applied=False,
                        production_configuration_modified=False,torque_kept_enabled=True,hold_status=hold_status,resumed_held_pose=resume_held_pose,tracking_restarted=False,created_at=utc(),
                        camera_view=dict(lookat=cam.lookat.tolist(),distance=cam.distance,azimuth=cam.azimuth,elevation=cam.elevation))
                    write_json(out/'candidate.json',record);write_json(out/'observations.json',dict(samples=[dict(timestamp=r['timestamp'],camera_object_pose=r['pose'].tolist()) for r in samples]))
                    if latest_jpeg:(out/'camera_frame.jpg').write_bytes(latest_jpeg)
                    active.fitted=T;active.use_fit=True;write_json(run/'latest_fit.json',dict(directory=str(out.relative_to(ROOT)),**record))
                    write_json('method5/latest_fit.json',dict(directory=str(out.relative_to(ROOT))))
                    message=f"FIT {fit_count} applied locally: {metrics['after']['position_rms_mm']:.2f} mm, {metrics['after']['orientation_rms_deg']:.2f} deg RMS. Torque/tracking remain ON."
                    print(message,flush=True)
                except (ValueError,RuntimeError) as exc:message=str(exc);print('Fit not applied:',message,flush=True)
            if editor.pose is not None:
                data.mocap_pos[refmid]=editor.pose[:3,3];data.mocap_quat[refmid]=matToQuat(editor.pose[:3,:3]);model.geom_rgba[refgeom,3]=cfg['reference_opacity']
            mujoco.mj_forward(model,data)
            width,height=glfw.get_framebuffer_size(window);viewport=mujoco.MjrRect(0,0,width,height)
            mujoco.mjv_updateScene(model,data,opt,pert,cam,mujoco.mjtCatBit.mjCAT_ALL,scene);mujoco.mjr_render(viewport,scene,render)
            step=cfg['translation_steps_m'][editor.step_index]*1000 if editor.mode=='translate' else cfg['rotation_steps_deg'][editor.step_index]
            error_text='Reference not initialized'
            if valid and editor.pose is not None:
                pos=np.linalg.norm(latest_world[:3,3]-editor.pose[:3,3])*1000;angle=np.degrees(Rotation.from_matrix(editor.pose[:3,:3].T@latest_world[:3,:3]).magnitude());error_text=f'Live overlap error: {pos:.2f} mm | {angle:.2f} deg'
            calibration_label='Method 5 fitted' if active.fitted is not None and active.use_fit else 'Method 3 baseline'
            text=(f'METHOD 5 | cyan translucent=tracked | gold=manual reference\nCalibration: {calibration_label}\nMode: {editor.mode.upper()} | step: {step:g} {"mm" if editor.mode=="translate" else "deg"} | world axes\n'
                'NumPad 3: copy tracked pose   5: translation/rotation   0: step size   9: undo\n4/6: -/+X   2/8: -/+Y   1/7: -/+Z   NumPad Enter: FIT CALIBRATION\nF11: baseline/fitted   F12: shutdown request   NumPad /: confirm cube removed\n'
                +error_text+'\n'+message+'\n'+status)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,text,'',render)
            glfw.swap_buffers(window)
            if now-last_status>.1:
                write_json(run/'editor_status.json',dict(displayed_at=now,tracking_valid=valid,reference_initialized=editor.pose is not None,
                    reference_pose=None if editor.pose is None else editor.pose.tolist(),active_transform=active.current()[1].tolist(),fit_count=fit_count,
                    mode=editor.mode,step_index=editor.step_index,message=message,last_key_time=now,status=status));last_status=now
    finally:
        if not (run/'object_removed.trigger').exists():write_json(run/'shutdown.request',dict(requested_at=time.time(),reason='reference viewer closed; hold until object removal'))
        feedback.close();sub.close();context.term();render.free();glfw.destroy_window(window);glfw.terminate()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='method5/config.json');p.add_argument('--run-dir',required=True);p.add_argument('--resume-held-pose',action='store_true');a=p.parse_args();view(a.config,a.run_dir,a.resume_held_pose)
