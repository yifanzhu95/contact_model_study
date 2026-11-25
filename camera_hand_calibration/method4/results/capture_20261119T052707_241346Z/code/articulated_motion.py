#!/usr/bin/env python3
"""Authorized calibration sweep: bounded smooth targets and encoder/RGB-D logs.

No calibration parameters are changed. Read-only inspection and offline path
screening are separate from --run. Every generated artifact is workspace-local.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(ROOT/'cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(ROOT/'cache/xdg')
sys.path.insert(0, str(REPO))

import argparse
import json
import shutil
import signal
import time
from collections import deque

import mujoco
import numpy as np

from workspace import local, identifier, utc, write_json, read_json, digest, verify_model
from ContactModelStudy.Hardware.Hand import HardwareHand, check_device_feedback

API = Path('/home/chetan/leap_agentic_sysid/third_party/LEAP_Hand_Hardware')


class CalibrationHand(HardwareHand):
    """Space single-register checks; bulk encoder reads/target writes unchanged."""
    read_retries = 0
    def register(self,motor,address,size):
        for attempt in range(3):
            time.sleep(.002)
            try:return super().register(motor,address,size)
            except OSError as exc:
                if '-3001' not in str(exc) or attempt==2:raise
                self.read_retries+=1
                self.client.port_handler.is_using=False
                self.client.port_handler.clearPort()
                time.sleep(.005)


def model():
    manifest = verify_model()
    if digest(local('model/hand.mjb')) != manifest['binary_sha256']:
        raise ValueError('Frozen model binary changed')
    return mujoco.MjModel.from_binary_path(str(local('model/hand.mjb')))


def inspect_hand():
    cfg = read_json('configs/articulated_hardware.json')
    out = local('results')/('articulated_precheck_'+identifier());out.mkdir()
    hand = CalibrationHand(API, cfg)
    record = dict(created_at_utc=utc(), hardware_config=cfg, writes_performed=False)
    try:
        hand.connect_read_only()
        q,v,current = hand.read()
        check_device_feedback(q,v,current,cfg)
        record.update(q=q.tolist(),v=v.tolist(),current_raw=current.tolist(),
            torque_by_native_motor=[hand.register(i,64,1) for i in range(16)],
            temperature_by_native_motor=[hand.register(i,146,1) for i in range(16)],
            hardware_error_by_native_motor=[hand.register(i,70,1) for i in range(16)])
    finally:
        hand.close_read_only()
        write_json(out/'inspection.json',record)
    print(json.dumps(dict(directory=str(out.relative_to(ROOT)),**record),indent=2))
    return out


def minimum_jerk(start, end, fraction):
    s = np.clip(fraction,0.,1.)
    return np.asarray(start)+(np.asarray(end)-start)*(10*s**3-15*s**4+6*s**5)


def duration(start,end,speed=.18,acceleration=.35):
    distance=float(np.max(np.abs(np.asarray(end)-start)))
    return max(1.,1.875*distance/speed,np.sqrt(5.774*distance/acceleration))


def collision_pairs(m):
    exclusions=set(map(int,m.exclude_signature))
    result=[]
    geoms=[g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    for i,g in enumerate(geoms):
        a=int(m.geom_bodyid[g])
        for h in geoms[i+1:]:
            b=int(m.geom_bodyid[h])
            if a==b or m.body_parentid[a]==b or m.body_parentid[b]==a:continue
            if ((min(a,b)<<16)+max(a,b)) in exclusions:continue
            if not ((m.geom_contype[g]&m.geom_conaffinity[h]) or (m.geom_contype[h]&m.geom_conaffinity[g])):continue
            result.append((g,h))
    return result


def clearance(m,data,q,pairs,limit=.02):
    data.qpos[:]=q;mujoco.mj_fwdPosition(m,data)
    minimum=limit;nearest=None
    for g,h in pairs:
        value=mujoco.mj_geomDistance(m,data,g,h,limit,None)
        if value<minimum:
            minimum=float(value);nearest=(m.geom(g).name,m.geom(h).name)
    return minimum,nearest


def screen(m,waypoints,pairs,margin=.004):
    data=mujoco.MjData(m);worst=.1;nearest=None;worst_index=None
    for segment,(a,b) in enumerate(zip(waypoints[:-1],waypoints[1:])):
        count=max(3,int(np.ceil(np.max(abs(b-a))/.008)))
        for fraction in np.linspace(0,1,count):
            value,pair=clearance(m,data,minimum_jerk(a,b,fraction),pairs)
            if value<worst:worst,nearest,worst_index=value,pair,segment
    return dict(passed=worst>=margin,minimum_clearance_m=worst,nearest_pair=nearest,
                worst_segment=worst_index,required_margin_m=margin)


def acquire(plan_path,preview=True):
    import cv2
    import pyrealsense2 as rs
    from capture import Writer, intrinsics, frame_epoch, match_joints
    plan=read_json(plan_path);cfg=read_json('configs/articulated_hardware.json')
    experiment=read_json('configs/experiment.json');cam=experiment['camera']
    experiment['joints']=dict(experiment['joints'],max_speed_rad_s=.06,max_sample_gap_s=.1,max_motion_rad=.012)
    m=model();data=mujoco.MjData(m);pairs=collision_pairs(m)
    out=local('datasets')/('articulated_'+identifier());out.mkdir()
    writer=Writer(out);hand=CalibrationHand(API,cfg);pipeline=rs.pipeline();opened=False;enabled=False;original=[]
    stop=[False];old_handlers={}
    for sig in [signal.SIGINT,signal.SIGTERM]:
        old_handlers[sig]=signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    report=dict(status='starting',created_at_utc=utc(),plan=plan,hardware_config=cfg,
        model_manifest=read_json('model/manifest.json'),motor_motion_authorized=True,
        max_speed_rad_s=plan['max_speed_rad_s'],max_acceleration_rad_s2=plan['max_acceleration_rad_s2'],
        camera_intrinsics_modified=False,production_files_modified=False,
        frames_saved=0,waypoints=[],tag_detection=False)
    write_json(out/'session.json',report);write_json(out/'plan.json',plan)
    code=out/'code';code.mkdir()
    for name in ['articulated_motion.py','capture.py','workspace.py']:shutil.copy2(ROOT/name,code/name)
    recent=deque(maxlen=2000);saved=0;last_saved=-1.;last_health=0.;health_motor=0;health_index=0
    previous_frame=None;source_gaps=0;last_command=None;last_command_mono=None;keyframes=[]
    try:
        hand.connect();q,v,current=hand.read();check_device_feedback(q,v,current,cfg)
        path=[q.copy()]+[np.array(w['target']) for w in plan['waypoints']]
        checked=screen(m,path,pairs,plan['collision_margin_m'])
        if not checked['passed']:raise RuntimeError(f'Fresh measured-pose path failed collision screen: {checked}')
        report.update(initial_q=q.tolist(),fresh_screen=checked)
        selection=rs.config();selection.enable_device(cam['serial'])
        selection.enable_stream(rs.stream.color,cam['width'],cam['height'],rs.format.rgb8,cam['fps'])
        selection.enable_stream(rs.stream.depth,cam['width'],cam['height'],rs.format.z16,cam['fps'])
        profile=pipeline.start(selection);opened=True;device=profile.get_device()
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                original.append((sensor,sensor.get_option(rs.option.global_time_enabled)));sensor.set_option(rs.option.global_time_enabled,1)
        K=intrinsics(profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics())
        if not np.allclose(K['distortion_coefficients'],0):raise ValueError('Requires rectified color images')
        depth_scale=float(device.first_depth_sensor().get_depth_scale());align=rs.align(rs.stream.color)
        report['camera_profile']=dict(serial=cam['serial'],intrinsics=K,depth_scale_m=depth_scale,fps=cam['fps'])
        for _ in range(90):pipeline.wait_for_frames(5000)
        if stop[0]:raise KeyboardInterrupt('Cancelled before torque enable')
        # Preload the latest measurement, not the pre-camera-start sample.
        q,v,current=hand.read();check_device_feedback(q,v,current,cfg)
        checked=screen(m,[q.copy()]+[np.array(w['target']) for w in plan['waypoints']],pairs,plan['collision_margin_m'])
        if not checked['passed']:raise RuntimeError('Initial hand pose changed into an unsafe approach path')
        hand.enable(q);enabled=True;report.update(status='moving',initial_q=q.tolist(),fresh_screen=checked,
            joint_capture_criteria=experiment['joints']);write_json(out/'session.json',report)
        print(f'Authorized smooth sweep started; data {out.relative_to(ROOT)}. Q/Escape stops.',flush=True)
        def cycle(command,waypoint,phase):
            nonlocal saved,last_saved,last_health,health_motor,health_index,previous_frame,source_gaps,last_command,last_command_mono
            if stop[0]:raise KeyboardInterrupt('Operator stop')
            begin=time.time();qm,vm,cur=hand.read();end=time.time();now=time.monotonic()
            check_device_feedback(qm,vm,cur,cfg)
            if np.max(abs(vm))>1.2:raise RuntimeError('Measured joint speed exceeds calibration envelope')
            value,pair=clearance(m,data,qm,pairs)
            if value<plan['runtime_abort_margin_m']:raise RuntimeError(f'Measured self-collision clearance too small: {value}, {pair}')
            if min(data.site(name).xpos[2] for name in ['if_tip','mf_tip','rf_tip','th_tip'])<plan['minimum_tip_z_m']:
                raise RuntimeError('Fingertip below the screened workspace height')
            measured=dict(timestamp_epoch_s=(begin+end)/2,received_epoch_s=end,q=qm.tolist(),v=vm.tolist())
            recent.append(measured)
            if last_command is not None and np.max(abs(command-last_command))>.012:
                raise RuntimeError('Unexpected discontinuous target; stopping')
            hand.write(command);last_command=command.copy();last_command_mono=now
            writer.submit('joints',dict(**measured,current_raw=cur.tolist(),command=command.tolist(),phase=phase,pose_id=waypoint['id'],
                read_started_epoch_s=begin,read_finished_epoch_s=end,minimum_clearance_m=value,nearest_pair=pair))
            if now-last_health>.08:
                hand.health_register(health_motor,health_index);health_index=1-health_index
                if not health_index:health_motor=(health_motor+1)%16
                last_health=now
            raw=pipeline.wait_for_frames(2000);received=time.time();frames=align.process(raw)
            color,depth=frames.get_color_frame(),frames.get_depth_frame()
            if not color or not depth:raise RuntimeError('Missing camera frames')
            stamp,basis=frame_epoch(color.get_timestamp(),str(color.get_frame_timestamp_domain()),received)
            if basis!='sdk_epoch':raise RuntimeError('Camera global timestamps unavailable')
            number=int(color.get_frame_number());gaps=0 if previous_frame is None else max(0,number-previous_frame-1)
            previous_frame=number;source_gaps+=gaps
            rgb=np.asanyarray(color.get_data());dep=np.asanyarray(depth.get_data())
            if now-last_saved>=1/plan['save_hz']:
                match=match_joints(list(recent),stamp,basis,experiment['joints'])
                meta=dict(index=saved,pose_id=waypoint['id'],split=waypoint['split'],phase=phase,
                    active_finger=waypoint.get('finger'),body=waypoint.get('body'),
                    timestamp_epoch_s=stamp,time_basis=basis,received_epoch_s=received,
                    camera_intrinsics=K['K'],distortion_coefficients=K['distortion_coefficients'],
                    camera_serial=cam['serial'],depth_scale_m=depth_scale,
                    color_timestamp_ms=float(color.get_timestamp()),depth_timestamp_ms=float(depth.get_timestamp()),
                    color_frame_number=number,depth_frame_number=int(depth.get_frame_number()),
                    source_frames_skipped=gaps,joints=match)
                writer.submit('frame',(saved,rgb.copy(),dep.copy(),meta))
                if phase=='capture' and match['usable']:keyframes.append(dict(index=saved,pose_id=waypoint['id'],split=waypoint['split'],finger=waypoint.get('finger'),body=waypoint.get('body')))
                saved+=1;last_saved=now
            if preview:
                image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                cv2.putText(image,f"{waypoint['id']} | {phase} | Q/Esc stop",(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
                cv2.imshow('Independent camera / encoder calibration sweep',image)
                if cv2.waitKey(1)&255 in [27,ord('q')]:stop[0]=True
            return qm,vm
        last=q.copy()
        for waypoint in plan['waypoints']:
            target=np.array(waypoint['target']);T=duration(last,target,plan['max_speed_rad_s'],plan['max_acceleration_rad_s2'])
            started=time.monotonic()
            while time.monotonic()-started<T:
                cycle(minimum_jerk(last,target,(time.monotonic()-started)/T),waypoint,'move')
            settle_start=time.monotonic();stable_since=None
            while True:
                qm,vm=cycle(target,waypoint,'settle')
                now=time.monotonic();window=[r for r in recent if r['timestamp_epoch_s']>time.time()-.5]
                stable=(len(window)>8 and np.max(np.ptp([r['q'] for r in window],axis=0))<plan['settle_motion_rad']
                    and np.max(np.abs(np.median([r['v'] for r in window],axis=0)))<plan['settle_max_speed_rad_s'])
                stable_since=now if stable and stable_since is None else stable_since if stable else None
                if stable_since is not None and now-stable_since>plan['settle_s']:break
                if now-settle_start>plan['settle_timeout_s']:raise RuntimeError(f"Hand failed to settle at {waypoint['id']}")
            capture_start=time.monotonic();first_index=len(keyframes)
            while time.monotonic()-capture_start<plan['static_capture_s']:qm,vm=cycle(target,waypoint,'capture')
            entry=dict(id=waypoint['id'],split=waypoint['split'],finger=waypoint.get('finger'),body=waypoint.get('body'),
                measured_q=qm.tolist(),command=target.tolist(),usable_capture_frames=len(keyframes)-first_index)
            report['waypoints'].append(entry);write_json(out/'session.json',report)
            print(f"{waypoint['id']}: {entry['usable_capture_frames']} settled RGB-D frames",flush=True)
            last=target.copy()
        report['status']='complete'
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        # Hand cleanup precedes camera/disk cleanup. No unplanned recovery move.
        try:
            hand.close();report['torque_disabled_verified']=True
        except Exception as exc:report['torque_cleanup_error']=str(exc)
        for sensor,value in original:
            try:sensor.set_option(rs.option.global_time_enabled,value)
            except RuntimeError:pass
        if opened:pipeline.stop()
        if preview:cv2.destroyAllWindows()
        try:writer.close()
        finally:
            for sig,handler in old_handlers.items():signal.signal(sig,handler)
            report.update(finished_at_utc=utc(),frames_saved=saved,writer_frames=writer.frames,source_frames_skipped=source_gaps)
            report.update(torque_ever_enabled=enabled,single_register_read_retries=hand.read_retries)
            write_json(out/'keyframes.json',keyframes);write_json(out/'session.json',report)
    print('Sweep complete; torque disabled. '+str(out.relative_to(ROOT)),flush=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inspect',action='store_true')
    p.add_argument('--run',action='store_true',help='AUTHORIZED hardware motion and RGB-D capture')
    p.add_argument('--plan',default='configs/articulated_sweep.json')
    p.add_argument('--no-preview',action='store_true')
    args=p.parse_args()
    if args.inspect:inspect_hand()
    elif args.run:acquire(args.plan,not args.no_preview)
    else:p.error('Choose --inspect or --run')


if __name__=='__main__':main()
