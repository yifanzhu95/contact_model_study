#!/usr/bin/env python3
"""Supervised method-4 capture: move empty hand, place, sample, remove, torque off.

Operator acknowledgements are per-run files. No automatic object-removal
inference. Tracking remains camera owner; raw camera poses/JPEGs are subscribed
from its existing ZMQ stream, so no object-only -15 mm offset enters the fit.
"""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent))
import argparse
import json
import signal
import time
from collections import deque
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from workspace import local,read_json,write_json,utc,digest
from articulated_motion import CalibrationHand,API,model,collision_pairs,clearance,screen,duration,minimum_jerk
from method4_reference import level_pose,palm_stopped_cube,refine_translation,cube_orientation_check,approach_path
from ContactModelStudy.Hardware.Hand import check_device_feedback
from ContactModelStudy.Hardware.LCM import RobotState,tracked_transform


def object_stability(samples):
    if len(samples)<2:return dict(position_std_m=float('inf'),orientation_rms_deg=float('inf'))
    poses=np.array([r['raw_camera_object_pose'] for r in samples]);rot=Rotation.from_matrix(poses[:,:3,:3]);mean=rot.mean()
    return dict(position_std_m=float(np.linalg.norm(np.std(poses[:,:3,3],axis=0))),
        orientation_rms_deg=float(np.degrees(np.sqrt(np.mean((mean.inv()*rot).magnitude()**2)))))


def port_owners(port):
    port=str(Path(port).resolve());owners=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:target=os.readlink(fd)
                except OSError:continue
                if target==port:owners.append(int(proc.name));break
        except (OSError,PermissionError):pass
    return owners


class Method4:
    def __init__(self,config_path,run_dir):
        import lcm
        import zmq
        self.zmq=zmq;self.config=read_json(config_path);self.run=local(run_dir);self.run.mkdir(parents=True,exist_ok=True)
        if (self.run/'session.json').exists():raise ValueError('Use a new capture directory; old acknowledgements cannot be reused')
        if not (self.run/'empty_hand_confirmed.trigger').exists():raise ValueError('Empty-hand confirmation is required before opening hardware')
        self.live=read_json('configs/live_tracking.json');self.hardware=read_json('configs/articulated_hardware.json')
        owners=port_owners(self.hardware['port'])
        if owners:raise RuntimeError(f'USB hand is already open in processes {owners}; stop the read-only source first')
        self.m=model();self.data=mujoco.MjData(self.m);self.pairs=collision_pairs(self.m);self.target=level_pose(self.m,self.config['inward_yaw_rad'])
        self.hand=CalibrationHand(API,self.hardware);self.lcm=lcm.LCM(self.live['lcm_url']);self.context=zmq.Context();self.sub=self.context.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE,b'');self.sub.setsockopt(zmq.RCVHWM,1);self.sub.setsockopt(zmq.LINGER,0);self.sub.connect(self.live['viewer_endpoint'])
        self.command=None;self.last_command=None;self.latest=None;self.recent=deque(maxlen=1000);self.samples=[];self.interrupts=0;self.may_have_object=False
        self.last_health=0.;self.health_index=0;self.last_status=0.;self.stage='initializing';self.enabled=False
        self.log=(self.run/'joints.jsonl').open('w',buffering=1)
        self.record=dict(status='initializing',created_at=utc(),config=self.config,torque_disabled_verified=False,object_only_correction_used=False,
            hardware_config=self.hardware,initial_transform_sha256=digest(local(self.config['initial_transform'])),
            initial_transform=read_json(self.config['initial_transform']),samples=[])
        write_json(self.run/'session.json',self.record)
        signal.signal(signal.SIGINT,self.interrupt);signal.signal(signal.SIGTERM,self.terminate)
    def interrupt(self,*_):self.interrupts+=1
    def terminate(self,*_):self.interrupts=2
    def status(self,stage,prompt):
        self.stage=stage;self.prompt=prompt
        write_json(self.run/'status.json',dict(stage=stage,prompt=prompt,updated_at=time.time(),torque_enabled=self.enabled,object_may_be_present=self.may_have_object,samples=len(self.samples)))
        print(prompt,flush=True)
    def cycle(self):
        if self.interrupts>1:raise KeyboardInterrupt('Emergency stop')
        started=time.time();q,v,current=self.hand.read();finished=time.time();stamp=(started+finished)/2
        check_device_feedback(q,v,current,self.hardware)
        distance,pair=clearance(self.m,self.data,q,self.pairs)
        if distance<.002:raise RuntimeError('Measured collision envelope breached')
        if self.command is not None:
            if self.last_command is not None and np.max(abs(self.command-self.last_command))>.012:raise RuntimeError('Discontinuous calibration command')
            self.hand.write(self.command);self.last_command=self.command.copy()
        self.lcm.publish(self.live['state_channel'],RobotState(int(stamp*1e6),q,v).encode())
        self.recent.append(dict(t=stamp,q=q.copy(),v=v.copy()))
        self.log.write(json.dumps(dict(timestamp=stamp,q=q.tolist(),v=v.tolist(),command=None if self.command is None else self.command.tolist(),
            current_raw=current.tolist(),minimum_clearance_m=distance,stage=self.stage))+'\n')
        now=time.monotonic()
        if now-self.last_health>.08:
            self.hand.health_register((self.health_index//2)%16,self.health_index%2);self.health_index+=1;self.last_health=now
        while self.sub.poll(0):
            parts=self.sub.recv_multipart()
            if len(parts)!=3:continue
            try:
                pose=tracked_transform(np.frombuffer(parts[0],np.float32).reshape(4,4))
                if pose[2,3]<=0:continue  # placement/mask-review streams are not object measurements
                self.latest=dict(pose=pose,timestamp=float(np.frombuffer(parts[2],np.float64)[0]),jpeg=parts[1])
            except (ValueError,TypeError,IndexError):continue
        if time.monotonic()-self.last_status>.25:
            write_json(self.run/'status.json',dict(stage=self.stage,prompt=getattr(self,'prompt',''),updated_at=time.time(),
                torque_enabled=self.enabled,object_may_be_present=self.may_have_object,samples=len(self.samples),measured_q=q.tolist()))
            self.last_status=time.monotonic()
        time.sleep(max(0,1/60-(time.time()-started)))
        return q,v
    def hold(self,seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:self.cycle()
    def gate(self,name,stage,prompt):
        self.status(stage,prompt);trigger=self.run/(name+'.trigger')
        while not trigger.exists():
            if name!='object_removed' and self.interrupts:raise KeyboardInterrupt('Operator interrupted capture/placement')
            self.cycle()
    def move(self,start,target):
        T=duration(start,target,self.config['max_speed_rad_s'],self.config['max_acceleration_rad_s2']);begin=time.monotonic()
        while time.monotonic()-begin<T:
            if self.interrupts:raise KeyboardInterrupt('Interrupted while hand empty')
            self.command=minimum_jerk(start,target,(time.monotonic()-begin)/T);self.cycle()
        self.command=target.copy();self.hold(1.)
    def trim_empty(self):
        """Small MCP target bias only, to make measured supports level unloaded.

        Runtime gains are unchanged. The final target is held fixed throughout
        placement/capture, so manual contact cannot integrate a growing bias.
        """
        start=time.monotonic();indices=np.array([0,4,8]);velocity=np.zeros(3);quiet=None;last=time.monotonic()
        while time.monotonic()-start<25:
            q,v=self.cycle();now=time.monotonic();dt=min(.04,now-last);last=now
            if self.interrupts:raise KeyboardInterrupt('Interrupted while hand empty')
            desired=np.clip((self.target[indices]-q[indices])*.8,-.025,.025)
            velocity+=np.clip(desired-velocity,-.15*dt,.15*dt)
            proposed=self.command.copy();proposed[indices]+=velocity*dt
            proposed[indices]=np.clip(proposed[indices],self.target[indices]-.12,self.target[indices]+.12)
            distance,_=clearance(self.m,self.data,proposed,self.pairs)
            if distance<self.config['collision_margin_m']:raise RuntimeError('Trim target would violate collision margin')
            self.command=proposed
            _,geometry=palm_stopped_cube(self.m,q,self.config['cube_edge_m'])
            good=geometry['max_support_tilt_deg']<.5 and geometry['support_height_spread_mm']<.5 and max(abs(v))<.06
            quiet=now if good and quiet is None else quiet if good else None
            if quiet is not None and now-quiet>.8:return
        raise RuntimeError('Empty supports did not become level; no object placement requested')
    def capture(self):
        begin=time.monotonic();seen=set();last_rejection=None;window_samples=deque(maxlen=self.config['stable_frames']);index=0
        while time.monotonic()-begin<40:
            self.cycle()
            if self.interrupts:raise KeyboardInterrupt('Capture interrupted')
            frame=self.latest
            if frame is None or frame['timestamp'] in seen:continue
            seen.add(frame['timestamp']);age=time.time()-frame['timestamp']
            if not -.1<age<.4:continue
            closest=min(self.recent,key=lambda r:abs(r['t']-frame['timestamp']))
            if abs(closest['t']-frame['timestamp'])>.025:continue
            window=[r for r in self.recent if closest['t']-.4<r['t']<=closest['t']]
            if len(window)<10 or np.max(np.ptp([r['q'] for r in window],axis=0))>.008:continue
            try:center,geometry=palm_stopped_cube(self.m,closest['q'],self.config['cube_edge_m'],self.config['pad_surface_offset_m'])
            except ValueError as exc:last_rejection=str(exc);continue
            if geometry['max_support_tilt_deg']>self.config['max_pad_tilt_deg'] or geometry['support_height_spread_mm']>self.config['max_pad_height_spread_mm']:
                last_rejection=f"Loaded supports: tilt {geometry['max_support_tilt_deg']:.2f} deg, height spread {geometry['support_height_spread_mm']:.2f} mm";continue
            row=dict(timestamp=frame['timestamp'],received_at=time.time(),raw_camera_object_pose=frame['pose'].tolist(),measured_q=closest['q'].tolist(),joint_skew_ms=(closest['t']-frame['timestamp'])*1000,known_world_center_m=center.tolist(),reference_geometry=geometry)
            row['frame']=f'frame_{index:04d}.jpg';(self.run/row['frame']).write_bytes(frame['jpeg']);index+=1
            window_samples.append(row);self.samples=list(window_samples)
            with (self.run/'object_samples.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            stable=object_stability(self.samples)
            if len(self.samples)>=self.config['stable_frames'] and stable['position_std_m']<.0015 and stable['orientation_rms_deg']<2.:
                self.record['tracking_stability']=stable;return
        raise ValueError('Could not collect a complete stable cube-pose window. '+str(last_rejection))
    def execute(self):
        completed=False;error=None
        try:
            self.hand.connect();q,v,c=self.hand.read();check_device_feedback(q,v,c,self.hardware)
            waypoints,path=approach_path(self.m,q,self.target,self.config['collision_margin_m'])
            self.record['initial_measured_q']=q.tolist()
            self.record['fresh_path_screen']=path
            self.hand.enable(q);self.enabled=True;self.command=q.copy()
            self.status('moving_empty','Moving the empty hand gradually to straight, parallel, level fingers.')
            for target in waypoints:self.move(q,target);q=target.copy()
            self.status('leveling_empty','Settling measured proximal finger surfaces to horizontal.');self.trim_empty();self.hold(1.)
            self.record['held_target']=self.command.tolist();self.may_have_object=True
            self.gate('object_placed','awaiting_object_placement','READY: place the cube centered over the middle finger, bottom rear corners against the fixed palm housings. Clear your hands and confirm placed. Holding fingers fixed.')
            self.status('settling_loaded','Holding fixed while the cube and measured fingers settle.');self.hold(2.)
            self.status('capturing','Capturing fresh raw camera poses and measured loaded joints.');self.capture();completed=True
        except (ValueError,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}';self.record['capture_error']=error
        finally:
            try:
                if self.enabled and self.may_have_object and self.interrupts<2:
                    self.interrupts=0
                    self.gate('object_removed','awaiting_object_removal',('CAPTURE COMPLETE. ' if completed else 'CAPTURE STOPPED. ')+
                        'Remove the cube completely, then confirm removed. Torque remains enabled until confirmation.')
                    self.may_have_object=False
            finally:
                try:self.hand.close();self.record['torque_disabled_verified']=True;self.enabled=False
                except Exception as exc:self.record['torque_cleanup_error']=str(exc)
                self.sub.close();self.context.term();self.log.close()
                self.record.update(status='captured_torque_off' if completed else 'failed',finished_at=utc(),samples=self.samples)
                write_json(self.run/'session.json',self.record)
        if not completed:raise RuntimeError(error or 'Capture failed')
        if not self.record['torque_disabled_verified']:raise RuntimeError('Torque disable was not verified; refusing refinement')
        self.status('refining','Cube removed and torque verified off. Refining method 4 now.')
        import trimesh
        mesh_center=trimesh.load(self.live['mesh'],process=False).bounds.mean(0)
        poses=np.array([r['raw_camera_object_pose'] for r in self.samples]);known=np.array([r['known_world_center_m'] for r in self.samples]);initial=np.array(self.record['initial_transform']['T_world_from_camera'])
        candidate,stats=refine_translation(initial,poses,known,mesh_center,self.config['max_translation_change_m'],self.config['min_stable_frames'])
        orientation=cube_orientation_check(initial,poses)
        write_json(self.run/'candidate.json',dict(T_world_from_camera=candidate.tolist(),method=4,parent_method=3,
            status='Single known-placement translation anchor; independent validation required',translation_fit=stats,rotation_verification=orientation,
            object_only_offset_applied=False,torque_disabled_verified=True,mesh_center_object_m=mesh_center.tolist(),production_configuration_modified=False))
        self.record['status']='complete';write_json(self.run/'session.json',self.record);write_json('method4/latest_capture.json',dict(directory=str(self.run.relative_to(ROOT))))
        self.status('complete','Method 4 saved; production calibration unchanged; torque is off.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='method4/config.json');p.add_argument('--run-dir',required=True);p.add_argument('--run',action='store_true',required=True);a=p.parse_args();Method4(a.config,a.run_dir).execute()
