#!/usr/bin/env python3
"""Capture an object-specific joint-cost reference by manually posing each finger.

Keys in the MuJoCo window: 1=index, 2=middle, 3=ring, 4=thumb; Enter locks the
selected finger at its measured pose. Backspace releases only the selected
finger for correction. All fingers start torque-off. Once all four are locked,
a stable measured pose is saved and held until Q/Escape/Space/window-close.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import shutil
import signal
import subprocess
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from ContactModelStudy.Hardware.Hand import HardwareHand, check_device_feedback, vector
from ContactModelStudy.Hardware.JointReferences import OBJECTS, reference_path, save_reference
from ContactModelStudy.Hardware.Scene import JOINTS, ROOT

FINGERS = {name: list(range(start,start+4)) for name,start in
           [('index',0),('middle',4),('ring',8),('thumb',12)]}


class FingerCapture:
    """Only explicit lock/release operations change a finger's torque state."""
    def __init__(self, hand, model, obj, output=None, settle=.75, velocity=.1, spread=.02,
                 metadata=None):
        self.hand, self.model, self.obj, self.output = hand, model, obj, output
        self.settle, self.velocity, self.spread = settle, velocity, spread
        self.metadata = metadata or {}
        self.selected = None
        self.locked = []
        self.targets = np.zeros(16)
        self.history = deque()
        self.saved = False
        self.last_saved = None
        self.message = 'All torque OFF. Select a finger: 1 index, 2 middle, 3 ring, 4 thumb.'

    def select(self, finger):
        if finger not in FINGERS:
            raise ValueError(f'Unknown finger: {finger}')
        self.selected = finger
        self.message = (f'{finger.upper()} locked. Backspace releases it for adjustment.'
                        if finger in self.locked else
                        f'{finger.upper()} torque OFF: position it by hand, hold still, then Enter.')

    def capture(self, q, v):
        if self.selected is None:
            self.message = 'Select a finger first (1 / 2 / 3 / 4).'
            return False
        if self.selected in self.locked:
            self.message = f'{self.selected} is already locked; Backspace releases it.'
            return False
        q, v = vector(q,'capture q'), vector(v,'capture velocity')
        indices = FINGERS[self.selected]
        low, high = self.model.actuator_ctrlrange[indices].T
        outside = (q[indices] < low) | (q[indices] > high)
        if outside.any():
            bad = ', '.join(JOINTS[indices[i]] for i in np.flatnonzero(outside))
            self.message = f'Not locked: {bad} outside calibrated actuator range. Reposition and Enter.'
            return False
        if np.max(np.abs(v[indices])) > self.velocity:
            self.message = 'Not locked: selected finger is moving. Hold still and press Enter again.'
            return False
        # Motor groups are contiguous by finger in both orders. Within a group,
        # HardwareHand maps the measured MuJoCo pose before preloading targets.
        self.hand.enable_motors(q,indices)
        self.targets[indices] = q[indices]
        self.locked.append(self.selected)
        self.history.clear()
        self.saved = False
        self.message = f'{self.selected.upper()} locked at its measured position. Select another finger.'
        print(self.message, 'joints:',q[indices].tolist(),flush=True)
        return True

    def release(self):
        if self.selected in self.locked:
            self.hand.disable_motors(FINGERS[self.selected])
            self.locked.remove(self.selected)
            self.history.clear()
            self.saved = False
            self.message = f'{self.selected.upper()} torque OFF. Reposition and Enter to lock again.'
            print(self.message,flush=True)

    def hold(self):
        motors = sorted(j for finger in self.locked for j in FINGERS[finger])
        if motors:
            self.hand.write_motors(self.targets,motors)

    def observe(self, q, v, now):
        if len(self.locked) != 4 or self.saved:
            return
        if np.max(np.abs(v)) > self.velocity:
            self.history.clear()
            self.message = 'All fingers locked. Waiting for the measured hand pose to settle...'
            return
        self.history.append((now,np.asarray(q).copy()))
        while len(self.history)>1 and self.history[1][0] <= now-self.settle:
            self.history.popleft()
        if now-self.history[0][0] < self.settle:
            self.message = 'All fingers locked. Measuring a stable pose before saving...'
            return
        samples = np.array([sample for _,sample in self.history])
        if np.max(np.ptp(samples,axis=0)) > self.spread:
            self.message = 'All fingers locked. Pose is still changing; waiting to save...'
            return
        reference = np.mean(samples,axis=0)
        violations = [JOINTS[i] for i in range(16) if
                      not self.model.jnt_range[i,0] <= reference[i] <= self.model.jnt_range[i,1]]
        self.last_saved = save_reference(self.obj,reference,{
            **self.metadata,'hold_targets':self.targets.tolist(),'capture_order':list(self.locked),
            'sample_count':len(samples),'sample_span_s':now-self.history[0][0],
            'sample_std_rad':np.std(samples,axis=0).tolist(),
            'sample_range_rad':np.ptp(samples,axis=0).tolist(),
            'outside_planning_joint_ranges':violations,
            'reference_semantics':'mean measured pose after all four fingers are locked; w_joint only',
        },path=self.output)
        self.saved = True
        self.message = f'SAVED {self.last_saved}. Holding all fingers. Q/Escape releases and exits.'
        print(self.message,flush=True)
        print('Joint-cost reference (MuJoCo order, radians):',reference.tolist(),flush=True)
        if violations:
            print('Measured reference outside planning joint ranges:',', '.join(violations),flush=True)

    def status(self):
        return dict(selected=self.selected,locked=list(self.locked),saved=self.saved,
                    output=str(self.last_saved or self.output or reference_path(self.obj)),
                    message=self.message)


def write_status(path, value):
    if path is None:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(dict(pid=os.getpid(),updated_at=time.time(),**value),indent=2)+'\n')
    temporary.replace(path)


def ensure_port_free(port):
    path = Path(port).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'LEAP serial port is absent: {path}')
    if shutil.which('fuser'):
        result = subprocess.run(['fuser',str(path.resolve())],capture_output=True,text=True)
        if result.returncode == 0:
            raise RuntimeError(f'Serial port is in use by PID(s) {result.stdout.strip()}; stop its bridge/controller first')
        if result.returncode != 1:
            raise RuntimeError(f'Cannot check serial port ownership: {result.stderr.strip()}')


def run_capture(hand, model, config, session, stopped, rate=50, status_file=None):
    for i,name in enumerate(JOINTS):
        if model.joint(name).qposadr[0] != i or model.actuator(i).name != name+'_act':
            raise ValueError('Capture model must use the calibrated MuJoCo hand joint order')
    if model.nq != 16 or model.nv != 16 or model.nu != 16:
        raise ValueError('Use the calibrated hand-only model for reference capture')
    # Explicitly switch all motors OFF before inspecting/configuring the hand.
    # Standard bridge connect() keeps its original torque-off startup check.
    hand.connect(torque_off=True)
    q,v,current = hand.read()
    check_device_feedback(q,v,current,config)
    data = mujoco.MjData(model)
    data.qpos[:],data.qvel[:],data.ctrl[:] = q,v,q
    if data.act.size:
        data.act[:] = q
    mujoco.mj_forward(model,data)
    events = SimpleQueue()

    def key_pressed(key):
        if key in (32,27,256,ord('q'),ord('Q')):
            stopped.set()
        elif key in (13,257,335):
            events.put('capture')
        elif key in (8,259):
            events.put('release')
        elif key in (49,50,51,52):
            events.put(list(FINGERS)[key-49])

    with mujoco.viewer.launch_passive(model,data,key_callback=key_pressed,
                                      show_left_ui=False,show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation = .45,130,-35
            visible_groups = viewer.opt.geomgroup.copy()
        print('VIEWER READY: all 16 motors TORQUE OFF. No automatic pose movement.',flush=True)
        print('Click viewer. 1 index / 2 middle / 3 ring / 4 thumb; move finger by hand, Enter locks it.',flush=True)
        print('Backspace releases the selected locked finger. Q / Escape / Space exits with all torque OFF.',flush=True)
        last_health = last_status = 0.
        health_motor = 0
        while viewer.is_running() and not stopped.is_set():
            started = time.monotonic()
            q,v,current = hand.read()
            check_device_feedback(q,v,current,config)
            while not stopped.is_set():
                try:
                    event = events.get_nowait()
                except Empty:
                    break
                if event == 'capture':
                    # Capture from a fresh sample, not the previous render frame.
                    q,v,current = hand.read()
                    check_device_feedback(q,v,current,config)
                    if not stopped.is_set():
                        session.capture(q,v)
                elif event == 'release':
                    session.release()
                else:
                    session.select(event)
                with viewer.lock():
                    # Numeric keys also toggle geom groups in MuJoCo. Preserve
                    # the hand's original visibility when using them to select.
                    viewer.opt.geomgroup[:] = visible_groups
            if stopped.is_set():
                break
            session.hold()
            session.observe(q,v,started)
            if started-last_health >= 1/16:
                hand.health(health_motor)
                health_motor = (health_motor+1)%16
                last_health = started
            state = session.status()
            overlay = ('1 index / 2 middle / 3 ring / 4 thumb | Enter: lock selected\n'
                       'Backspace: release selected | Q/Escape/Space: release all and exit\n'
                       + ' | '.join(f'{name.upper()}: {"LOCKED" if name in session.locked else "FREE"}' for name in FINGERS)
                       + '\n'+session.message)
            viewer.set_texts((None,None,overlay,None))
            with viewer.lock():
                data.qpos[:],data.qvel[:],data.ctrl[:] = q,v,q
                if data.act.size:
                    data.act[:] = q
                mujoco.mj_forward(model,data)
            viewer.sync()
            if started-last_status >= .5:
                write_status(status_file,dict(stage='ready',**state))
                last_status = started
            time.sleep(max(0,1/rate-(time.monotonic()-started)))
    return session.last_saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',choices=OBJECTS,default='cube')
    parser.add_argument('--output',type=Path,help='default: configs/leap_joint_references/<object>.json')
    parser.add_argument('--hardware-api',type=Path,default=Path.home()/'leap_agentic_sysid/third_party/LEAP_Hand_Hardware')
    parser.add_argument('--hardware-config',type=Path,default=ROOT/'configs/leap_hardware.json')
    parser.add_argument('--model',type=Path,default=ROOT/'scenes/leap/hardware/calibrated_hand.xml')
    parser.add_argument('--port')
    parser.add_argument('--rate',type=float,default=50)
    parser.add_argument('--settle',type=float,default=.75,help='stable measurement window in seconds')
    parser.add_argument('--max-capture-velocity',type=float,default=.1,help='selected finger must be below this speed to lock, rad/s')
    parser.add_argument('--max-pose-spread',type=float,default=.02,help='maximum joint range during the stable window, radians')
    parser.add_argument('--status-file',type=Path,default=ROOT/'.runtime/leap_reference_capture/status.json')
    args = parser.parse_args(argv)
    for name in ['rate','settle','max_capture_velocity','max_pose_spread']:
        if not np.isfinite(getattr(args,name)) or getattr(args,name)<=0:
            parser.error(f'--{name.replace("_","-")} must be finite and positive')
    config = json.loads(args.hardware_config.read_text())
    ensure_port_free(args.port or config['port'])
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if 'palm_quat_wxyz' in config:
        model.body_quat[model.body('palm').id] = config['palm_quat_wxyz']
    model.opt.gravity[:] = config.get('gravity_mujoco',[0,0,-9.81])
    hand = HardwareHand(args.hardware_api,config,args.port)
    stopped = threading.Event()
    previous = {s:signal.signal(s,lambda *_:stopped.set()) for s in (signal.SIGINT,signal.SIGTERM)}
    session = FingerCapture(hand,model,args.object,args.output,args.settle,args.max_capture_velocity,args.max_pose_spread,
        metadata=dict(model_path=str(args.model.resolve()),model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                      hardware_config=config))
    failure = None
    try:
        write_status(args.status_file,dict(stage='connecting',**session.status()))
        run_capture(hand,model,config,session,stopped,args.rate,args.status_file)
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        print('Capture stopped:',failure,flush=True)
        raise
    finally:
        for s in previous:
            signal.signal(s,signal.SIG_IGN)
        try:
            hand.close()
            write_status(args.status_file,dict(stage='stopped',torque_disabled=True,error=failure,**session.status()))
        finally:
            for s,handler in previous.items():
                signal.signal(s,handler)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
