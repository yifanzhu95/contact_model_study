"""Standalone LEAP recovery; no LCM, tracker, viewer or original process needed."""
import argparse
import fcntl
import json
import os
import signal
import time
from pathlib import Path
import mujoco
import numpy as np
from articulated_motion import (API,CalibrationHand,model,collision_pairs,clearance,
    screen,minimum_jerk,duration)
from method4_capture import port_owners
from method4_reference import approach_path
from workspace import ROOT,local,read_json,write_json,identifier,utc
from ContactModelStudy.Hardware.Hand import MOTORS,check_device_feedback
from ContactModelStudy.Hardware.TrackingSession import process_identity,process_alive


def release_port(port,grace=3.,kill_wait=2.,timeout=12.):
    """Terminate actual USB owners, escalating only if they retain the port.

    Rescan to catch replacements. Validate each recorded process identity before
    signalling, and never open a second serial client while an owner remains.
    """
    pending={};actions=[];deadline=time.monotonic()+timeout
    while True:
        owners=port_owners(port)
        if not owners:return actions
        now=time.monotonic()
        if now>=deadline:raise RuntimeError(f'USB is still owned by {owners} after termination attempts; refusing concurrent access')
        for pid in owners:
            identity=process_identity(pid)
            if identity is None:continue
            key=(pid,identity['start_ticks'],identity['boot_id'])
            action=pending.get(key)
            sig=None
            if action is None:
                sig=signal.SIGTERM;action=dict(identity=identity,terminate_at=now);pending[key]=action
            elif 'kill_at' not in action and now-action['terminate_at']>=grace:
                sig=signal.SIGKILL;action['kill_at']=now
            elif 'kill_at' in action and now-action['kill_at']>=kill_wait:
                raise RuntimeError(f'Process {pid} still holds USB after SIGKILL; check device/process state')
            if sig is None or not process_alive(identity):continue
            try:os.kill(pid,sig)
            except ProcessLookupError:continue
            except PermissionError as exc:raise RuntimeError(f'Permission denied terminating USB owner {pid}; stop it with the account that owns it') from exc
            print(f'USB owner {pid}: sent {signal.Signals(sig).name}.',flush=True)
            actions.append(dict(identity=identity,signal=signal.Signals(sig).name))
        time.sleep(.05)


def recovery_path(m,q,margin=.004):
    zero=np.zeros(16);pairs=collision_pairs(m)
    if np.any(zero<m.jnt_range[:16,0]) or np.any(zero>m.jnt_range[:16,1]):
        raise ValueError('Zero pose is outside the model joint limits')
    direct=screen(m,[q,zero],pairs,margin)
    if direct['passed']:return [zero],dict(direct=direct)
    points,checked=approach_path(m,q,zero,margin)
    return points,dict(direct=direct,via_clearance_pose=checked)


def recover(hand,m,config,run,record,*,speed=.3,acceleration=.35,hz=50.,settle_timeout=20.,off_only=False):
    """Take over an ownerless serial bus without first switching torque off.

    Runtime gains/current settings are preserved. The current measured pose
    is preloaded before torque enable, including when some motors are still on.
    Caller always performs verified torque cleanup, on success or failure.
    """
    hand.connect_read_only()
    for motor in MOTORS:
        for field,address,size in [('model_number',0,2),('mode',11,1)]:
            expected=int(np.broadcast_to(config['expected_registers'][field],(16,))[motor])
            if hand.register(motor,address,size)!=expected:raise RuntimeError(f'Motor {motor}: unexpected {field}; refusing motion')
    record['initial_torque']=[hand.register(i,64,1) for i in MOTORS]
    if off_only:
        record['status']='torque_off_requested';return
    hand.health();q,v,current=hand.read();check_device_feedback(q,v,current,config)
    max_velocity=config.get('manual_viewer_limits',{}).get('max_velocity',3.)
    if np.max(abs(v))>max_velocity:raise RuntimeError('Measured joints moving too fast for recovery motion')
    points,checked=recovery_path(m,q)
    record.update(initial_measured_q=q.tolist(),path=checked)
    write_json(run/'session.json',record)
    hand.enable(q);record['torque_enabled_for_return']=True
    pairs=collision_pairs(m);data=mujoco.MjData(m);last_health=0.;health_index=0
    log=(run/'joints.jsonl').open('w',buffering=1)
    def cycle(command):
        nonlocal last_health,health_index
        begin=time.monotonic();actual,velocity,current=hand.read();check_device_feedback(actual,velocity,current,config)
        distance,pair=clearance(m,data,actual,pairs)
        if distance<.002:raise RuntimeError(f'Measured collision clearance below 2 mm: {pair}')
        if np.max(abs(velocity))>max_velocity:raise RuntimeError('Measured velocity exceeded recovery limit')
        if np.max(abs(command-actual))>.35:raise RuntimeError('Measured joints are not following the recovery target within 0.35 rad')
        hand.write(command)
        if begin-last_health>.1:
            hand.health_register((health_index//2)%16,health_index%2);health_index+=1;last_health=begin
        log.write(json.dumps(dict(timestamp=time.time(),q=actual.tolist(),v=velocity.tolist(),target=command.tolist(),current_raw=current.tolist(),minimum_clearance_m=distance))+'\n')
        time.sleep(max(0,1/hz-(time.monotonic()-begin)))
        return actual,velocity
    try:
        start=q.copy()
        for index,target in enumerate(points):
            seconds=duration(start,target,speed,acceleration);begin=time.monotonic()
            print(f'Gradual recovery segment {index+1}/{len(points)}: {seconds:.1f} seconds.',flush=True)
            while time.monotonic()-begin<seconds:cycle(minimum_jerk(start,target,(time.monotonic()-begin)/seconds))
            cycle(target);start=target.copy()
        print('Zero commanded; waiting for the hand to settle before torque off.',flush=True)
        zero=np.zeros(16);end=time.monotonic()+settle_timeout;quiet=None
        tolerances=np.full(16,.08)
        while time.monotonic()<end:
            q,v=cycle(zero);now=time.monotonic()
            good=np.all(abs(q)<=tolerances) and np.max(abs(v))<=.08
            quiet=now if good and quiet is None else quiet if good else None
            if quiet is not None and now-quiet>=.5:
                record.update(status='zero_reached',zero_settled=True,final_measured_q=q.tolist());return
        record['final_measured_q']=q.tolist()
        raise RuntimeError(f'Zero was commanded but the hand did not settle within {settle_timeout:g} seconds')
    finally:log.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--empty-hand-confirmed',action='store_true',help='confirm the object is removed and hands are clear')
    parser.add_argument('--hardware-config',type=Path,default=ROOT.parent/'configs/leap_hardware.json')
    parser.add_argument('--port',help='optional USB serial port override')
    parser.add_argument('--speed',type=float,default=.3,help='maximum commanded joint speed in rad/s (default .3, maximum .5)')
    parser.add_argument('--torque-off-only',action='store_true',help='skip the zero move and only disable/verify torque')
    args=parser.parse_args(argv)
    if not args.empty_hand_confirmed:parser.error('Remove the object and clear the hand, then supply --empty-hand-confirmed')
    if not np.isfinite(args.speed) or not 0<args.speed<=.5:parser.error('--speed must be in (0, .5] rad/s')
    config=json.loads(args.hardware_config.expanduser().read_text())
    if args.port:config['port']=args.port
    try:terminated=release_port(config['port'])
    except RuntimeError as exc:parser.error(str(exc))
    lock=local('runtime/hand_recovery.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:parser.error('Another hand recovery utility is already running')
    try:terminated+=release_port(config['port'])
    except RuntimeError as exc:
        lock.close();parser.error(str(exc))
    run=local('results/hand_recovery_'+identifier());run.mkdir()
    record=dict(created_at=utc(),status='starting',hardware_config=config,gains_modified=False,
        max_command_speed_rad_s=args.speed,max_command_acceleration_rad_s2=.35,
        empty_hand_confirmed=True,zero_settled=False,torque_disabled_verified=False,usb_owner_termination=terminated)
    write_json(run/'session.json',record);print(f'Recovery log: {run}',flush=True)
    hand=None;previous={}
    def interrupt(sig,_):raise KeyboardInterrupt(signal.Signals(sig).name)
    for sig in (signal.SIGINT,signal.SIGTERM):previous[sig]=signal.signal(sig,interrupt)
    code=0
    try:
        hand=CalibrationHand(API,config)
        recover(hand,None if args.torque_off_only else model(),config,run,record,speed=args.speed,off_only=args.torque_off_only)
    except BaseException as exc:
        record.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}')
        print(f'Recovery stopped: {exc}. Attempting torque-off cleanup.',flush=True);code=1
    finally:
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        try:
            if hand is not None:
                connected=hand.client.is_connected
                hand.close()
                record['torque_disabled_verified']=bool(connected)
                if not connected:raise RuntimeError('USB was not connected; torque-off could not be verified')
            else:raise RuntimeError('No hardware connection was created; torque-off could not be verified')
        except Exception as exc:
            record['torque_cleanup_error']=str(exc);print(f'TORQUE-OFF NOT VERIFIED: {exc}',flush=True);code=1
        finally:
            record['finished_at']=utc();write_json(run/'session.json',record);lock.close()
            for sig,handler in previous.items():signal.signal(sig,handler)
    return code
