#!/usr/bin/env python3
"""Isolated vision launcher and strictly read-only hand-state publisher."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
sys.path.insert(0,str(ROOT.parent))
import argparse
import json
import signal
import time
from workspace import local,read_json,write_json,identifier


def joint_source(config_path):
    import lcm
    from articulated_motion import CalibrationHand,API
    from ContactModelStudy.Hardware.Hand import check_device_feedback
    from ContactModelStudy.Hardware.LCM import RobotState
    cfg=read_json(config_path);hardware=read_json('configs/articulated_hardware.json');hand=CalibrationHand(API,hardware)
    connection=lcm.LCM(cfg['lcm_url']);stopped=[False]
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    try:
        hand.connect_read_only()
        torque=[hand.register(i,64,1) for i in range(16)]
        print('Read-only joint source. Torque registers (unchanged):',torque,flush=True)
        while not stopped[0]:
            begin=time.time();q,v,current=hand.read();end=time.time();check_device_feedback(q,v,current,hardware)
            connection.publish(cfg['state_channel'],RobotState(int((begin+end)*.5e6),q,v).encode());time.sleep(.003)
    finally:hand.close_read_only()


def launch(args):
    from launch_leap_experiment import (Experiment,build_commands,prepare_tracking,tracking_mesh,controller_selection)
    from ContactModelStudy.Hardware.TrackingSession import TrackingSession,tracking_spec,process_alive
    from ContactModelStudy.Hardware.GPU import select_gpu
    cfg=read_json(args.config);session=TrackingSession(local('runtime/tracking'))
    session.acquire();experiment=None
    try:
        saved=session.load()
        if args.stop_tracking:
            session.stop(10);return
        if saved and not process_alive(saved['processes'].get('tracker')):
            session.stop(10);saved=None
        if saved and args.restart_tracking:session.stop(10);saved=None
        gpu=select_gpu(args.gpu,preferred_uuid=saved['spec'].get('cuda_visible_devices') if saved and args.gpu=='auto' else None)
        cfg['tracking_gpu_assignment']=gpu;cfg['gpu_assignment']=gpu
        print(f"Tracking/SAM GPU {gpu['index']}: {gpu['utilization_percent']}% load, {gpu['memory_used_mib']} MiB",flush=True)
        directory=local('results/live_tracking_'+identifier());directory.mkdir()
        cfg['init_mask_path']=str(directory/'mask.png');cfg['hardware_config']=str(local('configs/articulated_hardware.json'))
        commands=build_commands(cfg,directory)
        commands.pop('controller',None)
        commands['bridge']=[sys.executable,'-u',str(ROOT/'live_tracking.py'),'--config',args.config,'--joint-source']
        commands['mujoco']=[sys.executable,'-u',str(ROOT/'compare_tracking.py'),'--config',args.config,'--run-dir',str(directory)]
        write_json(directory/'launch.json',dict(config=cfg,commands=commands,read_only=True));write_json('runtime/live_tracking_latest.json',dict(directory=str(directory.relative_to(ROOT)),pid=os.getpid()))
        experiment=Experiment(directory,tracking_gpu_assignment=gpu,gpu_assignment=gpu)
        experiment.tracking_session=session
        mesh,scale=tracking_mesh(cfg,controller_selection(cfg['controller_args']))
        experiment.tracking_spec=tracking_spec(cfg,mesh,scale)
        state={'stop':None,'force_stop':False}
        for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda sig,_:state.__setitem__('stop',signal.Signals(sig).name))
        experiment.start('bridge',commands['bridge']);experiment.start('mujoco',commands['mujoco'])
        if saved:
            session.check_compatible(experiment.tracking_spec);experiment.reusing_tracking=experiment.keep_tracking=True
            experiment.tracking_directory=Path(saved['directory'])
            if not process_alive(saved['processes'].get('viewer')):
                experiment.start('viewer',saved['commands']['viewer']);session.replace_viewer(experiment.children['viewer'])
        else:
            result=prepare_tracking(experiment,commands,cfg,state)
            if result is not None:raise RuntimeError(str(result))
        print('Live calibration comparison running; no controller or torque commands. Tracking stays running when the comparison viewer closes.',flush=True)
        while not state['stop']:
            finished=experiment.finished()
            if finished:print('Comparison stopping:',finished,flush=True);break
            time.sleep(.1)
    finally:
        if experiment is not None:experiment.shutdown(10)
        session.release()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/live_tracking.json');p.add_argument('--gpu',default='auto');p.add_argument('--joint-source',action='store_true');p.add_argument('--restart-tracking',action='store_true');p.add_argument('--stop-tracking',action='store_true');a=p.parse_args()
    if a.joint_source:joint_source(a.config)
    else:launch(a)
