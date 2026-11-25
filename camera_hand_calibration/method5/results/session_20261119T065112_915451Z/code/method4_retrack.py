#!/usr/bin/env python3
"""Reacquire vision while method 4 independently holds the hand.

This process never opens USB or launches a joint publisher/controller. Only
the owned vision session is restarted. A per-capture ROI does not edit the
hardware experiment's default bounding box.
"""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent))
import argparse
import signal
import time
from workspace import local,read_json,write_json,identifier


def main(args):
    from launch_leap_experiment import (Experiment,build_commands,prepare_tracking,tracking_mesh,controller_selection)
    from ContactModelStudy.Hardware.TrackingSession import TrackingSession,tracking_spec,VISION_SERVICES
    from ContactModelStudy.Hardware.GPU import select_gpu
    cfg=read_json(getattr(args,'config','configs/live_tracking.json'));cfg['bbox_xywh']=args.bbox
    prefix=getattr(args,'prefix','method4')
    if prefix not in ('method4','method5'):raise ValueError('Unknown isolated vision workspace')
    directory=local(prefix+'/results/retrack_'+identifier());directory.mkdir(parents=True)
    session=TrackingSession(local('runtime/tracking'));session.acquire();experiment=None
    state={'stop':None,'force_stop':False}
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda sig,_:state.__setitem__('stop',signal.Signals(sig).name))
    try:
        saved=session.load();preferred=saved['spec'].get('cuda_visible_devices') if saved else None
        gpu=select_gpu('auto',preferred_uuid=preferred)
        if saved:session.stop(10)
        cfg.update(tracking_gpu_assignment=gpu,gpu_assignment=gpu,init_mask_path=str(directory/'mask.png'))
        commands={name:argv for name,argv in build_commands(cfg,directory).items() if name in VISION_SERVICES}
        write_json(directory/'launch.json',dict(config=cfg,commands=commands,hand_access=False,default_hardware_bbox_unchanged=True))
        write_json(prefix+'/retrack_active.json',dict(directory=str(directory.relative_to(ROOT)),pid=os.getpid()))
        experiment=Experiment(directory,tracking_gpu_assignment=gpu,gpu_assignment=gpu);experiment.tracking_session=session
        mesh,scale=tracking_mesh(cfg,controller_selection(cfg['controller_args']));experiment.tracking_spec=tracking_spec(cfg,mesh,scale)
        result=prepare_tracking(experiment,commands,cfg,state)
        if result is not None:raise RuntimeError(str(result))
        print('Tracking reacquired. Hand controller was untouched.',flush=True)
        # Vision children persist under their session registry. No additional
        # control loop or USB source is left running by this helper.
    finally:
        if experiment is not None:experiment.shutdown(10)
        session.release()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bbox',type=int,nargs=4,required=True,metavar=('X','Y','W','H'));p.add_argument('--config',default='configs/live_tracking.json');p.add_argument('--prefix',choices=['method4','method5'],default='method4');a=p.parse_args();main(a)
