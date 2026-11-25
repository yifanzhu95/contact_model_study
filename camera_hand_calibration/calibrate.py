#!/usr/bin/env python3
"""Isolated, marker-free camera/hand calibration workspace CLI."""
import os
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']=str(BASE/'cache/matplotlib')
os.environ['XDG_CACHE_HOME']=str(BASE/'cache/xdg')

import argparse
import logging
import traceback
from workspace import ROOT,local,identifier,utc,write_json,read_json,bootstrap,environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    boot=commands.add_parser('bootstrap',help='freeze source model/assets/calibration; refuses to replace an existing snapshot')
    boot.add_argument('--model-source',type=Path)
    boot.add_argument('--camera-source',type=Path)
    boot.add_argument('--hardware-source',type=Path)
    commands.add_parser('probe',help='list RealSense cameras without opening any hand device')
    commands.add_parser('status',help='regenerate workspace summary/index; no devices accessed')
    inspect=commands.add_parser('inspect',help='camera-only timing/profile/depth QC plots')
    inspect.add_argument('--session',required=True)
    cap=commands.add_parser('capture',help='record aligned RGB-D; joints are opt-in and passive LCM only')
    cap.add_argument('--config',default='configs/experiment.json')
    cap.add_argument('--frames',type=int,default=180)
    cap.add_argument('--joints-lcm',action='store_true',help='after operator authorization: passively subscribe to an already running joint-state publisher')
    cap.add_argument('--preview',action='store_true')
    cap.add_argument('--pose-id',default='camera_check')
    cap.add_argument('--split',choices=['unassigned','train','validation'],default='unassigned')
    catalog=commands.add_parser('catalog',help='sample actual visual CAD vertices as candidate natural landmarks')
    catalog.add_argument('--body',default='palm');catalog.add_argument('--count',type=int,default=16)
    ann=commands.add_parser('annotate',help='click matching natural features in captured RGB frames')
    ann.add_argument('--frame',required=True);ann.add_argument('--catalog',default='configs/landmarks_palm.json')
    ann.add_argument('--split',choices=['train','validation'],required=True)
    ann.add_argument('--output',default='datasets/observations.json')
    fit=commands.add_parser('fit',help='fit a camera-to-model-world transform from natural landmarks')
    fit.add_argument('--observations',default='datasets/observations.json')
    fit.add_argument('--initial',choices=['scratch','current'],default='scratch')
    fit.add_argument('--name')
    report=commands.add_parser('report',help='regenerate residual plots and overlays from a saved fit')
    report.add_argument('--result',required=True)
    overlay=commands.add_parser('overlay',help='project natural CAD landmarks onto a saved RGB frame; not a fitted calibration')
    overlay.add_argument('--frame',required=True);overlay.add_argument('--catalog',default='configs/landmarks_palm.json')
    overlay.add_argument('--transform',default='configs/current_transform.json')
    args=parser.parse_args()
    local('logs').mkdir(exist_ok=True)
    run_id=identifier();log=local('logs')/(run_id+'_'+args.command+'.log')
    logging.basicConfig(level=logging.INFO,handlers=[logging.StreamHandler(),logging.FileHandler(log)])
    record=dict(started_at_utc=utc(),command=args.command,argv=sys.argv[1:],environment=environment(),status='running')
    try:
        if args.command=='bootstrap':result=bootstrap(args.model_source,args.camera_source,args.hardware_source)
        elif args.command=='probe':
            from capture import devices
            result=dict(devices=devices(),hand_access=False)
        elif args.command=='capture':
            if args.frames<2:raise ValueError('--frames must be at least 2')
            from capture import record as capture
            result=capture(args.config,args.frames,args.joints_lcm,args.preview,args.pose_id,args.split)
        elif args.command=='inspect':
            from quality import inspect_session
            result=inspect_session(args.session)
        elif args.command=='status':
            from quality import workspace_status
            result=workspace_status()
        else:
            from study import catalog,annotate,fit,report,overlay
            if args.command=='catalog':result=catalog(args.body,args.count)
            elif args.command=='annotate':result=annotate(args.frame,args.catalog,args.split,args.output)
            elif args.command=='fit':result=fit(args.observations,args.initial,args.name)
            elif args.command=='overlay':result=overlay(args.frame,args.catalog,args.transform)
            else:result=report(args.result)
        record.update(status='complete',result=result)
        logging.info('%s',result)
        return 0
    except BaseException as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        logging.exception('Command failed')
        return 130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        record['finished_at_utc']=utc();write_json(log.with_suffix('.json'),record)


if __name__=='__main__':raise SystemExit(main())
