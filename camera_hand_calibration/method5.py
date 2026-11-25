#!/usr/bin/env python3
"""Start the supervised preparation hold, fresh normal tracking and editor."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent))
import argparse
import signal
import subprocess
import time
import shutil
from workspace import local,read_json,write_json,identifier,digest,utc
from ContactModelStudy.Hardware.TrackingSession import process_identity,process_alive


def release_owned_reader():
    path=local('runtime/live_tracking_latest.json')
    if not path.exists():return
    previous=read_json(path);identity=process_identity(previous['pid'])
    if not identity:return
    args=(Path('/proc')/str(previous['pid'])/'cmdline').read_bytes().decode().split('\0')
    if not any(Path(a).name=='live_tracking.py' for a in args):raise RuntimeError('Saved reader launcher identity does not match')
    os.kill(identity['pid'],signal.SIGINT);end=time.monotonic()+15
    while process_alive(identity) and time.monotonic()<end:time.sleep(.1)
    if process_alive(identity):raise RuntimeError('Read-only source has not released USB')


def launch(args):
    if not args.empty_hand_confirmed:raise ValueError('Start with the cube removed; supply --empty-hand-confirmed after checking the hand')
    cfg=read_json(args.config);tracking=read_json(cfg['tracking_config']);release_owned_reader()
    existing=local('method5/active.json')
    if existing.exists() and process_alive(read_json(existing).get('hold_process')):raise RuntimeError('A method-5 hold is already active')
    run=local('method5/results/session_'+identifier());run.mkdir(parents=True)
    write_json(run/'empty_hand_confirmed.trigger',dict(confirmed_at=time.time(),source='Operator setup confirms cube removed before preparation.'))
    shutil.copy2(local(args.config),run/'config.json');code=run/'code';code.mkdir()
    for name in ['method5.py','method5_hold.py','method5_viewer.py','method5_alignment.py','method5_zero.py','method4_capture.py','method4_reference.py','method4_retrack.py','articulated_motion.py']:
        shutil.copy2(ROOT/name,code/name)
    streams=[];children={};interrupts=[0]
    def start(name,argv):
        stream=(run/(name+'.log')).open('w');streams.append(stream)
        p=subprocess.Popen(argv,cwd=ROOT.parent,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children[name]=p;print(f'Started {name}, pid {p.pid}; {run/(name+".log")}',flush=True);return p
    def signal_handler(sig,_):
        interrupts[0]+=1;write_json(run/'shutdown.request',dict(requested_at=time.time(),reason=signal.Signals(sig).name))
        if interrupts[0]>1 or sig==signal.SIGTERM:
            if 'hold' in children and children['hold'].poll() is None:children['hold'].terminate()
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,signal_handler)
    hold=start('hold',[sys.executable,'-u',str(ROOT/'method5_hold.py'),'--config',args.config,'--run-dir',str(run)])
    write_json('method5/active.json',dict(directory=str(run.relative_to(ROOT)),launcher=process_identity(os.getpid()),hold_process=process_identity(hold.pid)))
    write_json(run/'launch.json',dict(created_at=utc(),config=cfg,hardware=read_json(cfg['hardware_config']),
        hardware_sha256=digest(local(cfg['hardware_config'])),initial_transform=read_json(cfg['initial_transform']),
        tracking_configuration=tracking,bbox=tracking['bbox_xywh'],production_configuration_modified=False))
    try:
        while not (run/'preparation_ready.trigger').exists():
            if hold.poll() is not None:raise RuntimeError('Preparation hold failed; see hold.log')
            if interrupts[0]:break
            time.sleep(.1)
        if not interrupts[0]:
            print('Preparation pose held. Opening the hardware experiment mask preview with its original bounding box.',flush=True)
            vision=start('vision',[sys.executable,'-u',str(ROOT/'method4_retrack.py'),'--prefix','method5','--config',cfg['tracking_config'],'--bbox',*map(str,tracking['bbox_xywh'])])
            while vision.poll() is None:
                if hold.poll() is not None:vision.terminate();raise RuntimeError('Hold exited during vision setup')
                if interrupts[0]:vision.terminate();break
                time.sleep(.1)
            if vision.returncode not in (0,None) and not interrupts[0]:
                write_json(run/'shutdown.request',dict(requested_at=time.time(),reason='vision setup failed; awaiting object removal'))
                print('Vision setup failed. Holding until object removal is confirmed.',flush=True)
            elif not interrupts[0]:
                editor=start('editor',[sys.executable,'-u',str(ROOT/'method5_viewer.py'),'--config',args.config,'--run-dir',str(run)])
                print('Place the cube on the hand. NumPad3 seeds the editable reference; NumPadEnter fits and keeps torque/tracking ON.',flush=True)
                record=read_json('method5/active.json');record['editor_process']=process_identity(editor.pid);write_json('method5/active.json',record)
        while hold.poll() is None:
            if 'editor' in children and children['editor'].poll() is not None and not (run/'shutdown.request').exists():
                write_json(run/'shutdown.request',dict(requested_at=time.time(),reason='editor exited; remove cube before torque off'))
            time.sleep(.1)
        print('Hardware hold exited. Inspect session.json for verified torque-off status. Tracking is preserved.',flush=True)
    finally:
        # Never terminate a healthy holding hand merely because vision/editor
        # failed. It remains supervised through the explicit removal gate.
        if hold.poll() is None and not (run/'shutdown.request').exists():
            write_json(run/'shutdown.request',dict(requested_at=time.time(),reason='launcher exited; awaiting object removal'))
        for name,p in children.items():
            if name=='editor' and p.poll() is None:p.terminate()
        for stream in streams:stream.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='method5/config.json');p.add_argument('--empty-hand-confirmed',action='store_true');p.add_argument('--run-dir');p.add_argument('--request-stop',action='store_true');p.add_argument('--finish',action='store_true',help='close editor, wait for cube removal, return to zero, disable torque');p.add_argument('--object-removed',action='store_true');a=p.parse_args()
    if a.request_stop or a.object_removed or a.finish:
        run=local(a.run_dir or read_json('method5/active.json')['directory'])
        if a.finish:
            session=read_json(run/'session.json')
            if not session.get('capabilities',{}).get('return_to_zero_on_finish'):
                p.error('This active hold predates zero-return support. It must be stopped before using the updated hold on the next launch.')
            write_json(run/'shutdown.request',dict(requested_at=time.time(),return_to_zero=True,reason='Operator requested finish at zero'))
            active=read_json('method5/active.json');editor=active.get('editor_process')
            if local(active['directory'])==run and process_alive(editor):os.kill(editor['pid'],signal.SIGTERM)
            print('Editor closed. Remove cube and run: python camera_hand_calibration/method5.py --object-removed',flush=True)
        if a.request_stop and not a.finish:
            write_json(run/'shutdown.request',dict(requested_at=time.time(),reason='Operator requested direct torque-off after cube removal'))
        if a.object_removed:
            write_json(run/'object_removed.trigger',dict(confirmed_at=time.time(),source='Operator CLI acknowledgement'))
    else:launch(a)
