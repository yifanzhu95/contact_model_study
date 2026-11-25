#!/usr/bin/env python3
"""Compare settled LEAP targets against the same sequence at a recorded rate.

Default: validate/describe the experiment, without opening USB or enabling
torque. --hardware executes it through an isolated LCM bridge. --analyze DIR
only analyzes an existing recording. Fingertips are encoder-based forward
kinematics, not optical measurements. No dynamics simulation or MPPI is run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import mujoco
import numpy as np

from ContactModelStudy.Hardware.Episodes import HardwareEpisodes
from ContactModelStudy.Hardware.Hand import control_limits, vector
from ContactModelStudy.Hardware.LCM import LCMFeedback, RobotPositionAction, RobotTargetState
from ContactModelStudy.Hardware.Scene import JOINTS

ROOT=Path(__file__).resolve().parent
TIPS=('if_tip','mf_tip','rf_tip','th_tip')
MOVING_JOINTS=(2,6,10,14)
DEFAULT_RUN=ROOT/'results/diagnostics/m2_tracking_baseline/leap_experiment_20261116_215909_720053/controller.json'
DEFAULT_CONFIG=ROOT/'configs/m2_tracking_diagnostic/hardware.json'
DEFAULT_MODEL=ROOT/'scenes/leap/hardware/calibrated_hand.xml'


def measured_period(source):
    source=json.loads(Path(source).read_text())
    intervals=[]
    for episode in source['episodes']:
        stamps=[s['action_timestamp_us']*1e-6 for s in episode['steps'] if s.get('action_timestamp_us')]
        intervals.extend(np.diff(stamps))
    intervals=np.asarray(intervals)
    if not intervals.size or not np.isfinite(intervals).all() or np.any(intervals<=0):
        raise ValueError('Source has no valid consecutive command timestamps')
    period=float(intervals.mean())
    if not .02<=period<=.2:
        raise ValueError('Recorded mean period must be between 20 and 200 ms')
    return period


def sequence(reference,joint,amplitude,cycles,closing_only=False,waveform='steps',sine_samples=16):
    reference=vector(reference,'reference')
    if joint not in MOVING_JOINTS or not 0<amplitude<=.1 or cycles<1:
        raise ValueError('Use a fingertip flexion joint, amplitude in (0,0.1], and positive cycles')
    if waveform not in ('steps','sine') or not isinstance(sine_samples,int) or sine_samples<8:
        raise ValueError('Use steps or sine and at least 8 sine samples per cycle')
    if waveform=='sine':
        phase=2*np.pi*np.arange(1,sine_samples+1)/sine_samples
        offsets=amplitude*(1-np.cos(phase))/2 if closing_only else amplitude*np.sin(phase)
        offsets[-1]=0.
    else:
        offsets=(amplitude,amplitude/2,0.,amplitude/2) if closing_only else (amplitude,0.,-amplitude,0.)
    targets=[]
    for cycle in range(cycles):
        for offset in offsets:
            target=reference.copy();target[joint]+=offset
            targets.append(target)
    return targets


def validate_targets(model,reference,amplitude,cycles,closing_only=False,waveform='steps',sine_samples=16):
    if (model.nq,model.nv,model.nu)!=(16,16,16):
        raise ValueError('Use the calibrated 16-joint hand-only scene')
    if [model.joint(i).name for i in range(16)]!=list(JOINTS):
        raise ValueError('Unexpected joint order')
    lower,upper=control_limits(model)
    targets=np.asarray([reference]+[t for j in MOVING_JOINTS for t in sequence(reference,j,amplitude,cycles,
        closing_only=closing_only,waveform=waveform,sine_samples=sine_samples)])
    if np.any(targets<lower) or np.any(targets>upper):
        raise ValueError('Reference or test target outside calibrated joint/actuator limits')
    if any(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,name)<0 for name in TIPS):
        raise ValueError('Missing fingertip sites')


def validate_screen(path,args,reference,period):
    report=json.loads(Path(path).read_text());candidate=report['hardware_candidate']
    if not candidate['approved'] or candidate['waveform']!=args.waveform:
        raise ValueError('Simulation did not approve the requested waveform')
    if (not np.isclose(candidate['amplitude_rad'],args.amplitude,rtol=0,atol=1e-12)
            or candidate['sine_samples']!=args.sine_samples or candidate['cycles']!=args.cycles
            or candidate['closing_only']!=args.loaded_object
            or not np.allclose(report['reference'],reference,rtol=0,atol=1e-10)
            or not np.isclose(report['command_period_s'],period,rtol=0,atol=1e-9)):
        raise ValueError('Hardware targets/schedule do not match the approved simulation screen')
    return report


class Experiment:
    def __init__(self,args,model,config,period,directory):
        self.args,self.model,self.config,self.period=args,model,config,period
        lower,upper=control_limits(model)
        self.reference=np.clip(np.asarray(config['robot_qref']),lower,upper)
        self.directory=directory
        self.bridge=self.viewer=self.feedback=None
        self.commands=[];self.phases=[]
        self.phase=None
        self.last_target=self.reference.copy()
        self.object_may_be_present=False
        self.command_channel='LEAP_RESPONSE_ACTION'
        self.state_channel='LEAP_RESPONSE_STATE'
        self.report=dict(status='starting',started_at_utc=datetime.now(timezone.utc).isoformat(),
            hardware_config=config,model=str(args.model.resolve()),source_run=str(args.rate_from_run.resolve()),
            test_reference=self.reference.tolist(),reference_adjustment=(self.reference-config['robot_qref']).tolist(),
            period_s=period,rate_hz=1/period,amplitude_rad=args.amplitude,cycles=args.cycles,
            waveform=args.waveform,sine_samples=args.sine_samples,simulation_screen=str(args.screen_report) if args.screen_report else None,
            protocol='Identical absolute targets in both conditions; bridge position packets, no MPPI/rebasing.',
            fingertip_measurement='Forward kinematics of measured joints; no independent optical fingertip measurement.',
            object_condition='cube' if args.loaded_object else 'empty_hand',
            closing_only=args.loaded_object,commands=self.commands,phases=self.phases)
        tty=Path(config['port']).resolve().name
        latency_path=Path('/sys/bus/usb-serial/devices')/tty/'latency_timer'
        try:
            latency=int(latency_path.read_text().strip())
        except (OSError,ValueError):latency=None
        self.report['usb_adapter']=dict(tty=tty,latency_timer_ms=latency,latency_timer_path=str(latency_path))

    def check(self,ignore_viewer=False):
        if self.bridge.poll() is not None:
            raise RuntimeError(f'Bridge exited ({self.bridge.returncode}); see bridge.log')
        if not ignore_viewer and (self.directory/'viewer_stop').exists():
            raise KeyboardInterrupt
        if not ignore_viewer and self.viewer is not None and self.viewer.poll() is not None:
            raise KeyboardInterrupt
        q,v=self.feedback.buffer.snapshot_robot(max_robot_age=.15)
        with self.feedback.buffer.lock:
            stamp=self.feedback.buffer.robot.timestamp
        return q,v,dict(robot_timestamp=stamp)

    def publish(self,target,kind='new_target'):
        target=vector(target,'test target')
        stamp=int(time.time()*1e6)
        self.feedback.connection.publish(self.command_channel,RobotPositionAction(stamp,target).encode())
        self.feedback.connection.publish(self.state_channel+'_TARGET',RobotTargetState(stamp,target).encode())
        self.commands.append(dict(timestamp_us=stamp,target=target.tolist(),kind=kind,
            phase=None if self.phase is None else self.phase['index']))
        self.last_target=target.copy()

    def object_gate(self,name,prompt,ignore_viewer=False):
        """Keep the grasp while the operator places/removes the object.

        The acknowledgement file is written by the supervising operator/session,
        never inferred from a timeout or an unattended object-pose estimate.
        """
        self.phase=None
        target=self.last_target.copy()
        gate=self.directory/name
        status=dict(mode=name,joint='cube',prompt=prompt,ready_us=int(time.time()*1e6))
        (self.directory/'phase.json').write_text(json.dumps(status)+'\n')
        print(prompt,flush=True)
        previous_status=self.report['status']
        self.report['status']=name
        while not gate.exists():
            self.check(ignore_viewer=ignore_viewer)
            self.publish(target,'operator_hold')
            time.sleep(.1)
        self.report[name+'_acknowledged_us']=int(time.time()*1e6)
        self.report['status']=previous_status

    def wait_until(self,deadline):
        while True:
            self.check()
            remaining=deadline-time.monotonic()
            if remaining<=0:
                return
            time.sleep(min(.005,remaining))

    def hold(self,target,minimum=1.,until_settled=True):
        started=time.monotonic();last_publish=-float('inf');quiet=None
        last_stamp=None
        while True:
            q,v,meta=self.check()
            now=time.monotonic()
            if now-last_publish>=.1:
                self.publish(target,'heartbeat');last_publish=now
            # Settling requires distinct, fresh feedback packets, not repeated
            # polling of the same cached measurement.
            if meta['robot_timestamp']!=last_stamp:
                last_stamp=meta['robot_timestamp']
                quiet=(now if quiet is None else quiet) if np.max(abs(v))<=.08 else None
            if now-started>=minimum and (not until_settled or (quiet is not None and now-quiet>=.3)):
                return
            if now-started>=5.:
                raise RuntimeError('Joint velocities did not settle within 5 s; stopping test')
            time.sleep(.005)

    def phase_start(self,mode,joint):
        self.phase=dict(index=len(self.phases),mode=mode,joint=JOINTS[joint],joint_index=joint,
            fingertip=TIPS[MOVING_JOINTS.index(joint)],started_us=int(time.time()*1e6),
            waveform=self.args.waveform,samples_per_cycle=self.args.sine_samples if self.args.waveform=='sine' else 4)
        self.phase['cycles']=1 if mode=='settled' and self.args.waveform=='sine' else self.args.cycles
        self.phases.append(self.phase)
        (self.directory/'phase.json').write_text(json.dumps(self.phase)+'\n')
        print(f'{mode}: {JOINTS[joint]}, {self.phase["samples_per_cycle"]*self.phase["cycles"]} targets'+
              (f' at {1/self.period:.3f} Hz' if mode=='fast' else '; each held for >=1 s and quiet >=0.3 s'),flush=True)

    def run(self):
        bridge_log=(self.directory/'bridge.log').open('w')
        viewer_log=(self.directory/'viewer.log').open('w')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        self.feedback=LCMFeedback(self.model,robot_channel=self.state_channel,
            pose_channel='LEAP_RESPONSE_UNUSED_POSE',action_channel=self.command_channel)
        motion=None
        try:
            self.bridge=subprocess.Popen([sys.executable,str(ROOT/'leap_hand_lcm_bridge.py'),
                '--hardware-config',str(self.args.hardware_config.resolve()),'--model',str(self.args.model.resolve()),
                '--state-channel',self.state_channel,'--action-channel',self.command_channel,
                '--telemetry-log',str(self.directory/'bridge_telemetry.jsonl')],
                stdout=bridge_log,stderr=subprocess.STDOUT,env=env)
            deadline=time.monotonic()+30
            while True:
                try:
                    self.check();break
                except RuntimeError:
                    if self.bridge.poll() is not None or time.monotonic()>=deadline:
                        raise
                    time.sleep(.02)
            if self.args.viewer:
                self.viewer=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                    '--view-only',str(self.directory),'--model',str(self.args.model.resolve())],
                    stdout=viewer_log,stderr=subprocess.STDOUT,env=env)
            motion=HardwareEpisodes(self.feedback,self.config,.15,logging_enabled=False)
            motion.timeout=15
            motion.reference=self.reference.copy()
            motion.prepare(0)
            reference=self.reference
            self.last_target=reference.copy()
            if self.args.loaded_object:
                self.object_may_be_present=True
                self.object_gate('object_placed',
                    'READY FOR PLACEMENT: reference held. Place cube securely in palm/fingertip cradle; '
                    'clear your fingers and confirm object placed before test starts.')
                self.hold(reference)
            self.report['status']='running'
            for joint in MOVING_JOINTS:
                for mode in ('settled','fast'):
                    self.phase=None
                    self.publish(reference);self.hold(reference)
                    self.phase_start(mode,joint)
                    targets=sequence(reference,joint,self.args.amplitude,self.phase['cycles'],
                        closing_only=self.args.loaded_object,waveform=self.args.waveform,sine_samples=self.args.sine_samples)
                    for index,target in enumerate(targets):
                        self.check()
                        self.publish(target)
                        if mode=='settled':
                            self.hold(target)
                        else:
                            self.wait_until(time.monotonic()+self.period)
                    self.phase['sequence_finished_us']=int(time.time()*1e6)
                    self.phase['ended_us']=self.phase['sequence_finished_us']
                    # Keep the last (reference) target for a separate settling
                    # tail. It is excluded from fast-sequence error statistics.
                    if mode=='fast':
                        self.hold(reference)
                        self.phase['tail_ended_us']=int(time.time()*1e6)
            self.phase=None
            if self.args.loaded_object:
                self.object_gate('object_removed',
                    'TEST FINISHED: grasp held. Remove and support the cube, then confirm object removed '
                    'before return to zero and torque disable.',ignore_viewer=True)
                self.object_may_be_present=False
            motion.return_to_zero(0)
            self.report['status']='complete'
        except BaseException as exc:
            self.report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                               error=f'{type(exc).__name__}: {exc}')
            print(self.report['error'],flush=True)
            if motion is not None and self.bridge.poll() is None:
                try:
                    if self.object_may_be_present:
                        self.object_gate('object_removed',
                            'TEST STOPPED: holding the last target. Remove/support the cube and confirm '
                            'object removed before return to zero.',ignore_viewer=True)
                        self.object_may_be_present=False
                    motion.timeout=15;motion.return_to_zero(0)
                except BaseException as reset_error:
                    self.report['reset_error']=str(reset_error)
        finally:
            # Repeated stop packets avoid racing unsubscribe/transport cleanup.
            # A bridge fault already disables torque through its own finally.
            if self.bridge is not None:
                deadline=time.monotonic()+3
                while self.bridge.poll() is None and time.monotonic()<deadline:
                    self.feedback.publish_stop();time.sleep(.05)
                if self.bridge.poll() is None:
                    self.bridge.terminate()
                try:
                    self.bridge.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.report['bridge_cleanup_timeout']=True
                    self.bridge.kill();self.bridge.wait(timeout=2)
                self.report['bridge_exit_code']=self.bridge.returncode
            if self.viewer is not None and self.viewer.poll() is None:
                self.viewer.terminate()
                try:self.viewer.wait(timeout=5)
                except subprocess.TimeoutExpired:self.viewer.kill();self.viewer.wait(timeout=2)
            self.feedback.close();bridge_log.close();viewer_log.close()
            self.report['torque_off_verified']='Torque disabled and verified on all 16 motors.' in (self.directory/'bridge.log').read_text()
            self.report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
            (self.directory/'experiment.json').write_text(json.dumps(self.report,indent=2)+'\n')
        return 0 if self.report['status']=='complete' and self.report['torque_off_verified'] else 1


def view(args):
    import mujoco.viewer
    model=mujoco.MjModel.from_xml_path(str(args.model));data=mujoco.MjData(model);goal=mujoco.MjData(model)
    sites=[model.site(name).id for name in TIPS]
    directory=args.view_only
    feedback=LCMFeedback(model,robot_channel='LEAP_RESPONSE_STATE',pose_channel='LEAP_RESPONSE_UNUSED_POSE',
        action_channel='LEAP_RESPONSE_ACTION',target_feedback=True)
    def stop(key):
        if key in (27,256,ord('q'),ord('Q')):(directory/'viewer_stop').touch()
    try:
        with mujoco.viewer.launch_passive(model,data,key_callback=stop,show_left_ui=False,show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:]=[0,0,.07];viewer.cam.distance=.42
                viewer.cam.azimuth=130;viewer.cam.elevation=-35
            while viewer.is_running():
                status='Waiting for feedback'
                try:
                    q,v=feedback.buffer.snapshot_robot(.15)
                    with viewer.lock():
                        data.qpos[:]=q;mujoco.mj_kinematics(model,data)
                        viewer.user_scn.ngeom=0
                        try:
                            target,_=feedback.buffer.snapshot_target(.2)
                            goal.qpos[:]=target;mujoco.mj_kinematics(model,goal)
                            for point in goal.site_xpos[sites]:
                                g=viewer.user_scn.geoms[viewer.user_scn.ngeom]
                                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.003),point,np.eye(3).ravel(),np.array([1.,.8,0.,.8]))
                                viewer.user_scn.ngeom+=1
                        except RuntimeError:pass
                    status='Measured hand; yellow markers = published target fingertips\nMouse: orbit / pan / zoom. Q/Escape/close: stop test.'
                    try:
                        phase=json.loads((directory/'phase.json').read_text());status=f"{phase['mode']} / {phase['joint']}\n"+status
                    except (OSError,ValueError):pass
                except RuntimeError:pass
                viewer.set_texts([(None,mujoco.mjtGridPos.mjGRID_TOPLEFT,status,None)])
                viewer.sync();time.sleep(1/60)
    finally:
        (directory/'viewer_stop').touch();feedback.close()


def analyze(directory):
    # Import plotting only after torque cleanup, in an offline process/phase.
    from ContactModelStudy.Hardware.TargetResponseAnalysis import analyze_response
    return analyze_response(directory)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware',action='store_true')
    parser.add_argument('--hardware-config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--rate-from-run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--amplitude',type=float,
        help='empty hand: +/-amplitude; loaded object: closing offsets 0..amplitude')
    parser.add_argument('--loaded-object',action='store_true',
        help='cube placement/removal gates and closing-only finger targets (default amplitude 0.03 rad)')
    parser.add_argument('--cycles',type=int,default=4)
    parser.add_argument('--waveform',choices=['steps','sine'],default='steps')
    parser.add_argument('--sine-samples',type=int,default=16,help='samples per sine cycle at the recorded command update rate')
    parser.add_argument('--screen-report',type=Path,help='simulation-screen result to retain with the experiment metadata')
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--view-only',type=Path)
    parser.add_argument('--analyze',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.amplitude is None:args.amplitude=.03 if args.loaded_object else .08
    if args.view_only:
        view(args);return 0
    if args.analyze:
        analyze(args.analyze);return 0
    config=json.loads(args.hardware_config.read_text())
    if config.get('reviewed_for_this_hand') is not True:parser.error('Hardware configuration must be reviewed for this hand')
    model=mujoco.MjModel.from_xml_path(str(args.model))
    lower,upper=control_limits(model)
    reference=np.clip(np.asarray(config['robot_qref']),lower,upper)
    validate_targets(model,reference,args.amplitude,args.cycles,closing_only=args.loaded_object,
                     waveform=args.waveform,sine_samples=args.sine_samples)
    period=measured_period(args.rate_from_run)
    if args.screen_report is not None:
        validate_screen(args.screen_report,args,reference,period)
    movement=f'closing offsets 0..{args.amplitude:.3f}' if args.loaded_object else f'+/-{args.amplitude:.3f}'
    print(f'Configured {1/period:.3f} Hz, {movement} rad, {args.cycles} cycles on each fingertip flexion joint.',flush=True)
    if not args.hardware:
        print('Validation only: USB not opened; no commands or torque. Add --hardware to execute.');return 0
    if config.get('port'):
        owners=subprocess.run(['fuser',str(config['port'])],capture_output=True,text=True)
        if owners.returncode!=1:
            parser.error(f'USB port may already be open, or ownership could not be checked: {owners.stdout} {owners.stderr}')
    directory=args.output or ROOT/'results/diagnostics'/('target_response_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    directory=directory.resolve();directory.mkdir(parents=True,exist_ok=False)
    import fcntl
    lock=(ROOT/'.runtime/leap_target_response.lock');lock.parent.mkdir(exist_ok=True)
    with lock.open('w') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=Experiment(args,model,config,period,directory).run()
    analyze(directory)
    print(f'Results: {directory}',flush=True)
    return result


if __name__=='__main__':
    raise SystemExit(main())
