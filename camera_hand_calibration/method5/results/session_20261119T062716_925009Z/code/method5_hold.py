#!/usr/bin/env python3
"""Preparation-pose hold for method 5, independent of vision/editor/optimizer."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent))
import argparse
import time
import numpy as np
from workspace import local,read_json,write_json,utc
from method4_capture import Method4,port_owners
from method4_reference import approach_path
from articulated_motion import CalibrationHand,API
from ContactModelStudy.Hardware.Hand import check_device_feedback
from ContactModelStudy.Hardware.Episodes import HardwareEpisodes


class PreparationHold(Method4):
    def __init__(self,config_path,run_dir):
        super().__init__(config_path,run_dir)
        self.target=np.array(self.hardware['robot_qref'],float)
        self.record.update(method=5,hardware_config=self.hardware,preparation_target=self.target.tolist(),
                           keep_torque_after_fit=True,reference_cube_never_commands_motors=True)
    def execute(self):
        try:
            self.hand.connect();q,v,current=self.hand.read();check_device_feedback(q,v,current,self.hardware)
            waypoints,path=approach_path(self.m,q,self.target,self.config['collision_margin_m'])
            self.record.update(initial_measured_q=q.tolist(),fresh_path_screen=path)
            write_json(self.run/'session.json',self.record)
            self.hand.enable(q);self.enabled=True;self.command=q.copy()
            self.status('moving_preparation','Moving gradually to the hardware experiment robot_qref preparation pose.')
            for target in waypoints:self.move(q,target);q=target.copy()
            settings=HardwareEpisodes(None,self.hardware,.3)
            quiet=None;last_report=0.
            while True:
                q,v=self.cycle();now=time.monotonic()
                if self.interrupts or (self.run/'shutdown.request').exists():raise KeyboardInterrupt('Preparation interrupted')
                reached=np.all(abs(q-self.target)<=settings.position_tolerances()) and max(abs(v))<=settings.settle_velocity
                quiet=now if reached and quiet is None else quiet if reached else None
                if quiet is not None and now-quiet>=settings.settle_time:break
                if now-last_report>2:
                    index=int(np.argmax(abs(q-self.target)));print(f'Preparation settling: joint {index}, error {abs(q[index]-self.target[index]):.4f} rad',flush=True);last_report=now
            self.may_have_object=True
            self.record.update(status='holding_preparation',held_target=self.command.tolist(),measured_at_ready=q.tolist())
            write_json(self.run/'session.json',self.record)
            self.status('holding_preparation','PREPARATION READY. Holding robot_qref. Start the normal mask/tracker, then place the cube for manual alignment. Torque stays on through fitting.')
            write_json(self.run/'preparation_ready.trigger',dict(ready_at=time.time()))
            while not (self.run/'shutdown.request').exists() and not self.interrupts:self.cycle()
            if self.interrupts<2:
                self.interrupts=0
                self.gate('object_removed','awaiting_object_removal','Remove the cube completely. Confirm removal before disabling torque. Holding the preparation target.')
                self.may_have_object=False
        except KeyboardInterrupt:
            if self.enabled and self.may_have_object and self.interrupts<2:
                self.interrupts=0
                self.gate('object_removed','awaiting_object_removal','Interrupted: remove the cube and confirm before torque is disabled.')
                self.may_have_object=False
        finally:
            try:self.hand.close();self.record['torque_disabled_verified']=True;self.enabled=False
            except Exception as exc:self.record['torque_cleanup_error']=str(exc)
            self.sub.close();self.context.term();self.log.close()
            self.record.update(status='stopped',finished_at=utc());write_json(self.run/'session.json',self.record)
            self.status('torque_off' if self.record['torque_disabled_verified'] else 'cleanup_failed','Method 5 hold ended. Inspect session torque verification.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='method5/config.json');p.add_argument('--run-dir',required=True);a=p.parse_args();PreparationHold(a.config,a.run_dir).execute()
