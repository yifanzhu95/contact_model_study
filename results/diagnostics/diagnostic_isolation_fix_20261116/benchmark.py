"""Bridge timing emulator: no serial devices, LCM network or physics steps.

Replay the measured 16 ms serial transaction latency. Compare legacy bundled
health checks to split checks, with diagnostics off/on. Host scheduling jitter
is included; this does not certify physical hardware deadlines.
"""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from ContactModelStudy.Hardware.BridgeTelemetry import BridgeTelemetry
from ContactModelStudy.Hardware.LCM import RobotDeltaAction,RobotState
from ContactModelStudy.Hardware.Scene import DEFAULT_SCENE
from leap_hand_lcm_bridge import LeapHandLCMBridge

OUT=Path(__file__).resolve().parent


class Connection:
    def __init__(self):
        self.pending=None
        self.stamps=[]
    def subscribe(self,channel,callback):
        self.callback=callback
        return SimpleNamespace(set_queue_capacity=lambda n:None)
    def unsubscribe(self,subscription):
        pass
    def handle_timeout(self,timeout):
        if self.pending is None:
            return 0
        packet,self.pending=self.pending,None
        self.callback('LEAP_ACTION_DELTA',packet)
        return 1
    def publish(self,channel,packet):
        if channel=='LEAP_STATE':
            self.stamps.append(RobotState.decode(packet).timestamp)


class Hand:
    def __init__(self,connection,split,cycles):
        self.connection=connection
        self.cycles=cycles
        self.reads=0
        self.writes=[]
        self.checks=[]
        if split:
            self.health_register=self.single_check
    def connect(self):
        pass
    def enable(self,q):
        pass
    def close(self):
        pass
    def read(self):
        self.reads+=1
        if self.reads>self.cycles+1:
            raise KeyboardInterrupt
        time.sleep(.016)
        self.connection.pending=RobotDeltaAction(int(time.time()*1e6),np.full(16,.03)).encode()
        return np.full(16,.2),np.zeros(16),np.zeros(16)
    def single_check(self,motor,index):
        self.checks.append((time.monotonic(),motor,index))
        time.sleep(.016)
    def health(self,motor=None):
        self.single_check(motor,0)
        self.single_check(motor,1)
    def write(self,q):
        self.writes.append(q.copy())
        time.sleep(.0001)


def stats(values,scale=1.):
    values=np.asarray(values)*scale
    return dict(count=len(values),mean=float(values.mean()),
        p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),max=float(values.max()))


def main():
    model=mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    results=[]
    for repeat in range(2):
        # Reverse diagnostic ordering on the second repetition.
        for split in (False,True):
            for enabled in ((False,True) if repeat==0 else (True,False)):
                name=f'{"split" if split else "bundled"}_{"on" if enabled else "off"}_{repeat}'
                connection=Connection()
                hand=Hand(connection,split,100)
                config=dict(max_current_raw=550,delta_reference='measured_position')
                recorder=BridgeTelemetry(OUT/f'{name}.jsonl') if enabled else None
                producer_times=[]
                if recorder is not None:
                    submit=recorder.submit
                    def timed_submit(record):
                        start=time.perf_counter()
                        submit(record)
                        producer_times.append(time.perf_counter()-start)
                    recorder.submit=timed_submit
                bridge=LeapHandLCMBridge(hand,connection,model,config,telemetry=recorder)
                try:
                    bridge.run()
                except KeyboardInterrupt:
                    pass
                finally:
                    bridge.close()
                np.testing.assert_allclose(hand.writes,.23)
                row=dict(name=name,split_health=split,diagnostics=enabled,
                    publication_interval_ms=stats(np.diff(connection.stamps),.001),
                    state_count=len(connection.stamps),write_count=len(hand.writes),
                    monitoring_register_reads=len(hand.checks))
                if recorder is not None:
                    row.update(producer_us=stats(producer_times,1e6),
                        recorder=dict(accepted=recorder.accepted,written=recorder.written,
                                      dropped=recorder.dropped,error=recorder.error))
                    assert recorder.written==recorder.accepted and not recorder.dropped and recorder.error is None
                results.append(row)
                print(json.dumps(row),flush=True)
    report=dict(scope=__doc__,serial_transaction_latency_ms=16.,cycles_per_case=100,results=results)
    (OUT/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
