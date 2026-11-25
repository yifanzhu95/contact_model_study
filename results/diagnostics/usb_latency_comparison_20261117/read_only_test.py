"""Read-only real-hand I/O benchmark and register audit; no torque/motor writes."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from ContactModelStudy.Hardware.Hand import HardwareHand
from ContactModelStudy.Hardware.BridgeTelemetry import BridgeTelemetry
from leap_hand_lcm_bridge import LeapHandLCMBridge


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=5.)
    args=parser.parse_args();out=Path(__file__).resolve().parent
    path=ROOT/'configs/m2_tracking_diagnostic/hardware.json';config=json.loads(path.read_text())
    hand=HardwareHand(Path.home()/'leap_agentic_sysid/third_party/LEAP_Hand_Hardware',config)
    fields=dict(model_number=(0,2),mode=(11,1),p=(84,2),i=(82,2),d=(80,2),goal_current=(102,2),
        torque_enabled=(64,1),return_delay=(9,1),drive_mode=(10,1),profile_acceleration=(108,4),
        profile_velocity=(112,4),feedforward_1=(90,2),feedforward_2=(88,2),current_limit=(38,2))
    actual=[]
    try:
        hand.connect_read_only()
        for motor in range(16):
            actual.append(dict(motor=motor,**{name:hand.register(motor,*address) for name,address in fields.items()}))
    finally:hand.close_read_only()
    (out/'registers.json').write_text(json.dumps(actual,indent=2)+'\n')
    for motor,row in enumerate(actual):
        assert row['torque_enabled']==0,'Expected idle torque-off hand before comparison'
        for key,expected in config['expected_registers'].items():
            value=expected[motor] if isinstance(expected,list) else expected
            assert row[key]==value,f'Motor {motor} changed {key}: {row[key]} versus baseline {value}'
    tty=Path(config['port']).resolve().name
    latency=int((Path('/sys/bus/usb-serial/devices')/tty/'latency_timer').read_text())
    timestamps=[];start=time.monotonic()
    hand=HardwareHand(Path.home()/'leap_agentic_sysid/third_party/LEAP_Hand_Hardware',config)
    connection=SimpleNamespace()
    def publish(channel,packet):
        timestamps.append(time.monotonic())
        if timestamps[-1]-start>=args.seconds:bridge.stop_requested=True
    connection.publish=publish
    telemetry=BridgeTelemetry(out/'read_only_telemetry.jsonl',dict(read_only=True,latency_timer_ms=latency))
    model=mujoco.MjModel.from_xml_path(str(ROOT/'scenes/leap/hardware/calibrated_hand.xml'))
    bridge=LeapHandLCMBridge(hand,connection,model,config,state_only=True,telemetry=telemetry)
    try:bridge.run()
    finally:bridge.close()
    records=[json.loads(line) for line in (out/'read_only_telemetry.jsonl').read_text().splitlines()]
    cycles=[r for r in records if r.get('event')=='cycle' and 'read_duration_s' in r]
    def stats(values):
        a=np.asarray(values)
        return dict(count=len(a),mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max()))
    result=dict(latency_timer_ms=latency,motor_registers_match_baseline=True,torque_changed=False,
        scope='Physical hand, bridge state-only path; no LCM network or position/torque writes',
        serial_read_ms=stats([r['read_duration_s']*1000 for r in cycles]),
        feedback_interval_ms=stats(np.diff([r['state_timestamp_us'] for r in cycles])*.001),
        health_ms=stats([r['health_duration_s']*1000 for r in cycles if 'health_duration_s' in r]),
        telemetry=json.loads((out/'read_only_telemetry.status.json').read_text()))
    (out/'read_only_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
