"""Opt-in diagnostic routing and bounded writer behavior; no hardware."""
import json
import pickle
import socket
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import launch_leap_experiment as launcher
from ContactModelStudy.Hardware.BridgeTelemetry import BridgeTelemetry


def config(tmp_path):
    value=json.loads((launcher.ROOT/'configs/leap_experiment.json').read_text())
    value['init_mask_path']=str(tmp_path/'mask.png')
    return value


def test_telemetry_only_changes_bridge_command(tmp_path):
    value=config(tmp_path)
    before=launcher.build_commands(value,tmp_path)
    value['bridge_telemetry']=True
    after=launcher.build_commands(value,tmp_path)
    assert after['bridge']==before['bridge']+['--telemetry-log',str(tmp_path/'bridge_telemetry.jsonl')]
    assert {k:v for k,v in after.items() if k!='bridge'}=={k:v for k,v in before.items() if k!='bridge'}


def test_telemetry_requires_logging(tmp_path):
    value=config(tmp_path);value.update(bridge_telemetry=True,logging=False)
    with pytest.raises(ValueError,match='requires logging'):
        launcher.build_commands(value,tmp_path)


def test_writer_producer_drops_instead_of_blocking_when_full():
    writer=BridgeTelemetry.__new__(BridgeTelemetry)
    packets=[]
    def send(payload):
        if packets:
            raise BlockingIOError('recorder backpressure')
        packets.append(payload)
        return len(payload)
    writer.socket=SimpleNamespace(send=send);writer.closed=False;writer.error=None
    writer.accepted=writer.dropped=0
    writer.submit({'sample':1});writer.submit({'sample':2})
    assert writer.accepted==1 and writer.dropped==1 and len(packets)==1
    assert pickle.loads(packets[0])=={'sample':1}


def test_writer_flushes_numpy_records_and_reports_loss(tmp_path):
    path=tmp_path/'records.jsonl';writer=BridgeTelemetry(path,{'diagnostic':True})
    for i in range(20):writer.submit(dict(event='sample',i=i,q=np.full(16,i,dtype=float)))
    writer.close()
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]['event']=='start' and rows[-1]['event']=='end'
    assert [r['i'] for r in rows if r['event']=='sample']==list(range(20))
    assert rows[2]['q']==[1.]*16
    status=json.loads(path.with_suffix('.status.json').read_text())
    assert status==dict(written=20,accepted=20,dropped=0,error=None,writer_finished=True)


def test_recorder_is_separate_and_survives_group_interrupt_until_socket_closes(tmp_path):
    import os
    writer=BridgeTelemetry(tmp_path/'isolated.jsonl')
    assert writer.process.pid!=os.getpid() and not hasattr(writer,'thread')
    assert writer.socket.getblocking() is False
    os.kill(writer.process.pid,signal.SIGINT)
    writer.submit(dict(event='sample',q=np.zeros(16)))
    writer.close()
    assert writer.process.returncode==0 and writer.written==1 and writer.error is None


def test_recorder_exit_does_not_block_or_crash_control_producer(tmp_path):
    writer=BridgeTelemetry(tmp_path/'dead.jsonl')
    writer.process.kill();writer.process.wait(timeout=5)
    writer.submit(dict(event='sample'))
    assert writer.dropped==1 and writer.error is not None
    writer.close()


def test_stopped_recorder_drops_samples_and_flushes_accepted_records_on_close(tmp_path):
    import os
    writer=BridgeTelemetry(tmp_path/'backpressure.jsonl',capacity=1)
    try:
        os.kill(writer.process.pid,signal.SIGSTOP)
        os.waitpid(writer.process.pid,os.WUNTRACED)
        for i in range(1000):
            writer.submit(dict(event='sample',i=i,q=np.zeros(256)))
        assert writer.accepted>0 and writer.dropped>0 and writer.error is None
    finally:
        os.kill(writer.process.pid,signal.SIGCONT)
        writer.close()
    assert writer.written==writer.accepted and writer.error is None
