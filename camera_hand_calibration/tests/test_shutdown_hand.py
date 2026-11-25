from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import pytest
import shutdown_hand as recovery
from workspace import local,read_json
from articulated_motion import model


class FakeHand:
    def __init__(self,q):self.q=q.copy();self.events=[];self.client=SimpleNamespace(is_connected=False)
    def connect_read_only(self):self.client.is_connected=True;self.events.append('connect_without_torque_change')
    def register(self,motor,address,size):return {0:1240,11:5,64:1}[address]
    def health(self):pass
    def read(self):return self.q.copy(),np.zeros(16),np.zeros(16)
    def enable(self,q):self.events.append(('preload_then_enable',q.copy()))
    def write(self,q):self.q=q.copy();self.events.append(('write',q.copy()))
    def health_register(self,*args):pass
    def close(self):self.events.append('verified_off');self.client.is_connected=False


def fast_clock(monkeypatch):
    clock=[1.]
    monkeypatch.setattr(recovery.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(recovery.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+max(seconds,.001)))


def test_torque_enabled_recovery_preloads_current_and_reaches_zero(monkeypatch):
    fast_clock(monkeypatch);cfg=read_json('method5/hardware.json');start=np.array(cfg['robot_qref']);hand=FakeHand(start);record={}
    with tempfile.TemporaryDirectory(dir=local('cache')) as folder:
        recovery.recover(hand,model(),cfg,Path(folder),record)
    assert hand.events[0]=='connect_without_torque_change'
    np.testing.assert_array_equal(hand.events[1][1],start)
    np.testing.assert_allclose(hand.q,0);assert record['zero_settled']
    writes=[event[1] for event in hand.events if isinstance(event,tuple) and event[0]=='write']
    assert np.max(abs(writes[0]-start))<.001
    assert max(np.max(abs(b-a)) for a,b in zip(writes[:-1],writes[1:]))<.007


def test_torque_off_only_never_enables_or_moves():
    cfg=read_json('method5/hardware.json');hand=FakeHand(np.ones(16));record={}
    with tempfile.TemporaryDirectory(dir=local('cache')) as folder:
        recovery.recover(hand,None,cfg,Path(folder),record,off_only=True)
    assert hand.events==['connect_without_torque_change']


def test_port_termination_failure_refuses_to_open_or_write(monkeypatch):
    def refused(port):raise RuntimeError('USB owner cannot be terminated')
    monkeypatch.setattr(recovery,'release_port',refused)
    monkeypatch.setattr(recovery,'CalibrationHand',lambda *args:pytest.fail('Occupied bus was opened'))
    with pytest.raises(SystemExit) as exc:recovery.main(['--empty-hand-confirmed'])
    assert exc.value.code==2


@pytest.mark.parametrize('ignore_term',[False,True])
def test_port_owners_are_terminated_before_recovery(ignore_term,tmp_path):
    import subprocess,sys,time
    port=tmp_path/'fake_usb';port.write_text('fake device')
    child=subprocess.Popen([sys.executable,'-u','-c',
        'import sys,time,signal\n'
        +('signal.signal(signal.SIGTERM,signal.SIG_IGN)\n' if ignore_term else '')
        +'f=open(sys.argv[1]);print("ready",flush=True);time.sleep(60)',str(port)],stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        actions=recovery.release_port(str(port),grace=.1,kill_wait=1,timeout=3)
        child.wait(timeout=3)
        assert child.pid not in recovery.port_owners(str(port))
        assert actions[0]['signal']=='SIGTERM'
        assert any(row['signal']=='SIGKILL' for row in actions)==ignore_term
    finally:
        if child.poll() is None:child.kill();child.wait()


def test_missing_empty_hand_ack_never_opens_usb(monkeypatch):
    monkeypatch.setattr(recovery,'CalibrationHand',lambda *args:pytest.fail('Missing acknowledgement opened USB'))
    with pytest.raises(SystemExit) as exc:recovery.main([])
    assert exc.value.code==2


def test_portable_api_and_xml_override_avoid_cached_model(monkeypatch):
    hand=FakeHand(np.zeros(16));seen={}
    monkeypatch.setattr(recovery,'release_port',lambda port:[])
    def create(api,config):seen['api']=api;return hand
    monkeypatch.setattr(recovery,'CalibrationHand',create)
    monkeypatch.setattr(recovery,'model',lambda:pytest.fail('Portable override loaded the source-machine MJB'))
    def record_recovery(hand,m,config,run,record,**kwargs):
        seen['joints']=m.nu;hand.client.is_connected=True;record['zero_settled']=True
    monkeypatch.setattr(recovery,'recover',record_recovery)
    with tempfile.TemporaryDirectory(dir=local('cache')) as folder:
        run=Path(folder)
        monkeypatch.setattr(recovery,'local',lambda value:run/'session' if str(value).startswith('results/') else run/'lock')
        xml=recovery.ROOT.parent/'scenes/leap/hardware/calibrated_hand.xml'
        assert recovery.main(['--empty-hand-confirmed','--hardware-api','third_party/LEAP_Hand_Hardware',
                              '--model',str(xml)])==0
        assert seen['api']==Path('third_party/LEAP_Hand_Hardware') and seen['joints']==16
        assert hand.events==['verified_off']


def test_feedback_failure_still_disables_torque_and_returns_failure(monkeypatch):
    hand=FakeHand(np.zeros(16))
    def broken():raise OSError('USB feedback timeout')
    hand.read=broken
    monkeypatch.setattr(recovery,'port_owners',lambda port:[])
    monkeypatch.setattr(recovery,'CalibrationHand',lambda *args:hand)
    with tempfile.TemporaryDirectory(dir=local('cache')) as folder:
        run=Path(folder)
        monkeypatch.setattr(recovery,'local',lambda value:run/'session' if str(value).startswith('results/') else run/'lock')
        assert recovery.main(['--empty-hand-confirmed'])==1
        assert hand.events[-1]=='verified_off'
        saved=read_json(run/'session/session.json')
        assert saved['torque_disabled_verified'] and not saved['zero_settled']
