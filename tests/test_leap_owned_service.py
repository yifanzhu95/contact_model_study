"""Container-side service lifetime without Docker, CUDA or hardware."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import leap_owned_service as service


def test_run_execs_service_and_stop_verifies_its_exit(tmp_path):
    pid_file=tmp_path/'service identity.json';ready=tmp_path/'ready'
    child=subprocess.Popen([sys.executable,str(Path(service.__file__)),'run','--pid-file',str(pid_file),'--',
        sys.executable,'-c',f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); time.sleep(60)"],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert ready.exists()
        record=json.loads(pid_file.read_text())
        assert record['pid']==child.pid and service.identity(child.pid)==record
        service.stop_service(pid_file,.5)
        child.wait(timeout=3)
        assert service.identity(child.pid) is None and not pid_file.exists()
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=3)


def test_stale_identity_never_signals_a_reused_pid(tmp_path,monkeypatch):
    pid_file=tmp_path/'old.json'
    recorded=dict(pid=123456,start_ticks=10,uid=os.getuid(),boot_id='old')
    pid_file.write_text(json.dumps(recorded))
    monkeypatch.setattr(service,'identity',lambda pid:dict(recorded,start_ticks=999))
    monkeypatch.setattr(service.os,'kill',lambda *_:pytest.fail('Signaled a reused PID'))
    service.stop_service(pid_file,.01)
    assert not pid_file.exists()


def test_stuck_service_prevents_false_shutdown_confirmation(tmp_path,monkeypatch):
    pid_file=tmp_path/'stuck.json';recorded=dict(pid=123456,start_ticks=10,uid=os.getuid(),boot_id='old')
    pid_file.write_text(json.dumps(recorded));signals=[]
    monkeypatch.setattr(service,'identity',lambda pid:recorded)
    monkeypatch.setattr(service.os,'kill',lambda pid,sig:signals.append(sig))
    ticks=iter(range(100))
    monkeypatch.setattr(service.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(service.time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='did not exit'):service.stop_service(pid_file,.01)
    assert signals==[service.signal.SIGINT,service.signal.SIGTERM,service.signal.SIGKILL]
    assert pid_file.exists()


def test_invalid_pid_is_not_signaled(tmp_path,monkeypatch):
    pid_file=tmp_path/'pid1.json';pid_file.write_text(json.dumps(dict(pid=1,uid=os.getuid())))
    monkeypatch.setattr(service.os,'kill',lambda *_:pytest.fail('Signaled PID 1'))
    with pytest.raises(ValueError,match='Invalid owned'):service.stop_service(pid_file,.01)
