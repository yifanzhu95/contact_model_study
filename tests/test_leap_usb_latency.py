"""Real helper-process lifecycle with temporary files; no USB, sudo or motors."""
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time

import pytest

from ContactModelStudy.Hardware import UsbLatency as usb


def command(path):
    code=(f"import runpy; module=runpy.run_path({usb.__file__!r},run_name='usb_latency_test'); "
          f"raise SystemExit(module['guard']({str(path)!r},1,16))")
    return [sys.executable,'-I','-c',code]


def response(process):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout,selectors.EVENT_READ)
        assert selector.select(10),'helper did not respond'
    return json.loads(process.stdout.readline())


@pytest.mark.parametrize('initial',[1,16,32])
def test_runtime_one_restores_explicit_default_sixteen(tmp_path,monkeypatch,initial):
    timer_file=tmp_path/'latency_timer';timer_file.write_text(f'{initial}\n')
    monkeypatch.delenv(usb.MANAGED_ENV,raising=False)
    monkeypatch.setattr(usb,'latency_path',lambda port:(Path('/dev/ttyUSB99'),timer_file))
    popen=subprocess.Popen
    def spawn(cmd,**kwargs):
        assert cmd[0]!='sudo','fake sysfs must never invoke sudo'
        return popen(command(timer_file),**kwargs)
    monkeypatch.setattr(usb.subprocess,'Popen',spawn)
    timer=usb.UsbLatencyTimer('/dev/ttyUSB99')
    try:
        timer.start();assert timer_file.read_text().splitlines()[0]=='1'
        assert timer.metadata['previous_ms']==initial
        timer.start()  # idempotent, no second helper
    finally:
        timer.close()
    assert timer_file.read_text().splitlines()[0]=='16'
    assert timer.metadata['restored_ms']==16 and not timer.metadata['active']
    timer.close()


def test_owner_sigkill_restores_without_finally_or_extra_polling(tmp_path):
    path=tmp_path/'latency_timer';path.write_text('16\n')
    code=("import subprocess,sys,json,time\n"
          f"child=subprocess.Popen({command(path)!r},stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)\n"
          "print(child.stdout.readline(),end='',flush=True)\n"
          "while True:time.sleep(1)\n")
    parent=subprocess.Popen([sys.executable,'-I','-c',code],stdout=subprocess.PIPE,text=True)
    try:
        ready=response(parent);assert ready['event']=='ready'
        assert path.read_text().splitlines()[0]=='1'
        parent.kill();parent.wait(timeout=5)
        deadline=time.monotonic()+10
        while path.read_text().splitlines()[0]!='16' and time.monotonic()<deadline:
            time.sleep(.02)
        assert path.read_text().splitlines()[0]=='16'
    finally:
        if parent.poll() is None:parent.kill();parent.wait(timeout=5)
        parent.stdout.close()


def test_second_guard_cannot_restore_first_guard_out_from_under_it(tmp_path):
    path=tmp_path/'latency_timer';path.write_text('16\n')
    first=subprocess.Popen(command(path),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    second=None
    try:
        assert response(first)['event']=='ready'
        second=subprocess.Popen(command(path),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        assert response(second)['event']=='error'
        assert second.wait(timeout=5)==1
        assert path.read_text().splitlines()[0]=='1'
    finally:
        for process in [second,first]:
            if process is not None:
                process.stdin.close();process.wait(timeout=5);process.stdout.close()
    assert path.read_text().splitlines()[0]=='16'


def test_cleanup_keeps_original_inode_if_device_path_is_replaced(tmp_path):
    path=tmp_path/'latency_timer';path.write_text('16\n')
    process=subprocess.Popen(command(path),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert response(process)['event']=='ready'
        original=tmp_path/'disconnected_device';path.rename(original)
        path.write_text('99\n')
        process.stdin.close();assert process.wait(timeout=5)==0
        assert original.read_text().splitlines()[0]=='16'
        assert path.read_text()=='99\n'
    finally:
        if not process.stdin.closed:process.stdin.close()
        process.wait(timeout=5);process.stdout.close()


def test_recovery_waits_for_previous_guard_to_finish_cleanup(tmp_path,monkeypatch):
    path=tmp_path/'latency_timer';path.write_text('16\n')
    monkeypatch.delenv(usb.MANAGED_ENV,raising=False)
    monkeypatch.setattr(usb,'latency_path',lambda port:(Path('/dev/ttyUSB99'),path))
    popen=subprocess.Popen
    first=popen(command(path),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    assert response(first)['event']=='ready'
    def spawn(cmd,**kwargs):
        assert cmd[0]!='sudo'
        return popen(command(path),**kwargs)
    monkeypatch.setattr(usb.subprocess,'Popen',spawn)
    release=threading.Timer(.25,first.stdin.close);release.start()
    timer=usb.UsbLatencyTimer('/dev/ttyUSB99')
    try:
        timer.start()
        assert first.wait(timeout=5)==0
        assert path.read_text().splitlines()[0]=='1'
    finally:
        timer.close();release.join();first.stdout.close()
    assert path.read_text().splitlines()[0]=='16'


def test_launcher_managed_connection_reuses_guard_until_launcher_cleanup(tmp_path,monkeypatch):
    path=tmp_path/'latency_timer';path.write_text('1\n')
    monkeypatch.setattr(usb,'latency_path',lambda port:(Path('/dev/ttyUSB99'),path))
    owner=dict(device='/dev/ttyUSB99',runtime_ms=1,restore_ms=16,owner_pid=os.getpid(),
               owner_identity=usb._identity(os.getpid()),previous_ms=16,active=True)
    monkeypatch.setenv(usb.MANAGED_ENV,json.dumps(owner))
    monkeypatch.setattr(usb.UsbLatencyTimer,'_spawn',lambda *a:pytest.fail('duplicate guard'))
    timer=usb.UsbLatencyTimer('/dev/ttyUSB99').start()
    assert timer.external
    timer.close();assert path.read_text()=='1\n'


def test_permission_fallback_is_noninteractive_and_failure_is_actionable(tmp_path,monkeypatch):
    monkeypatch.delenv(usb.MANAGED_ENV,raising=False)
    monkeypatch.setattr(usb,'latency_path',lambda port:(Path('/dev/ttyUSB99'),tmp_path/'latency_timer'))
    calls=[]
    def spawn(self,privileged):
        calls.append(privileged)
        raise PermissionError('denied')
    monkeypatch.setattr(usb.UsbLatencyTimer,'_spawn',spawn)
    with pytest.raises(PermissionError,match='sudo -v'):
        usb.UsbLatencyTimer('/dev/ttyUSB99').start()
    assert calls==[False,True]


@pytest.mark.parametrize('value',[0,256,True,1.5,'1',None])
def test_invalid_timers_rejected_before_any_helper(value):
    with pytest.raises(ValueError):usb.UsbLatencyTimer('/dev/unused',value)
