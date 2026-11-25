import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import method5_zero
from workspace import local,read_json
from articulated_motion import model,collision_pairs


def controller(run):
    m=model();hardware=read_json('method5/hardware.json');command=np.array(hardware['robot_qref']);events=[];clock=[0.]
    c=SimpleNamespace(run=run,m=m,pairs=collision_pairs(m),command=command,hardware=hardware,record={},
        config=read_json('method5/config.json'))
    def cycle():clock[0]+=.1;return c.command.copy(),np.zeros(16)
    def move(start,end):events.append(('move',start.copy(),end.copy()));c.command=end.copy()
    c.cycle=cycle;c.move=move;c.status=lambda *args:events.append(('status',args))
    return c,events,clock


def test_no_zero_motion_without_explicit_cube_removal():
    with tempfile.TemporaryDirectory(dir=local('cache')) as folder:
        c,events,_=controller(Path(folder))
        with pytest.raises(RuntimeError,match='confirmation'):method5_zero.return_to_zero(c)
        assert events==[]


def test_screened_zero_return_then_settles_without_changing_gains(monkeypatch):
    with tempfile.TemporaryDirectory(dir=local('cache')) as folder:
        c,events,clock=controller(Path(folder));(c.run/'object_removed.trigger').write_text('confirmed')
        monkeypatch.setattr(method5_zero.time,'monotonic',lambda:clock[0])
        method5_zero.return_to_zero(c)
        assert c.record['zero_return']['settled']
        np.testing.assert_allclose(c.command,0)
        assert len([r for r in events if r[0]=='move'])==1
        assert c.record['zero_return']['command_path']['passed']


def test_failed_fresh_screen_refuses_motion(monkeypatch):
    with tempfile.TemporaryDirectory(dir=local('cache')) as folder:
        c,events,_=controller(Path(folder));(c.run/'object_removed.trigger').write_text('confirmed')
        monkeypatch.setattr(method5_zero,'screen',lambda *args:dict(passed=False))
        with pytest.raises(RuntimeError,match='screen'):method5_zero.return_to_zero(c)
        assert not any(r[0]=='move' for r in events)
