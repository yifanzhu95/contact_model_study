"""Persistent vision ownership and concurrency without Docker or hardware."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ContactModelStudy.Hardware import TrackingSession as sessions


def test_only_one_experiment_can_own_a_tracking_session_at_a_time(tmp_path):
    first, second = sessions.TrackingSession(tmp_path), sessions.TrackingSession(tmp_path)
    try:
        first.acquire()
        with pytest.raises(RuntimeError, match='Another experiment'):
            second.acquire()
        first.release()
        second.acquire()
    finally:
        first.release()
        second.release()


def test_pid_reuse_never_signals_an_unrelated_process(tmp_path, monkeypatch):
    session = sessions.TrackingSession(tmp_path)
    recorded = dict(pid=12345, pgid=12345, uid=os.getuid(), start_ticks=10, boot_id='old')
    session.record = dict(version=1, processes=dict(tracker=recorded), directory=str(tmp_path/'old'), container=None)
    monkeypatch.setattr(sessions, 'process_identity', lambda pid: dict(recorded, start_ticks=999))
    monkeypatch.setattr(sessions.os, 'killpg', lambda *a: pytest.fail('Signaled a reused PID'))
    session.stop(.01)
    assert session.record is None


def test_owned_container_is_stopped_by_full_id_even_when_clients_are_gone(tmp_path, monkeypatch):
    session = sessions.TrackingSession(tmp_path)
    identifier = 'a'*64
    session.record = dict(version=1, processes={}, directory=str(tmp_path/'old'), container=['docker-wrapper', identifier])
    calls = []
    monkeypatch.setattr(sessions.subprocess, 'run', lambda argv, **kw: (calls.append(argv) or SimpleNamespace(returncode=0, stderr='')))
    session.stop(.01)
    assert calls == [['docker-wrapper', 'stop', '--time', '2', identifier]]


def test_stop_never_deletes_logged_experiment_data(tmp_path):
    workspace = tmp_path/'experiment'
    workspace.mkdir()
    (workspace/'controller.json').write_text('results')
    session = sessions.TrackingSession(tmp_path/'registry')
    session.record = dict(version=1, processes={}, directory=str(workspace), container=None, temporary_workspace=False)
    session.stop(.01)
    assert (workspace/'controller.json').read_text() == 'results'


def test_stop_only_works_on_launcher_created_process_groups(tmp_path, monkeypatch):
    session = sessions.TrackingSession(tmp_path)
    recorded = dict(pid=12345, pgid=1, uid=os.getuid(), start_ticks=10, boot_id='test')
    session.record = dict(version=1, processes=dict(tracker=recorded), directory=str(tmp_path/'old'), container=None)
    monkeypatch.setattr(sessions, 'process_identity', lambda pid: recorded)
    monkeypatch.setattr(sessions.os, 'killpg', lambda *a: pytest.fail('Signaled an unowned group'))
    # No waiting in this synthetic ownership check: advance its clock past each deadline.
    ticks = iter(range(20))
    monkeypatch.setattr(sessions.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(RuntimeError, match='did not stop'):
        session.stop(.01)
