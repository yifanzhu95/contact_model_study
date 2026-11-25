"""Persistent metrics, asynchronous checkpoints and snapshot isolation."""
import json
import threading

import numpy as np
import pytest

from ContactModelStudy.Hardware.RunLogging import RunLogger, quaternion_angle, snapshot


def record():
    return dict(args=dict(n_episodes=3), status='success_hold', started_at_utc='2026-11-10T00:00:00+00:00',
        episodes=[dict(episode=0, steps=[dict(step=0, planning_time=.08)], n_steps=1,
            finish_reason='success', completed=True, manipulation_wall_time_s=.1)])


def test_checkpoints_write_in_worker_and_do_not_follow_live_list_mutations(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def save(self, item):
        assert threading.current_thread() is not threading.main_thread()
        calls.append(item)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
    monkeypatch.setattr(RunLogger, '_save', save)
    logger = RunLogger(tmp_path/'controller.json')
    data = record()
    try:
        logger.checkpoint(data)
        assert entered.wait(3)
        # The worker is blocked; a checkpoint must still return to control.
        logger.checkpoint(data)
        data['episodes'][0]['steps'].append(dict(step=1, planning_time=.09))
        data['episodes'][0]['n_steps'] = 2
    finally:
        release.set()
        logger.close(data)
    assert len(calls[0]['episodes'][0]['steps']) == 1
    assert len(calls[-1]['episodes'][0]['steps']) == 2


def test_summary_distinguishes_success_failure_and_aborted_episodes():
    data = record()
    data['episodes'].extend([
        dict(episode=1, steps=[], n_steps=0, finish_reason='failure', completed=True, manipulation_wall_time_s=.01),
        dict(episode=2, steps=[], n_steps=0, finish_reason='error', completed=False, manipulation_wall_time_s=.02,
             failed_step=dict(planning_time=.2))])
    saved = snapshot(data)
    summary = saved['summary']
    assert summary['completed_episodes'] == 2 and summary['aborted_episodes'] == 1
    assert summary['successful_episodes'] == summary['unsuccessful_episodes'] == 1
    assert summary['success_rate_completed'] == .5
    assert summary['success_rate_requested'] == pytest.approx(1/3)
    assert summary['mppi_time']['count'] == 2
    assert summary['mppi_time']['total_s'] == pytest.approx(.28)
    assert quaternion_angle(0) == 0
    assert quaternion_angle(1) == pytest.approx(np.pi)
    assert quaternion_angle(.04) == pytest.approx(2*np.arcsin(.2))


def test_logging_failure_is_reported_when_flushing_after_control_cleanup(tmp_path, monkeypatch):
    def failed(self, data):
        raise OSError('disk unavailable')
    monkeypatch.setattr(RunLogger, '_save', failed)
    logger = RunLogger(tmp_path/'controller.json')
    with pytest.raises(OSError, match='disk unavailable'):
        logger.close(record())
    assert not logger.thread.is_alive()
