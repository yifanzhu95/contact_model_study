"""Budgets and signal requests, without physics stepping or PPO optimization."""
import signal

import pytest

from ContactModelStudy.RL.Execution import BoundaryStop, update_budget


def test_sample_budget_never_exceeds_cap_or_changes_rollout_length():
    budget = update_budget(10, 3, 8, max_transitions=100)
    assert budget['effective_updates'] == 4
    assert budget['effective_transition_limit'] == 96
    assert budget['transitions_per_update'] == 24
    assert update_budget(2, 3, 8, 100)['effective_updates'] == 2
    assert update_budget(0, 3, 8, 0)['effective_transition_limit'] == 0


@pytest.mark.parametrize('cap', [-1, True, 23, 1.5])
def test_invalid_or_less_than_one_update_sample_cap_is_rejected(cap):
    with pytest.raises(ValueError):
        update_budget(10, 3, 8, cap)


def test_wall_time_includes_setup_and_reserves_saving_time():
    now = [10.]
    stop = BoundaryStop(100., 20., started=0., clock=lambda: now[0])
    assert stop.reason() is None
    now[0] = 80.
    assert stop.reason() == 'wall_time_budget'


@pytest.mark.parametrize('seconds,margin', [(0, 0), (float('inf'), 0), (60, 60), (60, -1), (True, 0), (60, True)])
def test_invalid_time_budgets_are_rejected(seconds, margin):
    with pytest.raises(ValueError):
        BoundaryStop(seconds, margin)


def test_signal_requests_normal_control_flow_and_restores_handlers():
    previous = signal.getsignal(signal.SIGTERM)
    with BoundaryStop() as stop:
        signal.raise_signal(signal.SIGTERM)
        assert stop.reason() == 'signal:SIGTERM'
    assert signal.getsignal(signal.SIGTERM) == previous
