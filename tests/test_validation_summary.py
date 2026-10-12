"""CPU-only tests for the distinction between completion and scientific validity."""

from __future__ import annotations

from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder, _Episode


def test_kamino_validation_is_separate_from_task_success():
    rec = EpisodeRecorder.__new__(EpisodeRecorder)
    ep = _Episode(
        id="strict",
        configs={
            "rollout_simulator": {"class": "Kamino"},
            "planner": {"config": {"n_iterations": 1}},
        },
    )
    ep.q = [[0.0], [0.0]]
    valid = {
        "all_worlds_valid": True,
        "valid_worlds": 4,
        "invalid_worlds": 0,
        "max_iterations": 20,
        "max_active_contacts": 12,
        "blocking_reasons": {},
    }
    ep.planner_diagnostics = [[valid], [valid]]
    summary = rec._plannerValidation(ep)
    assert summary["status"] == "pass" and summary["scaling_eligible"]
    assert summary["expected_rollouts"] == summary["recorded_rollouts"] == 2

    ep.planner_diagnostics[1] = [{
        **valid,
        "all_worlds_valid": False,
        "valid_worlds": 3,
        "invalid_worlds": 1,
        "blocking_reasons": {"nonconverged_worlds": 1},
    }]
    summary = rec._plannerValidation(ep)
    assert summary["status"] == "fail" and not summary["scaling_eligible"]
    assert summary["blocking_reasons"] == {
        "nonconverged_worlds": 1,
        "invalid_rollouts": 1,
    }
