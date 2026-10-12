"""CPU compilation and fail-closed behavior of Kamino's device diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("newton")

from newton._src.solvers.kamino._src.solvers.padmm.types import PADMMStatus

from ContactModelStudy.Simulators._KaminoDiagnostics import KaminoRolloutDiagnostics


def _case(converged: bool):
    device, world_count = wp.get_device("cpu"), 2
    diagnostics = KaminoRolloutDiagnostics(world_count, device)
    status = wp.zeros(world_count, dtype=PADMMStatus, device=device)
    host = status.numpy()
    host["converged"] = int(converged)
    host["iterations"] = 7
    host["r_p"] = 1e-6
    host["r_d"] = 2e-6
    host["r_c"] = 3e-6
    status.assign(host)
    contacts = wp.array(np.array([3, 4], dtype=np.int32), dtype=wp.int32, device=device)
    reset = wp.array(np.array([True, True]), dtype=wp.bool, device=device)
    qpos = wp.zeros((world_count, 3), dtype=wp.float32, device=device)
    qvel = wp.zeros((world_count, 2), dtype=wp.float32, device=device)
    running = wp.array(np.array([1.0, 2.0], dtype=np.float32), dtype=wp.float32, device=device)
    terminal = wp.array(np.array([0.5, 0.25], dtype=np.float32), dtype=wp.float32, device=device)

    diagnostics.reset()
    diagnostics.accumulate(status, contacts, 5e-4)
    diagnostics.invalidate_costs(
        reset_success=reset,
        qpos=qpos,
        qvel=qvel,
        running_costs=running,
        terminal_costs=terminal,
        expected_fine_steps=1,
        iteration_limit=100,
        contact_capacity=128,
    )
    wp.synchronize_device(device)
    return diagnostics.summary(1, 100, 128), running.numpy(), terminal.numpy()


def test_valid_candidate_costs_survive():
    summary, running, terminal = _case(True)
    assert summary["all_worlds_valid"] and summary["valid_worlds"] == 2
    assert np.allclose(running, [1.0, 2.0])
    assert np.allclose(terminal, [0.5, 0.25])


def test_nonconverged_candidate_costs_become_nan():
    summary, running, terminal = _case(False)
    assert not summary["all_worlds_valid"] and summary["invalid_worlds"] == 2
    assert summary["blocking_reasons"] == {"nonconverged_worlds": 2}
    assert np.isnan(running).all() and np.isnan(terminal).all()


@pytest.mark.parametrize("fault,usable", [
    ("finite_residual", True), ("iteration_cap", True), ("capacity_hit", True),
    ("nonfinite_residual", False), ("state", False), ("transient_state", False),
    ("cost", False), ("reset", False), ("overflow", False),
])
def test_exploration_acceptance_keeps_quality_separate_from_hard_safety(fault, usable):
    device = wp.get_device("cpu")
    d = KaminoRolloutDiagnostics(1, device)
    status = wp.zeros(1, dtype=PADMMStatus, device=device)
    host = status.numpy()
    host["converged"] = 1
    host["iterations"] = 7
    host["r_p"], host["r_d"], host["r_c"] = 1e-6, 2e-6, 3e-6
    if fault == "finite_residual":
        host["converged"], host["r_d"] = 0, .0048
    if fault == "iteration_cap": host["iterations"] = 100
    if fault == "nonfinite_residual": host["r_d"] = np.nan
    status.assign(host)
    contact_count = 128 if fault == "capacity_hit" else 129 if fault == "overflow" else 3
    contacts = wp.array([contact_count], dtype=wp.int32, device=device)
    reset = wp.array([fault != "reset"], dtype=wp.bool, device=device)
    qpos = wp.zeros((1, 3), dtype=wp.float32, device=device)
    qvel = wp.zeros((1, 2), dtype=wp.float32, device=device)
    if fault == "state": qpos.assign(np.full((1, 3), np.inf, np.float32))
    running = wp.array([np.inf if fault == "cost" else 1.0], dtype=wp.float32, device=device)
    terminal = wp.array([.5], dtype=wp.float32, device=device)
    d.reset()
    if fault == "transient_state":
        native_q = wp.array([np.inf, 0, 0], dtype=wp.float32, device=device)
        native_v = wp.zeros(2, dtype=wp.float32, device=device)
        d.accumulate_state(native_q, native_v)
        native_q.zero_()
        d.accumulate_state(native_q, native_v)
    d.accumulate(status, contacts, 5e-4)
    d.invalidate_costs(reset_success=reset, qpos=qpos, qvel=qvel, running_costs=running,
                       terminal_costs=terminal, expected_fine_steps=1, iteration_limit=100,
                       contact_capacity=128, acceptance_policy="exploratory")
    summary = d.summary(1, 100, 128)
    assert summary["acceptance_policy"] == "exploratory"
    assert summary["usable_worlds"] == int(usable)
    assert not summary["all_worlds_valid"]
    assert bool(np.isfinite(running.numpy()).all()) == usable
    if usable:
        assert summary["quality_warnings"] and not summary["hard_blocking_reasons"]
        assert np.allclose(running.numpy(), [1])


def test_two_world_state_scan_does_not_cross_contiguous_world_boundaries():
    device = wp.get_device("cpu")
    d = KaminoRolloutDiagnostics(2, device)
    q = wp.array([0, 0, 0, 0, np.inf, 0], dtype=wp.float32, device=device)
    v = wp.zeros(4, dtype=wp.float32, device=device)
    d.accumulate_state(q, v)
    assert d.nonfinite_state_steps.numpy().tolist() == [0, 1]
