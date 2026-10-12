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
