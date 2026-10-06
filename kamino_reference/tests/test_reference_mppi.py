"""Tests for the common MPPI layer used in formulation comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from kamino_feasibility.reference_mppi import (
    mppi_softmin_update,
    run_reference_mppi,
)
from kamino_feasibility.reference_planning import (
    ReferencePlanningSpec,
    make_gaussian_reference_bundle,
)


def _bundle(*, n_iterations: int = 2):
    spec = ReferencePlanningSpec(
        task="grasp_reorient",
        geometry="cube_high_high",
        scene_sha256="a" * 64,
        cost_source_sha256="b" * 64,
        nq=23,
        nv=22,
        nu=2,
        n_samples=4,
        n_iterations=n_iterations,
        control_horizon=3,
        physics_steps_per_control=2,
        physics_dt_s=0.01,
        control_dt_s=0.02,
        requested_horizon_s=0.06,
        realized_horizon_s=0.06,
        temperature=0.5,
        noise_sigma=0.1,
    )
    return make_gaussian_reference_bundle(
        spec,
        qpos=np.zeros(23),
        qvel=np.zeros(22),
        current_ctrl=np.zeros(2),
        goal=np.zeros(10, dtype=np.float32),
        cost_weights=np.zeros(12, dtype=np.float32),
        seed=7,
    )


def test_equal_costs_produce_arithmetic_candidate_mean() -> None:
    candidates = np.array(
        [
            [[-1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 2.0]],
        ],
        dtype=np.float32,
    )
    update = mppi_softmin_update(
        candidates,
        np.array([4.0, 4.0, 4.0], dtype=np.float32),
        temperature=1.0,
    )
    np.testing.assert_allclose(update.nominal_actions, [[0.0, 1.0]], atol=1.0e-7)
    np.testing.assert_allclose(update.weights, np.full(3, 1.0 / 3.0), atol=1.0e-7)
    assert update.effective_sample_size == pytest.approx(3.0, abs=1.0e-6)


def test_lower_cost_candidate_dominates_at_low_temperature() -> None:
    candidates = np.array([[[0.8]], [[-0.2]]], dtype=np.float32)
    update = mppi_softmin_update(
        candidates,
        np.array([0.0, 10.0], dtype=np.float32),
        temperature=0.01,
    )
    assert update.first_action[0] == pytest.approx(0.8, abs=1.0e-6)
    assert update.weights[0] == pytest.approx(1.0, abs=1.0e-6)


def test_nan_rollout_gets_zero_weight() -> None:
    candidates = np.array([[[0.1]], [[0.9]]], dtype=np.float32)
    update = mppi_softmin_update(
        candidates,
        np.array([np.nan, 2.0], dtype=np.float32),
        temperature=1.0,
    )
    np.testing.assert_array_equal(update.weights, [0.0, 1.0])
    assert update.first_action[0] == pytest.approx(0.9)
    assert update.valid_sample_count == 1


def test_multi_iteration_runner_recenters_same_contract() -> None:
    bundle = _bundle(n_iterations=2)
    seen: list[np.ndarray] = []

    def evaluate(candidates: np.ndarray, iteration: int) -> np.ndarray:
        seen.append(candidates.copy())
        target = 0.04 * (iteration + 1)
        return np.sum((candidates - target) ** 2, axis=(1, 2), dtype=np.float32)

    nominal, history = run_reference_mppi(bundle, evaluate)
    assert len(history) == 2
    assert len(seen) == 2
    np.testing.assert_array_equal(history[0].candidates, bundle.first_iteration_candidates())
    np.testing.assert_array_equal(history[1].candidates, seen[1])
    np.testing.assert_array_equal(nominal, history[-1].update.nominal_actions)
    assert not np.array_equal(seen[1], bundle.perturbations[1])


def test_all_nan_costs_are_rejected() -> None:
    with pytest.raises(ValueError, match="all rollout costs"):
        mppi_softmin_update(
            np.zeros((2, 1, 1), dtype=np.float32),
            np.full(2, np.nan, dtype=np.float32),
            temperature=1.0,
        )
