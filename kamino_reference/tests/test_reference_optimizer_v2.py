"""CPU-only tests for repeat-aggregated reference optimization."""

from __future__ import annotations

import numpy as np
import pytest

from kamino_feasibility.reference_mppi import mppi_softmin_update
from kamino_feasibility.reference_optimizer_v2 import (
    aggregate_repeated_candidate_costs,
    bootstrap_repeat_aggregated_first_action,
    repeat_aggregated_mppi_update,
)


def test_costs_are_aggregated_before_the_nonlinear_update() -> None:
    candidates = np.array([[[0.0]], [[1.0]]], dtype=np.float32)
    costs = np.array([[0.0, 10.0], [2.0, 0.0]], dtype=np.float64)

    result = repeat_aggregated_mppi_update(
        candidates,
        costs,
        temperature=1.0,
    )
    per_repeat_actions = np.array(
        [
            mppi_softmin_update(candidates, row, temperature=1.0).first_action
            for row in costs
        ]
    )

    np.testing.assert_allclose(result.aggregation.aggregated_costs, [1.0, 5.0])
    assert result.update.first_action[0] == pytest.approx(0.01798621, abs=1e-7)
    assert abs(
        float(result.update.first_action[0])
        - float(np.mean(per_repeat_actions[:, 0]))
    ) > 0.4


def test_aggregation_reports_repeat_statistics() -> None:
    result = aggregate_repeated_candidate_costs(
        np.array([[1.0, 8.0], [2.0, 6.0], [3.0, 7.0]])
    )
    assert result.method == "arithmetic_mean_float64_then_float32"
    np.testing.assert_array_equal(result.aggregated_costs, [2.0, 7.0])
    np.testing.assert_array_equal(result.median_costs, [2.0, 7.0])
    np.testing.assert_allclose(result.sample_standard_deviation, [1.0, 1.0])
    np.testing.assert_array_equal(result.cost_ranges, [2.0, 2.0])


def test_failed_repeat_is_not_silently_dropped() -> None:
    with pytest.raises(ValueError, match="strict-valid and finite"):
        aggregate_repeated_candidate_costs(
            np.array([[1.0, 2.0], [1.1, np.nan]]),
            strict_valid=np.array([[True, True], [True, False]]),
        )


def test_bootstrap_is_deterministic_and_zero_for_identical_rows() -> None:
    candidates = np.array([[[0.0]], [[1.0]], [[2.0]]], dtype=np.float32)
    costs = np.tile(np.array([[0.0, 1.0, 2.0]]), (5, 1))
    left = bootstrap_repeat_aggregated_first_action(
        candidates,
        costs,
        temperature=1.0,
        draw_count=50,
        seed=17,
    )
    right = bootstrap_repeat_aggregated_first_action(
        candidates,
        costs,
        temperature=1.0,
        draw_count=50,
        seed=17,
    )
    np.testing.assert_array_equal(left.lower_first_action, right.lower_first_action)
    np.testing.assert_array_equal(left.upper_first_action, right.upper_first_action)
    np.testing.assert_allclose(left.lower_first_action, left.point_first_action)
    np.testing.assert_allclose(left.upper_first_action, left.point_first_action)
    assert left.maximum_l2_from_point == pytest.approx(0.0)
    assert left.maximum_linf_from_point == pytest.approx(0.0)


def test_bootstrap_rejects_invalid_configuration() -> None:
    candidates = np.zeros((2, 1, 1), dtype=np.float32)
    costs = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="draw_count"):
        bootstrap_repeat_aggregated_first_action(
            candidates, costs, temperature=1.0, draw_count=1
        )
