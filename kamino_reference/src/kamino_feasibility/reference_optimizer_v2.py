"""Repeat-aggregated MPPI semantics for the offline Kamino reference.

The important ordering is::

    repeated candidate costs -> aggregate each candidate -> one MPPI update

It is intentionally *not* ``MPPI update per repeat -> average actions`` because
soft-min weighting is nonlinear in cost. Keeping this operation engine-neutral
also lets the coordinator be tested without starting CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference_mppi import ReferenceMPPIUpdate, mppi_softmin_update


@dataclass(frozen=True)
class RepeatCostAggregation:
    """Per-candidate numerical-repeat statistics used by one MPPI update."""

    method: str
    replicate_count: int
    candidate_count: int
    aggregated_costs: np.ndarray
    mean_costs: np.ndarray
    median_costs: np.ndarray
    sample_standard_deviation: np.ndarray
    standard_error: np.ndarray
    minimum_costs: np.ndarray
    maximum_costs: np.ndarray
    cost_ranges: np.ndarray


@dataclass(frozen=True)
class RepeatAggregatedMPPIUpdate:
    """One update and the cost-level evidence from which it was produced."""

    aggregation: RepeatCostAggregation
    update: ReferenceMPPIUpdate


@dataclass(frozen=True)
class FirstActionBootstrap:
    """Repeat-row bootstrap diagnostics for the aggregated first action.

    The lower/upper arrays are component-wise percentile intervals. They are
    empirical numerical-repeat diagnostics, not real-world uncertainty bounds.
    """

    draw_count: int
    seed: int
    confidence: float
    point_first_action: np.ndarray
    lower_first_action: np.ndarray
    median_first_action: np.ndarray
    upper_first_action: np.ndarray
    maximum_l2_from_point: float
    maximum_linf_from_point: float


def aggregate_repeated_candidate_costs(
    costs_by_replicate: np.ndarray,
    *,
    strict_valid: np.ndarray | None = None,
    minimum_replicates: int = 2,
) -> RepeatCostAggregation:
    """Strictly aggregate an ``(repeat, candidate)`` cost matrix by mean.

    Arithmetic mean is the frozen primary estimator for Reference Optimizer
    v2. Median and spread are returned only as diagnostics. The routine does
    not silently drop a failed repeat: every candidate/repeat cell must be
    marked valid and finite before an action can be generated.
    """

    costs = np.asarray(costs_by_replicate, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("costs_by_replicate must have shape (repeat, candidate)")
    repeats, candidates = costs.shape
    if repeats < minimum_replicates:
        raise ValueError(
            f"at least {minimum_replicates} repeats are required, got {repeats}"
        )
    if candidates < 1:
        raise ValueError("at least one candidate is required")

    if strict_valid is None:
        valid = np.ones(costs.shape, dtype=bool)
    else:
        valid = np.asarray(strict_valid, dtype=bool)
        if valid.shape != costs.shape:
            raise ValueError(
                f"strict_valid must have shape {costs.shape}, got {valid.shape}"
            )
    accepted = valid & np.isfinite(costs)
    if not np.all(accepted):
        failed = np.argwhere(~accepted)
        preview = failed[:8].tolist()
        raise ValueError(
            "all repeated candidate evaluations must be strict-valid and "
            f"finite; failed cells begin with {preview}"
        )

    mean = np.mean(costs, axis=0, dtype=np.float64)
    median = np.median(costs, axis=0)
    sample_std = np.std(costs, axis=0, ddof=1, dtype=np.float64)
    standard_error = sample_std / np.sqrt(float(repeats))
    minimum = np.min(costs, axis=0)
    maximum = np.max(costs, axis=0)
    return RepeatCostAggregation(
        method="arithmetic_mean_float64_then_float32",
        replicate_count=repeats,
        candidate_count=candidates,
        aggregated_costs=mean.astype("<f4"),
        mean_costs=mean,
        median_costs=median,
        sample_standard_deviation=sample_std,
        standard_error=standard_error,
        minimum_costs=minimum,
        maximum_costs=maximum,
        cost_ranges=maximum - minimum,
    )


def repeat_aggregated_mppi_update(
    candidates: np.ndarray,
    costs_by_replicate: np.ndarray,
    *,
    temperature: float,
    strict_valid: np.ndarray | None = None,
    minimum_replicates: int = 2,
) -> RepeatAggregatedMPPIUpdate:
    """Aggregate candidate costs first, then perform exactly one MPPI update."""

    aggregation = aggregate_repeated_candidate_costs(
        costs_by_replicate,
        strict_valid=strict_valid,
        minimum_replicates=minimum_replicates,
    )
    update = mppi_softmin_update(
        candidates,
        aggregation.aggregated_costs,
        temperature=temperature,
    )
    return RepeatAggregatedMPPIUpdate(
        aggregation=aggregation,
        update=update,
    )


def bootstrap_repeat_aggregated_first_action(
    candidates: np.ndarray,
    costs_by_replicate: np.ndarray,
    *,
    temperature: float,
    draw_count: int = 2000,
    seed: int = 3901,
    confidence: float = 0.95,
    strict_valid: np.ndarray | None = None,
) -> FirstActionBootstrap:
    """Bootstrap repeat rows while preserving aggregate-then-update ordering."""

    if draw_count < 2:
        raise ValueError("draw_count must be at least two")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    costs = np.asarray(costs_by_replicate, dtype=np.float64)
    point = repeat_aggregated_mppi_update(
        candidates,
        costs,
        temperature=temperature,
        strict_valid=strict_valid,
    )
    repeats = costs.shape[0]
    generator = np.random.default_rng(seed)
    selections = generator.integers(
        0,
        repeats,
        size=(draw_count, repeats),
        endpoint=False,
    )
    samples = np.empty(
        (draw_count, point.update.first_action.size), dtype=np.float64
    )
    for draw, rows in enumerate(selections):
        sampled_update = repeat_aggregated_mppi_update(
            candidates,
            costs[rows],
            temperature=temperature,
        )
        samples[draw] = sampled_update.update.first_action

    alpha = (1.0 - confidence) / 2.0
    delta = samples - point.update.first_action.astype(np.float64)
    return FirstActionBootstrap(
        draw_count=draw_count,
        seed=seed,
        confidence=confidence,
        point_first_action=point.update.first_action.copy(),
        lower_first_action=np.quantile(samples, alpha, axis=0),
        median_first_action=np.quantile(samples, 0.5, axis=0),
        upper_first_action=np.quantile(samples, 1.0 - alpha, axis=0),
        maximum_l2_from_point=float(np.max(np.linalg.norm(delta, axis=1))),
        maximum_linf_from_point=float(np.max(np.abs(delta))),
    )
