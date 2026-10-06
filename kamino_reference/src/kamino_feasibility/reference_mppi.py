"""Engine-neutral MPPI update for contact-formulation comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .reference_planning import ReferencePlanningBundle


@dataclass(frozen=True)
class ReferenceMPPIUpdate:
    """One soft-min update shared by every dynamics formulation."""

    nominal_actions: np.ndarray
    first_action: np.ndarray
    weights: np.ndarray
    minimum_cost: float
    mean_valid_cost: float
    unnormalized_weight_sum: float
    effective_sample_size: float
    valid_sample_count: int


@dataclass(frozen=True)
class ReferenceMPPIIteration:
    """Auditable arrays and summary from one planning iteration."""

    iteration: int
    candidates: np.ndarray
    costs: np.ndarray
    update: ReferenceMPPIUpdate


def mppi_softmin_update(
    candidates: np.ndarray,
    costs: np.ndarray,
    *,
    temperature: float,
) -> ReferenceMPPIUpdate:
    """Match the main planner's soft-min weights and weighted sample mean.

    Candidate layout is ``(sample, control_node, actuator)``.  NaN-cost
    rollouts receive zero weight, exactly as in the live Warp kernel.  The
    returned nominal is a full replacement by the weighted mean of the already
    clipped candidates, not an unconstrained accumulated increment.
    """

    candidate_array = np.asarray(candidates, dtype=np.float32)
    cost_array = np.asarray(costs, dtype=np.float32)
    if candidate_array.ndim != 3:
        raise ValueError("candidates must have shape (sample, control, actuator)")
    if cost_array.shape != (candidate_array.shape[0],):
        raise ValueError(
            f"costs must have shape {(candidate_array.shape[0],)}, got {cost_array.shape}"
        )
    if not np.isfinite(candidate_array).all():
        raise ValueError("candidates contains NaN or Inf")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    valid = ~np.isnan(cost_array)
    if not np.any(valid):
        raise ValueError("all rollout costs are NaN")
    if np.any(np.isneginf(cost_array[valid])):
        raise ValueError("rollout costs cannot contain -Inf")
    minimum = np.min(cost_array[valid]).astype(np.float32)
    raw_weights = np.zeros(cost_array.shape, dtype=np.float32)
    raw_weights[valid] = np.exp(
        -(cost_array[valid] - minimum) / np.float32(temperature)
    )
    eta = np.sum(raw_weights, dtype=np.float32)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("soft-min weights are degenerate")
    # Preserve the live kernel's 1e-8 guard in the normalization denominator.
    weights = raw_weights / (eta + np.float32(1.0e-8))
    nominal = np.einsum(
        "n,nhu->hu",
        weights,
        candidate_array,
        dtype=np.float32,
        optimize=False,
    ).astype("<f4", copy=False)
    squared_weight_sum = np.sum(weights * weights, dtype=np.float32)
    effective_sample_size = float(1.0 / squared_weight_sum)
    return ReferenceMPPIUpdate(
        nominal_actions=nominal,
        first_action=nominal[0].copy(),
        weights=weights.astype("<f4", copy=False),
        minimum_cost=float(minimum),
        mean_valid_cost=float(np.mean(cost_array[valid], dtype=np.float32)),
        unnormalized_weight_sum=float(eta),
        effective_sample_size=effective_sample_size,
        valid_sample_count=int(np.count_nonzero(valid)),
    )


def run_reference_mppi(
    bundle: ReferencePlanningBundle,
    evaluate_costs: Callable[[np.ndarray, int], np.ndarray],
) -> tuple[np.ndarray, tuple[ReferenceMPPIIteration, ...]]:
    """Run the bundle's deterministic iterations around one dynamics callback."""

    bundle.validate()
    nominal = bundle.nominal_actions.copy()
    history: list[ReferenceMPPIIteration] = []
    for iteration in range(bundle.spec.n_iterations):
        candidates = bundle.candidates_for_iteration(
            iteration,
            nominal_actions=nominal,
        )
        costs = np.asarray(evaluate_costs(candidates, iteration), dtype="<f4")
        update = mppi_softmin_update(
            candidates,
            costs,
            temperature=bundle.spec.temperature,
        )
        history.append(
            ReferenceMPPIIteration(
                iteration=iteration,
                candidates=candidates.copy(),
                costs=costs.copy(),
                update=update,
            )
        )
        nominal = update.nominal_actions
    return nominal, tuple(history)
