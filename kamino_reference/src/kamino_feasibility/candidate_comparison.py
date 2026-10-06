"""Metrics for two formulations evaluated on an identical candidate tensor."""

from __future__ import annotations

from typing import Any

import numpy as np

from .candidate_evaluation import CandidateEvaluation
from .reference_mppi import mppi_softmin_update
from .reference_planning import ReferencePlanningBundle


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    begin = 0
    while begin < values.size:
        end = begin + 1
        while end < values.size and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end - 1)
        begin = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0.0:
        return float("nan")
    return float((left_centered @ right_centered) / denominator)


def compare_candidate_evaluations(
    bundle: ReferencePlanningBundle,
    reference: CandidateEvaluation,
    comparison: CandidateEvaluation,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compare rankings and MPPI actions without conflating cost scale."""

    bundle.validate()
    reference.validate()
    comparison.validate()
    fingerprint = bundle.fingerprint_sha256()
    for label, evaluation in (
        ("reference", reference),
        ("comparison", comparison),
    ):
        if evaluation.spec.input_fingerprint_sha256 != fingerprint:
            raise ValueError(f"{label} evaluation uses a different input bundle")
        if evaluation.spec.n_samples != bundle.spec.n_samples:
            raise ValueError(f"{label} evaluation has a different sample count")
        if evaluation.spec.action_semantics != bundle.spec.action_semantics:
            raise ValueError(f"{label} evaluation has different action semantics")
        if evaluation.spec.cost_semantics != bundle.spec.cost_semantics:
            raise ValueError(f"{label} evaluation has different cost semantics")
    if top_k < 1 or top_k > bundle.spec.n_samples:
        raise ValueError("top_k must be in [1, n_samples]")

    common_valid = reference.valid & comparison.valid
    indices = np.flatnonzero(common_valid)
    if indices.size < 2:
        raise ValueError("At least two candidates must be valid in both evaluations")
    reference_costs = reference.costs[indices].astype(np.float64)
    comparison_costs = comparison.costs[indices].astype(np.float64)
    reference_order_local = np.argsort(reference_costs, kind="stable")
    comparison_order_local = np.argsort(comparison_costs, kind="stable")
    reference_order = indices[reference_order_local]
    comparison_order = indices[comparison_order_local]
    effective_k = min(top_k, indices.size)
    top_reference = set(reference_order[:effective_k].tolist())
    top_comparison = set(comparison_order[:effective_k].tolist())

    candidates = bundle.first_iteration_candidates()
    reference_update = mppi_softmin_update(
        candidates,
        np.where(reference.valid, reference.costs, np.nan),
        temperature=bundle.spec.temperature,
    )
    comparison_update = mppi_softmin_update(
        candidates,
        np.where(comparison.valid, comparison.costs, np.nan),
        temperature=bundle.spec.temperature,
    )
    action_delta = comparison_update.first_action - reference_update.first_action
    action_denominator = (
        np.linalg.norm(reference_update.first_action)
        * np.linalg.norm(comparison_update.first_action)
    )
    action_cosine = (
        float(reference_update.first_action @ comparison_update.first_action)
        / float(action_denominator)
        if action_denominator > 0.0
        else float("nan")
    )

    reference_best = int(reference_order[0])
    comparison_best = int(comparison_order[0])
    reference_best_cost = float(reference.costs[reference_best])
    comparison_choice_cost_in_reference = float(reference.costs[comparison_best])
    return {
        "input_fingerprint_sha256": fingerprint,
        "reference_formulation": reference.spec.formulation,
        "comparison_formulation": comparison.spec.formulation,
        "common_valid_samples": int(indices.size),
        "n_samples": bundle.spec.n_samples,
        "cost_pearson": _correlation(reference_costs, comparison_costs),
        "rank_spearman": _correlation(
            _average_ranks(reference_costs),
            _average_ranks(comparison_costs),
        ),
        "top_k": effective_k,
        "top_k_overlap_count": len(top_reference & top_comparison),
        "top_k_jaccard": len(top_reference & top_comparison)
        / len(top_reference | top_comparison),
        "reference_best_sample": reference_best,
        "comparison_best_sample": comparison_best,
        "best_candidate_reference_regret": (
            comparison_choice_cost_in_reference - reference_best_cost
        ),
        "best_candidate_reference_regret_note": (
            "Discrete candidate-set regret only; not J_ref of the weighted MPPI action"
        ),
        "reference_ess": reference_update.effective_sample_size,
        "comparison_ess": comparison_update.effective_sample_size,
        "first_action_l2": float(np.linalg.norm(action_delta)),
        "first_action_linf": float(np.max(np.abs(action_delta))),
        "first_action_cosine": action_cosine,
        "full_nominal_l2": float(
            np.linalg.norm(
                comparison_update.nominal_actions
                - reference_update.nominal_actions
            )
        ),
        "reference_first_action": reference_update.first_action.tolist(),
        "comparison_first_action": comparison_update.first_action.tolist(),
    }
