"""Tests for fixed-candidate formulation comparison metrics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kamino_feasibility.candidate_comparison import compare_candidate_evaluations
from kamino_feasibility.candidate_evaluation import (
    CandidateEvaluation,
    CandidateEvaluationSpec,
)
from kamino_feasibility.reference_planning import (
    ReferencePlanningSpec,
    make_gaussian_reference_bundle,
)


def _inputs():
    planning_spec = ReferencePlanningSpec(
        task="task",
        geometry="geometry",
        scene_sha256="a" * 64,
        cost_source_sha256="b" * 64,
        nq=3,
        nv=2,
        nu=2,
        n_samples=4,
        n_iterations=1,
        control_horizon=2,
        physics_steps_per_control=1,
        physics_dt_s=0.01,
        control_dt_s=0.01,
        requested_horizon_s=0.02,
        realized_horizon_s=0.02,
        temperature=1.0,
        noise_sigma=0.1,
    )
    bundle = make_gaussian_reference_bundle(
        planning_spec,
        qpos=np.zeros(3),
        qvel=np.zeros(2),
        current_ctrl=np.zeros(2),
        goal=np.zeros(10, dtype=np.float32),
        cost_weights=np.zeros(12, dtype=np.float32),
        seed=2,
    )
    evaluation_spec = CandidateEvaluationSpec(
        input_fingerprint_sha256=bundle.fingerprint_sha256(),
        formulation="reference",
        engine="engine",
        engine_version="1",
        scene_sha256="a" * 64,
        n_samples=4,
        nq=3,
        nv=2,
        control_horizon=2,
        physics_steps_per_control=1,
        physics_dt_s=0.01,
        action_semantics=planning_spec.action_semantics,
        cost_semantics=planning_spec.cost_semantics,
        validity_semantics="finite",
    )

    def evaluation(costs, formulation="reference"):
        return CandidateEvaluation(
            spec=replace(evaluation_spec, formulation=formulation),
            costs=np.asarray(costs, dtype="<f4"),
            final_qpos=np.zeros((4, 3), dtype="<f8"),
            final_qvel=np.zeros((4, 2), dtype="<f8"),
            valid=np.ones(4, dtype=bool),
            contact_fine_steps=np.zeros(4, dtype="<i4"),
            max_contacts=np.zeros(4, dtype="<i4"),
            max_penetration_m=np.zeros(4, dtype="<f8"),
        )

    return bundle, evaluation


def test_self_comparison_is_identity() -> None:
    bundle, make_evaluation = _inputs()
    evaluation = make_evaluation([3, 1, 4, 2])
    report = compare_candidate_evaluations(
        bundle, evaluation, evaluation, top_k=2
    )
    assert report["cost_pearson"] == pytest.approx(1.0)
    assert report["rank_spearman"] == pytest.approx(1.0)
    assert report["top_k_jaccard"] == pytest.approx(1.0)
    assert report["best_candidate_reference_regret"] == pytest.approx(0.0)
    assert report["first_action_l2"] == pytest.approx(0.0)


def test_reversed_ranking_reports_disagreement() -> None:
    bundle, make_evaluation = _inputs()
    reference = make_evaluation([0, 1, 2, 3])
    comparison = make_evaluation([3, 2, 1, 0], formulation="comparison")
    report = compare_candidate_evaluations(
        bundle, reference, comparison, top_k=1
    )
    assert report["rank_spearman"] == pytest.approx(-1.0)
    assert report["top_k_overlap_count"] == 0
    assert report["best_candidate_reference_regret"] == pytest.approx(3.0)


def test_different_input_fingerprint_is_rejected() -> None:
    bundle, make_evaluation = _inputs()
    reference = make_evaluation([0, 1, 2, 3])
    comparison = replace(
        make_evaluation([0, 1, 2, 3], formulation="comparison"),
        spec=replace(reference.spec, input_fingerprint_sha256="c" * 64),
    )
    with pytest.raises(ValueError, match="different input bundle"):
        compare_candidate_evaluations(bundle, reference, comparison)
