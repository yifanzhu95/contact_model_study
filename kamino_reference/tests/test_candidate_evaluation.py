"""Tests for portable fixed-candidate evaluation artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from kamino_feasibility.candidate_evaluation import (
    CandidateEvaluation,
    CandidateEvaluationSpec,
    load_candidate_evaluation,
    save_candidate_evaluation,
)


def _evaluation() -> CandidateEvaluation:
    spec = CandidateEvaluationSpec(
        input_fingerprint_sha256="a" * 64,
        formulation="mujoco_soft_constraint",
        engine="mujoco",
        engine_version="test",
        scene_sha256="b" * 64,
        n_samples=2,
        nq=3,
        nv=2,
        control_horizon=4,
        physics_steps_per_control=5,
        physics_dt_s=0.01,
        action_semantics="relative",
        cost_semantics="cost-v1",
        validity_semantics="finite_state",
    )
    return CandidateEvaluation(
        spec=spec,
        costs=np.array([1.0, 2.0], dtype="<f4"),
        final_qpos=np.arange(6, dtype="<f8").reshape(2, 3),
        final_qvel=np.arange(4, dtype="<f8").reshape(2, 2),
        valid=np.array([True, True], dtype=bool),
        contact_fine_steps=np.array([3, 4], dtype="<i4"),
        max_contacts=np.array([1, 2], dtype="<i4"),
        max_penetration_m=np.array([0.0, 1.0e-5], dtype="<f8"),
    )


def test_exact_roundtrip(tmp_path) -> None:
    original = _evaluation()
    path = save_candidate_evaluation(original, tmp_path / "result.npz")
    loaded = load_candidate_evaluation(path)
    assert loaded.spec == original.spec
    assert loaded.fingerprint_sha256() == original.fingerprint_sha256()
    for name in (
        "costs",
        "final_qpos",
        "final_qvel",
        "valid",
        "contact_fine_steps",
        "max_contacts",
        "max_penetration_m",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(original, name))


def test_tampered_cost_is_rejected(tmp_path) -> None:
    path = save_candidate_evaluation(_evaluation(), tmp_path / "result.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["costs"][0] += 1.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_candidate_evaluation(path)


def test_unknown_manifest_field_is_rejected(tmp_path) -> None:
    path = save_candidate_evaluation(_evaluation(), tmp_path / "result.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    manifest = json.loads(str(arrays["manifest"].item()))
    manifest["spec"]["surprise"] = 1
    arrays["manifest"] = np.asarray(json.dumps(manifest))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="Unknown evaluation-spec"):
        load_candidate_evaluation(path)
