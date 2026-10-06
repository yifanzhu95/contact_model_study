"""Pure-schema tests for the cross-environment reference-planning contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kamino_feasibility.reference_planning import (
    ReferencePlanningSpec,
    load_reference_planning_bundle,
    make_gaussian_reference_bundle,
    save_reference_planning_bundle,
)


def _spec(**overrides) -> ReferencePlanningSpec:
    values = {
        "task": "grasp_reorient",
        "geometry": "cube_high_high",
        "scene_sha256": "1" * 64,
        "cost_source_sha256": "2" * 64,
        "nq": 23,
        "nv": 22,
        "nu": 16,
        "n_samples": 4,
        "n_iterations": 2,
        "control_horizon": 5,
        "physics_steps_per_control": 16,
        "physics_dt_s": 0.004,
        "control_dt_s": 0.064,
        "requested_horizon_s": 0.352,
        "realized_horizon_s": 0.320,
        "state_source": "unit_test",
    }
    values.update(overrides)
    return ReferencePlanningSpec(**values)


def _bundle(seed: int = 7):
    spec = _spec()
    return make_gaussian_reference_bundle(
        spec,
        qpos=np.linspace(-1.0, 1.0, spec.nq),
        qvel=np.zeros(spec.nv),
        current_ctrl=np.linspace(0.0, 1.0, spec.nu),
        goal=np.linspace(-0.5, 0.5, 7 + spec.nu + 1),
        cost_weights=np.arange(1, 13),
        seed=seed,
    )


def test_default_schedule_matches_main_project() -> None:
    spec = _spec()
    spec.validate()
    assert spec.physics_steps_per_control == 16
    assert spec.control_horizon == 5
    assert spec.realized_horizon_s == pytest.approx(0.320)


def test_inconsistent_time_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="control_dt_s must equal"):
        _spec(control_dt_s=0.060).validate()
    with pytest.raises(ValueError, match="floor-quantized"):
        _spec(requested_horizon_s=0.500).validate()


def test_deterministic_candidates_have_canonical_layout_and_clip() -> None:
    first = _bundle(seed=19)
    second = _bundle(seed=19)
    candidates = first.first_iteration_candidates()
    assert candidates.shape == (4, 5, 16)
    assert candidates.dtype == np.dtype("<f4")
    assert np.max(candidates) <= first.spec.action_delta_high
    assert np.min(candidates) >= first.spec.action_delta_low
    np.testing.assert_array_equal(first.perturbations, second.perturbations)
    assert first.fingerprint_sha256() == second.fingerprint_sha256()


def test_round_trip_preserves_exact_fingerprint(tmp_path: Path) -> None:
    bundle = _bundle()
    path = save_reference_planning_bundle(bundle, tmp_path / "reference.npz")
    loaded = load_reference_planning_bundle(path)
    assert loaded.fingerprint_sha256() == bundle.fingerprint_sha256()
    np.testing.assert_array_equal(loaded.perturbations, bundle.perturbations)
    np.testing.assert_array_equal(
        loaded.first_iteration_candidates(), bundle.first_iteration_candidates()
    )


def test_single_candidate_worker_subbatch_is_valid() -> None:
    spec = _spec(n_samples=1)
    bundle = make_gaussian_reference_bundle(
        spec,
        qpos=np.zeros(spec.nq, dtype="<f8"),
        qvel=np.zeros(spec.nv, dtype="<f8"),
        current_ctrl=np.zeros(spec.nu, dtype="<f8"),
        goal=np.zeros(7 + spec.nu + 1, dtype="<f4"),
        cost_weights=np.ones(12, dtype="<f4"),
        seed=7,
    )
    bundle.validate()
    assert bundle.first_iteration_candidates().shape == (1, 5, 16)


def test_tampered_array_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle()
    original = save_reference_planning_bundle(bundle, tmp_path / "reference.npz")
    with np.load(original, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    arrays["perturbations"][0, 0, 0, 0] += np.float32(0.01)
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_reference_planning_bundle(tampered)


def test_manifest_semantic_change_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle()
    original = save_reference_planning_bundle(bundle, tmp_path / "reference.npz")
    with np.load(original, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    manifest = json.loads(str(arrays["manifest_json"].item()))
    manifest["spec"]["action_semantics"] = "absolute_joint_target"
    arrays["manifest_json"] = np.asarray(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), dtype=np.str_
    )
    tampered = tmp_path / "semantic_drift.npz"
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(ValueError, match="Unsupported action semantics"):
        load_reference_planning_bundle(tampered)
