"""CPU-only checks for the repository-level Kamino comparison bridge."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from contact_study.evaluation.reference_bridge import (
    enable_reference_protocol,
    reference_paths,
)
from contact_study.evaluation.reference_optimizer import (
    FormulationOptimizerResult,
    load_formulation_result,
    save_formulation_result,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_reference_bundle_is_self_contained() -> None:
    paths = reference_paths()
    enable_reference_protocol()
    from kamino_feasibility.reference_planning import (
        load_reference_planning_bundle,
    )

    index = json.loads(paths["inputs_index"].read_text(encoding="utf-8"))
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    bundle_path = paths["root"] / index["development"]["path"]
    bundle = load_reference_planning_bundle(bundle_path)

    assert contract["status"] == "FROZEN_FOR_COLLABORATOR_SCALING"
    assert _sha256(paths["contract"]) == index["contract_sha256"]
    assert _sha256(bundle_path) == index["development"]["file_sha256"]
    assert (
        bundle.fingerprint_sha256()
        == index["development"]["fingerprint_sha256"]
    )


def test_formulation_result_round_trip(tmp_path) -> None:
    iterations, samples, horizon, nu, nq, nv = 1, 2, 3, 4, 5, 4
    result = FormulationOptimizerResult(
        metadata={
            "schema": "contact_study.formulation_optimizer_result.v1",
            "iterations": iterations,
            "samples": samples,
            "control_horizon": horizon,
            "nu": nu,
            "nq": nq,
            "nv": nv,
        },
        nominal_before=np.zeros((iterations, horizon, nu), dtype=np.float32),
        candidates=np.zeros(
            (iterations, samples, horizon, nu), dtype=np.float32
        ),
        costs=np.asarray([[1.0, 2.0]], dtype=np.float32),
        valid=np.ones((iterations, samples), dtype=bool),
        weights=np.asarray([[0.75, 0.25]], dtype=np.float32),
        nominal_after=np.ones((iterations, horizon, nu), dtype=np.float32),
        first_action=np.ones((iterations, nu), dtype=np.float32),
        final_qpos=np.zeros((iterations, samples, nq), dtype=np.float32),
        final_qvel=np.zeros((iterations, samples, nv), dtype=np.float32),
        rollout_wall_s=np.asarray([0.1], dtype=np.float64),
    )

    output = save_formulation_result(result, tmp_path / "M1.npz")
    restored = load_formulation_result(output)

    assert restored.metadata == result.metadata
    for field in (
        "nominal_before",
        "candidates",
        "costs",
        "valid",
        "weights",
        "nominal_after",
        "first_action",
        "final_qpos",
        "final_qvel",
        "rollout_wall_s",
    ):
        assert np.array_equal(getattr(restored, field), getattr(result, field))
