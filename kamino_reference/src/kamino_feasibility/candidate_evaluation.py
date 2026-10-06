"""Portable fixed-candidate rollout results for formulation comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATE_EVALUATION_SCHEMA = "kamino_feasibility.candidate_evaluation.v1"


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class CandidateEvaluationSpec:
    """Non-array provenance for one formulation's fixed-candidate costs."""

    input_fingerprint_sha256: str
    formulation: str
    engine: str
    engine_version: str
    scene_sha256: str
    n_samples: int
    nq: int
    nv: int
    control_horizon: int
    physics_steps_per_control: int
    physics_dt_s: float
    action_semantics: str
    cost_semantics: str
    validity_semantics: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not _is_sha256(self.input_fingerprint_sha256):
            raise ValueError("input_fingerprint_sha256 must be a SHA-256 digest")
        if not _is_sha256(self.scene_sha256):
            raise ValueError("scene_sha256 must be a SHA-256 digest")
        for name in (
            "formulation",
            "engine",
            "engine_version",
            "action_semantics",
            "cost_semantics",
            "validity_semantics",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "n_samples",
            "nq",
            "nv",
            "control_horizon",
            "physics_steps_per_control",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(self.physics_dt_s) or self.physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be finite and positive")
        try:
            _canonical_json(self.metadata)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be finite JSON data") from error

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateEvaluationSpec":
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown evaluation-spec fields: {sorted(unknown)}")
        result = cls(**payload)
        result.validate()
        return result


@dataclass(frozen=True)
class CandidateEvaluation:
    """One exact cost per candidate plus compact rollout diagnostics."""

    spec: CandidateEvaluationSpec
    costs: np.ndarray
    final_qpos: np.ndarray
    final_qvel: np.ndarray
    valid: np.ndarray
    contact_fine_steps: np.ndarray
    max_contacts: np.ndarray
    max_penetration_m: np.ndarray

    def validate(self) -> None:
        self.spec.validate()
        expected = {
            "costs": ((self.spec.n_samples,), np.dtype("<f4")),
            "final_qpos": (
                (self.spec.n_samples, self.spec.nq),
                np.dtype("<f8"),
            ),
            "final_qvel": (
                (self.spec.n_samples, self.spec.nv),
                np.dtype("<f8"),
            ),
            "valid": ((self.spec.n_samples,), np.dtype("bool")),
            "contact_fine_steps": (
                (self.spec.n_samples,),
                np.dtype("<i4"),
            ),
            "max_contacts": ((self.spec.n_samples,), np.dtype("<i4")),
            "max_penetration_m": (
                (self.spec.n_samples,),
                np.dtype("<f8"),
            ),
        }
        for name, (shape, dtype) in expected.items():
            array = np.asarray(getattr(self, name))
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
            if array.dtype != dtype:
                raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}")
        if np.any(self.contact_fine_steps < 0) or np.any(self.max_contacts < 0):
            raise ValueError("contact counts must be non-negative")
        finite_penetration = np.isfinite(self.max_penetration_m)
        if np.any(np.isinf(self.max_penetration_m)):
            raise ValueError("max_penetration_m cannot contain Inf")
        if np.any(self.max_penetration_m[finite_penetration] < 0.0):
            raise ValueError("max_penetration_m must be non-negative")
        if not np.isfinite(self.final_qpos[self.valid]).all():
            raise ValueError("valid final_qpos contains NaN or Inf")
        if not np.isfinite(self.final_qvel[self.valid]).all():
            raise ValueError("valid final_qvel contains NaN or Inf")
        if not np.isfinite(self.costs[self.valid]).all():
            raise ValueError("valid costs contains NaN or Inf")
        # NaN means that this engine did not expose a comparable penetration
        # diagnostic.  It is preserved explicitly rather than replaced by zero.

    def fingerprint_sha256(self) -> str:
        self.validate()
        digest = hashlib.sha256()
        digest.update(CANDIDATE_EVALUATION_SCHEMA.encode("utf-8"))
        digest.update(_canonical_json(self.spec.to_dict()).encode("utf-8"))
        for name in (
            "costs",
            "final_qpos",
            "final_qvel",
            "valid",
            "contact_fine_steps",
            "max_contacts",
            "max_penetration_m",
        ):
            array = np.ascontiguousarray(getattr(self, name))
            digest.update(name.encode("utf-8"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()


def save_candidate_evaluation(
    evaluation: CandidateEvaluation,
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.suffix != ".npz":
        raise ValueError("Candidate evaluation path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": CANDIDATE_EVALUATION_SCHEMA,
        "spec": evaluation.spec.to_dict(),
        "fingerprint_sha256": evaluation.fingerprint_sha256(),
    }
    np.savez_compressed(
        destination,
        manifest=np.asarray(_canonical_json(manifest)),
        costs=evaluation.costs,
        final_qpos=evaluation.final_qpos,
        final_qvel=evaluation.final_qvel,
        valid=evaluation.valid,
        contact_fine_steps=evaluation.contact_fine_steps,
        max_contacts=evaluation.max_contacts,
        max_penetration_m=evaluation.max_penetration_m,
    )
    return destination


def load_candidate_evaluation(path: str | Path) -> CandidateEvaluation:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        expected_names = {
            "manifest",
            "costs",
            "final_qpos",
            "final_qvel",
            "valid",
            "contact_fine_steps",
            "max_contacts",
            "max_penetration_m",
        }
        if set(archive.files) != expected_names:
            raise ValueError("Candidate evaluation has unexpected array names")
        manifest = json.loads(str(archive["manifest"].item()))
        if manifest.get("schema") != CANDIDATE_EVALUATION_SCHEMA:
            raise ValueError("Unsupported candidate-evaluation schema")
        evaluation = CandidateEvaluation(
            spec=CandidateEvaluationSpec.from_dict(manifest["spec"]),
            costs=archive["costs"].copy(),
            final_qpos=archive["final_qpos"].copy(),
            final_qvel=archive["final_qvel"].copy(),
            valid=archive["valid"].copy(),
            contact_fine_steps=archive["contact_fine_steps"].copy(),
            max_contacts=archive["max_contacts"].copy(),
            max_penetration_m=archive["max_penetration_m"].copy(),
        )
    evaluation.validate()
    expected_fingerprint = manifest.get("fingerprint_sha256")
    if evaluation.fingerprint_sha256() != expected_fingerprint:
        raise ValueError("Candidate evaluation fingerprint mismatch")
    return evaluation
