"""Portable, engine-neutral contract for full-NCP action-reference studies.

The Newton/Kamino and contact_model_study environments intentionally use
incompatible Warp/MuJoCo versions.  This module therefore depends only on
NumPy and the Python standard library.  Both environments can load the same
state, goal, cost, noise, and timing payload without importing one another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REFERENCE_PLANNING_SCHEMA = "kamino_feasibility.reference_planning.v1"
ACTION_SEMANTICS = "relative_qpos_delta_each_control_node"
COST_SEMANTICS = "contact_study.grasp_reorient.v1"
COST_ACCUMULATION = "running_nodes_0_to_h_minus_2_terminal_only_at_h_minus_1"


def file_sha256(path: str | Path) -> str:
    """Return the content hash of one source or scene file."""

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class ReferencePlanningSpec:
    """All non-array choices that must match across contact formulations."""

    task: str
    geometry: str
    scene_sha256: str
    cost_source_sha256: str
    nq: int
    nv: int
    nu: int
    n_samples: int
    n_iterations: int
    control_horizon: int
    physics_steps_per_control: int
    physics_dt_s: float
    control_dt_s: float
    requested_horizon_s: float
    realized_horizon_s: float
    action_semantics: str = ACTION_SEMANTICS
    candidate_layout: str = "sample_control_actuator"
    cost_semantics: str = COST_SEMANTICS
    cost_accumulation: str = COST_ACCUMULATION
    planner: str = "mppi"
    temperature: float = 10.0
    noise_sigma: float = 0.1
    action_delta_low: float = -0.1
    action_delta_high: float = 0.1
    reference_formulation: str = "kamino_full_ncp"
    state_source: str = "unspecified"
    main_repo_commit: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.task or not self.geometry:
            raise ValueError("task and geometry must be non-empty")
        if not _is_sha256(self.scene_sha256):
            raise ValueError("scene_sha256 must be a lowercase SHA-256 hex digest")
        if not _is_sha256(self.cost_source_sha256):
            raise ValueError("cost_source_sha256 must be a lowercase SHA-256 hex digest")
        for name in (
            "nq",
            "nv",
            "nu",
            "n_samples",
            "n_iterations",
            "control_horizon",
            "physics_steps_per_control",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        # A portable bundle may be a worker sub-batch containing one member of
        # a larger MPPI candidate set.  The outer planner still owns the full
        # multi-candidate update; the worker bundle only evaluates dynamics.
        if self.n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if self.action_semantics != ACTION_SEMANTICS:
            raise ValueError(f"Unsupported action semantics {self.action_semantics!r}")
        if self.candidate_layout != "sample_control_actuator":
            raise ValueError(f"Unsupported candidate layout {self.candidate_layout!r}")
        if self.cost_semantics != COST_SEMANTICS:
            raise ValueError(f"Unsupported cost semantics {self.cost_semantics!r}")
        if self.cost_accumulation != COST_ACCUMULATION:
            raise ValueError(f"Unsupported cost accumulation {self.cost_accumulation!r}")
        if self.planner != "mppi":
            raise ValueError(f"Unsupported planner {self.planner!r}")
        for name in (
            "physics_dt_s",
            "control_dt_s",
            "requested_horizon_s",
            "realized_horizon_s",
            "temperature",
            "noise_sigma",
            "action_delta_low",
            "action_delta_high",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.physics_dt_s <= 0.0 or self.control_dt_s <= 0.0:
            raise ValueError("physics_dt_s and control_dt_s must be positive")
        if self.temperature <= 0.0 or self.noise_sigma <= 0.0:
            raise ValueError("temperature and noise_sigma must be positive")
        if self.action_delta_low >= self.action_delta_high:
            raise ValueError("action_delta_low must be below action_delta_high")
        expected_control_dt = self.physics_dt_s * self.physics_steps_per_control
        if not np.isclose(self.control_dt_s, expected_control_dt, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "control_dt_s must equal physics_dt_s * physics_steps_per_control"
            )
        expected_realized = self.control_dt_s * self.control_horizon
        if not np.isclose(self.realized_horizon_s, expected_realized, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "realized_horizon_s must equal control_dt_s * control_horizon"
            )
        if self.requested_horizon_s + 1.0e-12 < self.realized_horizon_s:
            raise ValueError("requested_horizon_s cannot be shorter than the realized horizon")
        if self.requested_horizon_s >= self.realized_horizon_s + self.control_dt_s + 1.0e-12:
            raise ValueError("control_horizon is not the floor-quantized requested horizon")
        try:
            _canonical_json(self.metadata)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be finite JSON data") from error

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferencePlanningSpec":
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown planning-spec fields: {sorted(unknown)}")
        # Dataclass construction provides the useful missing-required-field error.
        spec = cls(**payload)
        spec.validate()
        return spec


@dataclass(frozen=True)
class ReferencePlanningBundle:
    """Exact planning input shared by Kamino and every relaxed simulator."""

    spec: ReferencePlanningSpec
    qpos: np.ndarray
    qvel: np.ndarray
    current_ctrl: np.ndarray
    goal: np.ndarray
    cost_weights: np.ndarray
    nominal_actions: np.ndarray
    perturbations: np.ndarray

    def validate(self) -> None:
        self.spec.validate()
        expected = {
            "qpos": ((self.spec.nq,), np.dtype("<f8")),
            "qvel": ((self.spec.nv,), np.dtype("<f8")),
            "current_ctrl": ((self.spec.nu,), np.dtype("<f8")),
            "goal": ((7 + self.spec.nu + 1,), np.dtype("<f4")),
            "cost_weights": ((12,), np.dtype("<f4")),
            "nominal_actions": (
                (self.spec.control_horizon, self.spec.nu),
                np.dtype("<f4"),
            ),
            "perturbations": (
                (
                    self.spec.n_iterations,
                    self.spec.n_samples,
                    self.spec.control_horizon,
                    self.spec.nu,
                ),
                np.dtype("<f4"),
            ),
        }
        for name, (shape, dtype) in expected.items():
            array = np.asarray(getattr(self, name))
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
            if array.dtype != dtype:
                raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains NaN or Inf")

    def first_iteration_candidates(self) -> np.ndarray:
        """Return canonical ``(sample, control, actuator)`` delta actions."""

        return self.candidates_for_iteration(0)

    def candidates_for_iteration(
        self,
        iteration: int,
        *,
        nominal_actions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one canonical candidate block around the supplied mean."""

        self.validate()
        if iteration < 0 or iteration >= self.spec.n_iterations:
            raise IndexError(
                f"iteration must be in [0, {self.spec.n_iterations}), got {iteration}"
            )
        nominal = (
            self.nominal_actions
            if nominal_actions is None
            else np.asarray(nominal_actions, dtype="<f4")
        )
        expected = (self.spec.control_horizon, self.spec.nu)
        if nominal.shape != expected:
            raise ValueError(f"nominal_actions must have shape {expected}")
        if not np.isfinite(nominal).all():
            raise ValueError("nominal_actions contains NaN or Inf")
        return np.clip(
            nominal[np.newaxis, :, :] + self.perturbations[iteration],
            self.spec.action_delta_low,
            self.spec.action_delta_high,
        ).astype("<f4", copy=False)

    def fingerprint_sha256(self) -> str:
        """Hash semantic choices, exact dtypes/shapes, and every array byte."""

        self.validate()
        digest = hashlib.sha256()
        digest.update(REFERENCE_PLANNING_SCHEMA.encode("utf-8"))
        digest.update(_canonical_json(self.spec.to_dict()).encode("utf-8"))
        for name in (
            "qpos",
            "qvel",
            "current_ctrl",
            "goal",
            "cost_weights",
            "nominal_actions",
            "perturbations",
        ):
            array = np.ascontiguousarray(getattr(self, name))
            digest.update(name.encode("utf-8"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()


def make_gaussian_reference_bundle(
    spec: ReferencePlanningSpec,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    current_ctrl: np.ndarray,
    goal: np.ndarray,
    cost_weights: np.ndarray,
    seed: int,
    nominal_actions: np.ndarray | None = None,
) -> ReferencePlanningBundle:
    """Create deterministic MPPI perturbations outside either physics engine."""

    spec.validate()
    nominal = (
        np.zeros((spec.control_horizon, spec.nu), dtype="<f4")
        if nominal_actions is None
        else np.asarray(nominal_actions, dtype="<f4")
    )
    rng = np.random.default_rng(seed)
    perturbations = rng.normal(
        0.0,
        spec.noise_sigma,
        size=(
            spec.n_iterations,
            spec.n_samples,
            spec.control_horizon,
            spec.nu,
        ),
    ).astype("<f4")
    bundle = ReferencePlanningBundle(
        spec=spec,
        qpos=np.asarray(qpos, dtype="<f8"),
        qvel=np.asarray(qvel, dtype="<f8"),
        current_ctrl=np.asarray(current_ctrl, dtype="<f8"),
        goal=np.asarray(goal, dtype="<f4"),
        cost_weights=np.asarray(cost_weights, dtype="<f4"),
        nominal_actions=nominal,
        perturbations=perturbations,
    )
    bundle.validate()
    return bundle


def save_reference_planning_bundle(
    bundle: ReferencePlanningBundle,
    path: str | Path,
) -> Path:
    """Save one portable NPZ with an internal semantic fingerprint."""

    destination = Path(path).expanduser().resolve()
    if destination.suffix != ".npz":
        raise ValueError("Reference planning bundle path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": REFERENCE_PLANNING_SCHEMA,
        "spec": bundle.spec.to_dict(),
        "fingerprint_sha256": bundle.fingerprint_sha256(),
    }
    np.savez_compressed(
        destination,
        manifest_json=np.asarray(_canonical_json(manifest), dtype=np.str_),
        qpos=bundle.qpos,
        qvel=bundle.qvel,
        current_ctrl=bundle.current_ctrl,
        goal=bundle.goal,
        cost_weights=bundle.cost_weights,
        nominal_actions=bundle.nominal_actions,
        perturbations=bundle.perturbations,
    )
    return destination


def load_reference_planning_bundle(path: str | Path) -> ReferencePlanningBundle:
    """Load a bundle without pickle and reject corruption or semantic drift."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        manifest = json.loads(str(payload["manifest_json"].item()))
        if manifest.get("schema") != REFERENCE_PLANNING_SCHEMA:
            raise ValueError(f"Unsupported reference-planning schema {manifest.get('schema')!r}")
        bundle = ReferencePlanningBundle(
            spec=ReferencePlanningSpec.from_dict(dict(manifest.get("spec") or {})),
            qpos=payload["qpos"].copy(),
            qvel=payload["qvel"].copy(),
            current_ctrl=payload["current_ctrl"].copy(),
            goal=payload["goal"].copy(),
            cost_weights=payload["cost_weights"].copy(),
            nominal_actions=payload["nominal_actions"].copy(),
            perturbations=payload["perturbations"].copy(),
        )
    expected = manifest.get("fingerprint_sha256")
    actual = bundle.fingerprint_sha256()
    if actual != expected:
        raise ValueError("Reference planning bundle fingerprint mismatch")
    return bundle
