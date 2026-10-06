"""Evaluate a frozen MPPI candidate tape under one M1--M4 formulation.

This module reuses the production sampling planner's GPU reset, control,
physics-step and cost kernels, but bypasses its random sampler.  The exact
candidate tensor comes from the engine-neutral Kamino reference bundle, so all
formulations see identical state, goal, timing, control and cost semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401  (register task implementations)
from contact_study.tasks import grasp_reorient as grasp_reorient_task_module
from contact_study.contact_models.config import ContactModelConfig
from contact_study.planners.mppi import MPPIConfig, MPPIController
from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole

from .reference_bridge import enable_reference_protocol


RESULT_SCHEMA = "contact_study.formulation_optimizer_result.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FormulationOptimizerResult:
    """Auditable multi-iteration result for one contact formulation."""

    metadata: dict[str, Any]
    nominal_before: np.ndarray
    candidates: np.ndarray
    costs: np.ndarray
    valid: np.ndarray
    weights: np.ndarray
    nominal_after: np.ndarray
    first_action: np.ndarray
    final_qpos: np.ndarray
    final_qvel: np.ndarray
    rollout_wall_s: np.ndarray

    def validate(self) -> None:
        if self.metadata.get("schema") != RESULT_SCHEMA:
            raise ValueError("unsupported formulation optimizer result schema")
        iterations = int(self.metadata["iterations"])
        samples = int(self.metadata["samples"])
        horizon = int(self.metadata["control_horizon"])
        nu = int(self.metadata["nu"])
        nq = int(self.metadata["nq"])
        nv = int(self.metadata["nv"])
        expected = {
            "nominal_before": (iterations, horizon, nu),
            "candidates": (iterations, samples, horizon, nu),
            "costs": (iterations, samples),
            "valid": (iterations, samples),
            "weights": (iterations, samples),
            "nominal_after": (iterations, horizon, nu),
            "first_action": (iterations, nu),
            "final_qpos": (iterations, samples, nq),
            "final_qvel": (iterations, samples, nv),
            "rollout_wall_s": (iterations,),
        }
        for name, shape in expected.items():
            array = np.asarray(getattr(self, name))
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        if not np.all(self.valid == (np.isfinite(self.costs))):
            raise ValueError("valid mask must match finite candidate costs")
        if not np.isfinite(self.costs[self.valid]).all():
            raise ValueError("valid costs contain NaN or Inf")
        if not np.isfinite(self.final_qpos[self.valid]).all():
            raise ValueError("valid final_qpos contains NaN or Inf")
        if not np.isfinite(self.final_qvel[self.valid]).all():
            raise ValueError("valid final_qvel contains NaN or Inf")
        if not np.isfinite(self.weights).all():
            raise ValueError("weights contain NaN or Inf")
        if not np.isfinite(self.nominal_after).all():
            raise ValueError("nominal_after contains NaN or Inf")
        if not np.isfinite(self.rollout_wall_s).all() or np.any(self.rollout_wall_s < 0):
            raise ValueError("rollout_wall_s must be finite and non-negative")


def save_formulation_result(
    result: FormulationOptimizerResult, path: str | Path
) -> Path:
    result.validate()
    destination = Path(path).expanduser().resolve()
    if destination.suffix != ".npz":
        raise ValueError("formulation result path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        metadata_json=np.asarray(
            json.dumps(result.metadata, sort_keys=True, allow_nan=False), dtype=np.str_
        ),
        nominal_before=result.nominal_before,
        candidates=result.candidates,
        costs=result.costs,
        valid=result.valid,
        weights=result.weights,
        nominal_after=result.nominal_after,
        first_action=result.first_action,
        final_qpos=result.final_qpos,
        final_qvel=result.final_qvel,
        rollout_wall_s=result.rollout_wall_s,
    )
    return destination


def load_formulation_result(path: str | Path) -> FormulationOptimizerResult:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        result = FormulationOptimizerResult(
            metadata=json.loads(str(archive["metadata_json"].item())),
            nominal_before=archive["nominal_before"].copy(),
            candidates=archive["candidates"].copy(),
            costs=archive["costs"].copy(),
            valid=archive["valid"].copy(),
            weights=archive["weights"].copy(),
            nominal_after=archive["nominal_after"].copy(),
            first_action=archive["first_action"].copy(),
            final_qpos=archive["final_qpos"].copy(),
            final_qvel=archive["final_qvel"].copy(),
            rollout_wall_s=archive["rollout_wall_s"].copy(),
        )
    result.validate()
    return result


def _configure_task(bundle: Any):
    spec = bundle.spec
    if spec.task != "grasp_reorient":
        raise ValueError(f"only grasp_reorient is supported, got {spec.task!r}")
    task = get_task(spec.task, geometry=spec.geometry, role=TaskRole.ROLLOUT)
    scene_path = task.resolve_scene_path()
    if _sha256(scene_path) != spec.scene_sha256:
        raise ValueError(
            "rollout scene hash disagrees with the frozen planning bundle: "
            f"{scene_path}"
        )
    cost_source = Path(grasp_reorient_task_module.__file__).resolve()
    if _sha256(cost_source) != spec.cost_source_sha256:
        raise ValueError(
            "grasp_reorient cost source hash disagrees with the frozen "
            f"planning bundle: {cost_source}"
        )
    model, _ = task.load()
    if (model.nq, model.nv, model.nu) != (spec.nq, spec.nv, spec.nu):
        raise ValueError(
            "scene dimensions disagree with the frozen bundle: "
            f"{(model.nq, model.nv, model.nu)} != {(spec.nq, spec.nv, spec.nu)}"
        )
    model.opt.timestep = spec.physics_dt_s
    task.goal_vector = bundle.goal.copy()
    task.weights_wp.assign(bundle.cost_weights)
    task.goal_vector_wp.assign(bundle.goal)
    return task, model


def evaluate_formulation(
    bundle: Any,
    contact_cfg: ContactModelConfig,
    *,
    samples: int,
    iterations: int,
    model_key: str,
    nconmax: int = 200,
    njmax: int = 500,
) -> FormulationOptimizerResult:
    """Run one formulation on a prefix of the frozen nested noise tape."""

    enable_reference_protocol()
    from kamino_feasibility.reference_mppi import mppi_softmin_update

    bundle.validate()
    spec = bundle.spec
    if samples < 1 or samples > spec.n_samples:
        raise ValueError(f"samples must be in [1, {spec.n_samples}]")
    if iterations < 1 or iterations > spec.n_iterations:
        raise ValueError(f"iterations must be in [1, {spec.n_iterations}]")

    task, model = _configure_task(bundle)
    planner_cfg = MPPIConfig(
        n_samples=samples,
        time_horizon=None,
        step_time=None,
        step_horizon=spec.control_horizon,
        step_substeps=spec.physics_steps_per_control,
        noise_sigma=spec.noise_sigma,
        n_iterations=1,
        warm_start=False,
        ctrl_relative_to_qpos=True,
        nconmax=nconmax,
        njmax=njmax,
        debug=False,
        delta_range=(spec.action_delta_low, spec.action_delta_high),
        use_full_graph=True,
        seed=0,
        resample_interval=None,
        temperature=spec.temperature,
    )
    controller = MPPIController(
        task=task,
        cfg=contact_cfg,
        mppi_cfg=planner_cfg,
        rng=np.random.default_rng(0),
    )

    shapes = {
        "nominal_before": (iterations, spec.control_horizon, spec.nu),
        "candidates": (iterations, samples, spec.control_horizon, spec.nu),
        "costs": (iterations, samples),
        "valid": (iterations, samples),
        "weights": (iterations, samples),
        "nominal_after": (iterations, spec.control_horizon, spec.nu),
        "first_action": (iterations, spec.nu),
        "final_qpos": (iterations, samples, spec.nq),
        "final_qvel": (iterations, samples, spec.nv),
        "rollout_wall_s": (iterations,),
    }
    nominal_before = np.empty(shapes["nominal_before"], dtype="<f4")
    candidate_log = np.empty(shapes["candidates"], dtype="<f4")
    costs = np.full(shapes["costs"], np.nan, dtype="<f4")
    valid = np.zeros(shapes["valid"], dtype=bool)
    weights = np.zeros(shapes["weights"], dtype="<f4")
    nominal_after = np.empty(shapes["nominal_after"], dtype="<f4")
    first_action = np.empty(shapes["first_action"], dtype="<f4")
    final_qpos = np.full(shapes["final_qpos"], np.nan, dtype="<f4")
    final_qvel = np.full(shapes["final_qvel"], np.nan, dtype="<f4")
    rollout_wall_s = np.zeros(shapes["rollout_wall_s"], dtype="<f8")

    nominal = bundle.nominal_actions.copy()
    for iteration in range(iterations):
        candidates = bundle.candidates_for_iteration(
            iteration, nominal_actions=nominal
        )[:samples]
        nominal_before[iteration] = nominal
        candidate_log[iteration] = candidates
        controller.V_wp.assign(candidates)
        controller.qpos_reset.assign(bundle.qpos)
        controller.qvel_reset.assign(bundle.qvel)
        controller.ctrl_reset.assign(bundle.current_ctrl)

        started = time.perf_counter()
        controller._rollout()
        controller._fold_costs()
        wp.synchronize()
        rollout_wall_s[iteration] = time.perf_counter() - started

        iteration_costs = controller.costs_wp.numpy().astype("<f4", copy=True)
        qpos = controller.d.qpos.numpy().astype("<f4", copy=True)
        qvel = controller.d.qvel.numpy().astype("<f4", copy=True)
        iteration_valid = (
            np.isfinite(iteration_costs)
            & np.isfinite(qpos).all(axis=1)
            & np.isfinite(qvel).all(axis=1)
        )
        iteration_costs[~iteration_valid] = np.nan
        qpos[~iteration_valid] = np.nan
        qvel[~iteration_valid] = np.nan
        update = mppi_softmin_update(
            candidates,
            iteration_costs,
            temperature=spec.temperature,
        )
        costs[iteration] = iteration_costs
        valid[iteration] = iteration_valid
        weights[iteration] = update.weights
        nominal_after[iteration] = update.nominal_actions
        first_action[iteration] = update.first_action
        final_qpos[iteration] = qpos
        final_qvel[iteration] = qvel
        nominal = update.nominal_actions

    metadata = {
        "schema": RESULT_SCHEMA,
        "bundle_fingerprint_sha256": bundle.fingerprint_sha256(),
        "task": spec.task,
        "geometry": spec.geometry,
        "scene_sha256": spec.scene_sha256,
        "cost_source_sha256": spec.cost_source_sha256,
        "model_key": model_key,
        "model_label": contact_cfg.label,
        "backend": contact_cfg.backend.value,
        "samples": samples,
        "iterations": iterations,
        "control_horizon": spec.control_horizon,
        "physics_steps_per_control": spec.physics_steps_per_control,
        "physics_dt_s": spec.physics_dt_s,
        "control_dt_s": spec.control_dt_s,
        "temperature": spec.temperature,
        "action_semantics": spec.action_semantics,
        "cost_semantics": spec.cost_semantics,
        "cost_accumulation": spec.cost_accumulation,
        "nq": spec.nq,
        "nv": spec.nv,
        "nu": spec.nu,
        "mujoco_version": mujoco.__version__,
        "warp_version": wp.__version__,
        "evaluation_note": (
            "Exact frozen candidates evaluated through SamplingPlanner reset/control/"
            "step/cost kernels; random sampling and planner.plan() are bypassed."
        ),
    }
    result = FormulationOptimizerResult(
        metadata=metadata,
        nominal_before=nominal_before,
        candidates=candidate_log,
        costs=costs,
        valid=valid,
        weights=weights,
        nominal_after=nominal_after,
        first_action=first_action,
        final_qpos=final_qpos,
        final_qvel=final_qvel,
        rollout_wall_s=rollout_wall_s,
    )
    result.validate()
    return result
