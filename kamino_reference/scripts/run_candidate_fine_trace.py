#!/usr/bin/env python3
"""Run one fixed candidate batch and record every Kamino fine step on device."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import time

import numpy as np

from kamino_feasibility.batched_target_rollout import (
    KaminoBatchedLeapRollout,
    LeapFineStepTrace,
)
from kamino_feasibility.reference_mppi import mppi_softmin_update
from kamino_feasibility.reference_planning import (
    ACTION_SEMANTICS,
    COST_SEMANTICS,
    load_reference_planning_bundle,
)


TRACE_SCHEMA = "kamino_feasibility.candidate_fine_trace.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--padmm-tolerance", type=float, default=5.0e-4)
    parser.add_argument("--padmm-max-iterations", type=int, default=800)
    parser.add_argument("--padmm-rho0", type=float, default=0.1)
    parser.add_argument(
        "--padmm-penalty-update-method",
        choices=("fixed", "balanced"),
        default="fixed",
    )
    parser.add_argument(
        "--dynamics-storage", choices=("dense", "sparse"), default="sparse"
    )
    parser.add_argument(
        "--sparse-linear-solver", choices=("CR", "CRF"), default="CRF"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def _strict_valid(result, expected_fine_steps: int) -> np.ndarray:
    return (
        np.asarray(result.reset_success, dtype=bool)
        & (result.converged_steps == expected_fine_steps)
        & np.isfinite(result.costs)
        & np.isfinite(result.final_qpos).all(axis=1)
        & np.isfinite(result.final_qvel).all(axis=1)
    )


def main() -> int:
    args = _parse_args()
    bundle = load_reference_planning_bundle(args.bundle)
    spec = bundle.spec
    rollout = KaminoBatchedLeapRollout(
        args.scene,
        world_count=spec.n_samples,
        horizon_steps=spec.control_horizon,
        device=args.device,
        dt=spec.physics_dt_s,
        physics_steps_per_control=spec.physics_steps_per_control,
        action_semantics=ACTION_SEMANTICS,
        cost_semantics=COST_SEMANTICS,
        padmm_tolerance=args.padmm_tolerance,
        padmm_max_iterations=args.padmm_max_iterations,
        padmm_rho0=args.padmm_rho0,
        padmm_penalty_update_method=args.padmm_penalty_update_method,
        dynamics_storage=args.dynamics_storage,
        sparse_linear_solver=args.sparse_linear_solver,
    )
    rollout.set_initial_state(bundle.qpos, bundle.qvel)
    rollout.set_grasp_reorient_cost(bundle.goal, bundle.cost_weights)
    candidates = bundle.first_iteration_candidates()
    rollout.upload_action_sequences(np.transpose(candidates, (1, 0, 2)))

    start = time.perf_counter()
    result, trace = rollout.trace_rollout()
    wall_s = time.perf_counter() - start
    expected_fine_steps = (
        spec.control_horizon * spec.physics_steps_per_control
    )
    valid = _strict_valid(result, expected_fine_steps)
    costs = np.where(valid, result.costs, np.nan)
    update = mppi_softmin_update(
        candidates, costs, temperature=spec.temperature
    )

    metadata = {
        "schema": TRACE_SCHEMA,
        "bundle_path": str(args.bundle.expanduser().resolve()),
        "bundle_fingerprint_sha256": bundle.fingerprint_sha256(),
        "scene_path": str(args.scene.expanduser().resolve()),
        "device": str(rollout.device),
        "world_count": rollout.world_count,
        "fine_step_count": expected_fine_steps,
        "physics_dt_s": spec.physics_dt_s,
        "physics_steps_per_control": spec.physics_steps_per_control,
        "control_horizon": spec.control_horizon,
        "temperature": spec.temperature,
        "object_qpos_address": 16,
        "padmm_tolerance": rollout.padmm_tolerance,
        "padmm_max_iterations": rollout.padmm_max_iterations,
        "padmm_rho0": rollout.padmm_rho0,
        "padmm_penalty_update_method": rollout.padmm_penalty_update_method,
        "dynamics_storage": rollout.dynamics_storage,
        "sparse_linear_solver": rollout.sparse_linear_solver,
        "rollout_wall_s": wall_s,
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True), dtype=np.str_
        ),
        "initial_qpos": bundle.qpos.copy(),
        "initial_qvel": bundle.qvel.copy(),
        "candidates": candidates.copy(),
        "costs": result.costs.copy(),
        "valid": valid.copy(),
        "weights": update.weights.copy(),
        "first_action": update.first_action.copy(),
        "final_qpos": result.final_qpos.copy(),
        "final_qvel": result.final_qvel.copy(),
        "converged_steps": result.converged_steps.copy(),
        "contact_fine_steps": result.contact_fine_steps.copy(),
        "max_iterations": result.max_iterations.copy(),
        "max_active_contacts": result.max_active_contacts.copy(),
    }
    for field in fields(LeapFineStepTrace):
        arrays[f"trace_{field.name}"] = np.asarray(getattr(trace, field.name))
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)

    summary = {
        **metadata,
        "output": str(destination),
        "wall_s": wall_s,
        "strict_valid_worlds": int(np.count_nonzero(valid)),
        "experiment_ready": bool(np.all(valid)),
        "minimum_cost": update.minimum_cost,
        "mean_cost": update.mean_valid_cost,
        "best_candidate": int(np.nanargmin(costs)),
        "effective_sample_size": update.effective_sample_size,
        "first_action": update.first_action.tolist(),
        "maximum_iterations": int(np.max(result.max_iterations)),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    summary_path = args.summary.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if summary["experiment_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
