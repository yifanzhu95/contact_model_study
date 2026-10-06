"""CPU coordinator for chunked, repeat-aggregated Kamino optimization.

This module deliberately never imports Newton or Warp and never starts a GPU
worker. It prepares fingerprinted chunk bundles, validates externally produced
results, aggregates repeated costs, performs one MPPI update, and prepares the
next iteration. A synthetic evaluator exercises the complete artifact protocol
without CUDA.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .reference_mppi import mppi_softmin_update
from .reference_optimizer_v2 import (
    bootstrap_repeat_aggregated_first_action,
    repeat_aggregated_mppi_update,
)
from .reference_planning import (
    ReferencePlanningBundle,
    file_sha256,
    load_reference_planning_bundle,
    save_reference_planning_bundle,
)


RUN_SCHEMA = "kamino_feasibility.reference_optimizer_v2_run.v1"
TASK_SCHEMA = "kamino_feasibility.reference_optimizer_v2_tasks.v1"
RESULT_SCHEMA = "kamino_feasibility.reference_optimizer_v2_iteration.v1"
SYNTHETIC_TRACE_SCHEMA = "kamino_feasibility.synthetic_candidate_chunk.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_CONTRACT_STATES = {
    (
        "kamino_feasibility.reference_optimizer_v2_preregistration.v1",
        "FROZEN_BEFORE_NEW_GPU_DATA",
    ),
    (
        "kamino_feasibility.reference_optimizer_v3_preregistration.v1",
        "FROZEN_FOR_COLLABORATOR_SCALING",
    ),
}


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_recorded(path: str, root: Path) -> Path:
    selected = Path(path).expanduser()
    return selected.resolve() if selected.is_absolute() else (root / selected).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _rank_positions(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _rank_positions(left).astype(np.float64)
    right_rank = _rank_positions(right).astype(np.float64)
    left_rank -= np.mean(left_rank)
    right_rank -= np.mean(right_rank)
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    return 1.0 if denominator == 0.0 else float(np.dot(left_rank, right_rank) / denominator)


def _budget(contract: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [row for row in contract["budget_ladder"] if row["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate budget label {label!r}")
    return dict(matches[0])


def _load_bound_inputs(
    contract_path: Path, inputs_index_path: Path
) -> tuple[dict[str, Any], dict[str, Any], ReferencePlanningBundle, Path]:
    contract_path = contract_path.expanduser().resolve()
    inputs_index_path = inputs_index_path.expanduser().resolve()
    repository_root = REPOSITORY_ROOT
    contract = _load_json(contract_path)
    index = _load_json(inputs_index_path)
    contract_state = (contract.get("schema"), contract.get("status"))
    if contract_state not in SUPPORTED_CONTRACT_STATES:
        raise ValueError(
            "unsupported or unfrozen optimizer contract: "
            f"schema={contract_state[0]!r}, status={contract_state[1]!r}"
        )
    if file_sha256(contract_path) != index.get("contract_sha256"):
        raise ValueError("contract SHA-256 disagrees with the inputs index")
    development = index["development"]
    bundle_path = _resolve_recorded(development["path"], repository_root)
    if file_sha256(bundle_path) != development["file_sha256"]:
        raise ValueError("development maximum bundle file SHA-256 mismatch")
    bundle = load_reference_planning_bundle(bundle_path)
    if bundle.fingerprint_sha256() != development["fingerprint_sha256"]:
        raise ValueError("development maximum bundle fingerprint mismatch")
    return contract, index, bundle, repository_root


def _immutable_run_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: manifest[key]
        for key in (
            "schema",
            "run_fingerprint_sha256",
            "contract_path",
            "contract_sha256",
            "inputs_index_path",
            "inputs_index_sha256",
            "maximum_bundle_path",
            "maximum_bundle_fingerprint_sha256",
            "budget",
            "chunk_size",
            "temperature",
            "solver_policy",
        )
    }


def initialize_run(
    *,
    contract_path: str | Path,
    inputs_index_path: str | Path,
    run_directory: str | Path,
    budget_label: str,
    chunk_size: int,
    scene_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create or verify one resumable optimizer run without starting workers."""

    contract_path = Path(contract_path).expanduser().resolve()
    inputs_index_path = Path(inputs_index_path).expanduser().resolve()
    contract, _, bundle, repository_root = _load_bound_inputs(
        contract_path, inputs_index_path
    )
    budget = _budget(contract, budget_label)
    maximum_safe_chunk = int(contract["safety_execution"]["initial_candidate_chunk_worlds_max"])
    if chunk_size < 1 or chunk_size > maximum_safe_chunk:
        raise ValueError(
            f"chunk_size must be in [1, {maximum_safe_chunk}] under the active safety contract"
        )
    if int(budget["samples"]) > bundle.spec.n_samples:
        raise ValueError("budget sample count exceeds the frozen maximum tape")
    if int(budget["iterations"]) > bundle.spec.n_iterations:
        raise ValueError("budget iteration count exceeds the frozen maximum tape")

    exported_scene = Path(str(bundle.spec.metadata["scene_path_at_export"]))
    selected_scene = (
        Path(scene_path).expanduser().resolve()
        if scene_path is not None
        else exported_scene
    )
    scene_verified = False
    if scene_path is not None:
        if not selected_scene.is_file():
            raise FileNotFoundError(f"scene override does not exist: {selected_scene}")
        selected_scene_sha256 = file_sha256(selected_scene)
        if selected_scene_sha256 != bundle.spec.scene_sha256:
            raise ValueError(
                "scene override SHA-256 does not match the frozen planning bundle"
            )
        scene_verified = True

    run_directory = Path(run_directory).expanduser().resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    immutable_seed = {
        "schema": RUN_SCHEMA,
        "contract_sha256": file_sha256(contract_path),
        "inputs_index_sha256": file_sha256(inputs_index_path),
        "maximum_bundle_fingerprint_sha256": bundle.fingerprint_sha256(),
        "budget": budget,
        "chunk_size": chunk_size,
        "temperature": float(contract["temperature_rule"]["primary_temperature"]),
        "solver_policy": dict(contract["solver_policy"]),
        "scene_sha256": bundle.spec.scene_sha256,
    }
    manifest = {
        **immutable_seed,
        "run_fingerprint_sha256": _json_sha256(immutable_seed),
        "contract_path": _relative(contract_path, repository_root),
        "inputs_index_path": _relative(inputs_index_path, repository_root),
        "maximum_bundle_path": _relative(
            _resolve_recorded(
                _load_json(inputs_index_path)["development"]["path"], repository_root
            ),
            repository_root,
        ),
        "repository_root": str(repository_root),
        "run_directory": str(run_directory),
        "scene_path": str(selected_scene),
        "scene_verified_at_initialization": scene_verified,
        "status": "initialized_no_workers_started",
        "current_iteration": 0,
        "completed_iterations": [],
        "worker_execution_authorized": False,
        "claim_scope": budget["claim_scope"],
    }
    manifest_path = run_directory / "run_manifest.json"
    if manifest_path.exists():
        existing = _load_json(manifest_path)
        if _immutable_run_fields(existing) != _immutable_run_fields(manifest):
            raise ValueError("existing run manifest disagrees with requested immutable inputs")
        return existing
    _write_json(manifest_path, manifest)
    _prepare_iteration(run_directory, iteration=0, nominal=bundle.nominal_actions)
    return _load_json(manifest_path)


def _chunk_bundle(
    maximum: ReferencePlanningBundle,
    *,
    candidates: np.ndarray,
    iteration: int,
    start: int,
    stop: int,
    run_fingerprint: str,
) -> ReferencePlanningBundle:
    metadata = dict(maximum.spec.metadata)
    metadata.update(
        {
            "reference_optimizer_run_fingerprint_sha256": run_fingerprint,
            "reference_optimizer_iteration": iteration,
            "global_candidate_start": start,
            "global_candidate_stop": stop,
            "chunk_encoding": "zero_nominal_plus_exact_clipped_candidate_actions",
        }
    )
    spec = replace(
        maximum.spec,
        n_samples=stop - start,
        n_iterations=1,
        metadata=metadata,
    )
    zeros = np.zeros_like(maximum.nominal_actions, dtype="<f4")
    return ReferencePlanningBundle(
        spec=spec,
        qpos=maximum.qpos.copy(),
        qvel=maximum.qvel.copy(),
        current_ctrl=maximum.current_ctrl.copy(),
        goal=maximum.goal.copy(),
        cost_weights=maximum.cost_weights.copy(),
        nominal_actions=zeros,
        perturbations=candidates[start:stop][np.newaxis].astype("<f4", copy=True),
    )


def _prepare_iteration(
    run_directory: Path, *, iteration: int, nominal: np.ndarray
) -> dict[str, Any]:
    manifest_path = run_directory / "run_manifest.json"
    manifest = _load_json(manifest_path)
    repository_root = Path(manifest["repository_root"])
    maximum = load_reference_planning_bundle(
        _resolve_recorded(manifest["maximum_bundle_path"], repository_root)
    )
    budget = manifest["budget"]
    sample_count = int(budget["samples"])
    repeat_count = int(budget["numerical_repeats"])
    if iteration < 0 or iteration >= int(budget["iterations"]):
        raise IndexError("iteration is outside the selected budget")
    candidates = maximum.candidates_for_iteration(
        iteration, nominal_actions=np.asarray(nominal, dtype="<f4")
    )[:sample_count]
    iteration_directory = run_directory / f"iteration_{iteration:02d}"
    chunks_directory = iteration_directory / "chunks"
    chunks_directory.mkdir(parents=True, exist_ok=True)
    tasks = []
    chunk_size = int(manifest["chunk_size"])
    solver = manifest["solver_policy"]
    scene = manifest["scene_path"]
    expected_fine_steps = maximum.spec.control_horizon * maximum.spec.physics_steps_per_control
    for start in range(0, sample_count, chunk_size):
        stop = min(sample_count, start + chunk_size)
        chunk = _chunk_bundle(
            maximum,
            candidates=candidates,
            iteration=iteration,
            start=start,
            stop=stop,
            run_fingerprint=manifest["run_fingerprint_sha256"],
        )
        chunk_path = chunks_directory / f"chunk_{start:04d}_{stop:04d}.npz"
        if chunk_path.exists():
            existing = load_reference_planning_bundle(chunk_path)
            if existing.fingerprint_sha256() != chunk.fingerprint_sha256():
                raise ValueError(f"existing chunk bundle mismatch: {chunk_path}")
        else:
            save_reference_planning_bundle(chunk, chunk_path)
        for repeat in range(repeat_count):
            output = (
                iteration_directory
                / "results"
                / f"repeat_{repeat + 1:02d}"
                / f"chunk_{start:04d}_{stop:04d}.npz"
            )
            summary = output.with_suffix(".json")
            worker_command = [
                sys.executable,
                "scripts/run_candidate_fine_trace.py",
                "--bundle",
                _relative(chunk_path, repository_root),
                "--scene",
                str(scene),
                "--device",
                str(solver["device"]),
                "--padmm-tolerance",
                str(solver["padmm_tolerance"]),
                "--padmm-max-iterations",
                str(solver["padmm_max_iterations"]),
                "--padmm-rho0",
                str(solver["padmm_rho0"]),
                "--padmm-penalty-update-method",
                str(solver["padmm_penalty_update_method"]),
                "--dynamics-storage",
                str(solver["dynamics_storage"]),
                "--sparse-linear-solver",
                str(solver["sparse_linear_solver"]),
                "--output",
                _relative(output, repository_root),
                "--summary",
                _relative(summary, repository_root),
            ]
            tasks.append(
                {
                    "task_id": f"i{iteration:02d}_r{repeat + 1:02d}_c{start:04d}_{stop:04d}",
                    "iteration": iteration,
                    "repeat": repeat,
                    "candidate_start": start,
                    "candidate_stop": stop,
                    "chunk_bundle_path": _relative(chunk_path, repository_root),
                    "chunk_bundle_file_sha256": file_sha256(chunk_path),
                    "chunk_bundle_fingerprint_sha256": chunk.fingerprint_sha256(),
                    "result_path": _relative(output, repository_root),
                    "summary_path": _relative(summary, repository_root),
                    "expected_fine_steps": expected_fine_steps,
                    "scene_sha256": manifest["scene_sha256"],
                    "worker_command": worker_command,
                    "must_run_through_gpu_guard": True,
                    "worker_started_by_coordinator": False,
                }
            )
    task_manifest = {
        "schema": TASK_SCHEMA,
        "run_fingerprint_sha256": manifest["run_fingerprint_sha256"],
        "iteration": iteration,
        "sample_count": sample_count,
        "repeat_count": repeat_count,
        "chunk_size": chunk_size,
        "nominal_before": np.asarray(nominal, dtype=np.float32).tolist(),
        "candidate_sha256": hashlib.sha256(
            np.ascontiguousarray(candidates).tobytes()
        ).hexdigest(),
        "task_count": len(tasks),
        "gpu_workers_started": False,
        "tasks": tasks,
    }
    _write_json(iteration_directory / "tasks.json", task_manifest)
    return task_manifest


def _write_synthetic_result(
    *, task: dict[str, Any], repository_root: Path, synthetic_seed: int
) -> bool:
    output = _resolve_recorded(task["result_path"], repository_root)
    if output.exists():
        return False
    chunk_path = _resolve_recorded(task["chunk_bundle_path"], repository_root)
    chunk = load_reference_planning_bundle(chunk_path)
    candidates = chunk.first_iteration_candidates()
    iteration = int(task["iteration"])
    repeat = int(task["repeat"])
    start = int(task["candidate_start"])
    stop = int(task["candidate_stop"])
    actuator_phase = np.arange(chunk.spec.nu, dtype=np.float64)[None, :]
    horizon_phase = np.arange(chunk.spec.control_horizon, dtype=np.float64)[:, None]
    target = 0.025 * np.sin(0.19 * actuator_phase + 0.31 * horizon_phase + 0.17 * iteration)
    base = np.sum((candidates.astype(np.float64) - target) ** 2, axis=(1, 2))
    generator = np.random.default_rng(
        synthetic_seed + iteration * 100_000 + repeat * 1_000 + start
    )
    candidate_noise = generator.normal(0.0, 2.0e-4, size=stop - start)
    costs = (base + candidate_noise + repeat * 1.0e-5).astype("<f4")
    valid = np.ones(stop - start, dtype=bool)
    converged = np.full(stop - start, int(task["expected_fine_steps"]), dtype="<i4")
    metadata = {
        "schema": SYNTHETIC_TRACE_SCHEMA,
        "backend": "cpu_synthetic_no_physics",
        "task_id": task["task_id"],
        "bundle_fingerprint_sha256": chunk.fingerprint_sha256(),
        "iteration": iteration,
        "repeat": repeat,
        "candidate_start": start,
        "candidate_stop": stop,
        "fine_step_count": int(task["expected_fine_steps"]),
        "synthetic_seed": synthetic_seed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        candidates=candidates,
        costs=costs,
        valid=valid,
        converged_steps=converged,
    )
    _write_json(_resolve_recorded(task["summary_path"], repository_root), metadata)
    return True


def _pairwise_diagnostics(costs: np.ndarray, candidates: np.ndarray, temperature: float) -> dict[str, Any]:
    repeat_updates = [
        mppi_softmin_update(candidates, row, temperature=temperature)
        for row in costs
    ]
    first_actions = np.stack([update.first_action for update in repeat_updates])
    top_k = min(5, costs.shape[1])
    pairs = []
    for left, right in combinations(range(costs.shape[0]), 2):
        left_top = set(np.argsort(costs[left], kind="stable")[:top_k])
        right_top = set(np.argsort(costs[right], kind="stable")[:top_k])
        delta = first_actions[right].astype(np.float64) - first_actions[left].astype(np.float64)
        pairs.append(
            {
                "left_repeat": left,
                "right_repeat": right,
                "top_k_overlap": len(left_top & right_top),
                "spearman": _spearman(costs[left], costs[right]),
                "first_action_l2_rad": float(np.linalg.norm(delta)),
                "first_action_linf_rad": float(np.max(np.abs(delta))),
            }
        )
    return {
        "top_k": top_k,
        "best_candidate_by_repeat": np.argmin(costs, axis=1).astype(int).tolist(),
        "minimum_pairwise_top_k_overlap": min(row["top_k_overlap"] for row in pairs),
        "minimum_pairwise_spearman": min(row["spearman"] for row in pairs),
        "maximum_pairwise_first_action_l2_rad": max(row["first_action_l2_rad"] for row in pairs),
        "maximum_pairwise_first_action_linf_rad": max(row["first_action_linf_rad"] for row in pairs),
        "per_repeat_first_actions": first_actions.tolist(),
        "pairs": pairs,
    }


def collect_current_iteration(run_directory: str | Path) -> dict[str, Any]:
    """Validate all current chunk results, update once, and prepare the next iteration."""

    run_directory = Path(run_directory).expanduser().resolve()
    manifest_path = run_directory / "run_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest["status"] == "complete":
        return manifest
    repository_root = Path(manifest["repository_root"])
    iteration = int(manifest["current_iteration"])
    task_manifest = _load_json(run_directory / f"iteration_{iteration:02d}" / "tasks.json")
    sample_count = int(task_manifest["sample_count"])
    repeat_count = int(task_manifest["repeat_count"])
    maximum = load_reference_planning_bundle(
        _resolve_recorded(manifest["maximum_bundle_path"], repository_root)
    )
    costs = np.full((repeat_count, sample_count), np.nan, dtype=np.float64)
    strict_valid = np.zeros((repeat_count, sample_count), dtype=bool)
    filled = np.zeros((repeat_count, sample_count), dtype=bool)
    backends = set()
    candidates = None
    nominal_before = np.asarray(task_manifest["nominal_before"], dtype="<f4")
    expected_candidates = maximum.candidates_for_iteration(
        iteration, nominal_actions=nominal_before
    )[:sample_count]
    for task in task_manifest["tasks"]:
        output = _resolve_recorded(task["result_path"], repository_root)
        if not output.exists():
            raise FileNotFoundError(f"missing worker result: {output}")
        chunk_path = _resolve_recorded(task["chunk_bundle_path"], repository_root)
        if file_sha256(chunk_path) != task["chunk_bundle_file_sha256"]:
            raise ValueError(f"chunk bundle file hash changed: {chunk_path}")
        chunk = load_reference_planning_bundle(chunk_path)
        if chunk.fingerprint_sha256() != task["chunk_bundle_fingerprint_sha256"]:
            raise ValueError(f"chunk bundle fingerprint changed: {chunk_path}")
        with np.load(output, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            result_candidates = payload["candidates"].copy()
            result_costs = payload["costs"].astype(np.float64, copy=True)
            result_valid = payload["valid"].astype(bool, copy=True)
            converged = payload["converged_steps"].astype(np.int64, copy=True)
        backends.add(str(metadata.get("backend", "kamino_worker")))
        if metadata.get("bundle_fingerprint_sha256") != chunk.fingerprint_sha256():
            raise ValueError(f"worker result bundle fingerprint mismatch: {output}")
        expected_chunk = chunk.first_iteration_candidates()
        if not np.array_equal(result_candidates, expected_chunk):
            raise ValueError(f"worker result candidate mismatch: {output}")
        start = int(task["candidate_start"])
        stop = int(task["candidate_stop"])
        repeat = int(task["repeat"])
        if np.any(filled[repeat, start:stop]):
            raise ValueError("duplicate candidate cells in task manifest")
        costs[repeat, start:stop] = result_costs
        strict_valid[repeat, start:stop] = (
            result_valid
            & np.isfinite(result_costs)
            & (converged == int(task["expected_fine_steps"]))
        )
        filled[repeat, start:stop] = True
    if not np.all(filled):
        raise ValueError("task manifest did not cover every repeat/candidate cell")
    if not np.all(strict_valid):
        raise ValueError("at least one repeat/candidate result failed strict validity")
    candidates = expected_candidates
    temperature = float(manifest["temperature"])
    contract = _load_json(
        _resolve_recorded(manifest["contract_path"], repository_root)
    )
    if repeat_count == 1:
        update = mppi_softmin_update(candidates, costs[0], temperature=temperature)
        aggregation = None
        bootstrap = None
        repeat_diagnostics = None
    else:
        combined = repeat_aggregated_mppi_update(
            candidates,
            costs,
            strict_valid=strict_valid,
            temperature=temperature,
            minimum_replicates=repeat_count,
        )
        update = combined.update
        aggregation = combined.aggregation
        estimator = contract["numerical_repeat_estimator"]
        bootstrap = bootstrap_repeat_aggregated_first_action(
            candidates,
            costs,
            strict_valid=strict_valid,
            temperature=temperature,
            draw_count=int(estimator["bootstrap_draws"]),
            seed=int(estimator["bootstrap_seed"]),
        )
        repeat_diagnostics = _pairwise_diagnostics(costs, candidates, temperature)

    iteration_directory = run_directory / f"iteration_{iteration:02d}"
    claim_policy = contract.get(
        "reference_claim",
        {
            "estimator_contract_supported": False,
            "single_budget_claim_allowed": False,
            "reason": (
                "legacy v2 five-repeat budgets were superseded by the "
                "Step-43 repeat-count convergence result"
            ),
        },
    )
    estimator_eligible = bool(
        claim_policy.get("estimator_contract_supported", False)
        and repeat_count
        >= int(claim_policy.get("minimum_numerical_repeats", 2**31 - 1))
        and manifest["claim_scope"] != "safety_and_plumbing_only"
    )
    metadata = {
        "schema": RESULT_SCHEMA,
        "run_fingerprint_sha256": manifest["run_fingerprint_sha256"],
        "iteration": iteration,
        "backend_values": sorted(backends),
        "sample_count": sample_count,
        "repeat_count": repeat_count,
        "strict_valid_cells": int(np.count_nonzero(strict_valid)),
        "expected_cells": int(strict_valid.size),
        "effective_sample_size": update.effective_sample_size,
        "reference_estimator_eligible": estimator_eligible,
        "reference_claim_eligible": bool(
            estimator_eligible
            and claim_policy.get("single_budget_claim_allowed", False)
        ),
        "reference_claim_policy": claim_policy,
        "repeat_diagnostics": repeat_diagnostics,
        "bootstrap": None if bootstrap is None else {
            key: value
            for key, value in asdict(bootstrap).items()
            if not isinstance(value, np.ndarray)
        },
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        "nominal_before": nominal_before,
        "candidates": candidates,
        "costs_by_replicate": costs,
        "strict_valid": strict_valid,
        "weights": update.weights,
        "nominal_after": update.nominal_actions,
        "first_action": update.first_action,
    }
    if aggregation is not None:
        arrays.update(
            {
                "aggregated_costs": aggregation.aggregated_costs,
                "mean_costs": aggregation.mean_costs,
                "median_costs": aggregation.median_costs,
                "cost_sample_std": aggregation.sample_standard_deviation,
                "cost_standard_error": aggregation.standard_error,
                "cost_minimum": aggregation.minimum_costs,
                "cost_maximum": aggregation.maximum_costs,
                "cost_ranges": aggregation.cost_ranges,
                "bootstrap_first_action_lower": bootstrap.lower_first_action,
                "bootstrap_first_action_median": bootstrap.median_first_action,
                "bootstrap_first_action_upper": bootstrap.upper_first_action,
            }
        )
    result_path = iteration_directory / "iteration_result.npz"
    np.savez_compressed(result_path, **arrays)
    summary = {
        **metadata,
        "result_path": str(result_path),
        "result_file_sha256": file_sha256(result_path),
        "minimum_aggregated_cost": update.minimum_cost,
        "mean_aggregated_cost": update.mean_valid_cost,
        "first_action": update.first_action.tolist(),
        "nominal_after_sha256": hashlib.sha256(
            np.ascontiguousarray(update.nominal_actions).tobytes()
        ).hexdigest(),
        "bootstrap_componentwise_95_width_max_rad": (
            None
            if bootstrap is None
            else float(np.max(bootstrap.upper_first_action - bootstrap.lower_first_action))
        ),
    }
    _write_json(iteration_directory / "iteration_result.json", summary)

    completed = list(manifest["completed_iterations"])
    if iteration not in completed:
        completed.append(iteration)
    manifest["completed_iterations"] = sorted(completed)
    next_iteration = iteration + 1
    if next_iteration < int(manifest["budget"]["iterations"]):
        manifest["current_iteration"] = next_iteration
        manifest["status"] = "waiting_for_external_worker_results"
        _write_json(manifest_path, manifest)
        _prepare_iteration(
            run_directory,
            iteration=next_iteration,
            nominal=update.nominal_actions,
        )
    else:
        manifest["status"] = "complete"
        manifest["current_iteration"] = iteration
        manifest["worker_result_backends"] = sorted(backends)
        manifest["final_first_action"] = update.first_action.tolist()
        manifest["final_iteration_result"] = str(result_path)
        _write_json(manifest_path, manifest)
    return summary


def run_cpu_synthetic_dry_run(
    run_directory: str | Path, *, synthetic_seed: int = 4001
) -> dict[str, Any]:
    """Exercise initialization, chunking, resume and collection without CUDA."""

    run_directory = Path(run_directory).expanduser().resolve()
    manifest_path = run_directory / "run_manifest.json"
    created = 0
    skipped = 0
    while True:
        manifest = _load_json(manifest_path)
        if manifest["status"] == "complete":
            return {"manifest": manifest, "created_results": created, "skipped_results": skipped}
        write_result = write_cpu_synthetic_current_iteration(
            run_directory, synthetic_seed=synthetic_seed
        )
        created += int(write_result["created_results"])
        skipped += int(write_result["skipped_results"])
        collect_current_iteration(run_directory)


def write_cpu_synthetic_current_iteration(
    run_directory: str | Path, *, synthetic_seed: int = 4001
) -> dict[str, int]:
    """Write only missing current-iteration results for resume testing."""

    run_directory = Path(run_directory).expanduser().resolve()
    manifest = _load_json(run_directory / "run_manifest.json")
    if manifest["status"] == "complete":
        return {"created_results": 0, "skipped_results": 0}
    repository_root = Path(manifest["repository_root"])
    iteration = int(manifest["current_iteration"])
    tasks = _load_json(run_directory / f"iteration_{iteration:02d}" / "tasks.json")
    created = 0
    skipped = 0
    for task in tasks["tasks"]:
        if _write_synthetic_result(
            task=task,
            repository_root=repository_root,
            synthetic_seed=synthetic_seed,
        ):
            created += 1
        else:
            skipped += 1
    return {"created_results": created, "skipped_results": skipped}
