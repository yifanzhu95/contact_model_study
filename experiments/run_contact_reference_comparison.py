#!/usr/bin/env python3
"""Run matched M1--M4 and isolated Kamino reference experiments.

One repository contains two incompatible environments.  This command runs the
main formulations in the current interpreter and invokes Kamino through
``kamino_reference/.venv/bin/python``.  Both sides consume one fingerprinted
planning bundle; generated runs are resumable and remain outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from contact_study.contact_models.config import ContactModelConfig
from contact_study.evaluation.reference_bridge import (
    REPOSITORY_ROOT,
    enable_reference_protocol,
    reference_paths,
)
from contact_study.evaluation.reference_optimizer import (
    evaluate_formulation,
    load_formulation_result,
    save_formulation_result,
)


MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _git_contains(commit: str) -> bool:
    """Whether the frozen source commit is an ancestor of this checkout."""

    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    return completed.returncode == 0


def _executable_path(path: Path) -> Path:
    """Return an absolute path without resolving a venv's Python symlink.

    Resolving ``.venv/bin/python`` to ``/usr/bin/python3`` discards the virtual
    environment because Python locates ``pyvenv.cfg`` from argv[0].
    """

    selected = path.expanduser()
    return selected if selected.is_absolute() else (Path.cwd() / selected).absolute()


def _protocol():
    enable_reference_protocol()
    from kamino_feasibility.reference_planning import load_reference_planning_bundle

    return load_reference_planning_bundle


def _contract_and_bundle(role: str = "development"):
    paths = reference_paths()
    contract = _load_json(paths["contract"])
    index = _load_json(paths["inputs_index"])
    if role not in ("template", "development", "held_out"):
        raise ValueError(f"unknown bundle role {role!r}")
    bundle_path = paths["root"] / index[role]["path"]
    bundle = _protocol()(bundle_path)
    return contract, index, bundle_path, bundle


def _budget(contract: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [row for row in contract["budget_ladder"] if row["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate budget {label!r}")
    return dict(matches[0])


def validate_checkout(
    *, kamino_python: Path, scene: Path, check_cuda: bool
) -> dict[str, Any]:
    paths = reference_paths()
    contract, index, bundle_path, bundle = _contract_and_bundle("development")
    task_source = REPOSITORY_ROOT / "contact_study" / "tasks" / "grasp_reorient.py"
    checks = {
        "kamino_python_present": kamino_python.is_file(),
        "scene_present": scene.is_file(),
        "contract_present": paths["contract"].is_file(),
        "inputs_index_present": paths["inputs_index"].is_file(),
        "bundle_present": bundle_path.is_file(),
        # The delivery commit necessarily comes after the source commit frozen
        # in the bundle.  Exact scene/cost hashes below detect semantic drift;
        # ancestry proves the checkout contains the recorded source history.
        "main_history_contains_frozen_commit": _git_contains(
            bundle.spec.main_repo_commit
        ),
        "scene_hash_matches": scene.is_file() and _sha256(scene) == bundle.spec.scene_sha256,
        "cost_source_hash_matches": (
            task_source.is_file() and _sha256(task_source) == bundle.spec.cost_source_sha256
        ),
        "contract_hash_matches": (
            _sha256(paths["contract"]) == index["contract_sha256"]
        ),
        "contract_frozen": contract.get("status") == "FROZEN_FOR_COLLABORATOR_SCALING",
    }
    if not kamino_python.is_file():
        return {"ready": False, "checks": checks, "kamino_validation": None}
    command = [
        str(kamino_python),
        str(paths["validator"]),
        "--scene",
        str(scene),
    ]
    if check_cuda:
        command.append("--check-cuda")
    completed = subprocess.run(
        command,
        cwd=paths["root"],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        kamino_validation = json.loads(completed.stdout)
    except json.JSONDecodeError:
        # Warp initialization may precede the JSON when --check-cuda is used.
        begin = completed.stdout.find("{")
        kamino_validation = (
            json.loads(completed.stdout[begin:]) if begin >= 0 else None
        )
    checks["kamino_validator_exit_zero"] = completed.returncode == 0
    checks["kamino_validator_ready"] = bool(
        kamino_validation and kamino_validation.get("ready")
    )
    return {
        "schema": "contact_study.reference_delivery_validation.v1",
        "ready": bool(all(checks.values())),
        "checks": checks,
        "main_commit": _git_head(),
        "scene": str(scene),
        "bundle": str(bundle_path),
        "bundle_fingerprint_sha256": bundle.fingerprint_sha256(),
        "kamino_command": command,
        "kamino_validation": kamino_validation,
        "kamino_stderr": completed.stderr.strip(),
    }


def run_main_formulations(
    *, budget_label: str, models: list[str], run_directory: Path
) -> dict[str, Any]:
    contract, _, bundle_path, bundle = _contract_and_bundle("development")
    budget = _budget(contract, budget_label)
    samples = int(budget["samples"])
    iterations = int(budget["iterations"])
    output_directory = run_directory / "main"
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_key in models:
        if model_key not in MODEL_FACTORIES:
            raise ValueError(f"unknown main formulation {model_key!r}")
        print(
            f"[main] {model_key}: {samples} candidates x {iterations} iterations",
            flush=True,
        )
        result = evaluate_formulation(
            bundle,
            MODEL_FACTORIES[model_key](),
            samples=samples,
            iterations=iterations,
            model_key=model_key,
        )
        destination = save_formulation_result(
            result, output_directory / f"{model_key}.npz"
        )
        rows.append(
            {
                "model": model_key,
                "path": str(destination),
                "valid_cells": int(np.count_nonzero(result.valid)),
                "expected_cells": int(result.valid.size),
                "total_rollout_wall_s": float(np.sum(result.rollout_wall_s)),
                "final_first_action": result.first_action[-1].tolist(),
            }
        )
    manifest = {
        "schema": "contact_study.main_formulation_run.v1",
        "budget": budget,
        "bundle": str(bundle_path),
        "bundle_fingerprint_sha256": bundle.fingerprint_sha256(),
        "main_commit": _git_head(),
        "models": rows,
    }
    _write_json(output_directory / "manifest.json", manifest)
    return manifest


def _run_command(command: list[str], *, cwd: Path) -> None:
    print("[subprocess] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def prepare_kamino(
    *, kamino_python: Path, scene: Path, budget_label: str,
    chunk_size: int, run_directory: Path,
) -> Path:
    paths = reference_paths()
    destination = run_directory / "kamino"
    command = [
        str(kamino_python),
        str(paths["coordinator"]),
        "initialize",
        "--contract",
        str(paths["contract"]),
        "--inputs-index",
        str(paths["inputs_index"]),
        "--budget",
        budget_label,
        "--chunk-size",
        str(chunk_size),
        "--scene",
        str(scene),
        "--run-directory",
        str(destination),
    ]
    _run_command(command, cwd=paths["root"])
    return destination


def run_kamino_workers(
    *, kamino_python: Path, run_directory: Path, gpu_index: int,
    maximum_runtime_s: float, max_new_tasks: int | None,
    allow_display_gpu: bool, headless: bool,
) -> None:
    paths = reference_paths()
    command = [
        str(kamino_python),
        str(paths["workers"]),
        "--run-directory",
        str(run_directory / "kamino"),
        "--gpu-index",
        str(gpu_index),
        "--maximum-runtime-s",
        str(maximum_runtime_s),
    ]
    if max_new_tasks is not None:
        command += ["--max-new-tasks", str(max_new_tasks)]
    if allow_display_gpu:
        command.append("--allow-display-gpu")
    if headless:
        command += ["--no-require-anydesk", "--no-require-x11"]
    _run_command(command, cwd=paths["root"])


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
    a = left - np.mean(left)
    b = right - np.mean(right)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else float("nan")


def _kamino_iteration(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "metadata": json.loads(str(archive["metadata_json"].item())),
            "candidates": archive["candidates"].copy(),
            "costs_by_replicate": archive["costs_by_replicate"].copy(),
            "strict_valid": archive["strict_valid"].copy(),
            "weights": archive["weights"].copy(),
            "nominal_after": archive["nominal_after"].copy(),
            "first_action": archive["first_action"].copy(),
        }


def build_report(
    *, models: list[str], run_directory: Path, top_k: int = 5
) -> dict[str, Any]:
    kamino_root = run_directory / "kamino"
    manifest = _load_json(kamino_root / "run_manifest.json")
    iterations = int(manifest["budget"]["iterations"])
    kamino = [
        _kamino_iteration(
            kamino_root / f"iteration_{iteration:02d}" / "iteration_result.npz"
        )
        for iteration in range(iterations)
    ]
    rows = []
    for model in models:
        main = load_formulation_result(run_directory / "main" / f"{model}.npz")
        if main.metadata["bundle_fingerprint_sha256"] != manifest[
            "maximum_bundle_fingerprint_sha256"
        ]:
            raise ValueError(f"{model} and Kamino use different planning bundles")
        iteration_rows = []
        for iteration in range(iterations):
            k = kamino[iteration]
            k_valid = np.all(k["strict_valid"], axis=0)
            k_cost = np.mean(k["costs_by_replicate"], axis=0, dtype=np.float64)
            m_valid = main.valid[iteration]
            common = k_valid & m_valid
            candidate_equal = np.array_equal(
                k["candidates"], main.candidates[iteration]
            )
            row: dict[str, Any] = {
                "iteration": iteration,
                "candidates_identical": candidate_equal,
                "common_valid_samples": int(np.count_nonzero(common)),
            }
            if candidate_equal and np.count_nonzero(common) >= 2:
                kc = k_cost[common]
                mc = main.costs[iteration, common].astype(np.float64)
                k_order = np.flatnonzero(common)[np.argsort(kc, kind="stable")]
                m_order = np.flatnonzero(common)[np.argsort(mc, kind="stable")]
                effective_k = min(top_k, k_order.size)
                row.update(
                    {
                        "cost_pearson": _correlation(kc, mc),
                        "rank_spearman": _correlation(
                            _average_ranks(kc), _average_ranks(mc)
                        ),
                        "top_k": effective_k,
                        "top_k_overlap_count": len(
                            set(k_order[:effective_k]) & set(m_order[:effective_k])
                        ),
                        "kamino_best_candidate": int(k_order[0]),
                        "main_best_candidate": int(m_order[0]),
                    }
                )
            else:
                row["cost_comparison"] = (
                    "not comparable: formulation-specific nominal makes candidate "
                    "tensors differ after the first optimizer iteration"
                )
            iteration_rows.append(row)

        kamino_action = kamino[-1]["first_action"].astype(np.float64)
        main_action = main.first_action[-1].astype(np.float64)
        delta = main_action - kamino_action
        denominator = np.linalg.norm(main_action) * np.linalg.norm(kamino_action)
        rows.append(
            {
                "model": model,
                "model_label": main.metadata["model_label"],
                "iterations": iteration_rows,
                "final_first_action_l2": float(np.linalg.norm(delta)),
                "final_first_action_linf": float(np.max(np.abs(delta))),
                "final_first_action_cosine": (
                    float(main_action @ kamino_action / denominator)
                    if denominator
                    else float("nan")
                ),
                "main_total_rollout_wall_s": float(np.sum(main.rollout_wall_s)),
                "main_final_first_action": main_action.tolist(),
                "kamino_final_first_action": kamino_action.tolist(),
                "cross_regret": "not computed by this fixed-candidate report",
            }
        )
    report = {
        "schema": "contact_study.kamino_comparison_report.v1",
        "claim_scope": manifest["claim_scope"],
        "reference_interpretation": (
            "simulator-defined offline Kamino/full-NCP reference; not real-world "
            "ground truth, global optimality, or online MPC"
        ),
        "run_directory": str(run_directory.resolve()),
        "budget": manifest["budget"],
        "bundle_fingerprint_sha256": manifest[
            "maximum_bundle_fingerprint_sha256"
        ],
        "models": rows,
    }
    destination = run_directory / "comparison_report.json"
    _write_json(destination, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=True))
    return report


def _add_shared(parser: argparse.ArgumentParser) -> None:
    paths = reference_paths()
    parser.add_argument(
        "--kamino-python", type=Path, default=paths["default_python"]
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "scenes"
            / "leap"
            / "env_leap_rollout_cube_high_high.xml"
        ),
    )


def _add_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--budget", default="G0_4x1")
    parser.add_argument("--models", nargs="+", default=list(MODEL_FACTORIES),
                        choices=list(MODEL_FACTORIES))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    _add_shared(validate)
    validate.add_argument("--check-cuda", action="store_true")

    main = sub.add_parser("run-main")
    _add_run(main)

    prepare = sub.add_parser("prepare-kamino")
    _add_shared(prepare)
    _add_run(prepare)
    prepare.add_argument("--chunk-size", type=int, default=4)

    workers = sub.add_parser("run-kamino")
    _add_shared(workers)
    workers.add_argument("--run-directory", required=True, type=Path)
    workers.add_argument("--gpu-index", type=int, default=0)
    workers.add_argument("--maximum-runtime-s", type=float, default=120.0)
    workers.add_argument("--max-new-tasks", type=int)
    workers.add_argument("--allow-display-gpu", action="store_true")
    workers.add_argument("--headless", action="store_true")

    report = sub.add_parser("report")
    report.add_argument("--run-directory", required=True, type=Path)
    report.add_argument("--models", nargs="+", default=list(MODEL_FACTORIES),
                        choices=list(MODEL_FACTORIES))
    report.add_argument("--top-k", type=int, default=5)

    smoke = sub.add_parser("smoke")
    _add_shared(smoke)
    _add_run(smoke)
    smoke.add_argument("--chunk-size", type=int, default=4)
    smoke.add_argument("--gpu-index", type=int, default=0)
    smoke.add_argument("--maximum-runtime-s", type=float, default=120.0)
    smoke.add_argument("--allow-display-gpu", action="store_true")
    smoke.add_argument("--headless", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "validate":
        result = validate_checkout(
            kamino_python=_executable_path(args.kamino_python),
            scene=args.scene.expanduser().resolve(),
            check_cuda=args.check_cuda,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ready"] else 1
    if args.command == "run-main":
        result = run_main_formulations(
            budget_label=args.budget,
            models=args.models,
            run_directory=args.run_directory.expanduser().resolve(),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-kamino":
        prepare_kamino(
            kamino_python=_executable_path(args.kamino_python),
            scene=args.scene.expanduser().resolve(),
            budget_label=args.budget,
            chunk_size=args.chunk_size,
            run_directory=args.run_directory.expanduser().resolve(),
        )
        return 0
    if args.command == "run-kamino":
        run_kamino_workers(
            kamino_python=_executable_path(args.kamino_python),
            run_directory=args.run_directory.expanduser().resolve(),
            gpu_index=args.gpu_index,
            maximum_runtime_s=args.maximum_runtime_s,
            max_new_tasks=args.max_new_tasks,
            allow_display_gpu=args.allow_display_gpu,
            headless=args.headless,
        )
        return 0
    if args.command == "report":
        build_report(
            models=args.models,
            run_directory=args.run_directory.expanduser().resolve(),
            top_k=args.top_k,
        )
        return 0
    if args.command == "smoke":
        run_directory = args.run_directory.expanduser().resolve()
        validation = validate_checkout(
            kamino_python=_executable_path(args.kamino_python),
            scene=args.scene.expanduser().resolve(),
            check_cuda=True,
        )
        if not validation["ready"]:
            print(json.dumps(validation, indent=2, sort_keys=True))
            return 1
        run_main_formulations(
            budget_label=args.budget,
            models=args.models,
            run_directory=run_directory,
        )
        prepare_kamino(
            kamino_python=_executable_path(args.kamino_python),
            scene=args.scene.expanduser().resolve(),
            budget_label=args.budget,
            chunk_size=args.chunk_size,
            run_directory=run_directory,
        )
        run_kamino_workers(
            kamino_python=_executable_path(args.kamino_python),
            run_directory=run_directory,
            gpu_index=args.gpu_index,
            maximum_runtime_s=args.maximum_runtime_s,
            max_new_tasks=None,
            allow_display_gpu=args.allow_display_gpu,
            headless=args.headless,
        )
        build_report(models=args.models, run_directory=run_directory)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
