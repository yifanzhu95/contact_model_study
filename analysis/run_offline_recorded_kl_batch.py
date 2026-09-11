"""Resumable episode-level batch runner for recorded-log offline KL."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ANALYSIS = Path(__file__).resolve().parent
REPO = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))

from offline_recorded_kl_common import (
    DEFAULT_CONVERGENCE_TOL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_MEASUREMENTS,
    DEFAULT_REFERENCE_SAMPLES,
    DEFAULT_REFERENCE_TEMPERATURE,
    DEFAULT_SHRINKAGE,
    SCHEMA,
    atomic_json,
    load_cells_manifest,
    output_filename,
    scientific_runtime_provenance,
    sha256_file,
)

WORKER = ANALYSIS / "run_offline_recorded_kl_episode.py"
COMMON = ANALYSIS / "offline_recorded_kl_common.py"
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--episodes-per-cell", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-new-episodes", type=int)
    parser.add_argument("--launch-budget-seconds", type=float)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-measurements", type=int, default=DEFAULT_MAX_MEASUREMENTS)
    parser.add_argument("--ref-samples", type=int, default=DEFAULT_REFERENCE_SAMPLES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_REFERENCE_TEMPERATURE)
    parser.add_argument("--convergence-tol", type=float, default=DEFAULT_CONVERGENCE_TOL)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE)
    return parser


def validate_args(args, parser):
    if not args.source_dir.is_dir() or not args.manifest.is_file():
        parser.error("source-dir and manifest must exist")
    if args.episodes_per_cell is not None and args.episodes_per_cell < 1:
        parser.error("episodes-per-cell must be positive when set")
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.max_new_episodes is not None and args.max_new_episodes < 1:
        parser.error("max-new-episodes must be positive when set")
    if args.launch_budget_seconds is not None and args.launch_budget_seconds <= 0:
        parser.error("launch-budget-seconds must be positive when set")
    if args.max_measurements < 1 or args.ref_samples < 1 or args.max_iterations < 2:
        parser.error("max-measurements/ref-samples must be positive and max-iterations >= 2")
    if not math.isfinite(args.temperature) or not math.isfinite(args.convergence_tol):
        parser.error("temperature and convergence-tol must be finite")
    if args.temperature <= 0 or args.convergence_tol <= 0 or not 0 < args.shrinkage <= 1:
        parser.error("temperature/tolerance must be positive and shrinkage must lie in (0, 1]")


def expected_settings(args):
    return {
        "n_samples": args.ref_samples,
        "temperature": args.temperature,
        "convergence_tol": args.convergence_tol,
        "max_iterations": args.max_iterations,
        "shrinkage": args.shrinkage,
        "max_measurements": args.max_measurements,
    }


def validate_cached(path, item, settings, provenance):
    record = json.loads(path.read_text())
    if not record.get("completed"):
        return False
    if record.get("schema") != SCHEMA:
        raise ValueError(f"Cached output has unknown schema: {path}")
    checks = {
        "source": item["cell"]["file"],
        "source_sha256": item["cell"]["sha256"],
        "episode": item["episode"],
        "analysis_script_sha256": provenance["worker_sha256"],
        "analysis_support_sha256": provenance["common_sha256"],
        "shrinkage": settings["shrinkage"],
        "reference_cross_state_history": "none",
        "source_outcome_unchanged": True,
        "closed_loop_rerun": False,
    }
    for key, expected in checks.items():
        if record.get(key) != expected:
            raise ValueError(f"Cached {path.name} has different {key}")
    expected_runtime = provenance["scientific_by_source"][item["cell"]["file"]]
    if record.get("scientific_runtime_provenance") != expected_runtime:
        raise ValueError(f"Cached {path.name} uses different scientific code or scene assets")
    reference = record["reference_config"]
    for key in ("n_samples", "temperature", "convergence_tol", "max_iterations"):
        if reference[key] != settings[key]:
            raise ValueError(f"Cached {path.name} has different reference {key}")
    if record["selection"]["max_measurements"] != settings["max_measurements"]:
        raise ValueError(f"Cached {path.name} has a different measurement cap")
    return True


def archive_incomplete(path, outdir):
    archive = outdir / "interrupted_attempts"
    archive.mkdir(exist_ok=True)
    target = archive / f"{path.stem}_{time.time_ns()}{path.suffix}"
    path.rename(target)
    log = path.with_suffix(".log")
    if log.exists():
        log.rename(archive / f"{log.stem}_{time.time_ns()}{log.suffix}")


def archive_orphan_log(log, outdir):
    """Preserve a log from a crash that happened before the first checkpoint."""
    if not log.exists():
        return
    archive = outdir / "interrupted_attempts"
    archive.mkdir(exist_ok=True)
    log.rename(archive / f"{log.stem}_{time.time_ns()}{log.suffix}")


def run_item(item, args, settings, provenance):
    output = args.outdir / output_filename(item["cell"]["file"], item["episode"])
    base = {
        "source": item["cell"]["file"],
        "source_sha256": item["cell"]["sha256"],
        "model": item["cell"]["model"],
        "episode": item["episode"],
        "output": str(output.resolve()),
    }
    if output.exists():
        if validate_cached(output, item, settings, provenance):
            record = json.loads(output.read_text())
            return {
                **base,
                "status": "cached",
                **record["summary"],
                "wall_seconds": 0.0,
            }
        archive_incomplete(output, args.outdir)

    source = args.source_dir / item["cell"]["file"]
    command = [
        args.python,
        "-u",
        str(WORKER),
        "--source-json", str(source),
        "--episode", str(item["episode"]),
        "--outdir", str(args.outdir),
        "--expected-source-sha256", item["cell"]["sha256"],
        "--max-measurements", str(settings["max_measurements"]),
        "--ref-samples", str(settings["n_samples"]),
        "--temperature", str(settings["temperature"]),
        "--convergence-tol", str(settings["convergence_tol"]),
        "--max-iterations", str(settings["max_iterations"]),
        "--shrinkage", str(settings["shrinkage"]),
    ]
    environment = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MUJOCO_GL="egl",
    )
    log = output.with_suffix(".log")
    archive_orphan_log(log, args.outdir)
    started = time.monotonic()
    with log.open("w") as stream:
        process = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode != 0 or not output.exists():
        return {**base, "status": "failed", "exit_code": process.returncode,
                "wall_seconds": time.monotonic() - started, "command": command}
    if not validate_cached(output, item, settings, provenance):
        raise ValueError(f"Worker returned an incomplete output: {output}")
    record = json.loads(output.read_text())
    return {
        **base,
        "status": "completed",
        "exit_code": process.returncode,
        "wall_seconds": time.monotonic() - started,
        "command": command,
        **record["summary"],
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    _, cells = load_cells_manifest(args.manifest)
    if args.episodes_per_cell is None:
        episode_counts = {int(cell["n"]) for cell in cells}
        if len(episode_counts) != 1:
            raise ValueError(
                "Manifest cells have different episode counts; set "
                "--episodes-per-cell explicitly to choose a balanced prefix"
            )
        args.episodes_per_cell = episode_counts.pop()
    if args.episodes_per_cell > min(cell["n"] for cell in cells):
        raise ValueError("episodes-per-cell exceeds at least one source cell")
    for cell in cells:
        path = args.source_dir / cell["file"]
        if not path.is_file() or sha256_file(path) != cell["sha256"]:
            raise ValueError(f"Missing or hash-mismatched source: {path}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    settings = expected_settings(args)
    provenance = {
        "worker_sha256": sha256_file(WORKER),
        "common_sha256": sha256_file(COMMON),
        "scientific_by_source": {
            cell["file"]: scientific_runtime_provenance(
                REPO, cell["geometry"], cell["model"]
            ) for cell in cells
        },
    }
    items = [
        {"cell": cell, "episode": episode}
        for episode in range(args.episodes_per_cell)
        for cell in cells
    ]
    manifest_path = args.outdir / "execution_manifest.json"
    with (args.outdir / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest_path.exists():
            prior = args.outdir / f"execution_manifest_{time.time_ns()}.prior.json"
            shutil.copy2(manifest_path, prior)
        manifest = {
            "status": "running",
            "source_directory": str(args.source_dir.resolve()),
            "input_manifest": str(args.manifest.resolve()),
            "input_manifest_sha256": sha256_file(args.manifest),
            "requested_cells": len(cells),
            "requested_episodes_per_cell": args.episodes_per_cell,
            "requested_episode_outputs": len(items),
            "reference_preset": settings,
            "protocol": "independent zero-mean restart at every selected recorded state",
            "sample_order": "episode index first, then input-manifest cell order",
            "workers": args.workers,
            "worker": str(WORKER),
            "worker_sha256": provenance["worker_sha256"],
            "analysis_files_sha256": {
                str(WORKER.relative_to(REPO)): provenance["worker_sha256"],
                str(COMMON.relative_to(REPO)): provenance["common_sha256"],
            },
            "scientific_runtime_provenance_by_source": provenance["scientific_by_source"],
            "source_outcomes_unchanged": True,
            "closed_loop_reruns": False,
            "jobs": [],
        }
        atomic_json(manifest_path, manifest)
        started = time.monotonic()
        position = 0
        newly_launched = 0
        failure = False
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            active = {}
            while active or position < len(items):
                while position < len(items) and len(active) < args.workers:
                    budget_hit = (
                        args.launch_budget_seconds is not None
                        and time.monotonic() - started >= args.launch_budget_seconds
                    )
                    count_hit = (
                        args.max_new_episodes is not None
                        and newly_launched >= args.max_new_episodes
                    )
                    if budget_hit or count_hit:
                        break
                    item = items[position]
                    output = args.outdir / output_filename(item["cell"]["file"], item["episode"])
                    is_cached = output.exists() and validate_cached(
                        output, item, settings, provenance
                    )
                    active[pool.submit(run_item, item, args, settings, provenance)] = item
                    position += 1
                    if not is_cached:
                        newly_launched += 1
                manifest.update(
                    active=[{"source": item["cell"]["file"], "episode": item["episode"]}
                            for item in active.values()],
                    not_scheduled=len(items) - position,
                    newly_launched=newly_launched,
                    wall_seconds=time.monotonic() - started,
                )
                atomic_json(manifest_path, manifest)
                if not active:
                    break
                finished, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                for future in finished:
                    item = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {"source": item["cell"]["file"], "episode": item["episode"],
                                  "status": "failed", "error": repr(error)}
                    failure |= result["status"] == "failed"
                    manifest["jobs"].append(result)
                    print(json.dumps(result), flush=True)
        manifest.update(
            status="attention_required" if failure else (
                "completed" if position == len(items) else "launch_limit_reached"
            ),
            active=[],
            not_scheduled=len(items) - position,
            newly_launched=newly_launched,
            wall_seconds=time.monotonic() - started,
        )
        atomic_json(manifest_path, manifest)
        print("FINISHED", manifest["status"], flush=True)


if __name__ == "__main__":
    main()
