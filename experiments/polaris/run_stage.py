#!/usr/bin/env python3
"""Guarded G1 calibration, G2 pooled throughput, and G3 four-GPU pilot.

--check only parses and prints workloads. Execution requires a clean source,
a matching compute-node preflight, and evidence from the preceding stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from ContactModelStudy.Utils.Provenance import git_provenance  # noqa: E402


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass
class Cell:
    name: str
    row: dict
    gpus: str
    workers: int | None = None

    def spec(self):
        return {"name": self.name, "row": self.row, "gpus": self.gpus, "workers": self.workers}


def batch_module():
    spec = importlib.util.spec_from_file_location("polaris_batches", REPO / "experiments/run_episode_batches.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cells_for(stage, samples):
    with (PACKAGE / "m5_smoke.csv").open() as stream:
        base = next(csv.DictReader(stream))
    base.update(debug="false", stop_on_success="false", save_steps="true")
    if stage in ("g1", "g3"):
        with (PACKAGE / f"{stage}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        return [Cell(row["label"], {**base, **row,
                     **({"n_samples": str(samples)} if stage == "g3" else {})},
                     "0" if stage == "g1" else str(i % 4)) for i, row in enumerate(rows)]
    row = {**base, "n_samples": str(samples), "n_episodes": "24", "steps": "100", "seed": "0"}
    return [Cell(f"g2_gpu{len(gpus.split(','))}_w{workers}", row.copy(), gpus, workers)
            for gpus, workers in (("0", 4), ("0,1,2,3", 4), ("0,1,2,3", 8), ("0,1,2,3", 16))]


def command(cell, attempt, batches):
    argv = batches.rowToArgv(cell.row, attempt, "result")
    if cell.workers is None:
        return [sys.executable, str(REPO / "ContactModelStudy/Drivers/run_episodes.py"), *argv]
    return [sys.executable, str(REPO / "ContactModelStudy/Drivers/run_episodes_pooled.py"), *argv,
            "--gpus", cell.gpus, "--planners-per-gpu", "1", "--workers", str(cell.workers),
            "--checkpoint-every", "1", "--pool-stats", str(attempt / "pool_stats.json")]


def assess_result(document, cell, expected_sha, pool_stats=None):
    """Completion and scientific validity are independent required gates."""
    reasons = []
    source = document.get("configs", {}).get("metadata", {}).get("provenance", {}).get("source", {})
    if source.get("commit") != expected_sha or source.get("dirty") is not False:
        reasons.append("source_mismatch_or_dirty")
    validation = document.get("validation", {})
    if validation.get("status") != "pass" or validation.get("scaling_eligible") is not True:
        reasons.append("solver_validation_failed_or_missing")
    episodes = document.get("episodes", [])
    expected = int(cell.row["n_episodes"])
    if len(episodes) != expected or document.get("n_episodes") != expected:
        reasons.append("incomplete_episode_count")
    summaries = [e.get("summary", {}) for e in episodes]
    if any(s.get("finish_reason") in (None, "interrupted", "error") for s in summaries):
        reasons.append("episode_interrupted_or_error")
    if any(int(s.get("n_steps", 0)) <= 0 for s in summaries):
        reasons.append("empty_episode")
    if cell.workers is not None:
        if sorted(s.get("episode", -1) for s in summaries) != list(range(expected)):
            reasons.append("episode_ids_incomplete")
        used = {g for s in summaries for g in s.get("planner_validation", {}).get("planner_devices", [])}
        if used != set(cell.gpus.split(",")):
            reasons.append("not_all_requested_gpus_served_plans")
        if not pool_stats or pool_stats.get("complete") is not True:
            reasons.append("pool_incomplete_or_missing_stats")
        elif pool_stats.get("pool") != {"gpus": cell.gpus.split(","), "workers": cell.workers, "planners_per_gpu": 1}:
            reasons.append("pool_topology_mismatch")
        from ContactModelStudy.Drivers.EpisodePool import episodeSeeds
        if any(s.get("goal_seed") != episodeSeeds(int(cell.row["seed"]), s.get("episode", -1))[0]
               for s in summaries if int(s.get("episode", -1)) >= 0):
            reasons.append("episode_goal_seed_mismatch")
    return reasons


def require_previous(stage, path, expected_sha, samples):
    previous = read_json(path)
    if previous.get("expected_source_sha") != expected_sha:
        raise ValueError("previous-stage source SHA differs; rerun gates for this deployment version")
    if stage == "g1":
        if previous.get("schema") != "contact_model_study.polaris_smoke_verdict.v1" or previous.get("passed") is not True:
            raise ValueError("G1 requires a passing G0 smoke verdict")
    elif stage == "g2":
        if previous.get("stage") != "g1" or previous.get("source_unchanged") is not True \
                or samples not in previous.get("validated_samples", []):
            raise ValueError("G2 requires the chosen N to have passed all three G1 seeds")
    elif previous.get("stage") != "g2" or previous.get("passed") is not True or previous.get("n_samples") != samples:
        raise ValueError("G3 requires passing G2 at the same chosen N")
    return previous


class Telemetry:
    """Sample assigned physical GPU IDs; sampled memory is not an exact peak."""

    def __init__(self, path, gpus):
        self.path, self.gpus = path, gpus
        self.rows, self.errors = [], []
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        fields = "index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        result = subprocess.run(["nvidia-smi", "-i", self.gpus, f"--query-gpu={fields}",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        rows = []
        for parts in csv.reader(result.stdout.splitlines()):
            if len(parts) != 7:
                raise ValueError("unexpected nvidia-smi telemetry row")
            gpu, uuid, util, used, total, temperature, power = [p.strip() for p in parts]
            rows.append({"time_unix": time.time(), "gpu": gpu, "uuid": uuid,
                         "utilization_percent": float(util), "memory_mib": float(used),
                         "total_memory_mib": float(total), "temperature_c": float(temperature),
                         "power_w": None if power in ("[N/A]", "N/A", "[Not Supported]") else float(power)})
        if {r["gpu"] for r in rows} != set(self.gpus.split(",")):
            raise ValueError("GPU telemetry did not return the assigned device IDs")
        self.rows.extend(rows)
        with self.path.open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def start(self):
        self.sample()  # Refuse to launch a workload with unavailable telemetry.
        def loop():
            while not self.stop.wait(1.0):
                try:
                    self.sample()
                except Exception as exc:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                    return
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=6)
        metrics = {}
        for gpu in self.gpus.split(","):
            rows = [r for r in self.rows if r["gpu"] == gpu]
            metrics[gpu] = {
                "samples": len(rows),
                "mean_utilization_percent": statistics.mean(r["utilization_percent"] for r in rows) if rows else None,
                "sampled_peak_memory_mib": max((r["memory_mib"] for r in rows), default=None),
            }
        return {"per_gpu": metrics, "errors": self.errors}


class RunningCell:
    def __init__(self, cell, directory, batches, expected_sha):
        self.cell, self.directory, self.expected_sha = cell, directory, expected_sha
        directory.mkdir(parents=True, exist_ok=True)
        number = 1
        while (directory / f"attempt_{number:04d}").exists():
            number += 1
        self.attempt = directory / f"attempt_{number:04d}"
        self.attempt.mkdir()
        self.log = (self.attempt / "driver.log").open("x")
        self.telemetry = Telemetry(self.attempt / "gpu_metrics.jsonl", cell.gpus)
        self.started, self.stop_sent = time.monotonic(), None
        self.proc, self.start_error = None, None
        self.argv = command(cell, self.attempt, batches)
        write_json(directory / "cell.status.json", {
            "execution_status": "running", "spec": cell.spec(), "expected_source_sha": expected_sha,
            "spec_sha256": fingerprint(cell.spec()), "attempt": str(self.attempt), "argv": self.argv})
        try:
            self.telemetry.start()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=cell.gpus, PYTHONUNBUFFERED="1")
            self.proc = subprocess.Popen(self.argv, cwd=REPO, env=env, stdout=self.log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        except Exception as exc:
            self.start_error = f"{type(exc).__name__}: {exc}"

    def cancel(self):
        if not self.proc or self.proc.poll() is not None:
            return
        if self.stop_sent is None:
            self.stop_sent = time.monotonic()
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif time.monotonic() - self.stop_sent > 90:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def ready(self):
        return self.proc is None or self.proc.poll() is not None

    def finish(self):
        rc = self.proc.wait() if self.proc else 127
        self.log.close()
        wall = time.monotonic() - self.started
        telemetry = self.telemetry.finish()
        reasons, document, stats = [], {}, None
        try:
            document = read_json(self.attempt / "result.json")
            if self.cell.workers is not None:
                stats = read_json(self.attempt / "pool_stats.json")
            reasons.extend(assess_result(document, self.cell, self.expected_sha, stats))
        except (OSError, ValueError, TypeError) as exc:
            reasons.append(f"result_unreadable: {exc}")
        if rc:
            reasons.append(f"driver_exit_{rc}")
        if self.stop_sent is not None:
            reasons.append("cancelled_or_budget_exhausted")
        if self.start_error or telemetry["errors"] or not telemetry["per_gpu"]:
            reasons.append("telemetry_or_launch_failed")
        summaries = [e.get("summary", {}) for e in document.get("episodes", [])]
        valid_episodes = sum(s.get("planner_validation", {}).get("status") == "pass"
                             and s.get("finish_reason") not in (None, "interrupted", "error") for s in summaries)
        means = [s["plan_s_mean"] for s in summaries if s.get("plan_s_mean") is not None]
        result = {
            "execution_status": "completed" if rc == 0 and self.stop_sent is None else "failed",
            "passed": not reasons, "blocking_reasons": reasons, "returncode": rc,
            "start_error": self.start_error, "spec": self.cell.spec(),
            "spec_sha256": fingerprint(self.cell.spec()), "expected_source_sha": self.expected_sha,
            "attempt": str(self.attempt), "argv": self.argv, "wall_s": wall,
            "valid_episodes": valid_episodes, "valid_episodes_per_hour": valid_episodes * 3600 / wall,
            "success_rate": document.get("success_rate"), "validation": document.get("validation"),
            "mean_episode_plan_s_after_first": statistics.mean(means) if means else None,
            "planner_request_wait_s": sum(s.get("planner_request_wait_s", 0) for s in summaries),
            "planner_wait_overhead_s": sum(s.get("planner_wait_overhead_s", 0) for s in summaries),
            "telemetry": telemetry, "pool_stats": stats,
        }
        write_json(self.directory / "cell.status.json", result)
        return result


def cached_result(directory, cell, expected_sha):
    try:
        status = read_json(directory / "cell.status.json")
        if status.get("passed") is not True or status.get("expected_source_sha") != expected_sha \
                or status.get("spec_sha256") != fingerprint(cell.spec()):
            return None
        attempt = Path(status["attempt"])
        stats = read_json(attempt / "pool_stats.json") if cell.workers is not None else None
        if assess_result(read_json(attempt / "result.json"), cell, expected_sha, stats):
            return None
        return status
    except (OSError, ValueError, KeyError, TypeError):
        return None


def verdict(stage, cells, results, expected_sha, samples):
    passed = len(results) == len(cells) and all(r.get("passed") is True for r in results.values())
    valid_samples = []
    if stage == "g1":
        for n in (32, 64, 128, 256):
            group = [c for c in cells if int(c.row["n_samples"]) == n]
            if len(group) == 3 and all(results.get(c.name, {}).get("passed") is True for c in group):
                valid_samples.append(n)
    speedups = {}
    baseline = results.get("g2_gpu1_w4", {})
    if stage == "g2" and baseline.get("passed"):
        speedups = {name: baseline["wall_s"] / r["wall_s"] for name, r in results.items() if r.get("passed")}
    return {"schema": "contact_model_study.polaris_stage_verdict.v1", "stage": stage,
            "passed": passed, "scaling_eligible": passed, "expected_source_sha": expected_sha,
            "n_samples": samples, "validated_samples": valid_samples, "speedup_vs_gpu1_w4": speedups,
            "pending_cells": [c.name for c in cells if c.name not in results], "cells": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("g1", "g2", "g3"))
    parser.add_argument("--n-samples", type=int, choices=(32, 64, 128, 256))
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--expected-sha")
    parser.add_argument("--budget-seconds", type=int, default=3000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.stage != "g1" and args.n_samples is None:
        parser.error("G2/G3 require an explicitly selected --n-samples from validated G1 evidence")
    if args.budget_seconds < 1 or args.budget_seconds > 3000:
        parser.error("--budget-seconds must be between 1 and 3000 for the one-hour debug job")
    cells = cells_for(args.stage, args.n_samples)
    batches = batch_module()
    for cell in cells:
        cmd = command(cell, args.outdir / cell.name / "attempt_preview", batches)
        if cell.workers is None:
            error = batches.checkRow(cmd[2:])
            if error:
                parser.error(error)
        else:
            from ContactModelStudy.Drivers import run_episodes as drv
            from ContactModelStudy.Drivers.run_episodes_pooled import build_parser
            drv.parseArgs(cmd[2:], parser=build_parser())
    plan = {"stage": args.stage, "n_samples": args.n_samples, "cells": [c.spec() for c in cells]}
    if args.check:
        if args.previous or args.expected_sha:
            if not args.previous or not args.expected_sha:
                parser.error("previous verdict checking needs both --previous and --expected-sha")
            try:
                require_previous(args.stage, args.previous, args.expected_sha, args.n_samples)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
        print(json.dumps(plan, indent=2))
        return 0
    if not args.expected_sha or not args.previous:
        parser.error("execution requires --expected-sha and --previous")
    args.outdir = args.outdir.resolve()
    source = git_provenance(REPO)
    if source["commit"] != args.expected_sha or source["dirty"] is not False:
        parser.error("execution requires the exact clean local deployment commit")
    try:
        require_previous(args.stage, args.previous, args.expected_sha, args.n_samples)
        preflight = read_json(args.outdir / "preflight.json")
        expected_gpus = "0" if args.stage == "g1" else "0,1,2,3"
        if preflight.get("passed") is not True or preflight.get("expected_source_sha") != args.expected_sha \
                or preflight.get("requested_gpus") != expected_gpus \
                or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpus:
            raise ValueError("passing compute-node preflight and stage GPU visibility are required")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    plan.update(expected_source_sha=args.expected_sha, previous=str(args.previous.resolve()))
    plan_path = args.outdir / "stage_plan.json"
    if plan_path.exists() and (not args.resume or read_json(plan_path) != plan):
        parser.error("existing stage requires --resume with an identical plan and previous verdict path")
    write_json(plan_path, plan)
    results, pending, running = {}, [], []
    for cell in cells:
        cached = cached_result(args.outdir / cell.name, cell, args.expected_sha) if args.resume else None
        if cached:
            results[cell.name] = cached
        else:
            pending.append(cell)
    deadline, stopped, interrupted = time.monotonic() + args.budget_seconds, False, False
    previous_handlers = {}
    def stop(signum, frame):
        nonlocal stopped, interrupted
        stopped, interrupted = True, True
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, stop)
    try:
        while pending or running:
            if time.monotonic() >= deadline:
                stopped = True
            if not stopped:
                used = {g for active in running for g in active.cell.gpus.split(",")}
                for cell in list(pending):
                    if (args.stage != "g3" and running) or used.intersection(cell.gpus.split(",")):
                        continue
                    running.append(RunningCell(cell, args.outdir / cell.name, batches, args.expected_sha))
                    pending.remove(cell)
                    used.update(cell.gpus.split(","))
                    print(f"started {cell.name} on GPU(s) {cell.gpus}", flush=True)
            for active in list(running):
                if interrupted or time.monotonic() >= deadline:
                    active.cancel()
                if active.ready():
                    result = active.finish()
                    running.remove(active)
                    results[active.cell.name] = result
                    stopped = stopped or not result["passed"]
                    print(f"finished {active.cell.name}: passed={result['passed']}", flush=True)
                    write_json(args.outdir / "stage_verdict.json",
                               verdict(args.stage, cells, results, args.expected_sha, args.n_samples))
            if stopped and not running:
                break
            time.sleep(0.2)
    except BaseException:
        interrupted = True
        raise
    finally:
        while running:
            for active in list(running):
                active.cancel()
                if active.ready():
                    results[active.cell.name] = active.finish()
                    running.remove(active)
            if running:
                time.sleep(0.2)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        final = verdict(args.stage, cells, results, args.expected_sha, args.n_samples)
        final["interrupted"] = interrupted
        last_source = git_provenance(REPO)
        final["source_unchanged"] = last_source["commit"] == args.expected_sha and last_source["dirty"] is False
        if not final["source_unchanged"] or interrupted:
            final["passed"] = final["scaling_eligible"] = False
        write_json(args.outdir / "stage_verdict.json", final)
    return 0 if final["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
