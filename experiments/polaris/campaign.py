#!/usr/bin/env python3
"""Full G1-G3 exploratory campaign in ONE Polaris PBS allocation.

The matrix is fixed, not an adaptive search. Execution, task success, strict
numerical qualification and performance eligibility are independent outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

REPO = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PACKAGE))
import direct  # noqa: E402
import run_stage as stages  # noqa: E402
from ContactModelStudy.Drivers.EpisodePool import episodeSeeds  # noqa: E402
from ContactModelStudy.Utils.Provenance import git_provenance  # noqa: E402

write_json, read_json, fingerprint = stages.write_json, stages.read_json, stages.fingerprint
STAGES = ("g1", "g2", "g3")
TERMINAL_EPISODES = ("success", "failure", "timeout", "planner_failure")
FATAL_LOG_MARKERS = ("No space left on device", "Input/output error", "Read-only file system",
                     "ModuleNotFoundError:", "ImportError:", "CUDA_ERROR_DEVICE_LOST",
                     "CUDA driver version is insufficient", "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix():
    return {s: stages.cells_for(s, 64) for s in STAGES}


def scientific_plan():
    return {"stages": {s: [c.spec() for c in cells] for s, cells in matrix().items()},
            "n_samples_g2_g3": 64, "padmm_max_iterations": 100, "padmm_tolerance": 5e-4,
            "acceptance_policy": "exploratory", "strict_qualification_always_recorded": True,
            "all_unusable": "end_episode_before_action", "automatic_retry": False,
            "finite_residual_miss": "warning_not_candidate_rejection",
            "contact_capacity_hit": "warning_not_proven_overflow"}


def make_plan(args, sha):
    seconds = direct.wall_seconds(args.walltime)
    limit = 3600 if args.queue == "debug" else 168 * 3600
    if not 900 <= seconds <= limit:
        raise ValueError("walltime must be >=15 minutes and within the queue's documented maximum")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.project):
        raise ValueError("project must be an existing ALCF allocation/account name")
    science = scientific_plan()
    return {"schema": "contact_model_study.polaris_campaign.v1", "source": str(REPO),
            "expected_sha": sha, "environment": str(args.env.resolve()), "project": args.project,
            "queue": args.queue, "walltime": args.walltime, "budget_seconds": seconds - 600,
            "wrapper": "run_campaign.pbs", "science": science, "science_sha256": fingerprint(science),
            "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
            "requested_cells": 43, "requested_episodes": 135, "max_control_plans": 12540}


def driver_command(cell, attempt, batches):
    return stages.command(cell, attempt, batches) + ["--kamino-max-iterations", "100",
            "--kamino-acceptance", "exploratory", "--stop-on-invalid-plan"]


def validate_commands():
    from ContactModelStudy.Drivers import run_episodes as drv
    from ContactModelStudy.Drivers.run_episodes_pooled import build_parser
    batches = stages.batch_module()
    for cells in matrix().values():
        for cell in cells:
            args = drv.parseArgs(driver_command(cell, Path("/preview"), batches)[2:],
                                 parser=build_parser() if cell.workers else None)
            task, _ = drv.buildTasks(args)
            from ContactModelStudy.Utils.ContactModelPresets import _presetConfig
            cfg = _presetConfig("M5", **drv.rolloutSimOverrides(args, task))
            if cfg.padmm_max_iterations != 100 or cfg.padmm_tolerance != 5e-4 \
                    or cfg.resolved_horizon != 5 or cfg.resolved_substeps != 16:
                raise ValueError("scientific defaults no longer match the reviewed campaign")


def assess(document, stats, cell, sha, rc):
    """Never confuse an underconverged, failed task with an incomplete process."""
    reasons = []
    source = document.get("configs", {}).get("metadata", {}).get("provenance", {}).get("source", {})
    if source.get("commit") != sha or source.get("dirty") is not False:
        reasons.append("source_mismatch")
    config = document.get("configs", {}).get("rollout_simulator", {}).get("config", {})
    if config.get("rollout_acceptance") != "exploratory" or config.get("padmm_max_iterations") != 100 \
            or config.get("padmm_tolerance") != 5e-4:
        reasons.append("numerical_policy_mismatch")
    episodes = document.get("episodes", [])
    summaries = [e.get("summary", {}) for e in episodes]
    expected = int(cell.row["n_episodes"])
    if document.get("n_episodes") != expected or len(episodes) != expected \
            or sorted(s.get("episode", -1) for s in summaries) != list(range(expected)):
        reasons.append("incomplete_episode_ids")
    if any(s.get("finish_reason") not in TERMINAL_EPISODES for s in summaries):
        reasons.append("episode_error_or_interruption")
    if rc:
        reasons.append(f"driver_exit_{rc}")
    used = {g for s in summaries for g in s.get("planner_validation", {}).get("planner_devices", [])}
    if cell.workers:
        if not stats or stats.get("complete") is not True:
            reasons.append("pool_incomplete")
        elif stats.get("pool") != {"gpus": cell.gpus.split(","), "workers": cell.workers, "planners_per_gpu": 1}:
            reasons.append("pool_topology_mismatch")
        for s in summaries:
            ep = s.get("episode", -1)
            if isinstance(ep, int) and ep >= 0:
                goal, noise = episodeSeeds(int(cell.row["seed"]), ep)
                if s.get("goal_seed") != goal or s.get("noise_seed") != noise:
                    reasons.append("paired_seed_mismatch")
                    break
    validation = document.get("validation", {})
    strict_ok = validation.get("status") == "pass" and validation.get("scaling_eligible") is True
    steps = [int(s.get("n_steps", 0)) for s in summaries]
    rejected = sum(s.get("rejected_plan") is not None for s in summaries)
    all_gpus = not cell.workers or used == set(cell.gpus.split(","))
    qualities = [s.get("planner_validation", {}) for s in summaries]
    valid_worlds = sum(int(q.get("valid_worlds", 0)) for q in qualities)
    invalid_worlds = sum(int(q.get("invalid_worlds", 0)) for q in qualities)
    return {"execution_status": "incomplete" if reasons else "completed", "execution_reasons": reasons,
            "global_failure": (bool(source) and "source_mismatch" in reasons)
                or (bool(config) and "numerical_policy_mismatch" in reasons),
            "validation_status": "pass" if strict_ok else "fail", "strict_qualification": validation,
            "reference_eligible": not reasons and strict_ok, "all_requested_gpus_observed": all_gpus,
            "planner_devices_observed": sorted(used), "recorded_episodes": len(episodes),
            "task_successes_observed": sum(bool(s.get("success")) for s in summaries),
            "task_failures": sum(s.get("finish_reason") == "failure" for s in summaries),
            "planner_failures": sum(s.get("finish_reason") == "planner_failure" for s in summaries),
            "numerical_quality": {"strict_valid_worlds": valid_worlds, "strict_invalid_worlds": invalid_worlds,
                "strict_invalid_fraction": invalid_worlds / (valid_worlds + invalid_worlds) if valid_worlds + invalid_worlds else None,
                "hard_unusable_worlds": sum(int(q.get("unusable_worlds", 0)) for q in qualities),
                "max_primal_residual": max((q.get("max_primal_residual", 0) for q in qualities), default=0),
                "max_dual_residual": max((q.get("max_dual_residual", 0) for q in qualities), default=0),
                "max_complementarity_residual": max((q.get("max_complementarity_residual", 0) for q in qualities), default=0)},
            "control_steps_applied": sum(steps), "plans_attempted": sum(steps) + rejected,
            "work_signature": [{"episode": s.get("episode"), "steps": s.get("n_steps"),
                               "finish_reason": s.get("finish_reason"), "goal_seed": s.get("goal_seed"),
                               "noise_seed": s.get("noise_seed")} for s in summaries],
            "full_step_budget_completed": len(steps) == expected and all(n == int(cell.row["steps"]) for n in steps),
            "first_plan_s": [s.get("plan_s_first") for s in summaries],
            "mean_plan_s_after_first": [s.get("plan_s_mean") for s in summaries],
            "applied_plan_s_total": sum(float(s.get("applied_plan_s_total", 0)) for s in summaries),
            "planner_request_wait_s": sum(float(s.get("planner_request_wait_s", 0)) for s in summaries),
            "planner_wait_overhead_s": sum(float(s.get("planner_wait_overhead_s", 0)) for s in summaries),
            "planner_initialization_s": (stats or {}).get("process_stats", {}).get("planner_initialization_s", {})}


class Telemetry(stages.Telemetry):
    """Telemetry is observational. Losing a sample never changes an experiment."""
    def start(self):
        def sample_safely():
            try:
                self.sample()
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
        sample_safely()
        def loop():
            while not self.stop.wait(1):
                sample_safely()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()


def artifact_hashes(attempt):
    return {str(p.relative_to(attempt)): sha256(p) for p in sorted(attempt.rglob("*")) if p.is_file()}


class Active(stages.RunningCell):
    def __init__(self, cell, directory, batches, sha):
        self.cell, self.directory, self.expected_sha = cell, directory, sha
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
        self.argv = driver_command(cell, self.attempt, batches)
        write_json(self.attempt / "command.json", self.argv)
        write_json(directory / "cell.status.json", {"execution_status": "running", "spec": cell.spec(),
                   "spec_sha256": fingerprint(cell.spec()), "expected_source_sha": sha, "attempt": str(self.attempt)})
        try:
            self.telemetry.start()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=cell.gpus, PYTHONUNBUFFERED="1")
            self.proc = subprocess.Popen(self.argv, cwd=REPO, env=env, stdout=self.log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        except Exception as exc:
            self.start_error = f"{type(exc).__name__}: {exc}"

    def finish(self):
        rc = self.proc.wait() if self.proc else 127
        self.log.close()
        telemetry = self.telemetry.finish()
        document, stats = {}, None
        try:
            document = read_json(self.attempt / "result.json")
            stats = read_json(self.attempt / "pool_stats.json") if self.cell.workers else None
        except (FileNotFoundError, ValueError):
            pass
        try:
            result = assess(document, stats, self.cell, self.expected_sha, rc)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            result = assess({}, None, self.cell, self.expected_sha, rc or 1)
            result["execution_reasons"].append(f"malformed_result:{type(exc).__name__}:{exc}")
        if self.stop_sent is not None or self.start_error:
            result["execution_status"] = "incomplete"
            result["execution_reasons"].append("cancelled" if self.stop_sent else "launch_failed")
        log = (self.attempt / "driver.log").read_text(errors="replace")
        markers = [s for s in FATAL_LOG_MARKERS if s in log]
        # An ordinary isolated failure or OOM is recorded and does not remove
        # other planned cells. Known infrastructure failures stop all work.
        result["global_failure"] = result["global_failure"] or bool(markers) or self.start_error is not None
        result.update(returncode=rc, start_error=self.start_error, fatal_log_markers=markers,
                      spec=self.cell.spec(), spec_sha256=fingerprint(self.cell.spec()),
                      expected_source_sha=self.expected_sha, attempt=str(self.attempt), argv=self.argv,
                      wall_s=time.monotonic() - self.started, telemetry=telemetry, pool_stats=stats)
        result["telemetry_complete"] = bool(telemetry["per_gpu"]) and not telemetry["errors"] and all(
            metrics.get("samples", 0) > 0 for metrics in telemetry["per_gpu"].values())
        result["performance_eligible"] = (result["execution_status"] == "completed"
            and result["reference_eligible"] and result["all_requested_gpus_observed"]
            and result["full_step_budget_completed"] and result["telemetry_complete"])
        result["artifact_sha256"] = artifact_hashes(self.attempt)
        # Immutable per-attempt record remains available when a later attempt runs.
        write_json(self.attempt / "outcome.json", result)
        write_json(self.directory / "cell.status.json", result)
        return result


def cached_result(directory, cell, sha):
    """Completed-with-warnings is done; verify artifacts rather than trusting flags."""
    try:
        status = read_json(directory / "cell.status.json")
        attempt = Path(status["attempt"])
        if status.get("execution_status") != "completed" or status.get("expected_source_sha") != sha \
                or status.get("spec_sha256") != fingerprint(cell.spec()) \
                or attempt.resolve().parent != directory.resolve() or attempt.is_symlink():
            return None
        hashes = status.get("artifact_sha256", {})
        if not hashes or "result.json" not in hashes:
            return None
        for name, digest in hashes.items():
            path = attempt / name
            if not path.resolve().is_relative_to(attempt.resolve()) or path.is_symlink() or sha256(path) != digest:
                return None
        stats = read_json(attempt / "pool_stats.json") if cell.workers else None
        check = assess(read_json(attempt / "result.json"), stats, cell, sha, status["returncode"])
        if check["execution_status"] != "completed":
            return None
        return status
    except (OSError, ValueError, KeyError, TypeError):
        return None


def import_completed(old, root, sha):
    results = {}
    for stage, cells in matrix().items():
        results[stage] = {}
        for cell in cells:
            olddir = old / stage / cell.name
            if not olddir.exists():
                continue
            if olddir.is_symlink() or any(p.is_symlink() for p in olddir.rglob("*")):
                raise ValueError("resume cell contains symlinked evidence")
            status = cached_result(olddir, cell, sha)
            prior = read_json(olddir / "cell.status.json") if (olddir / "cell.status.json").exists() else {}
            if prior.get("execution_status") == "completed" and status is None:
                raise ValueError("completed resume cell failed artifact/identity verification; not silently rerunning it")
            directory = root / stage / cell.name
            directory.parent.mkdir(parents=True, exist_ok=True)
            # Copy all historical attempts, even interrupted ones. A pending
            # cell will then create attempt_0002 etc., never overwrite evidence.
            shutil.copytree(olddir, directory, symlinks=False)
            write_json(directory / "resume_import.json", {"source": str(olddir), "previous_status": prior})
            if status is None:
                continue
            attempt = directory / Path(status["attempt"]).name
            imported = {**status, "attempt": str(attempt), "reused_from": str(olddir)}
            write_json(directory / "cell.status.json", imported)
            if cached_result(directory, cell, sha) is None:
                raise ValueError("resume artifacts changed while copying")
            results[stage][cell.name] = imported
    return results


def check_resume(plan, query=True):
    old = Path(plan["resume_from"])
    previous = read_json(old / "plan.json")
    if previous.get("schema") != plan["schema"] or previous.get("expected_sha") != plan["expected_sha"] \
            or previous.get("science") != plan["science"] or previous.get("environment") != plan["environment"]:
        raise ValueError("resume needs the same campaign version, exact source, environment and scientific matrix")
    ledger = read_json(old / "submission.json")
    if sha256(old / "plan.json") != ledger.get("plan_sha256"):
        raise ValueError("resume plan hash mismatch")
    if query:
        job = ledger.get("job_id", "")
        if not re.fullmatch(r"\d+(?:\.[A-Za-z0-9_.-]+)?", job):
            raise ValueError("resume job ID is missing or invalid")
        answer = subprocess.run(["qstat", "-xf", job], capture_output=True, text=True, timeout=30)
        if answer.returncode or not re.search(r"\bjob_state\s*=\s*[FC]\b", answer.stdout):
            raise ValueError("resume refused: PBS must confirm the previous job has finished; no overlapping writers")
    return {"root": str(old), "plan_sha256": sha256(old / "plan.json"), "job_id": ledger.get("job_id")}


def stage_verdict(stage, cells, results):
    complete = [c.name for c in cells if results.get(c.name, {}).get("execution_status") == "completed"]
    return {"stage": stage, "requested_cells": len(cells), "completed_cells": len(complete),
            "all_cells_completed": len(complete) == len(cells),
            "uncompleted_cells": [c.name for c in cells if c.name not in complete],
            "strict_qualified_cells": [name for name, r in results.items() if r.get("reference_eligible")],
            "cells": results}


def summary(plan, results, reason=None):
    verdicts = {s: stage_verdict(s, cells, results[s]) for s, cells in matrix().items()}
    all_results = [r for group in results.values() for r in group.values()]
    complete = all(v["all_cells_completed"] for v in verdicts.values())
    baseline = results["g2"].get("g2_gpu1_w4", {})
    comparisons = {}
    for name, r in results["g2"].items():
        matched = (baseline.get("execution_status") == "completed" and r.get("execution_status") == "completed"
                   and baseline.get("full_step_budget_completed") and r.get("full_step_budget_completed")
                   and baseline.get("all_requested_gpus_observed") and r.get("all_requested_gpus_observed")
                   and baseline.get("telemetry_complete") and r.get("telemetry_complete")
                   and r.get("work_signature") == baseline.get("work_signature")
                   and not baseline.get("reused_from") and not r.get("reused_from"))
        eligible = matched and baseline.get("performance_eligible") and r.get("performance_eligible")
        comparisons[name] = {"qualified_speedup_vs_gpu1_w4": baseline["wall_s"] / r["wall_s"] if eligible else None,
                             "exploratory_matched_work_wall_ratio": baseline["wall_s"] / r["wall_s"] if matched else None,
                             "eligible": bool(eligible), "wall_s": r.get("wall_s"),
                             "control_steps_applied": r.get("control_steps_applied"),
                             "note": "Exploratory ratio is descriptive and may include quality warnings. Qualified speedup additionally requires strict quality. Both require matched full work and same allocation; early failures are not acceleration."}
    strict_ok = complete and all(r.get("reference_eligible") for r in all_results)
    return {"schema": "contact_model_study.polaris_campaign_status.v1", "expected_sha": plan["expected_sha"],
            "execution_status": "completed" if complete and not reason else "incomplete",
            "stop_reason": reason, "requested_cells": 43, "requested_episodes": 135,
            "completed_cells": sum(v["completed_cells"] for v in verdicts.values()),
            "validation_status": "pass" if strict_ok else "fail", "reference_eligible": strict_ok and not reason,
            "scaling_eligible": strict_ok and not reason,
            "recorded_episodes": sum(r.get("recorded_episodes", 0) for r in all_results),
            "task_successes_observed": sum(r.get("task_successes_observed", 0) for r in all_results),
            "task_failures": sum(r.get("task_failures", 0) for r in all_results),
            "planner_failures": sum(r.get("planner_failures", 0) for r in all_results),
            "plans_attempted": sum(r.get("plans_attempted", 0) for r in all_results),
            "stages": verdicts, "g2_performance_comparison": comparisons,
            "g1_calibration_by_samples": {str(n): {c.name: results["g1"].get(c.name)
                 for c in matrix()["g1"] if int(c.row["n_samples"]) == n} for n in (32,64,128,256)},
            "note": "Exploratory policy, not a certified high-accuracy reference. Task failure and quality warnings never masquerade as success. No automatic retries."}


def check_environment():
    # Inspect installed metadata, not a stale version cached in a Python module.
    from preflight import EXPECTED_PACKAGES
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Python environment changed: Python3.12 required")
    for name, expected in EXPECTED_PACKAGES.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"runtime environment changed: {name} must be {expected}")


def run(root):
    root = Path(root).resolve()
    plan, ledger = read_json(root / "plan.json"), read_json(root / "submission.json")
    with (root / "run.lock").open("x") as lock:
        lock.write(str(os.environ.get("PBS_JOBID")) + "\n")
    results, running = {s: {} for s in STAGES}, []
    reason, interrupted, finalizing = None, False, False
    old_handlers = {}
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    def save():
        report = summary(plan, results, reason)
        report["job_id"] = os.environ.get("PBS_JOBID")
        report["resume_from"] = plan["resume_from"]
        report["active_cells"] = [str(a.directory.relative_to(root)) for a in running]
        if not finalizing and not reason:
            report["execution_status"] = "running"
        write_json(root / "status.json", report)
        for s, verdict in report["stages"].items():
            write_json(root / s / "stage_verdict.json", verdict)
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, stop)
        if not os.environ.get("PBS_JOBID") or ledger.get("job_id") != os.environ["PBS_JOBID"]:
            raise ValueError("unregistered PBS job")
        if sha256(root / "plan.json") != ledger.get("plan_sha256"):
            raise ValueError("submitted plan was modified")
        write_json(root / "submission_at_start.json", ledger)
        if plan["science"] != scientific_plan() or plan["science_sha256"] != fingerprint(scientific_plan()):
            raise ValueError("campaign matrix/policy differs from reviewed source")
        if plan["source"] != str(REPO) or not direct.source_matches(plan["expected_sha"]):
            raise ValueError("source must match clean submitted commit")
        if Path(sys.executable).resolve() != (Path(plan["environment"]) / "bin/python").resolve():
            raise ValueError("wrong Python environment")
        preflight = read_json(root / "preflight.json")
        if preflight.get("passed") is not True or preflight.get("expected_source_sha") != plan["expected_sha"] \
                or preflight.get("requested_gpus") != "0,1,2,3":
            raise ValueError("four-GPU preflight failed/missing")
        if plan["resume_from"]:
            resume = check_resume(plan)
            write_json(root / "resume_source.json", resume)
            old = Path(plan["resume_from"])
            for filename in ("pip-freeze.txt", "conda-explicit.txt"):
                if sha256(root / filename) != sha256(old / filename):
                    raise ValueError("resume runtime environment snapshot changed")
            for filename in ("artifact_manifest.json", "status.json", "wrapper_status.json", "submission.json"):
                if (old / filename).is_file():
                    shutil.copy2(old / filename, root / f"resume_previous_{filename}")
            results = import_completed(old, root, plan["expected_sha"])
        batches = stages.batch_module()
        # PBS walltime starts before preflight. Subtract elapsed wrapper time,
        # leaving the reviewed ten-minute startup/finalization margin.
        start = float(os.environ.get("CAMPAIGN_START_UNIX", time.time()))
        deadline = time.monotonic() + max(0, plan["budget_seconds"] - (time.time() - start))
        save()
        for stage, cells in matrix().items():
            pending = [c for c in cells if c.name not in results[stage]]
            while pending or running:
                if interrupted:
                    reason = "interrupted"
                elif time.monotonic() >= deadline:
                    reason = "walltime_budget_exhausted"
                if not reason:
                    if not direct.source_matches(plan["expected_sha"]):
                        raise ValueError("source changed during execution")
                    used = {g for active in running for g in active.cell.gpus.split(",")}
                    launched = False
                    for cell in list(pending):
                        if (stage != "g3" and running) or used.intersection(cell.gpus.split(",")):
                            continue
                        check_environment()
                        active = Active(cell, root / stage / cell.name, batches, plan["expected_sha"])
                        running.append(active)
                        launched = True
                        pending.remove(cell)
                        used.update(cell.gpus.split(","))
                        print(f"Started {stage}/{cell.name} on GPU(s) {cell.gpus}", flush=True)
                    if launched:
                        save()
                for active in list(running):
                    if reason:
                        active.cancel()
                    if active.ready():
                        running.remove(active)
                        result = active.finish()
                        results[stage][active.cell.name] = result
                        print(f"Finished {stage}/{active.cell.name}: execution={result['execution_status']}, strict={result['validation_status']}", flush=True)
                        if result["global_failure"]:
                            reason = f"global_failure:{stage}/{active.cell.name}"
                        save()
                if reason and not running:
                    break
                if pending or running:
                    time.sleep(0.2)
            if reason:
                break
        if not direct.source_matches(plan["expected_sha"]):
            reason = "source_changed_during_execution"
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        while running:
            for active in list(running):
                active.cancel()
                if active.ready():
                    running.remove(active)
                    try:
                        results[stage][active.cell.name] = active.finish()
                    except Exception as exc:
                        reason = f"cleanup_error:{type(exc).__name__}:{exc}"
            if running:
                time.sleep(0.2)
        finalizing = True
        save()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return 0 if not reason and summary(plan, results)["execution_status"] == "completed" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("submit")
    p.add_argument("--project", required=True)
    p.add_argument("--env", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--queue", choices=("debug", "capacity"), default="capacity")
    p.add_argument("--walltime", default="24:00:00")
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--check", action="store_true")
    p = commands.add_parser("run")
    p.add_argument("--outdir", required=True, type=Path)
    p = commands.add_parser("status")
    p.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action == "run":
        return run(args.outdir)
    if args.action == "status":
        print(json.dumps(read_json(args.root / "status.json"), indent=2))
        return 0
    try:
        source = git_provenance(REPO)
        plan = make_plan(args, source["commit"])
        validate_commands()
        if args.check:
            if args.resume_from:
                check_resume(plan, query=False)
            print(json.dumps({"preview_only": True, "source_dirty": source["dirty"], **plan}, indent=2))
            return 0
        if not direct.source_matches(plan["expected_sha"]):
            raise ValueError("a clean local deployment commit is required")
        for path in (REPO, args.env.resolve(), args.out_root.resolve(), args.resume_from):
            if path and any(c.isspace() or c in ",\"'\\" for c in str(path)):
                raise ValueError("PBS paths cannot contain whitespace, commas, quotes or backslashes")
        for filename in ("bin/python", "conda-explicit.txt", "pip-freeze.txt"):
            if not (args.env / filename).is_file():
                raise ValueError(f"prepared environment file missing: {filename}")
        output = args.out_root.resolve()
        if not str(output).startswith(("/eagle/", "/lus/eagle/projects/")) or not output.is_dir():
            raise ValueError("out-root must be an existing writable Eagle directory")
        if args.resume_from:
            check_resume(plan)
        root = Path(tempfile.mkdtemp(prefix="kamino_campaign_", dir=output))
        print(f"Results and submission record: {root}", flush=True)
        job = direct.submit(plan, root)
        print(f"Submitted and released ONE job: {job}\nPBS now owns the complete G1-G3 campaign; no SSH watcher is needed.")
        print(f"When finished, return the entire directory: {root}")
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
