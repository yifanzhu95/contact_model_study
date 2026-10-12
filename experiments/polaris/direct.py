#!/usr/bin/env python3
"""One PBS job running actual four-GPU Kamino episodes, without a G0 prerequisite.

Execution completion is not a solver-validation certificate. Invalid candidates
remain filtered; an all-invalid plan ends that episode before actuator application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ContactModelStudy.Utils.Provenance import git_provenance  # noqa: E402


def write_json(path, doc):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def finalize(root, exit_code):
    """Seal quiet, stable files, including failures before the driver started."""
    root = Path(root)
    write_json(root / "wrapper_status.json", {"job_id": os.environ.get("PBS_JOBID"),
               "exit_code": exit_code, "end_unix": time.time()})
    if not (root / "status.json").exists():
        write_json(root / "status.json", {"execution_status": "error",
                   "validation_status": "not_evaluated", "scaling_eligible": False,
                   "error": "wrapper stopped before experiment startup; see job.log and preflight.json"})
    # The collector's own printed path must not append bytes to the sealed log.
    answer = subprocess.run([sys.executable, str(REPO / "experiments/polaris/collect_manifest.py"),
                             "--outdir", str(root), "--job-exit-code", str(exit_code),
                             "--exclude", "submission.json", "--exclude", "pbs.stdout",
                             "--exclude", "pbs.stderr"],
                            capture_output=True, text=True, timeout=120)
    return answer.returncode


def wall_seconds(value):
    if not re.fullmatch(r"\d{2,3}:[0-5]\d:[0-5]\d", value):
        raise ValueError("walltime must be HH:MM:SS")
    h, m, s = map(int, value.split(":"))
    return h * 3600 + m * 60 + s


def driver_command(plan, root):
    root = Path(root)
    return [str(Path(plan["environment"]) / "bin/python"),
            str(REPO / "ContactModelStudy/Drivers/run_episodes_pooled.py"),
            "--task", "cube_reorient", "--rollout-model", "M5", "--eval-sim", "mujoco",
            "--n-samples", str(plan["n_samples"]), "--n-episodes", str(plan["n_episodes"]),
            "--steps", str(plan["steps"]), "--seed", "0", "--goal-difficulty", "8",
            "--timestep", "0.0005", "--eval-steps-per-rollout-step", "8",
            "--ctrl-time-step", "0.064", "--horizon", "5", "--n-iterations", "1",
            "--noise-sigma", "0.05", "--temperature", "40", "--control-mode", "pos_relative",
            "--kamino-max-iterations", str(plan["max_iterations"]), "--stop-on-invalid-plan",
            "--graph", "--no-warm-start", "--no-stop-on-success", "--no-video", "--save-steps",
            "--gpus", "0,1,2,3", "--planners-per-gpu", "1", "--workers", "8",
            "--checkpoint-every", "1", "--pool-stats", str(root / "pool_stats.json"),
            "--results", str(root / "results.json")]


def make_plan(args, sha):
    seconds = wall_seconds(args.walltime)
    limit = 3600 if args.queue == "debug" else 168 * 3600
    if not 900 <= seconds <= limit:
        raise ValueError(f"this wrapper needs 15 minutes to {limit // 3600} hours in {args.queue}")
    if min(args.n_samples, args.n_episodes, args.steps, args.max_iterations) < 1:
        raise ValueError("samples, episodes, steps and iteration cap must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.project):
        raise ValueError("project must be an existing ALCF account name")
    return {"schema": "contact_model_study.polaris_direct.v1", "source": str(REPO),
            "expected_sha": sha, "project": args.project, "environment": str(args.env.resolve()),
            "queue": args.queue, "walltime": args.walltime, "budget_seconds": seconds - 600,
            "n_samples": args.n_samples, "n_episodes": args.n_episodes, "steps": args.steps,
            "max_iterations": args.max_iterations,
            "policy": {"separate_diagnostic_job": False, "strict_candidate_filter": True,
                       "all_invalid": "end_episode_without_applying_action",
                       "validation_required_for_execution": False,
                       "automatic_retry": False, "numerical_preset_validated": False}}


def source_matches(sha):
    source = git_provenance(REPO)
    return source["commit"] == sha and source["dirty"] is False


def submit(plan, root):
    """Journal one held submission, register its ID, then release. Never retry."""
    root = Path(root)
    if (root / "plan.json").exists() or (root / "submission.json").exists():
        raise FileExistsError("submission directory is single-use; do not retry a previous submission")
    with (root / "submit.lock").open("x") as lock:
        lock.write(str(os.getpid()) + "\n")
    write_json(root / "plan.json", plan)
    ledger = {"state": "submitting", "job_id": None, "released": False, "operations": [],
              "plan_sha256": hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()}
    ledger_path = root / "submission.json"
    write_json(ledger_path, ledger)

    def operation(argv):
        entry = {"argv": argv, "status": "in_flight"}
        ledger["operations"].append(entry)
        write_json(ledger_path, ledger)
        try:
            answer = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            entry.update(status="unknown", error=str(exc))
            write_json(ledger_path, ledger)
            raise ValueError("scheduler response unknown; inspect submission.json, do not resubmit") from exc
        entry.update(status="returned", returncode=answer.returncode,
                     stdout=answer.stdout, stderr=answer.stderr)
        write_json(ledger_path, ledger)
        if answer.returncode:
            raise ValueError(f"{argv[0]} rejected the request: {answer.stderr.strip()}")
        return answer.stdout.strip()

    env = ",".join(f"{k}={v}" for k, v in {
        "PROJ_DIR": REPO, "ENV_DIR": plan["environment"], "OUTDIR": root,
        "EXPECTED_SHA": plan["expected_sha"], "ALCF_PROJECT": plan["project"]}.items())
    try:
        job = operation(["qsub", "-h", "-r", "n", "-A", plan["project"],
                         "-q", plan["queue"], "-l", f"walltime={plan['walltime']}",
                         "-v", env, "-o", str(root / "pbs.stdout"), "-e", str(root / "pbs.stderr"),
                         str(REPO / "experiments/polaris/run_direct.pbs")])
        if not re.fullmatch(r"\d+(?:\.[A-Za-z0-9_.-]+)?", job):
            raise ValueError("unrecognized PBS job ID; inspect the recorded response, do not resubmit")
        ledger.update(state="registered_held", job_id=job)
        write_json(ledger_path, ledger)
        if not source_matches(plan["expected_sha"]):
            raise ValueError("source changed; registered job remains held")
        # The compute job may start before qrls returns: registration already exists.
        operation(["qrls", "-h", "u", job])
        ledger.update(state="submitted", released=True)
        write_json(ledger_path, ledger)
    except Exception as exc:
        ledger.update(state="submission_incomplete", error=str(exc))
        write_json(ledger_path, ledger)
        raise
    return job


def assess(document, stats, plan, returncode):
    reasons = []
    source = document.get("configs", {}).get("metadata", {}).get("provenance", {}).get("source", {})
    if source.get("commit") != plan["expected_sha"] or source.get("dirty") is not False:
        reasons.append("source_mismatch")
    episodes = document.get("episodes", [])
    summaries = [e.get("summary", {}) for e in episodes]
    if len(episodes) != plan["n_episodes"] or sorted(s.get("episode", -1) for s in summaries) != list(range(plan["n_episodes"])):
        reasons.append("incomplete_episodes")
    if any(s.get("finish_reason") not in ("success", "failure", "timeout", "planner_failure") for s in summaries):
        reasons.append("execution_error_or_interruption")
    if returncode or stats.get("complete") is not True:
        reasons.append("driver_or_pool_incomplete")
    if stats.get("pool") != {"gpus": ["0", "1", "2", "3"], "workers": min(8, plan["n_episodes"]), "planners_per_gpu": 1}:
        reasons.append("pool_topology_mismatch")
    validation = document.get("validation", {})
    valid = validation.get("status") == "pass" and validation.get("scaling_eligible") is True
    planner_failures = sum(s.get("finish_reason") == "planner_failure" for s in summaries)
    used = sorted({g for s in summaries for g in s.get("planner_validation", {}).get("planner_devices", [])})
    return {"execution_status": "incomplete" if reasons else "completed",
            "execution_reasons": reasons, "driver_returncode": returncode,
            "requested_episodes": plan["n_episodes"], "recorded_episodes": len(episodes),
            "planner_failures": planner_failures, "task_failures": sum(s.get("finish_reason") == "failure" for s in summaries),
            "task_successes_observed": sum(bool(s.get("success")) for s in summaries),
            "validation_status": "pass" if valid else "fail",
            "scaling_eligible": not reasons and valid and used == ["0", "1", "2", "3"],
            "all_requested_gpus_observed": used == ["0", "1", "2", "3"],
            "strict_validation": validation, "planner_devices_observed": used,
            "note": "Completion includes recorded planner/task failures. It does not certify numerical accuracy or speedup."}


def run(root):
    root = Path(root).resolve()
    plan = read_json(root / "plan.json")
    ledger = read_json(root / "submission.json")
    # This directory is single-use, including a failed or interrupted execution.
    with (root / "run.lock").open("x") as lock:
        lock.write(str(os.environ.get("PBS_JOBID")) + "\n")
    status = {"execution_status": "starting", "expected_sha": plan["expected_sha"],
              "job_id": os.environ.get("PBS_JOBID"), "validation_status": "not_evaluated", "scaling_eligible": False}
    write_json(root / "status.json", status)
    child = None
    try:
        if not status["job_id"] or ledger.get("job_id") != status["job_id"]:
            raise ValueError("unregistered PBS job")
        if hashlib.sha256((root / "plan.json").read_bytes()).hexdigest() != ledger.get("plan_sha256"):
            raise ValueError("submitted plan was modified")
        # qrls may not have returned to the login process yet. Seal this snapshot,
        # not the live ledger that its independent submitter still owns.
        write_json(root / "submission_at_start.json", ledger)
        if plan["source"] != str(REPO) or not source_matches(plan["expected_sha"]):
            raise ValueError("source must match the clean submitted commit")
        if Path(sys.executable).resolve() != (Path(plan["environment"]) / "bin/python").resolve():
            raise ValueError("wrong Python environment")
        preflight = read_json(root / "preflight.json")
        if preflight.get("passed") is not True or preflight.get("expected_source_sha") != plan["expected_sha"] or preflight.get("requested_gpus") != "0,1,2,3":
            raise ValueError("four-GPU preflight failed or missing")
        argv = driver_command(plan, root)
        write_json(root / "command.json", argv)
        status.update(execution_status="running", start_unix=time.time())
        write_json(root / "status.json", status)
        with (root / "experiment.log").open("x") as log:
            child = subprocess.Popen(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                rc = child.wait(timeout=plan["budget_seconds"])
            except subprocess.TimeoutExpired:
                status["budget_exhausted"] = True
                os.killpg(child.pid, signal.SIGINT)
                try:
                    rc = child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    rc = child.wait()
        document = read_json(root / "results.json") if (root / "results.json").exists() else {}
        stats = read_json(root / "pool_stats.json") if (root / "pool_stats.json").exists() else {}
        status.update(assess(document, stats, plan, rc))
        if not source_matches(plan["expected_sha"]):
            status.update(execution_status="incomplete", scaling_eligible=False)
            status["execution_reasons"].append("source_changed_during_run")
        if status.get("budget_exhausted"):
            status.update(execution_status="incomplete", scaling_eligible=False)
            status["execution_reasons"].append("walltime_budget_exhausted")
        return 0 if status["execution_status"] == "completed" else 2
    except BaseException as exc:
        status.update(execution_status="error", error=f"{type(exc).__name__}: {exc}", scaling_eligible=False)
        return 2
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        status["end_unix"] = time.time()
        write_json(root / "status.json", status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("submit")
    p.add_argument("--project", required=True)
    p.add_argument("--env", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--queue", choices=("debug", "capacity"), default="capacity")
    p.add_argument("--walltime", default="04:00:00")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--n-episodes", type=int, default=24)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--max-iterations", type=int, default=400)
    p.add_argument("--check", action="store_true")
    p = commands.add_parser("run")
    p.add_argument("--outdir", required=True, type=Path)
    p = commands.add_parser("finalize")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args(argv)
    if args.action == "finalize":
        return finalize(args.outdir, args.exit_code)
    if args.action == "run":
        # PBS sends SIGTERM before its walltime kill; let run() retain partial results.
        def stop(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, stop)
        return run(args.outdir)
    try:
        source = git_provenance(REPO)
        plan = make_plan(args, source["commit"])
        from ContactModelStudy.Drivers import run_episodes as drv, run_episodes_pooled as pooled
        drv.parseArgs(driver_command(plan, Path("/preview"))[2:], parser=pooled.build_parser())
        if args.check:
            print(json.dumps({"preview_only": True, "source_dirty": source["dirty"], **plan}, indent=2))
            return 0
        if not source_matches(plan["expected_sha"]):
            raise ValueError("a clean local deployment commit is required")
        for path in (REPO, args.env.resolve(), args.out_root.resolve()):
            if any(c.isspace() or c in ",\"'\\" for c in str(path)):
                raise ValueError("PBS paths cannot contain whitespace, commas, quotes or backslashes")
        if not (args.env / "bin/python").is_file():
            raise ValueError("prepared environment Python is missing")
        for name in ("conda-explicit.txt", "pip-freeze.txt"):
            if not (args.env / name).is_file():
                raise ValueError(f"prepared environment record missing: {name}")
        output = args.out_root.resolve()
        if not str(output).startswith(("/eagle/", "/lus/eagle/projects/")) or not output.is_dir():
            raise ValueError("out-root must be an existing writable Eagle directory")
        root = Path(tempfile.mkdtemp(prefix="kamino_direct_", dir=output))
        print(f"Results and submission record: {root}", flush=True)
        job = submit(plan, root)
        print(f"Submitted and released ONE job: {job}\nPBS now owns execution; no SSH watcher is needed.")
        print(f"When finished, return this entire directory: {root}")
        return 0
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
