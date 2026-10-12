#!/usr/bin/env python3
"""Submit a held G0->G1->G2->G3 PBS chain, or inspect its file-based evidence.

No daemon, automatic resubmission, credentials, or GPU imports. Internal
start/finish commands are called by run_pipeline_stage.pbs on compute nodes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from ContactModelStudy.Utils.Provenance import git_provenance  # noqa: E402

STAGES = ("g0", "g1", "g2", "g3")
FIRST_DEPLOYMENT_SAMPLES = 64
SCHEMA = "contact_model_study.polaris_pipeline.v1"
JOB_ID = re.compile(r"[0-9]+(?:\.[A-Za-z0-9_.-]+)?")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return document


def write_json(path, document):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stage_dir(root, stage):
    return Path(root) / stage


def verdict_path(root, stage):
    name = "smoke_verdict.json" if stage == "g0" else "stage_verdict.json"
    return stage_dir(root, stage) / name


def status_path(root, stage):
    return stage_dir(root, stage) / "pipeline.job.status.json"


def make_plan(root, project, environment, sha, samples):
    root = Path(root).resolve()
    return {
        "schema": SCHEMA, "root": str(root), "repo": str(REPO),
        "environment": str(Path(environment).resolve()), "project": project,
        "expected_source_sha": sha, "n_samples": samples,
        "policy": {"full_stage_pass_required": True, "automatic_retries": 0,
                   "automatic_n_selection": False},
        "stages": [{"stage": stage, "outdir": str(stage_dir(root, stage)),
                    "walltime": "00:30:00" if stage == "g0" else "01:00:00",
                    "previous_verdict": str(verdict_path(root, STAGES[i - 1])) if i else None}
                   for i, stage in enumerate(STAGES)],
    }


def qsub_command(plan, item, previous_job=None):
    variables = {
        "PROJ_DIR": plan["repo"], "ENV_DIR": plan["environment"], "OUTDIR": item["outdir"],
        "EXPECTED_SHA": plan["expected_source_sha"], "ALCF_PROJECT": plan["project"],
        "PIPELINE_ROOT": plan["root"], "PIPELINE_STAGE": item["stage"],
        "STAGE": item["stage"], "NSAMPLES": str(plan["n_samples"]),
    }
    if item["previous_verdict"]:
        variables["PREVIOUS_VERDICT"] = item["previous_verdict"]
    # A short unique run token helps reconcile a lost qsub response using qstat.
    token = Path(plan["root"]).name[-6:]
    argv = ["qsub", "-h", "-r", "n", "-A", plan["project"], "-N", f"km_{item['stage']}_{token}",
            "-l", f"walltime={item['walltime']}",
            "-v", ",".join(f"{key}={value}" for key, value in variables.items()),
            "-o", str(Path(item["outdir"]) / "pbs.stdout"),
            "-e", str(Path(item["outdir"]) / "pbs.stderr")]
    if previous_job:
        argv.extend(["-W", f"depend=afterok:{previous_job}"])
    argv.append(str(PACKAGE / "run_pipeline_stage.pbs"))
    return argv


def check_workloads(samples, root):
    """Reuse the actual driver parsers; never simulate or call a scheduler."""
    commands = [[sys.executable, str(REPO / "experiments/run_episode_batches.py"),
                 str(PACKAGE / "m5_smoke.csv"), "--check", "--outdir", str(stage_dir(root, "g0"))]]
    for stage in STAGES[1:]:
        cmd = [sys.executable, str(PACKAGE / "run_stage.py"), stage, "--check",
               "--outdir", str(stage_dir(root, stage))]
        if stage != "g1":
            cmd.extend(["--n-samples", str(samples)])
        commands.append(cmd)
    for cmd in commands:
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise ValueError(f"workload check failed: {cmd}\n{result.stderr or result.stdout}")


def source_matches(sha):
    source = git_provenance(REPO)
    return source.get("commit") == sha and source.get("dirty") is False


def load_plan(root):
    root = Path(root).resolve()
    plan = read_json(root / "pipeline_plan.json")
    expected = make_plan(root, plan["project"], plan["environment"],
                         plan["expected_source_sha"], plan["n_samples"])
    if plan != expected or plan["n_samples"] not in (32, 64, 128, 256):
        raise ValueError("pipeline plan differs from this checkout or its fixed stage policy")
    return plan


def verdict_reasons(root, stage, plan):
    """Full-stage validity is mandatory in automatic mode, including G1."""
    try:
        document = read_json(verdict_path(root, stage))
        reasons = []
        schema = ("contact_model_study.polaris_smoke_verdict.v1" if stage == "g0"
                  else "contact_model_study.polaris_stage_verdict.v1")
        if document.get("schema") != schema:
            reasons.append("verdict_schema_mismatch")
        if document.get("expected_source_sha") != plan["expected_source_sha"]:
            reasons.append("verdict_source_sha_mismatch")
        if document.get("passed") is not True or document.get("scaling_eligible") is not True:
            reasons.append("stage_not_fully_passed")
        if stage != "g0":
            if document.get("stage") != stage or document.get("source_unchanged") is not True:
                reasons.append("stage_or_source_identity_mismatch")
            if document.get("interrupted") is not False:
                reasons.append("stage_interrupted_or_missing_completion")
            if document.get("pending_cells") != []:
                reasons.append("pending_cells_or_missing_completion")
        if stage == "g1" and plan["n_samples"] not in document.get("validated_samples", []):
            reasons.append("chosen_n_not_validated")
        if stage in ("g2", "g3") and document.get("n_samples") != plan["n_samples"]:
            reasons.append("chosen_n_mismatch")
        return reasons
    except (OSError, ValueError, TypeError) as exc:
        return [f"verdict_unreadable: {exc}"]


def artifact_reasons(root, stage):
    """A successful job's final status must also be sealed into its manifest."""
    try:
        manifest = read_json(stage_dir(root, stage) / "artifact_manifest.json")
        digest = hashlib.sha256(status_path(root, stage).read_bytes()).hexdigest()
        entries = [entry for entry in manifest.get("files", [])
                   if entry.get("path") == "pipeline.job.status.json"]
        if manifest.get("job_exit_code") != 0 or len(entries) != 1 or entries[0].get("sha256") != digest:
            return ["final_status_not_sealed_in_successful_artifact_manifest"]
        return []
    except (OSError, ValueError, TypeError) as exc:
        return [f"artifact_manifest_unreadable: {exc}"]


def job_start(root, stage):
    plan = load_plan(root)
    path = status_path(root, stage)
    # No silent rerun or overwriting of a previous job's evidence.
    if path.exists():
        raise ValueError("stage already has a pipeline job status; inspect it before recovery")
    job_id = os.environ.get("PBS_JOBID")
    status = {"schema": SCHEMA, "stage": stage, "job_id": job_id,
              "expected_source_sha": plan["expected_source_sha"], "n_samples": plan["n_samples"],
              "started_utc": now(), "execution_status": "checking", "passed": False}
    reasons = []
    try:
        ledger = read_json(Path(root) / "submission.json")
        if not job_id or ledger["jobs"].get(stage, {}).get("job_id") != job_id:
            reasons.append("unregistered_pbs_job")
        item = plan["stages"][STAGES.index(stage)]
        expected_environment = {
            "PROJ_DIR": plan["repo"], "ENV_DIR": plan["environment"], "OUTDIR": item["outdir"],
            "EXPECTED_SHA": plan["expected_source_sha"], "ALCF_PROJECT": plan["project"],
            "PIPELINE_ROOT": plan["root"], "PIPELINE_STAGE": stage, "STAGE": stage,
            "NSAMPLES": str(plan["n_samples"]),
        }
        if item["previous_verdict"]:
            expected_environment["PREVIOUS_VERDICT"] = item["previous_verdict"]
        if any(os.environ.get(key) != value for key, value in expected_environment.items()):
            reasons.append("pbs_environment_differs_from_plan")
        if not source_matches(plan["expected_source_sha"]):
            reasons.append("source_mismatch_or_dirty")
        if stage != "g0":
            previous = STAGES[STAGES.index(stage) - 1]
            previous_status = read_json(status_path(root, previous))
            if previous_status.get("passed") is not True \
                    or previous_status.get("job_id") != ledger["jobs"].get(previous, {}).get("job_id"):
                reasons.append("previous_pipeline_job_not_passed")
            reasons.extend(verdict_reasons(root, previous, plan))
            reasons.extend(artifact_reasons(root, previous))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        reasons.append(f"gate_unreadable: {exc}")
    status.update(execution_status="blocked" if reasons else "running", blocking_reasons=reasons)
    write_json(path, status)
    return 2 if reasons else 0


def job_finish(root, stage, exit_code):
    plan = load_plan(root)
    path = status_path(root, stage)
    status = read_json(path)
    if status.get("job_id") != os.environ.get("PBS_JOBID") \
            or status.get("execution_status") not in ("checking", "running", "blocked"):
        raise ValueError("refusing to overwrite another or already finalized job's evidence")
    reasons = list(status.get("blocking_reasons", []))
    if exit_code:
        reasons.append(f"job_exit_{exit_code}")
    if not source_matches(plan["expected_source_sha"]):
        reasons.append("source_mismatch_or_dirty")
    reasons.extend(verdict_reasons(root, stage, plan))
    status.update(finished_utc=now(), returncode=exit_code, passed=not reasons,
                  validation_status="fail" if reasons else "pass", blocking_reasons=reasons,
                  execution_status=("blocked" if status["execution_status"] == "blocked"
                                    else "completed" if exit_code == 0 else "failed"))
    write_json(path, status)
    return 2 if reasons else 0


def report(root):
    plan = load_plan(root)
    ledger = read_json(Path(root) / "submission.json")
    stages, blocked_by = {}, None
    for stage in STAGES:
        entry = {"job_id": ledger.get("jobs", {}).get(stage, {}).get("job_id")}
        path = status_path(root, stage)
        if path.exists():
            entry.update(read_json(path))
            if entry.get("passed") is True:
                reasons = verdict_reasons(root, stage, plan) + artifact_reasons(root, stage)
                if reasons:
                    entry.update(passed=False, blocking_reasons=reasons, execution_status="evidence_invalid")
            if entry.get("execution_status") in ("failed", "blocked", "evidence_invalid") \
                    or (entry.get("execution_status") == "completed" and not entry.get("passed")):
                blocked_by = blocked_by or stage
        else:
            entry["execution_status"] = "blocked_by_previous_stage" if blocked_by else "not_started_or_status_missing"
        stages[stage] = entry
    passed = all(entry.get("passed") is True for entry in stages.values())
    return {"schema": SCHEMA, "root": str(Path(root).resolve()), "passed": passed,
            "submission_state": ledger.get("state"), "blocked_by": blocked_by, "stages": stages,
            "note": "File evidence only. A running status may be stale after a hard kill. "
                    "Use qstat -xf <job-id> to check PBS; no scheduler query or monitor is started."}


def submit(root, plan, *, reserved=False):
    """Every scheduler operation is journaled before invocation; never retry it."""
    root = Path(root)
    if reserved:
        if not root.is_dir() or any(root.iterdir()):
            raise ValueError("reserved pipeline directory must be empty")
    else:
        root.mkdir(exist_ok=False)
    for stage in STAGES:
        stage_dir(root, stage).mkdir()
    write_json(root / "pipeline_plan.json", plan)
    ledger = {"schema": SCHEMA, "created_utc": now(), "state": "submitting_held", "jobs": {}, "operations": []}
    ledger_path = root / "submission.json"
    write_json(ledger_path, ledger)

    def operation(argv):
        entry = {"argv": argv, "started_utc": now(), "status": "in_flight"}
        ledger["operations"].append(entry)
        write_json(ledger_path, ledger)
        try:
            result = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            entry.update(status="unknown", error=f"{type(exc).__name__}: {exc}")
            write_json(ledger_path, ledger)
            raise ValueError("scheduler response unknown; inspect qstat and this ledger before any retry") from exc
        entry.update(status="returned", returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                     finished_utc=now())
        write_json(ledger_path, ledger)
        if result.returncode:
            raise ValueError(f"{argv[0]} failed: {result.stderr or result.stdout}")
        return result.stdout.strip()

    try:
        previous = None
        for item in plan["stages"]:
            answer = operation(qsub_command(plan, item, previous))
            if not JOB_ID.fullmatch(answer):
                raise ValueError("qsub returned an unrecognized job ID; reconcile qstat before any retry")
            ledger["jobs"][item["stage"]] = {"job_id": answer, "depends_on": previous, "released": False}
            write_json(ledger_path, ledger)
            previous = answer
            print(f"held {item['stage']}: {answer}", flush=True)
        if not source_matches(plan["expected_source_sha"]):
            raise ValueError("source changed during submission; all jobs remain held")
        ledger["state"] = "releasing"
        write_json(ledger_path, ledger)
        # Keep G0 held until every downstream user hold has been released.
        for stage in reversed(STAGES):
            operation(["qrls", "-h", "u", ledger["jobs"][stage]["job_id"]])
            ledger["jobs"][stage]["released"] = True
            write_json(ledger_path, ledger)
        ledger["state"] = "submitted"
        write_json(ledger_path, ledger)
        return 0
    except BaseException as exc:
        ledger.update(state="submission_incomplete", error=f"{type(exc).__name__}: {exc}",
                      recovery="Do not blindly resubmit or release G0. Inspect qstat and recorded IDs/operations. "
                               "No jobs were automatically deleted or retried.")
        write_json(ledger_path, ledger)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    launch = commands.add_parser("submit", help="one human launch, four ordered PBS jobs")
    launch.add_argument("--project", required=True)
    launch.add_argument("--env", required=True, type=Path)
    launch.add_argument("--out-root", required=True, type=Path)
    launch.add_argument("--n-samples", default=FIRST_DEPLOYMENT_SAMPLES, type=int, choices=(32, 64, 128, 256),
                        help="fixed G2/G3 candidate budget (reviewed first-deployment default: 64); "
                             "must pass G1, never selected or reduced automatically")
    launch.add_argument("--check", action="store_true", help="read-only plan and driver option checks; no qsub")
    for name in ("status", "start", "finish"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--root", type=Path, required=True)
        if name != "status":
            cmd.add_argument("--stage", choices=STAGES, required=True)
        if name == "finish":
            cmd.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "status":
            print(json.dumps(report(args.root), indent=2))
            return 0  # Read-only inspection succeeds even when the experiment failed.
        if args.action == "start":
            return job_start(args.root.resolve(), args.stage)
        if args.action == "finish":
            return job_finish(args.root.resolve(), args.stage, args.exit_code)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.project):
            raise ValueError("project name must contain only letters, digits, _, . or -")
        out_root, environment = args.out_root.resolve(), args.env.resolve()
        source = git_provenance(REPO)
        # Preview is allowed in a dirty local checkout and creates no files.
        preview_root = out_root / "kamino_pipeline_PREVIEW"
        check_workloads(args.n_samples, preview_root)
        preview = make_plan(preview_root, args.project, environment, source["commit"], args.n_samples)
        if args.check:
            preview["preview_only"] = True
            preview["source_dirty"] = source["dirty"]
            print(json.dumps(preview, indent=2))
            return 0
        if source["dirty"] is not False or not source["commit"]:
            raise ValueError("a clean local deployment commit is required; no GitHub push is needed")
        if not str(out_root).startswith(("/eagle/", "/lus/eagle/projects/")) or not out_root.is_dir():
            raise ValueError("--out-root must be an existing Eagle directory")
        if not (environment / "bin/python").is_file() or not os.access(environment / "bin/python", os.X_OK):
            raise ValueError("environment Python is missing or not executable")
        for value in (str(REPO), str(environment), str(out_root), args.project):
            if any(c.isspace() or c in ",\"'\\" for c in value):
                raise ValueError("PBS pipeline paths must not contain whitespace, commas, quotes or backslashes")
        for program in ("qsub", "qrls"):
            if not shutil.which(program):
                raise ValueError(f"{program} unavailable; submit from Polaris")
        root = Path(tempfile.mkdtemp(prefix="kamino_pipeline_", dir=out_root))
        plan = make_plan(root, args.project, environment, source["commit"], args.n_samples)
        print(f"pipeline results: {root}", flush=True)
        submit(root, plan, reserved=True)
        print("All four jobs accepted and released; PBS controls ordering. No local watcher is required.")
        print(f"Inspect: {sys.executable} {PACKAGE / 'pipeline.py'} status --root {root}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
