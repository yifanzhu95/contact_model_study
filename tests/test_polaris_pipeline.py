"""No CUDA or real PBS: fake CLI tools plus actual wrapper subprocess tests."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import REPO_ROOT

spec = importlib.util.spec_from_file_location("polaris_pipeline", REPO_ROOT / "experiments/polaris/pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
SHA = "a" * 40


@pytest.fixture
def fake_pbs(tmp_path, monkeypatch):
    """Exercise real subprocess argument handling, without a scheduler service."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "fake_pbs.py"
    script.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
log = Path(os.environ['FAKE_PBS_LOG'])
calls = json.loads(log.read_text()) if log.exists() else []
calls.append(sys.argv)
log.write_text(json.dumps(calls))
name = Path(sys.argv[0]).name
number = sum(Path(c[0]).name == name for c in calls)
if os.environ.get('FAKE_PBS_FAIL') == f'{name}:{number}':
    print('synthetic scheduler rejection', file=sys.stderr)
    sys.exit(1)
if name == 'qsub':
    print('unrecognized response' if os.environ.get('FAKE_PBS_BAD_ID') else f'{100 + number}.pbs.test')
""")
    script.chmod(0o755)
    for name in ("qsub", "qrls"):
        (bin_dir / name).symlink_to(script)
    log = tmp_path / "pbs_calls.json"
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_PBS_LOG", str(log))
    monkeypatch.setattr(pipeline, "source_matches", lambda sha: sha == SHA)
    root = tmp_path / "run"
    plan = pipeline.make_plan(root, "test_project", sys.prefix, SHA, 64)
    return root, plan, log


def calls_from(log):
    return json.loads(log.read_text()) if log.exists() else []


def test_four_held_jobs_then_reverse_release_without_future_verdicts(fake_pbs):
    root, plan, log = fake_pbs
    assert pipeline.submit(root, plan) == 0
    calls = calls_from(log)
    assert [Path(c[0]).name for c in calls] == ["qsub"] * 4 + ["qrls"] * 4
    for i, call in enumerate(calls[:4]):
        assert call[1:4] == ["-h", "-r", "n"]
        assert str(pipeline.stage_dir(root, pipeline.STAGES[i])) in call[call.index("-v") + 1]
        if i:
            assert call[call.index("-W") + 1] == f"depend=afterok:{100 + i}.pbs.test"
        else:
            assert "-W" not in call
    assert [c[-1] for c in calls[4:]] == [f"{n}.pbs.test" for n in (104, 103, 102, 101)]
    assert not any(pipeline.verdict_path(root, s).exists() for s in pipeline.STAGES)
    ledger = pipeline.read_json(root / "submission.json")
    assert ledger["state"] == "submitted" and all(j["released"] for j in ledger["jobs"].values())


@pytest.mark.parametrize("failure", ["qsub:3", "qrls:2"])
def test_partial_submission_never_releases_g0_and_is_not_retried(fake_pbs, monkeypatch, failure):
    root, plan, log = fake_pbs
    monkeypatch.setenv("FAKE_PBS_FAIL", failure)
    with pytest.raises(ValueError, match="failed"):
        pipeline.submit(root, plan)
    calls = calls_from(log)
    assert not any(Path(c[0]).name == "qrls" and c[-1] == "101.pbs.test" for c in calls)
    ledger = pipeline.read_json(root / "submission.json")
    assert ledger["state"] == "submission_incomplete"
    with pytest.raises(FileExistsError):
        pipeline.submit(root, plan)
    assert calls_from(log) == calls


def test_unknown_job_id_keeps_response_for_manual_reconciliation(fake_pbs, monkeypatch):
    root, plan, log = fake_pbs
    monkeypatch.setenv("FAKE_PBS_BAD_ID", "1")
    with pytest.raises(ValueError, match="unrecognized job ID"):
        pipeline.submit(root, plan)
    ledger = pipeline.read_json(root / "submission.json")
    assert len(calls_from(log)) == 1 and not ledger["jobs"]
    assert ledger["operations"][0]["stdout"].strip() == "unrecognized response"


def test_unknown_scheduler_timeout_is_journaled_before_call(fake_pbs, monkeypatch):
    root, plan, log = fake_pbs
    def timeout(argv, **kwargs):
        ledger = pipeline.read_json(root / "submission.json")
        assert ledger["operations"][-1]["status"] == "in_flight"
        raise subprocess.TimeoutExpired(argv, 30)
    monkeypatch.setattr(pipeline.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="response unknown"):
        pipeline.submit(root, plan)
    ledger = pipeline.read_json(root / "submission.json")
    assert len(ledger["operations"]) == 1 and ledger["operations"][0]["status"] == "unknown"


def test_source_change_keeps_every_job_held(fake_pbs, monkeypatch):
    root, plan, log = fake_pbs
    monkeypatch.setattr(pipeline, "source_matches", lambda sha: False)
    with pytest.raises(ValueError, match="source changed"):
        pipeline.submit(root, plan)
    assert [Path(c[0]).name for c in calls_from(log)] == ["qsub"] * 4


def test_preview_is_read_only_and_does_not_need_previous_verdicts(tmp_path, monkeypatch, capsys):
    checked = []
    monkeypatch.setattr(pipeline, "check_workloads", lambda samples, root: checked.append((samples, root)))
    monkeypatch.setattr(pipeline, "git_provenance", lambda repo: {"commit": SHA, "dirty": True})
    monkeypatch.setattr(pipeline, "submit", lambda *a, **kw: pytest.fail("preview submitted a job"))
    out = tmp_path / "not_created"
    assert pipeline.main(["submit", "--check", "--project", "test", "--env", sys.prefix,
                          "--out-root", str(out), "--n-samples", "64"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["preview_only"] and preview["source_dirty"] and checked[0][0] == 64
    assert not out.exists() and len(preview["stages"]) == 4


def test_real_submission_rejects_dirty_source_before_scheduler_calls(fake_pbs, monkeypatch, capsys):
    root, plan, log = fake_pbs
    monkeypatch.setattr(pipeline, "check_workloads", lambda *args: None)
    monkeypatch.setattr(pipeline, "git_provenance", lambda repo: {"commit": SHA, "dirty": True})
    assert pipeline.main(["submit", "--project", "test", "--env", sys.prefix,
                          "--out-root", str(root.parent), "--n-samples", "64"]) == 2
    assert "clean local deployment commit" in capsys.readouterr().err
    assert not root.exists() and not calls_from(log)


def test_submission_defaults_to_reviewed_first_deployment_budget(monkeypatch, capsys):
    checked = []
    monkeypatch.setattr(pipeline, "check_workloads", lambda samples, root: checked.append(samples))
    monkeypatch.setattr(pipeline, "git_provenance", lambda repo: {"commit": SHA, "dirty": False})
    assert pipeline.main(["submit", "--check", "--project", "test", "--env", "/env", "--out-root", "/out"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["n_samples"] == pipeline.FIRST_DEPLOYMENT_SAMPLES == 64
    assert checked == [64] and preview["policy"]["automatic_n_selection"] is False


def good_verdict(stage, samples=64):
    document = {"schema": ("contact_model_study.polaris_smoke_verdict.v1" if stage == "g0"
                            else "contact_model_study.polaris_stage_verdict.v1"),
                "stage": stage, "passed": True, "scaling_eligible": True, "expected_source_sha": SHA,
                "source_unchanged": True, "interrupted": False, "pending_cells": [],
                "validated_samples": [32, 64, 128, 256], "n_samples": samples}
    return document


def select_job(root, plan, stage, monkeypatch):
    item = plan["stages"][pipeline.STAGES.index(stage)]
    ledger = pipeline.read_json(root / "submission.json")
    for key, value in {
        "PBS_JOBID": ledger["jobs"][stage]["job_id"], "PROJ_DIR": plan["repo"],
        "ENV_DIR": plan["environment"], "OUTDIR": item["outdir"], "EXPECTED_SHA": SHA,
        "ALCF_PROJECT": plan["project"], "PIPELINE_ROOT": str(root), "PIPELINE_STAGE": stage,
        "STAGE": stage, "NSAMPLES": str(plan["n_samples"]),
        "PREVIOUS_VERDICT": item["previous_verdict"] or "",
    }.items():
        monkeypatch.setenv(key, value)


def complete_stage(root, plan, stage, monkeypatch):
    select_job(root, plan, stage, monkeypatch)
    assert pipeline.job_start(root, stage) == 0
    pipeline.write_json(pipeline.verdict_path(root, stage), good_verdict(stage))
    assert pipeline.job_finish(root, stage, 0) == 0
    seal_stage(root, stage)


def seal_stage(root, stage):
    # Use the actual producer, not a hand-assembled manifest fixture.
    spec = importlib.util.spec_from_file_location("pipeline_manifest", pipeline.PACKAGE / "collect_manifest.py")
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    assert collector.main(["--outdir", str(pipeline.stage_dir(root, stage)), "--job-exit-code", "0"]) == 0


def test_runtime_gates_and_full_chain_report(fake_pbs, monkeypatch):
    root, plan, log = fake_pbs
    pipeline.submit(root, plan)
    for stage in pipeline.STAGES:
        complete_stage(root, plan, stage, monkeypatch)
    result = pipeline.report(root)
    assert result["passed"] and result["blocked_by"] is None
    # Inspection must not query the scheduler or start monitoring.
    assert len(calls_from(log)) == 8


@pytest.mark.parametrize("fault", ["partial_g1", "missing_verdict", "sha", "n", "interrupted", "pending", "dirty"])
def test_forced_next_job_cannot_bypass_failed_scientific_gate(fake_pbs, monkeypatch, fault):
    root, plan, log = fake_pbs
    pipeline.submit(root, plan)
    complete_stage(root, plan, "g0", monkeypatch)
    complete_stage(root, plan, "g1", monkeypatch)
    path = pipeline.verdict_path(root, "g1")
    document = good_verdict("g1")
    if fault == "partial_g1":
        document.update(passed=False, scaling_eligible=False, validated_samples=[64])
    elif fault == "missing_verdict":
        path.unlink()
    elif fault == "sha":
        document["expected_source_sha"] = "b" * 40
    elif fault == "n":
        document["validated_samples"] = [32]
    elif fault == "interrupted":
        document["interrupted"] = True
    elif fault == "pending":
        document["pending_cells"] = ["unfinished"]
    else:
        monkeypatch.setattr(pipeline, "source_matches", lambda sha: False)
    if fault != "missing_verdict":
        pipeline.write_json(path, document)
    select_job(root, plan, "g2", monkeypatch)
    assert pipeline.job_start(root, "g2") == 2
    # Even a misleading zero process exit and a passing G2 verdict cannot clear the blocked start.
    pipeline.write_json(pipeline.verdict_path(root, "g2"), good_verdict("g2"))
    assert pipeline.job_finish(root, "g2", 0) == 2
    assert pipeline.read_json(pipeline.status_path(root, "g2"))["execution_status"] == "blocked"
    assert not pipeline.report(root)["passed"]


def test_job_failure_and_duplicate_start_preserve_evidence(fake_pbs, monkeypatch):
    root, plan, log = fake_pbs
    pipeline.submit(root, plan)
    select_job(root, plan, "g0", monkeypatch)
    assert pipeline.job_start(root, "g0") == 0
    pipeline.write_json(pipeline.verdict_path(root, "g0"), good_verdict("g0"))
    assert pipeline.job_finish(root, "g0", 7) == 2
    path = pipeline.status_path(root, "g0")
    before = path.read_text()
    with pytest.raises(ValueError, match="already has"):
        pipeline.job_start(root, "g0")
    with pytest.raises(ValueError, match="already finalized"):
        pipeline.job_finish(root, "g0", 0)
    assert path.read_text() == before
    result = pipeline.report(root)
    assert result["blocked_by"] == "g0"
    assert result["stages"]["g1"]["execution_status"] == "blocked_by_previous_stage"


@pytest.mark.parametrize("wrong", ["NSAMPLES", "PBS_JOBID"])
def test_runtime_rejects_unregistered_job_or_changed_parameters(fake_pbs, monkeypatch, wrong):
    root, plan, log = fake_pbs
    pipeline.submit(root, plan)
    select_job(root, plan, "g0", monkeypatch)
    monkeypatch.setenv(wrong, "wrong")
    assert pipeline.job_start(root, "g0") == 2


def test_missing_final_artifact_seal_blocks_next_stage_and_report(fake_pbs, monkeypatch):
    root, plan, log = fake_pbs
    pipeline.submit(root, plan)
    complete_stage(root, plan, "g0", monkeypatch)
    (pipeline.stage_dir(root, "g0") / "artifact_manifest.json").unlink()
    assert pipeline.report(root)["blocked_by"] == "g0"
    select_job(root, plan, "g1", monkeypatch)
    assert pipeline.job_start(root, "g1") == 2


def test_report_detects_changed_status_after_artifact_sealing(fake_pbs, monkeypatch):
    root, plan, log = fake_pbs
    pipeline.submit(root, plan)
    complete_stage(root, plan, "g0", monkeypatch)
    path = pipeline.status_path(root, "g0")
    status = pipeline.read_json(path)
    status["started_utc"] = "changed after sealing"
    pipeline.write_json(path, status)
    assert pipeline.report(root)["stages"]["g0"]["execution_status"] == "evidence_invalid"


@pytest.fixture
def wrapper_environment(tmp_path):
    package = tmp_path / "repo/experiments/polaris"
    package.mkdir(parents=True)
    helper = package / "pipeline.py"
    helper.write_text("""import json, os, sys
from pathlib import Path
with Path(os.environ['WRAPPER_LOG']).open('a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
sys.exit(int(os.environ.get('FAKE_START_RC' if sys.argv[1] == 'start' else 'FAKE_FINISH_RC', '0')))
""")
    (package / "collect_manifest.py").write_text("import os; raise SystemExit(int(os.environ.get('FAKE_MANIFEST_RC', '0')))\n")
    inner = package / "run_m5_smoke.pbs"
    inner.write_text('#!/bin/bash\nprintf started > "$INNER_MARKER"\nexit "${FAKE_INNER_RC:-0}"\n')
    out = tmp_path / "out"
    out.mkdir()
    env = dict(os.environ, PROJ_DIR=str(package.parents[1]), ENV_DIR=sys.prefix, OUTDIR=str(out),
               PBS_JOBID="123.test", PIPELINE_ROOT=str(out), PIPELINE_STAGE="g0",
               WRAPPER_LOG=str(tmp_path / "wrapper.log"), INNER_MARKER=str(tmp_path / "inner.marker"))
    return env, inner


@pytest.mark.parametrize("inner_rc,finish_rc,manifest_rc,expected", [(0, 0, 0, 0), (7, 0, 0, 7),
                                                                          (0, 2, 0, 2), (0, 0, 3, 3)])
def test_real_pbs_wrapper_preserves_failures(wrapper_environment, inner_rc, finish_rc, manifest_rc, expected):
    env, inner = wrapper_environment
    env.update(FAKE_INNER_RC=str(inner_rc), FAKE_FINISH_RC=str(finish_rc), FAKE_MANIFEST_RC=str(manifest_rc))
    result = subprocess.run(["bash", str(pipeline.PACKAGE / "run_pipeline_stage.pbs")], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == expected, result.stderr
    assert Path(env["INNER_MARKER"]).read_text() == "started"
    calls = [json.loads(line) for line in Path(env["WRAPPER_LOG"]).read_text().splitlines()]
    assert [c[0] for c in calls] == ["start", "finish"]
    assert calls[-1][-1] == str(inner_rc)


def test_real_wrapper_does_not_launch_inner_after_gate_failure(wrapper_environment):
    env, inner = wrapper_environment
    env["FAKE_START_RC"] = "2"
    result = subprocess.run(["bash", str(pipeline.PACKAGE / "run_pipeline_stage.pbs")], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and not Path(env["INNER_MARKER"]).exists()


def test_real_wrapper_forwards_termination_and_records_failure(wrapper_environment):
    env, inner = wrapper_environment
    inner.write_text('#!/bin/bash\ntrap \'exit 143\' TERM\nprintf started > "$INNER_MARKER"\nsleep 60\n')
    process = subprocess.Popen(["bash", str(pipeline.PACKAGE / "run_pipeline_stage.pbs")], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not Path(env["INNER_MARKER"]).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert Path(env["INNER_MARKER"]).exists()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 143, stderr
        calls = [json.loads(line) for line in Path(env["WRAPPER_LOG"]).read_text().splitlines()]
        assert calls[-1][-1] == "143"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
