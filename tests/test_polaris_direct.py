"""CPU/fake-PBS coverage for the direct research run; no GPU claim is made."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import REPO_ROOT
from ContactModelStudy.Drivers import run_episodes as drv
from ContactModelStudy.Drivers.EpisodePool import RemotePlanner, describeRollout
from ContactModelStudy.Utils.ContactModelPresets import _presetConfig
from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder, _Episode

spec = importlib.util.spec_from_file_location("polaris_direct", REPO_ROOT / "experiments/polaris/direct.py")
direct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(direct)
SHA = "a" * 40


def plan():
    args = argparse.Namespace(queue="capacity", walltime="04:00:00", project="test",
                              env=Path(sys.prefix), n_samples=64, n_episodes=24, steps=100, max_iterations=400)
    return direct.make_plan(args, SHA)


def result(p, failure=False, valid=True):
    episodes = [{"summary": {"episode": i, "finish_reason": "planner_failure" if failure else "timeout",
                              "success": False, "n_steps": 0 if failure else 100,
                              "planner_validation": {"planner_devices": [str(i % 4)]}}}
                for i in range(p["n_episodes"])]
    doc = {"configs": {"metadata": {"provenance": {"source": {"commit": SHA, "dirty": False}}}},
           "episodes": episodes, "validation": {"status": "pass" if valid else "fail", "scaling_eligible": valid}}
    stats = {"complete": True, "pool": {"gpus": ["0", "1", "2", "3"], "workers": 8, "planners_per_gpu": 1}}
    return doc, stats


def test_direct_command_preserves_tolerance_and_sets_only_explicit_cap():
    p = plan()
    argv = direct.driver_command(p, Path("/output"))
    from ContactModelStudy.Drivers.run_episodes_pooled import build_parser
    args = drv.parseArgs(argv[2:], parser=build_parser())
    task, _ = drv.buildTasks(args)
    cfg = _presetConfig("M5", **drv.rolloutSimOverrides(args, task))
    assert cfg.padmm_max_iterations == 400 and cfg.padmm_tolerance == 5e-4
    assert cfg.resolved_horizon == 5 and cfg.resolved_substeps == 16
    assert args.stop_on_invalid_plan and args.graph and args.workers == 8
    assert _presetConfig("M5").padmm_max_iterations == 100


@pytest.mark.parametrize("extra", [["--kamino-max-iterations", "0"],
                                    ["--rollout-model", "M2", "--kamino-max-iterations", "400"]])
def test_bad_override_rejected(extra):
    with pytest.raises(SystemExit):
        drv.parseArgs(extra)


def test_direct_completion_does_not_require_or_forge_validation():
    p = plan()
    doc, stats = result(p, failure=True, valid=False)
    status = direct.assess(doc, stats, p, 0)
    assert status["execution_status"] == "completed"
    assert status["planner_failures"] == 24
    assert status["validation_status"] == "fail" and not status["scaling_eligible"]
    doc["episodes"].pop()
    assert direct.assess(doc, stats, p, 0)["execution_status"] == "incomplete"


@pytest.mark.parametrize("mutation", ["source", "pool", "interrupted", "returncode"])
def test_direct_rejects_false_execution_completion(mutation):
    p = plan()
    doc, stats = result(p)
    rc = 0
    if mutation == "source":
        doc["configs"]["metadata"]["provenance"]["source"]["dirty"] = True
    elif mutation == "pool":
        stats["complete"] = False
    elif mutation == "interrupted":
        doc["episodes"][0]["summary"]["finish_reason"] = "interrupted"
    else:
        rc = 1
    s = direct.assess(doc, stats, p, rc)
    assert s["execution_status"] == "incomplete" and not s["scaling_eligible"]


def test_one_held_job_registration_before_release(tmp_path, monkeypatch):
    calls = []
    def scheduler(argv, **kwargs):
        ledger = direct.read_json(tmp_path / "submission.json")
        assert ledger["operations"][-1]["status"] == "in_flight"
        if argv[0] == "qrls":
            assert ledger["job_id"] == "123.polaris" and ledger["state"] == "registered_held"
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "123.polaris\n" if argv[0] == "qsub" else "", "")
    monkeypatch.setattr(direct.subprocess, "run", scheduler)
    monkeypatch.setattr(direct, "source_matches", lambda sha: sha == SHA)
    assert direct.submit(plan(), tmp_path) == "123.polaris"
    assert [c[0] for c in calls] == ["qsub", "qrls"]
    assert calls[0][1:4] == ["-h", "-r", "n"]
    assert "capacity" in calls[0] and "walltime=04:00:00" in calls[0]
    assert direct.read_json(tmp_path / "submission.json")["released"]
    original = (tmp_path / "submission.json").read_bytes()
    with pytest.raises(FileExistsError):
        direct.submit(plan(), tmp_path)
    assert len(calls) == 2 and (tmp_path / "submission.json").read_bytes() == original


@pytest.mark.parametrize("mode", ["rejected", "unknown", "release", "bad_id"])
def test_submission_failure_never_retries(tmp_path, monkeypatch, mode):
    calls = []
    def scheduler(argv, **kwargs):
        calls.append(argv)
        if mode == "unknown":
            raise subprocess.TimeoutExpired(argv, 60)
        rejected = mode == "rejected" or (mode == "release" and argv[0] == "qrls")
        return subprocess.CompletedProcess(argv, int(rejected), "bad" if mode == "bad_id" else "123.polaris", "rejected")
    monkeypatch.setattr(direct.subprocess, "run", scheduler)
    monkeypatch.setattr(direct, "source_matches", lambda sha: True)
    with pytest.raises(ValueError):
        direct.submit(plan(), tmp_path)
    assert sum(c[0] == "qsub" for c in calls) == 1
    assert direct.read_json(tmp_path / "submission.json")["state"] == "submission_incomplete"


def test_read_only_preview(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(direct, "git_provenance", lambda _: {"commit": SHA, "dirty": True})
    monkeypatch.setattr(direct, "submit", lambda *a: pytest.fail("preview submitted"))
    output = tmp_path / "not_created"
    assert direct.main(["submit", "--check", "--project", "test", "--env", sys.prefix,
                        "--out-root", str(output)]) == 0
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["budget_seconds"] == 13800
    assert direct.main(["submit", "--check", "--project", "test", "--env", sys.prefix,
                        "--out-root", str(output), "--queue", "debug"]) == 2


def test_remote_planner_carries_failure_flag():
    args = drv.parseArgs(["--rollout-model", "M5"])
    task, _ = drv.buildTasks(args)
    planner = RemotePlanner(describeRollout(args, task), task, queue.Queue(), queue.Queue(), 0, threading.Event())
    planner._replies.put({"state": {}, "min_cost": 1e30, "plan_s": 0.1, "sigma": None,
                          "rollout_diagnostics": [], "plan_ok": False, "action": np.zeros(16)})
    planner.Plan(np.zeros(23), np.zeros(22))
    assert planner.last_plan_ok is False


def test_all_invalid_stops_before_action_or_physics(monkeypatch):
    args = drv.parseArgs(["--stop-on-invalid-plan", "--settle", "0", "--steps", "1"])
    events = []
    task = SimpleNamespace(timestep=0.004)
    eval_task = SimpleNamespace(timestep=0.0005, setSimToInitialState=lambda s: None,
                               getInitialState=lambda: (None, None, np.ones(16)),
                               isFailure=lambda s: False, isSuccess=lambda s: False)
    sim = SimpleNamespace(GetState=lambda: (np.zeros(23), np.zeros(22)), GetControl=lambda: np.ones(16),
                          Step=lambda n: events.append(("physics", n)), SetControl=lambda u: events.append(("action", u)))
    planner = SimpleNamespace(Reset=lambda m: None, substeps=16, Plan=lambda *a, **k: np.zeros(16),
                              last_plan_ok=False, last_plan_seconds=0.1)
    recorder = SimpleNamespace(recordPlanFailure=lambda *a, **k: events.append(("rejected", k)),
                               recordStateAndAction=lambda *a, **k: pytest.fail("invalid action recorded as applied"))
    monkeypatch.setattr(drv, "_newGoal", lambda *a: None)
    assert drv.run_episode(args, task, eval_task, sim, planner, None, recorder) == "planner_failure"
    assert not any(e[0] == "action" for e in events)
    assert not any(e[0] == "physics" and e[1] > 0 for e in events)
    assert sum(e[0] == "rejected" for e in events) == 1


def test_rejected_attempt_is_durable_not_an_applied_step(tmp_path):
    rec = EpisodeRecorder.__new__(EpisodeRecorder)
    rec.configs = {"rollout_simulator": {"class": "Kamino"}, "planner": {"config": {"n_iterations": 1}}}
    rec._active = _Episode(id="failed", configs=rec.configs)
    rec.episodes = []
    rec.task = SimpleNamespace()
    rec.simulator = SimpleNamespace(nq=1, nv=1, nu=1, GetState=lambda: (np.zeros(1), np.zeros(1)))
    rec.planner = SimpleNamespace(last_min_cost=1e30, last_rollout_diagnostics=[{
        "all_worlds_valid": False, "valid_worlds": 0, "invalid_worlds": 64,
        "blocking_reasons": {"nonconverged_worlds": 64}}])
    rec.recordPlanFailure(np.zeros(1), np.zeros(1), 0.1)
    rec.episodeFinished("planner_failure")
    path = rec.Save(tmp_path / "results.json")
    doc = direct.read_json(path)
    ep = doc["episodes"][0]
    assert ep["n_steps"] == 0 and ep["summary"]["failed"]
    assert ep["summary"]["rejected_plan"]["action_applied"] is False
    assert ep["summary"]["planner_validation"]["expected_rollouts"] == 1
    assert ep["summary"]["planner_validation"]["invalid_worlds"] == 64
    assert doc["validation"]["status"] == "fail"
    assert np.load(tmp_path / ep["file"], allow_pickle=False).size == 0


def test_actual_runner_collects_batch_even_when_validation_fails(tmp_path, monkeypatch):
    p = plan()
    direct.write_json(tmp_path / "plan.json", p)
    digest = hashlib.sha256((tmp_path / "plan.json").read_bytes()).hexdigest()
    direct.write_json(tmp_path / "submission.json", {"job_id": "123", "plan_sha256": digest})
    direct.write_json(tmp_path / "preflight.json", {"passed": True, "expected_source_sha": SHA,
                                                   "requested_gpus": "0,1,2,3"})
    doc, stats = result(p, failure=True, valid=False)
    class Child:
        def __init__(self, *args, **kwargs):
            direct.write_json(tmp_path / "results.json", doc)
            direct.write_json(tmp_path / "pool_stats.json", stats)
        def wait(self, **kwargs):
            return 0
        def poll(self):
            return 0
    monkeypatch.setenv("PBS_JOBID", "123")
    monkeypatch.setattr(direct, "source_matches", lambda _: True)
    monkeypatch.setattr(direct.subprocess, "Popen", Child)
    assert direct.run(tmp_path) == 0
    status = direct.read_json(tmp_path / "status.json")
    assert status["execution_status"] == "completed" and status["validation_status"] == "fail"
    original = (tmp_path / "status.json").read_bytes()
    with pytest.raises(FileExistsError):
        direct.run(tmp_path)
    assert (tmp_path / "status.json").read_bytes() == original


def test_modified_plan_cannot_start_driver(tmp_path, monkeypatch):
    p = plan()
    direct.write_json(tmp_path / "plan.json", p)
    digest = hashlib.sha256((tmp_path / "plan.json").read_bytes()).hexdigest()
    direct.write_json(tmp_path / "submission.json", {"job_id": "123", "plan_sha256": digest})
    p["n_samples"] = 256
    direct.write_json(tmp_path / "plan.json", p)
    monkeypatch.setenv("PBS_JOBID", "123")
    monkeypatch.setattr(direct.subprocess, "Popen", lambda *a, **k: pytest.fail("modified plan launched"))
    assert direct.run(tmp_path) == 2
    assert "plan was modified" in direct.read_json(tmp_path / "status.json")["error"]


def test_preflight_failure_has_status_and_quiet_valid_seal(tmp_path, capsys):
    (tmp_path / "job.log").write_text("preflight failed\n")
    (tmp_path / "submission.json").write_text('{"state": "releasing"}')
    assert direct.finalize(tmp_path, 2) == 0
    assert capsys.readouterr().out == ""
    status = direct.read_json(tmp_path / "status.json")
    assert status["execution_status"] == "error" and not status["scaling_eligible"]
    manifest = direct.read_json(tmp_path / "artifact_manifest.json")
    assert manifest["job_exit_code"] == 2
    assert "submission.json" in manifest["excluded_paths"]
    assert not any(e["path"] == "submission.json" for e in manifest["files"])
    for entry in manifest["files"]:
        path = tmp_path / entry["path"]
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("preflight_rc", [0, 2])
def test_actual_pbs_wrapper_collects_logs_and_errors(tmp_path, monkeypatch, preflight_rc):
    import os
    # Substitute module/conda and compute-only commands; execute the actual PBS
    # wrapper, actual finalizer and actual collector, without touching a GPU.
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    fake = env_dir / "bin/python"
    fake.write_text(f'''#!{sys.executable}
import os, sys, json
from pathlib import Path
argv = sys.argv[1:]
if argv[0].endswith("preflight.py"):
    Path(argv[argv.index("--output") + 1]).write_text(json.dumps({{"passed": {preflight_rc == 0}}}))
    print("preflight called")
    sys.exit({preflight_rc})
if argv[0].endswith("direct.py") and argv[1] == "run":
    root = Path(argv[argv.index("--outdir") + 1])
    (root / "status.json").write_text(json.dumps({{"execution_status": "completed", "validation_status": "fail", "scaling_eligible": False}}))
    print("experiment called")
    sys.exit(0)
if argv[0].endswith("direct.py"):
    os.execv({sys.executable!r}, [{sys.executable!r}, *argv])
print("mock package command")
''')
    fake.chmod(0o755)
    hook = tmp_path / "hook.sh"
    hook.write_text('module() { :; }\nconda() { :; }\n')
    output = tmp_path / "output"
    output.mkdir()
    (tmp_path / "warp").mkdir()
    # mkdir on /local is substituted too; this test is not a Polaris host.
    hook.write_text(hook.read_text() + 'mkdir() { :; }\n')
    environ = dict(os.environ, PROJ_DIR=str(REPO_ROOT), ENV_DIR=str(env_dir), OUTDIR=str(output),
                   EXPECTED_SHA=SHA, ALCF_PROJECT="test", PBS_JOBID="test", BASH_ENV=str(hook),
                   PATH=str(env_dir / "bin") + os.pathsep + os.environ["PATH"])
    answer = subprocess.run(["bash", str(REPO_ROOT / "experiments/polaris/run_direct.pbs")],
                            capture_output=True, text=True, env=environ, timeout=20)
    assert answer.returncode == preflight_rc, answer.stderr
    manifest = direct.read_json(output / "artifact_manifest.json")
    assert manifest["job_exit_code"] == preflight_rc
    log = (output / "job.log").read_text()
    assert ("experiment called" in log) == (preflight_rc == 0)
    assert direct.read_json(output / "wrapper_status.json")["exit_code"] == preflight_rc
    assert direct.read_json(output / "status.json")["execution_status"] == ("completed" if preflight_rc == 0 else "error")
    for entry in manifest["files"]:
        path = output / entry["path"]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
