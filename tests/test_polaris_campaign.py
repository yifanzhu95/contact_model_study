"""Full-matrix CPU/fake-PBS tests. No A100 run or queue access is claimed."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from conftest import REPO_ROOT
from ContactModelStudy.Drivers import run_episodes as drv
from ContactModelStudy.Drivers.EpisodePool import episodeSeeds
from ContactModelStudy.Utils.ContactModelPresets import _presetConfig

spec = importlib.util.spec_from_file_location("polaris_campaign_test", REPO_ROOT / "experiments/polaris/campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
SHA = "a" * 40


def plan(resume=None):
    return campaign.make_plan(argparse.Namespace(queue="capacity", walltime="24:00:00", project="test",
        env=Path(sys.prefix), resume_from=resume), SHA)


def document(cell, valid=False, planner_failure=False):
    episodes = []
    for i in range(int(cell.row["n_episodes"])):
        goal, noise = episodeSeeds(int(cell.row["seed"]), i)
        episodes.append({"summary": {"episode": i, "goal_seed": goal, "noise_seed": noise,
            "finish_reason": "planner_failure" if planner_failure else "timeout",
            "n_steps": 0 if planner_failure else int(cell.row["steps"]), "success": False,
            "rejected_plan": {"action_applied": False} if planner_failure else None,
            "planner_validation": {"planner_devices": cell.gpus.split(",")}}})
    doc = {"configs": {"metadata": {"provenance": {"source": {"commit": SHA, "dirty": False}}},
        "rollout_simulator": {"config": {"rollout_acceptance": "exploratory",
                                 "padmm_max_iterations": 100, "padmm_tolerance": 5e-4}}},
        "episodes": episodes, "n_episodes": len(episodes),
        "validation": {"status": "pass" if valid else "fail", "scaling_eligible": valid}}
    stats = {"complete": True, "pool": {"gpus": cell.gpus.split(","),
             "workers": cell.workers, "planners_per_gpu": 1}}
    return doc, stats


def test_original_matrix_and_uniform_numerical_policy():
    cells = campaign.matrix()
    assert [len(cells[s]) for s in campaign.STAGES] == [12, 4, 27]
    flattened = sum(cells.values(), [])
    assert sum(int(c.row["n_episodes"]) for c in flattened) == 135
    assert sum(int(c.row["n_episodes"]) * int(c.row["steps"]) for c in flattened) == 12540
    assert {(c.gpus, c.workers) for c in cells["g2"]} == {
        ("0", 4), ("0,1,2,3", 4), ("0,1,2,3", 8), ("0,1,2,3", 16)}
    assert {int(c.row["n_samples"]) for c in cells["g1"]} == {32, 64, 128, 256}
    assert {(float(c.row["temperature"]), float(c.row["noise_sigma"]), int(c.row["seed"]))
        for c in cells["g3"]} == {(t, s, seed) for t in (10,40,100) for s in (.02,.05,.10) for seed in (0,1,2)}
    campaign.validate_commands()
    batches = campaign.stages.batch_module()
    from ContactModelStudy.Drivers.run_episodes_pooled import build_parser
    for cell in flattened:
        args = drv.parseArgs(campaign.driver_command(cell, Path("/output"), batches)[2:],
            parser=build_parser() if cell.workers else None)
        task, _ = drv.buildTasks(args)
        cfg = _presetConfig("M5", **drv.rolloutSimOverrides(args, task))
        assert cfg.rollout_acceptance == "exploratory" and cfg.padmm_max_iterations == 100
        assert cfg.padmm_tolerance == 5e-4 and cfg.resolved_horizon == 5 and cfg.resolved_substeps == 16
        assert args.graph and args.stop_on_invalid_plan
    assert _presetConfig("M5").rollout_acceptance == "strict"
    with pytest.raises(SystemExit): drv.parseArgs(["--rollout-model", "M2", "--kamino-acceptance", "exploratory"])


def test_warnings_and_zero_step_planner_failures_complete_without_fake_qualification():
    cell = campaign.matrix()["g2"][1]
    doc, stats = document(cell, planner_failure=True)
    status = campaign.assess(doc, stats, cell, SHA, 0)
    assert status["execution_status"] == "completed"
    assert status["planner_failures"] == 24 and status["plans_attempted"] == 24
    assert not status["reference_eligible"] and status["validation_status"] == "fail"
    assert status["control_steps_applied"] == 0 and not status["full_step_budget_completed"]


@pytest.mark.parametrize("mutation", ["source", "policy", "partial", "interrupted", "seed", "pool", "rc"])
def test_execution_integrity_is_not_relaxed(mutation):
    cell = campaign.matrix()["g2"][1]
    doc, stats = document(cell)
    rc = 0
    if mutation == "source": doc["configs"]["metadata"]["provenance"]["source"]["dirty"] = True
    if mutation == "policy": doc["configs"]["rollout_simulator"]["config"]["rollout_acceptance"] = "strict"
    if mutation == "partial": doc["episodes"].pop()
    if mutation == "interrupted": doc["episodes"][0]["summary"]["finish_reason"] = "interrupted"
    if mutation == "seed": doc["episodes"][0]["summary"]["noise_seed"] += 1
    if mutation == "pool": stats["pool"]["workers"] = 8
    if mutation == "rc": rc = 1
    status = campaign.assess(doc, stats, cell, SHA, rc)
    assert status["execution_status"] == "incomplete" and not status["reference_eligible"]
    assert status["global_failure"] == (mutation in ("source", "policy"))


def test_isolated_oom_without_result_is_not_global_infrastructure_failure():
    cell = campaign.matrix()["g1"][0]
    status = campaign.assess({}, None, cell, SHA, 1)
    assert status["execution_status"] == "incomplete" and not status["global_failure"]


def test_one_submit_for_full_campaign_registered_before_release(tmp_path, monkeypatch):
    calls = []
    def scheduler(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "qrls":
            assert campaign.read_json(tmp_path / "submission.json")["job_id"] == "123.polaris"
        return subprocess.CompletedProcess(argv, 0, "123.polaris" if argv[0] == "qsub" else "", "")
    monkeypatch.setattr(campaign.direct.subprocess, "run", scheduler)
    monkeypatch.setattr(campaign.direct, "source_matches", lambda sha: True)
    assert campaign.direct.submit(plan(), tmp_path) == "123.polaris"
    assert [c[0] for c in calls] == ["qsub", "qrls"]
    assert "walltime=24:00:00" in calls[0] and calls[0][-1].endswith("run_campaign.pbs")
    assert campaign.read_json(tmp_path / "plan.json")["requested_episodes"] == 135


def test_preview_does_not_create_files_or_submit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(campaign, "git_provenance", lambda _: {"commit": SHA, "dirty": True})
    monkeypatch.setattr(campaign.direct, "submit", lambda *a: pytest.fail("submitted preview"))
    output = tmp_path / "not-created"
    assert campaign.main(["submit", "--check", "--project", "test", "--env", sys.prefix,
                          "--out-root", str(output)]) == 0
    p = json.loads(capsys.readouterr().out)
    assert p["requested_cells"] == 43 and p["budget_seconds"] == 85800
    assert not output.exists()


def prepare_run(root, monkeypatch, p=None):
    root.mkdir(exist_ok=True)
    p = p or plan()
    campaign.write_json(root / "plan.json", p)
    campaign.write_json(root / "submission.json", {"job_id": "123.polaris", "plan_sha256": campaign.sha256(root / "plan.json")})
    campaign.write_json(root / "preflight.json", {"passed": True, "expected_source_sha": SHA, "requested_gpus": "0,1,2,3"})
    monkeypatch.setenv("PBS_JOBID", "123.polaris")
    monkeypatch.setattr(campaign.direct, "source_matches", lambda sha: True)
    monkeypatch.setattr(campaign.time, "sleep", lambda _: None)
    return p


def fake_active(monkeypatch, failure=None, fatal=False, interrupt=False):
    started, finished, live = [], [], set()
    class FakeActive:
        def __init__(self, cell, directory, batches, sha):
            self.cell, self.directory, self.cancelled = cell, directory, False
            stage = directory.parent.name
            if stage != "g3": assert not live
            assert not live.intersection(cell.gpus.split(","))
            live.update(cell.gpus.split(","))
            started.append((stage, cell.name, len(live)))
            if stage == "g2": assert len(finished) >= 12
            if stage == "g3": assert len(finished) >= 16
            if interrupt: signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        def ready(self): return self.cancelled if interrupt else True
        def cancel(self): self.cancelled = True
        def finish(self):
            live.difference_update(self.cell.gpus.split(","))
            finished.append(self.cell.name)
            doc, stats = document(self.cell)
            rc = 1 if self.cell.name == failure else 0
            result = campaign.assess(doc, stats, self.cell, SHA, rc)
            result.update(wall_s=1, performance_eligible=False, global_failure=fatal and bool(rc))
            if self.cancelled: result["execution_status"] = "incomplete"
            return result
    monkeypatch.setattr(campaign, "Active", FakeActive)
    return started, finished


@pytest.mark.parametrize("failure", [None, "g2_gpu4_w8"])
def test_full_scheduler_runs_all_43_cells_despite_warnings_and_isolated_error(tmp_path, monkeypatch, failure):
    prepare_run(tmp_path, monkeypatch)
    started, finished = fake_active(monkeypatch, failure)
    assert campaign.run(tmp_path) == (2 if failure else 0)
    assert len(started) == len(finished) == 43
    assert max(n for s, c, n in started if s == "g3") == 4
    report = campaign.read_json(tmp_path / "status.json")
    assert report["completed_cells"] == (42 if failure else 43)
    assert report["validation_status"] == "fail"
    assert not report["scaling_eligible"] and report["stages"]["g3"]["all_cells_completed"]
    assert all(r["qualified_speedup_vs_gpu1_w4"] is None for r in report["g2_performance_comparison"].values())


def test_global_failure_stops_later_cells(tmp_path, monkeypatch):
    prepare_run(tmp_path, monkeypatch)
    started, finished = fake_active(monkeypatch, "g2_gpu4_w8", fatal=True)
    assert campaign.run(tmp_path) == 2
    assert len(started) == 15 and len(finished) == 15
    assert "global_failure" in campaign.read_json(tmp_path / "status.json")["stop_reason"]


def test_signal_cancels_children_and_preserves_partial_status(tmp_path, monkeypatch):
    prepare_run(tmp_path, monkeypatch)
    started, finished = fake_active(monkeypatch, interrupt=True)
    assert campaign.run(tmp_path) == 2
    assert len(started) == len(finished) == 1
    report = campaign.read_json(tmp_path / "status.json")
    assert report["stop_reason"] == "interrupted" and report["completed_cells"] == 0


@pytest.mark.parametrize("fault", ["plan", "job", "preflight"])
def test_run_rejects_wrong_identity_before_launch(tmp_path, monkeypatch, fault):
    prepare_run(tmp_path, monkeypatch)
    if fault == "plan":
        p = campaign.read_json(tmp_path / "plan.json")
        p["science"]["padmm_max_iterations"] = 400
        campaign.write_json(tmp_path / "plan.json", p)
    if fault == "job": monkeypatch.setenv("PBS_JOBID", "999.polaris")
    if fault == "preflight": campaign.write_json(tmp_path / "preflight.json", {"passed": False})
    monkeypatch.setattr(campaign, "Active", lambda *a: pytest.fail("invalid run launched"))
    assert campaign.run(tmp_path) == 2
    assert campaign.read_json(tmp_path / "status.json")["completed_cells"] == 0


def save_cached(directory, cell, valid=False):
    attempt = directory / "attempt_0001"
    attempt.mkdir(parents=True)
    doc, stats = document(cell, valid=valid)
    campaign.write_json(attempt / "result.json", doc)
    if cell.workers: campaign.write_json(attempt / "pool_stats.json", stats)
    (attempt / "result_0.npy").write_bytes(b"test array")
    outcome = {**campaign.assess(doc, stats, cell, SHA, 0), "returncode": 0, "wall_s": 1,
        "performance_eligible": False, "expected_source_sha": SHA, "spec_sha256": campaign.fingerprint(cell.spec()),
        "attempt": str(attempt), "artifact_sha256": campaign.artifact_hashes(attempt)}
    campaign.write_json(directory / "cell.status.json", outcome)
    return outcome


def test_completed_warning_cell_reused_without_touching_old_evidence(tmp_path):
    cell = campaign.matrix()["g1"][0]
    old, new = tmp_path / "old", tmp_path / "new"
    directory = old / "g1" / cell.name
    save_cached(directory, cell)
    before = campaign.artifact_hashes(old)
    assert campaign.cached_result(directory, cell, SHA)["validation_status"] == "fail"
    imported = campaign.import_completed(old, new, SHA)
    assert len(imported["g1"]) == 1
    assert campaign.cached_result(new / "g1" / cell.name, cell, SHA)
    assert before == campaign.artifact_hashes(old)
    (directory / "attempt_0001" / "result_0.npy").write_bytes(b"modified")
    assert campaign.cached_result(directory, cell, SHA) is None
    assert campaign.cached_result(new / "g1" / cell.name, cell, SHA)
    with pytest.raises(ValueError, match="not silently rerunning"):
        campaign.import_completed(old, tmp_path / "tampered-resume", SHA)


def test_partial_attempts_preserved_for_explicit_resume(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    cell = campaign.matrix()["g1"][0]
    attempt = old / "g1" / cell.name / "attempt_0001"
    attempt.mkdir(parents=True)
    (attempt / "driver.log").write_text("partial evidence\n")
    campaign.write_json(attempt.parent / "cell.status.json", {"execution_status": "incomplete"})
    imported = campaign.import_completed(old, new, SHA)
    assert not imported["g1"]
    assert (new / "g1" / cell.name / "attempt_0001" / "driver.log").read_text() == "partial evidence\n"


@pytest.mark.parametrize("state", ["R", "Q", "F"])
def test_resume_requires_previous_job_confirmed_terminal(tmp_path, monkeypatch, state):
    previous = plan()
    campaign.write_json(tmp_path / "plan.json", previous)
    campaign.write_json(tmp_path / "submission.json", {"job_id": "111.polaris", "plan_sha256": campaign.sha256(tmp_path / "plan.json")})
    calls = []
    def qstat(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, f"Job Id: 111.polaris\n job_state = {state}\n", "")
    monkeypatch.setattr(campaign.subprocess, "run", qstat)
    if state == "F": assert campaign.check_resume(plan(tmp_path))["job_id"] == "111.polaris"
    else:
        with pytest.raises(ValueError, match="previous job has finished"): campaign.check_resume(plan(tmp_path))
    assert calls == [["qstat", "-xf", "111.polaris"]]


def test_resume_skips_done_warning_cell_and_keeps_full_matrix(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    prepare_run(old, monkeypatch)
    cell = campaign.matrix()["g1"][0]
    save_cached(old / "g1" / cell.name, cell)
    p = prepare_run(new, monkeypatch, plan(old))
    for root in (old, new):
        for name in ("pip-freeze.txt", "conda-explicit.txt"): (root / name).write_text("same environment\n")
    monkeypatch.setattr(campaign, "check_resume", lambda plan: {"root": str(old)})
    started, finished = fake_active(monkeypatch)
    # The scheduling assertions below intentionally count imported completion.
    # Use the actual state to establish stage ordering in the resumed run.
    finished.append(cell.name)
    assert campaign.run(new) == 0
    assert len(started) == 42 and len(finished) == 43
    assert all(name != cell.name for stage, name, n in started)
    assert campaign.read_json(new / "status.json")["completed_cells"] == 43


def test_telemetry_failure_is_nonblocking_measurement_warning(tmp_path, monkeypatch):
    monitor = campaign.Telemetry(tmp_path / "metrics.jsonl", "0")
    monkeypatch.setattr(monitor, "sample", lambda: (_ for _ in ()).throw(RuntimeError("measurement unavailable")))
    monitor.start()
    assert monitor.finish()["errors"]


@pytest.mark.parametrize("malformed", [False, True])
def test_real_child_and_attempt_hashes_without_gpu(tmp_path, monkeypatch, malformed):
    cell = campaign.matrix()["g3"][2]
    doc, _ = document(cell)
    if malformed:
        doc["episodes"][0]["summary"]["n_steps"] = "invalid step count"
    def command(cell, attempt, batches):
        code = ("import os; from pathlib import Path; "
                f"assert os.environ['CUDA_VISIBLE_DEVICES'] == {cell.gpus!r}; "
                f"Path({str(attempt / 'result.json')!r}).write_text({json.dumps(doc)!r}); print('mock complete')")
        return [sys.executable, "-c", code]
    class FakeTelemetry:
        def __init__(self, path, gpus): self.gpus = gpus
        def start(self): pass
        def finish(self): return {"per_gpu": {self.gpus: {"samples": 0}}, "errors": ["measurement unavailable"]}
    monkeypatch.setattr(campaign, "driver_command", command)
    monkeypatch.setattr(campaign, "Telemetry", FakeTelemetry)
    active = campaign.Active(cell, tmp_path / cell.name, None, SHA)
    deadline = time.monotonic() + 5
    while not active.ready() and time.monotonic() < deadline: time.sleep(.01)
    outcome = active.finish()
    assert outcome["execution_status"] == ("incomplete" if malformed else "completed")
    assert not outcome["global_failure"]
    assert not outcome["telemetry_complete"] and not outcome["performance_eligible"]
    assert bool(campaign.cached_result(active.directory, cell, SHA)) is (not malformed)
    assert (active.attempt / "outcome.json").exists()
    second = campaign.Active(cell, active.directory, None, SHA)
    while not second.ready(): time.sleep(.01)
    second.finish()
    assert second.attempt.name == "attempt_0002" and (active.attempt / "result.json").exists()


def test_fast_early_failures_never_generate_speedup():
    p, results = plan(), {s: {} for s in campaign.STAGES}
    for cell in campaign.matrix()["g2"]:
        doc, stats = document(cell, valid=True, planner_failure=True)
        r = campaign.assess(doc, stats, cell, SHA, 0)
        r.update(wall_s=.1, performance_eligible=False)
        results["g2"][cell.name] = r
    report = campaign.summary(p, results)
    assert all(not r["eligible"] for r in report["g2_performance_comparison"].values())


def test_exploratory_ratio_is_not_mislabeled_strict_speedup_and_resume_not_pooled():
    results = {s: {} for s in campaign.STAGES}
    for i, cell in enumerate(campaign.matrix()["g2"]):
        doc, stats = document(cell, valid=False)
        r = campaign.assess(doc, stats, cell, SHA, 0)
        r.update(wall_s=4/(i+1), performance_eligible=False, telemetry_complete=True)
        results["g2"][cell.name] = r
    comparisons = campaign.summary(plan(), results)["g2_performance_comparison"]
    assert comparisons["g2_gpu4_w8"]["exploratory_matched_work_wall_ratio"] == 3
    assert comparisons["g2_gpu4_w8"]["qualified_speedup_vs_gpu1_w4"] is None
    results["g2"]["g2_gpu4_w8"]["reused_from"] = "/previous/allocation"
    assert campaign.summary(plan(), results)["g2_performance_comparison"]["g2_gpu4_w8"]["exploratory_matched_work_wall_ratio"] is None


def test_real_controller_real_children_all_cells_and_all_hashes(tmp_path, monkeypatch):
    prepare_run(tmp_path, monkeypatch)
    # Keep the actual scheduler, Active lifecycle, subprocess launch, result
    # assessment, per-attempt hashes and final manifest. Only physics and the
    # device monitor are synthetic, and no scheduler is contacted.
    monkeypatch.undo()
    monkeypatch.setenv("PBS_JOBID", "123.polaris")
    monkeypatch.setattr(campaign.direct, "source_matches", lambda sha: True)
    def command(cell, attempt, batches):
        doc, stats = document(cell)
        code = ("import os; from pathlib import Path; "
                f"assert os.environ['CUDA_VISIBLE_DEVICES'] == {cell.gpus!r}; "
                f"Path({str(attempt / 'result.json')!r}).write_text({json.dumps(doc)!r}); "
                f"Path({str(attempt / 'pool_stats.json')!r}).write_text({json.dumps(stats)!r}); print('synthetic physics')")
        return [sys.executable, "-c", code]
    class FakeTelemetry:
        def __init__(self, path, gpus): self.gpus = gpus
        def start(self): pass
        def finish(self): return {"per_gpu": {g: {"samples": 1} for g in self.gpus.split(",")}, "errors": []}
    monkeypatch.setattr(campaign, "driver_command", command)
    monkeypatch.setattr(campaign, "Telemetry", FakeTelemetry)
    assert campaign.run(tmp_path) == 0
    report = campaign.read_json(tmp_path / "status.json")
    assert report["completed_cells"] == 43 and report["recorded_episodes"] == 135
    assert report["plans_attempted"] == 12540 and not report["reference_eligible"]
    for stage, cells in campaign.matrix().items():
        for cell in cells:
            assert campaign.cached_result(tmp_path / stage / cell.name, cell, SHA)
    assert campaign.direct.finalize(tmp_path, 0) == 0
    manifest = campaign.read_json(tmp_path / "artifact_manifest.json")
    for entry in manifest["files"]:
        assert campaign.sha256(tmp_path / entry["path"]) == entry["sha256"]


def test_budget_counts_preflight_time_and_never_launches_when_already_spent(tmp_path, monkeypatch):
    prepare_run(tmp_path, monkeypatch)
    monkeypatch.setenv("CAMPAIGN_START_UNIX", str(time.time() - 90000))
    monkeypatch.setattr(campaign, "Active", lambda *a: pytest.fail("launched after budget elapsed"))
    assert campaign.run(tmp_path) == 2
    assert campaign.read_json(tmp_path / "status.json")["stop_reason"] == "walltime_budget_exhausted"


def test_changed_environment_stops_before_launch(tmp_path, monkeypatch):
    prepare_run(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "check_environment", lambda: (_ for _ in ()).throw(ValueError("runtime environment changed")))
    monkeypatch.setattr(campaign, "Active", lambda *a: pytest.fail("launched changed environment"))
    assert campaign.run(tmp_path) == 2
    assert "environment changed" in campaign.read_json(tmp_path / "status.json")["stop_reason"]


@pytest.mark.parametrize("preflight_rc", [0, 2])
def test_actual_single_campaign_pbs_wrapper_seals_stable_logs(tmp_path, preflight_rc):
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    fake = env_dir / "bin/python"
    fake.write_text(f'''#!{sys.executable}
import os, sys, json
from pathlib import Path
argv = sys.argv[1:]
if argv[0].endswith("preflight.py"):
    Path(argv[argv.index("--output")+1]).write_text(json.dumps({{"passed": {preflight_rc == 0}}}))
    print("preflight called")
    sys.exit({preflight_rc})
if argv[0].endswith("campaign.py"):
    root = Path(argv[argv.index("--outdir")+1])
    (root/"status.json").write_text(json.dumps({{"execution_status": "completed", "validation_status": "fail", "scaling_eligible": False}}))
    print("full campaign called")
    sys.exit(0)
if argv[0].endswith("direct.py"):
    os.execv({sys.executable!r}, [{sys.executable!r}, *argv])
print("mock package command")
''')
    fake.chmod(0o755)
    hook = tmp_path / "hook.sh"
    hook.write_text('module() { :; }\nconda() { :; }\nmkdir() { :; }\n')
    output = tmp_path / "output"
    output.mkdir()
    environ = dict(os.environ, PROJ_DIR=str(REPO_ROOT), ENV_DIR=str(env_dir), OUTDIR=str(output),
        EXPECTED_SHA=SHA, ALCF_PROJECT="test", PBS_JOBID="123.polaris", BASH_ENV=str(hook),
        PATH=str(env_dir / "bin") + os.pathsep + os.environ["PATH"])
    answer = subprocess.run(["bash", str(REPO_ROOT / "experiments/polaris/run_campaign.pbs")],
        capture_output=True, text=True, env=environ, timeout=20)
    assert answer.returncode == preflight_rc, answer.stderr
    log = (output / "job.log").read_text()
    assert ("full campaign called" in log) == (preflight_rc == 0)
    manifest = campaign.read_json(output / "artifact_manifest.json")
    assert manifest["job_exit_code"] == preflight_rc
    for entry in manifest["files"]:
        assert campaign.sha256(output / entry["path"]) == entry["sha256"]
