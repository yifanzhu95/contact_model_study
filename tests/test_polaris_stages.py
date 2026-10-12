"""Offline checks of stage gates, device scheduling and incomplete results.

No CUDA kernel, simulator, PBS submission or real nvidia-smi is executed here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import queue
from types import SimpleNamespace

import pytest

from conftest import REPO_ROOT
from ContactModelStudy.Drivers.EpisodePool import episodeSeeds

spec = importlib.util.spec_from_file_location("polaris_stages", REPO_ROOT / "experiments/polaris/run_stage.py")
stages = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stages
spec.loader.exec_module(stages)
SHA = "a" * 40


def complete_result(cell):
    devices = cell.gpus.split(",")
    episodes = [{"summary": {"episode": i, "finish_reason": "steps", "n_steps": 100,
                  "goal_seed": episodeSeeds(int(cell.row["seed"]), i)[0],
                  "planner_validation": {"status": "pass", "planner_devices": devices}}}
                for i in range(int(cell.row["n_episodes"]))]
    return {"n_episodes": len(episodes), "episodes": episodes,
            "validation": {"status": "pass", "scaling_eligible": True},
            "configs": {"metadata": {"provenance": {"source": {"commit": SHA, "dirty": False}}}}}


def test_stage_budgets_and_scientific_settings():
    g1, g2, g3 = [stages.cells_for(s, 64) for s in ("g1", "g2", "g3")]
    assert len(g1) == 12 and len(g2) == 4 and len(g3) == 27
    assert sum(int(c.row["steps"]) * int(c.row["n_episodes"]) for c in g1) == 240
    assert sum(int(c.row["steps"]) * int(c.row["n_episodes"]) for c in g3) == 2700
    assert all(c.row["n_samples"] == "64" for c in g2 + g3)
    assert all(c.row["n_iterations"] == "1" and c.row["graph"] == "true" for c in g1 + g2 + g3)
    assert len({c.name for c in g3}) == 27
    assert [(c.gpus, c.workers) for c in g2] == [("0", 4), ("0,1,2,3", 4), ("0,1,2,3", 8), ("0,1,2,3", 16)]


def test_previous_gate_accepts_only_validated_n_and_same_version(tmp_path):
    previous = tmp_path / "g1.json"
    stages.write_json(previous, {"stage": "g1", "expected_source_sha": SHA,
                                "passed": False, "source_unchanged": True, "validated_samples": [32]})
    stages.require_previous("g2", previous, SHA, 32)
    with pytest.raises(ValueError, match="all three G1 seeds"):
        stages.require_previous("g2", previous, SHA, 64)
    with pytest.raises(ValueError, match="source SHA"):
        stages.require_previous("g2", previous, "b" * 40, 32)
    with pytest.raises(ValueError, match="passing G2"):
        stages.require_previous("g3", previous, SHA, 32)


@pytest.mark.parametrize("fault", ["partial", "interrupted", "missing_diagnostics", "unused_gpu", "bad_seed", "dirty"])
def test_successful_process_cannot_hide_invalid_or_incomplete_pool(fault):
    cell = stages.cells_for("g2", 32)[1]
    document = complete_result(cell)
    stats = {"complete": True, "pool": {"gpus": ["0", "1", "2", "3"], "workers": 4, "planners_per_gpu": 1}}
    assert stages.assess_result(document, cell, SHA, stats) == []
    if fault == "partial":
        document["episodes"].pop()
    elif fault == "interrupted":
        document["episodes"][0]["summary"]["finish_reason"] = "interrupted"
    elif fault == "missing_diagnostics":
        document["validation"] = {}
    elif fault == "unused_gpu":
        for e in document["episodes"]:
            e["summary"]["planner_validation"]["planner_devices"] = ["0"]
    elif fault == "bad_seed":
        document["episodes"][0]["summary"]["goal_seed"] += 1
    else:
        document["configs"]["metadata"]["provenance"]["source"]["dirty"] = True
    assert stages.assess_result(document, cell, SHA, stats)


def test_gpu_telemetry_is_pinned_and_preserves_uuid(tmp_path, monkeypatch):
    commands = []
    def fake_run(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0, stderr="", stdout="2, GPU-uuid, 80, 4096, 40960, 55, 120\n")
    monkeypatch.setattr(stages.subprocess, "run", fake_run)
    monitor = stages.Telemetry(tmp_path / "gpu.jsonl", "2")
    monitor.sample()
    result = monitor.finish()
    assert commands[0][1:3] == ["-i", "2"]
    assert result["per_gpu"]["2"]["sampled_peak_memory_mib"] == 4096
    assert json.loads(monitor.path.read_text())["uuid"] == "GPU-uuid"


def test_resume_rechecks_result_instead_of_trusting_status(tmp_path):
    cell = stages.cells_for("g1", None)[0]
    attempt = tmp_path / "attempt_0001"
    stages.write_json(attempt / "result.json", complete_result(cell))
    stages.write_json(tmp_path / "cell.status.json", {
        "passed": True, "expected_source_sha": SHA, "spec_sha256": stages.fingerprint(cell.spec()),
        "attempt": str(attempt)})
    assert stages.cached_result(tmp_path, cell, SHA)
    document = complete_result(cell)
    document["validation"] = {}
    stages.write_json(attempt / "result.json", document)
    assert stages.cached_result(tmp_path, cell, SHA) is None


@pytest.mark.parametrize("partial", [False, True])
def test_real_child_completion_and_failure_status_without_gpu(tmp_path, monkeypatch, partial):
    cell = stages.cells_for("g3", 32)[2]
    document = complete_result(cell)
    if partial:
        document["episodes"] = []
    def fake_command(cell, attempt, batches):
        code = ("import json, os; from pathlib import Path; "
                f"assert os.environ['CUDA_VISIBLE_DEVICES'] == {cell.gpus!r}; "
                f"Path({str(attempt / 'result.json')!r}).write_text({json.dumps(document)!r}); "
                "print('synthetic child completed')")
        return [sys.executable, "-c", code]
    class FakeTelemetry:
        def __init__(self, path, gpus): self.gpus = gpus
        def start(self): pass
        def finish(self): return {"per_gpu": {self.gpus: {"samples": 1}}, "errors": []}
    monkeypatch.setattr(stages, "command", fake_command)
    monkeypatch.setattr(stages, "Telemetry", FakeTelemetry)
    active = stages.RunningCell(cell, tmp_path / cell.name, None, SHA)
    deadline = time.monotonic() + 5
    while not active.ready() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert active.ready()
    result = active.finish()
    assert result["returncode"] == 0 and result["execution_status"] == "completed"
    assert result["passed"] is (not partial)
    assert (active.attempt / "driver.log").read_text().strip() == "synthetic child completed"


def test_child_cancellation_preserves_result_but_cannot_pass(tmp_path, monkeypatch):
    cell = stages.cells_for("g1", None)[0]
    document = complete_result(cell)
    def fake_command(cell, attempt, batches):
        code = ("from pathlib import Path; import time; "
                f"Path({str(attempt / 'result.json')!r}).write_text({json.dumps(document)!r}); "
                "time.sleep(60)")
        return [sys.executable, "-c", code]
    class FakeTelemetry:
        def __init__(self, path, gpus): pass
        def start(self): pass
        def finish(self): return {"per_gpu": {"0": {"samples": 1}}, "errors": []}
    monkeypatch.setattr(stages, "command", fake_command)
    monkeypatch.setattr(stages, "Telemetry", FakeTelemetry)
    active = stages.RunningCell(cell, tmp_path / cell.name, None, SHA)
    deadline = time.monotonic() + 5
    while not (active.attempt / "result.json").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    active.cancel()
    active.proc.wait(timeout=5)
    result = active.finish()
    assert not result["passed"] and result["execution_status"] == "failed"
    assert "cancelled_or_budget_exhausted" in result["blocking_reasons"]
    assert (active.attempt / "result.json").is_file()


def test_pool_close_kills_children_that_ignore_sigterm(monkeypatch):
    from ContactModelStudy.Drivers.EpisodePool import EpisodePool
    pool = EpisodePool.__new__(EpisodePool)
    pool._started, pool._closed = True, False
    pool.problems, pool._finished = [], []
    pool._job_queue = pool._requests = queue.Queue()
    pool._events = queue.Queue()
    class Child:
        alive, joined = True, False
        def is_alive(self): return self.alive
        def kill(self): self.alive = False
        def join(self, timeout): self.joined = True
    child = Child()
    pool._workers, pool._planners = [child], []
    monkeypatch.setattr(pool, "_drainUntil", lambda processes, deadline: None)
    assert pool.close() == []
    assert child.joined and not child.alive
    assert "forced_process_shutdown" in pool.problems


def stage_environment(tmp_path, monkeypatch, stage):
    out = tmp_path / stage
    gpus = "0" if stage == "g1" else "0,1,2,3"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpus)
    monkeypatch.setattr(stages, "git_provenance", lambda root: {"commit": SHA, "dirty": False})
    stages.write_json(out / "preflight.json", {"passed": True, "expected_source_sha": SHA, "requested_gpus": gpus})
    previous = tmp_path / "previous.json"
    document = ({"schema": "contact_model_study.polaris_smoke_verdict.v1"} if stage == "g1"
                else {"stage": "g2", "n_samples": 32})
    stages.write_json(previous, {**document, "passed": True, "expected_source_sha": SHA})
    return out, previous


def test_calibration_stops_before_escalating_failed_level(tmp_path, monkeypatch):
    out, previous = stage_environment(tmp_path, monkeypatch, "g1")
    launched = []
    class FakeCell:
        def __init__(self, cell, *args):
            self.cell = cell
            launched.append(cell.name)
        def ready(self): return True
        def finish(self): return {"passed": self.cell.row["n_samples"] == "32", "wall_s": 1.0}
    monkeypatch.setattr(stages, "RunningCell", FakeCell)
    monkeypatch.setattr(stages.time, "sleep", lambda seconds: None)
    assert stages.main(["g1", "--outdir", str(out), "--previous", str(previous), "--expected-sha", SHA]) == 2
    result = stages.read_json(out / "stage_verdict.json")
    assert result["validated_samples"] == [32]
    assert len(launched) == 4 and len(result["pending_cells"]) == 8


def test_pilot_uses_at_most_one_cell_per_gpu_and_all_four(tmp_path, monkeypatch):
    out, previous = stage_environment(tmp_path, monkeypatch, "g3")
    active, used, peaks = set(), set(), []
    class FakeCell:
        def __init__(self, cell, *args):
            self.cell = cell
            assert cell.gpus not in active
            active.add(cell.gpus)
            used.add(cell.gpus)
            peaks.append(len(active))
        def ready(self): return True
        def finish(self):
            active.remove(self.cell.gpus)
            return {"passed": True, "wall_s": 1.0}
    monkeypatch.setattr(stages, "RunningCell", FakeCell)
    monkeypatch.setattr(stages.time, "sleep", lambda seconds: None)
    assert stages.main(["g3", "--n-samples", "32", "--outdir", str(out),
                        "--previous", str(previous), "--expected-sha", SHA]) == 0
    result = stages.read_json(out / "stage_verdict.json")
    assert result["passed"] and len(result["cells"]) == 27
    assert max(peaks) == 4 and used == {"0", "1", "2", "3"}
