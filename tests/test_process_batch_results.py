"""experiments/process_batch_results.py: cells, checks, metrics, goals, end to end."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import types

import numpy as np
import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "experiments"))
_spec = importlib.util.spec_from_file_location("process_batch_results",
                                               REPO_ROOT / "experiments" / "process_batch_results.py")
pbr = importlib.util.module_from_spec(_spec)
sys.modules["process_batch_results"] = pbr
_spec.loader.exec_module(pbr)


def fake_batch(tmp_path, n=3):
    for i in range(n):
        (tmp_path / f"cell_{i:04d}_c{i}.status.json").write_text(json.dumps(
            {"cell": i, "name": f"cell_{i:04d}_c{i}", "status": "done", "row": {"label": f"c{i}"}}))
    return tmp_path


# -- building blocks ---------------------------------------------------------------------
def test_cells_are_listed_in_cell_order(tmp_path, capsys):
    fake_batch(tmp_path, 3)
    (tmp_path / "cell_0010_z.status.json").write_text(json.dumps({"cell": 10}))
    assert [c["name"] for c in pbr.listCells(tmp_path)] == \
        ["cell_0000_c0", "cell_0001_c1", "cell_0002_c2", "cell_0010_z"]
    assert pbr.main([str(tmp_path), "--count"]) == 0 and capsys.readouterr().out.strip() == "4"
    with pytest.raises(SystemExit):
        pbr.main([str(tmp_path / "missing"), "--count"])
    with pytest.raises(SystemExit):
        pbr.main([str(tmp_path), "--cell", "9"])


def test_missing_results_fail_the_cell_with_its_reason(tmp_path):
    fake_batch(tmp_path, 1)
    assert pbr.main([str(tmp_path), "--cell", "0"]) == 1
    status = json.loads((tmp_path / "analysis" / "cell_0000_c0.analysis_status.json").read_text())
    assert status["status"] == "failed" and "no results file" in status["error"]


def test_cell_checks():
    ep = types.SimpleNamespace(has_steps=True)
    pbr.checkCell([ep], argparse.Namespace(eval_sim="M2"))
    with pytest.raises(ValueError, match="not vectorized"):
        pbr.checkCell([ep], argparse.Namespace(eval_sim="mujoco"))
    with pytest.raises(ValueError, match="save_steps"):
        pbr.checkCell([types.SimpleNamespace(has_steps=False)], argparse.Namespace(eval_sim="M3"))
    with pytest.raises(ValueError, match="no episodes"):
        pbr.checkCell([], argparse.Namespace(eval_sim="M3"))


def test_optimal_args_plan_on_the_eval_model_over_the_same_durations():
    a = argparse.Namespace(rollout_model="M3", eval_sim="M1", plan_on_eval=False, substeps=16,
                           ctrl_time_step=None, horizon=5, time_horizon=None, temperature=7.0)
    o = pbr.optimalArgs(a, 0.064, 0.32)
    assert (o.plan_on_eval, o.rollout_model, o.temperature) == (True, "M1", 7.0)
    assert (o.substeps, o.ctrl_time_step, o.horizon, o.time_horizon) == (None, 0.064, None, 0.32)
    assert a.rollout_model == "M3"                                     # the original is untouched


def test_metrics():
    q = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    half = np.array([[np.cos(0.25), np.sin(0.25), 0, 0], [-1.0, 0, 0, 0]])   # 0.5 rad, and a sign flip
    assert np.allclose(pbr.quatAngle(q, half), [0.5, 0.0])
    idx = np.array([16, 16, 0, 16])
    qa = np.zeros((2, 23)); qa[:, 19] = 1.0
    qb = qa.copy(); qb[0, 16] += 0.003; qb[1, :16] += 0.1
    e = pbr.stateErrors(qa, np.zeros((2, 22)), qb, np.zeros((2, 22)), idx)
    assert np.allclose(e["obj_pos_err"], [0.003, 0]) and np.allclose(e["hand_rms_err"], [0, 0.1])
    assert np.allclose(pbr.goalIndexPerStep([(0, None), (3, None), (7, None)], 9), [0, 0, 0, 1, 1, 1, 1, 2, 2])
    s = pbr.stats("x", [1.0, 2.0, 3.0, np.nan])
    assert s["x_mean"] == 2.0 and s["x_median"] == 2.0 and s["x_n_nonfinite"] == 1


# -- end to end --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    """A real two-cell batch: M3 planning against an M2 world, and an M2 oracle."""
    import run_episode_batches as batches
    d = tmp_path_factory.mktemp("batch")
    csv_path = d / "b.csv"
    csv_path.write_text(
        "label,rollout_model,eval_sim,plan_on_eval,n_episodes,steps,settle,substeps,horizon,n_samples,"
        "save_steps,stop_on_success,goal_difficulty,timestep,eval_steps_per_rollout_step\n"
        "mismatch,M3,M2,false,2,6,0.1,4,4,32,true,false,4,0.002,1\n"
        "oracle,M2,M2,true,1,6,0.1,4,4,32,true,false,4,0.002,1\n")
    out = d / "out"
    for i, row in enumerate(batches.readRows(csv_path)):
        assert batches.runCell(i, row, out, overwrite=False)
    return out


@pytest.mark.gpu
@pytest.mark.slow
def test_process_a_batch(batch):
    assert pbr.main([str(batch)]) == 0
    a = json.loads((batch / "analysis" / "cell_0000_mismatch.analysis.json").read_text())
    o = json.loads((batch / "analysis" / "cell_0001_oracle.analysis.json").read_text())
    for rec in (a, o):
        assert np.isfinite(rec["metrics"]["kl_used_opt_mean"]) and np.isfinite(rec["metrics"]["fsim_qpos_err_mean"])
    assert a["optimal_planner"]["rollout_model"] == "M2" and a["optimal_planner"]["scene"] == "env_leap_eval_cube.xml"
    # The oracle's planner and rollout model are the eval model: forward-sim error at MJWarp's noise.
    assert o["metrics"]["fsim_obj_pos_err_mean"] < 1e-4 < a["metrics"]["fsim_obj_pos_err_mean"]
    rows = list(csv.DictReader(open(batch / "final_results.csv")))
    assert [r["name"] for r in rows] == ["cell_0000_mismatch", "cell_0001_oracle"]
    for col in ("rollout_model", "success_rate", "batch_status", "analysis_status", "kl_used_opt_mean",
                "kl_opt_used_mean", "fsim_obj_pos_err_mean", "eval_vs_recorded_qpos_err_mean"):
        assert col in rows[0]
    eps = list(csv.DictReader(open(batch / "final_episodes.csv")))
    assert len(eps) == 3 and all(e["kl_used_opt_mean"] for e in eps)
    # Rerun: nothing is redone.
    mtime = (batch / "analysis" / "cell_0000_mismatch.analysis.json").stat().st_mtime
    assert pbr.main([str(batch), "--cell", "0"]) == 0
    assert (batch / "analysis" / "cell_0000_mismatch.analysis.json").stat().st_mtime == mtime


@pytest.mark.gpu
@pytest.mark.slow
def test_goals_of_older_results_are_reconstructed(batch):
    from ContactModelStudy.Utils.EpisodeRecorder import EpisodeReplayer
    rep = EpisodeReplayer(batch / "cell_0000_mismatch.json")
    args = pbr.argsFromCliArgs(rep.configs["metadata"]["cli_args"])
    recorded = pbr.episodeGoals(rep, args)
    for ep in rep:                                       # as an older file: labels only
        for key in ("goal_values", "goal_steps"):
            ep.summary.pop(key)
    rebuilt = pbr.episodeGoals(rep, args)
    assert [[s for s, _ in g] for g in rebuilt] == [[s for s, _ in g] for g in recorded]
    assert all(np.allclose(a, b) for ga, gb in zip(rebuilt, recorded) for (_, a), (_, b) in zip(ga, gb))
    rep[0].summary["goals"] = ["?" for _ in rep[0].summary["goals"]]
    with pytest.raises(ValueError, match="cannot reconstruct"):
        pbr.episodeGoals(rep, args)
