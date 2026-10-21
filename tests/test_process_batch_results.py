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


def test_optimal_args_give_the_optimal_planner_settings_of_its_own():
    assert pbr.parseOptimalArgs(None) == pbr.parseOptimalArgs("  ") == {}
    over = pbr.parseOptimalArgs("--n-samples 1024 --temperature 0.1 --horizon 8 --cost-weight w_quat=2 "
                                "--no-graph")
    assert over == {"n_samples": 1024, "temperature": 0.1, "horizon": 8, "cost_weights": {"w_quat": 2.0},
                    "graph": False}
    a = argparse.Namespace(rollout_model="M3", eval_sim="M1", plan_on_eval=False, substeps=16,
                           ctrl_time_step=None, horizon=5, time_horizon=None, temperature=7.0, n_samples=256,
                           cost_weights={"w_pos_x": 3.0, "w_quat": 1.0})
    o = pbr.optimalArgs(a, 0.064, 0.32, over)
    assert (o.rollout_model, o.n_samples, o.temperature, o.horizon, o.time_horizon) == ("M1", 1024, 0.1, 8, None)
    assert o.ctrl_time_step == 0.064 and o.cost_weights == {"w_pos_x": 3.0, "w_quat": 2.0}
    assert pbr.optimalArgs(a, 0.064, 0.32, {"time_horizon": 0.5}).horizon is None
    for bad, match in (("--rollout-model M2", "cannot be set"), ("--warm-start hold", "cannot be set"),
                       ("--bogus 3", "unrecognised"), ("--n-samples x", "invalid"),
                       ("--cost-weight nope=1", "unknown weight"), ("--horizon 3 --time-horizon 1", "not both")):
        with pytest.raises(ValueError, match=match):
            pbr.parseOptimalArgs(bad)


def test_the_optimal_planner_defaults_give_way_to_the_flags_given():
    over, text = pbr.optimalOverrides(None)
    assert over == {"convergence_ratio": 2.0, "noise_sigma": 0.38, "temperature": 0.001}
    assert text == pbr.DEFAULT_OPTIMAL_ARGS
    over, text = pbr.optimalOverrides("--temperature 0.1 --n-samples 64")
    assert over == {"convergence_ratio": 2.0, "noise_sigma": 0.38, "temperature": 0.1, "n_samples": 64}
    assert text == pbr.DEFAULT_OPTIMAL_ARGS + " --temperature 0.1 --n-samples 64"


def test_bad_optimal_args_are_refused_before_anything_runs(tmp_path):
    fake_batch(tmp_path, 1)
    with pytest.raises(SystemExit):
        pbr.main([str(tmp_path), "--count", "--optimal-args=--eval-sim M1"])


@pytest.mark.parametrize("recorded, replayed", [(False, "hold"), (True, "shift"), ("none", "none")])
def test_recorded_on_off_warm_start_replays_as_it_ran(recorded, replayed):
    assert pbr.argsFromCliArgs({"warm_start": recorded}).warm_start == replayed


def test_recordings_from_before_the_noise_flags_replay_with_the_old_noise():
    old = pbr.argsFromCliArgs({"warm_start": "none"})
    assert old.resample_noise is False and old.center_noise is False
    new = pbr.argsFromCliArgs({"warm_start": "none", "resample_noise": True, "center_noise": True})
    assert new.resample_noise is True and new.center_noise is True


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
    # The mean's 95% t-interval over the finite values: 2 +/- t(0.975, 2) * 1 / sqrt(3).
    half = 4.302652729911275 / np.sqrt(3)
    assert (s["x_mean_lo"], s["x_mean_hi"]) == (pytest.approx(2 - half), pytest.approx(2 + half))
    assert list(s)[:3] == ["x_mean", "x_mean_lo", "x_mean_hi"]
    one = pbr.stats("y", [5.0])
    assert one["y_mean"] == 5.0 and one["y_mean_lo"] is None and one["y_mean_hi"] is None
    assert pbr.stats("z", [np.nan])["z_mean"] is None


def test_cell_and_episode_metrics_come_back_from_the_saved_steps(tmp_path):
    a, b = {"m": np.array([1.0, 2.0])}, {"m": np.array([3.0, np.nan, 6.0])}
    np.savez(tmp_path / "x.npz", m__aa=a["m"], step__aa=np.arange(2), m__bb=b["m"], step__bb=np.arange(3))
    cell, eps = pbr.metricsFromNpz(tmp_path / "x.npz")
    assert cell == pbr.stats("m", [1.0, 2.0, 3.0, np.nan, 6.0])           # pooled over every step
    assert eps == {"aa": pbr.episodeStats(a), "bb": pbr.episodeStats(b)}
    assert set(eps["bb"]) == {"m_mean", "m_mean_lo", "m_mean_hi"} and eps["bb"]["m_mean"] == 4.5


# -- end to end --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    """A real batch: M3 planning against an M2 world, an M2 oracle, and the first warm-started
    (hold) and saved every second step."""
    import run_episode_batches as batches
    d = tmp_path_factory.mktemp("batch")
    csv_path = d / "b.csv"
    csv_path.write_text(
        "label,rollout_model,eval_sim,plan_on_eval,n_episodes,steps,settle,substeps,horizon,n_samples,"
        "save_steps,stop_on_success,goal_difficulty,timestep,eval_steps_per_rollout_step,"
        "warm_start,save_stride\n"
        "mismatch,M3,M2,false,2,6,0.1,4,4,32,true,false,4,0.002,1,,\n"
        "oracle,M2,M2,true,1,6,0.1,4,4,32,true,false,4,0.002,1,,\n"
        "strided,M3,M2,false,1,6,0.1,4,4,32,true,false,4,0.002,1,hold,2\n")
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
        assert np.isfinite(rec["metrics"]["regret_mean"])
    assert a["optimal_planner"]["rollout_model"] == "M2" and a["optimal_planner"]["scene"] == "env_leap_eval_cube.xml"
    op, zp = a["optimal_planner"], a["zero_gap_planner"]
    assert (op["noise_sigma"], op["temperature"], op["convergence_ratio"]) == (0.38, 0.001, 2.0)
    # The zero-gap planner: the used planner's own settings, on the eval model.
    cli = json.loads((batch / "cell_0000_mismatch.json").read_text())["configs"]["metadata"]["cli_args"]
    assert (zp["rollout_model"], zp["scene"]) == ("M2", "env_leap_eval_cube.xml")
    assert (zp["noise_sigma"], zp["temperature"], zp["n_samples"], zp["convergence_ratio"]) == \
        (cli["noise_sigma"], cli["temperature"], cli["n_samples"], cli.get("convergence_ratio"))
    # The oracle already plans on the eval model, so its plan is the zero-gap plan: no regret.
    assert abs(o["metrics"]["regret_mean"]) < 1e-3 * abs(np.mean(np.load(
        batch / "analysis" / "cell_0001_oracle" / f"{o['episodes'][0]['id']}.npz")["regret_f_zero"]))
    # The oracle's planner and rollout model are the eval model: forward-sim error at MJWarp's noise.
    assert o["metrics"]["fsim_obj_pos_err_mean"] < 1e-4 < a["metrics"]["fsim_obj_pos_err_mean"]
    rows = list(csv.DictReader(open(batch / "final_results.csv")))
    assert [r["name"] for r in rows] == ["cell_0000_mismatch", "cell_0001_oracle", "cell_0002_strided"]
    for col in ("rollout_model", "success_rate", "batch_status", "analysis_status", "kl_used_opt_mean",
                "kl_opt_used_mean", "fsim_obj_pos_err_mean", "eval_vs_recorded_qpos_err_mean",
                "fsim_opt_obj_pos_err_mean", "eval_used_vs_opt_qpos_err_mean", "replay_action_err_mean",
                "regret_mean"):
        assert col in rows[0]
    # Every mean and rate has its 95% interval around it.
    for r in rows:
        for col in ("success_rate", "kl_used_opt_mean", "regret_mean", "fsim_obj_pos_err_mean",
                    "mean_goals_reached", "plan_ms_mean"):
            if r[f"{col}_lo"] == "":                    # a mean over one episode has no interval
                assert col in ("mean_goals_reached", "plan_ms_mean") and r["n_episodes_run"] == "1"
                continue
            lo, v, hi = (float(r[f"{col}{s}"]) for s in ("_lo", "", "_hi"))
            assert lo <= v <= hi, (r["name"], col)
    eps = list(csv.DictReader(open(batch / "final_episodes.csv")))
    assert len(eps) == 4 and all(e["kl_used_opt_mean"] for e in eps)
    assert all(float(e["regret_mean_lo"]) <= float(e["regret_mean"]) <= float(e["regret_mean_hi"]) for e in eps)
    # An analysis from before the intervals gains them from its saved steps on --summarize.
    path = batch / "analysis" / "cell_0000_mismatch.analysis.json"
    old = json.loads(path.read_text())
    stripped = {**old, "metrics": {k: v for k, v in old["metrics"].items() if not k.endswith(("_lo", "_hi"))}}
    path.write_text(json.dumps(stripped))
    assert pbr.main([str(batch), "--summarize"]) == 0
    again = list(csv.DictReader(open(batch / "final_results.csv")))[0]
    assert again["regret_mean_lo"] == rows[0]["regret_mean_lo"] != ""
    path.write_text(json.dumps(old))
    assert a["replay"] == o["replay"] == "independent" and a["n_steps_analyzed"] == 12
    # Rerun: nothing is redone.
    mtime = (batch / "analysis" / "cell_0000_mismatch.analysis.json").stat().st_mtime
    assert pbr.main([str(batch), "--cell", "0"]) == 0
    assert (batch / "analysis" / "cell_0000_mismatch.analysis.json").stat().st_mtime == mtime


@pytest.mark.gpu
@pytest.mark.slow
def test_a_strided_warm_started_cell_replays_each_saved_step_on_its_own(batch):
    from ContactModelStudy.Utils.EpisodeRecorder import EpisodeReplayer
    ep = EpisodeReplayer(batch / "cell_0002_strided.json")[0]
    assert list(ep.step) == [0, 2, 4] and ep.mean_start is not None
    cell = pbr.listCells(batch)[2]
    npz = batch / "analysis" / "cell_0002_strided.analysis.npz"
    steps = batch / "analysis" / "cell_0002_strided" / f"{ep.id}.npz"
    every = pbr.processCell(batch, cell, stride=1)
    kl_every = np.load(npz)[f"kl_used_opt__{ep.id}"]
    second = pbr.processCell(batch, cell, stride=2)
    kl_second, step_second = np.load(npz)[f"kl_used_opt__{ep.id}"], np.load(npz)[f"step__{ep.id}"]
    assert every["replay"] == "independent" and every["warm_start"] == "hold" and every["save_stride"] == 2
    assert (every["n_steps_analyzed"], second["n_steps_analyzed"]) == (3, 2)
    assert np.isfinite(kl_every).all() and list(step_second) == [0, 4]
    # Each step is planned on its own, so skipping one does not change the others.
    assert np.allclose(kl_second, kl_every[[0, 2]], rtol=1e-2, atol=1e-3)
    # The forward simulation is strided too: the saved steps hold only the analysed rows.
    d = np.load(steps)
    assert list(d["row"]) == [0, 2] and list(d["step"]) == [0, 4]
    for key in ("q", "mu_used", "mu_opt", "q_next_rollout_used", "q_next_eval_used", "q_next_rollout_opt",
                "q_next_eval_opt", "q_next_recorded"):
        assert len(d[key]) == 2 and np.isfinite(d[key]).all(), key
    assert d["cov_opt"].shape[1:] == (d["mu_opt"].shape[1],) * 2 and len(d["goal"]) == 2
    # A step count picks the first and last saved rows here, as stride 2 does.
    counted = pbr.processCell(batch, cell, stride=1, n_steps_analysed=2)
    assert (counted["n_steps_analyzed"], counted["n_steps_analysed"], counted["stride"]) == (2, 2, None)
    assert list(np.load(steps)["row"]) == [0, 2]


@pytest.mark.gpu
@pytest.mark.slow
def test_the_optimal_planner_takes_its_own_settings(batch):
    rec = pbr.processCell(batch, pbr.listCells(batch)[0], stride=3, max_episodes=1,
                          optimal_args="--n-samples 64 --temperature 0.5 --cost-weight w_quat=2")
    o = rec["optimal_planner"]
    assert (o["n_samples"], o["temperature"], o["cost_weights"]["w_quat"]) == (64, 0.5, 2.0)
    assert rec["optimal_args"] == pbr.DEFAULT_OPTIMAL_ARGS + " --n-samples 64 --temperature 0.5 --cost-weight w_quat=2"
    assert o["noise_sigma"] == 0.38 and rec["n_steps_analyzed"] == 2
    assert np.isfinite(rec["metrics"]["kl_used_opt_mean"]) and np.isfinite(rec["metrics"]["fsim_opt_qpos_err_mean"])


def test_plan_starts_come_from_the_record_or_are_rebuilt():
    u0 = np.array([1.0, 2.0])
    old = types.SimpleNamespace(step=np.array([0, 3]), u_prev=np.full((2, 2), np.nan), has_plan_starts=False,
                                summary={}, mean_start=None)
    none = argparse.Namespace(warm_start="none", temperature=5.0)
    assert pbr.planStart(old, 1, none, u0, 7) == {"noise_seed": 7, "resample_count": 3, "lam": 5.0,
                                                  "u_prev": None}
    assert pbr.planStart(old, 0, none, u0, 7)["u_prev"] is u0
    hold = argparse.Namespace(warm_start="hold", temperature=5.0)
    assert pbr.planStart(old, 1, hold, u0, 7) is None          # an older warm-started file: no mean
    new = types.SimpleNamespace(step=np.array([0, 3]), u_prev=np.array([[np.nan] * 2, [3.0, 4.0]]),
                                has_plan_starts=True, summary={"noise_seed": 11},
                                noise_index=np.array([0.0, 3.0]), temperature=np.array([5.0, 6.0]),
                                mean_start=np.ones((2, 4, 2)))
    s = pbr.planStart(new, 1, hold, u0, 7)
    assert (s["noise_seed"], s["resample_count"], s["lam"]) == (11, 3, 6.0)
    assert np.array_equal(s["u_prev"], [3.0, 4.0]) and s["U"].shape == (4, 2) and s["U"].dtype == np.float32
    assert "U" not in pbr.planStart(new, 1, none, u0, 7)


def test_unsaved_plan_starts_are_replayed_in_order_and_only_without_a_stride():
    every = types.SimpleNamespace(id="a", n_saved=6, n_steps=6)
    assert pbr.replayMode(every, [{}, {}], False, "hold") == "independent"
    assert pbr.replayMode(every, [None] * 6, True, "hold") == "sequential"
    with pytest.raises(ValueError, match="without --stride or --n-steps-analysed"):
        pbr.replayMode(every, [None, None], False, "hold")
    with pytest.raises(ValueError, match="cannot be replayed"):
        pbr.replayMode(types.SimpleNamespace(id="b", n_saved=3, n_steps=6), [None] * 3, True, "hold")


def test_a_step_count_spreads_the_analysed_rows_evenly_and_overrides_the_stride():
    assert list(pbr.analysedRows(10, 3)) == [0, 3, 6, 9]
    assert list(pbr.analysedRows(10, 3, n_steps_analysed=4)) == [0, 3, 6, 9]
    assert list(pbr.analysedRows(4000, 7, n_steps_analysed=5)) == [0, 1000, 2000, 2999, 3999]
    assert list(pbr.analysedRows(11, 1, n_steps_analysed=3)) == [0, 5, 10]
    assert list(pbr.analysedRows(1000, 1, n_steps_analysed=1)) == [0]
    # An episode with no more saved rows than that has all of them analysed.
    assert list(pbr.analysedRows(3, 2, n_steps_analysed=3)) == [0, 1, 2]
    assert list(pbr.analysedRows(3, 2, n_steps_analysed=500)) == [0, 1, 2]
    # Always exactly that many distinct rows, however close the count is to the episode's length.
    for n_saved in range(2, 60):
        for n in range(1, n_saved):
            rows = pbr.analysedRows(n_saved, 1, n_steps_analysed=n)
            assert len(np.unique(rows)) == n and rows[0] == 0 and rows[-1] == (n_saved - 1 if n > 1 else 0)


def test_episode_metrics_from_the_saved_steps():
    T, nu, idx = 2, 2, np.array([16, 16, 0, 16])
    qn = np.zeros((T, 23)); qn[:, 19] = 1.0
    moved = qn.copy(); moved[:, 16] += 0.002
    d = {"mu_used": np.zeros((T, nu)), "cov_used": np.stack([np.eye(nu)] * T),
         "mu_opt": np.array([[0.0, 0.0], [np.nan, np.nan]]), "cov_opt": np.stack([np.eye(nu)] * T),
         "u_used": np.zeros((T, nu)), "noise_sigma_used": 1.0, "noise_sigma_opt": 1.0, "indices": idx}
    for model, action, q in (("rollout", "used", moved), ("eval", "used", qn), ("rollout", "opt", qn),
                             ("eval", "opt", qn), ("recorded", None, qn)):
        key = f"{model}_{action}" if action else model
        d[f"q_next_{key}"], d[f"q_dot_next_{key}"] = q, np.zeros((T, 22))
    m = pbr.episodeMetrics(d, shrinkage=0.5)
    assert np.isnan(m["regret"]).all()                                  # saved before the regret was
    d["regret_f_used"], d["regret_f_zero"] = np.array([5.0, 3.0]), np.array([4.0, 3.5])
    m = pbr.episodeMetrics(d, shrinkage=0.5)
    assert np.allclose(m["regret"], [1.0, -0.5])
    assert m["kl_used_opt"][0] == pytest.approx(0.0) and np.isnan(m["kl_used_opt"][1])
    assert np.allclose(m["fsim_obj_pos_err"], 0.002) and np.allclose(m["fsim_opt_obj_pos_err"], 0.0)
    assert np.allclose(m["eval_vs_recorded_qpos_err"], 0.0) and np.allclose(m["replay_action_err"], 0.0)
    # Shrinkage pulls each covariance toward its own planner's noise.
    d["noise_sigma_opt"] = 2.0
    assert pbr.episodeMetrics(d, 0.5)["kl_used_opt"][0] > 0 == pbr.episodeMetrics(d, 0.0)["kl_used_opt"][0]


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
