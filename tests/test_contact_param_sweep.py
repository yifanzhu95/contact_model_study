"""experiments/contact_param_sweep.py: the grid, the episodes, the checkpoint, end to end."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types

import numpy as np
import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "experiments"))
_spec = importlib.util.spec_from_file_location("contact_param_sweep",
                                               REPO_ROOT / "experiments" / "contact_param_sweep.py")
cps = importlib.util.module_from_spec(_spec)
sys.modules["contact_param_sweep"] = cps
_spec.loader.exec_module(cps)
pbr = cps.pbr


def rollout(model: str):
    """``(task, sim overrides)`` of a default cube run planning with ``model``."""
    from ContactModelStudy.Drivers import run_episodes as drv
    args = drv.parseArgs(["--no-results", "--no-video", "--rollout-model", model])
    task, _ = drv.buildTasks(args)
    return task, drv.rolloutSimOverrides(args, task)


# -- the grid ------------------------------------------------------------------------------
@pytest.mark.parametrize("model, size", [("M1", 25), ("M2", 25), ("M3", 25), ("M4", 30), ("M6", 125)])
def test_each_model_sweeps_a_full_grid_around_its_nominal(model, size):
    task, ov = rollout(model)
    nominal = cps.nominalParams(model, task, ov)
    grid, dropped = cps.buildGrid(model, nominal, task, ov)
    assert len(grid) == size and dropped == []
    assert set(grid[0]) == set(cps.SWEEP_PARAMS[model])
    assert grid[cps.nominalIndex(grid, nominal)] == pytest.approx(nominal)


def test_the_nominal_is_the_cells_own_model():
    task, ov = rollout("M1")
    assert cps.nominalParams("M1", task, ov)["solref_timeconst"] == pytest.approx(2 * task.timestep)
    task, ov = rollout("M3")
    assert cps.nominalParams("mujoco_soft", task, ov) == cps.sceneSolref(task.getModelPath())
    assert cps.nominalParams("M3", task, ov) == task.contactParams("M3")       # the task's tuning
    assert cps.nominalParams("M4", task, ov) == {"xpbd_iterations": 2, "relaxation": 2.5}


def test_values_and_multipliers_replace_the_defaults():
    task, ov = rollout("M3")
    nominal = cps.nominalParams("M3", task, ov)
    values = cps.parseValueSpecs(["stiffness=0.5,1.0"], "--values")
    mults = cps.parseValueSpecs(["M3:damping=2", "M4:relaxation=1"], "--multipliers")
    per = cps.paramValues("M3", nominal, values, mults)
    assert per["stiffness"] == [0.5, nominal["stiffness"], 1.0]            # the nominal is always in
    assert per["damping"] == pytest.approx([nominal["damping"], 2 * nominal["damping"]])
    # Whole numbers stay whole, and repeats are dropped.
    assert cps.paramValues("M4", {"xpbd_iterations": 2, "relaxation": 2.5},
                           cps.parseValueSpecs(["xpbd_iterations=1.2,2,3"], "--values"))["xpbd_iterations"] == [1, 2, 3]
    for bad in ("stiffness", "stiffness=", "nope=1", "M3:alpha=1", "stiffness=a,b"):
        with pytest.raises(ValueError):
            cps.parseValueSpecs([bad], "--values")


def test_points_the_model_refuses_are_dropped_and_recorded():
    task, ov = rollout("M6")
    nominal = cps.nominalParams("M6", task, ov)
    grid, dropped = cps.buildGrid("M6", nominal, task, ov, values=cps.parseValueSpecs(
        ["gamma=0.5,10", "alpha=0.002", "refinement=2"], "--values"))
    assert grid == [{"alpha": 0.002, "gamma": 10.0, "refinement": 2}]
    assert len(dropped) == 1 and dropped[0]["gamma"] == 0.5 and "gamma" in dropped[0]["reason"]
    task, ov = rollout("M2")
    grid, dropped = cps.buildGrid("M2", cps.nominalParams("M2", task, ov), task, ov,
                                  multipliers=cps.parseValueSpecs(["solref_timeconst=0.1,1"], "--multipliers"))
    assert {d["reason"].split(" s ")[0] for d in dropped} == {"solref_timeconst 0.002"}
    assert all(g["solref_timeconst"] >= 2 * task.timestep for g in grid)


def test_score_blocks_put_u_a_beside_each_block_of_grid_points():
    assert cps.blocksOf(5, 4) == [[0, 1, 2], [3, 4]]
    assert cps.blocksOf(2, 2) == [[0], [1]]


# -- the episodes ------------------------------------------------------------------------------
def test_episodes_are_chosen_by_index_range_or_id():
    eps = [types.SimpleNamespace(id=f"id{k}", summary={"episode": k + 10}) for k in range(6)]
    assert cps.selectEpisodes(eps, None) == list(range(6))
    assert cps.selectEpisodes(eps, "15, 10,12-13") == [0, 2, 3, 5]            # the cell's order
    assert cps.selectEpisodes(eps, "id4,11") == [1, 4]
    assert cps.selectEpisodes(eps, "10-15", max_episodes=2) == [0, 1]
    with pytest.raises(ValueError, match="no episode 3, idx"):
        cps.selectEpisodes(eps, "3,idx")
    with pytest.raises(ValueError, match="backwards"):
        cps.selectEpisodes(eps, "12-10")


def test_the_sweep_key_follows_what_was_computed():
    key = lambda **kw: cps._sweepKey(**{"model": "M3", "grid": [{"stiffness": 1.0}], "episode_ids": ["a"],  # noqa: E731
                                        "rows": [np.arange(3)], "eval_kamino": {}, "score_block": 4, **kw})
    assert key() == key()
    for change in ({"grid": [{"stiffness": 2.0}]}, {"episode_ids": ["b"]}, {"rows": [np.arange(2)]},
                   {"eval_kamino": {"padmm_max_iterations": 100}}, {"score_block": 2}):
        assert key(**change) != key()


def test_m5_cells_are_skipped_and_the_summary_says_so(tmp_path, monkeypatch):
    (tmp_path / "cell_0000_gt.status.json").write_text(json.dumps({"cell": 0, "status": "done"}))
    (tmp_path / "cell_0000_gt.json").write_text("{}")

    def skip(*a, **kw):
        raise cps.CellSkipped("the rollout model is M5")
    monkeypatch.setattr(cps, "processCell", skip)
    assert cps.main([str(tmp_path), "--cell", "0"]) == 0
    assert cps.sweepDone(tmp_path, "cell_0000_gt")
    cps.summarize(tmp_path)
    row, = csv.DictReader(open(tmp_path / "sweep_results.csv"))
    assert row["sweep_status"] == "skipped"


def test_bad_flags_are_refused_before_anything_runs(tmp_path):
    for argv in (["--values", "nope=1"], ["--score-block", "1"], ["--stride", "0"]):
        with pytest.raises(SystemExit):
            cps.main([str(tmp_path), *argv])


# -- end to end --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    """A real batch: M3 planning against an M2 world, two episodes of six saved steps."""
    import run_episode_batches as batches
    d = tmp_path_factory.mktemp("batch")
    csv_path = d / "b.csv"
    csv_path.write_text(
        "label,rollout_model,eval_sim,plan_on_eval,n_episodes,steps,settle,substeps,horizon,n_samples,"
        "save_steps,stop_on_success,goal_difficulty,timestep,eval_steps_per_rollout_step\n"
        "mismatch,M3,M2,false,2,6,0.1,4,4,32,true,false,4,0.002,1\n")
    out = d / "out"
    for i, row in enumerate(batches.readRows(csv_path)):
        assert batches.runCell(i, row, out, overwrite=False)
    return out


SMALL_GRID = ["--values", "stiffness=0.375,0.75,1.5", "--values", "damping=3e-4"]


@pytest.mark.gpu
@pytest.mark.slow
def test_sweep_a_cell(batch):
    assert cps.main([str(batch), "--cell", "0", "--episodes", "1", *SMALL_GRID, "--score-block", "3"]) == 0
    rec = json.loads((batch / "sweep" / "cell_0000_mismatch.sweep.json").read_text())
    assert (rec["rollout_model"], rec["eval_sim"], rec["n_grid"], rec["n_blocks"]) == ("M3", "M2", 3, 2)
    assert rec["episodes"] == [1] and rec["n_steps_analyzed"] == 6
    assert [p["p_stiffness"] for p in rec["points"]] == [0.375, 0.75, 1.5]
    assert rec["points"][rec["nominal_index"]]["p_stiffness"] == 0.75
    d = np.load(batch / "sweep" / "cell_0000_mismatch.sweep.npz")
    assert d["regret"].shape == (6, 3) and np.isfinite(d["regret"]).all()
    assert set(d["episode"]) == {1} and d["U_used"].shape[:2] == (6, 3)
    assert np.array_equal(d["block"], [0, 0, 1])
    # One evaluation noise per step: U_A scores the same in both blocks, up to the eval model's noise.
    assert np.allclose(d["f_zero"][:, 0], d["f_zero"][:, 1], rtol=1e-4)
    assert not (batch / "sweep" / "cell_0000_mismatch.partial.npz").exists()
    # The nominal regret is process_batch_results' regret at the same steps.
    assert pbr.main([str(batch), "--cell", "0"]) == 0
    ep_id = str(d["episode_id"][0])
    a = np.load(batch / "analysis" / "cell_0000_mismatch" / f"{ep_id}.npz")
    assert np.allclose(d["regret"][:, rec["nominal_index"]], a["regret_f_used"] - a["regret_f_zero"],
                       rtol=1e-3, atol=1e-3 * np.abs(a["regret_f_zero"]).max())
    cps.summarize(batch)
    rows = list(csv.DictReader(open(batch / "sweep_results.csv")))
    assert len(rows) == 3 and [r["is_nominal"] for r in rows] == ["False", "True", "False"]
    for r in rows:
        assert float(r["regret_mean_lo"]) <= float(r["regret_mean"]) <= float(r["regret_mean_hi"])
    # Done: a rerun is skipped.
    mtime = (batch / "sweep" / "cell_0000_mismatch.sweep.json").stat().st_mtime
    assert cps.main([str(batch), "--cell", "0", "--episodes", "1", *SMALL_GRID, "--score-block", "3"]) == 0
    assert (batch / "sweep" / "cell_0000_mismatch.sweep.json").stat().st_mtime == mtime


@pytest.mark.gpu
@pytest.mark.slow
def test_a_cell_cut_off_resumes_from_its_checkpoint(batch, monkeypatch):
    argv = [str(batch), "--cell", "0", "--episodes", "0", *SMALL_GRID, "--overwrite"]
    built = []
    real = cps._usedPlanner

    def cut_off_after_one(args, task, ov, theta):
        if built:
            raise KeyboardInterrupt
        built.append(theta)
        return real(args, task, ov, theta)
    monkeypatch.setattr(cps, "_usedPlanner", cut_off_after_one)
    with pytest.raises(KeyboardInterrupt):
        cps.main(argv)
    ck = dict(np.load(batch / "sweep" / "cell_0000_mismatch.partial.npz"))
    assert ck["zero_done"] and ck["plans_done"].tolist() == [True, False, False] and not ck["blocks_done"].any()

    resumed = []
    monkeypatch.setattr(cps, "_usedPlanner", lambda *a: resumed.append(a[3]) or real(*a))
    assert cps.main(argv[:-1]) == 0                                          # without --overwrite
    assert resumed == [{"stiffness": 0.75, "damping": 3e-4}, {"stiffness": 1.5, "damping": 3e-4}]
    d = np.load(batch / "sweep" / "cell_0000_mismatch.sweep.npz")
    assert set(d["episode"]) == {0} and np.isfinite(d["regret"]).all()
