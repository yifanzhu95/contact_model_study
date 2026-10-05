"""experiments/run_bayes_opt.py: search space, seeds, validation, scoring, the optimizer, resuming."""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "experiments"))
spec_ = importlib.util.spec_from_file_location("run_bayes_opt", REPO_ROOT / "experiments" / "run_bayes_opt.py")
bo = importlib.util.module_from_spec(spec_)
sys.modules["run_bayes_opt"] = bo           # its dataclasses look themselves up here
spec_.loader.exec_module(bo)

CUBE_W = {"w_quat": 23.3929, "w_contact": 2.48369, "w_joint": 20.0, "w_velo": 0.0}


def write(tmp_path, text, name="bo.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def cell(tmp_path, **row):
    row = {"n_episodes": "2", **row}
    return bo.parseCell(row, tmp_path, tmp_path / "cell")


# -- the search space ------------------------------------------------------------------
def test_weight_specs():
    dims = bo.parseWeightSpecs("w_quat w_contact:0.1:100", CUBE_W)
    assert [(d.name, d.lo, d.hi) for d in dims] == [("w_quat", 23.3929 / 4, 23.3929 * 4),
                                                   ("w_contact", 0.1, 100.0)]
    assert bo.parseWeightSpecs("none", CUBE_W) == []
    for bad, msg in (("w_nope", "not a cost weight"), ("w_quat w_quat", "twice"),
                     ("w_velo", "no x4 bracket"), ("w_quat:5:1", "lo < hi"), ("w_quat:1", "bad entry")):
        with pytest.raises(ValueError, match=msg):
            bo.parseWeightSpecs(bad, CUBE_W)


def test_default_space_contains_the_objects_own_weights():
    duck = {"w_quat": 23.0, "w_pos_x": 0.01, "w_pos_y": 100.0, "w_pos_z": 0.36, "w_contact": 100.0,
            "w_joint": 6.2, "w_fallen": 93.6, "w_quat_term": 145.8, "w_pos_term": 500.0}
    dims = {d.name: d for d in bo.parseWeightSpecs(None, duck)}
    assert set(dims) == set(bo.DEFAULT_OPT_WEIGHTS)
    assert all(dims[n].lo <= v <= dims[n].hi for n, v in duck.items())
    assert (dims["w_pos_x"].lo, dims["w_pos_x"].hi) == (0.01, 50.0)      # widened, not moved


def test_ranges():
    assert bo.parseRange("0.01, 25", "t") == (0.01, 25.0)
    for bad in ("1", "5 1", "0 1", "a 2"):
        with pytest.raises(ValueError):
            bo.parseRange(bad, "t")


def test_cell_dims_and_settings(tmp_path):
    s = cell(tmp_path, rollout_models="M2 M3", opt_weights="w_quat", temperature_range="1 100",
             per_model_temperature="true", noise_sigma_range="0.01 0.3")
    assert s.names == ["w_quat", "temperature_M2", "temperature_M3", "noise_sigma"]
    x = [10.0, 2.0, 30.0, 0.05]
    assert s.modelSettings(x, "M3") == ({"temperature": 30.0, "noise_sigma": 0.05}, {"w_quat": 10.0})
    shared = cell(tmp_path, rollout_models="M2 M3", opt_weights="none", temperature_range="1 100")
    assert shared.names == ["temperature"]
    assert shared.modelSettings([7.0], "M2") == ({"temperature": 7.0}, {})


@pytest.mark.parametrize("row, message", [
    ({"opt_weights": "none"}, "search space is empty"),
    ({"temperature": "5", "temperature_range": "1 10"}, "pin it or search it"),
    ({"noise_sigma": "0.1", "noise_sigma_range": "0.01 1"}, "pin it or search it"),
    ({"w_quat": "5", "opt_weights": "w_quat"}, "pinned"),
    ({"per_model_temperature": "true"}, "needs a temperature_range"),
    ({"rollout_models": "M2 M9"}, "M9"),
    ({"rollout_models": "M2 M2"}, "twice"),
    ({"model_agg": "median"}, "model_agg"),
    ({"acq_func": "UCB"}, "acq_func"),
    ({"n_calls": "0"}, "n_calls"),
])
def test_bad_rows(tmp_path, row, message):
    with pytest.raises(ValueError, match=message):
        cell(tmp_path, **row)


def test_bad_columns(tmp_path):
    for header in ("label,rollout_model", "label,video", "label,n_calls,whatever"):
        with pytest.raises(SystemExit):
            bo.main([str(write(tmp_path, header + "\nx,1,2\n")), "--check"])


def test_example_csv_is_valid(tmp_path):
    assert bo.main([str(REPO_ROOT / "experiments" / "example_bayes_opt.csv"), "--check",
                    "--outdir", str(tmp_path)]) == 0


# -- seeds -------------------------------------------------------------------------------
def test_defaults_seed_is_the_objects_weights_and_pinned_knobs(tmp_path):
    s = cell(tmp_path, opt_weights="w_quat w_joint", noise_sigma_range="0.01 0.5")
    assert [x.source for x in s.seeds] == ["defaults"]
    assert s.params(s.seeds[0].x) == {"w_quat": CUBE_W["w_quat"], "w_joint": 20.0,
                                      "noise_sigma": s.pinned_noise_sigma}
    assert s.n_random == s.n_initial_points - 1
    off = cell(tmp_path, opt_weights="w_quat", seed_defaults="false")
    assert off.seeds == [] and off.n_random == off.n_initial_points


def test_seed_points_file(tmp_path):
    (tmp_path / "seeds.csv").write_text("w_quat,temperature\n30,\n40,5\n")
    s = cell(tmp_path, opt_weights="w_quat w_joint", temperature_range="1 100", seed_points="seeds.csv")
    assert [x.source for x in s.seeds] == ["defaults", "seed_points:0", "seed_points:1"]
    p0, p1 = s.params(s.seeds[1].x), s.params(s.seeds[2].x)
    assert p0 == {"w_quat": 30.0, "w_joint": 20.0, "temperature": s.pinned_temperature}   # filled in
    assert p1["temperature"] == 5.0
    (tmp_path / "out.csv").write_text("w_quat\n1000\n")
    with pytest.raises(ValueError, match="outside the search range"):
        cell(tmp_path, opt_weights="w_quat", seed_points="out.csv")
    (tmp_path / "pinned.csv").write_text("w_quat,w_joint\n30,3\n")
    with pytest.raises(ValueError, match="pins w_joint"):
        cell(tmp_path, opt_weights="w_quat", seed_points="pinned.csv")
    (tmp_path / "same.csv").write_text(f"w_quat,w_joint\n30,{20.0}\n")
    assert len(cell(tmp_path, opt_weights="w_quat", seed_points="same.csv").seeds) == 2
    with pytest.raises(ValueError, match="no such file"):
        cell(tmp_path, opt_weights="w_quat", seed_points="missing.csv")


def test_seed_from_an_earlier_run_takes_its_best(tmp_path):
    old = tmp_path / "old_cell"
    for i, (J, wq) in enumerate([(0.3, 10.0), (-0.5, 20.0), (0.1, 30.0), (-0.2, 500.0)]):
        (old / bo.trialName(i)).mkdir(parents=True)
        (old / bo.trialName(i) / "trial.json").write_text(json.dumps(
            {"trial": i, "status": "done", "objective": J, "params": {"w_quat": wq, "w_gone": 1.0}}))
    s = cell(tmp_path, opt_weights="w_quat", seed_defaults="false", seed_from="old_cell", seed_top_k="2")
    assert [x.source for x in s.seeds] == ["seed_from:1", "seed_from:3"]
    hi = s.dims[0].hi
    assert [x.x[0] for x in s.seeds] == [20.0, hi]                       # clipped into this box


def test_n_calls_must_cover_the_seeds(tmp_path):
    (tmp_path / "seeds.csv").write_text("w_quat\n30\n40\n")
    with pytest.raises(ValueError, match="less than"):
        cell(tmp_path, opt_weights="w_quat", seed_points="seeds.csv", n_calls="2")


# -- scoring -----------------------------------------------------------------------------
def test_scoring_and_folding(tmp_path):
    s = cell(tmp_path, opt_weights="w_quat")
    thr = s.thresholds
    at_threshold = {k: v for k, v in thr.items()}                       # sums to len(thr)
    eps = [{"finish_reason": "timeout", "success": True, "goal_errors_end": {k: 0 for k in thr},
            "steps_to_success": 10, "goals_reached": 1},
           {"finish_reason": "timeout", "success": False, "goal_errors_end": at_threshold,
            "steps_to_success": None, "goals_reached": 0},
           {"finish_reason": "error", "success": False}]
    m = bo.scoreModel(eps, s)
    err = (0 + len(thr) / s.err_clip + 1.0) / 3
    assert m["success_rate"] == pytest.approx(1 / 3) and m["n_errors"] == 1
    assert m["mean_norm_goal_err"] == pytest.approx(err)
    assert m["objective"] == pytest.approx(-1 / 3 + 0.1 * err)
    assert bo.normalizedGoalError({"pos": 1e9}, thr, 250) == 1.0
    per = {"M2": {"objective": -0.5}, "M3": {"objective": 0.1}}
    assert bo.foldModels(per, "mean") == pytest.approx(-0.2) and bo.foldModels(per, "worst") == 0.1


# -- the optimizer -----------------------------------------------------------------------
def test_constant_liar_proposes_distinct_points(tmp_path):
    s = cell(tmp_path, opt_weights="w_quat w_joint", noise_sigma_range="0.01 0.5")
    opt = bo.makeOptimizer(s)
    rng = np.random.default_rng(0)
    for _ in range(4):
        x = bo.randomPoint(opt)
        opt.tell(x, float(rng.normal()))
    pending = []
    for _ in range(3):
        pending.append(bo.askNext(opt, pending))
    assert len({tuple(np.round(p, 8)) for p in pending}) == 3
    assert all(d.lo <= v <= d.hi for p in pending for d, v in zip(s.dims, p))
    assert len(opt.yi) == 4                          # the lies never reach the real optimizer


def test_resume_refuses_a_changed_space(tmp_path):
    s = cell(tmp_path, opt_weights="w_quat:1:50")
    d = tmp_path / "c"
    d.mkdir()
    done = [{"trial": 0, "x": [40.0], "objective": 0.1, "source": "random"}]
    bo.writeState(s, d, done)
    bo.checkResume(s, d, done)                                         # same space: fine
    with pytest.raises(ValueError, match="searched over"):
        bo.checkResume(cell(tmp_path, opt_weights="w_joint:1:50"), d, done)
    with pytest.raises(ValueError, match="strands"):
        bo.checkResume(cell(tmp_path, opt_weights="w_quat:1:30"), d, done)


# -- end to end ----------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
def test_run_cell_records_resumes_and_summarizes(tmp_path):
    from ContactModelStudy.Utils.EpisodeRecorder import EpisodeReplayer
    p = write(tmp_path, "label,rollout_models,n_episodes,steps,settle,substeps,horizon,n_samples,"
                        "goal_difficulty,opt_weights,temperature_range,n_calls,n_initial_points,"
                        "trials_in_flight,save_steps,stop_on_success\n"
                        "t,M2 M3,1,4,0.1,4,4,32,4,w_quat,1 100,3,2,2,false,false\n")
    out = tmp_path / "out"
    assert bo.main([str(p), "--cell", "0", "--outdir", str(out), "--workers", "2"]) == 0
    cdir = out / "cell_0000_t"
    trials = [json.loads((cdir / bo.trialName(i) / "trial.json").read_text()) for i in range(3)]
    assert [t["source"] for t in trials] == ["defaults", "random", "gp"]
    goal_seeds = set()
    for t in trials:
        for m in ("M2", "M3"):
            r = EpisodeReplayer(cdir / bo.trialName(t["trial"]) / f"{m}.json")
            assert r.configs["planner"]["config"]["temperature"] == pytest.approx(t["params"]["temperature"])
            # The defaults trial runs the object's own weights, so the config has no overrides.
            weights = r.configs["rollout_task"]["config"]["cost_weights"]
            if t["source"] == "defaults":
                assert weights is None
            else:
                assert weights == pytest.approx({"w_quat": t["params"]["w_quat"]})
            goal_seeds.add(r[0].summary["goal_seed"])
    assert len(goal_seeds) == 1                       # every trial and model, the same episode
    state = json.loads((cdir / "bo_state.json").read_text())
    assert state["trials"] == [0, 1, 2] and len(state["func_vals"]) == 3
    summary = json.loads((cdir / "bo_summary.json").read_text())
    assert summary["best"]["trial"] == summary["ranked"][0] and summary["replay"]
    # Resume: nothing reruns.
    mtime = (cdir / "trial_0001" / "trial.json").stat().st_mtime
    assert bo.main([str(p), "--cell", "0", "--outdir", str(out), "--workers", "2"]) == 0
    assert (cdir / "trial_0001" / "trial.json").stat().st_mtime == mtime
    assert bo.main([str(p), "--summarize", "--outdir", str(out)]) == 0
    assert (out / "summary.csv").exists() and (out / "best.csv").exists()
