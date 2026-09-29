"""run_episodes_interwoven.py / run_episodes_pooled.py and the pieces they rest on."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import ROLLOUT_CUBE
from ContactModelStudy.Drivers.EpisodePool import episodeSeeds
from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig
from ContactModelStudy.Utils.EpisodeRecorder import EpisodeReplayer
from ContactModelStudy.Utils.EvalSimulators import evalSimConfig, makeEvalSim

BASE = ["--timestep", "0.0005", "--eval-steps-per-rollout-step", "8", "--substeps", "4",
        "--horizon", "8", "--n-samples", "64", "--noise-sigma", "0.1", "--temperature", "10",
        "--settle", "0.1", "--rollout-model", "M2", "--eval-sim", "mujoco", "--uncertainty",
        "--save-steps", "--no-video", "--no-stop-on-success", "--goal-difficulty", "4"]


# -- building blocks ---------------------------------------------------------
def test_episode_seeds_depend_only_on_seed_and_index():
    assert episodeSeeds(0, 3) == episodeSeeds(0, 3)
    assert len({episodeSeeds(0, k) for k in range(20)}) == 20
    assert episodeSeeds(0, 3) != episodeSeeds(1, 3)


def test_get_goal_and_reseed():
    a, b = CubeReorient(LeapReorientConfig(seed=0)), CubeReorient(LeapReorientConfig(seed=99))
    a.setGoal(a.sampleNewGoal())
    b.setGoal(a.getGoal())
    assert np.allclose(b.getGoal(), a.getGoal())
    a.reseed(5), b.reseed(5)
    assert np.allclose(a.sampleNewGoal(), b.sampleNewGoal(np.asarray(a._sample_ref)))
    assert a.config.seed == 0                     # the recorded config is untouched


def test_eval_sim_config_matches_what_is_built():
    from conftest import EVAL_CUBE
    name, cfg = evalSimConfig("mujoco", 0.001)
    sim = makeEvalSim("mujoco", EVAL_CUBE, 0.001)
    assert name == type(sim).__name__ and cfg == sim.config


@pytest.mark.gpu
def test_planner_state_roundtrip(cube_initial, cube_tasks):
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco, VectorizedMujocoConfig
    q0, v0, u0 = cube_initial
    p = MPPI(VectorizedMujoco(ROLLOUT_CUBE, VectorizedMujocoConfig(horizon=4, substeps=2), N=32),
             cube_tasks[0], MPPI_Config(seed=0, warm_start=True, adaptive_temp=True))
    p.Plan(q0, v0, u=u0)
    saved = p.SaveState()
    p.Reset()
    p.LoadState({**saved, "U": np.zeros_like(saved["U"]), "lam": 3.0, "resample_count": 7})
    assert p.lam == 3.0 and p._resample_count == 7 and np.all(p.U_wp.numpy() == 0)
    p.LoadState(saved)
    again = p.SaveState()
    assert np.array_equal(again["U"], saved["U"]) and again["noise_seed"] == saved["noise_seed"]
    assert again["resample_count"] == saved["resample_count"] and again["lam"] == saved["lam"]
    with pytest.raises(ValueError):
        p.LoadState({**saved, "U": np.zeros((3, 3))})


# -- the drivers, end to end ---------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
def test_interwoven_run(tmp_path):
    from ContactModelStudy.Drivers import run_episodes_interwoven as iw
    out = tmp_path / "iw.json"
    assert iw.main([*BASE, "--n-episodes", "4", "--steps", "10", "--results", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert [e["summary"]["episode"] for e in doc["episodes"]] == [0, 1, 2, 3]
    assert not any("configs" in e for e in doc["episodes"])    # one shared config set
    # Which worker takes which episode is up to scheduling; with short episodes
    # one worker can take them all before the other has started.
    assert {e["summary"]["worker"] for e in doc["episodes"]} <= {0, 1}
    c = doc["configs"]
    assert (c["planner"]["class"], c["rollout_simulator"]["class"], c["eval_simulator"]["class"]) == \
        ("MPPI", "VectorizedMujoco", "Mujoco")
    e = EpisodeReplayer(out)[1]
    assert len(e) == 10 and np.isfinite(e.sigma_u).all() and np.allclose(np.diff(e.t), 0.016)
    assert np.all(e.planning_time < 1.0)                         # solve time, not queue time


@pytest.mark.gpu
@pytest.mark.slow
def test_pooled_results_do_not_depend_on_pool_size(tmp_path):
    from ContactModelStudy.Drivers import run_episodes_pooled as pooled
    runs = {}
    for w in (1, 3):
        out = tmp_path / f"w{w}.json"
        assert pooled.main([*BASE, "--n-episodes", "4", "--steps", "6", "--workers", str(w),
                            "--results", str(out)]) == 0
        runs[w] = EpisodeReplayer(out)
    a, b = runs[1], runs[3]
    assert [e.summary["goals"] for e in a] == [e.summary["goals"] for e in b]
    assert [e.summary["goal_seed"] for e in a] == [e.summary["goal_seed"] for e in b]
    # Same state, same goal, same noise stream: the first plan agrees to MJWarp's noise.
    assert all(np.abs(x.u[0] - y.u[0]).max() < 5e-3 for x, y in zip(a, b))


def test_a_missing_scene_fails_before_anything_starts(tmp_path):
    from ContactModelStudy.Drivers import run_episodes_pooled as pooled
    with pytest.raises(FileNotFoundError):
        pooled.main([*BASE, "--n-episodes", "2", "--steps", "3", "--obj-acc", "foam4",
                     "--workers", "2", "--results", str(tmp_path / "bad.json")])


@pytest.mark.gpu
def test_a_missing_gpu_stops_the_run_promptly(tmp_path):
    import time
    from ContactModelStudy.Drivers import run_episodes_pooled as pooled
    t0 = time.time()
    assert pooled.main([*BASE, "--gpus", "7", "--n-episodes", "2", "--steps", "3", "--no-results"]) == 1
    assert time.time() - t0 < 60


def test_pool_flag_validation():
    from ContactModelStudy.Drivers import run_episodes_pooled as pooled
    for bad in (["--planners-per-gpu", "0"], ["--workers", "0"]):
        with pytest.raises(SystemExit):
            pooled.main([*BASE, *bad, "--no-results"])
