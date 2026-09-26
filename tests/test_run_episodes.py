"""run_episodes.py end to end. Slow: every test runs real episodes.

Every run passes its settings explicitly, so changing the driver's defaults
does not change what these tests check.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from ContactModelStudy.Drivers import run_episodes as drv
from ContactModelStudy.Utils.EpisodeRecorder import EpisodeReplayer

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

#: A small, fixed setup: 4 substeps x 4 ms = 16 ms control step, horizon 8.
BASE = ["--timestep", "0.0005", "--eval-steps-per-rollout-step", "8", "--substeps", "4",
        "--horizon", "8", "--n-samples", "128", "--noise-sigma", "0.1", "--temperature", "10",
        "--settle", "0", "--rollout-model", "M2", "--eval-sim", "mujoco", "--uncertainty",
        "--save-steps", "--no-video"]


def run(tmp_path, *extra, name="run.json"):
    """Run the driver; returns the results path (or None with --no-results)."""
    out = tmp_path / name
    argv = [*BASE, "--results", str(out), *extra]
    assert drv.main(argv) == 0
    return out


def test_two_episodes_are_recorded(tmp_path):
    rp = EpisodeReplayer(run(tmp_path, "--n-episodes", "2", "--steps", "20"))
    assert len(rp) == 2 and all(len(e) == 20 for e in rp)
    e = rp[0]
    assert e.q.shape == (20, 23)
    assert np.isfinite(e.sigma_u).all() and 0 < e.sigma_u.mean() < 0.2
    assert np.isfinite(e.planning_time).all()
    assert np.allclose(np.diff(e.t), 0.016)                    # one control step apart
    for k in ("goals", "goal_errors_end", "q_end", "plan_s_mean", "settle_s", "episode"):
        assert k in e.summary
    m = rp.configs["metadata"]
    assert m["eval_sim"] == "mujoco" and m["cli_args"]["steps"] == 20


def test_interrupted_run_keeps_what_finished(tmp_path, monkeypatch):
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI
    orig, calls = MPPI.Plan, [0]

    def flaky(self, *a, **k):
        calls[0] += 1
        if calls[0] == 15:
            raise KeyboardInterrupt
        return orig(self, *a, **k)

    monkeypatch.setattr(MPPI, "Plan", flaky)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, "--n-episodes", "2", "--steps", "10")
    rp = EpisodeReplayer(tmp_path / "run.json")
    assert [e.finish_reason for e in rp] == ["timeout", "interrupted"] and len(rp[1]) == 4


def test_no_uncertainty_stores_nan(tmp_path):
    rp = EpisodeReplayer(run(tmp_path, "--steps", "5", "--no-uncertainty"))
    assert np.isnan(rp[0].sigma_u).all()
    assert rp.configs["planner"]["config"]["return_uncertainty"] is False


def test_save_flags(tmp_path):
    out = run(tmp_path, "--steps", "5", "--no-save-steps")
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]
    assert not EpisodeReplayer(out)[0].has_steps
    empty = tmp_path / "none"
    empty.mkdir()
    assert drv.main([*BASE, "--steps", "5", "--no-results"]) == 0
    assert list(empty.iterdir()) == []


def test_settle_runs_before_the_first_step(tmp_path):
    e = EpisodeReplayer(run(tmp_path, "--steps", "5", "--settle", "0.5"))[0]
    assert e.summary["settle_s"] == 0.5 and np.isclose(e.t[0], 0.5)


@pytest.mark.parametrize("model, cls", [("M1", "VectorizedMujoco"), ("M2", "VectorizedMujoco"),
                                        ("M3", "ComFree"), ("M4", "XPBD")])
def test_each_rollout_model(tmp_path, model, cls):
    rp = EpisodeReplayer(run(tmp_path, "--steps", "5", "--rollout-model", model))
    assert rp.configs["rollout_simulator"]["class"] == cls and len(rp[0]) == 5


@pytest.mark.parametrize("sim", ["pinocchio", "drake"])
def test_each_eval_simulator(tmp_path, sim):
    pytest.importorskip({"pinocchio": "pinocchio", "drake": "pydrake"}[sim])
    rp = EpisodeReplayer(run(tmp_path, "--steps", "5", "--eval-sim", sim))
    assert rp.configs["eval_simulator"]["class"].lower() == sim and len(rp[0]) == 5


@pytest.mark.parametrize("task, scene", [("duck_reorient", "duck_high_high"), ("ball_reorient", "ball_high_high")])
def test_each_task(tmp_path, task, scene):
    rp = EpisodeReplayer(run(tmp_path, "--steps", "5", "--task", task))
    assert rp.configs["rollout_task"]["model_path"].endswith(f"env_leap_rollout_{scene}.xml")


def test_duration_flags_resolve(tmp_path):
    argv = [a for a in BASE]
    argv[argv.index("--substeps"):argv.index("--substeps") + 2] = ["--ctrl-time-step", "0.032"]
    argv[argv.index("--horizon"):argv.index("--horizon") + 2] = ["--time-horizon", "0.256"]
    out = tmp_path / "run.json"
    assert drv.main([*argv, "--steps", "5", "--results", str(out)]) == 0
    r = json.loads(out.read_text())["configs"]["rollout_simulator"]["resolved"]
    assert (r["resolved_substeps"], r["resolved_horizon"]) == (8, 8)
    assert np.allclose(np.diff(EpisodeReplayer(out)[0].t), 0.032)


@pytest.mark.parametrize("pair", [["--horizon", "8", "--time-horizon", "0.2"],
                                  ["--substeps", "2", "--ctrl-time-step", "0.01"],
                                  ["--settle", "-1"], ["--rollout-model", "M9"]])
def test_bad_flags_are_parser_errors(pair):
    with pytest.raises(SystemExit):
        drv.main(pair)


def test_buffer_defaults_reach_the_rollout(tmp_path):
    c = json.loads(run(tmp_path, "--steps", "2").read_text())["configs"]["rollout_simulator"]["config"]
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujocoConfig
    d = VectorizedMujocoConfig()
    assert (c["nconmax"], c["njmax"]) == (d.nconmax, d.njmax)
    c = json.loads(run(tmp_path, "--steps", "2", "--njmax", "128", name="r2.json")
                   .read_text())["configs"]["rollout_simulator"]["config"]
    assert c["njmax"] == 128


def test_video_length_matches_simulated_time(tmp_path):
    video = tmp_path / "v.mp4"
    argv = [a for a in BASE if a != "--no-video"]
    assert drv.main([*argv, "--steps", "20", "--settle", "0.2", "--video", str(video), "--no-results"]) == 0
    try:
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                    "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout)
    except (FileNotFoundError, ValueError):
        pytest.skip("ffprobe not available")
    assert abs(dur - (0.2 + 20 * 0.016)) < 0.1
