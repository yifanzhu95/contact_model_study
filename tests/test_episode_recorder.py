"""EpisodeRecorder and EpisodeReplayer: recording, summaries, saving, replaying, combining."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from conftest import ROLLOUT_CUBE
from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig
from ContactModelStudy.Tasks.TaskBase import TaskRole
from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder, EpisodeReplayer

pytestmark = pytest.mark.gpu
STEPS = 15


@pytest.fixture(scope="module")
def setup():
    """Eval task + CPU sim, rollout task + GPU planner (uncertainty on)."""
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config
    from ContactModelStudy.Simulators.Mujoco import Mujoco, MujocoConfig
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco, VectorizedMujocoConfig
    ev = CubeReorient(LeapReorientConfig(role=TaskRole.EVAL, seed=0))
    ro = CubeReorient(LeapReorientConfig(role=TaskRole.ROLLOUT))
    sim = Mujoco(ev.getModelPath(), MujocoConfig(timestep=ev.timestep))
    rsim = VectorizedMujoco(ro.getModelPath(), VectorizedMujocoConfig(horizon=8, substeps=4), N=128)
    planner = MPPI(rsim, ro, MPPI_Config(noise_sigma=0.1, seed=0, return_uncertainty=True))
    return ev, ro, sim, planner


def _episode(rec, ev, sim, planner, with_sigma=True):
    q0, v0, u0 = ev.getInitialState()
    sim.SetState(q0, v0)
    sim.SetControl(u0)
    planner.Reset()
    rows = []
    for _ in range(STEPS):
        q, qd = sim.GetState()
        t0 = time.perf_counter()
        u, sig = planner.Plan(q, qd, u=sim.GetControl())
        dt = time.perf_counter() - t0
        rec.recordStateAndAction(q, qd, u, sig if with_sigma else None, dt if with_sigma else None)
        rows.append((q, u, sig))
        sim.SetControl(u)
        sim.Step(4)
    return rows


@pytest.fixture(scope="module")
def saved(setup, tmp_path_factory):
    """Two recorded episodes, saved; returns (recorder, path, per-episode truth)."""
    ev, _, sim, planner = setup
    rec = EpisodeRecorder(ev, sim, planner, note="test", cli_args={"steps": STEPS})
    truth = [_episode(rec, ev, sim, planner, True)]
    rec.episodeFinished("timeout", extra_note=0)
    truth.append(_episode(rec, ev, sim, planner, False))
    rec.episodeFinished("failure", extra_note=1)
    path = rec.Save(tmp_path_factory.mktemp("rec") / "run.json")
    return rec, path, truth


def test_save_writes_one_json_and_one_npy_per_episode(saved):
    rec, path, _ = saved
    doc = json.loads(path.read_text())
    files = sorted(p.name for p in path.parent.iterdir())
    assert len([f for f in files if f.endswith(".npy")]) == 2
    assert len({e["id"] for e in doc["episodes"]}) == 2
    assert doc["format_version"] == 2 and doc["finish_reasons"] == {"failure": 1, "timeout": 1}


def test_configs_record_both_tasks_both_simulators_and_the_planner(saved, setup):
    ev, ro, _, _ = setup
    c = json.loads(saved[1].read_text())["configs"]
    assert set(c) == {"eval_task", "eval_simulator", "planner", "rollout_task",
                      "rollout_simulator", "metadata"}
    assert c["eval_task"]["role"] == "eval" and c["rollout_task"]["role"] == "rollout"
    assert c["eval_task"]["model_path"].endswith("env_leap_eval_cube.xml")
    assert c["rollout_task"]["model_path"].endswith("env_leap_rollout_cube_high_high.xml")
    assert c["rollout_task"]["timestep"] == ro.timestep and c["eval_task"]["timestep"] == ev.timestep
    assert c["planner"]["config"]["return_uncertainty"] is True
    assert c["rollout_simulator"]["resolved"] == {"resolved_substeps": 4, "control_timestep": 0.008,
                                                  "resolved_horizon": 8, "horizon_duration": 0.064}


def test_npy_is_a_plain_structured_array(saved):
    _, path, _ = saved
    entry = json.loads(path.read_text())["episodes"][0]
    arr = np.load(path.parent / entry["file"])          # allow_pickle=False by default
    assert arr.dtype.names == ("t", "q", "q_dot", "u", "sigma_u", "planning_time", "planner_cost")
    assert arr.shape == (STEPS,)


def test_replay_equals_what_was_recorded(saved):
    _, path, truth = saved
    rp = EpisodeReplayer(path)
    e0, e1 = rp[0], rp[1]
    assert np.allclose(e0.q, [r[0] for r in truth[0]])
    assert np.allclose(e0.u, [r[1] for r in truth[0]])
    assert np.allclose(e0.sigma_u, [r[2] for r in truth[0]])
    assert np.isnan(e1.sigma_u).all() and np.isnan(e1.planning_time).all()
    assert np.isfinite(e0.planning_time).all() and np.all(np.diff(e0.t) > 0)
    assert e0.finish_reason == "timeout" and e1.summary["extra_note"] == 1
    assert [e.id for e in rp] == rp.ids and rp.episode(rp.ids[1]).id == rp.ids[1]
    states = list(rp.iterStates())
    assert len(states) == 2 * STEPS and states[STEPS][:2] == (rp.ids[1], 0)


def test_save_refuses_mid_episode_and_combine_refuses_while_recording(setup, tmp_path):
    ev, _, sim, planner = setup
    rec = EpisodeRecorder(ev, sim, planner)
    q0, v0, u0 = ev.getInitialState()
    rec.recordStateAndAction(q0, v0, u0)
    with pytest.raises(RuntimeError):
        rec.Save(tmp_path / "run.json")
    with pytest.raises(RuntimeError):
        rec.Combine(rec)


def test_combine_and_clear(saved, setup):
    rec, _, _ = saved
    ev, _, sim, planner = setup
    q0, v0, u0 = ev.getInitialState()
    other = EpisodeRecorder(ev, sim, planner, note="test", cli_args={"steps": STEPS})
    other.recordStateAndAction(q0, v0, u0)
    other.episodeFinished("success")
    both = rec.Combine(other)
    assert (len(both), len(rec), len(other)) == (3, 2, 1)
    with pytest.raises(ValueError, match="metadata"):
        rec.Combine(EpisodeRecorder(ev, sim, planner, note="different"))
    other.Clear()
    assert len(other) == 0 and not other.recording


def test_save_without_steps_writes_only_the_summary(saved, tmp_path):
    rec, _, _ = saved
    out = rec.Save(tmp_path / "run.json", save_steps=False)
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]
    e = EpisodeReplayer(out)[0]
    assert len(e) == STEPS and not e.has_steps and e.q is None
    assert list(EpisodeReplayer(out).iterStates()) == []


# -- GenerateSummary ---------------------------------------------------------
def test_generate_summary(setup, tmp_path):
    ev, _, sim, planner = setup
    t = CubeReorient(LeapReorientConfig(role=TaskRole.EVAL, seed=1))
    rec = EpisodeRecorder(t, sim, planner)
    q0, v0, u0 = t.getInitialState()
    sim.SetState(q0, v0)
    # Episode 0: goal A, 3 steps, success at step 3, goal B, 2 more steps, timeout.
    g1 = t.sampleNewGoal(); t.setGoal(g1); rec.recordGoal(g1)
    start_err = t.goalErrors(q0, v0)
    for k, pt in enumerate([1.0, 0.02, 0.03]):
        planner.last_min_cost = 10.0 - k
        rec.recordStateAndAction(q0, v0, u0, planning_time=pt)
    rec.recordSuccess()
    g2 = t.sampleNewGoal(); t.setGoal(g2); rec.recordGoal(g2)
    for _ in range(2):
        rec.recordStateAndAction(q0, v0, u0, planning_time=0.04)
    sim.Step(10)
    q_end, qd_end = sim.GetState()
    rec.episodeFinished("timeout", video="v.mp4")
    s = rec.GenerateSummary(0)
    assert s["goals"] == [t.goalFace(g1), t.goalFace(g2)]
    assert s["success_steps"] == [3] and s["goals_reached"] == 1 and s["success"]
    assert s["finish_reason"] == "timeout" and not s["failed"]
    assert s["goal_errors_start"] == start_err
    assert np.allclose(s["q_end"], q_end) and s["goal_errors_end"] == t.goalErrors(q_end, qd_end)
    assert s["plan_s_first"] == 1.0 and s["plan_s_max"] == 0.04
    assert np.isclose(s["plan_s_mean"], (0.02 + 0.03 + 0.04 * 2) / 4)
    assert s["video"] == "v.mp4" and s["n_steps"] == 5
    # Episode 1: a failure before any step.
    rec.episodeFinished("failure")
    z = rec.GenerateSummary(-1)
    assert z["failed"] and not z["success"] and z["n_steps"] == 0 and z["plan_s_mean"] is None
    assert z["goal_errors_start"] == z["goal_errors_end"]
    b = rec.GenerateSummary()
    assert (b["n_episodes"], b["n_success"], b["n_failed"], b["success_rate"]) == (2, 1, 1, 0.5)
    out = rec.Save(tmp_path / "run.json")
    rp = EpisodeReplayer(out)
    assert rp[0].summary == json.loads(json.dumps(s))
    assert np.allclose(rp[0].planner_cost[:3], [10, 9, 8])
    assert rp[1].q.shape == (0, 23) and rp[1].u.shape == (0, 16)


def test_version_1_files_are_still_readable(saved, tmp_path):
    _, path, _ = saved
    doc = json.loads(path.read_text())
    doc["format_version"] = 1
    for e in doc["episodes"]:          # a v1 file has no planner_cost field
        arr = np.load(path.parent / e["file"])
        names = [n for n in arr.dtype.names if n != "planner_cost"]
        np.save(tmp_path / e["file"], arr[names])
    (tmp_path / "run.json").write_text(json.dumps(doc))
    e = EpisodeReplayer(tmp_path / "run.json")[0]
    assert len(e) == STEPS and np.isnan(e.planner_cost).all()
