"""run_episodes_async.py end to end. Slow: every test runs real episodes.

Every run passes its settings explicitly, so changing the driver's defaults
does not change what these tests check.
"""

from __future__ import annotations

import numpy as np
import pytest

from ContactModelStudy.Drivers import run_episodes as drv
from ContactModelStudy.Drivers import run_episodes_async as drv_async
from ContactModelStudy.Utils.EpisodeRecorder import EpisodeReplayer

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

#: As tests/test_run_episodes.py: 4 substeps x 4 ms = 16 ms control step, horizon 8.
BASE = ["--timestep", "0.0005", "--eval-steps-per-rollout-step", "8", "--substeps", "4",
        "--horizon", "8", "--n-samples", "128", "--noise-sigma", "0.1", "--temperature", "10",
        "--settle", "0", "--rollout-model", "M2", "--eval-sim", "mujoco", "--uncertainty",
        "--save-steps", "--no-video", "--warm-start", "shift"]


def run(main, tmp_path, *extra, name="run.json"):
    out = tmp_path / name
    assert main([*BASE, "--results", str(out), *extra]) == 0
    return EpisodeReplayer(out)[0]


@pytest.fixture
def deterministic_plan(monkeypatch):
    """Make ``MPPI.Plan`` a deterministic function of the state.

    MJWarp's rollouts are not bit-reproducible (two synchronous runs already
    differ by ~1e-5 at step 0), so comparing two drivers bit for bit needs a
    planner that is. The mean is written into the planner and resolved by its
    own ``_extractAction``, so the commands, tape and warm-start shift are the
    real ones; only the optimization is skipped.
    """
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI

    def plan(self, q, q_dot, u=None):
        q = np.asarray(q, dtype=float).ravel()
        self._setCommandBase(q, u)
        self.last_plan_start = {**self.SaveState(), "last_action_seq": None, "last_plan_ok": False}
        H, nu = self.horizon, self.nu
        mean = 0.02 * np.sin(np.outer(np.arange(1, H + 1), np.arange(nu)) + 50 * q[0])
        self.U_wp.assign(mean.astype(np.float32))
        self.last_plan_ok = True
        return self._result(self._extractAction(), np.zeros(nu, dtype=np.float32))

    monkeypatch.setattr(MPPI, "Plan", plan)


@pytest.mark.parametrize("mode", ["pos_relative", "absolute", "ctrl_relative"])
def test_zero_latency_is_the_synchronous_driver(tmp_path, mode, deterministic_plan):
    args = ("--steps", "20", "--control-mode", mode)
    sync = run(drv.main, tmp_path, *args, name="sync.json")
    asyn = run(drv_async.main, tmp_path, *args, "--plan-latency-ms", "0", name="async.json")
    assert len(sync) == len(asyn) == 20
    assert np.array_equal(sync.u, asyn.u)
    assert np.array_equal(sync.q, asyn.q)
    s = asyn.summary
    assert s["n_plans"] == 20 and s["stale_ticks"] == s["held_ticks"] == 0


def test_tape_plays_the_next_row_while_the_planner_is_busy(tmp_path):
    # 48 ms latency = 3 control steps: plan i starts at tick 3i and is played,
    # rows 0, 1, 2, over ticks 3i+3 .. 3i+5. Absolute control: the command is the row.
    e = run(drv_async.main, tmp_path, "--steps", "30", "--control-mode", "absolute",
            "--plan-latency-ms", "48", "--executor", "tape")
    s = e.summary
    assert s["finish_reason"] == "timeout" and len(e) == 30
    assert s["held_ticks"] == 3 and s["n_plans"] == 10 and s["stale_ticks"] == 18
    assert np.isclose(s["latency_ms_mean"], 48) and np.isclose(s["sim_seconds"], 30 * 0.016)
    assert np.isclose(s["staleness_ms_mean"], 48 + 16)       # ages 48, 64, 80 ms
    for tick in range(3, 30):
        assert np.array_equal(e.u[tick], e.action_seq[tick][(tick - 3) % 3])
        assert np.array_equal(e.action_seq[tick], e.action_seq[tick - (tick - 3) % 3])
    # One planning time per plan, on the tick that first applies it.
    assert np.isnan(e.planning_time[:3]).all()
    assert np.isfinite(e.planning_time[3::3]).all()
    assert np.isnan(e.planning_time[4::3]).all() and np.isnan(e.planning_time[5::3]).all()


def test_zoh_reapplies_row_zero(tmp_path):
    e = run(drv_async.main, tmp_path, "--steps", "12", "--control-mode", "absolute",
            "--plan-latency-ms", "48", "--executor", "zoh")
    assert all(np.array_equal(e.u[t], e.action_seq[t][0]) for t in range(3, 12))


def test_measured_latency(tmp_path):
    e = run(drv_async.main, tmp_path, "--steps", "20")
    s = e.summary
    assert len(e) == 20 and s["n_plans"] > 0 and s["latency_ms_mean"] > 0
    assert np.isclose(s["sim_seconds"], 20 * 0.016)
    assert np.allclose(np.diff(e.t), 0.016)                    # ticks stay on the control grid
    for k in ("staleness_ms_mean", "stale_ticks", "held_ticks", "tape_exhausted_ticks"):
        assert k in s


@pytest.mark.parametrize("flag", [["--plan-latency-ms", "-1"], ["--latency-scale", "0"],
                                  ["--plan-warmup", "-1"]])
def test_bad_flags_are_parser_errors(flag):
    with pytest.raises(SystemExit):
        drv_async.main([*BASE, "--steps", "1", "--no-results", *flag])
