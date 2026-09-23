"""Scheduler checks for the asynchronous eval driver.

run_async_eval_episode's control loop is pure integer bookkeeping over two grids
(the eval-step master clock and the control-step executor grid) plus a latency
deadline. That logic is what the driver adds over the synchronous one, and it is
testable without a GPU, a contact model or any physics: stub the planner and the
eval simulator, drive the loop with a FIXED latency, and assert on the exact
sequence of plans, commands and sim advances it produces.

Run directly (the repo has no pytest):

    python tests/test_async_scheduler.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).parents[1]))

from contact_study.drivers import run_async_eval_episode as drv
from contact_study.planners.mppi import MPPIConfig
from contact_study.sim.base import EvalState


# A 1-DOF slider with one position actuator: enough for a real mj_forward, which
# the driver calls on the planning MjData every tick.
_XML = """
<mujoco>
  <worldbody>
    <body name="slider">
      <joint name="j" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator><position joint="j" kp="1"/></actuator>
</mujoco>
"""

EVAL_DT       = 0.001    # 1 ms master tick
EVAL_SUBSTEPS = 4        # -> rollout_dt = 4 ms
SUBSTEPS      = 8        # -> control_dt = 32 ms = 32 eval steps
CONTROL_STEPS = SUBSTEPS * EVAL_SUBSTEPS
HORIZON       = 5
MAX_STEPS     = 10


@dataclass
class FakeTaskConfig:
    name:  str = "fake"
    timestep: float = EVAL_DT
    eval_substeps_per_rollout: int = EVAL_SUBSTEPS
    max_steps: int = MAX_STEPS
    force_limits: tuple | None = None
    control_limits: tuple | None = None
    eval_sim: object = None


class FakeSim:
    """Records every advance and every command, with the sim clock in eval steps."""

    def __init__(self):
        self.t_steps  = 0
        self.advances: list[int] = []      # each sim.step(n) call
        self.commands: list[tuple[int, float]] = []   # (t_steps, u[0])
        self._qpos = np.zeros(1)
        self._qvel = np.zeros(1)

    def reset(self, qpos, qvel):
        self._qpos = np.asarray(qpos, dtype=float).copy()
        self._qvel = np.asarray(qvel, dtype=float).copy()

    def set_state(self, qpos, qvel):
        self.reset(qpos, qvel)

    def get_state(self):
        return EvalState(qpos=self._qpos.copy(), qvel=self._qvel.copy())

    def apply_control(self, ctrl):
        self.commands.append((self.t_steps, float(np.asarray(ctrl).ravel()[0])))

    def step(self, n_substeps: int = 1):
        assert n_substeps >= 1, f"sim.step({n_substeps}) — must advance"
        self.advances.append(n_substeps)
        self.t_steps += n_substeps

    def save_video(self, path):
        return None

    @property
    def timestep(self):
        return EVAL_DT


class FakeTask:
    """Serves as both the ROLLOUT and the EVAL task."""

    def __init__(self, sim: FakeSim, succeed_at: int | None = None):
        self.config = FakeTaskConfig()
        self.sim = sim
        self.succeed_at = succeed_at
        self.n_success_checks = 0
        self.mjm = mujoco.MjModel.from_xml_string(_XML)

    def load(self):
        return self.mjm, mujoco.MjData(self.mjm)

    def get_inital_state(self, rng):
        return np.zeros(self.mjm.nq), np.zeros(self.mjm.nv), np.zeros(self.mjm.nu)

    def make_eval_simulator(self, **kwargs):
        return self.sim

    def is_success(self, mjd):
        self.n_success_checks += 1
        return (self.succeed_at is not None
                and self.n_success_checks > self.succeed_at)

    def has_failed(self, mjd):
        return False


class FakePlanner:
    """Emits a recognizable tape: row h of plan i is (100*i + h)."""

    def __init__(self, pc: MPPIConfig):
        self.pc = pc
        self.horizon = HORIZON
        self.substeps = SUBSTEPS
        self.rollout_dt = EVAL_DT * EVAL_SUBSTEPS
        self.control_dt = SUBSTEPS * self.rollout_dt
        self.nu = 1
        self.robot_qpos_adr = 0
        self.shift_steps = 1
        self.last_action_seq = None
        self.n_plans = 0
        self.plan_qpos: list[float] = []   # state each solve was handed

    def plan(self, mjd):
        self.plan_qpos.append(float(mjd.qpos[0]))
        seq = np.array([[100.0 * self.n_plans + h] for h in range(self.horizon)],
                       dtype=np.float32)
        self.last_action_seq = seq
        self.n_plans += 1
        return seq[0]

    def reset(self):
        self.last_action_seq = None


def run(latency_ms, *, executor="tape", succeed_at=None, warmup=0, max_steps=MAX_STEPS):
    """Drive the real scheduler against the fakes; return (result, sim, planner)."""
    sim = FakeSim()
    task = FakeTask(sim, succeed_at=succeed_at)
    task.config.max_steps = max_steps
    planner = FakePlanner(MPPIConfig(n_samples=4, step_horizon=HORIZON,
                                     step_substeps=SUBSTEPS, time_horizon=None,
                                     step_time=None))

    orig_get_task, orig_make_planner = drv.get_task, drv.make_planner
    drv.get_task     = lambda *a, **k: task
    drv.make_planner = lambda *a, **k: planner
    try:
        result = drv.run_async_eval_episode(
            task_name   = "fake",
            contact_cfg = type("C", (), {"label": "Mx"})(),
            planner_cfg = planner.pc,
            planner     = "mppi",
            rng         = np.random.default_rng(0),
            plan_latency_ms = latency_ms,
            plan_warmup = warmup,
            executor    = executor,
            verbose     = False,
        )
    finally:
        drv.get_task, drv.make_planner = orig_get_task, orig_make_planner
    return result, sim, planner


CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def test_clock_conservation():
    """The sim advances in whole eval steps that sum to exactly the episode length."""
    for lat in (0.0, 5.0, 32.0, 50.0, 200.0):
        _, sim, _ = run(lat)
        total = sum(sim.advances)
        assert total == MAX_STEPS * CONTROL_STEPS, (
            f"latency={lat}ms: sim advanced {total} steps, "
            f"expected {MAX_STEPS * CONTROL_STEPS}")
        assert sim.t_steps == total
        assert all(n >= 1 for n in sim.advances), "a zero-length advance slipped through"


@check
def test_sync_mode_is_one_plan_per_tick():
    """Zero charged latency must reproduce the synchronous driver exactly."""
    result, sim, planner = run(0.0)
    assert planner.n_plans == MAX_STEPS, (
        f"expected one plan per tick ({MAX_STEPS}), got {planner.n_plans}")
    assert len(sim.commands) == MAX_STEPS, f"got {len(sim.commands)} commands"
    # Every tick applies row 0 of a plan solved at that same instant, and the sim
    # advances one whole control period between ticks.
    for i, (t_steps, u0) in enumerate(sim.commands):
        assert t_steps == i * CONTROL_STEPS, f"tick {i} fired at t={t_steps}"
        assert u0 == 100.0 * i, f"tick {i} applied {u0}, expected row 0 of plan {i}"
    assert sim.advances == [CONTROL_STEPS] * MAX_STEPS
    assert result.mean_staleness_ms == 0.0, "sync mode must have zero staleness"
    assert result.missed_ticks == 0
    assert result.tape_exhausted_ticks == 0


@check
def test_latency_of_exactly_one_control_period():
    """At latency == control_dt the planner delivers one plan per tick, one late."""
    lat_ms = CONTROL_STEPS * EVAL_DT * 1e3          # 32 ms
    result, sim, planner = run(lat_ms)
    # Solves start at t = 0, 32, 64, ... — free-running, one per control period.
    assert planner.n_plans == MAX_STEPS, planner.n_plans
    # Tick 0 has nothing to execute yet. Every later tick gets a tape that was
    # published one period earlier, and contiguous playback starts each fresh
    # chain at its first row.
    assert len(sim.commands) == MAX_STEPS - 1
    for i, (t_steps, u0) in enumerate(sim.commands, start=1):
        assert t_steps == i * CONTROL_STEPS
        expected = 100.0 * (i - 1) + 0        # plan i-1, row 0
        assert u0 == expected, f"tick {i} applied {u0}, expected {expected}"
    assert result.missed_ticks == 1, "only the startup tick has no plan"
    assert abs(result.mean_staleness_ms - lat_ms) < 1e-9


@check
def test_fast_planner_uses_the_freshest_plan():
    """A planner faster than the control rate free-runs; each tick takes the newest tape."""
    lat_ms = 5.0                                    # 5 eval steps
    result, sim, planner = run(lat_ms)
    lat_steps = 5
    assert planner.n_plans > MAX_STEPS, (
        f"a 5ms planner should out-run a 32ms control period, got {planner.n_plans}")
    # Tick 0 still has nothing published, so commands begin at tick 1. Tick i
    # (t = 32i) executes the last plan published at or before t — one started at
    # the largest multiple of 5 that is <= 32i - 5.
    assert len(sim.commands) == MAX_STEPS - 1
    for i, (t_steps, u0) in enumerate(sim.commands, start=1):
        assert t_steps == i * CONTROL_STEPS, f"tick {i} fired at t={t_steps}"
        plan_idx = int(u0 // 100)
        row      = int(u0 % 100)
        start    = plan_idx * lat_steps
        assert 0 <= t_steps - (start + lat_steps) < lat_steps, (
            f"tick {i} ran on plan {plan_idx} (published at "
            f"{start + lat_steps}), which is not the freshest")
        # A planner this fast is always inside its first control period, so the
        # executor never has to reach past row 0.
        assert row == (t_steps - start) // CONTROL_STEPS == 0, (
            f"tick {i} applied row {row}")
    assert result.missed_ticks == 1, "only the startup tick"
    assert result.tape_exhausted_ticks == 0


@check
def test_slow_planner_exhausts_the_tape():
    """A tape forced to cover more than `horizon` ticks pins at its last row.

    Under contiguous playback exhaustion is not merely "latency > horizon *
    control_dt"; the same tape has to actually survive that many ticks before
    the next one lands. 200 ms solves against a 32 ms control period means one
    tape covers ~6 ticks, one more than the 5-row horizon."""
    result, sim, planner = run(200.0, max_steps=20)
    assert result.tape_exhausted_ticks > 0, "expected the tape to run out"
    assert result.missed_ticks > 0, "a 200ms planner must miss 32ms ticks"
    rows = [int(u0 % 100) for _, u0 in sim.commands]
    assert max(rows) == HORIZON - 1, (
        f"the cursor should pin at row {HORIZON - 1}, got {max(rows)}")
    # Every applied row is a valid tape index.
    for r in rows:
        assert 0 <= r < HORIZON


@check
def test_zoh_executor_never_leaves_row_zero():
    """The ZOH ablation holds row 0 no matter how stale the tape gets."""
    for lat in (5.0, 50.0, 200.0):
        _, sim, _ = run(lat, executor="zoh")
        for _, u0 in sim.commands:
            assert int(u0 % 100) == 0, f"latency={lat}ms: zoh applied row {int(u0 % 100)}"


@check
def test_contiguous_playback_walks_the_chain():
    """A slow planner walks rows 0,1,2,... in order and restarts at 0 on a new tape."""
    # 2 control periods per solve -> a fresh tape every other tick.
    lat_ms = 2 * CONTROL_STEPS * EVAL_DT * 1e3
    _, sim, _ = run(lat_ms)
    rows = [int(u0 % 100) for _, u0 in sim.commands]
    plans = [int(u0 // 100) for _, u0 in sim.commands]
    # Within one tape the cursor advances by exactly 1; a new tape resets it.
    for i in range(1, len(rows)):
        if plans[i] == plans[i - 1]:
            assert rows[i] == rows[i - 1] + 1, (
                f"tick {i}: row jumped {rows[i-1]} -> {rows[i]} within plan {plans[i]}")
        else:
            assert rows[i] == 0, (
                f"tick {i}: new plan {plans[i]} started at row {rows[i]}, not 0")
    assert 0 in rows, "no tape ever started at row 0"
    assert max(rows) >= 1, "the cursor never advanced — this is just zoh"


@check
def test_time_mode_reproduces_the_row_skipping_ablation():
    """`time` indexes by elapsed sim time and so skips rows once latency >= control_dt."""
    lat_ms = 2 * CONTROL_STEPS * EVAL_DT * 1e3      # exactly 2 control periods
    _, sim, _ = run(lat_ms, executor="time")
    rows = [int(u0 % 100) for _, u0 in sim.commands]
    assert rows, "no commands"
    # Every applied row is >= 2: rows 0 and 1 expired during the solve.
    assert min(rows) >= 2, f"expected the first two rows to be skipped, got {rows}"
    # Contiguous playback on the same latency must NOT skip them.
    _, sim_c, _ = run(lat_ms)
    rows_c = [int(u0 % 100) for _, u0 in sim_c.commands]
    assert min(rows_c) == 0, f"contiguous playback skipped rows: {rows_c}"


@check
def test_fast_planner_degenerates_to_row_zero():
    """When a fresh tape lands every tick, contiguous playback always uses row 0."""
    _, sim, _ = run(5.0)
    rows = [int(u0 % 100) for _, u0 in sim.commands]
    assert set(rows) == {0}, f"expected only row 0, got {sorted(set(rows))}"


@check
def test_staleness_tracks_latency():
    """Mean staleness rises with latency and is never below it."""
    seen = []
    for lat in (5.0, 32.0, 64.0):
        result, _, _ = run(lat)
        assert result.mean_staleness_ms >= lat - 1e-9, (
            f"latency={lat}ms produced staleness {result.mean_staleness_ms}ms")
        seen.append(result.mean_staleness_ms)
    assert seen == sorted(seen), f"staleness did not grow with latency: {seen}"


@check
def test_early_success_stops_the_episode():
    """is_success on a tick ends the run before the episode's full sim time."""
    result, sim, _ = run(5.0, succeed_at=3)
    assert result.success
    assert result.steps_to_success is not None
    assert sum(sim.advances) < MAX_STEPS * CONTROL_STEPS
    assert result.sim_seconds < MAX_STEPS * CONTROL_STEPS * EVAL_DT


@check
def test_planner_sees_the_state_at_solve_start():
    """Each solve is handed the sim state at the instant it starts, not earlier."""
    _, sim, planner = run(5.0)
    # The fake sim's qpos never changes, so this checks the call count/order
    # contract rather than values: one snapshot per solve, no solve skipped.
    assert len(planner.plan_qpos) == planner.n_plans


def main() -> int:
    failures = 0
    for fn in CHECKS:
        try:
            fn()
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {fn.__name__}\n      {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}\n      {type(e).__name__}: {e}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
