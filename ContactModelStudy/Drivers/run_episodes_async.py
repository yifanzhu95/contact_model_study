#!/usr/bin/env python3
"""Run closed-loop MPC episodes in which planning takes simulated time.

``run_episodes.py`` freezes the eval simulator while the planner thinks, so
planning latency is free: a contact model that is more accurate but slower
looks strictly better than a fast one. On hardware that is not what happens.
The robot keeps moving during the solve, the action that lands was computed
from a state that has gone stale, and a planner slower than its own control
step misses control ticks. This driver prices that latency in.

Clocks
------
The eval simulator runs nowhere near real time, so the planner and the sim
cannot both track a wall clock. **Simulated time is the master clock** instead:
each ``Plan`` is timed on the wall clock, and that duration is spent as
*simulated* seconds — the sim advances by exactly the planner's latency while
the executor keeps commanding from the plans it already has. Causally that is
asynchronous MPC on hardware, but single-threaded, and deterministic for a
given latency.

    eval_dt      eval timestep; the master clock counts whole eval steps
    control_dt   executor grid; the spacing of a plan's rows
    latency      measured wall seconds of Plan, spent as sim seconds

The planner free-runs: a new solve starts the moment the last one finishes,
from the state at that moment. A finished plan becomes visible to the executor
only once its latency has elapsed.

Executor
--------
Every control_dt the executor commands one row of the most recently published
plan, resolved against the live state for the control mode (``pos_relative``:
``q_robot + U[k]``; ``ctrl_relative``: ``ctrl + U[k]``; ``absolute``: ``U[k]``):

    tape  (default)  contiguous playback: row 0 when a plan is published, then
                     the next row each tick until a newer plan lands, holding
                     the last row once the plan runs out
    zoh              always row 0 of the freshest plan; a plan slower than a
                     control step has its first row re-applied, re-based
    time             the row for the elapsed sim time since the plan's solve
                     started, (t - t_plan) // control_dt; skips the rows that
                     expired during the solve

The old driver measured ~2.8x more per-tick command jump under the two cursor
modes than under ``zoh``: each publish resets the cursor to row 0, a sawtooth.
``tape`` is the default here because it plays out the plan that was made.

Latency
-------
``--plan-latency-ms`` charges a fixed latency instead of the measured one,
which makes a run reproducible across GPUs and makes latency an axis of its
own; ``--plan-latency-ms 0`` falls back to one plan per tick — exactly
``run_episodes.py``, kept as a regression check. ``--latency-scale``
multiplies the measured latency. ``--plan-warmup`` throwaway plans run before
the first episode, so kernel compilation and graph capture are not charged.

Episode length is in sim time (``--steps`` control steps), not in plans: a
slow planner is not handed extra time to recover in.

Takes every ``run_episodes.py`` option, and saves the same results; each
episode's summary also has the async statistics (``n_plans``, latency,
staleness, ``stale_ticks``, ``held_ticks``, ``tape_exhausted_ticks``).

Examples::

    python -m ContactModelStudy.Drivers.run_episodes_async
    python -m ContactModelStudy.Drivers.run_episodes_async --rollout-model M5 --n-episodes 5
    python -m ContactModelStudy.Drivers.run_episodes_async --plan-latency-ms 150 --executor zoh
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if not os.environ.get("MUJOCO_GL") and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ContactModelStudy.Drivers import run_episodes as drv  # noqa: E402
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI  # noqa: E402
from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim  # noqa: E402
from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder, StepSaving  # noqa: E402
from ContactModelStudy.Utils.EvalSimulators import makeEvalSim  # noqa: E402

EXECUTORS = ("tape", "zoh", "time")

#: What the recorder is told about a tick with no plan behind it.
_NO_PLAN = {"last_plan_start": None, "last_action_seq": None, "last_min_cost": None}


def build_parser():
    p = drv.build_parser()
    p.description = __doc__
    g = p.add_argument_group("async")
    g.add_argument("--executor", default="tape", choices=EXECUTORS,
                   help="which row of the latest plan each control tick applies: tape "
                        "(next row, one per tick), zoh (always row 0) or time (the row "
                        "for the time elapsed since the solve started)")
    g.add_argument("--plan-latency-ms", type=float, default=None,
                   help="charge this latency (ms of sim time) per plan instead of the "
                        "measured one; 0 plans once per tick, as run_episodes.py does")
    g.add_argument("--latency-scale", type=float, default=1.0,
                   help="multiply the measured latency (ignored with --plan-latency-ms)")
    g.add_argument("--plan-warmup", type=int, default=2,
                   help="throwaway plans before the first episode, so compilation and "
                        "graph capture are not charged as latency; they leave the "
                        "planner's state and noise stream as they found them")
    g.add_argument("--async-shift", action=argparse.BooleanOptionalAction, default=True,
                   help="under --warm-start shift, shift the mean by the control steps "
                        "the last solve took instead of by one")
    return p


@dataclass
class _Plan:
    """One finished solve, as the executor plays it out."""

    t0: int                     # eval step its solve started at
    mean: np.ndarray | None     # (H, nu), the planner's parameterization; None: failed
    seq: np.ndarray             # (H, nu) commands, as of the solve's state
    sigma: np.ndarray | None    # (nu,) uncertainty of row 0, when asked for
    seconds: float              # measured solve time
    record: dict                # what the recorder reads off the planner
    used: bool = False          # a tick has applied it


def _timedPlan(planner, sim, t: int) -> _Plan:
    q, q_dot = sim.GetState()
    t0 = time.perf_counter()
    out = planner.Plan(q, q_dot, u=sim.GetControl())
    seconds = time.perf_counter() - t0
    sigma = out[1] if isinstance(out, tuple) else None
    # Snapshot now: the next solve rebinds all of these before this plan is played out.
    return _Plan(t0=t, mean=planner.last_mean, seq=planner.last_action_seq, sigma=sigma,
                 seconds=seconds,
                 record={"last_plan_start": planner.last_plan_start,
                         "last_action_seq": planner.last_action_seq,
                         "last_min_cost": getattr(planner, "last_min_cost", None)})


def _command(plan: _Plan, k: int, mode: str, q_robot: np.ndarray, u_now: np.ndarray) -> np.ndarray:
    """Row ``k`` of ``plan`` as a command, resolved against the live state.

    Rebased the way the planner resolves row 0 (``_extractAction``), so a row-0
    command from a plan made this instant is bit-for-bit the synchronous one.
    """
    if plan.mean is None:
        # A failed solve publishes a zero command tape; the synchronous driver
        # applies its zero action too.
        return plan.seq[k].copy()
    row = plan.mean[k].astype(float)
    if mode == "absolute":
        cmd = row
    elif mode == "pos_relative":
        cmd = q_robot + row
    else:
        cmd = u_now + row
    return cmd.astype(np.float32)


class AsyncLoop:
    """The asynchronous control loop, for any number of control ticks from any state.

    One loop serves one planner; ``run`` drives a simulator from wherever it
    is, and may be called again for another stretch (each starts with no plan,
    as at an episode's start). The statistics accumulate across runs until
    ``resetStats``. The planner is not reset here: seed its mean, noise and
    goal before ``run`` as for a synchronous episode.

    Args:
        planner: The planner; its ``Plan`` is what gets timed.
        args: The driver's options: ``executor``, ``plan_latency_ms``,
            ``latency_scale``, ``async_shift``, ``warm_start`` and
            ``control_mode``.
        control_steps: Eval-simulator steps per control step.
        eval_dt: The eval simulator's timestep (s).
    """

    def __init__(self, planner, args, control_steps: int, eval_dt: float):
        self.planner = planner
        self.args = args
        self.control_steps = control_steps
        self.eval_dt = eval_dt
        self.control_dt = control_steps * eval_dt
        self.fixed = (None if args.plan_latency_ms is None
                      else int(round(args.plan_latency_ms * 1e-3 / eval_dt)))
        # Zero latency: a free-running planner would plan forever at one
        # instant, so plan once per tick instead — the synchronous driver.
        self.sync_mode = self.fixed == 0
        self.async_shift = args.async_shift and args.warm_start == "shift"
        self.resetStats()

    def resetStats(self) -> None:
        self.latencies: list[float] = []       # charged, in sim seconds
        self.plan_seconds: list[float] = []    # measured solve times
        self.staleness: list[float] = []       # age of each applied plan's state, s
        self.stale_ticks = self.held_ticks = self.exhausted_ticks = 0
        self.sim_steps = 0

    def stats(self) -> dict:
        """The statistics so far, as an episode summary keeps them."""
        lat_ms = 1e3 * np.asarray(self.latencies)
        return {
            "n_plans": len(self.latencies),
            "latency_ms_mean": float(lat_ms.mean()) if lat_ms.size else None,
            "latency_ms_std": float(lat_ms.std()) if lat_ms.size else None,
            "staleness_ms_mean": float(1e3 * np.mean(self.staleness)) if self.staleness else None,
            "stale_ticks": self.stale_ticks,
            "held_ticks": self.held_ticks,
            "tape_exhausted_ticks": self.exhausted_ticks,
            "sim_seconds": self.sim_steps * self.eval_dt,
        }

    def run(self, sim, clock, n_ticks: int, on_tick=None, recorder=None) -> str | None:
        """Drive ``sim`` for ``n_ticks`` control ticks, advancing it through ``clock``.

        Args:
            sim: The simulator being controlled.
            clock: Steps it: a ``run_episodes.EvalClock`` (which also films).
            n_ticks: Control ticks to run; the run covers ``n_ticks`` control
                steps of sim time.
            on_tick: Called as ``on_tick(i)`` at tick ``i``, before its
                command, with ``sim`` at the tick's state. Returning a string
                ends the run there, as its result; returning ``DROP`` throws
                away every plan, published and in flight, and replans from
                here (a new goal). ``None`` carries on.
            recorder: An ``EpisodeRecorder`` to report each tick to, or None.

        Returns:
            What ``on_tick`` ended the run with, or ``None`` if it ran all
            ``n_ticks``.
        """
        planner, args, eval_dt = self.planner, self.args, self.eval_dt
        control_steps = self.control_steps
        H, nu, adr = planner.horizon, planner.nu, planner.robot_qpos_adr
        planner.shift_steps = 1
        t_end = n_ticks * control_steps
        t = 0                           # eval steps since the run's first tick
        next_tick = 0
        tick = 0
        tape: _Plan | None = None       # published plan the executor plays out
        row = 0                         # tape cursor
        pending: _Plan | None = None    # finished, not yet visible
        deadline = 0                    # eval step `pending` becomes visible at
        n_lat = len(self.latencies)     # this run's plans start here

        result = None
        while t < t_end:
            if pending is not None and t >= deadline:
                tape, pending, row = pending, None, 0
            tick_due = t >= next_tick

            if tick_due and on_tick is not None:
                said = on_tick(tick)
                if said is DROP:
                    tape = pending = None
                elif said is not None:
                    result = said
                    break

            # The planner is idle: start the next solve from the state at `t`.
            # Before the tick, so under sync_mode this tick applies it.
            if pending is None and (tick_due or not self.sync_mode):
                if self.async_shift and len(self.latencies) > n_lat:
                    # The next solve starts one latency after this one; the last
                    # latency is the best estimate of it available.
                    planner.shift_steps = max(1, int(round(self.latencies[-1] / self.control_dt)))
                pending = _timedPlan(planner, sim, t)
                lat = self.fixed if self.fixed is not None else max(
                    1, int(round(pending.seconds * args.latency_scale / eval_dt)))
                deadline = t + lat
                self.latencies.append(lat * eval_dt)
                self.plan_seconds.append(pending.seconds)
                if t >= deadline:
                    tape, pending, row = pending, None, 0

            if tick_due:
                q, q_dot = sim.GetState()
                if tape is None:
                    # Nothing to play yet (startup, or a goal switch voided the
                    # plan): hold the current command.
                    self.held_ticks += 1
                    if recorder is not None:
                        recorder.recordStateAndAction(q, q_dot, sim.GetControl(), plan=_NO_PLAN)
                else:
                    age = t - tape.t0
                    k = {"tape": row, "zoh": 0, "time": age // control_steps}[args.executor]
                    if k >= H:
                        k = H - 1
                        self.exhausted_ticks += 1
                    row = k + 1
                    u = _command(tape, k, args.control_mode, np.asarray(q, dtype=float)[adr:adr + nu],
                                 np.asarray(sim.GetControl(), dtype=float))
                    first = not tape.used
                    tape.used = True
                    self.stale_ticks += not first
                    self.staleness.append(age * eval_dt)
                    if recorder is not None:
                        recorder.recordStateAndAction(
                            q, q_dot, u, tape.sigma if k == 0 else None,
                            planning_time=tape.seconds if first else None, plan=tape.record)
                    sim.SetControl(u)
                next_tick += control_steps
                tick += 1

            # On to the next event; both are strictly ahead of `t`.
            t_next = min(deadline if pending is not None else t_end, next_tick, t_end)
            if t_next <= t:
                raise RuntimeError(f"async loop failed to advance at t={t} (next_tick={next_tick}, "
                                   f"deadline={deadline}, pending={pending is not None})")
            clock.advance(t_next - t)
            t = t_next
        self.sim_steps += t
        return result


#: ``AsyncLoop.run``'s ``on_tick`` returns this to throw every plan away.
DROP = object()


def run_async_episode(args, task, eval_task, sim, planner, renderer, recorder):
    """Run one asynchronous episode, reporting it to ``recorder`` as it goes.

    Same setup, success and failure rules and recording as
    ``run_episodes.run_episode``; only the control loop differs (see the module
    docstring and ``AsyncLoop``). Every control tick is a recorded step, with
    the state it saw and the command it applied. A tick's ``planning_time`` is
    its plan's solve time on the first tick that applies the plan and NaN on
    the rest, so the planning-time summary is per plan.

    Returns:
        ``(end_reason, extra)``: why the episode ended, as ``run_episode``
        reports it, and the async statistics for its summary.
    """
    reset_mean, clock = drv.startEpisode(args, task, eval_task, sim, planner, renderer, recorder)
    loop = AsyncLoop(planner, args, drv.controlEvalSteps(task, eval_task, planner), eval_task.timestep)

    def on_tick(_tick):
        if eval_task.isFailure(sim):
            return "failure"
        if eval_task.isSuccess(sim):
            recorder.recordSuccess()
            if args.stop_on_success:
                return "success"
            # Any plan made for the old goal is wrong now: drop it, published
            # and in flight alike, and replan from here.
            drv._newGoal(task, eval_task, renderer, recorder)
            planner.Reset(reset_mean)
            return DROP
        return None

    end_reason = loop.run(sim, clock, args.steps, on_tick, recorder) or "timeout"
    clock.finish()
    return end_reason, loop.stats()


def warmUp(n: int, eval_task, sim, planner) -> list[float]:
    """``n`` throwaway plans from the initial state; their wall times, in ms.

    The planner's state, noise stream included, is put back afterwards, so the
    episodes run as they would have without them.
    """
    if n <= 0:
        return []
    eval_task.setSimToInitialState(sim)
    q, q_dot = sim.GetState()
    saved = planner.SaveState()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        planner.Plan(q, q_dot, u=sim.GetControl())
        times.append(1e3 * (time.perf_counter() - t0))
    planner.LoadState(saved)
    return times


def _printEpisode(episode: int, s: dict, args) -> None:
    drv._printEpisode(episode, s, args, first_plan_compiles=args.plan_warmup == 0)
    lat, stale = s.get("latency_ms_mean"), s.get("staleness_ms_mean")
    print(f"  async {s['n_plans']} plans, latency "
          + (f"{lat:.1f} ms" if lat is not None else "-")
          + ", staleness " + (f"{stale:.1f} ms" if stale is not None else "-")
          + f"; {s['stale_ticks']} stale, {s['held_ticks']} held, "
            f"{s['tape_exhausted_ticks']} exhausted ticks")


def main(argv=None) -> int:
    parser = build_parser()
    args = drv.parseArgs(argv, parser=parser, results_prefix="run_episodes_async")
    if args.plan_latency_ms is not None and args.plan_latency_ms < 0:
        parser.error(f"--plan-latency-ms must be >= 0, got {args.plan_latency_ms}")
    if args.latency_scale <= 0:
        parser.error(f"--latency-scale must be > 0, got {args.latency_scale}")
    if args.plan_warmup < 0:
        parser.error(f"--plan-warmup must be >= 0, got {args.plan_warmup}")

    task, eval_task = drv.buildTasks(args)
    sim = makeEvalSim(args.eval_sim, eval_task.getModelPath(), eval_task.timestep, task=eval_task)
    rollout_sim = GetContactModelSim(args.rollout_model, task.getModelPath(), N=args.n_samples,
                                     task=task, **drv.rolloutSimOverrides(args, task))
    planner = MPPI(rollout_sim, task, drv.buildPlannerConfig(args))
    recorder = EpisodeRecorder(eval_task, sim, planner, steps=StepSaving.fromArgs(args),
                               cli_args=vars(args), eval_sim=args.eval_sim, driver="async")
    drv.printHeader(args, task, eval_task, rollout_sim.config, type(rollout_sim).__name__, repr(planner))
    latency = ("measured" if args.plan_latency_ms is None else f"{args.plan_latency_ms:g} ms fixed"
               + (" (synchronous)" if args.plan_latency_ms == 0 else ""))
    if args.plan_latency_ms is None and args.latency_scale != 1.0:
        latency += f" x{args.latency_scale:g}"
    print(f"async      executor {args.executor}, latency {latency}")
    warm = warmUp(args.plan_warmup, eval_task, sim, planner)
    if warm:
        print(f"warm-up    {len(warm)} discarded plans: {', '.join(f'{m:.0f}' for m in warm)} ms")

    try:
        for episode in range(args.n_episodes):
            s = drv.runRecordedEpisode(episode, args, task, eval_task, sim, planner, recorder,
                                       episode_fn=run_async_episode)
            _printEpisode(episode, s, args)
        if args.n_episodes > 1:
            drv.printBatch(recorder)
    finally:
        # Also on an error or Ctrl-C, so the episodes already finished are kept.
        if recorder.recording:
            recorder.episodeFinished("interrupted")
        drv.saveResults(args, recorder)

    rollout_sim.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
