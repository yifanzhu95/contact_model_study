"""Asynchronous closed-loop eval episode: the sim keeps running while the planner thinks.

The synchronous driver (run_eval_episode.py) freezes the eval simulator for the
duration of controller.plan(), so planning latency is free: a contact model that
is more accurate but slower looks strictly better than a fast one. On hardware
that is not what happens. The robot keeps moving during the solve, the action
that lands is computed from a state that has already gone stale, and a planner
slower than its own control period simply misses control ticks.

This driver prices that latency in.

Clocks
------
The eval sim (Pinocchio at 2 kHz) runs far slower than real time, so the planner
and the sim cannot both track a wall clock. Instead **simulated time is the
master clock**: plan() is timed on the wall clock, and that duration is spent as
*simulated* seconds — the sim advances by exactly the planner's real latency
while the executor holds whatever it last commanded. The result is causally
identical to asynchronous MPC on hardware, but single-threaded, deterministic,
and free of the GIL contention that two Python-bound loops (the sim's substep
loop and the planner's Warp launch loop) would inflict on each other's timing.

    eval_dt     0.5 ms   master clock tick; sim integration step
    control_dt  64 ms    executor grid; spacing of the planned tape's rows
    T_plan      measured wall seconds of plan(), spent as sim seconds

The master clock counts whole eval_dt steps, so every advance is an integer
number of sim.step() calls and the grids never drift.

Executor
--------
Every control_dt the executor emits one row of the most recently published plan:

    u = qpos_measured[robot] + tape[k]      # ctrl_relative_to_qpos (default)
    u = u + tape[k]                         # otherwise

Latency is charged in full under every mode: the plan arrives late and was
solved from a stale state, which mean_staleness_ms reports. The executor choice
only decides WHICH row of that late plan gets executed.

    zoh  (default)  always the freshest plan's row 0, re-based on live qpos
    tape            contiguous playback: cursor starts at 0 on publish and
                    advances one row per tick, clamping at H-1
    time            index by elapsed sim time,
                    k = floor((t_now - tape_t0) / control_dt)

The default is `zoh`, and the reason is measured rather than assumed. Any moving
cursor produces a PHASE SAWTOOTH: it advances to row 1, then the next publish
resets it to row 0, so the command jumps by V[1]-V[0] — a whole control step of
planned motion — twice per cycle. `zoh` has no cursor and so no sawtooth, and it
still makes progress because the servo law re-reads qpos every tick.

Per-tick command discontinuity, grasp_reorient/M2/pinocchio, 25 ticks, one
episode, mean over ticks of max-over-joints |du| (sync baseline = 0.0228):

    128 ms latency    zoh 0.0190    tape 0.0533    time 0.0608

`zoh` matches the ZERO-latency synchronous baseline; both cursor modes are ~2.8x
worse. Chain-honoring (tape vs time) is only ~12% of that gap, so the sawtooth
dominates. `tape` and `time` are kept as ablations — `time` additionally skips
rows once latency >= control_dt, which breaks the chain the rollout builds (see
below) on top of the sawtooth.

The rows are a CHAIN, not independent setpoints. The rollout builds row t on top
of rows 0..t-1: _assign_ctrl_relative_kernel reads the rollout's OWN qpos after
executing them, and the accumulating variant does ctrl += V[t]. Applying row t to
a pose that never executed rows 0..t-1 therefore commands a jump toward a
configuration the robot never walked to.

Caveat: single 25-tick episodes on a task whose run-to-run spread is ~20% (see
tests/test_async_sync_equivalence.py on why grasp_reorient is not reproducible).
The zoh/cursor gap is far outside that floor; the tape-vs-time gap is not.

The planner free-runs: a new solve starts the instant the previous one finishes.

Latency
-------
--plan_latency_ms imposes a synthetic latency instead of the measured one, and
--latency_scale multiplies the measured one. Synthetic latency makes sweeps
reproducible across heterogeneous GPUs and turns latency into an independent
axis. --plan_latency_ms 0 is a special degenerate case: with zero latency a
free-running planner would spin forever at one instant, so the loop falls back
to one plan per executor tick, which reproduces the synchronous driver exactly
and exists as a regression check.

--time_constrained / --plan_budget_ms compose naturally here: they cap latency
by truncating the rollout horizon, i.e. the anytime-planner configuration.

Episode length is measured in SIM time (max_steps * control_dt seconds), not in
plan() calls. A slow planner must not be handed extra simulated time to recover
in — that would invert the very comparison this driver exists to make.

Run on a CUDA machine. Rendering is headless exactly as in run_eval_episode:

    python -m contact_study.drivers.run_async_eval_episode --task grasp_reorient
"""

from __future__ import annotations

import os
# Whatever the caller asked for, captured before the default below hides it:
# main() has to tell "the user picked a GL backend" from "we defaulted to one".
_USER_MUJOCO_GL = os.environ.get("MUJOCO_GL")
# For Drake eval, MuJoCo must not grab a GL backend (it shadows Drake's VTK GLX
# context). Default to "disable"; override to "egl" for MuJoCo-eval rendering.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time

import numpy as np
import mujoco
import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.contact_models.config import ContactModelConfig
from contact_study.evaluation import json_io
from contact_study.evaluation.metrics import EpisodeResult
from contact_study.evaluation.trajectory import (
    TrajectoryConfig, TrajectoryRecorder, add_cli_flags as add_record_flags,
)
from contact_study.planners import (
    PLANNER_NAMES, PlannerConfig, make_planner, make_planner_config,
    planner_name_for_config, resolve_planner_name,
)
from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole, EvalSimulatorKind
from contact_study.tasks.config import DEFAULT_SCENE_VARIANT
from contact_study.utils.headless_gl import configure_headless_gl

# Setup helpers are shared with the synchronous driver so the two agree on the
# rollout/eval split, the rollout timestep and the cost-weight overrides.
from contact_study.drivers.run_eval_episode import (
    MODEL_FACTORIES, RESULTS_DIR, VIDEOS_DIR,
    apply_cost_weight_overrides, apply_goal_difficulty, default_eval_sim,
)

EXECUTOR_MODES = ("zoh", "tape", "time")
DEFAULT_EXECUTOR = "zoh"


def run_async_eval_episode(
    task_name:   str,
    contact_cfg: ContactModelConfig,
    planner_cfg: PlannerConfig,
    rng:         np.random.Generator | None = None,
    geometry:    str = DEFAULT_SCENE_VARIANT,
    planner:     str | None = None,
    cost_weight_overrides: dict | None = None,
    goal_difficulty: int | None = None,
    settle_seconds: float = 0.0,
    eval_substeps: int | None = None,
    eval_sim:    EvalSimulatorKind | None = None,
    video_path:  str | None = None,
    use_mp4:     bool = True,
    ep_idx:      int  = 0,
    fin_ep_on_success: bool = True,
    # --- async knobs ------------------------------------------------------
    plan_latency_ms: float | None = None,
    latency_scale:   float = 1.0,
    plan_warmup:     int   = 2,
    executor:        str   = DEFAULT_EXECUTOR,
    async_shift:     bool  = True,
    debug:       bool = False,
    verbose:     bool = True,
    record:      TrajectoryConfig | None = None,
) -> EpisodeResult:
    """Run one asynchronous closed-loop eval episode and return an EpisodeResult.

    Same setup as run_eval_episode (ROLLOUT task + planner, EVAL task + "real"
    simulator, settle, goal sampling); only the control loop differs — see the
    module docstring.

    goal_difficulty: optional goal-difficulty level for tasks that have one
        (grasp_reorient levels 0-9); None keeps the task's own default.
    plan_latency_ms: impose this latency per plan() instead of measuring it.
        0 selects the degenerate synchronous mode. None (default) measures.
    latency_scale:   multiply the MEASURED latency (ignored when plan_latency_ms
        is given). >1 emulates a slower machine.
    plan_warmup:     throwaway plan() calls before the clock starts, so CUDA
        graph capture and Warp JIT don't land in the measured latency.
    executor:        "zoh" (default) always applies the freshest plan's row 0,
        re-based on live qpos — no cursor, so no phase sawtooth. "tape" plays
        each sequence out contiguously from row 0; "time" indexes by elapsed sim
        time and additionally skips rows. Both cursor modes measured ~2.8x more
        per-tick command discontinuity; kept as ablations. See the module
        docstring.
    async_shift:     when the planner warm-starts, shift its mean by the number
        of control steps the last solve consumed rather than by 1.
    """
    if executor not in EXECUTOR_MODES:
        raise ValueError(f"executor must be one of {EXECUTOR_MODES}, got {executor!r}")
    if plan_latency_ms is not None and plan_latency_ms < 0:
        raise ValueError(f"plan_latency_ms must be >= 0 or None, got {plan_latency_ms}")
    if latency_scale <= 0:
        raise ValueError(f"latency_scale must be > 0, got {latency_scale}")

    # The config type is authoritative; `planner` only has to agree with it.
    cfg_planner = planner_name_for_config(planner_cfg)
    planner = cfg_planner if planner is None else resolve_planner_name(planner)
    if planner != cfg_planner:
        raise ValueError(
            f"planner={planner!r} does not match the config type "
            f"{type(planner_cfg).__name__} (which selects {cfg_planner!r})"
        )
    rng = rng if rng is not None else np.random.default_rng()

    # ---- ROLLOUT task + planner ------------------------------------------
    rollout_task = get_task(task_name, geometry=geometry, role=TaskRole.ROLLOUT)
    # Before load(): this is the instance sample_new_goal runs on.
    if goal_difficulty is not None:
        apply_goal_difficulty(rollout_task, goal_difficulty)
    mjm, mjd = rollout_task.load()
    cfg = rollout_task.config
    if cost_weight_overrides:
        apply_cost_weight_overrides(rollout_task, cost_weight_overrides)

    # rollout_dt = eval_dt * eval_substeps_per_rollout, stamped onto the planning
    # model BEFORE the controller copies it to the GPU (api.put_model reads
    # mjm.opt.timestep at construction).
    eval_dt       = cfg.timestep
    eval_substeps = eval_substeps if eval_substeps is not None else cfg.eval_substeps_per_rollout
    rollout_dt    = eval_dt * eval_substeps
    mjm.opt.timestep = rollout_dt

    controller = make_planner(
        planner, task=rollout_task, cfg=contact_cfg, planner_cfg=planner_cfg, rng=rng,
    )

    # ---- EVAL task + "real" simulator ------------------------------------
    eval_task = get_task(task_name, geometry=geometry, role=TaskRole.EVAL)
    # Never samples goals; set so both instances report the same difficulty.
    if goal_difficulty is not None:
        apply_goal_difficulty(eval_task, goal_difficulty)
    eval_task.load()
    if eval_sim is not None:
        eval_task.config.eval_sim = eval_sim
    sim = eval_task.make_eval_simulator(
        video_path=video_path, render=video_path is not None, use_mp4=use_mp4
    )

    # ---- initial state ----------------------------------------------------
    q0, v0, u0 = rollout_task.get_inital_state(rng)
    sim.reset(np.asarray(q0, dtype=float), np.asarray(v0, dtype=float))
    u = np.asarray(u0, dtype=float).copy()

    if settle_seconds > 0.0:
        n_settle = int(settle_seconds / rollout_dt)
        for _ in range(n_settle):
            sim.apply_control(u)
            sim.step(eval_substeps)

    # Sample a fresh goal on the settled state BEFORE planning, so the controller
    # (which holds the in-place goal array) targets it. No-op for cart_pole.
    if hasattr(rollout_task, "sample_new_goal"):
        st = sim.get_state()
        mjd.qpos[:] = st.qpos
        mjd.qvel[:] = st.qvel
        mujoco.mj_forward(mjm, mjd)
        rollout_task.sample_new_goal(mjd, rng)

    # ---- absolute command clip (force for cart_pole; ctrlrange for hands) -
    if cfg.force_limits is not None:
        clip_lo, clip_hi = cfg.force_limits
    elif cfg.control_limits is not None:
        clip_lo, clip_hi = cfg.control_limits
    else:
        clip_lo = clip_hi = None

    # ---- the two grids, both in whole eval steps --------------------------
    control_dt    = controller.control_dt
    control_steps = controller.substeps * eval_substeps   # eval steps per control_dt
    t_end_steps   = cfg.max_steps * control_steps         # episode length in eval steps
    horizon       = controller.horizon
    relative_ctrl = controller.pc.ctrl_relative_to_qpos
    adr           = controller.robot_qpos_adr
    nu            = controller.nu

    rec = TrajectoryRecorder(
        record, controller, driver="async",
        control_dt=control_dt, rollout_dt=rollout_dt, eval_dt=eval_dt,
        eval_substeps=eval_substeps, max_steps=cfg.max_steps,
        clip=(clip_lo, clip_hi), settle_seconds=settle_seconds,
        extra_context={
            "task":            task_name,
            "geometry":        geometry,
            "goal_difficulty": getattr(rollout_task, "goal_difficulty", None),
            "planner":         planner,
            "model_label":     contact_cfg.label,
            "eval_sim":        getattr(eval_task.config.eval_sim, "value", None),
            "nq":              int(mjm.nq),
            "nv":              int(mjm.nv),
            "q0":              np.asarray(q0, dtype=float),
            "v0":              np.asarray(v0, dtype=float),
            "u0":              np.asarray(u0, dtype=float),
            "executor":        executor,
            "plan_latency_ms": plan_latency_ms,
            "latency_scale":   latency_scale,
            "async_shift":     async_shift,
            "plan_warmup":     plan_warmup,
        },
    )
    # Which of the loop's three exits was taken. Overwritten by either break in
    # the tick block; running out of episode length leaves the initial values.
    end_reason    = "timeout"
    n_steps_taken = cfg.max_steps

    # Fixed synthetic latency (in whole eval steps), or None to measure.
    fixed_lat_steps = None
    if plan_latency_ms is not None:
        fixed_lat_steps = int(round((plan_latency_ms * 1e-3) / eval_dt))
    # Degenerate zero-latency mode: a free-running planner would produce
    # infinitely many plans at one instant, so gate it to one plan per executor
    # tick. That is exactly the synchronous driver.
    sync_mode = fixed_lat_steps == 0

    if verbose:
        lat_str = ("measured" if fixed_lat_steps is None
                   else f"{plan_latency_ms:.1f}ms fixed")
        if fixed_lat_steps is None and latency_scale != 1.0:
            lat_str += f" x{latency_scale:g}"
        print(f"  [async] task={task_name}  planner={planner}  "
              f"eval_sim={eval_task.config.eval_sim.value}  "
              f"eval_dt={eval_dt*1e3:.2f}ms  rollout_dt={rollout_dt*1e3:.2f}ms  "
              f"control_dt={control_dt*1e3:.1f}ms  max_steps={cfg.max_steps}  "
              f"horizon={horizon} steps ({horizon*control_dt*1e3:.1f}ms)  "
              f"n_samples={planner_cfg.n_samples}  latency={lat_str}  "
              f"executor={executor}" + ("  [SYNC-EQUIVALENT]" if sync_mode else ""))

    def snapshot_into_mjd():
        """Mirror the eval state into the planning MjData and return it."""
        st = sim.get_state()
        mjd.qpos[:] = st.qpos
        mjd.qvel[:] = st.qvel
        mjd.ctrl[:] = u
        mujoco.mj_forward(mjm, mjd)
        return st

    def timed_plan() -> tuple[np.ndarray, float, int]:
        """One plan() call. Returns (tape, measured_ms, charged latency in eval steps)."""
        t0 = time.perf_counter()
        controller.plan(mjd)
        measured_ms = (time.perf_counter() - t0) * 1e3
        if fixed_lat_steps is not None:
            lat_steps = fixed_lat_steps
        else:
            lat_steps = max(1, int(round((measured_ms * 1e-3 * latency_scale) / eval_dt)))
        # last_action_seq is captured pre-shift inside _extract_action, so it is
        # the sequence that starts at the state this solve was given.
        tape = controller.last_action_seq
        if tape is None:                       # planner never populated one
            tape = np.zeros((horizon, nu), dtype=np.float32)
        return tape, measured_ms, lat_steps

    # ---- warm-up ----------------------------------------------------------
    # The first plan() calls pay CUDA graph capture and Warp JIT and are wildly
    # unrepresentative of steady-state latency, which here feeds straight back
    # into the dynamics. Burn them, then reset so the episode starts on a clean
    # mean exactly like the synchronous driver does.
    if plan_warmup > 0:
        snapshot_into_mjd()
        warmup_ms = []
        for _ in range(plan_warmup):
            t0 = time.perf_counter()
            controller.plan(mjd)
            warmup_ms.append((time.perf_counter() - t0) * 1e3)
        controller.reset()
        if verbose:
            print(f"  [async] warm-up: {plan_warmup} discarded plan(s), "
                  f"{', '.join(f'{m:.1f}' for m in warmup_ms)} ms")

    # ---- async control loop -----------------------------------------------
    t          = 0            # sim time, in whole eval steps
    next_tick  = 0            # next executor tick, in eval steps
    tape       = None         # published plan currently being played out
    tape_t0    = 0            # eval step at which that plan's solve STARTED
    tape_id    = -1           # monotonic id, to spot ticks with no fresh plan
    tape_row   = 0            # contiguous playback cursor into `tape`
    prev_tape_id = -2         # tape_id served at the previous tick
    pending    = None         # finished plan not yet visible to the executor
    pending_t0 = 0
    pending_id = -1
    deadline   = 0            # eval step at which `pending` becomes visible
    n_plans    = 0

    step_times:  list[float] = []   # raw measured plan() ms
    latency_ms:  list[float] = []   # charged latency, in ms of sim time
    staleness:   list[float] = []   # age of the applied command's state estimate
    missed_ticks = 0
    tape_exhausted_ticks = 0
    steps_to_success: int | None = None
    tick_idx = 0

    ep_start = time.perf_counter()

    while t < t_end_steps:
        # 1. A finished plan becomes visible.
        if pending is not None and t >= deadline:
            tape, tape_t0, tape_id = pending, pending_t0, pending_id
            tape_row = 0      # a fresh chain always starts at its first step
            pending = None

        tick_due = t >= next_tick
        # In sync_mode (zero charged latency) the planner must not free-run: it
        # would produce unboundedly many plans at a single instant. Gate it to
        # the tick that consumes its output, which is precisely the synchronous
        # driver's plan-then-apply order.
        plan_due = pending is None and (tick_due or not sync_mode)

        # One state read serves both the solve and the tick at this instant.
        st = snapshot_into_mjd() if (tick_due or plan_due) else None

        # 2. Planner is idle -> start the next solve from the state at `t`.
        #    Ordered before the tick so that in sync_mode the plan computed here
        #    is published (step 3) in time for this same tick to consume it.
        if plan_due:
            if controller.pc.warm_start and async_shift and latency_ms:
                # Align the carried-over mean with the next solve's t0. The next
                # solve starts one latency after this one, so the previous
                # measurement is the best estimate available at this point.
                controller.shift_steps = max(
                    1, int(round(latency_ms[-1] / (control_dt * 1e3)))
                )
            pending_t0 = t
            pending_id = n_plans
            pending, measured_ms, lat_steps = timed_plan()
            step_times.append(measured_ms)
            latency_ms.append(lat_steps * eval_dt * 1e3)
            deadline = t + lat_steps
            # OUTSIDE timed_plan(): this loop spends the measured plan_ms as
            # simulated seconds, so recorder work inside that region would change
            # the episode's dynamics, not merely its cost.
            rec.plan_event(plan_idx=n_plans, t_start=t * eval_dt,
                           t_visible=deadline * eval_dt, plan_ms=measured_ms,
                           latency_ms=lat_steps * eval_dt * 1e3)
            n_plans += 1

            # 3. Zero-latency plans land at the instant they started.
            if t >= deadline:
                tape, tape_t0, tape_id = pending, pending_t0, pending_id
                tape_row = 0
                pending = None

        # 4. Executor tick: evaluate the task and emit a command.
        if tick_due:
            if rollout_task.is_success(mjd):
                if steps_to_success is None:
                    steps_to_success = tick_idx
                    if debug:
                        print(f"  [ep {ep_idx:02d}] first success at tick {tick_idx}")
                if fin_ep_on_success:
                    end_reason, n_steps_taken = "success", tick_idx
                    break
                elif hasattr(rollout_task, "sample_new_goal"):
                    # Multi-goal mode. Any plan built for the old goal is now
                    # wrong, so drop it (published and in flight alike) and hold
                    # the last command until a fresh one lands — what a real
                    # system does when its target changes.
                    rollout_task.sample_new_goal(mjd, rng)
                    controller.reset()
                    tape = pending = None
                    rec.goal_switch(tick_idx)
            if rollout_task.has_failed(mjd):
                if verbose:
                    print(f"  tick {tick_idx:4d}: task failed")
                end_reason, n_steps_taken = "failed", tick_idx
                break

            if tape is None:
                # Nothing to execute yet (startup, or a goal switch just voided
                # the plan): hold the last command.
                missed_ticks += 1
                rec.tick(step=tick_idx, t=t * eval_dt, qpos=st.qpos, qvel=st.qvel,
                         action=None, ctrl=u, tape_id=-1, tape_row=-1,
                         staleness_ms=None, applied=False)
            else:
                age_steps = t - tape_t0
                if executor == "zoh":
                    k = 0
                elif executor == "time":
                    # Ablation: index by elapsed time. Skips the rows that
                    # expired during the solve, which breaks the delta chain.
                    k = age_steps // control_steps
                else:
                    # Contiguous playback: walk the chain one row per tick.
                    k = tape_row
                if k >= horizon:
                    k = horizon - 1
                    tape_exhausted_ticks += 1
                tape_row = k + 1
                row = tape[k]
                u = (st.qpos[adr : adr + nu] + row) if relative_ctrl else (u + row)
                if clip_lo is not None:
                    u = np.clip(u, clip_lo, clip_hi)
                # After the clip, before the apply: ctrl is exactly what the sim
                # gets, and action is the tape row it was built from — tape_id
                # indexes trajectory["plans"]["plan_idx"].
                rec.tick(step=tick_idx, t=t * eval_dt, qpos=st.qpos, qvel=st.qvel,
                         action=row, ctrl=u, tape_id=tape_id, tape_row=k,
                         staleness_ms=age_steps * eval_dt * 1e3, applied=True)
                sim.apply_control(u)
                staleness.append(age_steps * eval_dt * 1e3)
                if tape_id == prev_tape_id:
                    missed_ticks += 1
                prev_tape_id = tape_id

            if debug and tick_idx % 10 == 0:
                print(f"  [ep {ep_idx:02d} | tick {tick_idx:04d}]  "
                      f"t={t*eval_dt:6.3f}s  qpos_norm={np.linalg.norm(st.qpos):.4f}  "
                      f"qvel_norm={np.linalg.norm(st.qvel):.4f}  u[0]={float(u[0]):+8.3f}")
            elif verbose and tick_idx % 25 == 0:
                print(f"  tick {tick_idx:4d}  t={t*eval_dt:5.2f}s  u[0]={float(u[0]):+8.3f}")

            next_tick += control_steps
            tick_idx  += 1

        # 5. Advance the sim to whichever event comes first. Both candidates are
        #    strictly ahead of `t`: step 1 already published anything due, and
        #    step 4 already pushed next_tick past `t`.
        t_next = min(deadline if pending is not None else t_end_steps,
                     next_tick, t_end_steps)
        if t_next <= t:      # unreachable; guards against a silent spin
            raise RuntimeError(
                f"async scheduler failed to advance at t={t} "
                f"(next_tick={next_tick}, deadline={deadline}, pending={pending is not None})"
            )
        sim.step(t_next - t)
        t = t_next
    else:
        # Ran out of episode length rather than breaking: tick_idx is the number
        # of ticks completed. (Missed ticks count as ticks.)
        n_steps_taken = tick_idx

    elapsed = time.perf_counter() - ep_start

    if video_path is not None:
        written = sim.save_video(video_path)
        if verbose:
            print(f"  Saved video -> {written}")

    final_qpos = sim.get_state().qpos
    step_arr = np.asarray(step_times)
    lat_arr  = np.asarray(latency_ms)
    # Multi-goal mode (fin_ep_on_success=False) never breaks on success, so it
    # always exits by exhaustion: time_out stays True, but the episode succeeded.
    time_out = end_reason == "timeout"
    if time_out and steps_to_success is not None:
        end_reason = "success"

    return EpisodeResult(
        task_name        = cfg.name,
        model_label      = contact_cfg.label,
        success          = steps_to_success is not None,
        steps_to_success = steps_to_success,
        final_cost       = float(np.linalg.norm(final_qpos - np.asarray(q0, dtype=float))),
        n_samples_used   = planner_cfg.n_samples,
        planner          = planner,
        elapsed_seconds  = elapsed,
        mean_step_ms     = float(step_arr.mean()) if len(step_arr) else 0.0,
        std_step_ms      = float(step_arr.std())  if len(step_arr) else 0.0,
        n_plans          = n_plans,
        mean_latency_ms  = float(lat_arr.mean()) if len(lat_arr) else 0.0,
        std_latency_ms   = float(lat_arr.std())  if len(lat_arr) else 0.0,
        mean_staleness_ms    = float(np.mean(staleness)) if staleness else 0.0,
        missed_ticks         = missed_ticks,
        tape_exhausted_ticks = tape_exhausted_ticks,
        sim_seconds          = t * eval_dt,
        time_out             = time_out,
        end_reason           = end_reason,
        n_steps_taken        = n_steps_taken,
        trajectory           = rec.finish(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task",        type=str,   default="cart_pole")
    p.add_argument("--geometry",    type=str,   default=DEFAULT_SCENE_VARIANT,
                   help="Scene variant: '<object>' or "
                        "'<object>_<hand_acc>_<obj_acc>' (e.g. duck_low_high). "
                        "Legacy geometry names map to the default scene.")
    p.add_argument("--model",       type=str,   default="M2", choices=list(MODEL_FACTORIES))
    p.add_argument("--planner",     type=str,   default="mppi", choices=PLANNER_NAMES,
                   help="Sampling planner: mppi (softmax-weighted mean), cem (elite "
                        "refit) or predictive_sampler/ps (greedy best-of-N).")
    p.add_argument("--n_samples",   type=int,   default=256)
    p.add_argument("--horizon",     type=int,   default=None,
                   help="Planning horizon in control steps (ignored when "
                        "--time_horizon is given).")
    p.add_argument("--time_horizon", type=float, default=0.352,
                   help="Planning horizon in SECONDS; quantized down to whole "
                        "control steps. Overrides --horizon.")
    p.add_argument("--step_time",   type=float, default=0.064,
                   help="Control-step duration in SECONDS; quantized down to whole "
                        "rollout steps. Overrides --substeps.")
    p.add_argument("--n_iterations", type=int,  default=None,
                   help="Optimizer iterations per plan() call (default: the "
                        "planner's own — 1 for mppi/predictive_sampler, 3 for cem).")
    p.add_argument("--noise_sigma", type=float, default=0.2)
    p.add_argument("--delta",       type=float, default=None,
                   help="Per-step delta clip magnitude (action units); "
                        "omit to disable the delta clamp entirely.")
    p.add_argument("--substeps",    type=int,   default=None,
                   help="Rollout substeps per control step (control frequency knob).")
    p.add_argument("--eval_substeps", type=int, default=None,
                   help="Eval steps per rollout step (default: task config).")
    p.add_argument("--warm_start", action=argparse.BooleanOptionalAction, default=False,
                   help="Shift the planned sequence forward after each plan(); "
                        "--no-warm_start (default) keeps the running mean in "
                        "place, matching irisim_warp. See --async_shift for how "
                        "far it shifts here.")
    p.add_argument("--resample_interval", type=int, default=1,
                   help="Plan steps between noise resamples (1=every step).")
    # --- async-specific ----------------------------------------------------
    p.add_argument("--plan_latency_ms", type=float, default=None,
                   help="Impose this planning latency (ms of SIM time) per plan() "
                        "instead of measuring it — reproducible across GPUs and "
                        "sweepable as an independent axis. 0 selects the "
                        "degenerate synchronous-equivalent mode. Default: measure.")
    p.add_argument("--latency_scale", type=float, default=1.0,
                   help="Multiply the MEASURED latency (ignored with "
                        "--plan_latency_ms). >1 emulates a slower machine.")
    p.add_argument("--plan_warmup", type=int, default=2,
                   help="Throwaway plan() calls before the clock starts, so CUDA "
                        "graph capture and Warp JIT stay out of the measured "
                        "latency. Pass 0 to compare bit-exactly against the "
                        "synchronous driver (which has no warm-up).")
    p.add_argument("--executor", type=str, default=DEFAULT_EXECUTOR,
                   choices=list(EXECUTOR_MODES),
                   help="zoh (default): always the freshest plan's row 0, "
                        "re-based on live qpos. tape: contiguous playback, one "
                        "row per tick. time: index by elapsed sim time (also "
                        "skips rows). Both cursor modes measured ~2.8x more "
                        "per-tick command jump than zoh — see the module docstring.")
    p.add_argument("--async_shift", action=argparse.BooleanOptionalAction, default=True,
                   help="With --warm_start, shift the mean by the number of control "
                        "steps the last solve consumed instead of by 1.")
    # --- MPPI-only ---------------------------------------------------------
    p.add_argument("--temperature", type=float, default=20.0)
    # --- CEM-only ----------------------------------------------------------
    p.add_argument("--n_elites",    type=int,   default=None,
                   help="CEM elite-set size; overrides --elite_frac when set.")
    p.add_argument("--elite_frac",  type=float, default=None,
                   help="CEM elite fraction of --n_samples (default 0.1).")
    p.add_argument("--cem_alpha",   type=float, default=None,
                   help="CEM refit smoothing: new = alpha*old + (1-alpha)*fit.")
    p.add_argument("--min_sigma",   type=float, default=None,
                   help="CEM floor on the refit sigma (default 1e-3).")
    # --- predictive-sampler-only ------------------------------------------
    p.add_argument("--include_nominal", action=argparse.BooleanOptionalAction, default=True,
                   help="Predictive sampler: keep sample 0 unperturbed.")
    # ----------------------------------------------------------------------
    p.add_argument("--time_constrained", action=argparse.BooleanOptionalAction, default=False,
                   help="Stop rollouts once --plan_budget_ms elapses (capped at the "
                        "horizon). Combined with this driver, that is the anytime "
                        "planner: latency is capped by truncating the horizon.")
    p.add_argument("--plan_budget_ms", type=float, default=None,
                   help="Wall-clock rollout budget per plan() in ms; required with "
                        "--time_constrained.")
    p.add_argument("--eval_sim",    type=str,   default="none",
                   choices=["none", "mujoco", "drake", "pinocchio"],
                   help="Eval simulator: 'none' uses the task default, else override it.")
    p.add_argument("--settle",      type=float, default=1.0)
    p.add_argument("--goal_difficulty", type=int, default=None,
                   help="Goal-difficulty level for tasks that have one "
                        "(grasp_reorient levels 0-9; see run_eval_episode.py). "
                        "Default: the task's own.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--n_episodes",  type=int,   default=1,
                   help="Number of episodes to run; reports the aggregate success rate.")
    p.add_argument("--weights",     nargs="+", default=[],
                   help="Cost-weight overrides as name=value tokens "
                        "(e.g. --weights w_quat=50 w_pos_x=400).")
    p.add_argument("--video",       type=str,   default=None,
                   help="Video path (auto-named for a single episode if omitted).")
    p.add_argument("--video_format", type=str,  default="mp4", choices=["mp4", "gif"],
                   help="Video container; overrides --video's extension.")
    p.add_argument("--results",     type=str,   default=None,
                   help="JSON path for the episode result(s) (auto-named if omitted).")
    add_record_flags(p)
    p.add_argument("--debug",       action="store_true",
                   help="Verbose per-tick diagnostics (also enables planner debug).")
    args = p.parse_args()
    record_cfg = TrajectoryConfig.from_args(args)

    planner = resolve_planner_name(args.planner)
    seed_seq = np.random.SeedSequence(args.seed)
    episode_seeds = seed_seq.spawn(args.n_episodes)

    want_video = args.video is not None or args.n_episodes == 1
    use_mp4 = args.video_format == "mp4"
    base_video_path = args.video
    if base_video_path is None and want_video:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        base_video_path = str(
            VIDEOS_DIR / f"{args.task}_{planner}_async.{args.video_format}"
        )

    contact_cfg = MODEL_FACTORIES[args.model]()
    eval_sim = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)

    # Offscreen rendering needs an EGL context bound to the GPU; the dynamic
    # loader reads that env at exec time, so this may restart the process —
    # hence before wp.init(), and only when there is actually a video. Drake is
    # excluded: its VTK renderer wants the real (xvfb) display this would take.
    if want_video:
        kind = eval_sim if eval_sim is not None else default_eval_sim(args.task, args.geometry)
        if kind is not EvalSimulatorKind.DRAKE:
            configure_headless_gl(
                _USER_MUJOCO_GL or ("egl" if kind is EvalSimulatorKind.MUJOCO else "disable")
            )

    wp.init()

    overrides: dict = {}
    for tok in args.weights:
        if "=" not in tok:
            raise ValueError(f"bad --weights token {tok!r}; expected name=value")
        name, val = tok.split("=", 1)
        overrides[name.strip()] = float(val)

    results: list[EpisodeResult] = []
    for ep_idx in range(args.n_episodes):
        ep_seed = int(episode_seeds[ep_idx].generate_state(1)[0])
        rng = np.random.default_rng(episode_seeds[ep_idx])

        if want_video:
            if args.n_episodes == 1:
                video_path = base_video_path
            else:
                stem, suffix = base_video_path.rsplit(".", 1)
                video_path = f"{stem}_ep{ep_idx:03d}.{suffix}"
        else:
            video_path = None

        delta = (-args.delta, args.delta) if args.delta is not None else (None, None)

        # One flat knob set for every planner; make_planner_config keeps only the
        # fields the selected planner declares.
        planner_kwargs = dict(
            n_samples      = args.n_samples,
            step_horizon   = args.horizon,
            time_horizon   = args.time_horizon,
            step_time      = args.step_time,
            noise_sigma    = args.noise_sigma,
            step_substeps  = args.substeps,
            warm_start     = args.warm_start,
            use_full_graph = not args.time_constrained,
            delta_range    = delta,
            nconmax        = 50,
            njmax          = 300,
            seed           = ep_seed,
            debug          = args.debug,
            resample_interval = args.resample_interval,
            time_constrained  = args.time_constrained,
            plan_budget_ms    = args.plan_budget_ms,
            temperature       = args.temperature,
            include_nominal   = args.include_nominal,
        )
        for name, val in (("n_iterations", args.n_iterations),
                          ("n_elites",     args.n_elites),
                          ("elite_frac",   args.elite_frac),
                          ("alpha",        args.cem_alpha),
                          ("min_sigma",    args.min_sigma)):
            if val is not None:
                planner_kwargs[name] = val

        planner_cfg = make_planner_config(planner, **planner_kwargs)

        result = run_async_eval_episode(
            task_name   = args.task,
            geometry    = args.geometry,
            contact_cfg = contact_cfg,
            planner     = planner,
            planner_cfg = planner_cfg,
            rng         = rng,
            video_path  = video_path,
            use_mp4     = use_mp4,
            cost_weight_overrides = overrides or None,
            goal_difficulty = args.goal_difficulty,
            settle_seconds = args.settle,
            eval_substeps  = args.eval_substeps,
            eval_sim       = eval_sim,
            ep_idx         = ep_idx,
            plan_latency_ms = args.plan_latency_ms,
            latency_scale   = args.latency_scale,
            plan_warmup     = args.plan_warmup,
            executor        = args.executor,
            async_shift     = args.async_shift,
            debug          = args.debug,
            verbose        = args.debug or args.n_episodes == 1,
            fin_ep_on_success = True,
            record         = record_cfg,
        )
        results.append(result)

        label = "✓" if result.success else "✗"
        sstr  = f"tick {result.steps_to_success}" if result.steps_to_success is not None else "—"
        print(f"  [ep {ep_idx:03d}] {label}  success_tick={sstr}  "
              f"final_cost={result.final_cost:.4f}  "
              f"plan={result.mean_step_ms:.2f}±{result.std_step_ms:.2f} ms  "
              f"latency={result.mean_latency_ms:.2f} ms  "
              f"stale={result.mean_staleness_ms:.2f} ms  "
              f"n_plans={result.n_plans}  missed={result.missed_ticks}  "
              f"exhausted={result.tape_exhausted_ticks}")

    # ---- aggregate + save -------------------------------------------------
    n_success = sum(r.success for r in results)
    success_rate = n_success / len(results)
    mean_step_ms   = float(np.mean([r.mean_step_ms for r in results]))
    mean_latency   = float(np.mean([r.mean_latency_ms for r in results]))
    mean_staleness = float(np.mean([r.mean_staleness_ms for r in results]))
    total_missed   = int(sum(r.missed_ticks for r in results))
    total_exhausted = int(sum(r.tape_exhausted_ticks for r in results))

    print(f"\n{'='*70}")
    print(f"  ASYNC  task={args.task}  model={args.model}  planner={planner}  "
          f"n_episodes={args.n_episodes}")
    print(f"  success_rate={success_rate:.3f}  ({n_success}/{len(results)})")
    print(f"  mean_step_ms={mean_step_ms:.3f}  (raw plan cost)")
    print(f"  mean_latency_ms={mean_latency:.3f}  (charged against the dynamics)")
    print(f"  mean_staleness_ms={mean_staleness:.3f}  "
          f"missed_ticks={total_missed}  tape_exhausted={total_exhausted}")
    print(f"{'='*70}")

    results_path = args.results
    if results_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = str(
            RESULTS_DIR / f"{args.task}_{args.model}_{planner}_async.json"
        )
    json_io.dump({
        "task":            args.task,
        "model":           args.model,
        "planner":         planner,
        "driver":          "async",
        "n_episodes":      args.n_episodes,
        "success_rate":    success_rate,
        "mean_step_ms":    mean_step_ms,
        "mean_latency_ms": mean_latency,
        "mean_staleness_ms": mean_staleness,
        "missed_ticks":    total_missed,
        "tape_exhausted_ticks": total_exhausted,
        "async_config": {
            "plan_latency_ms": args.plan_latency_ms,
            "latency_scale":   args.latency_scale,
            "plan_warmup":     args.plan_warmup,
            "executor":        args.executor,
            "async_shift":     args.async_shift,
        },
        "episodes":        [r.to_dict() for r in results],
    }, results_path, precision=record_cfg.precision)
    print(f"  Saved result(s) -> {results_path}")

    return results


if __name__ == "__main__":
    main()
