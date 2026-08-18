"""Closed-loop eval episode: pluggable "real" simulator + GPU sampling planner.

Generalizes tests/test_drake.py. Two task instances share the same TaskConfig:

  * a ROLLOUT task — owns the planning MuJoCo model + cost arrays; handed to the
    planner (which still owns the batched GPU rollout env).
  * an EVAL task   — owns the high-fidelity "real" environment (an EvalSimulator,
    MuJoCo or Drake) that this loop actually steps.

Per control step: read eval state → mirror into the planning MjData →
controller.plan() on the GPU → integrate the delta into the command → apply to
the eval simulator → advance it → render.

The planner is selectable (--planner): MPPI, CEM or the predictive sampler. All
three share one rollout engine and differ only in how the sampled costs update
the control distribution, so they are directly comparable — see
contact_study/planners/__init__.py.

The eval ("real") simulator runs at a finer timestep than the rollout/planning
model: rollout_dt = eval_dt * eval_substeps_per_rollout (the eval sim takes that
many steps per rollout step). The rollout step is inferred from the eval step and
stamped onto the planning model. Control frequency stays a separate knob — one
control step spans the planner's substeps rollout steps (control_dt = substeps *
rollout_dt), so the eval sim advances substeps * eval_substeps_per_rollout steps.
The substep count and the horizon come from the planner config either as step
counts (step_substeps / step_horizon) or as durations (step_time /
time_horizon), which the controller quantizes against rollout_dt.

The contact model (M1..M4) used for rollouts is orthogonal to the eval simulator
(set by the task's TaskConfig.eval_sim).

Run on a CUDA machine (warp arrays live on the device). For Drake eval, its VTK
renderer needs a display, so run headless under xvfb:
    xvfb-run -a python -m contact_study.drivers.run_eval_episode --task cart_pole
"""

from __future__ import annotations

import os
# For Drake eval, MuJoCo must not grab a GL backend (it shadows Drake's VTK GLX
# context). Default to "disable"; override to "egl" for MuJoCo-eval rendering.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import mujoco
import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.contact_models.config import ContactModelConfig
from contact_study.evaluation.metrics import EpisodeResult
from contact_study.planners import (
    PLANNER_NAMES, PlannerConfig, make_planner, make_planner_config,
    planner_name_for_config, resolve_planner_name,
)
from contact_study.planners.mppi import MPPIController, MPPIConfig  # noqa: F401 — re-export
from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole, EvalSimulatorKind
from contact_study.tasks.config import DEFAULT_SCENE_VARIANT

#wp.init()

MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
}

VIDEOS_DIR  = Path(__file__).parents[2] / "videos"
RESULTS_DIR = Path(__file__).parents[2] / "results"


def load_rollout_task(task_name: str, geometry: str = DEFAULT_SCENE_VARIANT):
    """Load a task's ROLLOUT instance — handy for peeking nq/nv/nu/cost_weights
    before a sweep (the episode runner builds its own rollout + eval tasks)."""
    task = get_task(task_name, geometry=geometry, role=TaskRole.ROLLOUT)
    task.load()
    return task


def rollout_dt_for(task_config, eval_substeps: int | None = None) -> float:
    """The planning-model timestep this driver stamps onto mjm before planning.

    rollout_dt = eval_dt * eval_substeps_per_rollout. Exposed so sweep workers can
    resolve a planner config's time-based schedule (PlannerConfig.resolve_schedule)
    into step counts for labelling/logging *before* an episode runs, and get exactly
    the same numbers the controller will resolve internally.
    """
    n = eval_substeps if eval_substeps is not None else task_config.eval_substeps_per_rollout
    return task_config.timestep * n


def resolve_mppi_schedule(mppi_cfg: PlannerConfig, task_config,
                          eval_substeps: int | None = None) -> tuple[int, int, float]:
    """Resolve (horizon, substeps, rollout_dt) for a config against a task."""
    dt = rollout_dt_for(task_config, eval_substeps)
    horizon, substeps = mppi_cfg.resolve_schedule(dt)
    return horizon, substeps, dt


def apply_cost_weight_overrides(task, overrides: dict) -> None:
    """Merge cost-weight overrides into the (loaded) task and rebuild weights_wp.

    The dict key order of cost_weights must match the array order used in the
    task's initialize_task — this holds for all built-in tasks. Must be called
    before the planner is built (it captures task.cost_weights_wp)."""
    weights = dict(task.config.cost_weights)
    weights.update(overrides)
    weights_arr = np.array([weights[k] for k in task.config.cost_weights], dtype=np.float32)
    task.weights_wp = wp.array(weights_arr, dtype=wp.float32, device="cuda")


def run_eval_episode(
    task_name:   str,
    contact_cfg: ContactModelConfig,
    planner_cfg: PlannerConfig | None = None,
    rng:         np.random.Generator | None = None,
    geometry:    str = DEFAULT_SCENE_VARIANT,
    planner:     str | None = None,
    cost_weight_overrides: dict | None = None,
    settle_seconds: float = 0.0,
    eval_substeps: int | None = None,
    eval_sim:    EvalSimulatorKind | None = None,
    condition:   str  = "B",
    video_path:  str | None = None,
    use_mp4:     bool = True,
    ep_idx:      int  = 0,
    fin_ep_on_success: bool = True,
    debug:       bool = False,
    verbose:     bool = True,
    mppi_cfg:    PlannerConfig | None = None,
) -> EpisodeResult:
    """Run one closed-loop eval episode and return an EpisodeResult.

    Drop-in replacement for the legacy experiments/run_episode.py `run_episode`,
    but built on the eval/rollout split: a ROLLOUT task (planning MuJoCo model +
    cost arrays for the planner) and an EVAL task (the pluggable "real"
    EvalSimulator, MuJoCo or Drake). condition is a label only — only the
    warm-started planner ("B") path exists; the legacy fixed-budget rollout
    ("A") path is gone.

    planner_cfg picks the planner: an MPPIConfig runs MPPI, a CEMConfig runs CEM,
    a PredictiveSamplerConfig runs the predictive sampler. `planner` may name it
    explicitly (and is then checked against the config type). `mppi_cfg` is the
    old name of `planner_cfg`, still accepted so existing sweep workers keep
    working unchanged.

    The episode length is the task's TaskConfig.max_steps. mean_step_ms /
    std_step_ms hold per-control-step planning latency (controller.plan() only,
    excluding the eval-sim advance) in ms.

    cost_weight_overrides: optional {weight_name: value} merged into the rollout
        task's cost weights before planning (used by the weight grid search).
    fin_ep_on_success: stop at first success (default); if False, resample a new
        goal on each success and keep going (multi-goal mode).
    """
    planner_cfg = planner_cfg if planner_cfg is not None else mppi_cfg
    if planner_cfg is None:
        raise ValueError("run_eval_episode needs a planner_cfg (or the legacy mppi_cfg)")
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
    mjm, mjd = rollout_task.load()
    cfg = rollout_task.config
    if cost_weight_overrides:
        apply_cost_weight_overrides(rollout_task, cost_weight_overrides)

    # The eval ("real") simulator runs at the fine cfg.timestep; the rollout /
    # planning model runs coarser. Infer the rollout step from the eval step and
    # stamp it onto the planning model BEFORE the controller copies the model to
    # the GPU (api.put_model reads mjm.opt.timestep at construction):
    #   rollout_dt = eval_dt * eval_substeps_per_rollout
    eval_dt    = cfg.timestep
    eval_substeps = eval_substeps if eval_substeps is not None else cfg.eval_substeps_per_rollout
    rollout_dt = eval_dt * eval_substeps
    mjm.opt.timestep = rollout_dt

    controller = make_planner(
        planner, task=rollout_task, cfg=contact_cfg, planner_cfg=planner_cfg, rng=rng,
    )

    # ---- EVAL task + "real" simulator ------------------------------------
    eval_task = get_task(task_name, geometry=geometry, role=TaskRole.EVAL)
    eval_task.load()
    # eval_sim=None keeps the task's TaskConfig default; otherwise override it.
    if eval_sim is not None:
        eval_task.config.eval_sim = eval_sim
    # Only stand up a renderer when a video is requested (avoids a GL context per
    # episode during headless sweeps).
    sim = eval_task.make_eval_simulator(
        video_path=video_path, render=video_path is not None, use_mp4=use_mp4
    )

    # ---- initial state ----------------------------------------------------
    q0, v0, u0 = rollout_task.get_inital_state(rng)
    sim.reset(np.asarray(q0, dtype=float), np.asarray(v0, dtype=float))
    u = np.asarray(u0, dtype=float).copy()

    # Settle (objects fall to rest / hand closes onto the object). Hold the
    # initial command each rollout step so a position-controlled hand doesn't go
    # limp; advance the eval sim a full rollout step (eval_substeps fine steps).
    # This phase IS recorded: the eval sims capture frames on their own sim clock
    # inside step(), so the video opens with the settle just like Drake's has.
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

    # One control step = controller.substeps rollout steps. The eval sim covers
    # the same wall-clock with `eval_substeps` finer steps per rollout step. The
    # episode runs for the task's configured max_steps control steps. Read the
    # step counts off the controller: it resolved them from the config, which may
    # have specified them as durations (step_time / time_horizon).
    control_dt      = controller.control_dt
    eval_steps_per_control = controller.substeps * eval_substeps
    n_steps         = cfg.max_steps
    steps_to_success: int | None = None

    if verbose:
        print(f"  task={task_name}  planner={planner}  "
              f"eval_sim={eval_task.config.eval_sim.value}  "
              f"eval_dt={eval_dt*1e3:.2f}ms  rollout_dt={rollout_dt*1e3:.2f}ms  "
              f"control_dt={control_dt*1e3:.1f}ms  max_steps={n_steps}  "
              f"horizon={controller.horizon} steps "
              f"({controller.horizon*control_dt*1e3:.1f}ms)  "
              f"n_samples={planner_cfg.n_samples}  "
              f"n_iterations={planner_cfg.n_iterations}")

    step_times: list[float] = []
    ep_start = time.perf_counter()

    for t in range(n_steps):
        # 1. Read eval state (MuJoCo-ordered) and mirror into the planning MjData.
        st = sim.get_state()
        mjd.qpos[:] = st.qpos
        mjd.qvel[:] = st.qvel
        mjd.ctrl[:] = u
        mujoco.mj_forward(mjm, mjd)

        # Success/failure are evaluated on the current (start-of-step) state.
        if rollout_task.is_success(mjd):
            if steps_to_success is None:
                steps_to_success = t
                if debug:
                    print(f"  [ep {ep_idx:02d}] first success at step {t}")
            if fin_ep_on_success:
                break
            elif hasattr(rollout_task, "sample_new_goal"):
                # Multi-goal mode: target a fresh goal and keep going. reset()
                # also restores whatever adaptive state the planner carries
                # (MPPI's temperature, CEM's sigma).
                rollout_task.sample_new_goal(mjd, rng)
                controller.reset()
        if rollout_task.has_failed(mjd):
            if verbose:
                print(f"  step {t:4d}: task failed")
            break

        # 2. Plan on the GPU and turn the planned delta into the absolute command.
        # step_times measures only this plan() call, not the eval-sim advance below.
        plan_start = time.perf_counter()
        action = controller.plan(mjd)
        step_times.append((time.perf_counter() - plan_start) * 1e3)
        if controller.pc.ctrl_relative_to_qpos:
            # Servo parameterization (mirrors the rollout): command the current
            # measured robot joint qpos plus the planned delta, re-read each step,
            # instead of accumulating the delta onto the running command.
            adr = controller.robot_qpos_adr
            u = st.qpos[adr : adr + controller.nu] + action
        else:
            u = u + action
        if clip_lo is not None:
            u = np.clip(u, clip_lo, clip_hi)

        # 3. Apply and advance the eval sim (finer steps over the same control_dt).
        #    Video frames are captured inside step(), on the sim clock at cam_fps,
        #    so the recording plays back in real time at any control frequency.
        sim.apply_control(u)
        sim.step(eval_steps_per_control)

        if debug and t % 10 == 0:
            print(f"  [ep {ep_idx:02d} | step {t:04d}]  qpos_norm={np.linalg.norm(st.qpos):.4f}  "
                  f"qvel_norm={np.linalg.norm(st.qvel):.4f}  u[0]={float(u[0]):+8.3f}")
        elif verbose and t % 25 == 0:
            print(f"  step {t:4d}  t={t*control_dt:5.2f}s  u[0]={float(u[0]):+8.3f}")

    elapsed = time.perf_counter() - ep_start

    if video_path is not None:
        # save_video returns the path actually written (the container comes from
        # use_mp4, so the extension may differ from video_path's).
        written = sim.save_video(video_path)
        if verbose:
            print(f"  Saved video -> {written}")

    final_qpos = sim.get_state().qpos
    step_arr = np.asarray(step_times)
    return EpisodeResult(
        task_name        = cfg.name,
        model_label      = contact_cfg.label,
        condition        = condition,
        success          = steps_to_success is not None,
        steps_to_success = steps_to_success,
        final_cost       = float(np.linalg.norm(final_qpos - np.asarray(q0, dtype=float))),
        n_samples_used   = planner_cfg.n_samples,
        planner          = planner,
        elapsed_seconds  = elapsed,
        mean_step_ms     = float(step_arr.mean()) if len(step_arr) else 0.0,
        std_step_ms      = float(step_arr.std())  if len(step_arr) else 0.0,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument("--noise_sigma", type=float, default=0.1,)#0.1 # for M3)
    p.add_argument("--delta",       type=float, default=None,#0.1,
                   help="Per-step delta clip magnitude (action units); "
                        "pass 'none' to disable the delta clamp entirely.")
    p.add_argument("--substeps",    type=int,   default=None,
                   help="Rollout substeps per control step (control frequency knob).")
    p.add_argument("--eval_substeps", type=int, default=None,
                   help="Eval steps per rollout step (default: task config, usually 10).")
    p.add_argument("--warm_start", action=argparse.BooleanOptionalAction, default=False,
                   help="Shift the planned sequence forward one control step after "
                        "each plan(); --no-warm_start (default) keeps the running "
                        "mean in place, matching irisim_warp.")
    p.add_argument("--resample_interval", type=int, default=1,
                   help="Plan steps between noise resamples (1=every step; "
                        "omit=sample once and reuse, the default).")
    # --- MPPI-only ---------------------------------------------------------
    p.add_argument("--temperature", type=float, default=20.0)
    # --- CEM-only ----------------------------------------------------------
    p.add_argument("--n_elites",    type=int,   default=None,
                   help="CEM elite-set size; overrides --elite_frac when set.")
    p.add_argument("--elite_frac",  type=float, default=None,
                   help="CEM elite fraction of --n_samples (default 0.1).")
    p.add_argument("--cem_alpha",   type=float, default=None,
                   help="CEM refit smoothing: new = alpha*old + (1-alpha)*fit "
                        "(default 0.1).")
    p.add_argument("--min_sigma",   type=float, default=None,
                   help="CEM floor on the refit sigma (default 1e-3), keeping the "
                        "search from collapsing.")
    # --- predictive-sampler-only ------------------------------------------
    p.add_argument("--include_nominal", action=argparse.BooleanOptionalAction, default=True,
                   help="Predictive sampler: keep sample 0 unperturbed so the greedy "
                        "update can never pick a sequence worse than the nominal.")
    # ----------------------------------------------------------------------
    p.add_argument("--time_constrained", action=argparse.BooleanOptionalAction, default=False,
                   help="Stop rollouts once --plan_budget_ms elapses (capped at the horizon).")
    p.add_argument("--plan_budget_ms", type=float, default=None,
                   help="Wall-clock rollout budget per plan() in ms; required with "
                        "--time_constrained.")
    p.add_argument("--eval_sim",    type=str,   default="none",
                   choices=["none", "mujoco", "drake", "pinocchio"],
                   help="Eval simulator: 'none' uses the task default, else override it.")
    p.add_argument("--settle",      type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--n_episodes",  type=int,   default=1,
                   help="Number of episodes to run; reports the aggregate success rate.")
    p.add_argument("--weights",     nargs="+", default=[],
                   help="Cost-weight overrides as name=value tokens "
                        "(e.g. --weights w_quat=50 w_pos_x=400). Order must match "
                        "the task's config.cost_weights insertion order.")
    p.add_argument("--video",       type=str,   default="videos/grasp_reorient_eval.mp4")
    p.add_argument("--video_format", type=str,  default="mp4", choices=["mp4", "gif"],
                   help="Video container; overrides --video's extension.")
    p.add_argument("--results",     type=str,   default=None,
                   help="JSON path for the episode result(s) (auto-named if omitted).")
    p.add_argument("--debug",       action="store_true",
                   help="Verbose per-step diagnostics (also enables planner debug).")
    args = p.parse_args()

    wp.init()
    planner = resolve_planner_name(args.planner)
    seed_seq = np.random.SeedSequence(args.seed)
    episode_seeds = seed_seq.spawn(args.n_episodes)

    # Rendering every episode in a multi-episode run is slow and usually
    # unwanted; only render when the caller explicitly passed --video (or
    # there's a single episode, matching the old default behavior).
    want_video = args.video is not None or args.n_episodes == 1
    use_mp4 = args.video_format == "mp4"
    base_video_path = args.video
    if base_video_path is None and want_video:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        base_video_path = str(VIDEOS_DIR / f"{args.task}_{planner}_eval.{args.video_format}")

    contact_cfg = MODEL_FACTORIES[args.model]()
    eval_sim = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)

    # Parse `name=value` weight overrides (order must match config.cost_weights).
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

        if not args.delta is None:
            delta = (-args.delta, args.delta)
        else:
            delta = (None, None)

        # One flat knob set for every planner; make_planner_config keeps only the
        # fields the selected planner declares (--temperature for mppi, the
        # --n_elites/--cem_alpha family for cem, ...). Knobs left at None are
        # dropped so each config's own default stands.
        planner_kwargs = dict(
            n_samples      = args.n_samples,
            step_horizon   = args.horizon,
            time_horizon   = args.time_horizon,
            step_time      = args.step_time,
            noise_sigma    = args.noise_sigma,
            step_substeps  = args.substeps,
            warm_start     = args.warm_start,   # off: match irisim_warp (keep the running mean, no shift)
            use_full_graph = True,
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
        # These have no CLI default: leaving them out lets the planner's own
        # default apply (e.g. CEM's n_iterations=3, elite_frac=0.1).
        for name, val in (("n_iterations", args.n_iterations),
                          ("n_elites",     args.n_elites),
                          ("elite_frac",   args.elite_frac),
                          ("alpha",        args.cem_alpha),
                          ("min_sigma",    args.min_sigma)):
            if val is not None:
                planner_kwargs[name] = val

        planner_cfg = make_planner_config(planner, **planner_kwargs)

        result = run_eval_episode(
            task_name   = args.task,
            geometry    = args.geometry,
            contact_cfg = contact_cfg,
            planner     = planner,
            planner_cfg = planner_cfg,
            rng         = rng,
            video_path  = video_path,
            use_mp4     = use_mp4,
            cost_weight_overrides = overrides or None,
            settle_seconds = args.settle,
            eval_substeps  = args.eval_substeps,
            eval_sim       = eval_sim,
            ep_idx         = ep_idx,
            debug          = args.debug,
            verbose        = args.debug or args.n_episodes == 1,
            fin_ep_on_success = True,
        )
        results.append(result)

        label = "✓" if result.success else "✗"
        sstr  = f"step {result.steps_to_success}" if result.steps_to_success is not None else "—"
        print(f"  [ep {ep_idx:03d}] {label}  success_step={sstr}  "
              f"final_cost={result.final_cost:.4f}  "
              f"step={result.mean_step_ms:.3f}±{result.std_step_ms:.3f} ms")

    # ---- aggregate + save -------------------------------------------------
    n_success = sum(r.success for r in results)
    success_rate = n_success / len(results)
    mean_step_ms = float(np.mean([r.mean_step_ms for r in results]))

    print(f"\n{'='*60}")
    print(f"  task={args.task}  model={args.model}  planner={planner}  "
          f"n_episodes={args.n_episodes}")
    print(f"  success_rate={success_rate:.3f}  ({n_success}/{len(results)})  "
          f"mean_step_ms={mean_step_ms:.3f}")
    print(f"{'='*60}")

    results_path = args.results
    if results_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = str(RESULTS_DIR / f"{args.task}_{args.model}_{planner}_eval.json")
    with open(results_path, "w") as f:
        json.dump({
            "task":          args.task,
            "model":         args.model,
            "planner":       planner,
            "n_episodes":    args.n_episodes,
            "success_rate":  success_rate,
            "mean_step_ms":  mean_step_ms,
            "episodes":      [dataclasses.asdict(r) for r in results],
        }, f, indent=2)
    print(f"  Saved result(s) -> {results_path}")

    return results


if __name__ == "__main__":
    main()
