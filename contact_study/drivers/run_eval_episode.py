"""Closed-loop eval episode: pluggable "real" simulator + GPU-MPPI planner.

Generalizes tests/test_drake.py. Two task instances share the same TaskConfig:

  * a ROLLOUT task — owns the planning MuJoCo model + cost arrays; handed to the
    MPPIController (which still owns the batched GPU rollout env).
  * an EVAL task   — owns the high-fidelity "real" environment (an EvalSimulator,
    MuJoCo or Drake) that this loop actually steps.

Per control step: read eval state → mirror into the planning MjData →
controller.plan() on the GPU → integrate the delta into the command → apply to
the eval simulator → advance it → render.

The eval ("real") simulator runs at a finer timestep than the rollout/planning
model: rollout_dt = eval_dt * eval_substeps_per_rollout (the eval sim takes that
many steps per rollout step). The rollout step is inferred from the eval step and
stamped onto the planning model. Control frequency stays a separate knob — one
control step spans mppi_cfg.substeps rollout steps (control_dt = substeps *
rollout_dt), so the eval sim advances substeps * eval_substeps_per_rollout steps.

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
os.environ.setdefault("MUJOCO_GL", "disable")

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
from contact_study.planners.mppi import MPPIController, MPPIConfig
from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole

MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
}

VIDEOS_DIR  = Path(__file__).parents[2] / "videos"
RESULTS_DIR = Path(__file__).parents[2] / "results"


def run_eval_episode(
    task_name:   str,
    contact_cfg: ContactModelConfig,
    mppi_cfg:    MPPIConfig,
    rng:         np.random.Generator,
    video_path:  str | None = None,
    settle_seconds: float = 0.0,
    eval_substeps: int | None = None,
    condition:   str  = "B",
    debug:       bool = False,
    verbose:     bool = True,
) -> EpisodeResult:
    """Run one closed-loop eval episode and return an EpisodeResult.

    The episode length is the task's TaskConfig.max_steps. mean_step_ms /
    std_step_ms hold per-control-step latency (plan + eval advance) in ms.
    """
    # ---- ROLLOUT task + planner ------------------------------------------
    rollout_task = get_task(task_name, role=TaskRole.ROLLOUT)
    mjm, mjd = rollout_task.load()
    cfg = rollout_task.config

    # The eval ("real") simulator runs at the fine cfg.timestep; the rollout /
    # planning model runs coarser. Infer the rollout step from the eval step and
    # stamp it onto the planning model BEFORE the controller copies the model to
    # the GPU (api.put_model reads mjm.opt.timestep at construction):
    #   rollout_dt = eval_dt * eval_substeps_per_rollout
    eval_dt    = cfg.timestep
    eval_substeps = eval_substeps if eval_substeps is not None else cfg.eval_substeps_per_rollout
    rollout_dt = eval_dt * eval_substeps
    mjm.opt.timestep = rollout_dt

    controller = MPPIController(
        task=rollout_task, cfg=contact_cfg, mppi_cfg=mppi_cfg, rng=rng,
    )

    # ---- EVAL task + "real" simulator ------------------------------------
    eval_task = get_task(task_name, role=TaskRole.EVAL)
    eval_task.load()
    sim = eval_task.make_eval_simulator(video_path=video_path, render=True)

    # ---- initial state ----------------------------------------------------
    q0, v0, u0 = rollout_task.get_inital_state(rng)
    sim.reset(np.asarray(q0, dtype=float), np.asarray(v0, dtype=float))
    u = np.asarray(u0, dtype=float).copy()

    # Settle (objects fall to rest / hand closes onto the object). Hold the
    # initial command each rollout step so a position-controlled hand doesn't go
    # limp; advance the eval sim a full rollout step (eval_substeps fine steps).
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

    # One control step = mppi_cfg.substeps rollout steps. The eval sim covers the
    # same wall-clock with `eval_substeps` finer steps per rollout step. The
    # episode runs for the task's configured max_steps control steps.
    control_dt      = mppi_cfg.substeps * rollout_dt
    eval_steps_per_control = mppi_cfg.substeps * eval_substeps
    n_steps         = cfg.max_steps
    steps_to_success: int | None = None

    if verbose:
        print(f"  task={task_name}  eval_sim={cfg.eval_sim.value}  "
              f"eval_dt={eval_dt*1e3:.2f}ms  rollout_dt={rollout_dt*1e3:.2f}ms  "
              f"control_dt={control_dt*1e3:.1f}ms  max_steps={n_steps}  "
              f"horizon={mppi_cfg.horizon}  n_samples={mppi_cfg.n_samples}")

    step_times: list[float] = []
    ep_start = time.perf_counter()

    for t in range(n_steps):
        step_start = time.perf_counter()

        # 1. Read eval state (MuJoCo-ordered) and mirror into the planning MjData.
        st = sim.get_state()
        mjd.qpos[:] = st.qpos
        mjd.qvel[:] = st.qvel
        mjd.ctrl[:] = u
        mujoco.mj_forward(mjm, mjd)

        # Success/failure are evaluated on the current (start-of-step) state.
        if rollout_task.is_success(mjd) and steps_to_success is None:
            steps_to_success = t
            if debug:
                print(f"  first success at step {t}")
        if rollout_task.has_failed(mjd):
            if verbose:
                print(f"  step {t:4d}: task failed")
            break

        # 2. Plan on the GPU and integrate the delta into the absolute command.
        action = controller.plan(mjd)
        u = u + action
        if clip_lo is not None:
            u = np.clip(u, clip_lo, clip_hi)

        # 3. Apply, advance the eval sim (finer steps over the same control_dt),
        #    capture a frame.
        sim.apply_control(u)
        sim.step(eval_steps_per_control)
        sim.render()

        step_times.append((time.perf_counter() - step_start) * 1e3)

        if debug and t % 10 == 0:
            print(f"  [step {t:04d}]  qpos_norm={np.linalg.norm(st.qpos):.4f}  "
                  f"qvel_norm={np.linalg.norm(st.qvel):.4f}  u[0]={float(u[0]):+8.3f}")
        elif verbose and t % 25 == 0:
            print(f"  step {t:4d}  t={t*control_dt:5.2f}s  u[0]={float(u[0]):+8.3f}")

    elapsed = time.perf_counter() - ep_start

    if video_path is not None:
        sim.save_video(video_path)
        if verbose:
            print(f"  Saved video -> {video_path}")

    final_qpos = sim.get_state().qpos
    step_arr = np.asarray(step_times)
    return EpisodeResult(
        task_name        = cfg.name,
        model_label      = contact_cfg.label,
        condition        = condition,
        success          = steps_to_success is not None,
        steps_to_success = steps_to_success,
        final_cost       = float(np.linalg.norm(final_qpos - np.asarray(q0, dtype=float))),
        n_samples_used   = mppi_cfg.n_samples,
        elapsed_seconds  = elapsed,
        mean_step_ms     = float(step_arr.mean()) if len(step_arr) else 0.0,
        std_step_ms      = float(step_arr.std())  if len(step_arr) else 0.0,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",        type=str,   default="cart_pole")
    p.add_argument("--model",       type=str,   default="M2", choices=list(MODEL_FACTORIES))
    p.add_argument("--n_samples",   type=int,   default=256)
    p.add_argument("--horizon",     type=int,   default=128)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--noise_sigma", type=float, default=0.01)
    p.add_argument("--delta",       type=float, default=0.1,
                   help="Per-step MPPI delta clip magnitude (action units).")
    p.add_argument("--substeps",    type=int,   default=1,
                   help="MPPI rollout substeps per control step (control frequency knob).")
    p.add_argument("--eval_substeps", type=int, default=None,
                   help="Eval steps per rollout step (default: task config, usually 10).")
    p.add_argument("--settle",      type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--video",       type=str,   default=None)
    p.add_argument("--results",     type=str,   default=None,
                   help="JSON path for the episode result (auto-named if omitted).")
    p.add_argument("--debug",       action="store_true",
                   help="Verbose per-step diagnostics (also enables MPPI debug).")
    args = p.parse_args()

    wp.init()
    rng = np.random.default_rng(args.seed)

    video_path = args.video
    if video_path is None:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        video_path = str(VIDEOS_DIR / f"{args.task}_eval.gif")

    contact_cfg = MODEL_FACTORIES[args.model]()
    mppi_cfg = MPPIConfig(
        n_samples      = args.n_samples,
        horizon        = args.horizon,
        temperature    = args.temperature,
        noise_sigma    = args.noise_sigma,
        substeps       = args.substeps,
        warm_start     = True,
        use_full_graph = False,
        delta_range    = (-args.delta, args.delta),
        nconmax        = 200,
        njmax          = 500,
        seed           = args.seed,
        debug          = args.debug,
    )

    result = run_eval_episode(
        task_name   = args.task,
        contact_cfg = contact_cfg,
        mppi_cfg    = mppi_cfg,
        rng         = rng,
        video_path  = video_path,
        settle_seconds = args.settle,
        eval_substeps  = args.eval_substeps,
        debug          = args.debug,
    )

    # ---- print + save the episode result ---------------------------------
    label = "✓" if result.success else "✗"
    sstr  = f"step {result.steps_to_success}" if result.steps_to_success is not None else "—"
    print(f"\n{'='*60}")
    print(f"  task={result.task_name}  model={result.model_label}  condition={result.condition}")
    print(f"  {label}  success_step={sstr}  final_cost={result.final_cost:.4f}")
    print(f"  elapsed={result.elapsed_seconds*1e3:.1f} ms  "
          f"step={result.mean_step_ms:.3f}±{result.std_step_ms:.3f} ms")
    print(f"{'='*60}")

    results_path = args.results
    if results_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = str(RESULTS_DIR / f"{args.task}_{args.model}_eval.json")
    with open(results_path, "w") as f:
        json.dump(dataclasses.asdict(result), f, indent=2)
    print(f"  Saved result -> {results_path}")

    return result


if __name__ == "__main__":
    main()
