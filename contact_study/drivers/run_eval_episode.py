"""Closed-loop eval episode: pluggable "real" simulator + GPU-MPPI planner.

Generalizes tests/test_drake.py. Two task instances share the same TaskConfig:

  * a ROLLOUT task — owns the planning MuJoCo model + cost arrays; handed to the
    MPPIController (which still owns the batched GPU rollout env).
  * an EVAL task   — owns the high-fidelity "real" environment (an EvalSimulator,
    MuJoCo or Drake) that this loop actually steps.

Per control step: read eval state → mirror into the planning MjData →
controller.plan() on the GPU → integrate the delta into the command → apply to
the eval simulator → advance it → render.

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
from pathlib import Path

import numpy as np
import mujoco
import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.contact_models.config import ContactModelConfig
from contact_study.planners.mppi import MPPIController, MPPIConfig
from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole

MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
}

VIDEOS_DIR = Path(__file__).parents[2] / "videos"


def run_eval_episode(
    task_name:   str,
    contact_cfg: ContactModelConfig,
    mppi_cfg:    MPPIConfig,
    rng:         np.random.Generator,
    sim_time:    float = 10.0,
    video_path:  str | None = None,
    init_angle:  float | None = None,
    settle_seconds: float = 0.0,
    verbose:     bool = True,
) -> dict:
    # ---- ROLLOUT task + planner ------------------------------------------
    rollout_task = get_task(task_name, role=TaskRole.ROLLOUT)
    if init_angle is not None and hasattr(rollout_task, "init_angle"):
        rollout_task.init_angle = init_angle
    mjm, mjd = rollout_task.load()
    cfg = rollout_task.config

    controller = MPPIController(
        task=rollout_task, cfg=contact_cfg, mppi_cfg=mppi_cfg, rng=rng,
    )

    # ---- EVAL task + "real" simulator ------------------------------------
    eval_task = get_task(task_name, role=TaskRole.EVAL)
    if init_angle is not None and hasattr(eval_task, "init_angle"):
        eval_task.init_angle = init_angle
    eval_task.load()
    sim = eval_task.make_eval_simulator(video_path=video_path, render=True)

    # ---- initial state ----------------------------------------------------
    q0, v0, u0 = rollout_task.get_inital_state(rng)
    sim.reset(np.asarray(q0, dtype=float), np.asarray(v0, dtype=float))
    u = np.asarray(u0, dtype=float).copy()

    # Settle (objects fall to rest / hand closes onto the object). Hold the
    # initial command each substep so a position-controlled hand doesn't go limp.
    if settle_seconds > 0.0:
        n_settle = int(settle_seconds / sim.timestep)
        for _ in range(n_settle):
            sim.apply_control(u)
            sim.step(1)

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

    control_dt = mppi_cfg.substeps * mjm.opt.timestep
    n_steps    = int(sim_time / control_dt)
    steps_to_success: int | None = None

    if verbose:
        print(f"  task={task_name}  eval_sim={cfg.eval_sim.value}  "
              f"control_dt={control_dt*1e3:.1f}ms  n_steps={n_steps}  "
              f"horizon={mppi_cfg.horizon}  n_samples={mppi_cfg.n_samples}")

    for t in range(n_steps):
        # 1. Read eval state (MuJoCo-ordered) and mirror into the planning MjData.
        st = sim.get_state()
        mjd.qpos[:] = st.qpos
        mjd.qvel[:] = st.qvel
        mjd.ctrl[:] = u
        mujoco.mj_forward(mjm, mjd)

        # Success/failure are evaluated on the current (start-of-step) state.
        if rollout_task.is_success(mjd) and steps_to_success is None:
            steps_to_success = t
        if rollout_task.has_failed(mjd):
            if verbose:
                print(f"  step {t:4d}: task failed")
            break

        # 2. Plan on the GPU and integrate the delta into the absolute command.
        action = controller.plan(mjd)
        u = u + action
        if clip_lo is not None:
            u = np.clip(u, clip_lo, clip_hi)

        # 3. Apply, advance the eval sim, capture a frame.
        sim.apply_control(u)
        sim.step(mppi_cfg.substeps)
        sim.render()

        if verbose and t % 25 == 0:
            print(f"  step {t:4d}  t={t*control_dt:5.2f}s  "
                  f"u[0]={float(u[0]):+8.3f}")

    if video_path is not None:
        sim.save_video(video_path)
        if verbose:
            print(f"  Saved video -> {video_path}")

    success = steps_to_success is not None
    if verbose:
        print(f"  success={success}  steps_to_success={steps_to_success}")
    return {"success": success, "steps_to_success": steps_to_success}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",        type=str,   default="cart_pole")
    p.add_argument("--model",       type=str,   default="M2", choices=list(MODEL_FACTORIES))
    p.add_argument("--n_samples",   type=int,   default=256)
    p.add_argument("--horizon",     type=int,   default=50)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--noise_sigma", type=float, default=0.001)
    p.add_argument("--delta",       type=float, default=0.01,
                   help="Per-step MPPI delta clip magnitude (action units).")
    p.add_argument("--substeps",    type=int,   default=1)
    p.add_argument("--sim_time",    type=float, default=10.0)
    p.add_argument("--init_angle",  type=float, default=0.0,
                   help="cart_pole initial pole angle (rad): 0=down, pi=upright.")
    p.add_argument("--settle",      type=float, default=0.0)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--video",       type=str,   default=None)
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
        debug          = False,
    )

    run_eval_episode(
        task_name   = args.task,
        contact_cfg = contact_cfg,
        mppi_cfg    = mppi_cfg,
        rng         = rng,
        sim_time    = args.sim_time,
        video_path  = video_path,
        init_angle  = args.init_angle,
        settle_seconds = args.settle,
    )


if __name__ == "__main__":
    main()
