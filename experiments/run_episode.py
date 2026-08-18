"""run_episode.py

Single-episode MPPI runner — the importable building block for evaluations.

The core function `run_episode` runs one closed-loop MPPI episode given a
fully-configured task, contact model, and MPPI config. It returns an
EpisodeResult and is designed to be imported by experiment scripts.

The CLI mirrors test_mppi.py: it loads one (task, backend) pair, runs a
single episode (or a few), and optionally renders live or saves a video.

Usage:
    # Headless, default backend (mjwarp), grasp_reorient
    python experiments/run_episode.py

    # Live viewer, comfree backend
    python experiments/run_episode.py --render viewer --backend comfree

    # Save video, peg_in_hole task
    python experiments/run_episode.py --task peg_in_hole --render video

    # More episodes, fixed-budget planner (Condition A)
    python experiments/run_episode.py --n_episodes 5 --condition A --budget_seconds 0.2

    # All backends, headless comparison
    python experiments/run_episode.py --backend all --n_episodes 3

    # Physics noise ablation
    python experiments/run_episode.py --mass_sigma 0.05 --friction_sigma 0.1

    # Verbose per-step debug output
    python experiments/run_episode.py --debug --n_episodes 1
"""

from __future__ import annotations
import os
os.environ["MUJOCO_GL"] = "egl"  # must be set before importing mujoco

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import warp as wp
import mediapy as media

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.contact_models.config import ContactModelConfig
from contact_study.evaluation.metrics import EpisodeResult
from contact_study.planners.mppi import MPPIController, MPPIConfig
from contact_study.tasks.base import get_task
from contact_study.utils.physics_noise import PhysicsNoiseParams, apply_physics_noise
from contact_study.utils.rollout import fixed_budget_rollout
from contact_study.tasks.config import DEFAULT_SCENE_VARIANT

# ---------------------------------------------------------------------------
# Contact model factory table
# ---------------------------------------------------------------------------

MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
}

BACKEND_TO_MODEL = {
    "mjwarp":      "M2",
    "mjwarp_hard": "M1",
    "comfree":     "M3",
    "xpbd":        "M4",
}

wp.init()


# ---------------------------------------------------------------------------
# Core: single-episode runner
# ---------------------------------------------------------------------------

def run_episode(
    mjm:            mujoco.MjModel,
    task,
    cfg:            ContactModelConfig,
    mppi_cfg:       MPPIConfig,
    rng:            np.random.Generator,
    condition:      str   = "B",
    budget_seconds: float = 0.1,
    settle_seconds: float = 1.0,
    render_mode:    str   = "none",
    video_path:     str | None = None,
    debug:          bool  = False,
    ep_idx:         int   = 0,
    fin_ep_on_success: bool = True,
) -> EpisodeResult:
    """Run one closed-loop MPPI episode and return an EpisodeResult.

    mean_step_ms and std_step_ms on the result hold per-control-step latency
    (plan + substeps) in milliseconds.

    Parameters
    ----------
    mjm:
        MuJoCo model (already noise-applied if desired).
    task:
        A registered task instance with cost_fn_wp, is_success, has_failed, etc.
    cfg:
        Contact model configuration (M1–M4).
    mppi_cfg:
        Full MPPI hyperparameter config.
    rng:
        Numpy random generator (shared across episodes for reproducibility).
    condition:
        "A" = fixed_budget_rollout (open-loop, best-of-N),
        "B" = warm-started MPPIController.
    budget_seconds:
        Per-step wall-time budget for Condition A.
    settle_seconds:
        Seconds to simulate before planning starts (objects fall to rest).
    render_mode:
        "none" | "viewer" (live window) | "video" (save mp4).
    video_path:
        File path for video output; auto-generated if None and mode is "video".
    debug:
        Print per-step diagnostics every 10 steps.
    ep_idx:
        Episode index used in logging and auto video naming.
    """

    task_cfg = task.config or task.spec

    mjd = mujoco.MjData(mjm)
    q0, v0, u0 = task.get_inital_state(rng)
    mjd.qpos[:] = q0
    mjd.qvel[:] = v0
    if u0 is not None:
        mjd.ctrl[:] = u0
    mujoco.mj_forward(mjm, mjd)

    # ------------------------------------------------------------------
    # Rendering setup
    # ------------------------------------------------------------------
    mjd_view = mujoco.MjData(mjm)
    viewer   = None
    renderer = None
    frames   = []

    if render_mode == "viewer":
        viewer = mujoco.viewer.launch_passive(mjm, mjd_view)
    elif render_mode == "video":
        renderer = mujoco.Renderer(mjm, height=480, width=640)

    def _sync_viewer():
        if viewer is not None:
            mjd_view.qpos[:] = mjd.qpos
            mjd_view.qvel[:] = mjd.qvel
            mjd_view.ctrl[:] = mjd.ctrl
            mujoco.mj_forward(mjm, mjd_view)
            viewer.sync()

    def _capture_frame():
        if renderer is not None:
            renderer.update_scene(mjd, camera="top")
            frames.append(renderer.render())

    try:
        # ------------------------------------------------------------------
        # Settle physics
        # ------------------------------------------------------------------
        settle_steps = int(settle_seconds / mjm.opt.timestep)
        for _ in range(settle_steps):
            mujoco.mj_step(mjm, mjd)
            _sync_viewer()

        # Sample a fresh goal BEFORE creating the controller so it captures
        # the updated task.cost_goal_wp reference, not the stale XML default.
        if hasattr(task, "sample_new_goal"):
            task.sample_new_goal(mjd, rng)

        controller = MPPIController(
            task       = task,
            cfg        = cfg,
            mppi_cfg   = mppi_cfg,
            rng        = rng,
        )

        # ------------------------------------------------------------------
        # Episode loop
        # ------------------------------------------------------------------
        steps_to_success: int | None = None
        n_samples_used = mppi_cfg.n_samples
        step_times: list[float] = []
        ep_start = time.perf_counter()

        for t in range(task_cfg.max_steps):
            step_start = time.perf_counter()
            if condition == "A":
                result   = fixed_budget_rollout(
                    mjm            = mjm,
                    cfg            = cfg,
                    budget_seconds = budget_seconds,
                    horizon        = controller.horizon,
                    initial_qpos   = mjd.qpos,
                    initial_qvel   = mjd.qvel,
                    rng            = rng,
                )
                best_idx     = int(np.argmin(result["costs"]))
                ctrl         = result["final_qpos"][best_idx][:mjm.nu]
                n_samples_used = result["n_samples"]
            else:
                ctrl = controller.plan(mjd)

            mjd.ctrl[:] += ctrl

            for _ in range(controller.substeps):
                mujoco.mj_step(mjm, mjd)
                _sync_viewer()
                _capture_frame()

            step_times.append((time.perf_counter() - step_start) * 1e3)

            if debug and t % 10 == 0:
                print(f"  [ep {ep_idx:02d} | step {t:04d}]  "
                      f"qpos_norm={np.linalg.norm(mjd.qpos):.4f}  "
                      f"qvel_norm={np.linalg.norm(mjd.qvel):.4f}")

            if task.is_success(mjd):
                if steps_to_success is None:
                    steps_to_success = t + 1
                    if debug:
                        print(f"  [ep {ep_idx:02d}] First success at step {steps_to_success}")

                if fin_ep_on_success:
                    break

                if hasattr(task, "sample_new_goal"):
                    print(f"  [ep {ep_idx:02d}] Success — sampling new goal")
                    task.sample_new_goal(mjd, rng)
                    controller.reset()
                    controller.lam = controller.pc.temperature

            if task.has_failed(mjd):
                if debug:
                    print(f"  [ep {ep_idx:02d}] Task failed at step {t}")
                break

        elapsed = time.perf_counter() - ep_start

        # ------------------------------------------------------------------
        # Video save
        # ------------------------------------------------------------------
        if render_mode == "video" and frames:
            if video_path is None:
                Path("videos").mkdir(exist_ok=True)
                video_path = f"videos/ep{ep_idx}_{task_cfg.name}_{cfg.label}.mp4"
            media.write_video(video_path, frames, fps=int(1.0 / mjm.opt.timestep))
            print(f"  Saved video → {video_path}")

    finally:
        if viewer is not None:
            viewer.close()

    step_arr = np.asarray(step_times)
    return EpisodeResult(
        task_name        = task_cfg.name,
        model_label      = cfg.label,
        condition        = condition,
        success          = steps_to_success is not None,
        steps_to_success = steps_to_success,
        final_cost       = float(np.linalg.norm(mjd.qpos - q0)),
        n_samples_used   = n_samples_used,
        elapsed_seconds  = elapsed,
        mean_step_ms     = float(step_arr.mean()) if len(step_arr) else 0.0,
        std_step_ms      = float(step_arr.std())  if len(step_arr) else 0.0,
    )


# ---------------------------------------------------------------------------
# CLI helper: load + noise-apply a task model
# ---------------------------------------------------------------------------

def load_task(
    task_name: str,
    geometry:  str,
    noise:     PhysicsNoiseParams,
    rng:       np.random.Generator,
):
    task = get_task(task_name, geometry=geometry)
    mjm, _ = task.load()
    mjm = apply_physics_noise(mjm, noise, rng)
    task._mjm = mjm
    return mjm, task


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-episode MPPI runner — importable core for evaluations."
    )
    parser.add_argument("--task",    type=str, default="grasp_reorient",
                        help="Registered task name.")
    parser.add_argument("--backend", type=str, default="mjwarp",
                        choices=["mjwarp", "mjwarp_hard", "comfree", "xpbd", "all"])
    parser.add_argument("--condition", type=str, default="B", choices=["A", "B"],
                        help="A=fixed_budget_rollout  B=warm-started MPPIController")
    parser.add_argument("--n_episodes",     type=int,   default=1)
    parser.add_argument("--budget_seconds", type=float, default=0.1,
                        help="Per-step time budget for Condition A")
    parser.add_argument("--n_samples",      type=int,   default=256)
    parser.add_argument("--horizon",        type=int,   default=128)
    parser.add_argument("--temperature",    type=float, default=0.05)
    parser.add_argument("--noise_sigma",    type=float, default=0.01)
    parser.add_argument("--seed",           type=int,   default=None)
    parser.add_argument("--geometry",       type=str,   default=DEFAULT_SCENE_VARIANT,
                        help="Scene variant: '<object>' or "
                             "'<object>_<hand_acc>_<obj_acc>' (e.g. duck_low_high). "
                             "Legacy geometry names map to the default scene.")
    parser.add_argument("--mass_sigma",     type=float, default=0.0)
    parser.add_argument("--inertia_sigma",  type=float, default=0.0)
    parser.add_argument("--friction_sigma", type=float, default=0.0)
    parser.add_argument("--com_sigma",      type=float, default=0.0)
    parser.add_argument("--settle",         type=float, default=10.0,
                        help="Seconds to allow physics to settle before planning starts")
    parser.add_argument("--render", type=str, default="video",
                        choices=["none", "viewer", "video"],
                        help="none | viewer (live window) | video (save mp4)")
    parser.add_argument("--use_full_graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--nconmax",        type=int,   default=200)
    parser.add_argument("--njmax",          type=int,   default=500)
    parser.add_argument("--debug",          action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    noise    = PhysicsNoiseParams(
        mass_sigma     = args.mass_sigma,
        inertia_sigma  = args.inertia_sigma,
        friction_sigma = args.friction_sigma,
        com_sigma      = args.com_sigma,
    )
    geometry = args.geometry

    backends = (
        ["mjwarp", "mjwarp_hard", "comfree", "xpbd"]
        if args.backend == "all"
        else [args.backend]
    )

    all_results = []
    for i, backend in enumerate(backends):
        model_key = BACKEND_TO_MODEL[backend]
        cfg       = MODEL_FACTORIES[model_key]()
        mjm, task = load_task(args.task, geometry, noise, rng)

        mppi_cfg = MPPIConfig(
            n_samples      = args.n_samples,
            step_horizon   = args.horizon,
            temperature    = args.temperature,
            noise_sigma    = args.noise_sigma,
            warm_start     = True,
            use_full_graph = args.use_full_graph,
            nconmax        = args.nconmax,
            njmax          = args.njmax,
            seed           = args.seed,
            debug          = args.debug,
        )

        print(f"\n{'='*60}")
        print(f"  backend   : {backend}  ({model_key})")
        print(f"  task      : {args.task}")
        print(f"  condition : {args.condition}")
        print(f"  n_episodes: {args.n_episodes}")
        print(f"  horizon   : {args.horizon}    n_samples: {args.n_samples}")
        if args.condition == "A":
            print(f"  budget    : {args.budget_seconds*1e3:.1f} ms")
        print(f"{'='*60}")
        print(f"  nq={mjm.nq}  nv={mjm.nv}  nu={mjm.nu}  "
              f"max_steps={(task.config or task.spec).max_steps}")

        # Only show viewer for the first backend when comparing all
        render_mode = args.render if (i == 0 or args.render == "video") else "none"

        ep_results = []
        for ep in range(args.n_episodes):
            result = run_episode(
                mjm            = mjm,
                task           = task,
                cfg            = cfg,
                mppi_cfg       = mppi_cfg,
                rng            = rng,
                condition      = args.condition,
                budget_seconds = args.budget_seconds,
                settle_seconds = args.settle,
                render_mode    = render_mode,
                debug          = args.debug,
                ep_idx         = ep,
                fin_ep_on_success = False,
            )
            ep_results.append(result)
            label = "✓" if result.success else "✗"
            sstr  = f"step {result.steps_to_success}" if result.steps_to_success else "—"
            print(f"  ep {ep:02d}  {label}  success_step={sstr:<8}  "
                  f"elapsed={result.elapsed_seconds*1e3:.1f} ms  "
                  f"step={result.mean_step_ms:.3f}±{result.std_step_ms:.3f} ms")

        successes   = [r.success for r in ep_results]
        elapsed_arr = np.array([r.elapsed_seconds for r in ep_results])
        succ_steps  = [r.steps_to_success for r in ep_results if r.steps_to_success is not None]

        print(f"\n  Success rate : {np.mean(successes)*100:.1f}%")
        if succ_steps:
            print(f"  Mean steps   : {np.mean(succ_steps):.1f}")
        print(f"  Mean ep time : {elapsed_arr.mean()*1e3:.1f} ms")
        print(f"  Step time    : {np.mean([r.mean_step_ms for r in ep_results]):.3f} ms")

        all_results.extend(ep_results)

    if len(backends) > 0:
        print(f"\n{'='*60}")
        print(f"  Summary  (task={args.task}  condition={args.condition})")
        print(f"{'='*60}")
        print(f"  {'backend':<16}  {'succ%':>6}  {'ep_ms':>9}  {'step_ms':>10}  {'step_std':>9}")
        print(f"  {'-'*16}  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*9}")
        n = args.n_episodes
        for i, backend in enumerate(backends):
            batch   = all_results[i*n:(i+1)*n]
            sr      = np.mean([r.success for r in batch]) * 100
            et      = np.mean([r.elapsed_seconds for r in batch]) * 1e3
            step_mu = np.mean([r.mean_step_ms for r in batch])
            step_sd = np.mean([r.std_step_ms  for r in batch])
            print(f"  {backend:<16}  {sr:>5.1f}%  {et:>8.1f}ms  "
                  f"{step_mu:>9.3f}ms  {step_sd:>8.3f}ms")


if __name__ == "__main__":
    main()
