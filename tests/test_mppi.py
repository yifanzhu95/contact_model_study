"""test_mppi.py

Closed-loop MPPI debugger for contact_study.
Runs an MPPI controller (via MPPIController or fixed_budget_rollout) against
any registered task or raw XML scene, with an optional MuJoCo viewer.

Mirrors the episode logic in run_experiment.py but is structured like
test_allegro.py: a single `run()` function that returns timing/success
stats, plus a --backend flag to swap contact models.

Usage:
    # Headless, task=push, default backend (M2/mjwarp)
    python test_mppi.py

    # With viewer, comfree backend
    python test_mppi.py --viewer --backend comfree

    # Different task, more episodes, budget-based planning (Condition A)
    python test_mppi.py --task peg_in_hole --n_episodes 10 \
                        --condition A --budget_seconds 0.2

    # All backends, headless comparison
    python test_mppi.py --backend all --n_episodes 5

    # Raw XML (no task success check, random cost)
    python test_mppi.py --xml scenes/test_data/allegro/env_allegro_cube.xml \
                        --backend comfree --n_episodes 1 --viewer

    # Verbose per-step debug output
    python test_mppi.py --debug --n_episodes 1
"""

from __future__ import annotations
import os
os.environ["MUJOCO_GL"] = "egl"   # must be set before importing mujoco

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import warp as wp
try:
    import mediapy as media
except ImportError:
    media = None

from contact_study.contact_models.config import ContactModelConfig, GeometryVariant
from contact_study.planners.mppi import MPPIController, MPPIConfig
from contact_study.utils.rollout import fixed_budget_rollout
from contact_study.utils.physics_noise import PhysicsNoiseParams, apply_physics_noise

# Ensure tasks are registered before calling get_task
import contact_study.tasks.tasks  # noqa: F401
from contact_study.tasks.base import get_task

#$from contact_study.contact_models.config import GeometryVariant
#from .base import BaseTask, ContactComplexity, TaskSpec, register

# ---------------------------------------------------------------------------
# Contact model factory table (matches run_experiment.py)
# ---------------------------------------------------------------------------

MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,   # mjwarp_hard
    "M2": ContactModelConfig.M2,   # mjwarp (default)
    "M3": ContactModelConfig.M3,   # comfree
    "M4": ContactModelConfig.M4,   # xpbd
}

BACKEND_TO_MODEL = {
    "mjwarp":      "M2",
    "mjwarp_hard": "M1",
    "comfree":     "M3",
    "xpbd":        "M4",
}

wp.init()

# ---------------------------------------------------------------------------
# Fallback cost: penalise distance from a fixed reference qpos.
# Used when running against a raw XML that has no registered task.
# ---------------------------------------------------------------------------

@wp.func
def _fallback_cost_func(
    qpos: wp.array(dtype=float),
    qvel: wp.array(dtype=float),
    ctrl: wp.array(dtype=float),
    terminal: bool,
    goal: wp.array(dtype=float),
    indices: wp.array(dtype=int)
) -> float:
    # Dummy cost for raw XML testing without a registered task
    return 0.0


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(
    task_name:         str | None = "grasp_reorient",
    xml_path:          str | None = None,
    backend:           str   = "comfree",      # "mjwarp" | "comfree" | "mjwarp_hard"
    condition:         str        = "B",        # "A" = fixed_budget_rollout, "B" = MPPIController
    n_episodes:        int        = 3,
    budget_seconds:    float      = 0.1,
    n_samples:         int        = 256,
    horizon:           int        = 30,
    seed:              int        = 42,
    geometry:          str        = "accurate",
    mass_sigma:        float      = 0.0,
    inertia_sigma:     float      = 0.0,
    friction_sigma:    float      = 0.0,
    com_sigma:         float      = 0.0,
    settle_seconds:    float      = 10.0,
    render_mode:       str        = "none",
    warmup_episodes:   int        = 1,
    debug:             bool       = False,
) -> dict:
    """Run n_episodes of closed-loop MPPI and return aggregate stats.

    Parameters
    ----------
    task_name:
        Registered task name (push / grasp_reorient / peg_in_hole).
        If None, xml_path must be provided; a fallback cost is used.
    xml_path:
        Path to a raw XML scene. Overrides task_name geometry loading
        when provided alongside a task; used standalone when task_name
        is None.
    backend:
        One of mjwarp | mjwarp_hard | comfree | xpbd.
    condition:
        "A" = fixed_budget_rollout (open-loop sample, pick best action),
        "B" = MPPIController warm-started across steps.
    render_mode:
        "none", "viewer" (live window), or "video" (save mp4).
    """

    model_key = BACKEND_TO_MODEL[backend]
    cfg       = MODEL_FACTORIES[model_key]()
    geo       = GeometryVariant(geometry)
    noise     = PhysicsNoiseParams(
        mass_sigma     = mass_sigma,
        inertia_sigma  = inertia_sigma,
        friction_sigma = friction_sigma,
        com_sigma      = com_sigma,
    )
    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"  backend   : {backend}  ({model_key})")
    print(f"  task      : {task_name or '(raw xml)'}")
    print(f"  condition : {condition}")
    print(f"  n_episodes: {n_episodes}")
    print(f"  horizon   : {horizon}    n_samples: {n_samples}")
    if condition == "A":
        print(f"  budget    : {budget_seconds*1e3:.1f} ms")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Load task / model
    # ------------------------------------------------------------------
    task      = None
    cost_fn   = None
    max_steps = 200

    if task_name is not None:
        task = get_task(task_name, geometry=geo)
        if xml_path is not None:
            mjm, _ = task.load(xml_path)
        else:
            mjm, _ = task.load()
        mjm     = apply_physics_noise(mjm, noise, rng)
        task._mjm = mjm
        cost_fn   = task.cost_fn
        max_steps = task.spec.max_steps
    else:
        # Raw XML path — no registered task
        assert xml_path is not None, "Provide --task or --xml."
        mjm = mujoco.MjSpec.from_file(xml_path).compile()
        mjm = apply_physics_noise(mjm, noise, rng)

    # if xml_path is not None and task_name is not None:
    #     # User supplied an override XML — reload geometry from that file
    #     mjm = mujoco.MjSpec.from_file(xml_path).compile()
    #     mjm = apply_physics_noise(mjm, noise, rng)
    #     if task is not None:
    #         task._mjm = mjm

    # Fallback cost if no task
    mjd_ref   = mujoco.MjData(mjm)
    if mjm.nkey > 0:
        key = mjm.key(0)
        mjd_ref.qpos[:] = key.qpos
        mjd_ref.qvel[:] = key.qvel
        mjd_ref.ctrl[:] = key.ctrl
    ref_qpos  = mjd_ref.qpos.copy()
    # ref_qpos_wp = wp.array(ref_qpos, dtype=wp.float32, device="cuda")

    if task is not None:
        # Use the GPU-accelerated cost function defined in the task
        cost_fn_for_mppi = task.cost_fn_wp
    else:
        # Pass the wp.func directly, no lambdas!
        cost_fn_for_mppi = _fallback_cost_func

    print(f"  nq={mjm.nq}  nv={mjm.nv}  nu={mjm.nu}  max_steps={max_steps}")
    print(f"  integrator      = {mjm.opt.integrator}")
    print(f"  dof_damping     : min={mjm.dof_damping.min():.2e}  "
          f"max={mjm.dof_damping.max():.2e}")

    # ------------------------------------------------------------------
    # MPPI config (shared across episodes; controller is re-built per
    # episode so warm-start state doesn't carry over between runs)
    # ------------------------------------------------------------------
    mppi_cfg = MPPIConfig(
        n_samples  = n_samples,
        horizon    = horizon,
        temperature = 0.75,
        noise_sigma = 0.02,
        warm_start = True,
        debug = debug
    )

    # ------------------------------------------------------------------
    # Rendering setup
    # ------------------------------------------------------------------
    mjd_view = mujoco.MjData(mjm)
    v = None
    renderer = None
    if render_mode == "viewer":
        v = mujoco.viewer.launch_passive(mjm, mjd_view)
    elif render_mode == "video":
        renderer = mujoco.Renderer(mjm)

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------
    episode_times:   list[float] = []
    step_times:      list[float] = []
    successes:       list[bool]  = []
    steps_to_succ:   list[int]   = []

    try:
        for ep in range(n_episodes):
            # Fresh controller each episode (prevents stale warm-start)
            if task is not None:
                q0, v0, u0 = task.sample_initial_state(rng)
            else:
                q0 = ref_qpos.copy()
                v0 = np.zeros(mjm.nv)
                u0 = None

            controller = MPPIController(
                mjm      = mjm,
                cfg      = cfg,
                mppi_cfg = mppi_cfg,
                cost_fn  = cost_fn_for_mppi, # Pass the wp.func
                rng      = rng,
                initial_ctrl_sequence = u0,
            )

            mjd = mujoco.MjData(mjm)

            mjd.qpos[:] = q0
            mjd.qvel[:] = v0
            if u0 is not None:
                mjd.ctrl[:] = u0
            mujoco.mj_forward(mjm, mjd)

            # Allow model to settle (e.g., objects falling to rest)
            settle_steps = int(settle_seconds / mjm.opt.timestep)
            for _ in range(settle_steps):
                mujoco.mj_step(mjm, mjd)
                if render_mode == "viewer" and v is not None:
                    mjd_view.qpos[:] = mjd.qpos
                    mjd_view.qvel[:] = mjd.qvel
                    mujoco.mj_forward(mjm, mjd_view)
                    v.sync()

            success          = False
            steps_to_success = None
            ep_start         = time.perf_counter()
            frames           = []

            for t in range(max_steps):
                step_start = time.perf_counter()

                # --- Plan ---
                if condition == "A":
                    # Fixed-budget open-loop: run as many samples as
                    # possible within budget_seconds, pick best first action.
                    result   = fixed_budget_rollout(
                        mjm            = mjm,
                        cfg            = cfg,
                        budget_seconds = budget_seconds,
                        horizon        = horizon,
                        cost_fn        = cost_fn,
                        initial_qpos   = mjd.qpos,
                        initial_qvel   = mjd.qvel,
                        rng            = rng,
                    )
                    best_idx = int(np.argmin(result["costs"]))
                    ctrl     = result["final_qpos"][best_idx][:mjm.nu]  # first action
                else:
                    # Condition B: warm-started MPPI
                    ctrl = controller.plan(mjd)

                mjd.ctrl[:] += ctrl
                mujoco.mj_step(mjm, mjd)

                step_times.append(time.perf_counter() - step_start)

                # --- Debug output ---
                if debug and t % 10 == 0:
                    terminal = False
                    #cost_val = controller.cost_fn(mjd.qpos, mjd.qvel, mjd.ctrl, terminal)
                    print(f"  [ep {ep:02d} | step {t:04d}]  "
                          f"qpos_norm={np.linalg.norm(mjd.qpos):.4f}  "
                          f"qvel_norm={np.linalg.norm(mjd.qvel):.4f}")

                # --- Success check ---
                if task is not None and task.is_success(mjd):
                    if steps_to_success is None:
                        steps_to_success = t + 1
                        success = True
                        if debug:
                            print(f"  ✓  ep {ep:02d} succeeded at step {steps_to_success}")
                    # Keep simulating so timing is comparable across episodes

                # --- Rendering ---
                if render_mode == "viewer" and v is not None:
                    mjd_view.qpos[:] = mjd.qpos
                    mjd_view.qvel[:] = mjd.qvel
                    mjd_view.ctrl[:] = mjd.ctrl
                    mujoco.mj_forward(mjm, mjd_view)
                    v.sync()
                elif render_mode == "video" and renderer is not None:
                    renderer.update_scene(mjd)
                    frames.append(renderer.render())

            ep_elapsed = time.perf_counter() - ep_start

            if render_mode == "video" and frames and media is not None:
                video_path = f"video_{task_name or 'raw'}_{backend}_{condition}_ep{ep}.mp4"
                media.write_video(video_path, frames, fps=int(1.0/mjm.opt.timestep))
                print(f"  Saved video to {video_path}")

            if ep >= warmup_episodes:
                episode_times.append(ep_elapsed)
                successes.append(success)
                if steps_to_success is not None:
                    steps_to_succ.append(steps_to_success)

            label = "✓" if success else "✗"
            sstr  = f"step {steps_to_success}" if steps_to_success else "—"
            print(f"  ep {ep:02d}  {label}  success_step={sstr:<8}  "
                  f"elapsed={ep_elapsed*1e3:.1f} ms")

    finally:
        if v is not None:
            v.close()

    # ------------------------------------------------------------------
    # Stats (excluding warmup)
    # ------------------------------------------------------------------
    dts      = np.array(step_times)
    ep_arr   = np.array(episode_times) if episode_times else np.array([0.0])
    succ_arr = np.array(successes)     if successes     else np.array([False])

    stats = {
        "backend":               backend,
        "model_key":             model_key,
        "task":                  task_name or "(raw xml)",
        "condition":             condition,
        "n_episodes_measured":   len(ep_arr),
        "success_rate":          float(succ_arr.mean()),
        "mean_steps_to_success": float(np.mean(steps_to_succ)) if steps_to_succ else float("nan"),
        "mean_ep_time_ms":       float(ep_arr.mean() * 1e3),
        "mean_step_time_ms":     float(dts.mean() * 1e3) if len(dts) else float("nan"),
        "std_step_time_ms":      float(dts.std()  * 1e3) if len(dts) else float("nan"),
    }

    print(f"\n  Success rate     : {stats['success_rate']*100:.1f}%")
    if steps_to_succ:
        print(f"  Mean steps/succ  : {stats['mean_steps_to_success']:.1f}")
    print(f"  Mean ep time     : {stats['mean_ep_time_ms']:.1f} ms")
    print(f"  Mean step time   : {stats['mean_step_time_ms']:.3f} ms  "
          f"(±{stats['std_step_time_ms']:.3f})")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop MPPI debugger — mirrors test_allegro.py structure."
    )
    parser.add_argument("--task",    type=str, default="grasp_reorient",
                        help="Registered task name. Set to 'none' to use --xml only.")
    parser.add_argument("--xml",     type=str, default="scenes/test_data/allegro/env_allegro_cube.xml",
                        help="Override / standalone XML scene path.")
    parser.add_argument("--backend", type=str, default="mjwarp",
                        choices=["mjwarp", "mjwarp_hard", "comfree", "xpbd", "all"])
    parser.add_argument("--condition", type=str, default="B", choices=["A", "B"],
                        help="A=fixed_budget_rollout  B=warm-started MPPIController")
    parser.add_argument("--n_episodes",     type=int,   default=1)
    parser.add_argument("--budget_seconds", type=float, default=0.1,
                        help="Per-step time budget for Condition A")
    parser.add_argument("--n_samples",      type=int,   default=256)
    parser.add_argument("--horizon",        type=int,   default=48)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--geometry",       type=str,   default="accurate",
                        choices=[g.value for g in GeometryVariant])
    parser.add_argument("--mass_sigma",     type=float, default=0.0)
    parser.add_argument("--inertia_sigma",  type=float, default=0.0)
    parser.add_argument("--friction_sigma", type=float, default=0.0)
    parser.add_argument("--com_sigma",      type=float, default=0.0)
    parser.add_argument("--settle",         type=float, default=1.0,
                        help="Seconds to allow physics to settle before planning starts")
    parser.add_argument("--render", type=str, default="none", choices=["none", "viewer", "video"],
                        help="Rendering mode: none, viewer (live), or video (save mp4)")
    parser.add_argument("--warmup",         type=int,   default=1,
                        help="Episodes to skip when computing aggregate stats")
    parser.add_argument("--debug",          action="store_true",
                        help="Print per-step diagnostics")
    args = parser.parse_args()

    task_name = None if args.task.lower() == "none" else args.task

    backends = (
        ["mjwarp", "mjwarp_hard", "comfree", "xpbd"]
        if args.backend == "all"
        else [args.backend]
    )

    all_stats = []
    for i, backend in enumerate(backends):
        # Only open viewer for first backend; save video for all backends if requested
        current_render = args.render
        if args.render == "viewer" and i > 0:
            current_render = "none"

        stats = run(
            task_name         = task_name,
            xml_path          = args.xml,
            backend           = backend,
            condition         = args.condition,
            n_episodes        = args.n_episodes,
            budget_seconds    = args.budget_seconds,
            n_samples         = args.n_samples,
            horizon           = args.horizon,
            seed              = args.seed,
            geometry          = args.geometry,
            mass_sigma        = args.mass_sigma,
            inertia_sigma     = args.inertia_sigma,
            friction_sigma    = args.friction_sigma,
            com_sigma         = args.com_sigma,
            settle_seconds    = args.settle,
            render_mode       = current_render,
            warmup_episodes   = args.warmup,
            debug             = args.debug,
        )
        all_stats.append(stats)

    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print(f"  Summary  (task={task_name or 'raw xml'}  condition={args.condition})")
        print(f"{'='*60}")
        print(f"  {'backend':<16}  {'succ%':>6}  {'step_ms':>9}  {'ep_ms':>9}")
        print(f"  {'-'*16}  {'-'*6}  {'-'*9}  {'-'*9}")
        for s in all_stats:
            print(f"  {s['backend']:<16}  "
                  f"{s['success_rate']*100:>5.1f}%  "
                  f"{s['mean_step_time_ms']:>8.3f}ms  "
                  f"{s['mean_ep_time_ms']:>8.1f}ms")


if __name__ == "__main__":
    main()