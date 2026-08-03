"""Closed-loop eval episode that also records the applied control command from
every MPPI step to a .npy file.

Same eval/rollout split as run_eval_episode.py (see that module's docstring for
the full picture): a ROLLOUT task drives the MPPIController on the GPU, an EVAL
task owns the "real" simulator that this loop actually steps. This variant just
appends the absolute command `u` applied to the eval sim after each control
step and dumps it to disk at the end, alongside the usual video.

Run on a CUDA machine (warp arrays live on the device). For Drake eval, its VTK
renderer needs a display, so run headless under xvfb:
    xvfb-run -a python -m contact_study.drivers.run_eval_episode_record_controls --task cart_pole
"""

from __future__ import annotations

import os
# For Drake eval, MuJoCo must not grab a GL backend (it shadows Drake's VTK GLX
# context). Default to "disable"; override to "egl" for MuJoCo-eval rendering.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.contact_models.config import ContactModelConfig, GeometryVariant
from contact_study.evaluation.metrics import EpisodeResult
from contact_study.planners.mppi import MPPIController, MPPIConfig
from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole, EvalSimulatorKind

from contact_study.drivers.run_eval_episode import apply_cost_weight_overrides

MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
}

VIDEOS_DIR   = Path(__file__).parents[2] / "videos"
RESULTS_DIR  = Path(__file__).parents[2] / "results"
CONTROLS_DIR = Path(__file__).parents[2] / "results" / "controls"


def run_eval_episode_record_controls(
    task_name:   str,
    contact_cfg: ContactModelConfig,
    mppi_cfg:    MPPIConfig,
    rng:         np.random.Generator,
    geometry:    GeometryVariant = GeometryVariant.ACCURATE,
    cost_weight_overrides: dict | None = None,
    settle_seconds: float = 0.0,
    eval_substeps: int | None = None,
    eval_sim:    EvalSimulatorKind | None = None,
    condition:   str  = "B",
    video_path:  str | None = None,
    use_mp4:     bool = True,
    controls_path: str | None = None,
    ep_idx:      int  = 0,
    fin_ep_on_success: bool = True,
    debug:       bool = False,
    verbose:     bool = True,
) -> tuple[EpisodeResult, np.ndarray]:
    """Run one closed-loop eval episode, save a video (if requested), record
    every applied control command `u` to `controls_path`, and return the
    (EpisodeResult, controls) pair. `controls` has shape (n_steps_taken, nu).

    Identical to run_eval_episode.run_eval_episode otherwise — see that
    function's docstring for the argument semantics.
    """
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

    controller = MPPIController(
        task=rollout_task, cfg=contact_cfg, mppi_cfg=mppi_cfg, rng=rng,
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

    # One control step = controller.substeps rollout steps (resolved from the
    # config, which may specify durations instead of step counts). The eval sim
    # covers the same wall-clock with `eval_substeps` finer steps per rollout
    # step. The episode runs for the task's configured max_steps control steps.
    control_dt      = controller.control_dt
    eval_steps_per_control = controller.substeps * eval_substeps
    n_steps         = cfg.max_steps
    steps_to_success: int | None = None

    if verbose:
        print(f"  task={task_name}  eval_sim={eval_task.config.eval_sim.value}  "
              f"eval_dt={eval_dt*1e3:.2f}ms  rollout_dt={rollout_dt*1e3:.2f}ms  "
              f"control_dt={control_dt*1e3:.1f}ms  max_steps={n_steps}  "
              f"horizon={controller.horizon}  n_samples={mppi_cfg.n_samples}")

    step_times: list[float] = []
    controls_log: list[np.ndarray] = []
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
                # Multi-goal mode: target a fresh goal and keep going.
                rollout_task.sample_new_goal(mjd, rng)
                controller.reset()
                controller.lam = controller.pc.temperature
        if rollout_task.has_failed(mjd):
            if verbose:
                print(f"  step {t:4d}: task failed")
            break

        # 2. Plan on the GPU and turn the planned delta into the absolute command.
        # step_times measures only this MPPI call, not the eval-sim advance below.
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

        # Record the control command actually applied to the eval sim this step.
        controls_log.append(u.copy())

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

    controls_arr = np.stack(controls_log) if controls_log else np.empty((0,) + u.shape)
    if controls_path is not None:
        Path(controls_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(controls_path, controls_arr)
        if verbose:
            print(f"  Saved controls {controls_arr.shape} -> {controls_path}")

    final_qpos = sim.get_state().qpos
    step_arr = np.asarray(step_times)
    result = EpisodeResult(
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
    return result, controls_arr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",        type=str,   default="grasp_reorient")
    p.add_argument("--model",       type=str,   default="M2", choices=list(MODEL_FACTORIES))
    p.add_argument("--n_samples",   type=int,   default=256)
    p.add_argument("--horizon",     type=int,   default=48)
    p.add_argument("--temperature", type=float, default=12.50)
    p.add_argument("--noise_sigma", type=float, default=0.01)
    p.add_argument("--delta",       type=float, default=0.1,
                   help="Per-step MPPI delta clip magnitude (action units).")
    p.add_argument("--substeps",    type=int,   default=16,
                   help="MPPI rollout substeps per control step (control frequency knob).")
    p.add_argument("--eval_substeps", type=int, default=None,
                   help="Eval steps per rollout step (default: task config, usually 10).")
    p.add_argument("--eval_sim",    type=str,   default="mujoco",
                   choices=["none", "mujoco", "drake", "pinocchio"],
                   help="Eval simulator: 'none' uses the task default, else override it.")
    p.add_argument("--settle",      type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--video",       type=str,   default="videos/grasp_reorient_eval.mp4")
    p.add_argument("--video_format", type=str,  default="mp4", choices=["mp4", "gif"],
                   help="Video container; overrides --video's extension.")
    p.add_argument("--controls",    type=str,   default=None,
                   help="Output .npy path for the per-step control log (auto-named if omitted).")
    p.add_argument("--debug",       action="store_true",
                   help="Verbose per-step diagnostics (also enables MPPI debug).")
    args = p.parse_args()

    wp.init()
    rng = np.random.default_rng(args.seed)

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    video_path = args.video if args.video is not None else str(
        VIDEOS_DIR / f"{args.task}_eval.{args.video_format}")

    controls_path = args.controls
    if controls_path is None:
        CONTROLS_DIR.mkdir(parents=True, exist_ok=True)
        controls_path = str(CONTROLS_DIR / f"{args.task}_{args.model}_controls.npy")

    contact_cfg = MODEL_FACTORIES[args.model]()
    eval_sim = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)

    if not args.delta is None:
        delta = (-args.delta, args.delta)
    else:
        delta = (None, None)

    mppi_cfg = MPPIConfig(
        n_samples      = args.n_samples,
        step_horizon   = args.horizon,
        temperature    = args.temperature,
        noise_sigma    = args.noise_sigma,
        step_substeps  = args.substeps,
        warm_start     = False,   # match irisim_warp: keep the running mean, no shift
        use_full_graph = False,
        delta_range    = delta,
        nconmax        = 50,
        njmax          = 200,
        seed           = args.seed,
        debug          = args.debug,
    )

    result, controls = run_eval_episode_record_controls(
        task_name     = args.task,
        contact_cfg   = contact_cfg,
        mppi_cfg      = mppi_cfg,
        rng           = rng,
        video_path    = video_path,
        use_mp4       = args.video_format == "mp4",
        controls_path = controls_path,
        eval_substeps = args.eval_substeps,
        eval_sim      = eval_sim,
        settle_seconds= args.settle,
        debug         = args.debug,
        verbose       = True,
    )

    label = "✓" if result.success else "✗"
    sstr  = f"step {result.steps_to_success}" if result.steps_to_success is not None else "—"
    print(f"\n{'='*60}")
    print(f"  task={args.task}  model={args.model}  {label}  success_step={sstr}  "
          f"final_cost={result.final_cost:.4f}  "
          f"step={result.mean_step_ms:.3f}±{result.std_step_ms:.3f} ms")
    print(f"  controls shape={controls.shape} -> {controls_path}")
    print(f"{'='*60}")

    return result, controls


if __name__ == "__main__":
    main()
