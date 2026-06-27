"""test_robot_model.py

Per-joint range-of-motion sanity check, driven through the task / EvalSimulator
interface (the same pluggable MuJoCo-or-Drake "real" environment used by
contact_study/drivers/run_eval_episode.py), so it exercises the actual eval
simulator a task will be evaluated against, not a bare XML.

For each control dimension of the task (in turn): ramp from the task's home
command to the actuator's lower limit, sweep back and forth between its lower
and upper limit a few times, then ramp back home before moving to the next
joint. Control dimensions / limits come from the task's loaded MuJoCo model
(mjm.nu, mjm.actuator_ctrlrange) — never a directly-loaded XML.

Usage:
    python tests/test_robot_model.py --task grasp_reorient
    python tests/test_robot_model.py --task cart_pole --eval_sim mujoco
    python tests/test_robot_model.py --task grasp_reorient --eval_sim drake --video out.gif
"""

from __future__ import annotations

import os
# For Drake eval, MuJoCo must not grab a GL backend (it shadows Drake's VTK GLX
# context). Default to "disable"; override to "egl" for MuJoCo-eval rendering.
os.environ.setdefault("MUJOCO_GL", "disable")

import argparse
from typing import Callable

import numpy as np
#import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole, EvalSimulatorKind


def _joint_limits(mjm, home_ctrl: np.ndarray, fallback_amplitude: float) -> np.ndarray:
    """Per-actuator (lo, hi) sweep range: ctrlrange where ctrllimited, else
    home +/- fallback_amplitude."""
    lim = np.empty((mjm.nu, 2), dtype=float)
    for i in range(mjm.nu):
        if mjm.actuator_ctrllimited[i]:
            lim[i] = mjm.actuator_ctrlrange[i]
        else:
            lim[i] = (home_ctrl[i] - fallback_amplitude, home_ctrl[i] + fallback_amplitude)
    return lim


def _build_phases(
    nu: int,
    limits: np.ndarray,
    home_ctrl: np.ndarray,
    sweep_seconds: float,
    transition_seconds: float,
    cycles: float,
) -> list[tuple[str, float, Callable[[float], np.ndarray]]]:
    """Per-joint (name, duration, ctrl_fn(elapsed_in_phase)) phase list:
    ramp home->lo, sine sweep lo<->hi for `cycles` periods, ramp back->home."""
    phases: list[tuple[str, float, Callable[[float], np.ndarray]]] = []
    freq = (cycles / sweep_seconds) if sweep_seconds > 0 else 0.0

    for i in range(nu):
        lo, hi = limits[i]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)

        def ramp_to_lo(elapsed, i=i, lo=lo):
            frac = min(elapsed / transition_seconds, 1.0) if transition_seconds > 0 else 1.0
            c = home_ctrl.copy()
            c[i] = home_ctrl[i] + (lo - home_ctrl[i]) * frac
            return c

        def sweep(elapsed, i=i, mid=mid, half=half):
            c = home_ctrl.copy()
            c[i] = mid + half * np.sin(-np.pi / 2 + 2 * np.pi * freq * elapsed)
            return c

        end_val = mid + half * np.sin(-np.pi / 2 + 2 * np.pi * freq * sweep_seconds)

        def ramp_home(elapsed, i=i, end_val=end_val):
            frac = min(elapsed / transition_seconds, 1.0) if transition_seconds > 0 else 1.0
            c = home_ctrl.copy()
            c[i] = end_val + (home_ctrl[i] - end_val) * frac
            return c

        phases.append((f"joint {i:2d}/{nu}: -> lo={lo:+.3f}",    transition_seconds, ramp_to_lo))
        phases.append((f"joint {i:2d}/{nu}: sweep [{lo:+.3f}, {hi:+.3f}]", sweep_seconds, sweep))
        phases.append((f"joint {i:2d}/{nu}: -> home={home_ctrl[i]:+.3f}", transition_seconds, ramp_home))

    return phases


def run_robot_model_test(
    task_name: str,
    eval_sim: EvalSimulatorKind | None,
    sweep_seconds: float,
    transition_seconds: float,
    cycles: float,
    fallback_amplitude: float,
    video_path: str | None,
    video_width: int | None,
    video_height: int | None,
    video_fps: float | None,
    print_every: int,
    seed: int,
) -> None:
    task = get_task(task_name, role=TaskRole.EVAL)
    task.load()
    if eval_sim is not None:
        task.config.eval_sim = eval_sim
    # Video resolution/fps: both MujocoSimulator and DrakeSimulator read these
    # off task.config (cam_width/cam_height/cam_fps). At a small eval timestep,
    # render() is called once per fine sim step, so a high resolution/fps stores
    # far more frames than the video needs and can exhaust memory on long
    # episodes — lower these if the process gets OOM-killed.
    if video_width is not None:
        task.config.cam_width = video_width
    if video_height is not None:
        task.config.cam_height = video_height
    if video_fps is not None:
        task.config.cam_fps = video_fps

    mjm = task.mjm
    nu = mjm.nu
    print(f"task={task_name}  eval_sim={task.config.eval_sim.value}  "
          f"nq={mjm.nq}  nv={mjm.nv}  nu={nu}")
    if video_path is not None:
        print(f"  video: {task.config.cam_width}x{task.config.cam_height} "
              f"@ {task.config.cam_fps}fps -> {video_path}")

    rng = np.random.default_rng(seed)
    q0, v0, ctrl0 = task.get_inital_state(rng)
    home_ctrl = np.asarray(ctrl0 if ctrl0 is not None else np.zeros(nu), dtype=float)

    limits = _joint_limits(mjm, home_ctrl, fallback_amplitude)
    phases = _build_phases(nu, limits, home_ctrl, sweep_seconds, transition_seconds, cycles)

    sim = task.make_eval_simulator(video_path=video_path, render=video_path is not None)
    sim.reset(np.asarray(q0, dtype=float), np.asarray(v0, dtype=float))
    dt = sim.timestep

    step = 0
    for name, duration, ctrl_fn in phases:
        print(f"  {name}")
        n_steps = max(1, int(round(duration / dt))) if duration > 0 else 0
        for k in range(n_steps):
            elapsed = k * dt
            ctrl = ctrl_fn(elapsed)
            sim.apply_control(ctrl)
            sim.step(1)
            sim.render()
            step += 1
            if step % print_every == 0:
                st = sim.get_state()
                print(f"    [step {step:6d}] t={elapsed:6.3f}s  "
                      f"ctrl={np.array2string(ctrl, precision=3)}  "
                      f"qpos={np.array2string(st.qpos, precision=3)}")

    if video_path is not None:
        sim.save_video(video_path)
        print(f"  Saved video -> {video_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="cart_pole",
                         help="Registered task name (e.g. cart_pole, grasp_reorient).")
    parser.add_argument("--eval_sim", type=str, default="none",
                         choices=["none", "mujoco", "drake"],
                         help="Eval simulator: 'none' uses the task default, else override it.")
    parser.add_argument("--sweep_seconds", type=float, default=1.5,
                         help="Duration of the back-and-forth sweep at each joint's limits.")
    parser.add_argument("--transition_seconds", type=float, default=0.5,
                         help="Ramp duration to/from each joint's limit and home.")
    parser.add_argument("--cycles", type=float, default=2.0,
                         help="Number of back-and-forth oscillations during sweep_seconds.")
    parser.add_argument("--fallback_amplitude", type=float, default=1.0,
                         help="+/- range to sweep for actuators with no ctrlrange (rare).")
    parser.add_argument("--print_every", type=int, default=50, help="Print state every N steps.")
    parser.add_argument("--video", type=str, default="videos/test_robot_model.gif", help="Optional video output path.")
    parser.add_argument("--video_width", type=int, default=None,
                         help="Video frame width in pixels (default: task config, usually 640). "
                              "Lower this if the process gets OOM-killed.")
    parser.add_argument("--video_height", type=int, default=None,
                         help="Video frame height in pixels (default: task config, usually 480).")
    parser.add_argument("--video_fps", type=float, default=None,
                         help="Video capture rate in fps (default: task config, usually 30). "
                              "Frames are throttled to this rate, so lowering it also cuts memory use.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    #wp.init()

    eval_sim = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)

    run_robot_model_test(
        task_name           = args.task,
        eval_sim            = eval_sim,
        sweep_seconds       = args.sweep_seconds,
        transition_seconds  = args.transition_seconds,
        cycles              = args.cycles,
        fallback_amplitude  = args.fallback_amplitude,
        video_width         = args.video_width,
        video_height        = args.video_height,
        video_fps           = args.video_fps,
        video_path          = args.video,
        print_every         = args.print_every,
        seed                = args.seed,
    )


if __name__ == "__main__":
    main()
