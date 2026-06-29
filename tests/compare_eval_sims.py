"""compare_eval_sims.py

Side-by-side sanity check of all available eval simulators (currently MuJoCo
and Drake), driven through the same EvalSimulator interface used by
contact_study/drivers/run_eval_episode.py. Builds the same per-joint
ramp/sweep/ramp-home control sequence as test_robot_model.py, applies it to
both simulators in lockstep (same task, same initial state, same controls,
same timestep), and prints qpos/qvel from each at every step.

At the end, plots each qpos element over time (MuJoCo vs Drake overlaid) and
saves the figures as PDFs. Also records a per-simulator video (gif) of the
same control sequence into the videos/ folder, by shelling out to
test_robot_model.py once per simulator — MuJoCo's GL renderer and Drake's VTK
renderer can't safely share a process (see MUJOCO_GL note below), so each
video is captured in its own subprocess.

Usage:
    python tests/compare_eval_sims.py --task grasp_reorient
    python tests/compare_eval_sims.py --task cart_pole --out_dir figures/compare_eval_sims
    python tests/compare_eval_sims.py --task cart_pole --no_video
"""

from __future__ import annotations

import os
# Drake's VTK GLX context and MuJoCo's GL backend can't coexist; the
# comparison loop below doesn't render either simulator, so keep MuJoCo off
# the GPU entirely. Video capture happens in separate subprocesses (one
# simulator per process), each with its own MUJOCO_GL setting.
os.environ.setdefault("MUJOCO_GL", "disable")

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.tasks.base import get_task
from contact_study.tasks.config import EvalSimulatorKind, TaskRole

ALL_EVAL_SIMS = [EvalSimulatorKind.MUJOCO, EvalSimulatorKind.DRAKE]
REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_ROBOT_MODEL = Path(__file__).resolve().parent / "test_robot_model.py"


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


def _record_videos(
    task_name: str,
    sweep_seconds: float,
    transition_seconds: float,
    cycles: float,
    fallback_amplitude: float,
    seed: int,
    video_dir: Path,
    video_width: int | None,
    video_height: int | None,
    video_fps: float | None,
) -> None:
    """Capture one gif per eval simulator by running test_robot_model.py in
    its own subprocess (one simulator per process avoids MuJoCo's GL renderer
    and Drake's VTK renderer fighting over the same GL context)."""
    video_dir.mkdir(parents=True, exist_ok=True)
    for kind in ALL_EVAL_SIMS:
        video_path = video_dir / f"compare_eval_sims_{task_name}_{kind.value}.gif"
        cmd = [
            sys.executable, str(TEST_ROBOT_MODEL),
            "--task", task_name,
            "--eval_sim", kind.value,
            "--sweep_seconds", str(sweep_seconds),
            "--transition_seconds", str(transition_seconds),
            "--cycles", str(cycles),
            "--fallback_amplitude", str(fallback_amplitude),
            "--seed", str(seed),
            "--video", str(video_path),
        ]
        if video_width is not None:
            cmd += ["--video_width", str(video_width)]
        if video_height is not None:
            cmd += ["--video_height", str(video_height)]
        if video_fps is not None:
            cmd += ["--video_fps", str(video_fps)]

        env = os.environ.copy()
        # MuJoCo needs a real GL backend to render; Drake's own VTK renderer
        # is unaffected by MUJOCO_GL, but only one simulator runs per
        # subprocess, so there's no shadowing conflict to avoid here.
        env["MUJOCO_GL"] = "egl" if kind == EvalSimulatorKind.MUJOCO else "disable"

        print(f"  Recording {kind.value} video -> {video_path}")
        subprocess.run(cmd, check=True, env=env, cwd=str(REPO_ROOT))


def run_compare(
    task_name: str,
    sweep_seconds: float,
    transition_seconds: float,
    cycles: float,
    fallback_amplitude: float,
    print_every: int,
    seed: int,
    out_dir: str,
    record_video: bool,
    video_dir: str,
    video_width: int | None,
    video_height: int | None,
    video_fps: float | None,
) -> None:
    # One task object per simulator: make_eval_simulator() reads its kind off
    # task.config.eval_sim, and each task carries its own MjModel/MjData.
    tasks = {}
    sims = {}
    for kind in ALL_EVAL_SIMS:
        task = get_task(task_name, role=TaskRole.EVAL)
        task.load()
        task.config.eval_sim = kind
        tasks[kind] = task

    mjm = tasks[ALL_EVAL_SIMS[0]].mjm
    nu = mjm.nu
    print(f"task={task_name}  nq={mjm.nq}  nv={mjm.nv}  nu={nu}  "
          f"eval_sims={[k.value for k in ALL_EVAL_SIMS]}")

    rng = np.random.default_rng(seed)
    q0, v0, ctrl0 = tasks[ALL_EVAL_SIMS[0]].get_inital_state(rng)
    q0 = np.asarray(q0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    home_ctrl = np.asarray(ctrl0 if ctrl0 is not None else np.zeros(nu), dtype=float)

    limits = _joint_limits(mjm, home_ctrl, fallback_amplitude)
    phases = _build_phases(nu, limits, home_ctrl, sweep_seconds, transition_seconds, cycles)

    for kind in ALL_EVAL_SIMS:
        sim = tasks[kind].make_eval_simulator(video_path=None, render=False)
        sim.reset(q0.copy(), v0.copy())
        sims[kind] = sim

    dts = {kind: sims[kind].timestep for kind in ALL_EVAL_SIMS}
    dt = dts[ALL_EVAL_SIMS[0]]
    for kind in ALL_EVAL_SIMS[1:]:
        if not np.isclose(dts[kind], dt):
            raise ValueError(
                f"Eval simulators disagree on timestep: "
                f"{ {k.value: v for k, v in dts.items()} }"
            )

    history_t: list[float] = []
    history_qpos = {kind: [] for kind in ALL_EVAL_SIMS}
    history_qvel = {kind: [] for kind in ALL_EVAL_SIMS}

    step = 0
    t = 0.0
    for name, duration, ctrl_fn in phases:
        print(f"  {name}")
        n_steps = max(1, int(round(duration / dt))) if duration > 0 else 0
        for k in range(n_steps):
            elapsed = k * dt
            ctrl = ctrl_fn(elapsed)

            states = {}
            for kind in ALL_EVAL_SIMS:
                sim = sims[kind]
                sim.apply_control(ctrl)
                sim.step(1)
                states[kind] = sim.get_state()

            history_t.append(t)
            for kind in ALL_EVAL_SIMS:
                history_qpos[kind].append(states[kind].qpos.copy())
                history_qvel[kind].append(states[kind].qvel.copy())

            step += 1
            t += dt
            if step % print_every == 0:
                print(f"    [step {step:6d}] t={t:6.3f}s  ctrl={np.array2string(ctrl, precision=3)}")
                for kind in ALL_EVAL_SIMS:
                    st = states[kind]
                    print(f"      {kind.value:7s} qpos={np.array2string(st.qpos, precision=3)}  "
                          f"qvel={np.array2string(st.qvel, precision=3)}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t_arr = np.asarray(history_t)
    qpos = {kind: np.asarray(history_qpos[kind]) for kind in ALL_EVAL_SIMS}

    npz_path = out_path / f"{task_name}_qpos_qvel.npz"
    np.savez(
        npz_path,
        t=t_arr,
        **{f"qpos_{kind.value}": qpos[kind] for kind in ALL_EVAL_SIMS},
        **{f"qvel_{kind.value}": np.asarray(history_qvel[kind]) for kind in ALL_EVAL_SIMS},
    )
    print(f"  Saved raw qpos/qvel -> {npz_path}")

    for i in range(mjm.nq):
        fig, ax = plt.subplots(figsize=(8, 4))
        for kind in ALL_EVAL_SIMS:
            ax.plot(t_arr, qpos[kind][:, i], label=kind.value)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"qpos[{i}]")
        ax.set_title(f"{task_name}: qpos[{i}] vs time")
        ax.legend()
        fig.tight_layout()
        pdf_path = out_path / f"{task_name}_qpos_{i:02d}.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"  Saved plot -> {pdf_path}")

    if record_video:
        _record_videos(
            task_name           = task_name,
            sweep_seconds       = sweep_seconds,
            transition_seconds  = transition_seconds,
            cycles              = cycles,
            fallback_amplitude  = fallback_amplitude,
            seed                = seed,
            video_dir           = Path(video_dir),
            video_width         = video_width,
            video_height        = video_height,
            video_fps           = video_fps,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="cart_pole",
                         help="Registered task name (e.g. cart_pole, grasp_reorient).")
    parser.add_argument("--sweep_seconds", type=float, default=1.5,
                         help="Duration of the back-and-forth sweep at each joint's limits.")
    parser.add_argument("--transition_seconds", type=float, default=0.5,
                         help="Ramp duration to/from each joint's limit and home.")
    parser.add_argument("--cycles", type=float, default=1.0,
                         help="Number of back-and-forth oscillations during sweep_seconds.")
    parser.add_argument("--fallback_amplitude", type=float, default=1.0,
                         help="+/- range to sweep for actuators with no ctrlrange (rare).")
    parser.add_argument("--print_every", type=int, default=50, help="Print state every N steps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="figures/compare_eval_sims",
                         help="Directory to save qpos PDFs and raw data into.")
    parser.add_argument("--video", dest="record_video", action="store_true", default=True,
                         help="Record a gif per simulator (default: on).")
    parser.add_argument("--no_video", dest="record_video", action="store_false",
                         help="Skip video recording.")
    parser.add_argument("--video_dir", type=str, default="videos",
                         help="Directory to save per-simulator gifs into.")
    parser.add_argument("--video_width", type=int, default=None,
                         help="Video frame width in pixels (default: task config).")
    parser.add_argument("--video_height", type=int, default=None,
                         help="Video frame height in pixels (default: task config).")
    parser.add_argument("--video_fps", type=float, default=None,
                         help="Video capture rate in fps (default: task config).")
    args = parser.parse_args()

    run_compare(
        task_name           = args.task,
        sweep_seconds       = args.sweep_seconds,
        transition_seconds  = args.transition_seconds,
        cycles              = args.cycles,
        fallback_amplitude  = args.fallback_amplitude,
        print_every         = args.print_every,
        seed                = args.seed,
        out_dir             = args.out_dir,
        record_video        = args.record_video,
        video_dir           = args.video_dir,
        video_width         = args.video_width,
        video_height        = args.video_height,
        video_fps           = args.video_fps,
    )


if __name__ == "__main__":
    main()
