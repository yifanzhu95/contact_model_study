"""compare_eval_sims.py

Side-by-side check of all available eval simulators (currently MuJoCo and
Drake), driven by an MPPI controller instead of a hand-scripted control
sweep. Generalizes contact_study/drivers/run_eval_episode.py to two eval
simulators stepped in lockstep:

  * a ROLLOUT task — owns the planning MuJoCo model + MPPIController (GPU
    rollouts), exactly as in run_eval_episode.
  * two EVAL tasks  — one MuJoCo, one Drake EvalSimulator, both reset to the
    same initial state.

Per control step: read state from the MuJoCo eval sim (the "primary" sim that
closes the loop) → mirror into the planning MjData → controller.plan() on the
GPU → integrate the delta into the command → apply the *same* command to both
eval sims → advance both → print qpos/qvel from each.

MuJoCo drives the planner; Drake is stepped open-loop on the identical command
stream so any divergence between the two is purely a contact-model/physics
effect, not a difference in what each one was told to do.

At the end, saves the qpos/qvel/control history to an .npz and plots each
qpos element over time (MuJoCo vs Drake overlaid) as a PDF. Optionally records
a gif per simulator by replaying the saved control history in its own
subprocess (MuJoCo's GL renderer and Drake's VTK renderer can't safely share a
process, so each replay happens in isolation, after the fact).

Run on a CUDA machine (warp arrays live on the device):
    python tests/compare_eval_sims.py --task cart_pole
    python tests/compare_eval_sims.py --task grasp_reorient --model M2 --video
"""

from __future__ import annotations

import os
# The comparison loop below never renders in-process (MuJoCo's GL renderer and
# Drake's VTK renderer can't safely share a process); keep MuJoCo off the GPU
# entirely here. Video capture (if requested) replays the recorded controls in
# its own subprocess per simulator, each with its own MUJOCO_GL setting.
os.environ.setdefault("MUJOCO_GL", "disable")

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.contact_models.config import ContactModelConfig, GeometryVariant
from contact_study.planners.mppi import MPPIController, MPPIConfig
from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole, EvalSimulatorKind

MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
}

ALL_EVAL_SIMS = [EvalSimulatorKind.MUJOCO, EvalSimulatorKind.DRAKE]
PRIMARY_SIM = EvalSimulatorKind.MUJOCO  # drives the MPPI feedback loop

REPO_ROOT = Path(__file__).resolve().parent.parent
THIS_SCRIPT = Path(__file__).resolve()


def run_compare(
    task_name: str,
    contact_cfg: ContactModelConfig,
    mppi_cfg: MPPIConfig,
    rng: np.random.Generator,
    geometry: GeometryVariant = GeometryVariant.ACCURATE,
    settle_seconds: float = 1.0,
    eval_substeps: int | None = None,
    max_steps: int | None = None,
    print_every: int = 25,
    out_dir: str = "figures/compare_eval_sims",
) -> Path:
    """Run one MPPI-driven episode against both eval sims in lockstep; returns
    the path of the saved .npz (qpos/qvel/control history)."""
    # ---- ROLLOUT task + planner (same as run_eval_episode) -----------------
    rollout_task = get_task(task_name, geometry=geometry, role=TaskRole.ROLLOUT)
    mjm, mjd = rollout_task.load()
    cfg = rollout_task.config

    eval_dt = cfg.timestep
    eval_substeps = eval_substeps if eval_substeps is not None else cfg.eval_substeps_per_rollout
    rollout_dt = eval_dt * eval_substeps
    mjm.opt.timestep = rollout_dt

    controller = MPPIController(task=rollout_task, cfg=contact_cfg, mppi_cfg=mppi_cfg, rng=rng)

    # ---- two EVAL tasks, one per simulator ----------------------------------
    eval_tasks = {}
    sims = {}
    for kind in ALL_EVAL_SIMS:
        t = get_task(task_name, geometry=geometry, role=TaskRole.EVAL)
        t.load()
        t.config.eval_sim = kind
        eval_tasks[kind] = t

    q0, v0, u0 = rollout_task.get_inital_state(rng)
    q0 = np.asarray(q0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    u = np.asarray(u0, dtype=float).copy() if u0 is not None else np.zeros(mjm.nu)

    for kind in ALL_EVAL_SIMS:
        sim = eval_tasks[kind].make_eval_simulator(video_path=None, render=False)
        sim.reset(q0.copy(), v0.copy())
        sims[kind] = sim

    dts = {kind: sims[kind].timestep for kind in ALL_EVAL_SIMS}
    for kind in ALL_EVAL_SIMS[1:]:
        if not np.isclose(dts[kind], dts[PRIMARY_SIM]):
            raise ValueError(f"Eval simulators disagree on timestep: "
                              f"{ {k.value: v for k, v in dts.items()} }")

    # Recorded event log (settle + control-loop commands) so a later video
    # pass can replay the exact same input each simulator saw, without
    # re-running MPPI (which would resample noise and diverge).
    event_ctrl: list[np.ndarray] = []
    event_substeps: list[int] = []

    # ---- settle phase (hold u0, advance both sims) --------------------------
    if settle_seconds > 0.0:
        n_settle = int(settle_seconds / rollout_dt)
        for _ in range(n_settle):
            for kind in ALL_EVAL_SIMS:
                sims[kind].apply_control(u)
                sims[kind].step(eval_substeps)
            event_ctrl.append(u.copy())
            event_substeps.append(eval_substeps)

    # Sample a fresh goal on the settled MuJoCo state before planning (no-op
    # for tasks without sample_new_goal, e.g. cart_pole).
    if hasattr(rollout_task, "sample_new_goal"):
        st = sims[PRIMARY_SIM].get_state()
        mjd.qpos[:] = st.qpos
        mjd.qvel[:] = st.qvel
        mujoco.mj_forward(mjm, mjd)
        rollout_task.sample_new_goal(mjd, rng)

    if cfg.force_limits is not None:
        clip_lo, clip_hi = cfg.force_limits
    elif cfg.control_limits is not None:
        clip_lo, clip_hi = cfg.control_limits
    else:
        clip_lo = clip_hi = None

    control_dt = mppi_cfg.substeps * rollout_dt
    eval_steps_per_control = mppi_cfg.substeps * eval_substeps
    n_steps = max_steps if max_steps is not None else cfg.max_steps

    print(f"task={task_name}  model={contact_cfg.label}  "
          f"eval_sims={[k.value for k in ALL_EVAL_SIMS]} (primary={PRIMARY_SIM.value})  "
          f"eval_dt={eval_dt*1e3:.2f}ms  rollout_dt={rollout_dt*1e3:.2f}ms  "
          f"control_dt={control_dt*1e3:.1f}ms  max_steps={n_steps}  "
          f"horizon={mppi_cfg.horizon}  n_samples={mppi_cfg.n_samples}")

    history_t: list[float] = []
    history_qpos = {kind: [] for kind in ALL_EVAL_SIMS}
    history_qvel = {kind: [] for kind in ALL_EVAL_SIMS}

    steps_to_success: int | None = None
    for t in range(n_steps):
        st = sims[PRIMARY_SIM].get_state()
        mjd.qpos[:] = st.qpos
        mjd.qvel[:] = st.qvel
        mjd.ctrl[:] = u
        mujoco.mj_forward(mjm, mjd)

        if rollout_task.is_success(mjd):
            if steps_to_success is None:
                steps_to_success = t
                print(f"  step {t:4d}: success (primary={PRIMARY_SIM.value})")
            break
        if rollout_task.has_failed(mjd):
            print(f"  step {t:4d}: task failed (primary={PRIMARY_SIM.value})")
            break

        action = controller.plan(mjd)
        u = u + action
        if clip_lo is not None:
            u = np.clip(u, clip_lo, clip_hi)

        states = {}
        for kind in ALL_EVAL_SIMS:
            sims[kind].apply_control(u)
            sims[kind].step(eval_steps_per_control)
            states[kind] = sims[kind].get_state()
        event_ctrl.append(u.copy())
        event_substeps.append(eval_steps_per_control)

        history_t.append(t * control_dt)
        for kind in ALL_EVAL_SIMS:
            history_qpos[kind].append(states[kind].qpos.copy())
            history_qvel[kind].append(states[kind].qvel.copy())

        if t % print_every == 0:
            print(f"  step {t:4d}  t={t*control_dt:5.2f}s  u={np.array2string(u, precision=3)}")
            for kind in ALL_EVAL_SIMS:
                st = states[kind]
                print(f"    {kind.value:7s} qpos={np.array2string(st.qpos, precision=3)}  "
                      f"qvel={np.array2string(st.qvel, precision=3)}")

    # ---- save history + control replay log ----------------------------------
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t_arr = np.asarray(history_t)
    qpos = {kind: np.asarray(history_qpos[kind]) for kind in ALL_EVAL_SIMS}
    qvel = {kind: np.asarray(history_qvel[kind]) for kind in ALL_EVAL_SIMS}

    npz_path = out_path / f"{task_name}_qpos_qvel.npz"
    np.savez(
        npz_path,
        task_name=np.array(task_name),
        t=t_arr,
        q0=q0, v0=v0,
        event_ctrl=np.asarray(event_ctrl),
        event_substeps=np.asarray(event_substeps),
        **{f"qpos_{kind.value}": qpos[kind] for kind in ALL_EVAL_SIMS},
        **{f"qvel_{kind.value}": qvel[kind] for kind in ALL_EVAL_SIMS},
    )
    print(f"  Saved qpos/qvel/control history -> {npz_path}")

    for i in range(mjm.nq):
        fig, ax = plt.subplots(figsize=(8, 4))
        for kind in ALL_EVAL_SIMS:
            ax.plot(t_arr, qpos[kind][:, i], label=kind.value)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"qpos[{i}]")
        ax.set_title(f"{task_name}: qpos[{i}] vs time (MPPI/{contact_cfg.label})")
        ax.legend()
        fig.tight_layout()
        pdf_path = out_path / f"{task_name}_qpos_{i:02d}.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"  Saved plot -> {pdf_path}")

    return npz_path


def _replay_and_record(
    npz_path: Path,
    kind: EvalSimulatorKind,
    video_path: Path,
    video_width: int | None,
    video_height: int | None,
    video_fps: float | None,
) -> None:
    """Replay a recorded control history on a single eval simulator with
    rendering on, and save the video. Runs in its own process (see module
    docstring) so MuJoCo's and Drake's renderers never coexist."""
    data = np.load(npz_path, allow_pickle=False)
    task_name = str(data["task_name"])
    q0, v0 = data["q0"], data["v0"]
    event_ctrl, event_substeps = data["event_ctrl"], data["event_substeps"]

    task = get_task(task_name, role=TaskRole.EVAL)
    task.load()
    task.config.eval_sim = kind
    if video_width is not None:
        task.config.cam_width = video_width
    if video_height is not None:
        task.config.cam_height = video_height
    if video_fps is not None:
        task.config.cam_fps = video_fps

    sim = task.make_eval_simulator(video_path=str(video_path), render=True)
    sim.reset(q0.copy(), v0.copy())
    for ctrl, substeps in zip(event_ctrl, event_substeps):
        sim.apply_control(ctrl)
        sim.step(int(substeps))
        sim.render()
    sim.save_video(str(video_path))
    print(f"  Saved video -> {video_path}")


def _record_videos(
    npz_path: Path,
    task_name: str,
    video_dir: Path,
    video_width: int | None,
    video_height: int | None,
    video_fps: float | None,
) -> None:
    """Spawn one subprocess per simulator to replay the recorded control
    history with rendering on, and save a gif each."""
    video_dir.mkdir(parents=True, exist_ok=True)
    for kind in ALL_EVAL_SIMS:
        video_path = video_dir / f"compare_eval_sims_{task_name}_{kind.value}.gif"
        cmd = [
            sys.executable, str(THIS_SCRIPT),
            "--replay_npz", str(npz_path),
            "--replay_kind", kind.value,
            "--replay_video", str(video_path),
        ]
        if video_width is not None:
            cmd += ["--video_width", str(video_width)]
        if video_height is not None:
            cmd += ["--video_height", str(video_height)]
        if video_fps is not None:
            cmd += ["--video_fps", str(video_fps)]

        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl" if kind == EvalSimulatorKind.MUJOCO else "disable"

        print(f"  Recording {kind.value} video -> {video_path}")
        subprocess.run(cmd, check=True, env=env, cwd=str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="cart_pole")
    parser.add_argument("--model", type=str, default="M2", choices=list(MODEL_FACTORIES))
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--noise_sigma", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.1,
                         help="Per-step MPPI delta clip magnitude (action units).")
    parser.add_argument("--substeps", type=int, default=10,
                         help="MPPI rollout substeps per control step (control frequency knob).")
    parser.add_argument("--eval_substeps", type=int, default=None,
                         help="Eval steps per rollout step (default: task config, usually 10).")
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=None,
                         help="Override the task's configured episode length.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--out_dir", type=str, default="figures/compare_eval_sims",
                         help="Directory to save qpos PDFs and raw data into.")
    parser.add_argument("--eval_sims", type=str, default="mujoco,drake",
                         help="Comma-separated eval simulators to compare (e.g. "
                              "'mujoco,pinocchio'). The first is the primary sim "
                              "that drives the MPPI loop. Only sims the task "
                              "supports are valid (pinocchio: grasp_reorient).")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--video", dest="record_video", action="store_true", default=False,
                         help="Record a gif per simulator by replaying the episode (default: off).")
    parser.add_argument("--video_dir", type=str, default="videos")
    parser.add_argument("--video_width", type=int, default=None)
    parser.add_argument("--video_height", type=int, default=None)
    parser.add_argument("--video_fps", type=float, default=None)

    # Internal: invoked by _record_videos() in a subprocess, one simulator at
    # a time. Not meant to be passed by hand.
    parser.add_argument("--replay_npz", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--replay_kind", type=str, default=None,
                         choices=[k.value for k in EvalSimulatorKind], help=argparse.SUPPRESS)
    parser.add_argument("--replay_video", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Select which eval simulators to compare; the first is the MPPI-driving
    # primary. run_compare/_record_videos read these module globals at runtime.
    global ALL_EVAL_SIMS, PRIMARY_SIM
    ALL_EVAL_SIMS = [EvalSimulatorKind(s.strip()) for s in args.eval_sims.split(",") if s.strip()]
    PRIMARY_SIM = ALL_EVAL_SIMS[0]

    if args.replay_npz is not None:
        _replay_and_record(
            npz_path=Path(args.replay_npz),
            kind=EvalSimulatorKind(args.replay_kind),
            video_path=Path(args.replay_video),
            video_width=args.video_width,
            video_height=args.video_height,
            video_fps=args.video_fps,
        )
        return

    wp.init()
    rng = np.random.default_rng(args.seed)
    contact_cfg = MODEL_FACTORIES[args.model]()
    mppi_cfg = MPPIConfig(
        n_samples=args.n_samples,
        horizon=args.horizon,
        temperature=args.temperature,
        noise_sigma=args.noise_sigma,
        substeps=args.substeps,
        warm_start=True,
        use_full_graph=False,
        delta_range=(-args.delta, args.delta),
        nconmax=50,
        njmax=200,
        seed=args.seed,
        debug=args.debug,
    )

    npz_path = run_compare(
        task_name=args.task,
        contact_cfg=contact_cfg,
        mppi_cfg=mppi_cfg,
        rng=rng,
        settle_seconds=args.settle,
        eval_substeps=args.eval_substeps,
        max_steps=args.max_steps,
        print_every=args.print_every,
        out_dir=args.out_dir,
    )

    if args.record_video:
        _record_videos(
            npz_path=npz_path,
            task_name=args.task,
            video_dir=Path(args.video_dir),
            video_width=args.video_width,
            video_height=args.video_height,
            video_fps=args.video_fps,
        )


if __name__ == "__main__":
    main()
