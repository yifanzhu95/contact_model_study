"""compare_eval_sims.py

Side-by-side check of all available eval simulators (MuJoCo, Drake,
Pinocchio — default MuJoCo vs. Pinocchio; pass --eval_sims mujoco,drake to
compare against Drake instead), driven by an MPPI controller instead of a
hand-scripted control sweep. Generalizes
contact_study/drivers/run_eval_episode.py to two eval simulators stepped in
lockstep:

  * a ROLLOUT task — owns the planning MuJoCo model + MPPIController (GPU
    rollouts), exactly as in run_eval_episode.
  * two EVAL tasks  — one MuJoCo, one Pinocchio (or Drake) EvalSimulator,
    both reset to the same initial state.

Per control step: read state from the MuJoCo eval sim (the "primary" sim that
closes the loop) → mirror into the planning MjData → controller.plan() on the
GPU → integrate the delta into the command → apply the *same* command to both
eval sims → advance both → print qpos/qvel from each.

MuJoCo drives the planner; the second sim is stepped open-loop on the
identical command stream so any divergence between the two is purely a
contact-model/physics effect, not a difference in what each one was told to
do. Pinocchio is the default second sim (Drake has been prone to stalling on
some tasks); pass --eval_sims mujoco,drake to fall back to Drake. Note
cart_pole has no Pinocchio eval sim — use --eval_sims mujoco,drake for it.

Two control modes (--control_mode):
  * mppi (default) — as above: MuJoCo closes the loop through MPPI, the
    second sim replays the same command stream open-loop.
  * step — no planner/rollout model involved (no GPU needed). Each actuator
    in turn is stepped home -> home+step_amplitude -> home-step_amplitude ->
    home, holding each target for --step_hold_seconds, with both eval sims
    driven by the identical command stream. Useful for reading off step-
    response overshoot per joint, per contact model.

At the end, saves the qpos/qvel/control history to an .npz and plots each
qpos element over time (both eval sims overlaid, plus the commanded/desired
control value for any qpos slot with a driving actuator) as a PDF. Optionally
records a gif per simulator by replaying the saved control history in its own
subprocess (MuJoCo's GL renderer can't safely share a process with Drake's
VTK renderer or Pinocchio's panda3d renderer, so each replay happens in
isolation, after the fact).

--remove_cube (grasp_reorient only, ignored elsewhere): shifts the 'obj' cube
sideways out of the hand's workspace before reset, so finger motion can be
compared without any cube-contact interference — isolates the actuator/joint
model from the contact model. The offset is lateral (x), not vertical: with
zero initial velocity and uniform gravity the cube free-falls straight down
with no lateral drift, so it stays clear for the whole episode without having
to be re-teleported every step. Combining this with --control_mode mppi still
runs, but the cost function's cube pos/quat/contact terms become meaningless
once the cube is gone — pair --remove_cube with --control_mode step to isolate
pure actuator step-response instead.

Run on a CUDA machine (warp arrays live on the device) for --control_mode mppi;
--control_mode step needs no GPU:
    python tests/compare_eval_sims.py --task cart_pole
    python tests/compare_eval_sims.py --task grasp_reorient --model M2 --video
    python tests/compare_eval_sims.py --task grasp_reorient --control_mode step
    python tests/compare_eval_sims.py --task grasp_reorient --control_mode step --remove_cube
    # Step just the first finger's tip (fingertip) joint, actuator 3 — the
    # distal-most link in that finger's chain, so this exercises the PID/
    # actuator model with minimal inter-joint mass-matrix coupling, unlike
    # actuator 0 (mcp) which drags pip/dip/fingertip along with it:
    python tests/compare_eval_sims.py --task grasp_reorient --control_mode step \\
        --step_joint_start 3 --step_n_joints 1 --remove_cube
"""

from __future__ import annotations

import os
# The comparison loop below never renders in-process (MuJoCo's GL renderer
# can't safely share a process with Drake's VTK renderer or Pinocchio's
# panda3d renderer); keep MuJoCo off the GPU entirely here. Video capture (if
# requested) replays the recorded controls in its own subprocess per
# simulator, each with its own MUJOCO_GL setting.
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

ALL_EVAL_SIMS = [EvalSimulatorKind.MUJOCO, EvalSimulatorKind.PINOCCHIO]
PRIMARY_SIM = EvalSimulatorKind.MUJOCO  # drives the MPPI feedback loop

REPO_ROOT = Path(__file__).resolve().parent.parent
THIS_SCRIPT = Path(__file__).resolve()

# --remove_cube lateral offset (meters); see module docstring for why lateral
# (not vertical) keeps the cube clear of the hand for the whole episode.
_CUBE_REMOVE_OFFSET_M = 2.0


def _relocate_cube_away(mjm: mujoco.MjModel, q0: np.ndarray) -> np.ndarray:
    """grasp_reorient only: shift the 'obj' free body's initial x-position by
    _CUBE_REMOVE_OFFSET_M, well outside the hand's workspace. Returns a
    modified copy of q0; a no-op if the model has no 'obj_joint' (i.e. any
    task other than grasp_reorient)."""
    obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint")
    if obj_jnt < 0:
        return q0
    q0 = q0.copy()
    q0[mjm.jnt_qposadr[obj_jnt]] += _CUBE_REMOVE_OFFSET_M
    return q0

def _actuator_qpos_map(mjm: mujoco.MjModel) -> dict[int, int]:
    """qpos index -> actuator index, for actuators that drive a single hinge/
    slide joint 1:1 (the usual position-actuated robot joint). Used to overlay
    the commanded/desired control value on the matching qpos subplot."""
    mapping: dict[int, int] = {}
    for a in range(mjm.nu):
        jid = int(mjm.actuator_trnid[a, 0])
        if jid < 0:
            continue
        if mjm.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            mapping.setdefault(int(mjm.jnt_qposadr[jid]), a)
    return mapping


def _step_command_sequence(
    nu: int,
    home_ctrl: np.ndarray,
    ctrllimited: np.ndarray,
    ctrlrange: np.ndarray,
    step_amplitude: float,
    n_joints: int | None = None,
    joint_start: int = 0,
) -> list[tuple[str, np.ndarray]]:
    """Per-joint step sequence: home -> home+amplitude -> home-amplitude ->
    home, one joint at a time (all others held at home), clipped to the
    actuator's ctrlrange where limited. Steps actuators
    [joint_start, joint_start + n_joints) (default: joint_start=0, all nu).

    joint_start lets you target a specific actuator instead of always starting
    from 0 — e.g. for grasp_reorient, actuator 3 is the first finger's tip
    (fingertip) joint, the distal-most link in that finger's chain. Stepping it
    (vs. actuator 0, the mcp/base joint) minimizes inter-joint mass-matrix
    coupling, since the fingertip has no child links to drag along — useful for
    isolating PID/actuator-model divergence from chain-coupling effects."""
    seq: list[tuple[str, np.ndarray]] = []
    start = max(0, min(joint_start, nu - 1))
    n = (nu - start) if n_joints is None else min(n_joints, nu - start)
    for i in range(start, start + n):
        for sign, label in ((+1, "+"), (-1, "-"), (0, "home")):
            c = home_ctrl.copy()
            if sign != 0:
                target = home_ctrl[i] + sign * step_amplitude
                if ctrllimited[i]:
                    target = float(np.clip(target, ctrlrange[i, 0], ctrlrange[i, 1]))
                c[i] = target
            seq.append((f"joint {i:2d}/{nu} {label}", c))
    return seq


def run_compare(
    task_name: str,
    contact_cfg: ContactModelConfig,
    mppi_cfg: MPPIConfig,
    rng: np.random.Generator,
    control_mode: str = "mppi",
    geometry: GeometryVariant = GeometryVariant.ACCURATE,
    settle_seconds: float = 1.0,
    eval_substeps: int | None = None,
    max_steps: int | None = None,
    print_every: int = 25,
    out_dir: str = "figures/compare_eval_sims",
    step_amplitude: float = np.deg2rad(45.0),
    step_hold_seconds: float = 1.0,
    step_n_joints: int | None = None,
    step_joint_start: int = 0,
    remove_cube: bool = False,
) -> Path:
    """Run one episode against both eval sims in lockstep; returns the path of
    the saved .npz (qpos/qvel/control history).

    control_mode="mppi": MuJoCo closes the loop through MPPI (GPU rollouts);
    Drake replays the identical command stream open-loop.
    control_mode="step": no planner — each actuator is stepped in turn
    home -> +step_amplitude -> -step_amplitude -> home, driving both eval sims
    with the identical command stream. No GPU/rollout model needed.

    remove_cube: grasp_reorient only (silently ignored otherwise) — shifts the
    'obj' cube out of the hand's workspace before reset, isolating actuator/
    joint dynamics from cube-contact effects. See module docstring.
    """
    if control_mode not in ("mppi", "step"):
        raise ValueError(f"Unknown control_mode: {control_mode!r}")

    # ---- two EVAL tasks, one per simulator ----------------------------------
    eval_tasks = {}
    sims = {}
    for kind in ALL_EVAL_SIMS:
        t = get_task(task_name, geometry=geometry, role=TaskRole.EVAL)
        t.load()
        t.config.eval_sim = kind
        eval_tasks[kind] = t

    ref_task = eval_tasks[PRIMARY_SIM]
    mjm = ref_task.mjm
    cfg = ref_task.config
    eval_dt = cfg.timestep
    eval_substeps = eval_substeps if eval_substeps is not None else cfg.eval_substeps_per_rollout

    # ---- ROLLOUT task + planner (mppi mode only; same as run_eval_episode) --
    if control_mode == "mppi":
        rollout_task = get_task(task_name, geometry=geometry, role=TaskRole.ROLLOUT)
        rollout_mjm, mjd = rollout_task.load()
        rollout_dt = eval_dt * eval_substeps
        rollout_mjm.opt.timestep = rollout_dt
        controller = MPPIController(task=rollout_task, cfg=contact_cfg, mppi_cfg=mppi_cfg, rng=rng)
        state_task = rollout_task
    else:
        state_task = ref_task

    q0, v0, u0 = state_task.get_inital_state(rng)
    q0 = np.asarray(q0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    u = np.asarray(u0, dtype=float).copy() if u0 is not None else np.zeros(mjm.nu)

    if remove_cube:
        if task_name == "grasp_reorient":
            q0 = _relocate_cube_away(mjm, q0)
            print(f"  --remove_cube: shifted 'obj' +{_CUBE_REMOVE_OFFSET_M:.1f}m in x "
                  f"(clear of the hand for the whole episode)")
        else:
            print(f"  --remove_cube ignored: task={task_name!r} is not grasp_reorient")

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
    settle_dt = (eval_dt * eval_substeps) if control_mode == "mppi" else eval_dt
    settle_substeps = eval_substeps if control_mode == "mppi" else 1
    if settle_seconds > 0.0:
        n_settle = int(settle_seconds / settle_dt)
        for _ in range(n_settle):
            for kind in ALL_EVAL_SIMS:
                sims[kind].apply_control(u)
                sims[kind].step(settle_substeps)
            event_ctrl.append(u.copy())
            event_substeps.append(settle_substeps)

    history_t: list[float] = []
    history_qpos = {kind: [] for kind in ALL_EVAL_SIMS}
    history_qvel = {kind: [] for kind in ALL_EVAL_SIMS}
    history_ctrl: list[np.ndarray] = []

    if control_mode == "mppi":
        # Sample a fresh goal on the settled MuJoCo state before planning
        # (no-op for tasks without sample_new_goal, e.g. cart_pole).
        if hasattr(rollout_task, "sample_new_goal"):
            st = sims[PRIMARY_SIM].get_state()
            mjd.qpos[:] = st.qpos
            mjd.qvel[:] = st.qvel
            mujoco.mj_forward(rollout_mjm, mjd)
            rollout_task.sample_new_goal(mjd, rng)

        if cfg.force_limits is not None:
            clip_lo, clip_hi = cfg.force_limits
        elif cfg.control_limits is not None:
            clip_lo, clip_hi = cfg.control_limits
        else:
            clip_lo = clip_hi = None

        control_dt = controller.control_dt
        eval_steps_per_control = controller.substeps * eval_substeps
        n_steps = max_steps if max_steps is not None else cfg.max_steps

        print(f"task={task_name}  control_mode=mppi  model={contact_cfg.label}  "
              f"eval_sims={[k.value for k in ALL_EVAL_SIMS]} (primary={PRIMARY_SIM.value})  "
              f"eval_dt={eval_dt*1e3:.2f}ms  rollout_dt={rollout_dt*1e3:.2f}ms  "
              f"control_dt={control_dt*1e3:.1f}ms  max_steps={n_steps}  "
              f"horizon={controller.horizon}  n_samples={mppi_cfg.n_samples}")

        for t in range(n_steps):
            st = sims[PRIMARY_SIM].get_state()
            mjd.qpos[:] = st.qpos
            mjd.qvel[:] = st.qvel
            mjd.ctrl[:] = u
            mujoco.mj_forward(rollout_mjm, mjd)

            if rollout_task.is_success(mjd):
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
            history_ctrl.append(u.copy())
            for kind in ALL_EVAL_SIMS:
                history_qpos[kind].append(states[kind].qpos.copy())
                history_qvel[kind].append(states[kind].qvel.copy())

            if t % print_every == 0:
                print(f"  step {t:4d}  t={t*control_dt:5.2f}s  u={np.array2string(u, precision=3)}")
                for kind in ALL_EVAL_SIMS:
                    st = states[kind]
                    print(f"    {kind.value:7s} qpos={np.array2string(st.qpos, precision=3)}  "
                          f"qvel={np.array2string(st.qvel, precision=3)}")

    else:  # control_mode == "step"
        seq = _step_command_sequence(
            mjm.nu, u, mjm.actuator_ctrllimited.astype(bool), mjm.actuator_ctrlrange, step_amplitude,
            n_joints=step_n_joints, joint_start=step_joint_start,
        )
        n_hold = max(1, int(round(step_hold_seconds / eval_dt)))
        max_total = max_steps if max_steps is not None else len(seq) * n_hold
        start = max(0, min(step_joint_start, mjm.nu - 1))
        n_stepped = (mjm.nu - start) if step_n_joints is None else min(step_n_joints, mjm.nu - start)

        print(f"task={task_name}  control_mode=step  "
              f"eval_sims={[k.value for k in ALL_EVAL_SIMS]} (primary={PRIMARY_SIM.value})  "
              f"eval_dt={eval_dt*1e3:.2f}ms  step_amplitude={np.rad2deg(step_amplitude):.1f}deg  "
              f"step_hold={step_hold_seconds:.2f}s  joints=[{start},{start+n_stepped})/{mjm.nu}  "
              f"max_steps={max_total}")

        t_elapsed = 0.0
        step_idx = 0
        for name, target_ctrl in seq:
            if step_idx >= max_total:
                break
            print(f"  {name}: ctrl={np.array2string(target_ctrl, precision=3)}")
            for _ in range(n_hold):
                if step_idx >= max_total:
                    break
                states = {}
                for kind in ALL_EVAL_SIMS:
                    sims[kind].apply_control(target_ctrl)
                    sims[kind].step(1)
                    states[kind] = sims[kind].get_state()
                event_ctrl.append(target_ctrl.copy())
                event_substeps.append(1)

                history_t.append(t_elapsed)
                history_ctrl.append(target_ctrl.copy())
                for kind in ALL_EVAL_SIMS:
                    history_qpos[kind].append(states[kind].qpos.copy())
                    history_qvel[kind].append(states[kind].qvel.copy())

                if step_idx % print_every == 0:
                    for kind in ALL_EVAL_SIMS:
                        st = states[kind]
                        print(f"    step {step_idx:6d} t={t_elapsed:6.3f}s  {kind.value:7s} "
                              f"qpos={np.array2string(st.qpos, precision=3)}")

                t_elapsed += eval_dt
                step_idx += 1

    # ---- save history + control replay log ----------------------------------
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t_arr = np.asarray(history_t)
    ctrl_arr = np.asarray(history_ctrl)
    qpos = {kind: np.asarray(history_qpos[kind]) for kind in ALL_EVAL_SIMS}
    qvel = {kind: np.asarray(history_qvel[kind]) for kind in ALL_EVAL_SIMS}

    npz_path = out_path / f"{task_name}_qpos_qvel.npz"
    np.savez(
        npz_path,
        task_name=np.array(task_name),
        t=t_arr,
        q0=q0, v0=v0,
        ctrl=ctrl_arr,
        event_ctrl=np.asarray(event_ctrl),
        event_substeps=np.asarray(event_substeps),
        **{f"qpos_{kind.value}": qpos[kind] for kind in ALL_EVAL_SIMS},
        **{f"qvel_{kind.value}": qvel[kind] for kind in ALL_EVAL_SIMS},
    )
    print(f"  Saved qpos/qvel/control history -> {npz_path}")

    actuator_qpos_map = _actuator_qpos_map(mjm)
    title_tag = contact_cfg.label if control_mode == "mppi" else f"step/{contact_cfg.label}"
    for i in range(mjm.nq):
        fig, ax = plt.subplots(figsize=(8, 4))
        for kind in ALL_EVAL_SIMS:
            ax.plot(t_arr, qpos[kind][:, i], label=kind.value)
        if i in actuator_qpos_map:
            ax.plot(t_arr, ctrl_arr[:, actuator_qpos_map[i]], label="desired (ctrl)",
                     linestyle="--", color="black")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"qpos[{i}]")
        ax.set_title(f"{task_name}: qpos[{i}] vs time ({title_tag})")
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
    print(f"  Saved video -> {sim.save_video(str(video_path))}")


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
        video_path = video_dir / f"compare_eval_sims_{task_name}_{kind.value}.mp4"
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
    parser.add_argument("--control_mode", type=str, default="mppi", choices=["mppi", "step"],
                         help="'mppi': MuJoCo closes the loop through MPPI, Drake replays "
                              "the same commands open-loop (needs a CUDA machine). "
                              "'step': no planner — steps each actuator in turn through "
                              "+/-step_amplitude and back home (no GPU needed).")
    parser.add_argument("--step_amplitude_deg", type=float, default=45.0,
                         help="[step mode] +/- step size per actuator, in degrees.")
    parser.add_argument("--step_hold_seconds", type=float, default=0.5,
                         help="[step mode] How long to hold each step target before moving on.")
    parser.add_argument("--step_n_joints", type=int, default=1,
                         help="[step mode] Only step N actuators starting at --step_joint_start "
                              "(default: all).")
    parser.add_argument("--step_joint_start", type=int, default=0,
                         help="[step mode] First actuator index to step (default: 0). "
                              "For grasp_reorient, actuator 3 is the first finger's tip "
                              "(fingertip) joint — the distal-most link, with no children to "
                              "drag along — so '--step_joint_start 3 --step_n_joints 1' isolates "
                              "PID/actuator-model divergence from the inter-joint mass-matrix "
                              "coupling that stepping actuator 0 (mcp) exercises.")
    parser.add_argument("--remove_cube", action="store_true", default=False,
                         help="grasp_reorient only (ignored for other tasks): shift the "
                              "cube out of the hand's workspace before reset so actuator/"
                              "joint dynamics can be compared without cube-contact "
                              "interference.")
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
    parser.add_argument("--settle", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=None,
                         help="Override the task's configured episode length.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--out_dir", type=str, default="results/compare_eval_sims",
                         help="Directory to save qpos PDFs and raw data into.")
    parser.add_argument("--eval_sims", type=str, default="mujoco,pinocchio",
                         help="Comma-separated eval simulators to compare (e.g. "
                              "'mujoco,drake'). The first is the primary sim "
                              "that drives the MPPI loop. Only sims the task "
                              "supports are valid (pinocchio: grasp_reorient; "
                              "cart_pole has no pinocchio eval sim, pass "
                              "--eval_sims mujoco,drake for that task).")
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

    rng = np.random.default_rng(args.seed)
    contact_cfg = MODEL_FACTORIES[args.model]()

    if not args.delta is None:
        delta = (-args.delta, args.delta)
    else:
        delta = (None, None)

    mppi_cfg = MPPIConfig(
        n_samples=args.n_samples,
        step_horizon=args.horizon,
        temperature=args.temperature,
        noise_sigma=args.noise_sigma,
        step_substeps=args.substeps,
        warm_start=True,
        use_full_graph=False,
        delta_range=delta,
        nconmax=50,
        njmax=200,
        seed=args.seed,
        debug=args.debug,
    )

    if args.control_mode == "mppi":
        wp.init()

    npz_path = run_compare(
        task_name=args.task,
        contact_cfg=contact_cfg,
        mppi_cfg=mppi_cfg,
        rng=rng,
        control_mode=args.control_mode,
        settle_seconds=args.settle,
        eval_substeps=args.eval_substeps,
        max_steps=args.max_steps,
        print_every=args.print_every,
        out_dir=args.out_dir,
        step_amplitude=np.deg2rad(args.step_amplitude_deg),
        step_hold_seconds=args.step_hold_seconds,
        step_n_joints=args.step_n_joints,
        step_joint_start=args.step_joint_start,
        remove_cube=args.remove_cube,
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
