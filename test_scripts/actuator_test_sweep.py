"""actuator_test_sweep.py

Standalone actuator step-response check for the `actuator_test` task
(scenes/actuator_test.xml — a hinge chain, each joint driven by its own
position actuator). Drives a MuJoCo eval simulator and a Pinocchio eval
simulator in lockstep, open loop, with the identical command stream (no
MPPI/planner, no GPU needed): the selected joints move together through
0 -> +45deg -> 0 -> -45deg -> 0, holding each target for --hold_seconds, while
any non-selected joints are held at 0. At the end, plots the commanded
(desired) joint position alongside each simulator's resulting joint angle, one
subplot per driven joint.

--n_joints / --joint_start (mirroring compare_eval_sims.py's --step_n_joints /
--step_joint_start) select which actuators are driven: [joint_start,
joint_start + n_joints). By default only the first joint (index 0) is driven,
so a fresh chain gives the simplest single-actuator check; widen the range to
also exercise inter-joint mass-matrix coupling.

Run:
    python tests/actuator_test_sweep.py
    python tests/actuator_test_sweep.py --hold_seconds 2.0 --out_dir results/actuator_test
    python tests/actuator_test_sweep.py --n_joints 2                # drive both joints
    python tests/actuator_test_sweep.py --joint_start 1 --n_joints 1  # drive only joint 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import contact_study.tasks  # noqa: F401 -- registers all tasks

from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole, EvalSimulatorKind

SIMS = [EvalSimulatorKind.MUJOCO, EvalSimulatorKind.PINOCCHIO]
TARGETS_DEG = [0.0, 45.0, 0.0, -45.0, 0.0]


def run_sweep(
    hold_seconds: float,
    out_dir: str,
    n_joints: int | None = 1,
    joint_start: int = 0,
) -> Path:
    tasks = {}
    sims = {}
    for kind in SIMS:
        t = get_task("actuator_test", role=TaskRole.EVAL)
        t.load()
        t.config.eval_sim = kind
        tasks[kind] = t
        sims[kind] = t.make_eval_simulator(video_path=None, render=False)

    nu = tasks[SIMS[0]].mjm.nu
    for kind in SIMS[1:]:
        if tasks[kind].mjm.nu != nu:
            raise ValueError(f"Eval simulators disagree on actuator count: "
                              f"{ {k.value: tasks[k].mjm.nu for k in SIMS} }")

    start = max(0, min(joint_start, nu - 1))
    n = (nu - start) if n_joints is None else min(n_joints, nu - start)
    driven = list(range(start, start + n))

    dt = sims[SIMS[0]].timestep
    for kind in SIMS[1:]:
        if not np.isclose(sims[kind].timestep, dt):
            raise ValueError(
                f"Eval simulators disagree on timestep: "
                f"{ {k.value: sims[k].timestep for k in SIMS} }"
            )

    q0 = np.zeros(nu, dtype=np.float64)
    v0 = np.zeros(nu, dtype=np.float64)
    for kind in SIMS:
        sims[kind].reset(q0.copy(), v0.copy())

    n_hold = max(1, int(round(hold_seconds / dt)))

    print(f"task=actuator_test  eval_sims={[k.value for k in SIMS]}  "
          f"joints=[{start},{start+n})/{nu} driven (others held at 0)  "
          f"dt={dt*1e3:.3f}ms  hold={hold_seconds:.2f}s ({n_hold} steps/target)  "
          f"targets_deg={TARGETS_DEG}")

    history_t: list[float] = []
    history_ctrl: list[float] = []
    history_qpos = {kind: [] for kind in SIMS}

    t_elapsed = 0.0
    for target_deg in TARGETS_DEG:
        ctrl = np.zeros(nu, dtype=np.float64)
        ctrl[driven] = np.deg2rad(target_deg)
        print(f"  target={target_deg:+.1f} deg  ctrl={ctrl}")
        for _ in range(n_hold):
            for kind in SIMS:
                sims[kind].apply_control(ctrl)
                sims[kind].step(1)
            history_t.append(t_elapsed)
            history_ctrl.append(float(np.deg2rad(target_deg)))
            for kind in SIMS:
                history_qpos[kind].append(sims[kind].get_state().qpos.copy())
            t_elapsed += dt

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t_arr = np.asarray(history_t)
    ctrl_deg = np.rad2deg(np.asarray(history_ctrl))
    qpos_deg = {kind: np.rad2deg(np.asarray(history_qpos[kind])) for kind in SIMS}

    npz_path = out_path / "actuator_test_sweep.npz"
    np.savez(
        npz_path,
        t=t_arr,
        ctrl=history_ctrl,
        **{f"qpos_{kind.value}": history_qpos[kind] for kind in SIMS},
    )
    print(f"  Saved history -> {npz_path}")

    fig, axes = plt.subplots(len(driven), 1, figsize=(9, 4 * len(driven)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, j in zip(axes, driven):
        ax.plot(t_arr, ctrl_deg, label="desired (ctrl)", linestyle="--", color="black")
        for kind in SIMS:
            ax.plot(t_arr, qpos_deg[kind][:, j], label=kind.value)
        ax.set_ylabel(f"joint {j} angle [deg]")
        ax.legend()
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("actuator_test: joint angles vs time (mujoco vs pinocchio)")
    fig.tight_layout()
    pdf_path = out_path / "actuator_test_sweep.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved plot -> {pdf_path}")

    return pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold_seconds", type=float, default=1.0,
                         help="How long to hold each target before moving to the next.")
    parser.add_argument("--out_dir", type=str, default="results/actuator_test",
                         help="Directory to save the plot and raw history into.")
    parser.add_argument("--n_joints", type=int, default=1,
                         help="Number of actuators to drive, starting at --joint_start "
                              "(default: 1, i.e. just the first joint). Pass a larger "
                              "value (or omit for 'all') to also exercise inter-joint "
                              "mass-matrix coupling.")
    parser.add_argument("--joint_start", type=int, default=0,
                         help="First actuator index to drive (default: 0).")
    args = parser.parse_args()
    run_sweep(
        hold_seconds=args.hold_seconds,
        out_dir=args.out_dir,
        n_joints=args.n_joints,
        joint_start=args.joint_start,
    )


if __name__ == "__main__":
    main()
