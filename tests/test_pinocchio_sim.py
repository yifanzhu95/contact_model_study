"""Standalone sanity check for PinocchioSimulator's constraint-based PD.

Builds the grasp_reorient EVAL task (real PinocchioSimulator, not a mock),
resets to the task's default grasp state, then commands all 16 finger joints
first to their lower ctrlrange limit and then to their upper limit, holding
each extreme for a while. Logs the max coal-detected finger<->cube /
finger<->finger penetration and the cube's linear speed every substep, so a
fling or persistent deep penetration shows up immediately.

Run (needs pinocchio; add --video to also render):
    python tests/test_pinocchio_sim.py
    python tests/test_pinocchio_sim.py --video videos/test_pinocchio_sim.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import contact_study.tasks  # noqa: F401 — registers all tasks
from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole


def run(seconds_per_extreme: float, video_path: str | None):
    task = get_task("grasp_reorient", role=TaskRole.EVAL)
    task.load()
    mjm = task.mjm

    q0, v0, ctrl0 = task.get_inital_state(np.random.default_rng(0))
    sim = task.make_eval_simulator(video_path=video_path, render=video_path is not None)
    sim.reset(q0, v0)

    # Probe coal penetration each substep (same pattern as
    # tests/test_pinocchio_baumgarte_ab.py): wrap _detect_contacts.
    probe = {"pens": None, "ncon": 0}
    orig_detect = sim._detect_contacts

    def _detect_and_log():
        cms, cds, pens = orig_detect()
        probe["pens"] = pens
        probe["ncon"] = len(cms)
        return cms, cds, pens

    sim._detect_contacts = _detect_and_log

    lo = mjm.actuator_ctrlrange[:, 0].copy()
    hi = mjm.actuator_ctrlrange[:, 1].copy()

    n_substeps = int(round(seconds_per_extreme / sim.timestep))
    obj_qadr = int(task.index_vector[0])
    obj_vadr = int(task.index_vector[1])

    print(f"grasp_reorient / pinocchio eval: nq={mjm.nq} nv={mjm.nv} nu={mjm.nu}  "
          f"dt={sim.timestep*1e3:.3f}ms  n_substeps/extreme={n_substeps}")

    overall_peak_pen = 0.0
    overall_peak_speed = 0.0
    for label, target in [("default->lower limits", lo), ("lower->upper limits", hi)]:
        pen_hist, speed_hist = [], []
        for i in range(n_substeps):
            print(i)
            sim.apply_control(target)
            sim.step(1)
            st = sim.get_state()
            pens = probe["pens"]
            pmax = float(np.max(pens)) * 1e3 if pens is not None and pens.size else 0.0
            speed = float(np.linalg.norm(st.qvel[obj_vadr:obj_vadr + 3]))
            pen_hist.append(pmax)
            speed_hist.append(speed)
            if video_path is not None:
                sim.render()
            if not np.all(np.isfinite(st.qpos)):
                print(f"  !!! NON-FINITE state during '{label}' at substep {i}")
                break

        peak_pen = max(pen_hist) if pen_hist else 0.0
        final_pen = pen_hist[-1] if pen_hist else 0.0
        peak_speed = max(speed_hist) if speed_hist else 0.0
        overall_peak_pen = max(overall_peak_pen, peak_pen)
        overall_peak_speed = max(overall_peak_speed, peak_speed)
        print(f"  [{label:24s}] peak_pen={peak_pen:6.3f}mm  final_pen={final_pen:6.3f}mm  "
              f"peak_cube_speed={peak_speed:7.4f}m/s  final_ncon={probe['ncon']}")

    if video_path is not None:
        sim.save_video(video_path)
        print(f"  video -> {video_path}")

    print(f"\nOverall peak penetration: {overall_peak_pen:.3f}mm   "
          f"overall peak cube speed: {overall_peak_speed:.4f}m/s")
    return overall_peak_pen, overall_peak_speed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds_per_extreme", type=float, default=0.1,
                   help="Sim seconds spent driving toward (and holding at) each "
                        "joint-limit extreme.")
    p.add_argument("--video", type=str, default=None,
                   help="If set, render and save an mp4/gif to this path.")
    args = p.parse_args()
    run(args.seconds_per_extreme, args.video)


if __name__ == "__main__":
    main()
