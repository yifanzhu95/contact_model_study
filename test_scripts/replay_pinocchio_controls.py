"""Closed-loop MPPI eval on the real contact_study backend, to test whether the
Pinocchio/ADMM eval simulator lets the cube penetrate and stick to the LEAP
hand's fingers.

This used to be a standalone, hand-rolled duplicate of the Pinocchio model-
build + ADMM contact solve (same MJCF, same collision pairs, same PD/Baumgarte
scheme as contact_study.contact_models.pinocchio_sim, reimplemented from
scratch) that replayed a pre-recorded control log through it. That duplication
is exactly the risk when chasing a penetration/sticking bug: a discrepancy
could be a bug in the real eval sim, or just a divergence between this script
and it. This version instead drives the ACTUAL production path —

  * the grasp_reorient TASK (contact_study/tasks/grasp_reorient.py) — same
    scene, same initial state, same goal sampling as a real eval run;
  * the MPPI PLANNER (contact_study/planners/mppi.py) — closed-loop control,
    not a fixed pre-recorded log;
  * the PinocchioSimulator EVAL BACKEND (contact_study/contact_models/
    pinocchio_sim.py) — via GraspReorientTask.make_eval_simulator(), the same
    object run_eval_episode.py builds for a real sweep;

by calling contact_study.drivers.run_eval_episode.run_eval_episode() directly,
with PinocchioSimulator's own contact/ADMM diagnostics (pinocchio_sim.py's
PRINT_CONTACT_DEBUG flag and per-instance diagnostics()) turned on so
penetration depth and ADMM convergence can actually be inspected across the
episode instead of only being visible (or not) in the rendered video.

Run headless under xvfb (Panda3D's EGL pipe still needs a display context) on
a CUDA machine (MPPI's rollouts live on the GPU):
    xvfb-run -a python tests/replay_pinocchio_controls.py --n_episodes 1
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

# Registers all tasks (incl. "grasp_reorient") as a side effect, and sets
# MUJOCO_GL=egl (default) so MuJoCo doesn't grab a GL context Pinocchio/Panda3d
# would otherwise need for offscreen rendering.
from contact_study.drivers.run_eval_episode import run_eval_episode, MODEL_FACTORIES
from contact_study.planners import make_planner_config
from contact_study.tasks.config import EvalSimulatorKind
from contact_study.tasks.grasp_reorient import GraspReorientTask
import contact_study.contact_models.pinocchio_sim as pinocchio_sim


def _print_diag(ep_idx: int, diag: dict) -> None:
    n_sub = diag["n_substeps"] or 1
    worst_mm = diag["min_penetration_m"] * 1e3
    print(
        f"  [ep {ep_idx:02d}] diagnostics: {diag['n_substeps']} fine substeps, "
        f"{diag['n_contact_substeps']} with >=1 contact "
        f"({100.0 * diag['n_contact_substeps'] / n_sub:.1f}%), "
        f"max simultaneous contacts={diag['max_n_contacts']}, "
        f"worst penetration={worst_mm:+.3f} mm "
        f"(negative = overlapping; 0 = never penetrated), "
        f"ADMM non-converged substeps={diag['n_nonconverged']} "
        f"({100.0 * diag['n_nonconverged'] / n_sub:.1f}%)"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, default="M2", choices=list(MODEL_FACTORIES),
                    help="Rollout contact model (orthogonal to the Pinocchio eval sim).")
    p.add_argument("--difficulty", type=int, default=None,
                    help="Override GraspReorientTask.goal_difficulty (default: the "
                         "task's own default, 6 = adjacent face, no twist).")
    # --- MPPI knobs (mirrors run_eval_episode.py's CLI defaults) -----------
    p.add_argument("--n_samples", type=int, default=256)
    p.add_argument("--time_horizon", type=float, default=0.352)
    p.add_argument("--step_time", type=float, default=0.064)
    p.add_argument("--temperature", type=float, default=20.0)
    p.add_argument("--noise_sigma", type=float, default=0.5)
    # --- episode / diagnostics ----------------------------------------------
    p.add_argument("--settle", type=float, default=1.0,
                    help="Seconds to hold the initial grasp command before planning.")
    p.add_argument("--n_episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_substeps", type=int, default=None,
                    help="Fine Pinocchio steps per rollout step (default: task config).")
    p.add_argument("--video", type=str,
                    default="videos/grasp_reorient_pinocchio_mppi_test.mp4")
    p.add_argument("--print_contacts", action="store_true",
                    help="Also print every substep's contact penetration + ADMM "
                         "convergence live (pinocchio_sim.PRINT_CONTACT_DEBUG). "
                         "Very verbose (one line per fine substep); the per-episode "
                         "summary from diagnostics() prints regardless of this flag.")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    wp.init()

    if args.difficulty is not None:
        GraspReorientTask.goal_difficulty = args.difficulty
    if args.print_contacts:
        pinocchio_sim.PRINT_CONTACT_DEBUG = True

    contact_cfg = MODEL_FACTORIES[args.model]()
    planner_cfg = make_planner_config(
        "mppi",
        n_samples=args.n_samples,
        time_horizon=args.time_horizon,
        step_time=args.step_time,
        temperature=args.temperature,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )

    seed_seq = np.random.SeedSequence(args.seed)
    episode_seeds = seed_seq.spawn(args.n_episodes)

    for ep_idx in range(args.n_episodes):
        rng = np.random.default_rng(episode_seeds[ep_idx])
        video_path = args.video
        if args.n_episodes > 1:
            stem, suffix = args.video.rsplit(".", 1)
            video_path = f"{stem}_ep{ep_idx:03d}.{suffix}"

        result = run_eval_episode(
            task_name      = "grasp_reorient",
            contact_cfg    = contact_cfg,
            planner        = "mppi",
            planner_cfg    = planner_cfg,
            rng            = rng,
            eval_sim       = EvalSimulatorKind.PINOCCHIO,
            video_path     = video_path,
            settle_seconds = args.settle,
            eval_substeps  = args.eval_substeps,
            ep_idx         = ep_idx,
            debug          = args.debug,
            verbose        = True,
        )

        label = "SUCCESS" if result.success else "no success"
        sstr = f"step {result.steps_to_success}" if result.steps_to_success is not None else "-"
        print(f"  [ep {ep_idx:02d}] {label}  success_step={sstr}  "
              f"final_cost={result.final_cost:.4f}  "
              f"step={result.mean_step_ms:.3f}+/-{result.std_step_ms:.3f} ms")

        sim = GraspReorientTask._active_eval_sim
        if sim is not None and hasattr(sim, "diagnostics"):
            _print_diag(ep_idx, sim.diagnostics())
        else:
            print(f"  [ep {ep_idx:02d}] no PinocchioSimulator diagnostics available "
                  f"(eval sim was {type(sim).__name__ if sim is not None else None})")


if __name__ == "__main__":
    main()
