"""run_num_rollout_eval.py

Sweep over MPPI sample counts to evaluate how rollout budget affects task
success rate and per-step planning time.

For each (model × n_samples) cell the script runs `--n_episodes` episodes via
`run_episode`, aggregates the results, and saves a JSON file in the same
format as run_experiment.py.

Edit N_SAMPLES_SWEEP at the top of this file to change the sweep values.

Usage:
    # Default sweep, grasp_reorient, all models, 10 episodes each
    python experiments/run_num_rollout_eval.py --n_episodes 10

    # Single model, custom sweep task
    python experiments/run_num_rollout_eval.py --models M2 M3 --task push --n_episodes 5

    # Physics noise
    python experiments/run_num_rollout_eval.py --friction_sigma 0.1 --n_episodes 10

    # Specify output path
    python experiments/run_num_rollout_eval.py --output results/n_samples_sweep.json
"""

from __future__ import annotations
import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import datetime
from pathlib import Path

import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401

from contact_study.contact_models.config import ContactModelConfig, GeometryVariant
from contact_study.evaluation.metrics import aggregate_episodes, save_results
from contact_study.planners.mppi import MPPIConfig
from contact_study.tasks.base import get_task
from contact_study.utils.physics_noise import PhysicsNoiseParams, apply_physics_noise

from run_episode import run_episode, load_task, MODEL_FACTORIES

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ---------------------------------------------------------------------------
# Edit this list to change the sweep values
# ---------------------------------------------------------------------------
N_SAMPLES_SWEEP = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

wp.init()


def main():
    parser = argparse.ArgumentParser(
        description="Sweep MPPI n_samples and evaluate task performance."
    )
    parser.add_argument("--task",    type=str, default="grasp_reorient",
                        help="Registered task name.")
    parser.add_argument("--models",  nargs="+", default=list(MODEL_FACTORIES.keys()),
                        choices=list(MODEL_FACTORIES.keys()),
                        help="Contact model keys to evaluate (e.g. M1 M2 M3 M4).")
    parser.add_argument("--condition", type=str, default="B", choices=["A", "B"],
                        help="A=fixed_budget_rollout  B=warm-started MPPIController")
    parser.add_argument("--n_episodes",     type=int,   default=20,
                        help="Episodes to run per (model, n_samples) cell.")
    parser.add_argument("--budget_seconds", type=float, default=0.1,
                        help="Per-step wall-time budget for Condition A.")
    parser.add_argument("--horizon",        type=int,   default=48)
    parser.add_argument("--temperature",    type=float, default=0.5)
    parser.add_argument("--noise_sigma",    type=float, default=0.01)
    parser.add_argument("--seed",           type=int,   default=None)
    parser.add_argument("--geometry",       type=str,   default="accurate",
                        choices=[g.value for g in GeometryVariant])
    parser.add_argument("--mass_sigma",     type=float, default=0.0)
    parser.add_argument("--inertia_sigma",  type=float, default=0.0)
    parser.add_argument("--friction_sigma", type=float, default=0.0)
    parser.add_argument("--com_sigma",      type=float, default=0.0)
    parser.add_argument("--settle",         type=float, default=10.0)
    parser.add_argument("--use_full_graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--nconmax",        type=int,   default=200)
    parser.add_argument("--njmax",          type=int,   default=500)
    parser.add_argument("--output",         type=str,   default=None,
                        help="Path for results JSON (auto-timestamped if omitted).")
    parser.add_argument("--debug",          action="store_true")
    args = parser.parse_args()

    rng      = np.random.default_rng(args.seed)
    noise    = PhysicsNoiseParams(
        mass_sigma     = args.mass_sigma,
        inertia_sigma  = args.inertia_sigma,
        friction_sigma = args.friction_sigma,
        com_sigma      = args.com_sigma,
    )
    geometry = GeometryVariant(args.geometry)

    print(f"\n{'='*65}")
    print(f"  n_samples sweep — {args.task}  condition={args.condition}")
    print(f"  models     : {args.models}")
    print(f"  n_samples  : {N_SAMPLES_SWEEP}")
    print(f"  n_episodes : {args.n_episodes}  horizon={args.horizon}")
    print(f"{'='*65}")

    aggregated = []

    for model_key in args.models:
        cfg = MODEL_FACTORIES[model_key]()

        # Load once just to print model dimensions before the sweep.
        _mjm, _task = load_task(args.task, geometry, noise, rng)
        print(f"\n  [{model_key}]  nq={_mjm.nq}  nv={_mjm.nv}  nu={_mjm.nu}  "
              f"max_steps={_task.spec.max_steps}")

        for n_samples in N_SAMPLES_SWEEP:
            mppi_cfg = MPPIConfig(
                n_samples      = n_samples,
                horizon        = args.horizon,
                temperature    = args.temperature,
                noise_sigma    = args.noise_sigma,
                warm_start     = True,
                use_full_graph = args.use_full_graph,
                nconmax        = args.nconmax,
                njmax          = args.njmax,
                seed           = args.seed,
                debug          = args.debug,
            )

            label = f"{model_key}_n{n_samples}"
            print(f"\n  {label}  ({args.n_episodes} episodes)")
            print(f"  {'-'*50}")

            episodes = []
            for ep in range(args.n_episodes):
                mjm, task = load_task(args.task, geometry, noise, rng)
                result = run_episode(
                    mjm            = mjm,
                    task           = task,
                    cfg            = cfg,
                    mppi_cfg       = mppi_cfg,
                    rng            = rng,
                    condition      = args.condition,
                    budget_seconds = args.budget_seconds,
                    settle_seconds = args.settle,
                    render_mode    = "none",
                    debug          = args.debug,
                    ep_idx         = ep,
                )
                episodes.append(result)
                tick = "✓" if result.success else "✗"
                sstr = f"step {result.steps_to_success}" if result.steps_to_success else "—"
                print(f"    ep {ep:02d}  {tick}  success_step={sstr:<8}  "
                      f"step={result.mean_step_ms:.3f}±{result.std_step_ms:.3f} ms")

            agg = aggregate_episodes(episodes, args.task, label, args.condition)
            aggregated.append(agg)

            succ_steps = [e.steps_to_success for e in episodes if e.steps_to_success is not None]
            print(f"  → success={agg.success_rate*100:.1f}%  "
                  f"step_ms={agg.mean_step_ms:.3f}±{agg.std_step_ms:.3f}  "
                  + (f"mean_steps={agg.mean_steps_to_success:.1f}" if succ_steps else ""))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or str(RESULTS_DIR / f"n_samples_sweep_{args.task}_{ts}.json")
    save_results(aggregated, out)

    # ------------------------------------------------------------------
    # Summary table: rows = n_samples, cols = model
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Summary — {args.task}  condition={args.condition}")
    print(f"{'='*65}")

    col_w = 18
    header = f"  {'n_samples':<10}" + "".join(f"  {m:<{col_w}}" for m in args.models)
    print(header)
    print(f"  {'-'*10}" + (f"  {'-'*col_w}") * len(args.models))

    idx = {(r.model_label,): r for r in aggregated}

    for n_samples in N_SAMPLES_SWEEP:
        row = f"  {n_samples:<10}"
        for model_key in args.models:
            label = f"{model_key}_n{n_samples}"
            r = idx.get((label,))
            if r is None:
                cell = "—"
            else:
                cell = f"{r.success_rate*100:.0f}% / {r.mean_step_ms:.2f}ms"
            row += f"  {cell:<{col_w}}"
        print(row)

    print(f"\n  Columns show: success_rate% / mean_step_ms")


if __name__ == "__main__":
    main()
