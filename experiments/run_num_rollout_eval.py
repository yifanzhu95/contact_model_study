"""run_num_rollout_eval.py

Sweep over MPPI sample counts to evaluate how rollout budget affects task
success rate and per-step planning time.

For each (model × n_samples) cell the script runs `--n_episodes` episodes via
`run_eval_episode` (the eval/rollout driver) and aggregates the results into a
JSON file in the same format as run_experiment.py.

Edit N_SAMPLES_SWEEP at the top of this file to change the sweep values.

Usage:
    # Default sweep, grasp_reorient, all models, 10 episodes each
    python experiments/run_num_rollout_eval.py --n_episodes 10

    # Single model, custom task, MuJoCo eval (faster than Drake)
    python experiments/run_num_rollout_eval.py --models M2 M3 --task push --eval_sim mujoco

    # Specify output path
    python experiments/run_num_rollout_eval.py --output results/n_samples_sweep.json
"""

from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import datetime
from pathlib import Path

import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401

from contact_study.evaluation.metrics import aggregate_episodes, save_results
from contact_study.planners.mppi import MPPIConfig
from contact_study.tasks.config import EvalSimulatorKind

from contact_study.drivers.run_eval_episode import (
    run_eval_episode, load_rollout_task, MODEL_FACTORIES,
)
from contact_study.tasks.config import DEFAULT_SCENE_VARIANT

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ---------------------------------------------------------------------------
# Edit this list to change the sweep values
# ---------------------------------------------------------------------------
N_SAMPLES_SWEEP = [256, 512, 1024, 2048, 4096]

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
    parser.add_argument("--n_episodes",     type=int,   default=5,
                        help="Episodes to run per (model, n_samples) cell.")
    parser.add_argument("--horizon",        type=int,   default=48)
    parser.add_argument("--temperature",    type=float, default=0.125)
    parser.add_argument("--noise_sigma",    type=float, default=0.01)
    parser.add_argument("--seed",           type=int,   default=None)
    parser.add_argument("--geometry",       type=str,   default=DEFAULT_SCENE_VARIANT,
                        help="Scene variant: '<object>' or "
                             "'<object>_<hand_acc>_<obj_acc>' (e.g. duck_low_high). "
                             "Legacy geometry names map to the default scene.")
    parser.add_argument("--eval_sim",       type=str,   default="none",
                        choices=["none", "mujoco", "drake"],
                        help="Eval simulator: 'none' uses the task default, else override it.")
    parser.add_argument("--settle",         type=float, default=1.0)
    parser.add_argument("--use_full_graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--nconmax",        type=int,   default=50)
    parser.add_argument("--njmax",          type=int,   default=200)
    parser.add_argument("--output",         type=str,   default=None,
                        help="Path for results JSON (auto-timestamped if omitted).")
    parser.add_argument("--debug",          action="store_true")
    args = parser.parse_args()

    rng      = np.random.default_rng(args.seed)
    geometry = args.geometry
    eval_sim = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)

    print(f"\n{'='*65}")
    print(f"  n_samples sweep — {args.task}")
    print(f"  models     : {args.models}")
    print(f"  n_samples  : {N_SAMPLES_SWEEP}")
    print(f"  n_episodes : {args.n_episodes}  horizon={args.horizon}")
    print(f"{'='*65}")

    aggregated = []

    for model_key in args.models:
        cfg = MODEL_FACTORIES[model_key]()

        # Load once just to print model dimensions before the sweep.
        _task = load_rollout_task(args.task, geometry)
        _mjm = _task.mjm
        print(f"\n  [{model_key}]  nq={_mjm.nq}  nv={_mjm.nv}  nu={_mjm.nu}  "
              f"max_steps={_task.config.max_steps}")

        for n_samples in N_SAMPLES_SWEEP:
            mppi_cfg = MPPIConfig(
                n_samples      = n_samples,
                step_horizon   = args.horizon,
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
                result = run_eval_episode(
                    task_name      = args.task,
                    contact_cfg    = cfg,
                    mppi_cfg       = mppi_cfg,
                    rng            = rng,
                    geometry       = geometry,
                    settle_seconds = args.settle,
                    eval_sim       = eval_sim,
                    ep_idx         = ep,
                    debug          = args.debug,
                    verbose        = False,
                )
                episodes.append(result)
                tick = "✓" if result.success else "✗"
                sstr = f"step {result.steps_to_success}" if result.steps_to_success else "—"
                print(f"    ep {ep:02d}  {tick}  success_step={sstr:<8}  "
                      f"step={result.mean_step_ms:.3f}±{result.std_step_ms:.3f} ms")

            agg = aggregate_episodes(episodes, args.task, label, "B")
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
    print(f"  Summary — {args.task}")
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
