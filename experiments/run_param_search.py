"""run_param_search.py

Grid search over cost-function weights for a given task.

For each (model × weight combination) cell the script runs `--n_episodes`
episodes via `run_eval_episode` (the eval/rollout driver), passing the weight
overrides through `cost_weight_overrides`. Results are saved as two JSON files:
  - A standard aggregated JSON (same format as run_experiment.py)
  - A rich search JSON with explicit weight dicts for easy analysis/plotting

Edit WEIGHT_SEARCH_SPACE at the top to define the sweep axes.  Any key that
appears in the task's `config.cost_weights` dict is valid.  Keys not listed keep
their default value from the task config.

NOTE: The override merges into the task's `cost_weights` dict in insertion order
— which must match the order used in the task's `initialize_task`.  This holds
for all built-in tasks.

Usage:
    # Default search (grasp_reorient, all models, 5 episodes)
    python experiments/run_param_search.py --n_episodes 5

    # Single model, more episodes
    python experiments/run_param_search.py --models M3 --n_episodes 10

    # Different task, MuJoCo eval
    python experiments/run_param_search.py --task push --eval_sim mujoco --n_episodes 5
"""

from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import datetime
import itertools
import json
from pathlib import Path

import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401

from contact_study.contact_models.config import GeometryVariant
from contact_study.evaluation.metrics import aggregate_episodes, save_results
from contact_study.planners.mppi import MPPIConfig
from contact_study.tasks.config import EvalSimulatorKind

from contact_study.drivers.run_eval_episode import (
    run_eval_episode, load_rollout_task, MODEL_FACTORIES,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ---------------------------------------------------------------------------
# Edit this dict to define the search space.
# Keys must match entries in the task's config.cost_weights.
# Each key maps to a list of candidate values.
# ---------------------------------------------------------------------------
WEIGHT_SEARCH_SPACE: dict[str, list[float]] = {
    "w_quat":    [3.0, 5.0, 7.0],
    "w_pos":     [35.0, 40.0, 45.0],
    "w_contact": [2.5, 5.0, 7.5],
    "w_joint":   [0.05, 0.1, 0.2],
}

wp.init()


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def combo_label(model_key: str, overrides: dict[str, float]) -> str:
    """Short label encoding model + overridden weight values."""
    parts = [f"{k.lstrip('w_')}={v:g}" for k, v in overrides.items()]
    return f"{model_key}__" + "_".join(parts)


def build_grid(search_space: dict[str, list[float]]) -> list[dict[str, float]]:
    """Return all combinations of the search space as a list of dicts."""
    keys   = list(search_space.keys())
    combos = list(itertools.product(*[search_space[k] for k in keys]))
    return [dict(zip(keys, vals)) for vals in combos]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Grid search over task cost-function weights."
    )
    parser.add_argument("--task",    type=str, default="grasp_reorient")
    parser.add_argument("--models",  nargs="+", default=list(MODEL_FACTORIES.keys()),
                        choices=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--n_episodes",     type=int,   default=5,
                        help="Episodes per (model × weight combo) cell.")
    parser.add_argument("--n_samples",      type=int,   default=512)
    parser.add_argument("--horizon",        type=int,   default=48)
    parser.add_argument("--temperature",    type=float, default=2.0)
    parser.add_argument("--noise_sigma",    type=float, default=0.05)
    parser.add_argument("--seed",           type=int,   default=None)
    parser.add_argument("--geometry",       type=str,   default="accurate",
                        choices=[g.value for g in GeometryVariant])
    parser.add_argument("--eval_sim",       type=str,   default="none",
                        choices=["none", "mujoco", "drake"],
                        help="Eval simulator: 'none' uses the task default, else override it.")
    parser.add_argument("--settle",         type=float, default=1.0)
    parser.add_argument("--use_full_graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--nconmax",        type=int,   default=50)
    parser.add_argument("--njmax",          type=int,   default=200)
    parser.add_argument("--top_n",          type=int,   default=10,
                        help="How many top results to show in the final summary.")
    parser.add_argument("--output",         type=str,   default=None,
                        help="Output path prefix (suffixes _agg.json and _rich.json added).")
    parser.add_argument("--debug",          action="store_true")
    args = parser.parse_args()

    rng      = np.random.default_rng(args.seed)
    geometry = GeometryVariant(args.geometry)
    eval_sim = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)

    grid = build_grid(WEIGHT_SEARCH_SPACE)
    n_cells = len(args.models) * len(grid)

    print(f"\n{'='*65}")
    print(f"  weight grid search — {args.task}")
    print(f"  models      : {args.models}")
    print(f"  search axes : {list(WEIGHT_SEARCH_SPACE.keys())}")
    print(f"  combos      : {len(grid)}  ×  {len(args.models)} models  =  {n_cells} cells")
    print(f"  n_episodes  : {args.n_episodes}  (total episodes: {n_cells * args.n_episodes})")
    print(f"{'='*65}")

    mppi_cfg = MPPIConfig(
        n_samples      = args.n_samples,
        horizon        = args.horizon,
        temperature    = args.temperature,
        noise_sigma    = args.noise_sigma,
        warm_start     = True,
        use_full_graph = args.use_full_graph,
        nconmax        = args.nconmax,
        njmax          = args.njmax,
        seed           = args.seed,
        debug          = args.debug,
        delta_range    = (-0.5, 0.5),
    )

    aggregated  = []   # AggregatedResult list for save_results
    rich_rows   = []   # list[dict] for the rich JSON

    cell_idx = 0
    for model_key in args.models:
        cfg = MODEL_FACTORIES[model_key]()

        # Peek at model dimensions + default weights once.
        _task = load_rollout_task(args.task, geometry)
        _mjm = _task.mjm
        default_weights = dict(_task.config.cost_weights)
        print(f"\n  [{model_key}]  nq={_mjm.nq}  nv={_mjm.nv}  nu={_mjm.nu}  "
              f"max_steps={_task.config.max_steps}")
        print(f"  default weights: {default_weights}")

        for overrides in grid:
            cell_idx += 1
            label = combo_label(model_key, overrides)
            full_weights = {**default_weights, **overrides}

            print(f"\n  [{cell_idx}/{n_cells}]  {label}")
            print(f"  weights: { {k: full_weights[k] for k in WEIGHT_SEARCH_SPACE} }")
            print(f"  {'-'*50}")

            episodes = []
            for ep in range(args.n_episodes):
                result = run_eval_episode(
                    task_name             = args.task,
                    contact_cfg           = cfg,
                    mppi_cfg              = mppi_cfg,
                    rng                   = rng,
                    geometry              = geometry,
                    cost_weight_overrides = overrides,
                    settle_seconds        = args.settle,
                    eval_sim              = eval_sim,
                    ep_idx                = ep,
                    debug                 = args.debug,
                    verbose               = False,
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

            rich_rows.append({
                "model":                model_key,
                "label":               label,
                "overrides":           overrides,
                "full_weights":        full_weights,
                "success_rate":        agg.success_rate,
                "mean_steps_to_success": agg.mean_steps_to_success,
                "mean_step_ms":        agg.mean_step_ms,
                "std_step_ms":         agg.std_step_ms,
                "mean_elapsed_s":      agg.mean_elapsed,
                "n_episodes":          agg.n_episodes,
            })

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix  = args.output or str(RESULTS_DIR / f"param_search_{args.task}_{ts}")

    agg_path  = f"{prefix}_agg.json"
    rich_path = f"{prefix}_rich.json"

    save_results(aggregated, agg_path)

    Path(rich_path).parent.mkdir(parents=True, exist_ok=True)
    with open(rich_path, "w") as f:
        json.dump(rich_rows, f, indent=2)
    print(f"Saved {len(rich_rows)} rich results to {rich_path}")

    # ------------------------------------------------------------------
    # Ranked summary: top-N by success rate, then mean_step_ms
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Top-{args.top_n} configurations — {args.task}")
    print(f"{'='*65}")

    ranked = sorted(rich_rows, key=lambda r: (-r["success_rate"], r["mean_step_ms"]))
    col_w  = max(len(r["label"]) for r in ranked) + 2

    header_parts = list(WEIGHT_SEARCH_SPACE.keys())
    print(f"  {'label':<{col_w}}  {'succ%':>6}  {'step_ms':>9}"
          + "".join(f"  {k:>12}" for k in header_parts))
    print(f"  {'-'*col_w}  {'-'*6}  {'-'*9}"
          + "".join(f"  {'-'*12}" for _ in header_parts))

    for row in ranked[:args.top_n]:
        weight_vals = "".join(f"  {row['overrides'].get(k, '—'):>12}" for k in header_parts)
        print(f"  {row['label']:<{col_w}}  "
              f"{row['success_rate']*100:>5.1f}%  "
              f"{row['mean_step_ms']:>8.3f}ms"
              + weight_vals)


if __name__ == "__main__":
    main()
