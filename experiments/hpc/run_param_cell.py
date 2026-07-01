"""run_param_cell.py

HPC worker for the cost-weight parameter search — one SLURM array task = one cell.

A "cell" is a single (model x weight-combo) point of the grid. The grid is the
Cartesian product of the swept `--models` and the per-weight value lists passed
via `--weights`. Every array task independently reconstructs the *same* grid from
identical CLI args and selects its own cell with `--combo_index` (typically
`$SLURM_ARRAY_TASK_ID`). It runs `--n_episodes` episodes for that cell via
`run_eval_episode`, aggregates the success rate, and writes ONE JSON file to
`--outdir` (`cell_<index>.json`). `combine_results.py` merges those per-cell files.

This is the HPC counterpart of experiments/run_param_search.py: same search, but
the outer grid loop is parallelized across the job array instead of run serially.

Every parameter is a command-line input, and the swept axes accept lists:

    # Count the cells (use this to size the SLURM --array range)
    python run_param_cell.py --num_combos \
        --models M2 M3 \
        --weights w_quat=15,20,25 w_pos=15,20,25 w_contact=7.5,10,12.5

    # Run one cell (what SLURM calls per array task)
    python run_param_cell.py --combo_index 4 --outdir results/param_search_run \
        --task grasp_reorient --n_episodes 5 \
        --models M2 M3 \
        --weights w_quat=15,20,25 w_pos=15,20,25 w_contact=7.5,10,12.5

    # Run ALL cells serially in one process (local, no job array)
    python run_param_cell.py --outdir results/param_search_run \
        --n_episodes 5 --weights w_quat=15,20,25 w_pos=15,20,25
"""

from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.contact_models.config import GeometryVariant
from contact_study.planners.mppi import MPPIConfig
from contact_study.tasks.config import EvalSimulatorKind

from contact_study.drivers.run_eval_episode import (
    run_eval_episode, load_rollout_task, MODEL_FACTORIES,
)


# ---------------------------------------------------------------------------
# Grid helpers (shared shape with experiments/run_param_search.py)
# ---------------------------------------------------------------------------

def parse_weights(tokens: list[str]) -> dict[str, list[float]]:
    """Parse `name=v1,v2,...` tokens into an ordered {name: [values]} dict.

    Order is preserved from the command line, which must match the insertion
    order of the task's `config.cost_weights` for the override merge — this
    holds for all built-in tasks (see apply_cost_weight_overrides)."""
    space: dict[str, list[float]] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"bad --weights token {tok!r}; expected name=v1,v2,...")
        name, vals = tok.split("=", 1)
        space[name.strip()] = [float(v) for v in vals.split(",") if v != ""]
    return space


def combo_label(model_key: str, overrides: dict[str, float]) -> str:
    """Short label encoding model + overridden weight values."""
    parts = [f"{k.lstrip('w_')}={v:g}" for k, v in overrides.items()]
    return f"{model_key}__" + "_".join(parts)


def build_cells(models: list[str],
                search_space: dict[str, list[float]]) -> list[tuple[str, dict]]:
    """Enumerate every (model, weight-override) cell of the grid, deterministically.

    Outer axis is the model, inner axes are the weight lists in CLI order. Every
    array task calls this with identical args, so cell `i` is stable everywhere."""
    keys   = list(search_space.keys())
    combos = list(itertools.product(*[search_space[k] for k in keys])) or [()]
    cells: list[tuple[str, dict]] = []
    for model_key in models:
        for vals in combos:
            cells.append((model_key, dict(zip(keys, vals))))
    return cells


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------

def run_cell(cell_index: int,
             model_key:  str,
             overrides:  dict,
             args) -> dict:
    """Run one grid cell (n_episodes episodes) and return the result dict."""
    geometry = GeometryVariant(args.geometry)
    eval_sim = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)
    cfg      = MODEL_FACTORIES[model_key]()

    label = combo_label(model_key, overrides)

    # Peek at the rollout task once for default weights + dimensions.
    peek_task       = load_rollout_task(args.task, geometry)
    default_weights = dict(peek_task.config.cost_weights)
    full_weights    = {**default_weights, **overrides}

    print(f"[cell {cell_index}]  {label}")
    print(f"  task={args.task}  model={model_key}  n_episodes={args.n_episodes}")
    print(f"  overrides={overrides}")

    # Reproducible per-episode seeds derived from the base seed AND the cell
    # index, so different cells never share a seed but a run is repeatable.
    seed_seq      = np.random.SeedSequence([s for s in (args.seed, cell_index)
                                            if s is not None] or None)
    episode_seeds = seed_seq.spawn(args.n_episodes)

    episodes = []
    for ep in range(args.n_episodes):
        ep_seed = int(episode_seeds[ep].generate_state(1)[0])
        rng     = np.random.default_rng(episode_seeds[ep])

        mppi_cfg = MPPIConfig(
            n_samples      = args.n_samples,
            horizon        = args.horizon,
            temperature    = args.temperature,
            noise_sigma    = args.noise_sigma,
            substeps       = args.substeps,
            warm_start     = True,
            use_full_graph = args.use_full_graph,
            delta_range    = (-args.delta, args.delta),
            nconmax        = args.nconmax,
            njmax          = args.njmax,
            seed           = ep_seed,
            debug          = args.debug,
        )

        result = run_eval_episode(
            task_name             = args.task,
            contact_cfg           = cfg,
            mppi_cfg              = mppi_cfg,
            rng                   = rng,
            geometry              = geometry,
            cost_weight_overrides = overrides,
            settle_seconds        = args.settle,
            eval_substeps         = args.eval_substeps,
            eval_sim              = eval_sim,
            ep_idx                = ep,
            fin_ep_on_success     = True,
            debug                 = args.debug,
            verbose               = args.debug,
        )
        episodes.append(result)
        tick = "✓" if result.success else "✗"
        sstr = f"step {result.steps_to_success}" if result.steps_to_success is not None else "—"
        print(f"    ep {ep:02d}  {tick}  success_step={sstr:<8}  "
              f"step={result.mean_step_ms:.3f}±{result.std_step_ms:.3f} ms")

    n_success    = sum(r.success for r in episodes)
    success_rate = n_success / len(episodes)
    succ_steps   = [r.steps_to_success for r in episodes if r.steps_to_success is not None]
    mean_sts     = float(np.mean(succ_steps)) if succ_steps else None
    step_ms      = [r.mean_step_ms for r in episodes]
    step_sd      = [r.std_step_ms  for r in episodes]
    elapsed      = [r.elapsed_seconds for r in episodes]

    print(f"  → success={success_rate*100:.1f}%  ({n_success}/{len(episodes)})  "
          f"step_ms={float(np.mean(step_ms)):.3f}±{float(np.mean(step_sd)):.3f}")

    return {
        "combo_index":           cell_index,
        "task":                  args.task,
        "model":                 model_key,
        "label":                 label,
        "overrides":             overrides,
        "full_weights":          full_weights,
        "n_episodes":            len(episodes),
        "n_success":             n_success,
        "success_rate":          success_rate,
        "mean_steps_to_success": mean_sts,
        "mean_step_ms":          float(np.mean(step_ms)),
        "std_step_ms":           float(np.mean(step_sd)),
        "mean_elapsed_s":        float(np.mean(elapsed)),
        "mppi": {
            "n_samples":   args.n_samples,
            "horizon":     args.horizon,
            "temperature": args.temperature,
            "noise_sigma": args.noise_sigma,
            "substeps":    args.substeps,
            "delta":       args.delta,
        },
        "seed":     args.seed,
        "eval_sim": args.eval_sim,
        "geometry": args.geometry,
        "settle":   args.settle,
        "episodes": [dataclasses.asdict(r) for r in episodes],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HPC worker: run one (or all) parameter-search cells.")
    # --- what to sweep -----------------------------------------------------
    p.add_argument("--task",    type=str, default="grasp_reorient")
    p.add_argument("--models",  nargs="+", default=list(MODEL_FACTORIES.keys()),
                   choices=list(MODEL_FACTORIES.keys()),
                   help="Contact models to sweep (outer grid axis).")
    p.add_argument("--weights", nargs="+", default=[
                       "w_quat=15,20,25",
                       "w_pos=15,20,25",
                       "w_contact=7.5,10,12.5",
                       "w_joint=0.025,0.05,0.1",
                   ],
                   help="Swept weight axes as name=v1,v2,... tokens (inner grid axes).")
    p.add_argument("--n_episodes", type=int, default=5,
                   help="Episodes per cell (used to estimate the success rate).")
    # --- which cell(s) -----------------------------------------------------
    p.add_argument("--combo_index", type=int, default=None,
                   help="Run only this cell (e.g. $SLURM_ARRAY_TASK_ID). "
                        "Omit to run all cells serially in one process.")
    p.add_argument("--num_combos", action="store_true",
                   help="Print the number of cells in the grid and exit "
                        "(use to size the SLURM --array range).")
    p.add_argument("--outdir", type=str, default="results/param_search_run",
                   help="Directory for per-cell JSON output.")
    # --- MPPI / eval knobs (all command-line inputs) -----------------------
    p.add_argument("--n_samples",     type=int,   default=256)
    p.add_argument("--horizon",       type=int,   default=48)
    p.add_argument("--temperature",   type=float, default=1.0)
    p.add_argument("--noise_sigma",   type=float, default=0.01)
    p.add_argument("--delta",         type=float, default=0.1,
                   help="Per-step MPPI delta clip magnitude (action units).")
    p.add_argument("--substeps",      type=int,   default=16,
                   help="MPPI rollout substeps per control step (control-freq knob).")
    p.add_argument("--eval_substeps", type=int,   default=None,
                   help="Eval steps per rollout step (default: task config).")
    p.add_argument("--eval_sim",      type=str,   default="none",
                   choices=["none", "mujoco", "drake", "pinocchio"])
    p.add_argument("--settle",        type=float, default=1.0)
    p.add_argument("--geometry",      type=str,   default="accurate",
                   choices=[g.value for g in GeometryVariant])
    p.add_argument("--nconmax",       type=int,   default=50)
    p.add_argument("--njmax",         type=int,   default=200)
    p.add_argument("--use_full_graph",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed",          type=int,   default=None)
    p.add_argument("--debug",         action="store_true")
    return p


def main():
    args  = build_parser().parse_args()
    space = parse_weights(args.weights)
    cells = build_cells(args.models, space)

    if args.num_combos:
        print(len(cells))
        return

    wp.init()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.combo_index is not None:
        if not (0 <= args.combo_index < len(cells)):
            raise SystemExit(
                f"--combo_index {args.combo_index} out of range [0, {len(cells)})")
        indices = [args.combo_index]
    else:
        indices = list(range(len(cells)))
        print(f"No --combo_index: running all {len(cells)} cells serially.")

    for idx in indices:
        model_key, overrides = cells[idx]
        record   = run_cell(idx, model_key, overrides, args)
        out_path = outdir / f"cell_{idx:05d}.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  saved -> {out_path}\n")


if __name__ == "__main__":
    main()
