"""run_param_cell.py

HPC worker for the cost-weight parameter search — one SLURM array task = one cell.

A "cell" is a single (model, weight set) point of the grid. The SLURM script owns
the grid: it defines the value lists, maps `$SLURM_ARRAY_TASK_ID` to one value per
axis, and calls this script with the *selected* model and weight values. This
worker just runs `--n_episodes` episodes for that one cell via `run_eval_episode`,
aggregates the success rate, and writes ONE JSON file to `--outdir`
(`cell_<cell_id>.json`). `combine_results.py` merges those per-cell files.

Every parameter is a command-line input:

    python run_param_cell.py \
        --cell_id 4 --outdir results/param_search_run \
        --task grasp_reorient --model M2 --n_episodes 5 \
        --weights w_quat=20 w_pos_x=15 w_pos_y=15 w_pos_z=15 w_contact=10 w_joint=0.05
"""

from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.planners.mppi import MPPIConfig
from contact_study.tasks.config import EvalSimulatorKind

from contact_study.drivers.run_eval_episode import (
    run_eval_episode, load_rollout_task, resolve_mppi_schedule, MODEL_FACTORIES,
)
from contact_study.tasks.config import DEFAULT_SCENE_VARIANT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_overrides(tokens: list[str]) -> dict[str, float]:
    """Parse `name=value` tokens into an ordered {name: value} override dict.

    Order is preserved from the command line, which must match the insertion
    order of the task's `config.cost_weights` for the override merge — this
    holds for all built-in tasks (see apply_cost_weight_overrides)."""
    overrides: dict[str, float] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"bad --weights token {tok!r}; expected name=value")
        name, val = tok.split("=", 1)
        overrides[name.strip()] = float(val)
    return overrides


def combo_label(model_key: str, overrides: dict[str, float]) -> str:
    """Short label encoding model + overridden weight values."""
    parts = [f"{k.lstrip('w_')}={v:g}" for k, v in overrides.items()]
    return f"{model_key}__" + "_".join(parts)


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------

def run_cell(cell_id: int, overrides: dict, args) -> dict:
    """Run one grid cell (n_episodes episodes) and return the result dict."""
    geometry = args.geometry
    eval_sim = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)
    cfg      = MODEL_FACTORIES[args.model]()

    label = combo_label(args.model, overrides)

    # Peek at the rollout task once for default weights + dimensions.
    peek_task       = load_rollout_task(args.task, geometry)
    default_weights = dict(peek_task.config.cost_weights)
    full_weights    = {**default_weights, **overrides}

    # Quantize the requested durations into the step counts the controller will
    # resolve internally, for the log line and the result record.
    horizon, substeps, rollout_dt = resolve_mppi_schedule(
        MPPIConfig(time_horizon=args.time_horizon, step_time=args.step_time),
        peek_task.config, args.eval_substeps,
    )

    print(f"[cell {cell_id}]  {label}")
    print(f"  task={args.task}  model={args.model}  n_episodes={args.n_episodes}")
    print(f"  rollout_dt={rollout_dt*1e3:.3f}ms  step_time={args.step_time:g}s -> "
          f"{substeps} substeps  time_horizon={args.time_horizon:g}s -> {horizon} steps")
    print(f"  overrides={overrides}")

    # Reproducible per-episode seeds derived from the base seed AND the cell id,
    # so different cells never share a seed but a run is repeatable.
    seed_seq      = np.random.SeedSequence([s for s in (args.seed, cell_id)
                                            if s is not None] or None)
    episode_seeds = seed_seq.spawn(args.n_episodes)

    episodes = []
    for ep in range(args.n_episodes):
        ep_seed = int(episode_seeds[ep].generate_state(1)[0])
        rng     = np.random.default_rng(episode_seeds[ep])

        if not args.delta is None:
            delta = (-args.delta, args.delta)
        else:
            delta = (None, None)

        mppi_cfg = MPPIConfig(
            n_samples      = args.n_samples,
            time_horizon   = args.time_horizon,
            temperature    = args.temperature,
            noise_sigma    = args.noise_sigma,
            step_time      = args.step_time,
            warm_start     = False,
            use_full_graph = args.use_full_graph,
            delta_range    = delta,
            nconmax        = args.nconmax,
            njmax          = args.njmax,
            seed           = ep_seed,
            debug          = args.debug,
            resample_interval = args.resample_interval,
            time_constrained  = args.time_constrained,
            plan_budget_ms    = args.plan_budget_ms,
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
        "combo_index":           cell_id,
        "task":                  args.task,
        "model":                 args.model,
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
            "n_samples":         args.n_samples,
            "time_horizon":      args.time_horizon,
            "temperature":       args.temperature,
            "noise_sigma":       args.noise_sigma,
            "step_time":         args.step_time,
            # Resolved against rollout_dt — what the controller actually ran.
            "step_horizon":      horizon,
            "step_substeps":     substeps,
            "rollout_dt":        rollout_dt,
            "delta":             args.delta,
            "resample_interval": args.resample_interval,
            "time_constrained":  args.time_constrained,
            "plan_budget_ms":    args.plan_budget_ms,
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
    p = argparse.ArgumentParser(description="HPC worker: run one parameter-search cell.")
    # --- which cell (selected by the SLURM script) -------------------------
    p.add_argument("--task",    type=str, default="grasp_reorient")
    p.add_argument("--model",   type=str, default="M2", choices=list(MODEL_FACTORIES),
                   help="Contact model for this cell.")
    p.add_argument("--weights", nargs="+", default=[],
                   help="Weight overrides for this cell as name=value tokens.")
    p.add_argument("--cell_id", type=int, default=0,
                   help="Cell index (e.g. $SLURM_ARRAY_TASK_ID); names the output file.")
    p.add_argument("--n_episodes", type=int, default=5,
                   help="Episodes for this cell (used to estimate the success rate).")
    p.add_argument("--outdir", type=str, default="results/param_search_run",
                   help="Directory for the per-cell JSON output.")
    # --- MPPI / eval knobs (all command-line inputs) -----------------------
    p.add_argument("--n_samples",     type=int,   default=256)
    p.add_argument("--time_horizon",  type=float, default=0.256,
                   help="MPPI planning horizon in SECONDS (quantized down to whole "
                        "control steps).")
    p.add_argument("--temperature",   type=float, default=1.0)
    p.add_argument("--noise_sigma",   type=float, default=0.01)
    p.add_argument("--delta",         type=float, default=0.1,
                   help="Per-step MPPI delta clip magnitude (action units).")
    p.add_argument("--step_time",     type=float, default=0.032,
                   help="Control-step duration in SECONDS, i.e. the control-frequency "
                        "knob (quantized down to whole rollout steps).")
    p.add_argument("--eval_substeps", type=int,   default=None,
                   help="Eval steps per rollout step (default: task config).")
    p.add_argument("--eval_sim",      type=str,   default="none",
                   choices=["none", "mujoco", "drake", "pinocchio"])
    p.add_argument("--settle",        type=float, default=1.0)
    p.add_argument("--geometry",      type=str,   default=DEFAULT_SCENE_VARIANT,
                   help="Scene variant: '<object>' or "
                        "'<object>_<hand_acc>_<obj_acc>' (e.g. duck_low_high). "
                        "Legacy geometry names map to the default scene.")
    p.add_argument("--nconmax",       type=int,   default=50)
    p.add_argument("--njmax",         type=int,   default=200)
    p.add_argument("--use_full_graph",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resample_interval", type=int, default=1,
                   help="Plan steps between MPPI noise resamples (1=every step; "
                        "omit=sample once and reuse, the default).")
    p.add_argument("--time_constrained", action=argparse.BooleanOptionalAction, default=False,
                   help="Stop rollouts once --plan_budget_ms elapses (capped at the horizon).")
    p.add_argument("--plan_budget_ms", type=float, default=None,
                   help="Wall-clock rollout budget per plan() in ms; required with "
                        "--time_constrained.")
    p.add_argument("--seed",          type=int,   default=None)
    p.add_argument("--debug",         action="store_true")
    return p


def main():
    args      = build_parser().parse_args()
    overrides = parse_overrides(args.weights)

    wp.init()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    record   = run_cell(args.cell_id, overrides, args)
    out_path = outdir / f"cell_{args.cell_id:05d}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
