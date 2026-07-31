"""run_horizon_cell.py

HPC worker for the MPPI horizon-length sweep — one (model, time_horizon) cell.
Mirrors run_num_rollout_cell.py but sweeps MPPIConfig.time_horizon instead of
n_samples.

The horizon is swept in SECONDS. The worker quantizes it against the task's
rollout timestep (rollout_dt = eval_dt * eval_substeps_per_rollout) into whole
control steps and labels the cell with that resolved step count, so the sweep
plotter (analysis/plot_horizon_sweep_dir.py, which parses "<model>_h<int>")
keeps working unchanged.

The SLURM script (horizon_eval.slurm) owns the sweep: it defines the list of
horizon durations (TIME_HORIZONS) and the MODELS inline, maps
$SLURM_ARRAY_TASK_ID to one horizon value, and calls this worker once per model.
This worker runs `--n_episodes` episodes for that one cell via
`run_eval_episode` (no video), aggregates them into an AggregatedResult labelled
"<model>_h<horizon_steps>", and writes ONE JSON to `--outdir`.

    python run_horizon_cell.py \
        --outdir results/horizon_eval_run \
        --task grasp_reorient --model M2 --time_horizon 0.256 \
        --n_episodes 5
"""

from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.contact_models.config import GeometryVariant
from contact_study.evaluation.metrics import aggregate_episodes, save_results
from contact_study.planners.mppi import MPPIConfig
from contact_study.tasks.config import EvalSimulatorKind

from contact_study.drivers.run_eval_episode import (
    run_eval_episode, load_rollout_task, resolve_mppi_schedule, MODEL_FACTORIES,
)


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------
def run_cell(args):
    """Run one (model, time_horizon) cell (n_episodes episodes) -> AggregatedResult."""
    geometry = GeometryVariant(args.geometry)
    eval_sim = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)
    cfg      = MODEL_FACTORIES[args.model]()

    # Peek at the rollout task once for dimensions (before the episode loop).
    peek = load_rollout_task(args.task, geometry)
    mjm  = peek.mjm

    # Quantize the requested durations into the step counts the controller will
    # resolve internally, so the label / seed key / log line report what actually
    # runs rather than the requested seconds.
    horizon, substeps, rollout_dt = resolve_mppi_schedule(
        MPPIConfig(time_horizon=args.time_horizon, step_time=args.step_time),
        peek.config, args.eval_substeps,
    )
    label      = f"{args.model}_h{horizon}"
    control_dt = substeps * rollout_dt

    print(f"[{label}]  nq={mjm.nq} nv={mjm.nv} nu={mjm.nu}  "
          f"max_steps={peek.config.max_steps}  n_episodes={args.n_episodes}")
    print(f"[{label}]  rollout_dt={rollout_dt*1e3:.3f}ms  "
          f"step_time={args.step_time:g}s -> {substeps} substeps ({control_dt*1e3:.1f}ms)  "
          f"time_horizon={args.time_horizon:g}s -> {horizon} steps "
          f"({horizon*control_dt*1e3:.1f}ms)")

    # Reproducible per-episode seeds keyed by (seed, horizon, model) so cells
    # never share a stream but a run repeats.
    model_ord = list(MODEL_FACTORIES).index(args.model)
    seed_seq  = np.random.SeedSequence(
        [s for s in (args.seed, horizon, model_ord) if s is not None] or None
    )
    episode_seeds = seed_seq.spawn(args.n_episodes)

    episodes = []
    for ep in range(args.n_episodes):
        ep_seed = int(episode_seeds[ep].generate_state(1)[0])
        rng     = np.random.default_rng(episode_seeds[ep])

        mppi_cfg = MPPIConfig(
            n_samples      = args.n_samples,
            time_horizon   = args.time_horizon,
            temperature    = args.temperature,
            noise_sigma    = args.noise_sigma,
            step_time      = args.step_time,
            warm_start     = False,
            resample_interval = 1,
            use_full_graph = args.use_full_graph,
            delta_range    = (-args.delta, args.delta),
            nconmax        = args.nconmax,
            njmax          = args.njmax,
            seed           = ep_seed,
            debug          = args.debug,
        )

        # video_path is left unset -> run_eval_episode builds no renderer.
        result = run_eval_episode(
            task_name         = args.task,
            contact_cfg       = cfg,
            mppi_cfg          = mppi_cfg,
            rng               = rng,
            geometry          = geometry,
            settle_seconds    = args.settle,
            eval_substeps     = args.eval_substeps,
            eval_sim          = eval_sim,
            ep_idx            = ep,
            fin_ep_on_success = True,
            debug             = args.debug,
            verbose           = False,
        )
        episodes.append(result)
        tick = "✓" if result.success else "✗"
        sstr = f"step {result.steps_to_success}" if result.steps_to_success is not None else "—"
        print(f"    ep {ep:02d}  {tick}  success_step={sstr:<8}  "
              f"step={result.mean_step_ms:.3f}±{result.std_step_ms:.3f} ms")

    agg = aggregate_episodes(episodes, args.task, label, "B")
    succ = [e.steps_to_success for e in episodes if e.steps_to_success is not None]
    print(f"  → success={agg.success_rate*100:.1f}%  "
          f"step_ms={agg.mean_step_ms:.3f}±{agg.std_step_ms:.3f}  "
          + (f"mean_steps={agg.mean_steps_to_success:.1f}" if succ else ""))
    return agg, label


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HPC worker: run one horizon-length sweep cell.")
    p.add_argument("--task",       type=str, default="grasp_reorient")
    p.add_argument("--model",      type=str, default="M2", choices=list(MODEL_FACTORIES),
                   help="Contact model for this cell.")
    p.add_argument("--time_horizon", type=float, required=True,
                   help="MPPI planning horizon in SECONDS for this cell (the swept "
                        "axis); quantized down to whole control steps.")
    p.add_argument("--n_episodes", type=int, default=10,
                   help="Episodes for this cell (used to estimate the success rate).")
    p.add_argument("--outdir",     type=str, default="results/horizon_eval_run",
                   help="Directory for the per-cell JSON output.")
    # --- MPPI / eval knobs (all command-line inputs; n_samples/step_time fixed) -
    p.add_argument("--n_samples",     type=int,   default=256,
                   help="Fixed MPPI sample count (held constant across the sweep).")
    p.add_argument("--temperature",   type=float, default=10.0)
    p.add_argument("--noise_sigma",   type=float, default=0.01)
    p.add_argument("--delta",         type=float, default=0.1,
                   help="Per-step MPPI delta clip magnitude (action units).")
    p.add_argument("--step_time",     type=float, default=0.032,
                   help="Control-step duration in SECONDS (held constant across the "
                        "sweep); quantized down to whole rollout steps.")
    p.add_argument("--eval_substeps", type=int,   default=None,
                   help="Eval steps per rollout step (default: task config).")
    p.add_argument("--eval_sim",      type=str,   default="none",
                   choices=["none", "mujoco", "drake", "pinocchio"])
    p.add_argument("--settle",        type=float, default=1.0)
    p.add_argument("--geometry",      type=str,   default="accurate",
                   choices=[g.value for g in GeometryVariant])
    p.add_argument("--nconmax",       type=int,   default=50)
    p.add_argument("--njmax",         type=int,   default=300)
    p.add_argument("--use_full_graph",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed",          type=int,   default=None)
    p.add_argument("--debug",         action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    wp.init()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # The label carries the RESOLVED horizon in steps, so run_cell owns it.
    agg, label = run_cell(args)
    out_path = outdir / f"{label}.json"
    # save_results writes a JSON *list* of AggregatedResult dicts (one here), which
    # is exactly what the directory plotter merges across every cell file.
    save_results([agg], out_path)


if __name__ == "__main__":
    main()
