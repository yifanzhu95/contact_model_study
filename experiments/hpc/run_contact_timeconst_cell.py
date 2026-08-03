"""run_contact_timeconst_cell.py

HPC worker for the contact-time-constant sweep — one (model, timeconst) cell.
Mirrors run_cntrl_freq_cell.py but sweeps the rollout model's contact stiffness
(solref timeconst) instead of the control period, which is held fixed.

What is being swept
-------------------
A MuJoCo contact row's solref is (timeconst, dampratio): timeconst sets how
quickly position-level penetration is driven out, so a smaller value stiffens
contact and a larger one softens it. This worker writes it via
ContactModelConfig.mujoco.solref_timeconst, which api._apply_solref_override
applies to every geom/pair solref at put_model time WITHOUT touching solimp —
so the sweep is a pure contact-stiffness axis rather than the coupled
stiffness+regularizer change the M1 hard-contact preset makes.

M2's default is MuJoCo's compiler default solref = (0.02, 1): the leap scenes
declare no solref of their own. M1's default is the preset's 2*dt, and it is
clamped there, so M1 cannot be swept stiffer without editing that clamp —
hence the sweep's default model is M2.

Only the ROLLOUT model changes. The eval simulator (grasp_reorient defaults to
Pinocchio) is untouched, so this measures how much rollout-model contact
fidelity buys in closed-loop success against a fixed ground truth.

Stability floor
---------------
timeconst < 2 * rollout_dt makes the semi-implicit integrator ring or diverge.
The override warns rather than clamping, and this worker prints the
timeconst / rollout_dt ratio up front, so a cell below the floor is visible in
the log rather than silently corrected. With grasp_reorient's rollout_dt = 4 ms
the floor is 8 ms.

The SLURM script (contact_timeconst_eval.slurm) owns the sweep: it defines the
list of time constants (TIMECONST_SWEEP) and the MODELS inline, maps
$SLURM_ARRAY_TASK_ID to one cell, and calls this worker once for it. This
worker runs `--n_episodes` episodes for that one cell via `run_eval_episode`
(no video), aggregates them into an AggregatedResult labelled
"<model>_tc<microseconds>", and writes ONE JSON to `--outdir`.

The label carries the time constant in MICROSECONDS so it stays an integer,
matching the "<model>_<key><int>" convention the other sweep plotters parse:
0.02 s -> "M2_tc20000". `<outdir>/meta.json` records eval_dt,
eval_substeps_per_rollout and the resolved schedule so
analysis/plot_contact_timeconst_sweep_dir.py can annotate the 2*rollout_dt
stability floor without guessing.

    python run_contact_timeconst_cell.py \
        --outdir results/contact_timeconst_eval_run \
        --task grasp_reorient --model M2 --solref_timeconst 0.02 \
        --n_episodes 5
"""

from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
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
    """Run one (model, timeconst) cell -> (AggregatedResult, label, schedule dict)."""
    geometry = GeometryVariant(args.geometry)
    eval_sim = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)

    # Stamp the swept contact stiffness onto the config once; put_model re-reads
    # cfg on every episode, so this covers the whole cell.
    cfg = MODEL_FACTORIES[args.model]()
    cfg.mujoco.solref_timeconst = args.solref_timeconst
    if args.solref_dampratio is not None:
        cfg.mujoco.solref_dampratio = args.solref_dampratio

    # Peek at the rollout task once for dimensions + eval-dt/eval_substeps.
    peek = load_rollout_task(args.task, geometry)
    mjm  = peek.mjm
    eval_dt       = peek.config.timestep
    eval_substeps = args.eval_substeps if args.eval_substeps is not None \
        else peek.config.eval_substeps_per_rollout

    # The control period is FIXED here, but still quantize it so the printed
    # schedule describes what actually runs (and so rollout_dt — which sets the
    # stability floor for the swept timeconst — is the real one).
    horizon, substeps, rollout_dt = resolve_mppi_schedule(
        MPPIConfig(time_horizon=args.time_horizon, step_time=args.step_time),
        peek.config, args.eval_substeps,
    )

    # Microseconds keep the label an integer: 0.02 s -> M2_tc20000.
    timeconst_us = int(round(args.solref_timeconst * 1e6))
    label        = f"{args.model}_tc{timeconst_us}"
    control_freq = 1.0 / (eval_dt * eval_substeps * substeps)
    ratio        = args.solref_timeconst / rollout_dt

    print(f"[{label}]  nq={mjm.nq} nv={mjm.nv} nu={mjm.nu}  "
          f"max_steps={peek.config.max_steps}  n_episodes={args.n_episodes}")
    print(f"[{label}]  solref_timeconst={args.solref_timeconst*1e3:.3f} ms "
          f"= {ratio:.2f} x rollout_dt ({rollout_dt*1e3:.3f} ms)"
          + (f"  dampratio={args.solref_dampratio:g}"
             if args.solref_dampratio is not None else "  dampratio=<unchanged>"))
    if ratio < 2.0:
        print(f"[{label}]  WARNING: below the 2 x rollout_dt "
              f"({2*rollout_dt*1e3:.3f} ms) stability floor — contact may ring "
              f"or diverge. Running it as requested (no clamp).")
    print(f"[{label}]  control_freq={control_freq:.3f} Hz  "
          f"(eval_dt={eval_dt}, eval_substeps_per_rollout={eval_substeps}, "
          f"step_time={args.step_time:g}s -> substeps={substeps})")
    print(f"[{label}]  time_horizon={args.time_horizon:g}s -> {horizon} steps "
          f"({horizon*substeps*rollout_dt*1e3:.1f}ms)")

    # Reproducible per-episode seeds keyed by (seed, timeconst, model) so cells
    # never share a stream but a run repeats.
    model_ord = list(MODEL_FACTORIES).index(args.model)
    seed_seq  = np.random.SeedSequence(
        [s for s in (args.seed, timeconst_us, model_ord) if s is not None] or None
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

    schedule = {
        "eval_dt":                  eval_dt,
        "eval_substeps_per_rollout": eval_substeps,
        "rollout_dt":               rollout_dt,
        "substeps":                 substeps,
        "horizon":                  horizon,
        "control_freq":             control_freq,
    }
    return agg, label, schedule


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HPC worker: run one contact-time-constant sweep cell."
    )
    p.add_argument("--task",       type=str, default="grasp_reorient")
    p.add_argument("--model",      type=str, default="M2", choices=list(MODEL_FACTORIES),
                   help="Contact model for this cell. M2 is the default: its "
                        "solref comes from the XML/MuJoCo default (0.02 s) and "
                        "is freely overridable, whereas M1's is clamped at 2*dt.")
    p.add_argument("--solref_timeconst", type=float, required=True,
                   help="Contact time constant in SECONDS for this cell (the "
                        "swept axis). Smaller = stiffer. Applied to every "
                        "geom/pair solref of the rollout model; solimp is left "
                        "alone. Values below 2*rollout_dt are unstable and are "
                        "warned about, not clamped.")
    p.add_argument("--solref_dampratio", type=float, default=None,
                   help="Optional solref dampratio override (default: leave the "
                        "XML/preset value, which is 1.0 for these scenes).")
    p.add_argument("--n_episodes", type=int, default=10,
                   help="Episodes for this cell (used to estimate the success rate).")
    p.add_argument("--outdir",     type=str, default="results/contact_timeconst_eval_run",
                   help="Directory for the per-cell JSON output.")
    # --- MPPI / eval knobs (all held fixed across this sweep) ---------------
    p.add_argument("--n_samples",     type=int,   default=256,
                   help="Fixed MPPI sample count (held constant across the sweep).")
    p.add_argument("--time_horizon",  type=float, default=0.256,
                   help="MPPI planning horizon in SECONDS (held constant across the "
                        "sweep); quantized down to whole control steps.")
    p.add_argument("--step_time",     type=float, default=0.032,
                   help="Control-step duration in SECONDS (held constant across "
                        "the sweep; the nominal period used by the other sweeps).")
    p.add_argument("--temperature",   type=float, default=0.01)
    p.add_argument("--noise_sigma",   type=float, default=0.001)
    p.add_argument("--delta",         type=float, default=0.1,
                   help="Per-step MPPI delta clip magnitude (action units).")
    p.add_argument("--eval_substeps", type=int,   default=None,
                   help="Eval steps per rollout step (default: task config).")
    p.add_argument("--eval_sim",      type=str,   default="none",
                   choices=["none", "mujoco", "drake", "pinocchio"])
    p.add_argument("--settle",        type=float, default=10.0)
    p.add_argument("--geometry",      type=str,   default="accurate",
                   choices=[g.value for g in GeometryVariant])
    p.add_argument("--nconmax",       type=int,   default=200)
    p.add_argument("--njmax",         type=int,   default=500)
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

    # The label carries the timeconst in microseconds, so run_cell owns it.
    agg, label, schedule = run_cell(args)
    out_path = outdir / f"{label}.json"
    # save_results writes a JSON *list* of AggregatedResult dicts (one here), which
    # is exactly what the directory plotter merges across every cell file.
    save_results([agg], out_path)

    # One shared meta.json per outdir. Every cell in a sweep shares the same
    # task/schedule (only the timeconst varies), so this is safe to overwrite
    # from any cell. rollout_dt lets the plotter draw the 2*rollout_dt floor.
    meta_path = outdir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "task":      args.task,
            "model":     args.model,
            "step_time": args.step_time,
            **schedule,
        }, f, indent=2)


if __name__ == "__main__":
    main()
