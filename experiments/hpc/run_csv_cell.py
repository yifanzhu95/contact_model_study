"""run_csv_cell.py

HPC worker for CSV-driven experiment sweeps — one SLURM array task = one CSV row.

Generalizes run_param_cell.py: instead of a grid hardcoded as bash arrays, every
experiment is one row of an arbitrary CSV. Reserved columns (RESERVED_COLUMNS)
pick the task/model/planner/episode-count settings; `w_<name>` columns become
cost-weight overrides; every other column is forwarded straight to
`make_planner_config(planner, **kwargs)`, which already drops keys the selected
planner's config doesn't declare (contact_study/planners/__init__.py) — so a
single CSV can mix mppi/cem/predictive_sampler rows, and a blank cell simply
falls back to that planner's own default. `delta` is the one special-cased knob:
a single clip magnitude, expanded to the `delta_range=(-delta, delta)` field,
matching run_eval_episode.py's own --delta convention. `record_trajectory` /
`record_planner_dist` / `planner_dist_every` columns override the CLI's
recording flags per row, so a sweep can run lean by default and still record
a handful of interesting cells in full.

Two rows can also differ in HOW the episode is run, not just in what is planned:

  driver=async        runs contact_study/drivers/run_async_eval_episode.py
                      instead of run_eval_episode: the eval sim keeps running
                      while the planner solves, so planning latency is charged
                      as simulated time rather than being free. The five
                      ASYNC_COLUMNS (plan_latency_ms, latency_scale,
                      plan_warmup, executor, async_shift) tune it and are
                      rejected on a sync row.
  time_constrained    an ordinary forwarded planner field, but one that
                      PlannerConfig rejects unless plan_budget_ms and
                      use_full_graph agree with it; resolve_time_constraint
                      fills both in so the single cell is enough. Together with
                      driver=async this is the anytime planner: latency capped
                      by truncating the horizon.

Output is written as cell_<row>.json in the same shape run_param_cell.py's
run_cell() returns, so combine_results.py / combine.slurm merge it unchanged.

    python run_csv_cell.py --csv params.csv --row 0 --outdir results/csv_run
"""

from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import csv
from pathlib import Path

import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.evaluation import json_io
from contact_study.evaluation.trajectory import (
    TrajectoryConfig, add_cli_flags as add_record_flags,
)
from contact_study.drivers.run_eval_episode import (
    run_eval_episode, load_rollout_task, MODEL_FACTORIES,
)
from contact_study.drivers.run_async_eval_episode import run_async_eval_episode
from contact_study.planners import (
    PLANNERS, make_planner_config, planner_config_cls, resolve_planner_name,
)
from contact_study.tasks.config import (
    DEFAULT_HAND_ACC, DEFAULT_OBJ_ACC, DEFAULT_SCENE_VARIANT, EvalSimulatorKind,
)


# ---------------------------------------------------------------------------
# CSV row parsing
# ---------------------------------------------------------------------------

# Columns that pick experiment-level settings rather than a planner knob or a
# cost-weight override; handled explicitly instead of forwarded to
# make_planner_config. `eval_substeps` belongs here rather than with the planner
# knobs: it is a run_eval_episode argument (eval steps per rollout step), not a
# field of any planner config, so forwarding it would silently drop it.
# The three recording-flag columns mirror add_cli_flags's --record_trajectory /
# --record_planner_dist / --planner_dist_every, letting a row opt into full
# recording (e.g. to inspect one interesting cell) while the rest of the sweep
# stays lean.
# `hand_acc` / `obj_acc` split the collision-geometry fidelity out of
# `geometry` (see contact_study/tasks/config.py:SceneVariant) into their own
# columns, so a sweep can vary them independently of the object; when either is
# set, `geometry` is taken as the bare object name and the three are joined
# into "<geometry>_<hand_acc>_<obj_acc>" (bayes_opt.slurm's own convention).
# `driver` picks the control loop: "sync" (default) runs run_eval_episode, which
# freezes the eval sim for the duration of plan(); "async" runs
# run_async_eval_episode, which keeps the sim running and charges the planning
# latency as simulated time. The two take the same arguments and return the same
# EpisodeResult, so only the call target changes. ASYNC_COLUMNS are
# run_async_eval_episode ARGUMENTS rather than planner-config fields, so they
# have to be reserved here — forwarding them would hit validate_columns as
# unknown columns, and they are rejected outright on a sync row rather than
# silently ignored.
RESERVED_COLUMNS = (
    "task", "model", "planner", "n_episodes", "seed", "geometry", "hand_acc",
    "obj_acc", "eval_sim", "settle", "eval_substeps",
    "record_trajectory", "record_planner_dist", "planner_dist_every",
    "driver", "plan_latency_ms", "latency_scale", "plan_warmup",
    "executor", "async_shift",
)

DRIVERS = ("sync", "async")
ASYNC_COLUMNS = ("plan_latency_ms", "latency_scale", "plan_warmup",
                 "executor", "async_shift")


def planner_field_names() -> set[str]:
    """Union of the dataclass fields every registered planner config declares.

    Used only to catch typo'd CSV columns: make_planner_config silently ignores
    keys the selected planner does not declare (that is what lets one CSV mix
    planners), which would otherwise turn a misspelled `temprature` column into
    a whole sweep quietly running the default.
    """
    import dataclasses
    names: set[str] = set()
    for cfg_cls, _ in PLANNERS.values():
        names |= {f.name for f in dataclasses.fields(cfg_cls)}
    # Accepted as a CSV column but expanded into delta_range (see split_row).
    names.add("delta")
    return names


def validate_columns(fieldnames) -> None:
    """Fail fast on a column that is neither reserved, nor a `w_` weight, nor a
    field of ANY planner config — almost always a typo."""
    known = planner_field_names()
    unknown = [c for c in (fieldnames or [])
               if c and c not in RESERVED_COLUMNS and not c.startswith("w_")
               and c not in known]
    if unknown:
        raise SystemExit(
            f"unknown CSV column(s): {', '.join(unknown)}\n"
            f"  expected a reserved column ({', '.join(RESERVED_COLUMNS)}), "
            f"a 'w_<name>' cost weight, or a planner-config field "
            f"({', '.join(sorted(known))})"
        )


def coerce(raw: str):
    """CSV cells arrive as strings; recover the type a dataclass field expects.
    Tries int -> float -> bool -> leaves it as a string (e.g. 'accurate')."""
    value = raw.strip()
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def load_row(csv_path: Path, row: int) -> dict:
    """Row `row` (0-indexed over data rows, i.e. excluding the header) of
    csv_path, as {column: raw string}. Validates the header first, then bounds —
    this is the authoritative range check (the SLURM script does not re-count)."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        validate_columns(reader.fieldnames)
        rows = list(reader)
    if not (0 <= row < len(rows)):
        raise SystemExit(
            f"--row {row} is outside {csv_path} ({len(rows)} data rows, "
            f"valid range 0-{len(rows) - 1})"
        )
    return rows[row]


def split_row(raw_row: dict) -> tuple[dict, dict, dict]:
    """Split one CSV row into (experiment_kwargs, weight_overrides, planner_kwargs).

    Blank cells are dropped entirely (not passed as ""), so that field's own
    default applies downstream, per make_planner_config's contract."""
    experiment, overrides, planner_kwargs = {}, {}, {}
    for name, raw in raw_row.items():
        if raw is None or raw.strip() == "":
            continue
        value = coerce(raw)
        if name in RESERVED_COLUMNS:
            experiment[name] = value
        elif name.startswith("w_"):
            overrides[name] = value
        else:
            planner_kwargs[name] = value

    # delta_range is a (low, high) tuple, awkward to spell in a CSV cell; let a
    # `delta` column give a single clip magnitude instead, same as
    # run_eval_episode.py's --delta.
    if "delta" in planner_kwargs:
        d = planner_kwargs.pop("delta")
        planner_kwargs["delta_range"] = (-d, d)

    return experiment, overrides, planner_kwargs


def resolve_time_constraint(planner: str, planner_kwargs: dict, row_id: int) -> None:
    """Make a `time_constrained` row runnable, editing planner_kwargs in place.

    PlannerConfig rejects time_constrained without plan_budget_ms > 0, and again
    with use_full_graph=True (the full-graph unroll is one captured CUDA graph and
    cannot stop mid-horizon), so the obvious two-cell row would crash in
    __post_init__ before anything ran. Default the budget to the row's control
    period and force the step-graph path, matching run_async_eval_episode.py's
    `use_full_graph = not time_constrained` and run_cntrl_freq_cell.py's
    resolve_plan_budget_ms.
    """
    if not planner_kwargs.get("time_constrained"):
        return
    if planner_kwargs.get("plan_budget_ms") is None:
        # A blank step_time cell means that config's own default applies; read it
        # off the dataclass so the budget matches the period that will actually
        # run (a dataclass field's class attribute IS its default).
        step_time = planner_kwargs.get("step_time", planner_config_cls(planner).step_time)
        planner_kwargs["plan_budget_ms"] = float(step_time) * 1e3
        print(f"[row {row_id}]  time_constrained with no plan_budget_ms -> "
              f"{planner_kwargs['plan_budget_ms']:g} ms (step_time={step_time:g}s)")
    if planner_kwargs.get("use_full_graph", True):
        print(f"[row {row_id}]  time_constrained forces use_full_graph=false "
              f"(the full-graph unroll cannot stop early); overriding.")
        planner_kwargs["use_full_graph"] = False


def combo_label(model_key: str, planner: str, axes: dict) -> str:
    """Short label encoding model + planner + every non-blank axis value."""
    parts = []
    for k, v in axes.items():
        short = k[2:] if k.startswith("w_") else k
        parts.append(f"{short}={v:g}" if isinstance(v, (int, float)) else f"{short}={v}")
    tag = f"{model_key}_{planner}"
    return f"{tag}__" + "_".join(parts) if parts else tag


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------

def run_cell(row_id: int, experiment: dict, overrides: dict, planner_kwargs: dict, args) -> dict:
    """Run one CSV row (n_episodes episodes) and return the result dict."""
    task_name = experiment.get("task", args.task)
    if task_name is None:
        raise ValueError(f"row {row_id}: no 'task' column and no --task default given")
    model      = experiment.get("model", args.model)
    planner    = resolve_planner_name(experiment.get("planner", args.planner))
    n_episodes = int(experiment.get("n_episodes", args.n_episodes))

    # geometry/hand_acc/obj_acc together pick the scene variant (see
    # SceneVariant.parse in contact_study/tasks/config.py). hand_acc/obj_acc
    # split the collision-fidelity axes out of geometry so they can be swept
    # independently; when either is given, `geometry` is read as the bare
    # object name (falling back to DEFAULT_OBJECT, not the full default
    # variant, which already carries its own accuracy suffix) and the three
    # are joined. With neither given, geometry behaves exactly as before
    # (a bare object name or an already-composed "<obj>_<hand>_<obj>" string).
    geometry_col = experiment.get("geometry")
    hand_acc = experiment.get("hand_acc", args.hand_acc)
    obj_acc  = experiment.get("obj_acc", args.obj_acc)
    if hand_acc is not None or obj_acc is not None:
        obj = geometry_col if geometry_col is not None else DEFAULT_OBJECT
        hand_acc = hand_acc if hand_acc is not None else DEFAULT_HAND_ACC
        obj_acc  = obj_acc  if obj_acc  is not None else DEFAULT_OBJ_ACC
        geometry = f"{obj}_{hand_acc}_{obj_acc}"
    else:
        geometry = geometry_col if geometry_col is not None else args.geometry

    eval_sim_raw = experiment.get("eval_sim", args.eval_sim)
    eval_sim   = None if eval_sim_raw in (None, "none") else EvalSimulatorKind(eval_sim_raw)
    settle     = float(experiment.get("settle", args.settle))
    base_seed  = experiment.get("seed", args.seed)
    eval_substeps = experiment.get("eval_substeps", args.eval_substeps)
    if eval_substeps is not None:
        eval_substeps = int(eval_substeps)

    # Recording defaults to the CLI flags, overridable per row so a sweep can
    # run lean (--no-record_trajectory --no-record_planner_dist) while a few
    # rows opt into full recording for inspection.
    cli_record = TrajectoryConfig.from_args(args)
    record_cfg = TrajectoryConfig(
        record_trajectory   = bool(experiment.get("record_trajectory", args.record_trajectory)),
        record_planner_dist = bool(experiment.get("record_planner_dist", args.record_planner_dist)),
        planner_dist_every  = max(1, int(experiment.get("planner_dist_every", args.planner_dist_every))),
        precision           = cli_record.precision,
        shrinkage           = cli_record.shrinkage,
    )

    # Which control loop runs the episode. The async knobs are meaningless to the
    # synchronous driver, so a row that sets one without asking for async is a
    # mistake worth failing on rather than silently ignoring — the same reasoning
    # behind validate_columns' typo guard.
    driver = experiment.get("driver", "sync")
    if driver not in DRIVERS:
        raise SystemExit(
            f"row {row_id}: driver must be one of {', '.join(DRIVERS)}, got {driver!r}"
        )
    async_kwargs = {k: experiment[k] for k in ASYNC_COLUMNS if k in experiment}
    if driver != "async" and async_kwargs:
        names = sorted(async_kwargs)
        raise SystemExit(
            f"row {row_id}: {', '.join(names)} "
            f"{'only applies' if len(names) == 1 else 'only apply'} with "
            f"driver=async, but this row is driver={driver!r}"
        )
    # Must run before make_planner_config: the rejection it works around happens
    # in PlannerConfig.__post_init__.
    resolve_time_constraint(planner, planner_kwargs, row_id)

    cfg = MODEL_FACTORIES[model]()
    # driver/async knobs are reserved columns, so unlike the planner kwargs they
    # are not in the label by construction — fold them in, or a sweep comparing
    # sync against async produces two cells with identical labels and
    # combine_results.py collapses them into one row. Keyed off the column being
    # PRESENT (not its value) so a CSV without a driver column keeps the labels
    # it produces today, exactly as a blank cell does.
    sweep_axes = {**overrides, **planner_kwargs,
                  **({"driver": driver} if "driver" in experiment else {}),
                  **async_kwargs}
    axes  = {**sweep_axes, "model": model, "planner": planner}
    label = combo_label(model, planner, sweep_axes)

    # Peek at the rollout task once for default cost weights (to report the
    # fully-resolved weight set alongside the overrides).
    peek_task       = load_rollout_task(task_name, geometry)
    default_weights = dict(peek_task.config.cost_weights)
    full_weights    = {**default_weights, **overrides}

    print(f"[row {row_id}]  {label}")
    print(f"  task={task_name}  model={model}  planner={planner}  "
          f"n_episodes={n_episodes}  driver={driver}")
    if async_kwargs:
        print(f"  async kwargs    ={async_kwargs}")
    print(f"  weight overrides={overrides}")
    if planner_kwargs:
        print(f"  planner kwargs  ={planner_kwargs}")

    # Reproducible per-episode seeds derived from the base seed AND the row id,
    # so different rows never share a seed but a run is repeatable.
    seed_seq      = np.random.SeedSequence(
        [s for s in (base_seed, row_id) if s is not None] or None)
    episode_seeds = seed_seq.spawn(n_episodes)

    episodes = []
    for ep in range(n_episodes):
        ep_seed = int(episode_seeds[ep].generate_state(1)[0])
        rng     = np.random.default_rng(episode_seeds[ep])

        # Merged as one dict rather than passed as separate keywords: `debug` is
        # itself a planner-config field, so a CSV `debug` column would otherwise
        # collide with debug=args.debug ("multiple values for keyword"). The CSV
        # wins over the CLI flag; seed is always ours (it is a reserved column,
        # so it can never appear in planner_kwargs).
        planner_cfg = make_planner_config(
            planner, **{"debug": args.debug, **planner_kwargs, "seed": ep_seed},
        )

        # Same arguments either way: run_async_eval_episode takes every one of
        # these and returns the same EpisodeResult, so only the target changes.
        run_episode = run_async_eval_episode if driver == "async" else run_eval_episode
        result = run_episode(
            task_name             = task_name,
            contact_cfg           = cfg,
            planner_cfg           = planner_cfg,
            planner                = planner,
            rng                   = rng,
            geometry              = geometry,
            cost_weight_overrides = overrides,
            settle_seconds        = settle,
            eval_substeps         = eval_substeps,
            eval_sim              = eval_sim,
            ep_idx                = ep,
            fin_ep_on_success     = True,
            debug                 = args.debug,
            verbose               = args.debug,
            record                = record_cfg,
            **(async_kwargs if driver == "async" else {}),
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
        "combo_index":           row_id,
        "task":                  task_name,
        "model":                 model,
        "planner":               planner,
        "label":                 label,
        "overrides":             overrides,
        # Every grid axis (weights + planner kwargs) — what the analysis
        # scripts group rows by. "overrides" stays weights-only for
        # combine_results.py.
        "axes":                  axes,
        "swept_knobs":           list(planner_kwargs),
        "full_weights":          full_weights,
        "n_episodes":            len(episodes),
        "n_success":             n_success,
        "success_rate":          success_rate,
        "mean_steps_to_success": mean_sts,
        "mean_step_ms":          float(np.mean(step_ms)),
        "std_step_ms":           float(np.mean(step_sd)),
        "mean_elapsed_s":        float(np.mean(elapsed)),
        "planner_kwargs":        planner_kwargs,
        "seed":     base_seed,
        "driver":       driver,
        "async_kwargs": async_kwargs,
        "eval_sim": eval_sim_raw,
        "eval_substeps": eval_substeps,
        "geometry": geometry,
        "hand_acc": hand_acc,
        "obj_acc":  obj_acc,
        "settle":   settle,
        "episodes": [r.to_dict() for r in episodes],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv",  type=str, required=True,
                   help="CSV of experiments; one row = one cell. See RESERVED_COLUMNS "
                        "for the task/model/planner/n_episodes/seed/geometry/eval_sim/"
                        "settle columns; 'w_<name>' columns are cost-weight overrides; "
                        "every other column is a planner-config field.")
    p.add_argument("--row",  type=int, default=None,
                   help="0-indexed data row to run (default: $SLURM_ARRAY_TASK_ID).")
    p.add_argument("--outdir", type=str, default="results/csv_sweep_run",
                   help="Directory for the per-row JSON output.")
    # --- defaults used when a row's RESERVED_COLUMNS cell is blank/absent ---
    p.add_argument("--task",     type=str, default=None)
    p.add_argument("--model",    type=str, default="M2", choices=list(MODEL_FACTORIES))
    p.add_argument("--planner",  type=str, default="mppi")
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed",     type=int, default=None)
    p.add_argument("--geometry", type=str, default=DEFAULT_SCENE_VARIANT)
    p.add_argument("--hand_acc", type=str, default=None,
                   help="Hand collision-geometry fidelity (e.g. low/high); combined "
                        "with geometry/obj_acc into the scene variant when set.")
    p.add_argument("--obj_acc",  type=str, default=None,
                   help="Object collision-geometry fidelity (e.g. low/high); combined "
                        "with geometry/hand_acc into the scene variant when set.")
    p.add_argument("--eval_sim", type=str, default="none",
                   choices=["none", "mujoco", "drake", "pinocchio"])
    p.add_argument("--settle",   type=float, default=1.0)
    p.add_argument("--eval_substeps", type=int, default=None,
                   help="Eval steps per rollout step (default: task config).")
    add_record_flags(p)
    p.add_argument("--debug", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    if args.row is not None:
        row_id = args.row
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        row_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        raise SystemExit("pass --row, or run under a SLURM array (SLURM_ARRAY_TASK_ID)")

    csv_path = Path(args.csv)
    raw_row  = load_row(csv_path, row_id)
    experiment, overrides, planner_kwargs = split_row(raw_row)

    wp.init()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    record   = run_cell(row_id, experiment, overrides, planner_kwargs, args)
    out_path = outdir / f"cell_{row_id:05d}.json"
    json_io.dump(record, out_path,
                 precision=TrajectoryConfig.from_args(args).precision)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
