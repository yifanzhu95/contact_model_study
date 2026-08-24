"""bayes_opt_to_csv_dir.py

Summarize the output of contact_study/drivers/run_bayes_opt.py — the sibling of
analysis/param_search_to_csv_dir.py for Bayesian optimization runs.

The two searches need different summaries. A param search is a grid, so its CSV
has one row per hyperparameter combination and rows are ranked against each
other. A BO run is a *sequence*: every trial is a distinct point, chosen in
response to the ones before it, so the interesting object is the trial order and
the running best — the convergence trace — not a ranking of unique combos.

What it reads
-------------
experiments/hpc/bayes_opt.slurm submits one INDEPENDENT optimization per
(object, contact model) cell, all under one parent directory keyed by the array
job id:

    results/bayes_opt_<array_job_id>/       <- a job directory
        grasp_reorient_cube_M1_mppi/        <- one BO run per cell
        grasp_reorient_cube_M2_mppi/
        grasp_reorient_duck_M1_mppi/
        ...

With no arguments this script picks the most recently modified
results/bayes_opt_* directory and summarizes every run inside it. A path that is
itself a run (bo_state.json sits directly in it, as with the driver's own
auto-named --outdir) is summarized on its own, so both layouts work.

Resume chains
-------------
A long BO run is usually several SLURM jobs chained with --resume, and only the
trials a job actually ran leave cell_<id>.json behind: a 500-trial run may hold
cells 300-499 and inherit 0-299 from the two jobs before it. Its bo_state.json,
however, carries the whole 500-trial trace. So this script takes bo_state.json
as authoritative for the trace and walks args.resume backwards to collect the
per-trial detail from every job in the chain.

Trials that have a trace entry but no cell file anywhere (a lost or moved job
directory) still produce a row — objective and coordinates come from bo_state —
with the episode-level columns left blank.

Outputs
-------
1. <job_dir>/bayes_opt_summary.csv — ONE row per BO run, carrying the inputs,
   the best hyperparameters it found, and the result summary side by side:

     job, run, task, object, geometry, model, planner, eval_sim   — identity
     n_episodes, seed, n_samples, time_horizon, ...               — episode inputs
     w_success, w_cost, err_clip                                  — objective inputs
     n_calls, n_initial_points, acq_func, gp_noise, bo_seed       — optimizer inputs
     n_dims, dims, noise_sigma_range, temperature_range           — search space
     best_<dim>, ...                                              — the tuned values
     n_trials, best_trial, best_objective, best_success_rate, ... — results
     status                                                       — complete / partial

2. <run_dir>/<run_dir_name>_trials.csv — one row per trial, in trial order:

     trial, run, <axis_key>..., objective, best_so_far, improved,
     success_rate, n_success, n_episodes, mean_norm_goal_err,
     mean_steps_to_success, mean_step_ms, mean_elapsed_s

   best_so_far is the running minimum — the convergence trace to plot.

Usage:
    python analysis/bayes_opt_to_csv_dir.py                       # newest job dir
    python analysis/bayes_opt_to_csv_dir.py results/bayes_opt_1234567
    python analysis/bayes_opt_to_csv_dir.py results/bayes_opt_1234567/grasp_reorient_cube_M2_mppi
    python analysis/bayes_opt_to_csv_dir.py --all --top_n 0       # every job dir, no tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from param_search_to_csv import write_csv

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Locating runs
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def is_run_dir(path: Path) -> bool:
    """A BO run directory holds bo_state.json directly."""
    return (path / "bo_state.json").is_file()


def expand(path: Path) -> list[Path]:
    """The BO runs under `path`: itself if it is a run, else its subdirectories.

    This is what makes one invocation cover both layouts — the driver's own
    auto-named --outdir (a run) and bayes_opt.slurm's parent job directory (a
    directory of runs, one per array cell).
    """
    if is_run_dir(path):
        return [path]
    return [p for p in sorted(path.iterdir()) if p.is_dir()]


def newest_job_dir() -> Path:
    """The most recently modified results/bayes_opt_* directory."""
    cands = [p for p in RESULTS_DIR.glob("bayes_opt_*") if p.is_dir()]
    if not cands:
        raise FileNotFoundError(f"No bayes_opt_* directories found in {RESULTS_DIR}")
    return max(cands, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Loading a run (and its resume chain)
# ---------------------------------------------------------------------------

def _resolve_resume(resume: str, run_dir: Path) -> Path | None:
    """The run directory a bo_state.json's args.resume points at, or None.

    The stored path is whatever the job passed on the command line — relative to
    the project root on the HPC. Fall back to matching the directory's name next
    to the current run, so a results tree that was zipped, moved or unpacked
    somewhere else still resolves.
    """
    p = Path(resume)
    for cand in (p, Path.cwd() / p, run_dir.parent / p.parent.name / p.name):
        if cand.is_file():
            return cand.parent
    return None


def resolve_chain(run_dir: Path) -> list[Path]:
    """The run's resume chain, oldest job first, ending at run_dir."""
    chain: list[Path] = []
    seen: set[Path] = set()
    cur: Path | None = run_dir

    while cur is not None:
        key = cur.resolve()
        if key in seen:            # malformed/cyclic chain — stop rather than loop
            break
        seen.add(key)
        chain.append(cur)

        state = _read_json(cur / "bo_state.json")
        resume = (state or {}).get("args", {}).get("resume")
        cur = _resolve_resume(resume, cur) if resume else None

    return list(reversed(chain))


def load_run(run_dir: Path) -> dict | None:
    """Load one BO run: its trace, its resume chain, and every trial record."""
    state = _read_json(run_dir / "bo_state.json")
    if state is None:
        return None

    chain = resolve_chain(run_dir)

    # combo_index -> (record, which job wrote it). Later jobs in the chain win,
    # so a re-run trial id resolves to its most recent evaluation.
    cells: dict[int, tuple[dict, str]] = {}
    for job in chain:
        for path in sorted(job.glob("cell_*.json")):
            rec = _read_json(path)
            if rec is not None and "combo_index" in rec:
                cells[int(rec["combo_index"])] = (rec, job.name)

    return {
        "dir":       run_dir,
        "chain":     chain,
        "args":      state.get("args", {}),
        "dim_names": list(state.get("dim_names", [])),
        "x_iters":   state.get("x_iters", []),
        "func_vals": [float(v) for v in state.get("func_vals", [])],
        "cells":     cells,
    }


def best_index(run: dict) -> int | None:
    ys = run["func_vals"]
    return min(range(len(ys)), key=lambda i: ys[i]) if ys else None


# ---------------------------------------------------------------------------
# Per-trial rows
# ---------------------------------------------------------------------------

DETAIL_COLS = [
    "success_rate", "n_success", "n_episodes", "mean_norm_goal_err",
    "mean_steps_to_success", "mean_step_ms", "mean_elapsed_s",
]


def build_trial_rows(run: dict) -> tuple[list[str], list[dict]]:
    """(fieldnames, rows) — one row per trial, in the order the BO ran them."""
    dim_names = run["dim_names"]
    fieldnames = (["trial", "run"] + dim_names
                  + ["objective", "best_so_far", "improved"] + DETAIL_COLS)

    rows: list[dict] = []
    best = float("inf")

    for trial, (x, y) in enumerate(zip(run["x_iters"], run["func_vals"])):
        improved = y < best
        best = min(best, y)

        row: dict = {"trial": trial}
        # Coordinates come from the trace, so a trial with no cell file still
        # gets its full hyperparameter vector.
        for name, val in zip(dim_names, x):
            row[name] = f"{float(val):.6g}"
        row["objective"]   = f"{y:.6f}"
        row["best_so_far"] = f"{best:.6f}"
        row["improved"]    = 1 if improved else 0

        rec_job = run["cells"].get(trial)
        if rec_job is None:
            row["run"] = ""
            for c in DETAIL_COLS:
                row[c] = ""
        else:
            rec, job = rec_job
            steps = rec.get("mean_steps_to_success")
            gerr  = rec.get("mean_norm_goal_err")
            row["run"]                   = job
            row["success_rate"]          = f"{rec['success_rate']:.4f}"
            row["n_success"]             = rec.get("n_success", "")
            row["n_episodes"]            = rec.get("n_episodes", "")
            row["mean_norm_goal_err"]    = f"{gerr:.6f}" if gerr is not None else ""
            row["mean_steps_to_success"] = f"{steps:.1f}" if steps is not None else ""
            row["mean_step_ms"]          = f"{rec['mean_step_ms']:.3f}"
            row["mean_elapsed_s"]        = f"{rec.get('mean_elapsed_s', 0.0):.1f}"
        rows.append(row)

    return fieldnames, rows


# ---------------------------------------------------------------------------
# The combined per-run row: inputs | best hyperparameters | results
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    """CSV-safe scalar: lists become space-joined, None becomes blank."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return v


def run_inputs(run: dict, job: str) -> dict:
    """What the run was ASKED to do — its CLI arguments, as columns."""
    a = run["args"]
    geometry = a.get("geometry", "") or ""
    return {
        "job":      job,
        "run":      run["dir"].name,
        "task":     _fmt(a.get("task")),
        # bayes_opt.slurm builds geometry as "<object>_<hand_acc>_<obj_acc>",
        # and the object is the axis worth grouping by.
        "object":   geometry.split("_")[0],
        "geometry": geometry,
        "model":    _fmt(a.get("model")),
        "planner":  _fmt(a.get("planner")),
        "eval_sim": _fmt(a.get("eval_sim")),

        "n_episodes":        _fmt(a.get("n_episodes")),
        "seed":              _fmt(a.get("seed")),
        "n_samples":         _fmt(a.get("n_samples")),
        "time_horizon":      _fmt(a.get("time_horizon")),
        "step_time":         _fmt(a.get("step_time")),
        "settle":            _fmt(a.get("settle")),
        "resample_interval": _fmt(a.get("resample_interval")),
        "eval_substeps":     _fmt(a.get("eval_substeps")),
        "delta":             _fmt(a.get("delta")),
        "warm_start":        _fmt(a.get("warm_start")),

        "w_success": _fmt(a.get("w_success")),
        "w_cost":    _fmt(a.get("w_cost")),
        "err_clip":  _fmt(a.get("err_clip")),

        "n_calls":          _fmt(a.get("n_calls")),
        "n_initial_points": _fmt(a.get("n_initial_points")),
        "acq_func":         _fmt(a.get("acq_func")),
        "gp_noise":         _fmt(a.get("gp_noise")),
        "bo_seed":          _fmt(a.get("bo_seed")),

        "n_dims":            len(run["dim_names"]),
        # The realized space, not args.opt_weights — which is null whenever the
        # driver's own default subset was accepted.
        "dims":              " ".join(run["dim_names"]),
        "noise_sigma_range": _fmt(a.get("noise_sigma_range")),
        "temperature_range": _fmt(a.get("temperature_range")),
    }


INPUT_COLS = list(run_inputs(
    {"args": {}, "dir": Path("x"), "dim_names": []}, "").keys())


def run_best_params(run: dict) -> dict:
    """The tuned values, as best_<dim> columns.

    Read off the trace's argmin rather than bo_best.json, so a cell that was
    killed before it could write its summary still reports its best point.
    """
    i = best_index(run)
    if i is None:
        return {}
    return {f"best_{name}": f"{float(v):.6g}"
            for name, v in zip(run["dim_names"], run["x_iters"][i])}


def run_results(run: dict) -> dict:
    """How the run turned out."""
    ys    = run["func_vals"]
    a     = run["args"]
    cells = run["cells"]

    i = best_index(run)
    rec = cells.get(i, (None, None))[0] if i is not None else None
    rates = [r["success_rate"] for r, _ in cells.values()
             if r.get("success_rate") is not None]

    elapsed = sum(
        float((_read_json(job / "bo_best.json") or {}).get("elapsed_seconds", 0.0) or 0.0)
        for job in run["chain"]
    )

    n_calls = a.get("n_calls")
    status = ""
    if isinstance(n_calls, int) and n_calls > 0:
        status = "complete" if len(ys) >= n_calls else f"partial {len(ys)}/{n_calls}"

    steps = rec.get("mean_steps_to_success") if rec else None
    gerr  = rec.get("mean_norm_goal_err") if rec else None
    return {
        "n_trials":  len(ys),
        "n_jobs":    len(run["chain"]),
        "best_trial":     i if i is not None else "",
        "best_objective": f"{ys[i]:.6f}" if i is not None else "",
        "best_success_rate":     f"{rec['success_rate']:.4f}" if rec else "",
        "best_norm_goal_err":    f"{gerr:.6f}" if gerr is not None else "",
        "best_steps_to_success": f"{steps:.1f}" if steps is not None else "",
        "best_step_ms":          f"{rec['mean_step_ms']:.3f}" if rec else "",
        # How stale the incumbent is: a large value means the search has stopped
        # finding anything and the budget is probably spent.
        "trials_since_best":    (len(ys) - 1 - i) if i is not None else "",
        "n_trials_all_success": sum(1 for r in rates if r >= 1.0),
        "n_trials_any_success": sum(1 for r in rates if r > 0.0),
        "n_detailed_trials":    len(cells),
        "n_missing_cells":      len(ys) - len(cells),
        "elapsed_hours":        f"{elapsed / 3600:.2f}" if elapsed else "",
        "status":               status,
    }


RESULT_COLS = list(run_results(
    {"func_vals": [], "args": {}, "cells": {}, "chain": [], "x_iters": [],
     "dim_names": []}).keys())


def summarize(run: dict, job: str) -> dict:
    """Inputs, best hyperparameters and results as one flat row."""
    return {**run_inputs(run, job), **run_best_params(run), **run_results(run)}


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_run_report(run: dict, row: dict, top_n: int) -> None:
    ys        = run["func_vals"]
    dim_names = run["dim_names"]
    print(f"\n{'-' * 78}")
    print(f"  {row['run']}")
    print(f"{'-' * 78}")
    if len(run["chain"]) > 1:
        print(f"  resume chain ({len(run['chain'])} jobs, oldest first):")
        for job in run["chain"]:
            n = sum(1 for _, j in run["cells"].values() if j == job.name)
            print(f"      {job.name}  ({n} cells)")
    print(f"  task={row['task']}  object={row['object']}  model={row['model']}  "
          f"planner={row['planner']}  eval_sim={row['eval_sim']}")
    print(f"  n_episodes={row['n_episodes']}  seed={row['seed']}  "
          f"trials={row['n_trials']}  dims={row['n_dims']}  "
          f"elapsed={row['elapsed_hours'] or '?'} h  {row['status']}")
    if row["n_missing_cells"]:
        print(f"  ! {row['n_missing_cells']} trial(s) have no cell file; "
              f"their episode columns are blank")

    if not ys:
        print("  (no trials)")
        return

    print(f"  trials with every episode successful: "
          f"{row['n_trials_all_success']}/{row['n_detailed_trials']}   "
          f"with any success: {row['n_trials_any_success']}/{row['n_detailed_trials']}")
    print(f"  best objective={row['best_objective']} at trial {row['best_trial']}  "
          f"({row['trials_since_best']} trials since)")

    i = int(row["best_trial"])
    print(f"\n  Best hyperparameters (trial {i}):")
    for name, val in zip(dim_names, run["x_iters"][i]):
        print(f"      {name:<16} {float(val):.6g}")

    if top_n <= 0:
        return
    # Ranked table — for a BO run this is a view, not the primary output; the
    # trial-ordered CSV with best_so_far is what you plot.
    order = sorted(range(len(ys)), key=lambda k: ys[k])[:top_n]
    cols  = ["trial", "objective", "success", "goal_err"] + dim_names
    print(f"\n  Top-{len(order)} trials by objective:")
    print("  " + "  ".join(f"{c:>13}" for c in cols))
    for k in order:
        rec = run["cells"].get(k, (None, None))[0]
        sr  = f"{rec['success_rate']:.3f}" if rec else "-"
        ge  = (f"{rec['mean_norm_goal_err']:.4f}"
               if rec and rec.get("mean_norm_goal_err") is not None else "-")
        vals = [str(k), f"{ys[k]:.5f}", sr, ge] + [
            f"{float(v):.4g}" for v in run["x_iters"][k]
        ]
        print("  " + "  ".join(f"{v:>13}" for v in vals))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize Bayesian-optimization runs (a bayes_opt.slurm job "
                    "directory, or a single run directory) as tidy CSVs.")
    parser.add_argument(
        "indirs", nargs="*", type=Path,
        help="Job directories (one subdirectory per BO run) or single run "
             "directories. Defaults to the most recently modified "
             "results/bayes_opt_* directory.")
    parser.add_argument(
        "--all", action="store_true",
        help="Summarize every results/bayes_opt_* directory instead of only the newest.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Combined summary CSV path. Defaults to <job_dir>/bayes_opt_summary.csv.")
    parser.add_argument(
        "--top_n", type=int, default=10,
        help="Trials to show in each ranked preview; 0 suppresses the tables (default 10).")
    parser.add_argument(
        "--no_trial_csv", action="store_true",
        help="Print the reports but do not write the per-run trial CSVs.")
    args = parser.parse_args()

    if args.indirs:
        roots = args.indirs
    elif args.all:
        roots = sorted(p for p in RESULTS_DIR.glob("bayes_opt_*") if p.is_dir())
    else:
        roots = [newest_job_dir()]
        print(f"Newest BO directory: {roots[0]}")

    rows:    list[dict] = []
    skipped: list[Path] = []
    dim_cols: list[str] = []       # union of best_<dim> columns, first-seen order

    for root in roots:
        if not root.is_dir():
            skipped.append(root)
            continue
        run_dirs = expand(root)
        print(f"\n{'=' * 78}")
        print(f"  {root}  ({len(run_dirs)} run(s))")
        print(f"{'=' * 78}")

        for run_dir in run_dirs:
            run = load_run(run_dir)
            if run is None or not run["func_vals"]:
                skipped.append(run_dir)
                continue

            # The submission a run belonged to: its parent, unless that is
            # just results/ (the driver's own auto-named outdir, not an array).
            parent = run_dir.resolve().parent
            job = "" if parent == RESULTS_DIR.resolve() else parent.name
            row = summarize(run, job)
            for name in run["dim_names"]:
                if f"best_{name}" not in dim_cols:
                    dim_cols.append(f"best_{name}")
            rows.append(row)
            print_run_report(run, row, args.top_n)

            if not args.no_trial_csv:
                fieldnames, trial_rows = build_trial_rows(run)
                write_csv(fieldnames, trial_rows,
                          run_dir / f"{run_dir.name}_trials.csv")

    if skipped:
        print(f"\n  Skipped {len(skipped)} directory(ies) with no usable "
              f"bo_state.json (crashed or never started):")
        for p in skipped:
            print(f"      {p.name}")

    if not rows:
        raise ValueError("No BO runs with trials were found.")

    # --- the combined CSV ---------------------------------------------------
    fieldnames = INPUT_COLS + dim_cols + RESULT_COLS
    for row in rows:                       # runs with fewer dims leave blanks
        for c in fieldnames:
            row.setdefault(c, "")
    rows.sort(key=lambda r: float(r["best_objective"]))

    if args.output is not None:
        out = args.output
    elif len(roots) == 1 and roots[0].is_dir():
        # Beside whatever was pointed at — the job directory, or the single run.
        out = roots[0] / "bayes_opt_summary.csv"
    else:
        out = RESULTS_DIR / "bayes_opt_summary.csv"

    print(f"\n{'=' * 78}")
    print(f"  {len(rows)} run(s), best first")
    print(f"{'=' * 78}")
    cols = ["object", "model", "n_trials", "best_objective",
            "best_success_rate", "elapsed_hours", "status"]
    print("  " + "  ".join(f"{c:>17}" for c in cols))
    for row in rows:
        print("  " + "  ".join(f"{str(row[c])[-17:]:>17}" for c in cols))
    write_csv(fieldnames, rows, out)


if __name__ == "__main__":
    main()
