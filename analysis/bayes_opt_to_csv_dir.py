"""bayes_opt_to_csv_dir.py

Summarize the output of contact_study/drivers/run_bayes_opt.py — the sibling of
analysis/param_search_to_csv_dir.py for Bayesian optimization runs.

The two searches need different summaries. A param search is a grid, so its CSV
has one row per hyperparameter combination and rows are ranked against each
other. A BO run is a *sequence*: every trial is a distinct point, chosen in
response to the ones before it, so the interesting object is the trial order and
the running best — the convergence trace — not a ranking of unique combos.

Resume chains
-------------
A long BO run is usually several SLURM jobs chained with --resume, and only the
trials a job actually ran leave cell_<id>.json behind: the 500-trial run in
results/bayes_opt_..._144130 holds cells 300-499 and inherits 0-299 from the two
jobs before it. Its bo_state.json, however, carries the whole 500-trial trace.
So this script takes bo_state.json as authoritative for the trace and walks
args.resume backwards to collect the per-trial detail from every job in the
chain. Point it at the LAST directory of a chain and you get the whole history.

Trials that have a trace entry but no cell file anywhere (a lost or moved job
directory) still produce a row — objective and coordinates come from bo_state —
with the episode-level columns left blank.

Outputs
-------
<run_dir>/<run_dir_name>_trials.csv, one row per trial in trial order:

    trial                  — index within the chain (matches cell_<id>.json)
    run                    — which job in the chain ran it
    <axis_key>, ...        — one column per search dimension (w_quat, ...,
                             noise_sigma, temperature)
    objective              — the minimized value, -(w_success * success_rate)
                             + w_cost * norm_goal_err
    best_so_far            — running minimum: the convergence trace to plot
    improved               — 1 when this trial set a new best, else 0
    success_rate, n_success, n_episodes
    mean_norm_goal_err     — normalized final goal error (0 = at goal, 1 = clipped)
    mean_steps_to_success  — blank when no episode succeeded
    mean_step_ms, mean_elapsed_s

With more than one run directory, also writes a cross-run summary CSV with one
row per chain.

Usage:
    python analysis/bayes_opt_to_csv_dir.py results/bayes_opt_grasp_reorient_M2_mppi_20260821_144130
    python analysis/bayes_opt_to_csv_dir.py                    # every results/bayes_opt_* run
    python analysis/bayes_opt_to_csv_dir.py results/bayes_opt_* --top_n 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from param_search_to_csv import write_csv

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Loading a run (and its resume chain)
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


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
        "best":      _read_json(run_dir / "bo_best.json"),
    }


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
# Per-run summary
# ---------------------------------------------------------------------------

def summarize(run: dict) -> dict:
    """One-row summary of a whole chain: budget, best trial, convergence state."""
    ys    = run["func_vals"]
    args  = run["args"]
    cells = run["cells"]

    best_i = min(range(len(ys)), key=lambda i: ys[i]) if ys else None
    best_rec = cells.get(best_i, (None, None))[0] if best_i is not None else None

    rates = [r["success_rate"] for r, _ in cells.values() if r.get("success_rate") is not None]
    elapsed = sum(
        float((_read_json(job / "bo_best.json") or {}).get("elapsed_seconds", 0.0) or 0.0)
        for job in run["chain"]
    )

    return {
        "run":            run["dir"].name,
        "task":           args.get("task", ""),
        "model":          args.get("model", ""),
        "planner":        args.get("planner", ""),
        "n_trials":       len(ys),
        "n_jobs":         len(run["chain"]),
        "n_episodes":     args.get("n_episodes", ""),
        "seed":           args.get("seed", ""),
        "n_dims":         len(run["dim_names"]),
        "best_trial":     best_i if best_i is not None else "",
        "best_objective": f"{ys[best_i]:.6f}" if best_i is not None else "",
        "best_success_rate": (f"{best_rec['success_rate']:.4f}" if best_rec else ""),
        "best_norm_goal_err": (
            f"{best_rec['mean_norm_goal_err']:.6f}"
            if best_rec and best_rec.get("mean_norm_goal_err") is not None else ""
        ),
        # How stale the incumbent is: a large value means the search has stopped
        # finding anything and the budget is probably spent.
        "trials_since_best": (len(ys) - 1 - best_i) if best_i is not None else "",
        "n_trials_any_success": sum(1 for r in rates if r > 0.0),
        "n_trials_all_success": sum(1 for r in rates if r >= 1.0),
        "n_missing_cells": len(ys) - len(cells),
        "elapsed_hours":  f"{elapsed / 3600:.2f}" if elapsed else "",
    }


def print_run_report(run: dict, summary: dict, top_n: int) -> None:
    ys        = run["func_vals"]
    dim_names = run["dim_names"]
    print(f"\n{'=' * 78}")
    print(f"  {run['dir'].name}")
    print(f"{'=' * 78}")
    if len(run["chain"]) > 1:
        print(f"  resume chain ({len(run['chain'])} jobs, oldest first):")
        for job in run["chain"]:
            n = sum(1 for _, j in run["cells"].values() if j == job.name)
            print(f"      {job.name}  ({n} cells)")
    print(f"  task={summary['task']}  model={summary['model']}  "
          f"planner={summary['planner']}  n_episodes={summary['n_episodes']}  "
          f"seed={summary['seed']}")
    print(f"  trials={summary['n_trials']}  dims={summary['n_dims']}  "
          f"elapsed={summary['elapsed_hours'] or '?'} h")
    if summary["n_missing_cells"]:
        print(f"  ! {summary['n_missing_cells']} trial(s) have no cell file; "
              f"their episode columns are blank")

    if not ys:
        print("  (no trials)")
        return

    rates_all  = summary["n_trials_all_success"]
    rates_any  = summary["n_trials_any_success"]
    n_detailed = len(run["cells"])
    print(f"  trials with every episode successful: {rates_all}/{n_detailed}   "
          f"with any success: {rates_any}/{n_detailed}")
    print(f"  best objective={summary['best_objective']} at trial "
          f"{summary['best_trial']}  ({summary['trials_since_best']} trials since)")

    best_i = int(summary["best_trial"])
    print(f"\n  Best hyperparameters (trial {best_i}):")
    for name, val in zip(dim_names, run["x_iters"][best_i]):
        print(f"      {name:<16} {float(val):.6g}")

    # Ranked table — for a BO run this is a view, not the primary output; the
    # trial-ordered CSV with best_so_far is what you plot.
    order = sorted(range(len(ys)), key=lambda i: ys[i])[:top_n]
    cols  = ["trial", "objective", "success", "goal_err"] + dim_names
    print(f"\n  Top-{len(order)} trials by objective:")
    print("  " + "  ".join(f"{c:>13}" for c in cols))
    for i in order:
        rec = run["cells"].get(i, (None, None))[0]
        sr  = f"{rec['success_rate']:.3f}" if rec else "-"
        ge  = (f"{rec['mean_norm_goal_err']:.4f}"
               if rec and rec.get("mean_norm_goal_err") is not None else "-")
        vals = [str(i), f"{ys[i]:.5f}", sr, ge] + [
            f"{float(v):.4g}" for v in run["x_iters"][i]
        ]
        print("  " + "  ".join(f"{v:>13}" for v in vals))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize Bayesian-optimization run directories "
                    "(bo_state.json + cell_*.json) as tidy CSVs.")
    parser.add_argument(
        "indirs", nargs="*", type=Path,
        help="BO output directories. Defaults to every results/bayes_opt_* directory.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Cross-run summary CSV path. Defaults to results/bayes_opt_summary.csv.")
    parser.add_argument(
        "--top_n", type=int, default=10,
        help="How many trials to show in the ranked preview (default 10).")
    parser.add_argument(
        "--no_trial_csv", action="store_true",
        help="Print the reports but do not write the per-run trial CSVs.")
    args = parser.parse_args()

    indirs = args.indirs or sorted(RESULTS_DIR.glob("bayes_opt_*"))
    if not indirs:
        raise FileNotFoundError(f"No BO run directories found in {RESULTS_DIR}")

    summaries: list[dict] = []
    skipped:   list[Path] = []

    for indir in indirs:
        if not indir.is_dir():
            skipped.append(indir)
            continue
        run = load_run(indir)
        if run is None or not run["func_vals"]:
            skipped.append(indir)
            continue

        summary = summarize(run)
        summaries.append(summary)
        print_run_report(run, summary, args.top_n)

        if not args.no_trial_csv:
            fieldnames, rows = build_trial_rows(run)
            write_csv(fieldnames, rows, indir / f"{indir.name}_trials.csv")

    if skipped:
        print(f"\n  Skipped {len(skipped)} directory(ies) with no usable "
              f"bo_state.json (crashed or never started):")
        for p in skipped:
            print(f"      {p.name}")

    if not summaries:
        raise ValueError("No BO runs with trials were found.")

    if len(summaries) > 1:
        out = args.output or RESULTS_DIR / "bayes_opt_summary.csv"
        fieldnames = list(summaries[0])
        # Rank chains by their best objective so the best run reads first.
        summaries.sort(key=lambda s: float(s["best_objective"]))
        print(f"\n{'=' * 78}")
        print(f"  {len(summaries)} runs, best first")
        print(f"{'=' * 78}")
        cols = ["run", "n_trials", "best_trial", "best_objective",
                "best_success_rate", "elapsed_hours"]
        print("  " + "  ".join(f"{c:>16}" for c in cols))
        for s in summaries:
            print("  " + "  ".join(f"{str(s[c])[-16:]:>16}" for c in cols))
        write_csv(fieldnames, summaries, out)


if __name__ == "__main__":
    main()
