"""csv_sweep_to_csv.py

Summarize the output of experiments/hpc/run_csv_sweep.slurm — one
cell_<row>.json per row of the params CSV, written by run_csv_cell.py — as ONE
CSV ROW PER SWEEP ROW, in the order of the input CSV.

The sweep's own combine step (combine_results.py) collapses cells that share a
label and drops the per-episode records; this script keeps every row separate
(row 3 and row 7 stay rows 3 and 7 even if identical) and pulls the per-episode
detail up into the summary. Each output row carries

    row, status, label                                 — identity
    task, model, planner, geometry, hand_acc, obj_acc,
    driver, eval_sim, eval_substeps, settle, seed,
    goal_difficulty                                     — the reserved columns
    <axis>, ...                                         — every non-weight knob
                                                          the row set
                                                          (temperature, n_samples,
                                                          plan_latency_ms, ...),
                                                          union over all rows
    n_episodes, n_success, success_rate,
    mean_steps_to_success, mean_step_ms, std_step_ms    — the outcome
    n_timeout, n_failed, n_error, error                 — how the episodes ended
    mean_n_steps_taken, mean_final_cost,
    mean_final_<k>_err                                  — final-state detail
                                                          (pos/quat/vel for
                                                          grasp_reorient)
    mean_elapsed_s, total_elapsed_min                   — wall clock, per episode
                                                          and per row (to size
                                                          the job's --time)
    mean_latency_ms, mean_staleness_ms, ...             — async-driver telemetry,
                                                          only when any row ran
                                                          driver=async
    w_*                                                 — the FULL resolved cost
                                                          weights the row ran
                                                          under, overrides and
                                                          task defaults alike

Rows the sweep never finished
-----------------------------
A row whose task hit the job's --time limit (or crashed) leaves no
cell_<row>.json, and a gap in the numbering is easy to miss in a long sweep.
Gaps are always reported on the console; pass the params CSV the sweep ran
(`--csv`) and each missing row is written out too, with its inputs filled from
the CSV and `status=missing`, so the summary lines up with the input file.

Usage:
    python analysis/csv_sweep_to_csv.py                          # newest job dir
    python analysis/csv_sweep_to_csv.py results/csv_sweep_1234567
    python analysis/csv_sweep_to_csv.py results/csv_sweep_1234567 --csv params.csv
    python analysis/csv_sweep_to_csv.py results/csv_sweep_1234567/combined_csv_sweep_rich.json
    python analysis/csv_sweep_to_csv.py --rank                   # best rows first
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from param_search_to_csv import write_csv

RESULTS_DIR = Path(__file__).parent.parent / "results"

# run_csv_cell.py's RESERVED_COLUMNS that describe the experiment (as opposed to
# the recording flags, which do not change the outcome). Each is a top-level key
# of the cell record; they come first so a row reads like the input CSV did.
RESERVED_COLS = ("task", "model", "planner", "geometry", "hand_acc", "obj_acc",
                 "driver", "eval_sim", "eval_substeps", "settle", "seed",
                 "goal_difficulty")

OUTCOME_COLS = ("n_episodes", "n_success", "success_rate",
                "mean_steps_to_success", "mean_step_ms", "std_step_ms",
                "n_timeout", "n_failed", "n_error", "error",
                "mean_n_steps_taken", "mean_final_cost",
                "mean_eff_horizon_steps", "mean_eff_horizon_s",
                "mean_elapsed_s", "total_elapsed_min")

# EpisodeResult's async-driver fields (contact_study/evaluation/metrics.py); all
# zero on a synchronous row, so the columns are only emitted when some row
# actually ran driver=async.
ASYNC_EP_FIELDS = ("n_plans", "mean_latency_ms", "mean_staleness_ms",
                   "missed_ticks", "tape_exhausted_ticks", "sim_seconds")
# Per-episode means of the fields; the two that are already per-episode means
# keep their name rather than becoming mean_mean_latency_ms.
ASYNC_COLS = tuple(f if f.startswith("mean_") else f"mean_{f}" for f in ASYNC_EP_FIELDS)


# ---------------------------------------------------------------------------
# Locating and loading runs
# ---------------------------------------------------------------------------

def newest_job_dir() -> Path:
    """The most recently modified results/csv_sweep_* directory."""
    cands = [p for p in RESULTS_DIR.glob("csv_sweep_*") if p.is_dir()]
    if not cands:
        raise FileNotFoundError(f"No csv_sweep_* directories found in {RESULTS_DIR}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def load_cells(root: Path) -> list[dict]:
    """Every cell record under `root`, sorted by row index.

    Accepts the array's --outdir of cell_*.json files, or combine_results.py's
    merged *_rich.json. The rich file has already dropped the per-episode
    records (and the reserved columns), so the end-reason / goal-error /
    wall-clock columns come out blank from it, and it has no row index either:
    `row` is then positional, so a missing cell shifts every later row down by
    one. Prefer the directory.
    """
    if root.is_file():
        with open(root) as f:
            recs = json.load(f)
        if not isinstance(recs, list):
            raise ValueError(f"{root} is not a combined *_rich.json record list")
        # rich rows carry no combo_index; the file is written in index order
        # (with gaps closed up — see the docstring).
        return [dict(r, combo_index=r.get("combo_index", i))
                for i, r in enumerate(recs) if isinstance(r, dict)]

    records = []
    for path in sorted(root.glob("cell_*.json")):
        try:
            with open(path) as f:
                records.append(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! could not read {path.name} ({e}); skipping it")
    if not records:
        raise FileNotFoundError(f"No cell_*.json files found in {root}")
    records.sort(key=lambda c: c.get("combo_index", 0))
    return records


def load_params_csv(path: Path) -> list[dict]:
    """The sweep's input rows as {column: raw string}, blank cells dropped —
    the same rows run_csv_cell.py's load_row indexes by SLURM_ARRAY_TASK_ID."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k: v.strip() for k, v in r.items() if k and v and v.strip()}
            for r in rows]


# ---------------------------------------------------------------------------
# One summary row per cell
# ---------------------------------------------------------------------------

def _num(v, fmt: str) -> str:
    """Format a number, or blank when the record holds None."""
    return "" if v is None else format(float(v), fmt)


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def axis_columns(rec: dict) -> dict:
    """The row's non-weight knobs, named as the input CSV named them.

    `axes` is every swept knob plus model/planner (already reserved columns) and
    the w_ overrides (reported via full_weights instead, so the CSV shows the
    weights the row actually ran under, not only the ones it changed).
    delta_range is folded back to the single `delta` cell the CSV spelled it as.
    """
    out = {}
    for k, v in (rec.get("axes") or {}).items():
        if k in RESERVED_COLS or k.startswith("w_"):
            continue
        if k == "delta_range" and isinstance(v, (list, tuple)) and len(v) == 2 \
                and v[0] == -v[1]:
            out["delta"] = v[1]
            continue
        out[k] = " ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v
    return out


def episode_columns(episodes: list[dict]) -> dict:
    """Everything only the per-episode records can tell: how each episode ended,
    the final goal error, wall clock, and the async telemetry."""
    if not episodes:
        return {}
    reasons = [e.get("end_reason", "unknown") for e in episodes]
    errors  = [e["error"] for e in episodes if e.get("error")]
    elapsed = [e.get("elapsed_seconds") for e in episodes]

    row = {
        "n_timeout":          reasons.count("timeout"),
        "n_failed":           reasons.count("failed"),
        "n_error":            reasons.count("error"),
        # The first error message only; they are usually all the same one.
        "error":              errors[0][:120] if errors else "",
        "mean_n_steps_taken": _num(_mean([e.get("n_steps_taken") for e in episodes]), ".1f"),
        "mean_final_cost":    _num(_mean([e.get("final_cost") for e in episodes]), ".4f"),
        # Effective planning horizon (blank on records written before the
        # field existed); only differs from the configured horizon under
        # --time_constrained.
        "mean_eff_horizon_steps": _num(
            _mean([e.get("mean_eff_horizon_steps") for e in episodes]), ".2f"),
        "mean_eff_horizon_s": _num(
            _mean([e.get("mean_eff_horizon_s") for e in episodes]), ".4f"),
        "mean_elapsed_s":     _num(_mean(elapsed), ".1f"),
        "total_elapsed_min":  _num(sum(v for v in elapsed if v is not None) / 60, ".1f"),
    }

    # final_goal_errs is keyed per task (grasp_reorient: pos/quat/vel); union the
    # keys so a task with different criteria gets its own columns.
    keys: list[str] = []
    for e in episodes:
        for k in (e.get("final_goal_errs") or {}):
            if k not in keys:
                keys.append(k)
    for k in keys:
        row[f"mean_final_{k}_err"] = _num(
            _mean([(e.get("final_goal_errs") or {}).get(k) for e in episodes]), ".4f")

    for f, col in zip(ASYNC_EP_FIELDS, ASYNC_COLS):
        row[col] = _num(_mean([e.get(f) for e in episodes]), ".2f")
    return row


def summarize_cell(rec: dict) -> dict:
    """One CSV row: identity, reserved columns, axes, outcome, weights."""
    n_ep = rec.get("n_episodes")
    n_success = rec.get("n_success")
    if n_success is None and n_ep and rec.get("success_rate") is not None:
        n_success = round(rec["success_rate"] * n_ep)     # rich rows lack it

    row = {
        "row":    rec.get("combo_index", ""),
        "status": "ok",
        "label":  rec.get("label", ""),
    }
    for k in RESERVED_COLS:
        v = rec.get(k)
        row[k] = "" if v is None else v
    row.update(axis_columns(rec))
    row.update({
        "n_episodes":            "" if n_ep is None else n_ep,
        "n_success":             "" if n_success is None else n_success,
        "success_rate":          _num(rec.get("success_rate"), ".4f"),
        "mean_steps_to_success": _num(rec.get("mean_steps_to_success"), ".1f"),
        "mean_step_ms":          _num(rec.get("mean_step_ms"), ".3f"),
        "std_step_ms":           _num(rec.get("std_step_ms"), ".3f"),
        "mean_elapsed_s":        _num(rec.get("mean_elapsed_s"), ".1f"),
    })
    row.update(episode_columns(rec.get("episodes") or []))

    for k, v in (rec.get("full_weights") or rec.get("overrides") or {}).items():
        row[k if str(k).startswith("w_") else f"w_{k}"] = v
    return row


def missing_row(idx: int, params: dict | None) -> dict:
    """A placeholder for a sweep row with no cell file, its inputs copied from
    the params CSV when one was given (the column names already agree)."""
    row = {"row": idx, "status": "missing", "label": ""}
    if params:
        for k, v in params.items():
            if k in ("record_trajectory", "record_planner_dist", "planner_dist_every"):
                continue
            row[k] = v
    return row


def build_rows(cells: list[dict], params: list[dict] | None) -> tuple[list[str], list[dict]]:
    """(fieldnames, rows): every cell, plus a placeholder per missing index."""
    by_idx = {c.get("combo_index", i): c for i, c in enumerate(cells)}
    n_rows = len(params) if params is not None else max(by_idx) + 1
    if params is not None and max(by_idx) >= n_rows:
        print(f"  ! cell index {max(by_idx)} is beyond the {n_rows} rows of the "
              f"params CSV given — is it the file this sweep ran?")
        n_rows = max(by_idx) + 1

    rows, missing = [], []
    for i in range(n_rows):
        if i in by_idx:
            rows.append(summarize_cell(by_idx[i]))
        else:
            missing.append(i)
            rows.append(missing_row(i, params[i] if params and i < len(params) else None))
    if missing:
        print(f"  ! {len(missing)} row(s) have no cell file (timed out / crashed?): "
              f"{' '.join(map(str, missing))}")

    # Stable column order: identity, reserved, axes (first-seen), outcome,
    # goal errors, async telemetry (only if any row ran async), weights.
    seen: list[str] = []
    for r in rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    fixed = ["row", "status", "label", *RESERVED_COLS]
    axes    = [k for k in seen if k not in fixed and k not in OUTCOME_COLS
               and k not in ASYNC_COLS and not k.startswith("w_")
               and not k.startswith("mean_final_")]
    goal    = [k for k in seen if k.startswith("mean_final_") and k != "mean_final_cost"]
    weights = [k for k in seen if k.startswith("w_")]
    any_async = any(str(r.get("driver", "")) == "async" for r in rows)
    fieldnames = (fixed + axes + list(OUTCOME_COLS) + goal
                  + (list(ASYNC_COLS) if any_async else []) + weights)
    if not any_async:
        for r in rows:
            for k in ASYNC_COLS:
                r.pop(k, None)
    return fieldnames, rows


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(rows: list[dict], axes: list[str]) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    rates = [float(r["success_rate"]) for r in ok if r.get("success_rate") != ""]
    print(f"\n{'=' * 78}")
    print(f"  {len(ok)} of {len(rows)} row(s) finished"
          + (f"; mean success {sum(rates) / len(rates) * 100:.1f}%" if rates else ""))
    print(f"{'=' * 78}")
    show_axes = axes[:4]
    cols = ["row", "model", "planner", *show_axes, "succ%", "n_to", "n_fail",
            "steps", "step_ms", "min"]
    print("  " + "  ".join(f"{c:>10}" for c in cols))
    for r in rows:
        if r["status"] != "ok":
            print(f"  {str(r['row']):>10}  {'-- missing --':>10}")
            continue
        vals = [str(r["row"]), r.get("model", ""), r.get("planner", ""),
                *(str(r.get(a, "")) for a in show_axes),
                _num(float(r["success_rate"]) * 100, ".1f") if r["success_rate"] != "" else "",
                str(r.get("n_timeout", "")), str(r.get("n_failed", "")),
                r.get("mean_steps_to_success") or "—",
                r.get("mean_step_ms") or "—",
                r.get("total_elapsed_min", "")]
        print("  " + "  ".join(f"{v[-10:]:>10}" for v in vals))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize a run_csv_sweep.slurm job as one CSV row per sweep row.")
    parser.add_argument(
        "indir", nargs="?", type=Path,
        help="csv_sweep job directory (cell_*.json inside), or a combined "
             "*_rich.json. Defaults to the newest results/csv_sweep_*.")
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="The params CSV the sweep ran. Rows with no cell file are then "
             "still written, inputs filled from it, with status=missing.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path. Defaults to <job_dir>/csv_sweep_summary.csv.")
    parser.add_argument(
        "--rank", action="store_true",
        help="Sort by success rate desc, then mean_step_ms asc, instead of "
             "keeping the input CSV's row order.")
    args = parser.parse_args()

    root = args.indir or newest_job_dir()
    if args.indir is None:
        print(f"Newest csv_sweep directory: {root}")
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    cells  = load_cells(root)
    params = load_params_csv(args.csv) if args.csv else None
    print(f"Loaded {len(cells)} cell(s) from {root}"
          + (f", {len(params)} input row(s) from {args.csv}" if params else ""))

    fieldnames, rows = build_rows(cells, params)
    if args.rank:
        rows.sort(key=lambda r: (r["status"] != "ok",
                                 -float(r.get("success_rate") or 0),
                                 float(r.get("mean_step_ms") or float("inf"))))

    if args.output is not None:
        out = args.output
    elif root.is_dir():
        out = root / "csv_sweep_summary.csv"
    else:
        out = root.parent / f"{root.stem.replace('_rich', '')}_summary.csv"

    fixed = {"row", "status", "label", *RESERVED_COLS, *OUTCOME_COLS, *ASYNC_COLS}
    axes = [k for k in fieldnames if k not in fixed and not k.startswith("w_")
            and not k.startswith("mean_final_")]
    print_report(rows, axes)
    write_csv(fieldnames, rows, out)


if __name__ == "__main__":
    main()
