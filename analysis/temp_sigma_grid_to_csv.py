"""temp_sigma_grid_to_csv.py

Summarize the output of experiments/hpc/temp_sigma_grid.slurm —
contact_study/drivers/run_temp_sigma_grid.py — as ONE ROW PER CELL.

A "cell" is what the SLURM array hands one GPU: one (object, contact model,
n_iterations, n_samples). Inside it the driver walks the whole
temperature x noise_sigma grid, writing one cell_<id>.json per GRID POINT plus a
grid_summary_<array index>.json for the cell. So the files are per grid point,
while the question this CSV answers is per cell: *which temperature and
noise_sigma should this object/model run at?*

Each row therefore carries

    job, array_index, task, object, geometry, model, planner, eval_sim
    n_samples, n_iterations, convergence_tol, max_iterations   — the cell axes
    best_temperature, best_noise_sigma                         — the answer
    best_success_rate, best_n_success, best_mean_steps_to_success, ...
    has_tie, n_best_ties, tied_points, tie_broken_by           — see below
    n_points, temperatures, noise_sigmas, grid_status          — grid coverage
    n_episodes, seed, settle, time_horizon, step_time, ...     — episode inputs
    w_*                                                        — the fixed cost
                                                                 weights the whole
                                                                 grid ran under

Ties
----
Points are ranked exactly as the driver ranks them: success rate descending,
mean steps-to-success ascending. With 10 episodes a point's success rate takes
one of 11 values, so several temperatures sharing the top rate is the norm, not
the exception, and reporting only the steps-tiebreak winner would hide it.

    has_tie        1 when more than one grid point reaches the best success rate
    n_best_ties    how many points reach it (1 when there is no tie)
    tied_points    all of them, best first: "T=25 sigma=0.1 | T=50 sigma=0.1"
    tie_broken_by  mean_steps_to_success  — the reported point genuinely won the
                                            tiebreak
                   unbroken               — the tied points are indistinguishable
                                            on steps too (e.g. all-zero success),
                                            so best_temperature is arbitrary
                   (blank)                — no tie

Usage:
    python analysis/temp_sigma_grid_to_csv.py                      # newest job dir
    python analysis/temp_sigma_grid_to_csv.py results/temp_sigma_grid_1234567
    python analysis/temp_sigma_grid_to_csv.py results/combined_temp_sigma_grid_rich.json
    python analysis/temp_sigma_grid_to_csv.py --all --top_n 0      # every job dir, no tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from param_search_to_csv import write_csv

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Success rates are k/n_episodes, so exact equality would work; a tolerance
# keeps the comparison honest against JSON round-tripping at the driver's
# recording precision.
RATE_TOL = 1e-9


# ---------------------------------------------------------------------------
# Locating and loading runs
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def newest_job_dir() -> Path:
    """The most recently modified results/temp_sigma_grid_* directory."""
    cands = [p for p in RESULTS_DIR.glob("temp_sigma_grid_*") if p.is_dir()]
    if not cands:
        raise FileNotFoundError(
            f"No temp_sigma_grid_* directories found in {RESULTS_DIR}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def load_points(root: Path) -> list[dict]:
    """Every grid-point record under `root`.

    Accepts both layouts the sweep produces: the array's shared --outdir of
    cell_*.json files, and combine_results.py's merged *_rich.json (the same
    records in one list), so a job that has already been combined needs no
    special handling here.
    """
    if root.is_file():
        recs = _read_json(root)
        if not isinstance(recs, list):
            raise ValueError(f"{root} is not a combined *_rich.json record list")
        return [r for r in recs if isinstance(r, dict)]

    records = []
    for path in sorted(root.glob("cell_*.json")):
        rec = _read_json(path)
        if rec is None:
            print(f"  ! could not read {path.name}; skipping it")
        else:
            records.append(rec)
    return records


def load_cell_summaries(root: Path) -> dict[int, dict]:
    """array_index -> grid_summary_<index>.json, for the cells that wrote one.

    The summary is the only place the REQUESTED grid (and the cell's wall clock)
    is recorded, so it is what tells a half-finished cell from a complete one.
    A cell killed before it finished has none; every column it feeds is then
    left blank rather than guessed.
    """
    if root.is_file():
        return {}
    out: dict[int, dict] = {}
    for path in sorted(root.glob("grid_summary_*.json")):
        summary = _read_json(path)
        if summary is None:
            continue
        idx = summary.get("array_index")
        if idx is not None:
            out[int(idx)] = summary
    return out


# ---------------------------------------------------------------------------
# Grouping grid points into cells
# ---------------------------------------------------------------------------

def object_of(rec: dict) -> str:
    """The object a record ran on.

    `object` is written at the top level and inside `axes`; geometry is
    "<object>_<hand_acc>_<obj_acc>", so its first field is the last resort.
    """
    obj = rec.get("object") or (rec.get("axes") or {}).get("object")
    if obj:
        return str(obj)
    return str(rec.get("geometry", "")).split("_")[0]


def cell_key(rec: dict) -> tuple:
    """What makes two grid points members of the same cell.

    The SLURM array's four axes (object and model, plus the iteration and sample
    counts that live under `mppi`), qualified by task/geometry/planner so two
    submissions merged into one directory cannot collapse into one row. NOT
    array_index: a resumed submission re-runs under a new array job id, and the
    same cell should stay one row.
    """
    m = rec.get("mppi") or {}
    return (
        rec.get("task", ""),
        object_of(rec),
        rec.get("geometry", ""),
        rec.get("model", ""),
        rec.get("planner", ""),
        m.get("n_samples"),
        m.get("n_iterations"),
        m.get("convergence_tol"),
    )


def rank_key(rec: dict):
    """The driver's own ranking: success rate desc, mean steps-to-success asc.

    A point with no successes has mean_steps_to_success=None and must sort LAST
    within its success-rate group, which a bare None would not do.
    """
    steps = rec.get("mean_steps_to_success")
    return (-(rec.get("success_rate") or 0.0),
            float("inf") if steps is None else float(steps))


def group_cells(records: list[dict]) -> dict[tuple, list[dict]]:
    """Grid points grouped by cell, each cell's points ranked best first.

    A point evaluated twice (a resume that re-ran it) keeps its LAST record:
    same cell, same temperature/sigma, so the newer evaluation wins.
    """
    cells: dict[tuple, dict[tuple, dict]] = {}
    for rec in records:
        m = rec.get("mppi") or {}
        point = (m.get("temperature"), m.get("noise_sigma"))
        cells.setdefault(cell_key(rec), {})[point] = rec
    return {k: sorted(v.values(), key=rank_key) for k, v in cells.items()}


# ---------------------------------------------------------------------------
# One row per cell
# ---------------------------------------------------------------------------

def _num(v, fmt: str) -> str:
    """Format a number, or blank when the driver recorded None."""
    return "" if v is None else format(float(v), fmt)


def _point_label(rec: dict) -> str:
    m = rec.get("mppi") or {}
    return f"T={float(m.get('temperature', 0)):g} sigma={float(m.get('noise_sigma', 0)):g}"


def tie_columns(points: list[dict]) -> dict:
    """The tie among the best-scoring grid points.

    `points` is already ranked, so points[0] is the reported winner and the tie
    is the run of points sharing its success rate.
    """
    best_rate = points[0].get("success_rate") or 0.0
    tied = [p for p in points
            if abs((p.get("success_rate") or 0.0) - best_rate) <= RATE_TOL]

    if len(tied) > 1:
        # Whether the steps tiebreak actually separated the winner from the rest:
        # if it did not, best_temperature/best_noise_sigma are one arbitrary pick
        # out of `tied` and should not be read as the answer.
        first, second = rank_key(tied[0])[1], rank_key(tied[1])[1]
        broken = "mean_steps_to_success" if first != second else "unbroken"
    else:
        broken = ""

    return {
        "has_tie":       1 if len(tied) > 1 else 0,
        "n_best_ties":   len(tied),
        "tied_points":   " | ".join(_point_label(p) for p in tied),
        "tie_broken_by": broken,
    }


def grid_columns(points: list[dict], summary: dict | None,
                 fallback: dict | None = None) -> dict:
    """Coverage of the temperature x noise_sigma grid inside this cell.

    The temperatures/noise_sigmas columns are the values the cell was ASKED for
    when a summary says so, and the values actually evaluated otherwise — so a
    partial cell still names its axes, and `grid_status` says it is partial.

    A cell killed mid-grid never wrote a summary, which is exactly the cell whose
    completeness matters most. TEMPERATURES/NOISE_SIGMAS are set once for the
    whole submission in temp_sigma_grid.slurm, so `fallback` — any sibling cell's
    summary from the same job — names the grid it was meant to cover.
    """
    temps  = sorted({(p.get("mppi") or {}).get("temperature") for p in points} - {None})
    sigmas = sorted({(p.get("mppi") or {}).get("noise_sigma") for p in points} - {None})

    src = summary or fallback or {}
    want_t = src.get("temperatures") or temps
    want_s = src.get("noise_sigmas")  or sigmas
    expected = src.get("n_points")
    if expected is None:
        expected = len(want_t) * len(want_s)

    status = ("complete" if len(points) >= expected
              else f"partial {len(points)}/{expected}")
    elapsed = (summary or {}).get("elapsed_seconds")

    return {
        "n_points":         len(points),
        "n_temperatures":   len(want_t),
        "n_noise_sigmas":   len(want_s),
        "temperatures":     " ".join(f"{float(t):g}" for t in want_t),
        "noise_sigmas":     " ".join(f"{float(s):g}" for s in want_s),
        "grid_status":      status,
        "elapsed_hours":    _num(elapsed / 3600 if elapsed else None, ".2f"),
    }


def summarize_cell(points: list[dict], summary: dict | None, job: str,
                   fallback: dict | None = None) -> dict:
    """One CSV row: identity, cell axes, the winning point, ties, inputs."""
    best = points[0]
    m    = best.get("mppi") or {}

    rates = [p.get("success_rate") or 0.0 for p in points]
    idxs  = {p.get("array_index") for p in points} - {None}

    row = {
        "job":         job,
        "array_index": " ".join(str(i) for i in sorted(idxs)),
        "task":        best.get("task", ""),
        "object":      object_of(best),
        "geometry":    best.get("geometry", ""),
        "model":       best.get("model", ""),
        "planner":     best.get("planner", ""),
        "eval_sim":    best.get("eval_sim", ""),

        # The cell axes: what the SLURM array varied between GPUs.
        "n_samples":       m.get("n_samples", ""),
        "n_iterations":    m.get("n_iterations") if m.get("n_iterations") is not None else "",
        "convergence_tol": m.get("convergence_tol") if m.get("convergence_tol") is not None else "",
        "max_iterations":  m.get("max_iterations") if m.get("max_iterations") is not None else "",

        # The answer this whole sweep exists to produce.
        "best_temperature": _num(m.get("temperature"), "g"),
        "best_noise_sigma": _num(m.get("noise_sigma"), "g"),

        "best_success_rate":          _num(best.get("success_rate"), ".4f"),
        "best_n_success":             best.get("n_success", ""),
        "best_mean_steps_to_success": _num(best.get("mean_steps_to_success"), ".1f"),
        "best_mean_step_ms":          _num(best.get("mean_step_ms"), ".3f"),
        "best_mean_elapsed_s":        _num(best.get("mean_elapsed_s"), ".1f"),
    }

    row.update(tie_columns(points))
    row.update(grid_columns(points, summary, fallback))

    # How the cell did overall, not just at its best point: a cell whose every
    # point scores the same is telling you the knob does not bite here.
    row.update({
        "mean_success_rate":     f"{sum(rates) / len(rates):.4f}",
        "worst_success_rate":    f"{min(rates):.4f}",
        "n_points_any_success":  sum(1 for r in rates if r > 0.0),
        "n_points_full_success": sum(1 for r in rates if r >= 1.0),

        "n_episodes":        best.get("n_episodes", ""),
        "seed":              best.get("seed", ""),
        "settle":            best.get("settle", ""),
        "time_horizon":      m.get("time_horizon", ""),
        "step_time":         m.get("step_time", ""),
        "step_horizon":      m.get("step_horizon", ""),
        "step_substeps":     m.get("step_substeps", ""),
        "rollout_dt":        m.get("rollout_dt", ""),
        "delta":             m.get("delta") if m.get("delta") is not None else "",
        "resample_interval": m.get("resample_interval") if m.get("resample_interval") is not None else "",
    })

    # The cost weights are FIXED for a whole grid (weights are param_search's and
    # bayes_opt's axis, not this sweep's), so they belong on the row: without
    # them a winning temperature cannot be reproduced.
    for k, v in (best.get("full_weights") or best.get("overrides") or {}).items():
        row[f"w_{k}" if not str(k).startswith("w_") else str(k)] = v

    return row


# The stable column order; the w_* columns are appended in first-seen order.
BASE_COLS = list(summarize_cell(
    [{"mppi": {}, "success_rate": 0.0}], None, "").keys())


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_cell_report(row: dict, points: list[dict], top_n: int) -> None:
    iters = row["convergence_tol"] and f"converge<{row['convergence_tol']}" \
        or f"n_iterations={row['n_iterations']}"
    print(f"\n{'-' * 78}")
    print(f"  {row['object']} / {row['model']}   {iters}  n_samples={row['n_samples']}")
    print(f"{'-' * 78}")
    print(f"  {row['n_points']} grid point(s), {row['grid_status']}, "
          f"{row['n_episodes']} episodes each, seed={row['seed']}, "
          f"{row['elapsed_hours'] or '?'} h")
    print(f"  BEST  T={row['best_temperature']}  sigma={row['best_noise_sigma']}  "
          f"success={float(row['best_success_rate'] or 0) * 100:.1f}%  "
          f"mean_steps={row['best_mean_steps_to_success'] or '—'}")
    if row["has_tie"]:
        print(f"  ! {row['n_best_ties']} points tie at that success rate "
              f"({row['tie_broken_by']}): {row['tied_points']}")

    if top_n <= 0:
        return
    cols = ["temperature", "noise_sigma", "succ%", "mean_steps", "step_ms"]
    print("  " + "  ".join(f"{c:>12}" for c in cols))
    for p in points[:top_n]:
        m = p.get("mppi") or {}
        vals = [
            f"{float(m.get('temperature', 0)):g}",
            f"{float(m.get('noise_sigma', 0)):g}",
            _num((p.get("success_rate") or 0.0) * 100, ".1f"),
            _num(p.get("mean_steps_to_success"), ".1f") or "—",
            _num(p.get("mean_step_ms"), ".3f") or "—",
        ]
        print("  " + "  ".join(f"{v:>12}" for v in vals))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize a temperature x noise_sigma sweep as one CSV row "
                    "per (object, model, n_iterations, n_samples) cell.")
    parser.add_argument(
        "indirs", nargs="*", type=Path,
        help="temp_sigma_grid job directories (cell_*.json inside), or a combined "
             "*_rich.json. Defaults to the newest results/temp_sigma_grid_*.")
    parser.add_argument(
        "--all", action="store_true",
        help="Summarize every results/temp_sigma_grid_* directory, not only the newest.")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path. Defaults to <job_dir>/temp_sigma_grid_summary.csv.")
    parser.add_argument(
        "--top_n", type=int, default=5,
        help="Grid points to show per cell in the console tables; 0 suppresses "
             "them (default 5).")
    args = parser.parse_args()

    if args.indirs:
        roots = args.indirs
    elif args.all:
        roots = sorted(p for p in RESULTS_DIR.glob("temp_sigma_grid_*") if p.is_dir())
    else:
        roots = [newest_job_dir()]
        print(f"Newest temp_sigma_grid directory: {roots[0]}")

    rows:      list[dict] = []
    cells_by_row: list[list[dict]] = []
    weight_cols:  list[str] = []          # union of w_* columns, first-seen order

    for root in roots:
        if not root.exists():
            print(f"  ! {root} does not exist; skipping")
            continue

        records   = load_points(root)
        summaries = load_cell_summaries(root)
        if not records:
            print(f"  ! no grid points found in {root}; skipping")
            continue

        cells = group_cells(records)
        print(f"\n{'=' * 78}")
        print(f"  {root}  ({len(records)} grid point(s), {len(cells)} cell(s))")
        print(f"{'=' * 78}")

        job = root.stem if root.is_file() else root.name
        # Any cell's summary: they all record the submission's single grid.
        fallback = next(iter(summaries.values()), None)
        for key in sorted(cells, key=lambda k: tuple(str(v) for v in k)):
            points = cells[key]
            idx = points[0].get("array_index")
            row = summarize_cell(points, summaries.get(idx), job, fallback)
            for col in row:
                if col.startswith("w_") and col not in weight_cols:
                    weight_cols.append(col)
            rows.append(row)
            cells_by_row.append(points)
            print_cell_report(row, points, args.top_n)

    if not rows:
        raise ValueError("No temp_sigma_grid cells were found.")

    fieldnames = BASE_COLS + weight_cols
    for row in rows:                       # cells with fewer weights leave blanks
        for col in fieldnames:
            row.setdefault(col, "")

    # Grouped for reading, not ranked: neighbouring rows should be the same
    # object/model at different iteration and sample counts.
    rows.sort(key=lambda r: (str(r["object"]), str(r["model"]),
                             str(r["n_iterations"]), float(r["n_samples"] or 0)))

    if args.output is not None:
        out = args.output
    elif len(roots) == 1 and roots[0].is_dir():
        out = roots[0] / "temp_sigma_grid_summary.csv"
    else:
        out = RESULTS_DIR / "temp_sigma_grid_summary.csv"

    print(f"\n{'=' * 78}")
    print(f"  {len(rows)} cell(s)")
    print(f"{'=' * 78}")
    cols = ["object", "model", "n_iterations", "n_samples", "best_temperature",
            "best_noise_sigma", "best_success_rate", "n_best_ties", "grid_status"]
    print("  " + "  ".join(f"{c:>17}" for c in cols))
    for row in rows:
        print("  " + "  ".join(f"{str(row[c])[-17:]:>17}" for c in cols))
    write_csv(fieldnames, rows, out)


if __name__ == "__main__":
    main()
