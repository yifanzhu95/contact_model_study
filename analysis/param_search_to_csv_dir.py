"""param_search_to_csv_dir.py

Same output as analysis/param_search_to_csv.py, but reads directly from a
param-search output directory (the `--outdir` of experiments/hpc/param_search.slurm)
instead of a combined *_rich.json file. That directory holds one
cell_<id>.json per SLURM array task, written by experiments/hpc/run_param_cell.py —
this script loads all of them and skips the combine_results.py step.

Output columns (identical to param_search_to_csv.py):
    <axis_key>, ...            — one column per search axis: the overridden
                                 weight values (w_quat, w_pos_x, w_quat_term,
                                 w_pos_term, ...) plus any swept non-weight knobs
                                 (temperature, noise_sigma, ...)
    success_rate_<M>           — per-model success rate (0–1)
    mean_success_rate          — average success rate across all evaluated models
    mean_step_ms_<M>           — per-model mean MPPI step latency
    mean_step_ms_avg           — average step latency across models
    mean_steps_to_success_<M>  — per-model mean steps to success (blank if none succeeded)
    n_episodes                 — episodes per cell (same for all cells)

Rows are sorted by descending mean_success_rate, then ascending mean_step_ms_avg.

Usage:
    python analysis/param_search_to_csv_dir.py results/param_search_12345
    python analysis/param_search_to_csv_dir.py results/param_search_12345 --output custom.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from param_search_to_csv import axes_of, build_csv_rows, write_csv


def load_cells(indir: Path) -> list[dict]:
    files = sorted(indir.glob("cell_*.json"))
    if not files:
        raise FileNotFoundError(f"No cell_*.json files found in {indir}")
    records = []
    for f in files:
        with open(f) as fh:
            records.append(json.load(fh))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Convert a param-search output directory (cell_*.json files) to a tidy CSV."
    )
    parser.add_argument(
        "indir", type=Path,
        help="Directory containing cell_*.json files, as produced by param_search.slurm.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path. Defaults to <indir>/<indir name>.csv",
    )
    args = parser.parse_args()

    indir = args.indir
    print(f"Loading cells from: {indir}")

    records = load_cells(indir)
    if not records:
        raise ValueError("No cell records found.")

    fieldnames, rows = build_csv_rows(records)

    models = sorted({r["model"] for r in records})
    axis_keys = [k for k in fieldnames if k in axes_of(records[0])]
    print(f"  Hyperparameter axes : {axis_keys}")
    print(f"  Models              : {models}")
    print(f"  Combinations        : {len(rows)}")

    out_path = args.output or indir / f"{indir.name}.csv"
    write_csv(fieldnames, rows, out_path)

    # Print a small preview of the top rows
    top_n = min(5, len(rows))
    print(f"\n  Top-{top_n} by mean success rate:")
    rate_col = "mean_success_rate"
    succ_cols  = [f"success_rate_{m}" for m in models]
    preview_cols = axis_keys + succ_cols + [rate_col]
    header = "  " + "  ".join(f"{c:>22}" for c in preview_cols)
    print(header)
    for r in rows[:top_n]:
        line = "  " + "  ".join(f"{str(r.get(c, '')):>22}" for c in preview_cols)
        print(line)


if __name__ == "__main__":
    main()
