"""param_search_to_csv.py

Reprocess a param-search rich JSON (produced by experiments/run_param_search.py)
into a tidy CSV.  One row per hyperparameter combination, with a column for
each model's success rate and an average across all models.

Output columns:
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
    python analysis/param_search_to_csv.py                            # latest file
    python analysis/param_search_to_csv.py results/param_search_*.json # explicit path
    python analysis/param_search_to_csv.py --output custom.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _latest_rich_file() -> Path:
    files = sorted(RESULTS_DIR.glob("param_search_*_rich.json"))
    if not files:
        raise FileNotFoundError(
            f"No param_search_*_rich.json files found in {RESULTS_DIR}"
        )
    return files[-1]


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def axes_of(record: dict) -> dict:
    """The search axes of one cell: cost-weight overrides plus any swept
    non-weight knobs (temperature, noise_sigma, ...).

    Cells written before `axes` existed only carry the weight overrides, so fall
    back to those."""
    return record.get("axes") or record.get("overrides", {})


def build_csv_rows(records: list[dict]) -> tuple[list[str], list[dict]]:
    """Group records by search-axis combination and return (fieldnames, rows)."""
    # Group by the sorted axis items so command-line ordering doesn't matter
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    override_keys_order: list[str] = []

    for r in records:
        overrides = axes_of(r)
        if not override_keys_order:
            override_keys_order = list(overrides.keys())
        key = tuple(sorted(overrides.items()))
        groups[key][r["model"]] = r

    # Discover all model keys (sorted for stable column order)
    all_models: list[str] = sorted(
        {r["model"] for r in records}
    )

    # Build one output row per hyperparameter combo
    rows = []
    for key, model_map in groups.items():
        overrides = dict(key)
        row: dict = {}

        # Weight columns
        for k in override_keys_order:
            row[k] = overrides.get(k, "")

        # Per-model success rates and step latencies
        success_rates = []
        step_ms_vals = []
        for m in all_models:
            if m in model_map:
                r = model_map[m]
                row[f"success_rate_{m}"] = f"{r['success_rate']:.4f}"
                row[f"mean_step_ms_{m}"]  = f"{r['mean_step_ms']:.3f}"
                steps = r.get("mean_steps_to_success")
                row[f"mean_steps_to_success_{m}"] = (
                    f"{steps:.1f}" if steps is not None else ""
                )
                success_rates.append(r["success_rate"])
                step_ms_vals.append(r["mean_step_ms"])
            else:
                row[f"success_rate_{m}"]           = ""
                row[f"mean_step_ms_{m}"]            = ""
                row[f"mean_steps_to_success_{m}"]   = ""

        row["mean_success_rate"] = (
            f"{sum(success_rates) / len(success_rates):.4f}" if success_rates else ""
        )
        row["mean_step_ms_avg"] = (
            f"{sum(step_ms_vals) / len(step_ms_vals):.3f}" if step_ms_vals else ""
        )

        # n_episodes — should be identical for all cells, take any
        n_ep = next(iter(model_map.values())).get("n_episodes", "")
        row["n_episodes"] = n_ep

        # Sort key stored for later (not written to CSV)
        row["_sort_key"] = (
            -float(row["mean_success_rate"] or 0),
            float(row["mean_step_ms_avg"] or 0),
        )
        rows.append(row)

    rows.sort(key=lambda r: r["_sort_key"])
    for r in rows:
        del r["_sort_key"]

    # Fieldnames in logical order
    fieldnames = (
        override_keys_order
        + [f"success_rate_{m}" for m in all_models]
        + ["mean_success_rate"]
        + [f"mean_step_ms_{m}" for m in all_models]
        + ["mean_step_ms_avg"]
        + [f"mean_steps_to_success_{m}" for m in all_models]
        + ["n_episodes"]
    )

    return fieldnames, rows


def write_csv(fieldnames: list[str], rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path}  ({len(rows)} rows × {len(fieldnames)} columns)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a param-search rich JSON to a tidy CSV."
    )
    parser.add_argument(
        "results_file", nargs="?", type=Path,
        help="Path to *_rich.json. Defaults to latest param_search_*_rich.json in results/",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path. Defaults to <stem>.csv next to the input file.",
    )
    args = parser.parse_args()

    path = args.results_file or _latest_rich_file()
    print(f"Loading: {path}")

    records = load(path)
    if not records:
        raise ValueError("JSON file is empty.")

    fieldnames, rows = build_csv_rows(records)

    models = sorted({r["model"] for r in records})
    axis_keys = [k for k in fieldnames if k in axes_of(records[0])]
    print(f"  Hyperparameter axes : {axis_keys}")
    print(f"  Models              : {models}")
    print(f"  Combinations        : {len(rows)}")

    out_path = args.output or path.parent / (path.stem.replace("_rich", "") + ".csv")
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
