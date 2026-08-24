"""plot_results.py

Plot aggregated experiment results from a JSON file produced by
run_experiment.py / evaluation.metrics.save_results.

Usage:
    python tests/plot_results.py results/experiment_20260604_112018.json
    python tests/plot_results.py          # auto-picks latest file in results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"

# (display_title, json_field, error_field_or_None, y_axis_label, is_fraction_0_1)
METRICS = [
    ("Success Rate",              "success_rate",           "success_rate_se",   "Rate",    True),
    ("Mean Steps to Success",     "mean_steps_to_success",  None,                "Steps",   False),
    ("Mean Final Cost",           "mean_final_cost",        "std_final_cost",    "Cost",    False),
    ("Speedup vs Baseline",       "speedup_vs_baseline",    None,                "×",       False),
    ("Approx. Error vs Baseline", "approx_err_vs_baseline", None,                "L2 err",  False),
    ("Mean Episode Time (s)",     "mean_elapsed",           None,                "seconds", False),
]

# Consistent per-model colours — fall back to the prop_cycle for unknown labels
_MODEL_COLORS: dict[str, str] = {
    "M1_stiff_pyramidal": "#4C72B0",
    "M2_mjwarp_soft":     "#DD8452",
    "M3_comfree":         "#55A868",
    "M4_xpbd":            "#C44E52",
}
_FALLBACK_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _color(label: str, idx: int) -> str:
    return _MODEL_COLORS.get(label, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _latest_results_file() -> Path:
    files = sorted(RESULTS_DIR.glob("experiment_*.json"))
    if not files:
        raise FileNotFoundError(f"No experiment_*.json files found in {RESULTS_DIR}")
    return files[-1]


def _bar_label(field: str, value: float, is_pct: bool, missing: bool) -> str:
    if missing:
        return "N/A"
    if is_pct:
        return f"{value:.0%}"
    if field in ("mean_steps_to_success",):
        return f"{value:.0f}"
    if field == "speedup_vs_baseline":
        return f"{value:.2f}×"
    return f"{value:.3g}"


def plot_task(task_name: str, entries: list[dict], save_dir: Path) -> None:
    """Create one PNG per task with six metric subplots."""
    models = [e["model_label"] for e in entries]
    x = np.arange(len(models))
    width = 0.55

    ncols = 3
    nrows = (len(METRICS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle(
        f"{task_name}  ({entries[0]['n_episodes']} episodes / model)",
        fontsize=13, fontweight="bold",
    )

    for ax_idx, (ax, (title, field, err_field, ylabel, is_pct)) in enumerate(
        zip(axes.flat, METRICS)
    ):
        values, errors, missing = [], [], []
        for e in entries:
            raw = e.get(field)
            missing.append(raw is None)
            values.append(float(raw) if raw is not None else 0.0)
            if err_field:
                ev = e.get(err_field)
                errors.append(float(ev) if ev is not None else 0.0)
            else:
                errors.append(0.0)

        colors = [_color(lbl, i) for i, lbl in enumerate(models)]
        bars = ax.bar(
            x, values, width,
            yerr=errors if any(e > 0 for e in errors) else None,
            color=colors,
            capsize=4,
            error_kw={"linewidth": 1.2, "ecolor": "black"},
            zorder=3,
        )

        # Value labels above each bar
        y_top = max(values) if values else 1.0
        label_pad = y_top * 0.03 if y_top > 0 else 0.02
        for bar, v, miss in zip(bars, values, missing):
            label = _bar_label(field, v, is_pct, miss)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_pad,
                label,
                ha="center", va="bottom", fontsize=7.5, color="#333333",
            )

        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, color="#dddddd", zorder=0)

        if is_pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            ax.set_ylim(0, 1.15)
        else:
            ax.set_ylabel(ylabel, fontsize=8)

    # Hide any unused subplot slots
    for ax in list(axes.flat)[len(METRICS):]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = save_dir / f"results_{task_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results bar charts")
    parser.add_argument(
        "results_file", nargs="?", type=Path,
        help="Path to experiment JSON. Defaults to latest file in results/",
    )
    args = parser.parse_args()

    path = args.results_file or _latest_results_file()
    print(f"Loading: {path}")

    results = load_results(path)
    save_dir = path.parent

    # Group by task name
    tasks: dict[str, list[dict]] = {}
    for entry in results:
        tasks.setdefault(entry["task_name"], []).append(entry)

    for task_name, entries in tasks.items():
        print(f"\nTask: {task_name}  |  {len(entries)} rows")
        plot_task(task_name, entries, save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
