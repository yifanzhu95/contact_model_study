"""plot_n_samples_sweep.py

Two-panel clustered bar chart from a num-rollouts sweep JSON produced by
experiments/run_num_rollout_eval.py.

Top panel    — success rate (%) with ±1 SE whiskers.
Bottom panel — mean MPPI step time (ms) with ±1 SD whiskers.

Each cluster on the x-axis is one n_samples value (8, 16, …, 1024).
Each bar within a cluster is one contact model (M1–M4).

Usage:
    python analysis/plot_n_samples_sweep.py                          # latest file
    python analysis/plot_n_samples_sweep.py results/n_samples_*.json # explicit path
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"

MODEL_COLORS = {
    "M1": "#4C72B0",
    "M2": "#DD8452",
    "M3": "#55A868",
    "M4": "#C44E52",
}
MODEL_LABELS = {
    "M1": "M1 (stiff pyramidal)",
    "M2": "M2 (soft pyramidal)",
    "M3": "M3 (ComFree)",
    "M4": "M4 (XPBD)",
}

_LABEL_RE = re.compile(r"^(M\d+)_n(\d+)$")


def _latest_file() -> Path:
    files = sorted(RESULTS_DIR.glob("n_samples_sweep_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No n_samples_sweep_*.json files found in {RESULTS_DIR}"
        )
    return files[-1]


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def parse_records(records: list[dict]):
    """Return (models, n_values, data).

    data[model][n] = (success_rate, success_rate_se, mean_step_ms, std_step_ms)
    """
    data: dict[str, dict[int, tuple]] = {}
    for r in records:
        m = _LABEL_RE.match(r["model_label"])
        if m is None:
            continue
        model = m.group(1)
        n     = int(m.group(2))
        data.setdefault(model, {})[n] = (
            float(r["success_rate"]),
            float(r["success_rate_se"]),
            float(r.get("mean_step_ms", 0.0)),
            float(r.get("std_step_ms",  0.0)),
        )

    models   = sorted(data.keys())
    n_values = sorted({n for mdict in data.values() for n in mdict})
    return models, n_values, data


def _clustered_bars(ax, models, x_keys, data,
                    value_idx, error_idx,
                    group_centres, offsets, bar_w,
                    scale=1.0, add_legend=False):
    """Draw one panel's worth of clustered bars + whiskers onto *ax*."""
    for j, model in enumerate(models):
        vals   = [data[model].get(k, (0,)*4)[value_idx] * scale for k in x_keys]
        errors = [data[model].get(k, (0,)*4)[error_idx] * scale for k in x_keys]
        xs     = group_centres + offsets[j]
        color  = MODEL_COLORS.get(model, f"C{j}")
        label  = MODEL_LABELS.get(model, model) if add_legend else "_nolegend_"
        ax.bar(xs, vals, width=bar_w, color=color, label=label, zorder=3)
        ax.errorbar(xs, vals, yerr=errors, fmt="none",
                    ecolor="black", elinewidth=1.2,
                    capsize=3.5, capthick=1.2, zorder=4)


def plot(models, n_values, data, title: str, out_path: Path):
    n_groups  = len(n_values)
    n_models  = len(models)
    bar_w     = 0.7 / n_models
    group_gap = 1.0

    fig, (ax_succ, ax_time) = plt.subplots(
        2, 1,
        figsize=(max(8, n_groups * 1.4), 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
    )

    group_centres = np.arange(n_groups) * group_gap
    offsets       = (np.arange(n_models) - (n_models - 1) / 2) * bar_w

    # ── Top panel: success rate ──────────────────────────────────────
    _clustered_bars(ax_succ, models, n_values, data,
                    value_idx=0, error_idx=1,
                    group_centres=group_centres, offsets=offsets,
                    bar_w=bar_w, scale=100.0, add_legend=True)

    ax_succ.set_ylabel("Success rate  (%)", fontsize=11)
    ax_succ.set_ylim(0, 115)
    ax_succ.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_succ.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax_succ.spines[["top", "right"]].set_visible(False)
    ax_succ.grid(axis="y", linewidth=0.4, color="#cccccc", zorder=0)
    legend = ax_succ.legend(
        title="Contact model", title_fontsize=9, fontsize=9,
        framealpha=0.9, loc="upper left",
    )
    legend.get_frame().set_linewidth(0.5)

    # ── Bottom panel: mean step time ─────────────────────────────────
    _clustered_bars(ax_time, models, n_values, data,
                    value_idx=2, error_idx=3,
                    group_centres=group_centres, offsets=offsets,
                    bar_w=bar_w, scale=1.0, add_legend=False)

    ax_time.set_ylabel("Mean step time  (ms)", fontsize=11)
    ax_time.set_ylim(bottom=0)
    ax_time.spines[["top", "right"]].set_visible(False)
    ax_time.grid(axis="y", linewidth=0.4, color="#cccccc", zorder=0)

    # x-axis labels only on the bottom panel (shared axis hides top labels)
    ax_time.set_xticks(group_centres)
    ax_time.set_xticklabels([str(n) for n in n_values], fontsize=10)
    ax_time.set_xlabel("Number of rollouts  (n_samples)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot n_samples sweep: success rate and step time bar charts."
    )
    parser.add_argument(
        "results_file", nargs="?", type=Path,
        help="Path to JSON. Defaults to latest n_samples_sweep_*.json in results/",
    )
    args = parser.parse_args()

    path    = args.results_file or _latest_file()
    print(f"Loading: {path}")

    records = load(path)
    models, n_values, data = parse_records(records)

    if not models:
        raise ValueError("No parseable records found (expected labels like M1_n64).")

    print(f"  Models  : {models}")
    print(f"  n values: {n_values}")

    task_name = records[0].get("task_name", "unknown") if records else "unknown"
    condition = records[0].get("condition", "")
    title     = f"Success rate vs. rollouts — {task_name}"
    if condition:
        title += f"  (condition {condition})"

    out_path = path.parent / (path.stem + "_plot.pdf")
    plot(models, n_values, data, title, out_path)


if __name__ == "__main__":
    main()
