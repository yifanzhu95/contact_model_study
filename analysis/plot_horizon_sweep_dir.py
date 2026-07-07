"""plot_horizon_sweep_dir.py

Directory plotter for the MPPI horizon-length sweep — reads a whole directory
of per-cell JSONs written by experiments/hpc/horizon_eval.slurm
(results/horizon_eval_<id>/<model>_h<horizon>.json) and merges every cell's
records for the same (model, horizon) label before plotting. Same merge rule
and two-panel bar-chart style as plot_n_samples_sweep_dir.py /
plot_cntrl_freq_sweep_dir.py, with no single-file counterpart since the
horizon sweep is only ever produced as a directory of per-cell JSONs.

Merge rule: each label's stats are pooled across files weighted by that file's
n_episodes. The success-rate whisker (±1 SE) is recomputed from the pooled
success rate and total episode count (binomial SE = sqrt(p(1-p)/N)); step-time
mean/SD are episode-weighted averages.

Top panel    — success rate (%) with ±1 SE whiskers.
Bottom panel — mean MPPI step time (ms) with ±1 SD whiskers.
Each cluster on the x-axis is one horizon length (6, 12, 24, 48); each bar
within a cluster is one contact model (M1-M4).

Usage:
    python analysis/plot_horizon_sweep_dir.py                        # latest horizon_eval_* dir
    python analysis/plot_horizon_sweep_dir.py results/horizon_eval_12345
"""

from __future__ import annotations

import argparse
import json
import math
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

_LABEL_RE = re.compile(r"^(M\d+)_h(\d+)$")


def _latest_dir() -> Path:
    dirs = sorted(d for d in RESULTS_DIR.glob("horizon_eval_*") if d.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"No horizon_eval_* directories found in {RESULTS_DIR}. "
            f"Pass a directory explicitly."
        )
    return dirs[-1]


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def merge_dir(directory: Path) -> tuple[list[dict], list[Path]]:
    """Pool every *.json in *directory* into one record per model_label.

    Returns records with the fields parse_records reads: model_label,
    task_name, condition, n_episodes, success_rate, success_rate_se,
    mean_step_ms, std_step_ms.
    """
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.json files found in {directory}")

    # label -> episode-weighted accumulators
    acc: dict[str, dict] = {}
    for path in files:
        for r in load(path):
            label = r.get("model_label")
            if label is None:
                continue
            n = int(r.get("n_episodes", 0) or 0)
            a = acc.setdefault(label, {
                "n": 0, "sr": 0.0, "ms": 0.0, "sd": 0.0,
                "task_name": r.get("task_name", "unknown"),
                "condition": r.get("condition", ""),
            })
            a["n"]  += n
            a["sr"] += float(r.get("success_rate", 0.0)) * n
            a["ms"] += float(r.get("mean_step_ms", 0.0)) * n
            a["sd"] += float(r.get("std_step_ms",  0.0)) * n

    merged: list[dict] = []
    for label, a in acc.items():
        n_tot = a["n"]
        w = n_tot if n_tot else 1               # guard against all-zero weights
        sr = a["sr"] / w
        se = math.sqrt(max(sr * (1.0 - sr), 0.0) / n_tot) if n_tot > 1 else 0.0
        merged.append({
            "model_label":     label,
            "task_name":       a["task_name"],
            "condition":       a["condition"],
            "n_episodes":      n_tot,
            "success_rate":    sr,
            "success_rate_se": se,
            "mean_step_ms":    a["ms"] / w,
            "std_step_ms":     a["sd"] / w,
        })

    return merged, files


def parse_records(records: list[dict]):
    """Return (models, horizon_values, data).

    data[model][horizon] = (success_rate, success_rate_se, mean_step_ms, std_step_ms)
    """
    data: dict[str, dict[int, tuple]] = {}
    for r in records:
        m = _LABEL_RE.match(r["model_label"])
        if m is None:
            continue
        model   = m.group(1)
        horizon = int(m.group(2))
        data.setdefault(model, {})[horizon] = (
            float(r["success_rate"]),
            float(r["success_rate_se"]),
            float(r.get("mean_step_ms", 0.0)),
            float(r.get("std_step_ms",  0.0)),
        )

    models          = sorted(data.keys())
    horizon_values  = sorted({h for mdict in data.values() for h in mdict})
    return models, horizon_values, data


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


def plot(models, horizon_values, data, title: str, out_path: Path):
    n_groups  = len(horizon_values)
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
    _clustered_bars(ax_succ, models, horizon_values, data,
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
    _clustered_bars(ax_time, models, horizon_values, data,
                    value_idx=2, error_idx=3,
                    group_centres=group_centres, offsets=offsets,
                    bar_w=bar_w, scale=1.0, add_legend=False)

    ax_time.set_ylabel("Mean step time  (ms)", fontsize=11)
    ax_time.set_ylim(bottom=0)
    ax_time.spines[["top", "right"]].set_visible(False)
    ax_time.grid(axis="y", linewidth=0.4, color="#cccccc", zorder=0)

    # x-axis labels only on the bottom panel (shared axis hides top labels)
    ax_time.set_xticks(group_centres)
    ax_time.set_xticklabels([str(h) for h in horizon_values], fontsize=10)
    ax_time.set_xlabel("Planning horizon  (steps)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a whole directory of horizon-sweep JSONs (merged)."
    )
    parser.add_argument(
        "results_dir", nargs="?", type=Path,
        help="Directory of sweep JSONs. Defaults to latest "
             "horizon_eval_* dir in results/",
    )
    args = parser.parse_args()

    directory = args.results_dir or _latest_dir()
    print(f"Loading directory: {directory}")

    records, files = merge_dir(directory)
    print(f"  Merged {len(files)} file(s)")

    models, horizon_values, data = parse_records(records)
    if not models:
        raise ValueError("No parseable records found (expected labels like M1_h24).")

    print(f"  Models   : {models}")
    print(f"  Horizons : {horizon_values}")

    task_name = records[0].get("task_name", "unknown") if records else "unknown"
    condition = records[0].get("condition", "")
    title     = f"Success rate vs. planning horizon — {task_name}"
    if condition:
        title += f"  (condition {condition})"

    out_path = directory / f"{directory.name}_plot.pdf"
    plot(models, horizon_values, data, title, out_path)


if __name__ == "__main__":
    main()
