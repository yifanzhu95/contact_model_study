"""plot_cntrl_freq_sweep.py

Two-panel clustered bar chart from a control-frequency sweep JSON produced by
experiments/run_cntrl_freq_eval.py.

Top panel    — success rate (%) with ±1 SE whiskers.
Bottom panel — mean MPPI step time (ms) with ±1 SD whiskers.

Each cluster on the x-axis is one control frequency, derived from the number
of physics substeps per planning step:

    control_frequency = 1 / (substeps × dt)

where dt defaults to 0.005 s (200 Hz simulation).  Pass --dt to override.

Bars within each cluster are one per contact model (M1–M4).

Usage:
    python analysis/plot_cntrl_freq_sweep.py                         # latest file
    python analysis/plot_cntrl_freq_sweep.py results/substeps_*.json # explicit path
    python analysis/plot_cntrl_freq_sweep.py --dt 0.002              # 500 Hz sim
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

_LABEL_RE = re.compile(r"^(M\d+)_sub(\d+)$")


def _latest_file() -> Path:
    files = sorted(RESULTS_DIR.glob("substeps_sweep_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No substeps_sweep_*.json files found in {RESULTS_DIR}"
        )
    return files[-1]


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _freq_label(freq: float) -> str:
    if freq == int(freq):
        return f"{int(freq)}"
    return f"{freq:.4g}"


def parse_records(records: list[dict], dt: float):
    """Return (models, freq_values, substep_map, data).

    data[model][freq] = (success_rate, success_rate_se, mean_step_ms, std_step_ms)
    substep_map[freq] = substeps  (for the secondary axis label)
    """
    data: dict[str, dict[float, tuple]] = {}
    substep_map: dict[float, int] = {}

    for r in records:
        m = _LABEL_RE.match(r["model_label"])
        if m is None:
            continue
        model    = m.group(1)
        substeps = int(m.group(2))
        freq     = 1.0 / (substeps * dt)
        data.setdefault(model, {})[freq] = (
            float(r["success_rate"]),
            float(r["success_rate_se"]),
            float(r.get("mean_step_ms", 0.0)),
            float(r.get("std_step_ms",  0.0)),
        )
        substep_map[freq] = substeps

    models      = sorted(data.keys())
    freq_values = sorted({f for mdict in data.values() for f in mdict})
    return models, freq_values, substep_map, data


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


def plot(models, freq_values, substep_map, data, title: str, dt: float, out_path: Path):
    n_groups  = len(freq_values)
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
    _clustered_bars(ax_succ, models, freq_values, data,
                    value_idx=0, error_idx=1,
                    group_centres=group_centres, offsets=offsets,
                    bar_w=bar_w, scale=100.0, add_legend=True)

    ax_succ.set_ylabel("Success rate  (%)", fontsize=11)
    ax_succ.set_ylim(0, 115)
    ax_succ.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_succ.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax_succ.spines[["top", "right"]].set_visible(False)
    ax_succ.grid(axis="y", linewidth=0.4, color="#cccccc", zorder=0)
    ax_succ.annotate(
        f"dt = {dt*1e3:.3g} ms",
        xy=(1.0, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=7.5, color="#888888",
    )
    legend = ax_succ.legend(
        title="Contact model", title_fontsize=9, fontsize=9,
        framealpha=0.9, loc="upper left",
    )
    legend.get_frame().set_linewidth(0.5)

    # ── Bottom panel: mean step time ─────────────────────────────────
    _clustered_bars(ax_time, models, freq_values, data,
                    value_idx=2, error_idx=3,
                    group_centres=group_centres, offsets=offsets,
                    bar_w=bar_w, scale=1.0, add_legend=False)

    ax_time.set_ylabel("Mean step time  (ms)", fontsize=11)
    ax_time.set_ylim(bottom=0)
    ax_time.spines[["top", "right"]].set_visible(False)
    ax_time.grid(axis="y", linewidth=0.4, color="#cccccc", zorder=0)

    # Primary x-axis: control frequency (Hz)
    ax_time.set_xticks(group_centres)
    ax_time.set_xticklabels([_freq_label(f) for f in freq_values], fontsize=10)

    # Secondary x-axis: substep count, sitting below the frequency labels
    ax2 = ax_time.secondary_xaxis("bottom")
    ax2.set_xticks(group_centres)
    ax2.set_xticklabels(
        [f"({substep_map[f]}×)" for f in freq_values],
        fontsize=7.5, color="#666666",
    )
    ax2.tick_params(axis="x", length=0, pad=14)
    ax2.spines["bottom"].set_visible(False)

    ax_time.set_xlabel(
        "Control frequency  (Hz)\n(substep multiplier in parentheses)",
        fontsize=10, labelpad=18,
    )

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot control-frequency sweep: success rate and step time bar charts."
    )
    parser.add_argument(
        "results_file", nargs="?", type=Path,
        help="Path to JSON. Defaults to latest substeps_sweep_*.json in results/",
    )
    parser.add_argument(
        "--dt", type=float, default=0.005,
        help="Simulation timestep in seconds used to convert substeps → Hz. "
             "Default: 0.005 s (200 Hz simulation).",
    )
    args = parser.parse_args()

    path = args.results_file or _latest_file()
    print(f"Loading: {path}")
    print(f"Using dt = {args.dt*1e3:.3g} ms  →  substeps × dt = control period")

    records = load(path)
    models, freq_values, substep_map, data = parse_records(records, args.dt)

    if not models:
        raise ValueError("No parseable records found (expected labels like M1_sub4).")

    print(f"  Models      : {models}")
    print(f"  Substeps    : {sorted(substep_map.values())}")
    print(f"  Ctrl freqs  : {[_freq_label(f) + ' Hz' for f in freq_values]}")

    task_name = records[0].get("task_name", "unknown") if records else "unknown"
    condition = records[0].get("condition", "")
    title     = f"Success rate vs. control frequency — {task_name}"
    if condition:
        title += f"  (condition {condition})"

    out_path = path.parent / (path.stem + "_plot.pdf")
    plot(models, freq_values, substep_map, data, title, args.dt, out_path)


if __name__ == "__main__":
    main()
