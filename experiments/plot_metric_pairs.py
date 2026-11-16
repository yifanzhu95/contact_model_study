#!/usr/bin/env python3
"""Every metric against every other metric, one figure per fixed temperature.

Companion to ``plot_final_results.py``, which pools all temperatures into each
scatter. Here the cells (and episodes) are split by ``temperature`` first, so
nothing is compared across temperatures (the regret in particular is scored at
each cell's own temperature and is only comparable within one).

For each temperature it draws two lower-triangle scatter matrices:

* ``pairs_cells_T_<temp>.png`` from the per-cell table: one point per cell,
  coloured by rollout model, marker by task. Success rate is the cell's rate,
  regret the cell mean, every other metric the cell median (the same summaries
  ``plot_final_results.py`` uses).
* ``pairs_episodes_T_<temp>.png`` from the per-episode table: one point per
  episode (its per-episode mean of each metric), coloured by success.

Each panel is annotated with Spearman's rank correlation (rho) over the points
it shows. KL axes are log; the rest are linear.

Usage:
    python experiments/plot_metric_pairs.py \
        --episodes results/final_episodes.csv --cells results/final_results.csv \
        --out results/plots/metric_pairs
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.lines import Line2D

from plot_final_results import FAILURE_COLOR, SUCCESS_COLOR, readTable

# Fixed per model, as in plot_sr_by_task_model.py, so a model keeps its colour across figures.
MODEL_COLORS = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#eda100",
                "M5": "#6250d6", "M6": "#e87ba4"}
TASK_MARKERS = {"cube_reorient": "s", "ball_reorient": "o", "duck_reorient": "^"}

# (key, short axis label, scale, log axis); key is the metric's base column name.
METRICS = [
    ("success", "Success rate", 1.0, False),
    ("regret", "Regret", 1.0, False),
    ("kl_used_opt", "KL(used ‖ opt)", 1.0, True),
    ("kl_opt_used", "KL(opt ‖ used)", 1.0, True),
    ("fsim_obj_pos_err", "Fwd pos err (mm)", 1e3, False),
    ("fsim_obj_rot_err", "Fwd rot err (rad)", 1.0, False),
    ("fsim_qvel_err", "Fwd qvel err", 1.0, False),
]


def cellValues(cells: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each metric's per-cell summary: success rate, regret mean, otherwise the median."""
    out = {}
    for key, _, scale, _ in METRICS:
        if key == "success":
            col = "success_rate"
        elif key == "regret":
            col = "regret_mean"
        else:
            col = f"{key}_median"
        out[key] = cells[col].to_numpy(float) * scale
    return out


def episodeValues(episodes: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each metric's per-episode mean (success is the colour, not an axis)."""
    return {key: episodes[f"{key}_mean"].to_numpy(float) * scale
            for key, _, scale, _ in METRICS if key != "success"}


def logAxis(axis):
    """Log ticks at 1, 2 and 5 per decade, labelled compactly (e.g. 2e5), no minor labels."""
    axis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0e}".replace("e+0", "e").replace("e-0", "e-")))
    axis.set_minor_formatter(ticker.NullFormatter())


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank())


def pairMatrix(values: dict, groups: list[tuple[np.ndarray, dict]], handles, title: str, path: Path,
               point_size: float):
    """Lower-triangle scatter matrix of ``values``; ``groups`` is a list of (mask, scatter kwargs)."""
    metrics = [m for m in METRICS if m[0] in values]
    n = len(metrics)
    fig, axes = plt.subplots(n - 1, n - 1, figsize=(2.6 * (n - 1) + 1.5, 2.4 * (n - 1) + 0.8), squeeze=False)
    for i in range(1, n):                 # row: y metric
        ykey, ylabel, _, ylog = metrics[i]
        for j in range(n - 1):            # column: x metric
            ax = axes[i - 1, j]
            if j >= i:
                ax.set_visible(False)
                continue
            xkey, xlabel, _, xlog = metrics[j]
            x, y = values[xkey], values[ykey]
            for mask, kw in groups:
                ax.scatter(x[mask], y[mask], s=point_size, linewidths=0.4, **kw)
            if xlog:
                ax.set_xscale("log")
                logAxis(ax.xaxis)
            if ylog:
                ax.set_yscale("log")
                logAxis(ax.yaxis)
            rho = spearman(x, y)
            ax.text(0.03, 0.97, f"ρ={rho:+.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    color="#222222", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.grid(color="#eeeeee", linewidth=0.6)
            ax.set_axisbelow(True)
            if i == n - 1:
                ax.set_xlabel(xlabel, fontsize=9)
            else:
                ax.tick_params(which="both", labelbottom=False)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            else:
                ax.tick_params(which="both", labelleft=False)
    # Share axes along rows/columns by metric so the grid reads as one matrix.
    for j in range(n - 1):
        col_axes = [axes[i - 1, j] for i in range(j + 1, n)]
        lims = [a.get_xlim() for a in col_axes]
        for a in col_axes:
            a.set_xlim(min(l[0] for l in lims), max(l[1] for l in lims))
    for i in range(1, n):
        row_axes = [axes[i - 1, j] for j in range(i)]
        lims = [a.get_ylim() for a in row_axes]
        for a in row_axes:
            a.set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.95), frameon=False, fontsize=10)
    fig.suptitle(title, x=0.02, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plotCells(cells: pd.DataFrame, temp: float, out_dir: Path):
    values = cellValues(cells)
    models = sorted(cells["rollout_model"].unique())
    tasks = [t for t in TASK_MARKERS if t in set(cells["task"])]
    groups = []
    for m in models:
        for t in tasks:
            mask = ((cells["rollout_model"] == m) & (cells["task"] == t)).to_numpy()
            groups.append((mask, dict(color=MODEL_COLORS.get(m, "#777777"), marker=TASK_MARKERS[t],
                                      edgecolor="white", alpha=0.9)))
    handles = [Line2D([], [], ls="", marker="o", ms=8, color=MODEL_COLORS.get(m, "#777777"), label=f"rollout {m}")
               for m in models]
    handles += [Line2D([], [], ls="", marker=TASK_MARKERS[t], ms=8, color="#777777", label=t.removesuffix("_reorient"))
                for t in tasks]
    eval_sims = ", ".join(sorted(cells["eval_sim"].astype(str).unique()))
    pairMatrix(values, groups, handles,
               f"Per-cell metrics at temperature {temp:g}  ({len(cells)} cells, eval sim {eval_sims}; "
               f"regret = cell mean, others = cell median; ρ = Spearman)",
               out_dir / f"pairs_cells_T_{temp:g}.png", point_size=40)


def plotEpisodes(episodes: pd.DataFrame, temp: float, out_dir: Path):
    values = episodeValues(episodes)
    success = (episodes["success"].astype(str).str.lower() == "true").to_numpy()
    groups = [(~success, dict(color=FAILURE_COLOR, alpha=0.45, edgecolor="none")),
              (success, dict(color=SUCCESS_COLOR, alpha=0.35, edgecolor="none"))]
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=SUCCESS_COLOR, label=f"success ({success.sum()})"),
               Line2D([], [], ls="", marker="o", ms=7, color=FAILURE_COLOR, label=f"not success ({(~success).sum()})")]
    pairMatrix(values, groups, handles,
               f"Per-episode metrics at temperature {temp:g}  ({len(episodes)} episodes, per-episode means; "
               f"ρ = Spearman)",
               out_dir / f"pairs_episodes_T_{temp:g}.png", point_size=6)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, default=Path("results/final_episodes.csv"))
    p.add_argument("--cells", type=Path, default=Path("results/final_results.csv"))
    p.add_argument("--out", type=Path, default=Path("results/plots/metric_pairs"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cells = readTable(args.cells)
    if "analysis_status" in cells:
        cells = cells[cells["analysis_status"] == "done"]
    episodes = readTable(args.episodes)
    for temp in sorted(cells["temperature"].unique(), reverse=True):
        plotCells(cells[cells["temperature"] == temp].reset_index(drop=True), temp, args.out)
    for temp in sorted(episodes["temperature"].unique(), reverse=True):
        plotEpisodes(episodes[episodes["temperature"] == temp].reset_index(drop=True), temp, args.out)


if __name__ == "__main__":
    main()
