#!/usr/bin/env python3
"""Plot the processed batch results (``process_batch_results.py`` output).

From the per-episode table it draws, for each metric, a grid of histograms with
one panel per cell (rows = rollout model, columns = hand accuracy), each
episode's bar segment coloured by whether it succeeded. From the per-cell table
it draws success rate against each metric, one marker per cell. The x position is
the cell's median with a whisker up to its p90; ``process_batch_results.stats``
computes both over every analysed step of every episode in the cell pooled
together (no lower percentile is stored, so the whisker is one-sided). The
vertical bar is a 95% Wilson interval on the success rate. It also draws each KL
against each forward error, both axes as median with a whisker to p90.

Metrics:
    KL divergence           KL(used || opt)       kl_used_opt_mean
    Reverse KL              KL(opt || used)       kl_opt_used_mean
    Forward position error  object position (mm) fsim_obj_pos_err_mean
    Forward orientation err object rotation (rad) fsim_obj_rot_err_mean
    Forward qvel error      full qvel             fsim_qvel_err_mean

KL values span several orders of magnitude, so they are binned and plotted on a
log axis; the forward errors use linear axes.

Usage:
    python experiments/plot_final_results.py \
        --episodes results/final_episodes.csv --cells results/final_results.csv \
        --out results/plots
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# (column, label, scale applied to values, log axis)
METRICS = [
    ("kl_used_opt_mean", "KL divergence  KL(used ‖ opt)", 1.0, True),
    ("kl_opt_used_mean", "Reverse KL  KL(opt ‖ used)", 1.0, True),
    ("fsim_obj_pos_err_mean", "Forward position error (mm)", 1e3, False),
    ("fsim_obj_rot_err_mean", "Forward orientation error (rad)", 1.0, False),
    ("fsim_qvel_err_mean", "Forward qvel error", 1.0, False),
]

SUCCESS_COLOR = "#2a78d6"
FAILURE_COLOR = "#eb6834"
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4", "#008300"]
ACC_ORDER = ["high", "med", "low"]
ACC_MARKERS = {"high": "o", "med": "s", "low": "^"}
N_BINS = 20


def styleAxes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def orderedLevels(values, preferred):
    levels = list(dict.fromkeys(values))
    return [v for v in preferred if v in levels] + sorted(v for v in levels if v not in preferred)


def sharedBins(values, log):
    values = values[np.isfinite(values)]
    if log:
        values = values[values > 0]
        return np.logspace(np.log10(values.min()), np.log10(values.max()), N_BINS + 1)
    return np.linspace(values.min(), values.max(), N_BINS + 1)


def wilsonInterval(successes, n, z=1.96):
    """95% Wilson score interval for a binomial proportion, elementwise."""
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def plotEpisodeHistograms(episodes: pd.DataFrame, out_dir: Path):
    success = episodes["success"].astype(str).str.lower() == "true"
    models = orderedLevels(episodes["rollout_model"], sorted(episodes["rollout_model"].unique()))
    accs = orderedLevels(episodes["hand_acc"], ACC_ORDER)

    for col, label, scale, log in METRICS:
        values = episodes[col].to_numpy(dtype=float) * scale
        bins = sharedBins(values, log)
        fig, axes = plt.subplots(len(models), len(accs), figsize=(4 * len(accs), 3 * len(models)),
                                 sharex=True, squeeze=False)
        for i, model in enumerate(models):
            for j, acc in enumerate(accs):
                ax = axes[i, j]
                cell = (episodes["rollout_model"] == model) & (episodes["hand_acc"] == acc)
                if not cell.any():
                    ax.set_visible(False)
                    continue
                v = values[cell.to_numpy()]
                s = success[cell].to_numpy()
                ax.hist([v[s], v[~s]], bins=bins, stacked=True, color=[SUCCESS_COLOR, FAILURE_COLOR],
                        edgecolor="white", linewidth=1)
                name = episodes.loc[cell, "label"].iloc[0]
                ax.set_title(f"{name}   ({s.sum()}/{len(s)} success)", fontsize=10)
                if log:
                    ax.set_xscale("log")
                styleAxes(ax)
                if j == 0:
                    ax.set_ylabel("Episodes")
                if i == len(models) - 1:
                    ax.set_xlabel(label)
        fig.legend(handles=[Patch(color=SUCCESS_COLOR, label="Success"),
                            Patch(color=FAILURE_COLOR, label="Not success")],
                   loc="upper right", frameon=False, ncol=2)
        fig.suptitle(f"{label} per episode", x=0.02, ha="left", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = out_dir / f"hist_{col}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")


def cellStyles(cells: pd.DataFrame):
    """Colour per rollout model and marker per hand accuracy, plus the legend handles for both."""
    models = orderedLevels(cells["rollout_model"], sorted(cells["rollout_model"].unique()))
    accs = orderedLevels(cells["hand_acc"], ACC_ORDER)
    model_color = {m: MODEL_COLORS[k % len(MODEL_COLORS)] for k, m in enumerate(models)}
    acc_marker = {a: ACC_MARKERS.get(a, "D") for a in accs}
    handles = [Line2D([], [], linestyle="", marker="o", markersize=9, color=model_color[m],
                      label=f"rollout {m}") for m in models]
    handles += [Line2D([], [], linestyle="", marker=acc_marker[a], markersize=9, color="#777777",
                       label=f"hand acc {a}") for a in accs]
    return model_color, acc_marker, handles


def medianP90(cells: pd.DataFrame, col: str, scale: float):
    """Each cell's median and the one-sided (median -> p90) error, as errorbar expects."""
    base = col.removesuffix("_mean")
    median = cells[f"{base}_median"].to_numpy(float) * scale
    p90 = cells[f"{base}_p90"].to_numpy(float) * scale
    return median, np.vstack([np.zeros_like(median), p90 - median])


def scatterCells(cells, x, y, xerr, yerr, xlabel, ylabel, title, xlog, ylog, path, ylim=None):
    """One marker per cell with its error bars; saved to ``path``."""
    model_color, acc_marker, handles = cellStyles(cells)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, (_, row) in enumerate(cells.iterrows()):
        color = model_color[row["rollout_model"]]
        ax.errorbar(x[k], y[k], xerr=xerr[:, k:k + 1], yerr=yerr[:, k:k + 1], fmt="none", ecolor=color,
                    alpha=0.5, elinewidth=1.5, capsize=3, zorder=2)
        ax.scatter(x[k], y[k], s=90, color=color, marker=acc_marker[row["hand_acc"]],
                   edgecolor="white", linewidth=1.5, zorder=3)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    styleAxes(ax)
    ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plotSuccessScatter(cells: pd.DataFrame, out_dir: Path):
    lo, hi = wilsonInterval(cells["n_success"].to_numpy(float), cells["n_episodes_run"].to_numpy(float))
    rate = cells["success_rate"].to_numpy(float)
    yerr = np.vstack([rate - lo, hi - rate]).clip(min=0)
    for col, label, scale, log in METRICS:
        x, xerr = medianP90(cells, col, scale)
        scatterCells(cells, x, rate, xerr, yerr, f"{label}, cell median (whisker to p90)",
                     "Success rate (95% Wilson interval)", f"Success rate vs {label}", log, False,
                     out_dir / f"scatter_success_vs_{col}.png", ylim=(-0.05, 1.05))


def plotKLvsForwardScatter(cells: pd.DataFrame, out_dir: Path):
    """Each KL against each forward error; both axes median with a whisker to p90."""
    kls = [m for m in METRICS if m[0].startswith("kl_")]
    errors = [m for m in METRICS if m[0].startswith("fsim_")]
    for kcol, klabel, kscale, klog in kls:
        y, yerr = medianP90(cells, kcol, kscale)
        for ecol, elabel, escale, elog in errors:
            x, xerr = medianP90(cells, ecol, escale)
            scatterCells(cells, x, y, xerr, yerr, f"{elabel}, cell median (whisker to p90)",
                         f"{klabel}, cell median (whisker to p90)", f"{klabel} vs {elabel}", elog, klog,
                         out_dir / f"scatter_{kcol}_vs_{ecol}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, default=Path("results/final_episodes.csv"))
    p.add_argument("--cells", type=Path, default=Path("results/final_results.csv"))
    p.add_argument("--out", type=Path, default=Path("results/plots"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    plotEpisodeHistograms(pd.read_csv(args.episodes), args.out)
    cells = pd.read_csv(args.cells)
    plotSuccessScatter(cells, args.out)
    plotKLvsForwardScatter(cells, args.out)


if __name__ == "__main__":
    main()
