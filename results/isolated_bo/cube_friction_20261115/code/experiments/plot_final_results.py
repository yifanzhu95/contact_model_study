#!/usr/bin/env python3
"""Plot the processed batch results (``process_batch_results.py`` output).

Everything is plotted per cell (a row of the batch CSV), never pooled across
cells, so cells that differ only in a setting such as the temperature stay
apart. A bar chart of success rate, one bar per cell (95% Wilson interval),
needs only the batch's own results and is always drawn. The plots below need
the analysis metrics and are skipped if the tables lack them.

From the per-episode table it draws, for each metric, a grid of histograms with
one panel per cell (a row of panels per rollout model, wrapped at
``MAX_COLS``), each episode's bar segment coloured by whether it succeeded.
From the per-cell table it draws success rate against each metric, one
labelled marker per cell. The x position is
the cell's median with a whisker up to its p90; ``process_batch_results.stats``
computes both over every analysed step of every episode in the cell pooled
together (no lower percentile is stored, so the whisker is one-sided). The
vertical bar is a 95% Wilson interval on the success rate. It also draws each KL
against each forward error, both axes as median with a whisker to p90.

The regret (``regret_mean``: what the model gap cost the used plan, in free
energy on the eval model) gets a bar chart, one bar per cell laid out as the
success bars, and its own scatters: success rate against it, and it against
each KL and each forward error. Regret is skewed and can be negative, so it is
shown as the cell's mean with its 95% interval on a linear axis, the bar chart
marking the median too. It is scored at each cell's own temperature, so it
compares cells within a temperature, not across them. Tables without it skip
these plots.

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
# hand accuracy is ordinal, so one hue from dark (high) to light (low)
ACC_BAR_COLORS = {"high": "#0d4a8f", "med": "#2a78d6", "low": "#8dbbee"}
#: The regret's label in the plots.
REGRET_LABEL = "Regret  F_A(U_used) − F_A(U_zero-gap)"
N_BINS = 20
#: Histogram panels per row; a rollout model with more cells wraps onto further rows.
MAX_COLS = 9


def readTable(path: Path) -> pd.DataFrame:
    """Read a processed table, dropping rows without a rollout model or hand accuracy
    (a cell whose settings were not recorded)."""
    return pd.read_csv(path).dropna(subset=["rollout_model", "hand_acc"]).reset_index(drop=True)


def analysedRows(df: pd.DataFrame, path: Path, suffixes: tuple[str, ...]) -> pd.DataFrame | None:
    """The rows of ``df`` whose analysis metrics are present, or None (with a message) if there are none.

    Rows whose cell analysis has not run are dropped, so a partly analysed batch
    still plots. ``suffixes`` are the metric column suffixes the plots need.
    """
    cols = [c.removesuffix("_mean") + s for c, *_ in METRICS for s in suffixes]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"skipping analysis plots: {path} has no analysis metrics ({', '.join(missing)}); "
              "run process_batch_results.py on the batch, then --summarize")
        return None
    if "analysis_status" in df.columns:
        df = df[df["analysis_status"] == "done"]
    df = df.dropna(subset=cols, how="all")
    if df.empty:
        print(f"skipping analysis plots: {path} has no analysed rows")
        return None
    return df.reset_index(drop=True)


def styleAxes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def orderedLevels(values, preferred):
    levels = list(dict.fromkeys(values))
    return [v for v in preferred if v in levels] + sorted(v for v in levels if v not in preferred)


def cellLabel(row) -> str:
    """A cell's CSV label, or its name when the row had none."""
    label = row.get("label")
    return str(label) if isinstance(label, str) and label else str(row["name"])


def shortLabels(cells: pd.DataFrame) -> list[str]:
    """Each cell's label without the ``_``-separated parts every label shares in the same place.

    ``cube_M1_M5_high_T_0.5`` among cells that all start ``cube_`` and share
    the eval sim ``M5`` becomes ``M1_high_T_0.5``. Labels with differing
    numbers of parts are kept whole.
    """
    parts = [cellLabel(row).split("_") for _, row in cells.iterrows()]
    if len(parts) < 2 or len({len(p) for p in parts}) != 1:
        return ["_".join(p) for p in parts]
    keep = [k for k in range(len(parts[0])) if len({p[k] for p in parts}) > 1]
    return ["_".join(p[k] for k in keep) for p in parts]


def cellsInOrder(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cell of ``df`` (its first), in batch order."""
    order = "cell" if "cell" in df.columns else "name"
    return df.sort_values(order, kind="stable").drop_duplicates("name").reset_index(drop=True)


def cellGrid(cells: pd.DataFrame) -> list[list]:
    """Rows of cell rows for a panel grid: each rollout model's cells in batch order, ``MAX_COLS`` a row."""
    models = orderedLevels(cells["rollout_model"], sorted(cells["rollout_model"].unique()))
    grid = []
    for model in models:
        mine = [row for _, row in cells[cells["rollout_model"] == model].iterrows()]
        grid += [mine[k:k + MAX_COLS] for k in range(0, len(mine), MAX_COLS)]
    return grid


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


def plotEpisodeHistograms(episodes: pd.DataFrame, out_dir: Path, layout: pd.DataFrame | None = None):
    """One histogram panel per cell of ``layout`` (default: the cells in ``episodes``).

    Laying out every cell of the batch keeps the panels in step from row to row
    when some cells are not analysed yet; those panels are left empty.
    """
    success = (episodes["success"].astype(str).str.lower() == "true").to_numpy()
    grid = cellGrid(cellsInOrder(episodes if layout is None else layout))
    n_rows, n_cols = len(grid), max(len(r) for r in grid)

    for col, label, scale, log in METRICS:
        values = episodes[col].to_numpy(dtype=float) * scale
        bins = sharedBins(values, log)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, squeeze=False)
        for i, row in enumerate(grid):
            for j in range(n_cols):
                ax = axes[i, j]
                if j >= len(row):
                    ax.set_visible(False)
                    continue
                cell = (episodes["name"] == row[j]["name"]).to_numpy()
                if not cell.any():
                    # An empty but otherwise normal panel: the x axis is shared, so hiding this
                    # one's ticks would hide them for its whole column when it is in the bottom row.
                    ax.set_title(cellLabel(row[j]), fontsize=10, color="#777777")
                    ax.text(0.5, 0.5, "not analysed", transform=ax.transAxes, ha="center", va="center",
                            color="#777777")
                    ax.set_yticks([])
                    if log:
                        ax.set_xscale("log")
                    styleAxes(ax)
                    continue
                v, s = values[cell], success[cell]
                ax.hist([v[s], v[~s]], bins=bins, stacked=True, color=[SUCCESS_COLOR, FAILURE_COLOR],
                        edgecolor="white", linewidth=1)
                ax.set_title(f"{cellLabel(row[j])}   ({s.sum()}/{len(s)} success)", fontsize=10)
                if log:
                    ax.set_xscale("log")
                styleAxes(ax)
                if j == 0:
                    ax.set_ylabel("Episodes")
        # Shared x: only the bottom row is labelled, so a column whose bottom slot is empty
        # gets its labels on its lowest panel instead.
        for j in range(n_cols):
            lowest = max(i for i, row in enumerate(grid) if j < len(row))
            axes[lowest, j].tick_params(axis="x", which="both", labelbottom=True)
        fig.supxlabel(label)
        fig.legend(handles=[Patch(color=SUCCESS_COLOR, label="Success"),
                            Patch(color=FAILURE_COLOR, label="Not success")],
                   loc="upper right", frameon=False, ncol=2)
        fig.suptitle(f"{label} per episode, one panel per cell", x=0.02, ha="left", fontsize=13)
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


def regretMean(cells: pd.DataFrame):
    """Each cell's mean regret and the two-sided error to its 95% interval, as errorbar expects."""
    mean = cells["regret_mean"].to_numpy(float)
    lo = cells["regret_mean_lo"].to_numpy(float) if "regret_mean_lo" in cells else mean
    hi = cells["regret_mean_hi"].to_numpy(float) if "regret_mean_hi" in cells else mean
    return mean, np.vstack([mean - lo, hi - mean]).clip(min=0)


def hasRegret(cells: pd.DataFrame) -> bool:
    """Whether any cell has a regret; says so when none has."""
    if "regret_mean" in cells and np.isfinite(cells["regret_mean"].to_numpy(float)).any():
        return True
    print("skipping regret plots: the cells table has no regret (analysed before it was computed?)")
    return False


def scatterCells(cells, x, y, xerr, yerr, xlabel, ylabel, title, xlog, ylog, path, ylim=None):
    """One marker per cell with its error bars, labelled with the cell's label; saved to ``path``."""
    model_color, acc_marker, handles = cellStyles(cells)
    labels = shortLabels(cells)
    fig, ax = plt.subplots(figsize=(9, 6))
    for k, (_, row) in enumerate(cells.iterrows()):
        color = model_color[row["rollout_model"]]
        ax.errorbar(x[k], y[k], xerr=xerr[:, k:k + 1], yerr=yerr[:, k:k + 1], fmt="none", ecolor=color,
                    alpha=0.5, elinewidth=1.5, capsize=3, zorder=2)
        ax.scatter(x[k], y[k], s=90, color=color, marker=acc_marker[row["hand_acc"]],
                   edgecolor="white", linewidth=1.5, zorder=3)
        if np.isfinite(x[k]) and np.isfinite(y[k]):
            ax.annotate(labels[k], (x[k], y[k]), xytext=(5, 4), textcoords="offset points",
                        fontsize=6, color="#444444", zorder=4)
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


def cellBars(cells: pd.DataFrame):
    """Bar positions for a per-cell bar chart: each rollout model's cells in batch order, a gap between models.

    Returns the cells as a list of rows, their x positions, a figure and its axes.
    """
    cells = cellsInOrder(cells)
    models = orderedLevels(cells["rollout_model"], sorted(cells["rollout_model"].unique()))
    xs, rows, x = [], [], 0
    for model in models:
        for _, row in cells[cells["rollout_model"] == model].iterrows():
            xs.append(x)
            rows.append(row)
            x += 1
        x += 1                                       # the gap before the next rollout model
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(rows) + 3), 5.5))
    return rows, np.array(xs), fig, ax


def finishCellBars(fig, ax, rows, xs, title: str, path: Path, extra_handles=()):
    """Cell labels under the bars, the hand-accuracy legend, the title; saved to ``path``."""
    accs = orderedLevels([r["hand_acc"] for r in rows], ACC_ORDER)
    ax.set_xticks(xs, [cellLabel(r) for r in rows], rotation=90, fontsize=7)
    ax.set_title(title, loc="left")
    styleAxes(ax)
    handles = [Patch(color=ACC_BAR_COLORS.get(a, "#777777"), label=f"hand acc {a}") for a in accs]
    ax.legend(handles=handles + list(extra_handles), frameon=False, loc="center left",
              bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def barColors(rows) -> list[str]:
    return [ACC_BAR_COLORS.get(r["hand_acc"], "#777777") for r in rows]


def plotSuccessBars(cells: pd.DataFrame, out_dir: Path):
    """Success rate of each cell, with a 95% Wilson interval, grouped by rollout model.

    Needs only the batch's own results, not the analysis. One bar per cell,
    coloured by hand accuracy; a gap separates the rollout models.
    """
    cells = cells[cells["n_episodes_run"] > 0]
    rows, xs, fig, ax = cellBars(cells)
    n_success = np.array([r["n_success"] for r in rows], dtype=float)
    n = np.array([r["n_episodes_run"] for r in rows], dtype=float)
    rate = n_success / n
    lo, hi = wilsonInterval(n_success, n)

    ax.bar(xs, rate, 0.8, color=barColors(rows), edgecolor="white", linewidth=1, zorder=2)
    ax.errorbar(xs, rate, yerr=np.vstack([rate - lo, hi - rate]).clip(min=0), fmt="none", ecolor="#444444",
                elinewidth=1.2, capsize=3, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate (95% Wilson interval)")
    eval_sims = ", ".join(sorted(cells["eval_sim"].astype(str).unique()))
    finishCellBars(fig, ax, rows, xs, f"Success rate per cell, grouped by rollout model (eval sim {eval_sims})",
                   out_dir / "bar_success_by_cell.png")


def plotRegretBars(cells: pd.DataFrame, out_dir: Path):
    """Mean regret of each cell, with its 95% interval and a dot at the median, laid out as the success bars.

    Cells not analysed yet keep their place, marked n/a, so the two charts line up.
    """
    rows, xs, fig, ax = cellBars(cells[cells["n_episodes_run"] > 0])
    sub = pd.DataFrame(rows)
    mean, err = regretMean(sub)
    median = sub["regret_median"].to_numpy(float) if "regret_median" in sub else np.full(len(sub), np.nan)
    ok = np.isfinite(mean)

    ax.axhline(0, color="#444444", linewidth=0.8, zorder=1)
    ax.bar(xs[ok], mean[ok], 0.8, color=[c for c, k in zip(barColors(rows), ok) if k], edgecolor="white",
           linewidth=1, zorder=2)
    ax.errorbar(xs[ok], mean[ok], yerr=err[:, ok], fmt="none", ecolor="#444444", elinewidth=1.2, capsize=3,
                zorder=3)
    ax.scatter(xs[ok], median[ok], s=14, color="#111111", zorder=4)
    for x in xs[~ok]:
        ax.text(x, 0, "n/a", rotation=90, ha="center", va="bottom", fontsize=7, color="#777777")
    ax.set_ylabel("Regret, cell mean (95% interval)")
    median_handle = Line2D([], [], linestyle="", marker="o", markersize=4, color="#111111", label="median")
    finishCellBars(fig, ax, rows, xs, f"{REGRET_LABEL} per cell (at each cell's own temperature)",
                   out_dir / "bar_regret_by_cell.png", extra_handles=[median_handle])


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


def plotRegretScatter(cells: pd.DataFrame, out_dir: Path):
    """Success rate against the regret, and the regret against each KL and each forward error.

    The regret is its cell mean with the 95% interval; the other metrics are
    median with a whisker to p90, as in the other scatters.
    """
    regret, regret_err = regretMean(cells)
    regret_label = f"{REGRET_LABEL}, cell mean (95% interval)"
    lo, hi = wilsonInterval(cells["n_success"].to_numpy(float), cells["n_episodes_run"].to_numpy(float))
    rate = cells["success_rate"].to_numpy(float)
    scatterCells(cells, regret, rate, regret_err, np.vstack([rate - lo, hi - rate]).clip(min=0), regret_label,
                 "Success rate (95% Wilson interval)", "Success rate vs regret", False, False,
                 out_dir / "scatter_success_vs_regret_mean.png", ylim=(-0.05, 1.05))
    for col, label, scale, log in METRICS:
        x, xerr = medianP90(cells, col, scale)
        scatterCells(cells, x, regret, xerr, regret_err, f"{label}, cell median (whisker to p90)", regret_label,
                     f"Regret vs {label}", log, False, out_dir / f"scatter_regret_mean_vs_{col}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, default=Path("results/final_episodes.csv"))
    p.add_argument("--cells", type=Path, default=Path("results/final_results.csv"))
    p.add_argument("--out", type=Path, default=Path("results/plots"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    all_cells = readTable(args.cells)
    plotSuccessBars(all_cells, args.out)
    if hasRegret(all_cells):
        plotRegretBars(all_cells, args.out)

    all_episodes = readTable(args.episodes)
    episodes = analysedRows(all_episodes, args.episodes, ("_mean",))
    if episodes is not None:
        plotEpisodeHistograms(episodes, args.out, layout=all_episodes)
    cells = analysedRows(all_cells, args.cells, ("_median", "_p90"))
    if cells is not None:
        plotSuccessScatter(cells, args.out)
        plotKLvsForwardScatter(cells, args.out)
        if hasRegret(cells):
            plotRegretScatter(cells, args.out)


if __name__ == "__main__":
    main()
