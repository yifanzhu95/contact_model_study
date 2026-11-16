#!/usr/bin/env python3
"""Success rate per cell of a CSV run, one figure per task, grouped by hand accuracy.

Each figure has one panel per hand (geometric) accuracy level. Within a panel
there is one cluster per rollout model and one bar per MPPI temperature, with
its 95% Wilson interval and ``successes/episodes`` below. Cells listed in the
run CSV that have no result JSON yet are drawn as a hatched "no data" box.

Usage:
    python experiments/plot_sr_by_geom_acc.py results/csv_run_11_3_26.csv \
        results/episode_batches_csv_run_11_3_26_31940749 --out-dir results/sr_by_geom_acc
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_sr_by_task_model import wilson, TEXT, MUTED, GRID

ACC_ORDER = ["high", "med", "low"]
# Temperature is ordinal, so one blue ramp light -> dark (low -> high temperature).
TEMP_RAMP = ["#86b6ef", "#2a78d6", "#104281"]
NODATA = "#b5b4ae"


def loadCells(run_csv: Path, batch_dir: Path) -> list[dict]:
    cells = []
    for i, row in enumerate(csv.DictReader(open(run_csv))):
        path = batch_dir / f"cell_{i:04d}_{row['label']}.json"
        r = dict(task=row["task"].removesuffix("_reorient"), model=row["rollout_model"],
                 acc=row["hand_acc"], temp=float(row["temperature"]), k=None, n=None)
        if path.exists():
            d = json.load(open(path))
            r.update(k=d["n_success"], n=d["n_episodes"])
        cells.append(r)
    return cells


def plotTask(task: str, cells: list[dict], out: Path):
    models = sorted({c["model"] for c in cells})
    temps = sorted({c["temp"] for c in cells})
    accs = [a for a in ACC_ORDER if any(c["acc"] == a for c in cells)]
    by = {(c["acc"], c["model"], c["temp"]): c for c in cells}
    width = 0.8 / len(temps)

    fig, axes = plt.subplots(1, len(accs), figsize=(5.2 * len(accs), 4.8), sharey=True)
    for ax, acc in zip(axes, accs):
        for mi, model in enumerate(models):
            for ti, temp in enumerate(temps):
                c = by.get((acc, model, temp))
                if c is None:
                    continue
                x = mi + (ti - (len(temps) - 1) / 2) * width
                if c["k"] is None:
                    ax.bar(x, 1.0, width * 0.9, color="none", edgecolor=NODATA, hatch="///",
                           lw=0.8, zorder=2)
                    ax.text(x, 0.5, "no data", rotation=90, ha="center", va="center",
                            fontsize=8, color=MUTED, zorder=3)
                    continue
                sr = c["k"] / c["n"]
                lo, hi = wilson(c["k"], c["n"])
                ax.bar(x, sr, width * 0.9, color=TEMP_RAMP[ti % len(TEMP_RAMP)], zorder=2)
                ax.errorbar(x, sr, yerr=[[sr - lo], [hi - sr]], color="#444444", capsize=2,
                            lw=1.0, zorder=3)
                ax.annotate(f"{c['k']}", (x, 0), xytext=(0, -3), textcoords="offset points",
                            ha="center", va="top", fontsize=6.5, color=MUTED,
                            annotation_clip=False)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.tick_params(axis="x", length=0, pad=14, colors=TEXT)
        ax.tick_params(axis="y", length=0, colors=TEXT)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color=GRID, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        n_done = sum(1 for c in cells if c["acc"] == acc and c["k"] is not None)
        n_all = sum(1 for c in cells if c["acc"] == acc)
        ax.set_title(f"hand accuracy: {acc}   ({n_done}/{n_all} cells done)", loc="left",
                     fontsize=11, color=TEXT)
    axes[0].set_ylabel("Success rate (95% Wilson interval)", color=TEXT)
    ns = sorted({c["n"] for c in cells if c["n"] is not None})
    fig.suptitle(f"{task}: success rate by rollout model and MPPI temperature "
                 f"({'/'.join(map(str, ns))} episodes per cell; number under bar = successes)",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TEMP_RAMP[i % len(TEMP_RAMP)])
               for i in range(len(temps))]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=NODATA, hatch="///"))
    fig.tight_layout()
    fig.legend(handles, [f"T = {t:g}" for t in temps] + ["no data yet"], loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(handles), frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_csv", type=Path)
    p.add_argument("batch_dir", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    cells = loadCells(a.run_csv, a.batch_dir)
    for task in sorted({c["task"] for c in cells}):
        plotTask(task, [c for c in cells if c["task"] == task], a.out_dir / f"sr_{task}_by_geom_acc.png")
    for c in cells:
        if c["k"] is None:
            print(f"  no data: {c['task']} {c['model']} {c['acc']} T={c['temp']:g}")


if __name__ == "__main__":
    main()
