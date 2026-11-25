#!/usr/bin/env python3
"""Success rate per task and rollout model, from ``run_episodes.py`` result JSONs.

One group of bars per task, one bar per rollout model, each with its 95% Wilson
interval and, below it, ``successes/episodes`` and the timeouts (T) and
failures (F) that make up the rest. The task and model are read from each
file's recorded configs, so the files can be named anything. Two files for the
same task and model are an error, not pooled.

Usage:
    python experiments/plot_sr_by_task_model.py results/bo_check/*.json results/m6/*.json \
        --out results/m6/sr_by_task_model.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK_ORDER = ["cube", "duck", "ball"]
# Fixed per model, so a model keeps its colour whichever others are plotted
# (M1-M4 as in results/bo_check/sr_by_task_model.png).
MODEL_COLORS = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#eda100",
                "M6": "#e87ba4", "M5": "#6250d6"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval on k successes of n."""
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def readRun(path: Path) -> dict:
    d = json.load(open(path))
    cli = d["configs"]["metadata"]["cli_args"]
    reasons = d["finish_reasons"]
    return dict(task=cli["task"].removesuffix("_reorient"), model=cli["rollout_model"].upper(),
                eval_sim=cli["eval_sim"], temperature=cli["temperature"], n=d["n_episodes"],
                k=d["n_success"], timeouts=reasons.get("timeout", 0),
                failures=reasons.get("failure", 0), path=path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", type=Path, help="run_episodes.py result JSONs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default=None)
    a = p.parse_args()

    runs = {}
    for path in a.results:
        r = readRun(path)
        key = (r["task"], r["model"])
        if key in runs:
            raise SystemExit(f"two results for {key}: {runs[key]['path']} and {path}")
        runs[key] = r
    tasks = [t for t in TASK_ORDER if any(k[0] == t for k in runs)]
    tasks += sorted({k[0] for k in runs} - set(tasks))
    models = sorted({k[1] for k in runs})
    evals = sorted({r["eval_sim"] for r in runs.values()})
    temps = sorted({r["temperature"] for r in runs.values()})
    ns = sorted({r["n"] for r in runs.values()})

    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(4.0 + 0.6 * len(tasks) * len(models), 5.2))
    for ti, task in enumerate(tasks):
        for mi, model in enumerate(models):
            r = runs.get((task, model))
            if r is None:
                continue
            x = ti + (mi - (len(models) - 1) / 2) * width
            sr = r["k"] / r["n"]
            lo, hi = wilson(r["k"], r["n"])
            ax.bar(x, sr, width * 0.92, color=MODEL_COLORS.get(model, "#888888"), zorder=2)
            ax.errorbar(x, sr, yerr=[[sr - lo], [hi - sr]], color="#444444", capsize=4, lw=1.3, zorder=3)
            ax.annotate(f"{r['k']}/{r['n']}", (x, 0), xytext=(0, -16), textcoords="offset points",
                        ha="center", va="top", fontsize=9, color=MUTED, annotation_clip=False)
            ax.annotate(f"T{r['timeouts']} F{r['failures']}", (x, 0), xytext=(0, -29),
                        textcoords="offset points", ha="center", va="top", fontsize=7.5,
                        color="#8a8984", annotation_clip=False)
        ax.annotate(task, (ti, 0), xytext=(0, -62), textcoords="offset points", ha="center",
                    va="top", fontsize=12, color=TEXT, annotation_clip=False)

    ax.set_xticks([])
    ax.set_xlim(-0.6, len(tasks) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Success rate (95% Wilson interval)", color=TEXT)
    ax.grid(axis="y", color=GRID, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, colors=TEXT)
    n_txt = f"{ns[0]} episodes per pair" if len(ns) == 1 else f"{'/'.join(map(str, ns))} episodes"
    title = a.title or (f"Rollout model SR, {'/'.join(evals)} eval, {n_txt}, "
                        f"MPPI temperature {'/'.join(f'{t:g}' for t in temps)}")
    ax.set_title(title + "\nSR = successes / episodes; timeouts (T) and failures (F) both count as "
                 "non-successes", loc="left", fontsize=11, color=TEXT)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS.get(m, "#888888")) for m in models]
    fig.tight_layout()
    # below the task names, which sit a fixed number of points under the axis
    fig.legend(handles, [f"{m} rollout" for m in models], loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=len(models), frameon=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}")
    for (task, model), r in sorted(runs.items()):
        print(f"  {task:5s} {model}: {r['k']}/{r['n']}  T{r['timeouts']} F{r['failures']}  ({r['path']})")


if __name__ == "__main__":
    main()
