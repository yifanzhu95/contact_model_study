"""analysis/plot_results.py

Generates all figures for the study.

Figures produced:
  1. accuracy_speed_frontier.pdf  - Pareto plot (approx_err vs speedup), one per task
  2. success_rate_table.pdf        - heatmap: models × tasks × condition
  3. condition_ab_delta.pdf        - bar chart: success rate change A→B per model
  4. contact_complexity_curve.pdf  - success rate vs. contact complexity for each Mk
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from contact_study.evaluation.metrics import (
    load_results,
    build_results_table,
    accuracy_speed_frontier,
    AggregatedResult,
)

FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

TASK_ORDER  = ["push", "grasp_reorient", "peg_in_hole"]
TASK_LABELS = {"push": "Push (Low)", "grasp_reorient": "Grasp/Reorient (Med)", "peg_in_hole": "Peg-in-Hole (High)"}
COND_COLORS = {"A": "#2196F3", "B": "#FF5722"}


# ---------------------------------------------------------------------------
# Figure 1: Accuracy-Speed Frontier
# ---------------------------------------------------------------------------

def plot_frontier(results: list[AggregatedResult], task_name: str, condition: str = "A"):
    errs, speedups, labels = accuracy_speed_frontier(results, task_name, condition)
    if len(errs) == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(errs, speedups, s=60, zorder=3)
    for x, y, lbl in zip(errs, speedups, labels):
        ax.annotate(lbl, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)

    ax.set_xlabel("Approximation error $\\epsilon_k$ (L2 pos, meters)")
    ax.set_ylabel("Speedup vs M2 baseline")
    ax.set_title(f"Accuracy–Speed Frontier\n{TASK_LABELS.get(task_name, task_name)} — Condition {condition}")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Baseline (M2)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    out = FIG_DIR / f"frontier_{task_name}_cond{condition}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Success Rate Heatmap
# ---------------------------------------------------------------------------

def plot_success_heatmap(results: list[AggregatedResult], condition: str = "A"):
    model_labels = sorted(set(r.model_label for r in results))
    task_names   = TASK_ORDER

    mat = build_results_table(results, task_names, model_labels, condition, "success_rate")

    fig, ax = plt.subplots(figsize=(len(task_names) * 1.8 + 1, len(model_labels) * 0.5 + 1))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Success Rate")

    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in task_names], rotation=20, ha="right")
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=8)
    ax.set_title(f"Success Rate — Condition {condition}")

    # Annotate cells
    for i in range(len(model_labels)):
        for j in range(len(task_names)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.4 or v > 0.8 else "black")

    out = FIG_DIR / f"success_heatmap_cond{condition}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Condition A vs B delta
# ---------------------------------------------------------------------------

def plot_condition_delta(results: list[AggregatedResult]):
    """Bar chart: success_rate(A) - success_rate(B) per model, for each task."""
    model_labels = sorted(set(r.model_label for r in results))

    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(4 * len(TASK_ORDER), 4), sharey=True)

    for ax, task_name in zip(axes, TASK_ORDER):
        deltas = []
        for ml in model_labels:
            sr_a = next((r.success_rate for r in results
                         if r.model_label == ml and r.task_name == task_name and r.condition == "A"), np.nan)
            sr_b = next((r.success_rate for r in results
                         if r.model_label == ml and r.task_name == task_name and r.condition == "B"), np.nan)
            deltas.append(sr_a - sr_b)

        colors = ["#2196F3" if d >= 0 else "#FF5722" for d in deltas]
        ax.bar(range(len(model_labels)), deltas, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(TASK_LABELS.get(task_name, task_name), fontsize=9)
        ax.set_ylim(-0.5, 0.5)

    axes[0].set_ylabel("$\\Delta$ Success Rate (A − B)")
    fig.suptitle("Condition A vs B: Effect of Sample Count vs Model Speed", fontsize=10)
    out = FIG_DIR / "condition_ab_delta.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: Success rate vs. contact complexity
# ---------------------------------------------------------------------------

def plot_complexity_curve(results: list[AggregatedResult], condition: str = "A"):
    """Line plot: x=task (ordered by complexity), y=success_rate, one line per model."""
    model_labels = sorted(set(r.model_label for r in results))
    x = np.arange(len(TASK_ORDER))

    fig, ax = plt.subplots(figsize=(6, 4))
    for ml in model_labels:
        y = [
            next((r.success_rate for r in results
                  if r.model_label == ml and r.task_name == t and r.condition == condition), np.nan)
            for t in TASK_ORDER
        ]
        ax.plot(x, y, marker="o", label=ml, linewidth=1.5, markersize=5)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in TASK_ORDER])
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Success Rate vs. Contact Complexity — Condition {condition}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    out = FIG_DIR / f"complexity_curve_cond{condition}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", type=str)
    args = parser.parse_args()

    results = load_results(args.results_json)
    print(f"Loaded {len(results)} aggregated results.")

    for task in TASK_ORDER:
        plot_frontier(results, task, "A")
        plot_frontier(results, task, "B")

    plot_success_heatmap(results, "A")
    plot_success_heatmap(results, "B")
    plot_condition_delta(results)
    plot_complexity_curve(results, "A")
    plot_complexity_curve(results, "B")

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
