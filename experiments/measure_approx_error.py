"""experiments/measure_approx_error.py

Measures approximation error epsilon_k = d(Mk, M*) for each contact model
relative to the M2 baseline (MuJoCo soft contact), across a grid of test
states and rollout horizons.

Produces:
  - results/approx_error_{timestamp}.json
  - figures/error_vs_horizon.pdf   (error growth over planning horizon)
  - figures/error_heatmap.pdf      (task × model heatmap at H=20)

Usage:
    python experiments/measure_approx_error.py \
        --tasks push grasp_reorient peg_in_hole \
        --models M1 M3 M4 M5 M7 \
        --horizons 5 10 20 40 \
        --n_states 50
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import contact_study.tasks.tasks  # noqa: F401
from contact_study.contact_models.benchmarks import measure_approximation_error
from contact_study.contact_models.config import ContactModelConfig
from contact_study.tasks.base import get_task

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
    "M4d": lambda: ContactModelConfig.M4(damping_friction=True),
    "M5": ContactModelConfig.M5,
    "M6": ContactModelConfig.M6,
    "M7": ContactModelConfig.M7,
    "M8": ContactModelConfig.M8,
    "M9": ContactModelConfig.M9,
    "M10": ContactModelConfig.M10,
}

TASK_LABELS = {
    "push":           "Push (Low)",
    "grasp_reorient": "Grasp/Reorient (Med)",
    "peg_in_hole":    "Peg-in-Hole (High)",
}


def plot_error_vs_horizon(records: list[dict], out_path: Path):
    """One subplot per task, lines per model."""
    tasks   = sorted(set(r["task"] for r in records))
    models  = sorted(set(r["model"] for r in records if r["model"] != "M2"))
    horizons = sorted(set(r["horizon"] for r in records))

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4), sharey=False)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        for model in models:
            subset = sorted(
                [r for r in records if r["task"] == task and r["model"] == model],
                key=lambda r: r["horizon"],
            )
            if not subset:
                continue
            xs  = [r["horizon"] for r in subset]
            ys  = [r["mean_err"] for r in subset]
            yes = [r["std_err"] for r in subset]
            ax.errorbar(xs, ys, yerr=yes, marker="o", label=model, linewidth=1.5, capsize=3)

        ax.set_xlabel("Rollout horizon H (steps)")
        ax.set_ylabel("Mean L2 pos error (m)")
        ax.set_title(TASK_LABELS.get(task, task))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Approximation Error vs. Horizon", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_error_heatmap(records: list[dict], horizon: int, out_path: Path):
    """Model × task heatmap at a fixed horizon."""
    tasks  = ["push", "grasp_reorient", "peg_in_hole"]
    models = sorted(set(r["model"] for r in records if r["model"] != "M2"))

    mat = np.full((len(models), len(tasks)), np.nan)
    idx = {(r["model"], r["task"]): r for r in records if r["horizon"] == horizon}
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            key = (m, t)
            if key in idx:
                mat[i, j] = idx[key]["mean_err"]

    fig, ax = plt.subplots(figsize=(len(tasks) * 2 + 1, len(models) * 0.6 + 1))
    vmax = np.nanpercentile(mat, 95)
    im = ax.imshow(mat, vmin=0, vmax=vmax, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean L2 pos error (m)")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], rotation=20, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_title(f"Approximation Error (H={horizon})")

    for i in range(len(models)):
        for j in range(len(tasks)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",    nargs="+",
                        default=["push", "grasp_reorient", "peg_in_hole"])
    parser.add_argument("--models",   nargs="+",
                        default=["M1", "M3", "M4", "M5", "M7"])
    parser.add_argument("--horizons", nargs="+", type=int,
                        default=[5, 10, 20, 40])
    parser.add_argument("--n_states", type=int, default=50)
    args = parser.parse_args()

    rng     = np.random.default_rng(0)
    records = []
    cfg_gt  = ContactModelConfig.M2()

    for task_name in args.tasks:
        task      = get_task(task_name)
        mjm, _    = task.load()
        test_states = np.stack([
            np.concatenate(task.sample_initial_state(rng))
            for _ in range(args.n_states)
        ])

        for model_name in args.models:
            if model_name == "M2":
                continue
            cfg_ap = MODEL_FACTORIES[model_name]()

            for H in args.horizons:
                print(f"  {task_name} | {model_name} | H={H} ...", end=" ", flush=True)
                mean_err, std_err = measure_approximation_error(
                    mjm, cfg_gt, cfg_ap, test_states, horizon=H
                )
                print(f"err={mean_err:.4f} ± {std_err:.4f}")
                records.append({
                    "task":     task_name,
                    "model":    model_name,
                    "horizon":  H,
                    "mean_err": mean_err,
                    "std_err":  std_err,
                })

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / f"approx_error_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {json_path}")

    FIGURES_DIR.mkdir(exist_ok=True)
    plot_error_vs_horizon(records, FIGURES_DIR / f"error_vs_horizon_{ts}.pdf")
    plot_error_heatmap(records, horizon=20, out_path=FIGURES_DIR / f"error_heatmap_{ts}.pdf")


if __name__ == "__main__":
    main()
