"""plot_n_samples_sweep_dir.py

Directory version of plot_n_samples_sweep.py. Instead of one JSON file, this
reads a whole directory of num-rollout sweep JSONs — e.g. the per-replicate
files written by experiments/hpc/num_rollout_eval.slurm
(results/num_rollout_eval_<id>/sweep_rep*.json) — and merges every replicate's
records for the same (model, n_samples) cell before plotting.

Merge rule: each cell's stats are pooled across files weighted by that file's
n_episodes. The success-rate whisker (±1 SE) is recomputed from the pooled
success rate and total episode count (binomial SE = sqrt(p(1-p)/N)); step-time
mean/SD are episode-weighted averages.

The plotting itself reuses plot_n_samples_sweep.py so the figure is identical.

Usage:
    python analysis/plot_n_samples_sweep_dir.py                        # latest num_rollout_eval_* dir
    python analysis/plot_n_samples_sweep_dir.py results/num_rollout_eval_12345
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Reuse the loader + plotting helpers from the single-file script (same dir).
sys.path.insert(0, str(Path(__file__).parent))
import plot_n_samples_sweep as base  # noqa: E402

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _latest_dir() -> Path:
    dirs = sorted(d for d in RESULTS_DIR.glob("num_rollout_eval_*") if d.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"No num_rollout_eval_* directories found in {RESULTS_DIR}. "
            f"Pass a directory explicitly."
        )
    return dirs[-1]


def merge_dir(directory: Path) -> list[dict]:
    """Pool every *.json in *directory* into one record per model_label.

    Returns records with the same fields plot_n_samples_sweep.parse_records
    reads: model_label, task_name, condition, n_episodes, success_rate,
    success_rate_se, mean_step_ms, std_step_ms.
    """
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No *.json files found in {directory}")

    # label -> episode-weighted accumulators
    acc: dict[str, dict] = {}
    for path in files:
        for r in base.load(path):
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


def main():
    parser = argparse.ArgumentParser(
        description="Plot a whole directory of num-rollout sweep JSONs (merged)."
    )
    parser.add_argument(
        "results_dir", nargs="?", type=Path,
        help="Directory of sweep JSONs. Defaults to latest "
             "num_rollout_eval_* dir in results/",
    )
    args = parser.parse_args()

    directory = args.results_dir or _latest_dir()
    print(f"Loading directory: {directory}")

    records, files = merge_dir(directory)
    print(f"  Merged {len(files)} file(s)")

    models, n_values, data = base.parse_records(records)
    if not models:
        raise ValueError("No parseable records found (expected labels like M1_n64).")

    print(f"  Models  : {models}")
    print(f"  n values: {n_values}")

    task_name = records[0].get("task_name", "unknown") if records else "unknown"
    condition = records[0].get("condition", "")
    title     = f"Success rate vs. rollouts — {task_name}"
    if condition:
        title += f"  (condition {condition})"

    out_path = directory / f"{directory.name}_plot.pdf"
    base.plot(models, n_values, data, title, out_path)


if __name__ == "__main__":
    main()
