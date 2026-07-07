"""plot_cntrl_freq_sweep_dir.py

Directory version of plot_cntrl_freq_sweep.py. Instead of one JSON file, this
reads a whole directory of control-frequency sweep JSONs — e.g. the per-cell
files written by experiments/hpc/cntrl_freq_eval.slurm
(results/cntrl_freq_eval_<id>/<model>_sub<n>.json) — and merges every cell's
records for the same (model, substeps) label before plotting.

Merge rule: each label's stats are pooled across files weighted by that file's
n_episodes. The success-rate whisker (±1 SE) is recomputed from the pooled
success rate and total episode count (binomial SE = sqrt(p(1-p)/N)); step-time
mean/SD are episode-weighted averages.

Control frequency uses the exact formula the HPC cell worker computes:
    control_freq = 1 / (eval_dt * eval_substeps_per_rollout * substeps)
eval_dt and eval_substeps_per_rollout are read from the directory's
meta.json (written by run_cntrl_freq_cell.py); pass --dt to override with a
literal effective dt (= eval_dt * eval_substeps_per_rollout) if meta.json is
missing.

The plotting itself reuses plot_cntrl_freq_sweep.py so the figure is
identical.

Usage:
    python analysis/plot_cntrl_freq_sweep_dir.py                        # latest cntrl_freq_eval_* dir
    python analysis/plot_cntrl_freq_sweep_dir.py results/cntrl_freq_eval_12345
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Reuse the loader + plotting helpers from the single-file script (same dir).
sys.path.insert(0, str(Path(__file__).parent))
import plot_cntrl_freq_sweep as base  # noqa: E402

RESULTS_DIR = Path(__file__).parent.parent / "results"

DEFAULT_DT = 0.005   # fallback effective dt (eval_dt * eval_substeps_per_rollout) if meta.json is missing


def _latest_dir() -> Path:
    dirs = sorted(d for d in RESULTS_DIR.glob("cntrl_freq_eval_*") if d.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"No cntrl_freq_eval_* directories found in {RESULTS_DIR}. "
            f"Pass a directory explicitly."
        )
    return dirs[-1]


def _load_effective_dt(directory: Path, dt_override: float | None) -> float:
    """Return eval_dt * eval_substeps_per_rollout from meta.json, or the override."""
    if dt_override is not None:
        return dt_override
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        print(f"  Warning: no meta.json in {directory}; falling back to dt={DEFAULT_DT}")
        return DEFAULT_DT
    with open(meta_path) as f:
        meta = json.load(f)
    return float(meta["eval_dt"]) * float(meta["eval_substeps_per_rollout"])


def merge_dir(directory: Path) -> tuple[list[dict], list[Path]]:
    """Pool every *.json (excluding meta.json) in *directory* into one record
    per model_label.

    Returns records with the same fields plot_cntrl_freq_sweep.parse_records
    reads: model_label, task_name, condition, n_episodes, success_rate,
    success_rate_se, mean_step_ms, std_step_ms.
    """
    files = sorted(p for p in directory.glob("*.json") if p.name != "meta.json")
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
        description="Plot a whole directory of control-frequency sweep JSONs (merged)."
    )
    parser.add_argument(
        "results_dir", nargs="?", type=Path,
        help="Directory of sweep JSONs. Defaults to latest "
             "cntrl_freq_eval_* dir in results/",
    )
    parser.add_argument(
        "--dt", type=float, default=None,
        help="Effective dt (= eval_dt * eval_substeps_per_rollout) used to convert "
             "substeps -> Hz. Defaults to reading eval_dt / eval_substeps_per_rollout "
             "from the directory's meta.json.",
    )
    args = parser.parse_args()

    directory = args.results_dir or _latest_dir()
    print(f"Loading directory: {directory}")

    dt = _load_effective_dt(directory, args.dt)
    print(f"Using effective dt = {dt*1e3:.3g} ms  →  substeps × dt = control period")

    records, files = merge_dir(directory)
    print(f"  Merged {len(files)} file(s)")

    models, freq_values, substep_map, data = base.parse_records(records, dt)
    if not models:
        raise ValueError("No parseable records found (expected labels like M1_sub5).")

    print(f"  Models      : {models}")
    print(f"  Substeps    : {sorted(substep_map.values())}")

    task_name = records[0].get("task_name", "unknown") if records else "unknown"
    condition = records[0].get("condition", "")
    title     = f"Success rate vs. control frequency — {task_name}"
    if condition:
        title += f"  (condition {condition})"

    out_path = directory / f"{directory.name}_plot.pdf"
    base.plot(models, freq_values, substep_map, data, title, dt, out_path)


if __name__ == "__main__":
    main()
