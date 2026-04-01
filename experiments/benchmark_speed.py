"""experiments/benchmark_speed.py

Standalone speed benchmark: measures rollout throughput (steps/sec) for
every contact model variant across a range of batch sizes.

Produces:
  - results/speed_benchmark_{timestamp}.json
  - figures/speed_vs_batch.pdf

Usage:
    python experiments/benchmark_speed.py \
        --task push \
        --models M1 M2 M3 M4 \
        --batch_sizes 64 128 256 512 1024 2048 4096 \
        --horizon 50
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import warp as wp

import contact_study.tasks.tasks  # noqa: F401  (register tasks)
from contact_study.contact_models.config import ContactModelConfig, GeometryVariant
from contact_study.contact_models import api
from contact_study.tasks.base import get_task

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
    "M4d": lambda: ContactModelConfig.M4(damping_friction=True),
}


def benchmark_one(
    mjm:         mujoco.MjModel,
    cfg:         ContactModelConfig,
    batch_size:  int,
    horizon:     int,
    n_warmup:    int = 3,
    n_trials:    int = 10,
) -> dict:
    """Return timing stats for one (cfg, batch_size) combination."""
    m = api.put_model(mjm, cfg, nworld=batch_size)
    d = api.make_data(mjm, m)

    def _run():
        api.reset_data(mjm, m, d)
        for _ in range(horizon):
            api.step(m, d)
        wp.synchronize()

    for _ in range(n_warmup):
        _run()

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _run()
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    steps_per_sec = batch_size * horizon / times

    return {
        "label":              cfg.label,
        "batch_size":         batch_size,
        "horizon":            horizon,
        "mean_ms":            float(np.mean(times) * 1e3),
        "std_ms":             float(np.std(times) * 1e3),
        "mean_steps_per_sec": float(np.mean(steps_per_sec)),
        "std_steps_per_sec":  float(np.std(steps_per_sec)),
    }


def plot_speed(records: list[dict], out_path: Path):
    """Line plot: batch_size vs steps/sec per model."""
    labels = sorted(set(r["label"] for r in records))
    fig, ax = plt.subplots(figsize=(6, 4))

    for label in labels:
        subset = sorted([r for r in records if r["label"] == label], key=lambda r: r["batch_size"])
        xs = [r["batch_size"] for r in subset]
        ys = [r["mean_steps_per_sec"] / 1e6 for r in subset]
        es = [r["std_steps_per_sec"] / 1e6 for r in subset]
        ax.errorbar(xs, ys, yerr=es, marker="o", label=label, linewidth=1.8, capsize=3)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size (parallel worlds)")
    ax.set_ylabel("Throughput (M steps / sec)")
    ax.set_title("Contact Model Throughput vs. Batch Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",        default="push")
    parser.add_argument("--models",      nargs="+", default=["M1", "M2", "M3", "M4"])
    parser.add_argument("--batch_sizes", nargs="+", type=int,
                        default=[64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--horizon",     type=int, default=50)
    parser.add_argument("--n_warmup",    type=int, default=3)
    parser.add_argument("--n_trials",    type=int, default=10)
    args = parser.parse_args()

    task = get_task(args.task)
    mjm, _ = task.load()

    records = []
    for model_name in args.models:
        cfg = MODEL_FACTORIES[model_name]()
        for bs in args.batch_sizes:
            print(f"  {model_name} | batch={bs} ...", end=" ", flush=True)
            rec = benchmark_one(mjm, cfg, bs, args.horizon, args.n_warmup, args.n_trials)
            print(f"{rec['mean_steps_per_sec']/1e6:.2f} M steps/sec")
            records.append(rec)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / f"speed_benchmark_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {json_path}")

    FIGURES_DIR.mkdir(exist_ok=True)
    plot_speed(records, FIGURES_DIR / f"speed_vs_batch_{ts}.pdf")


if __name__ == "__main__":
    main()
