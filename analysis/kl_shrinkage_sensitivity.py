"""Recompute Gaussian KL on recorded, fixed weighted first-action moments.

Run after a worker cell with --record_kl_moments. This is an estimator
sensitivity diagnostic: no new planning, simulation, or candidate sampling.
Separate scientific families remain in separate plots. SR is not recomputed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_kl_divergence_dir import merge_dir
from contact_study.evaluation.distributions import gaussian_kl


def kl_at_alpha(record: dict, alpha: float, sigma: float) -> tuple[float, float]:
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("shrinkage must be finite and in (0, 1]")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    means, covariances = [], []
    for name in ("reference", "degraded"):
        mean = np.asarray(record[name]["mean"], dtype=float)
        raw = np.asarray(record[name]["covariance_raw"], dtype=float)
        if (mean.ndim != 1 or raw.shape != (mean.size, mean.size)
                or not np.isfinite(mean).all() or not np.isfinite(raw).all()):
            raise ValueError(f"Invalid {name} raw moments")
        means.append(mean)
        covariances.append((1 - alpha) * raw + alpha * sigma ** 2 * np.eye(mean.size))
    if means[0].shape != means[1].shape:
        raise ValueError("Reference and degraded action dimensions differ")
    return (gaussian_kl(means[0], covariances[0], means[1], covariances[1]),
            gaussian_kl(means[1], covariances[1], means[0], covariances[0]))


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(a.mean()) if a.size else None,
            "min": float(a.min()) if a.size else None,
            "max": float(a.max()) if a.size else None}


def analyze_cell(cell: dict, alphas: list[float]) -> dict:
    episode_values = {a: [] for a in alphas}
    diagnostics = {name: {"rank": [], "ess": [], "min_eigenvalue": [],
                          "max_eigenvalue": []} for name in ("reference", "degraded")}
    mean_distances, reproduction_errors = [], []
    source_count = measurement_count = skipped_files = 0
    for source in cell["source_paths"]:
        payload = json.loads(Path(source).read_text())
        omitted = payload.get("artifact", {}).get("omitted_fields", [])
        if (not payload["config"].get("record_kl_moments", False)
                or "per_step.moments" in omitted):
            skipped_files += 1
            continue
        source_count += 1
        sigma = float(payload["config"]["noise_sigma"])
        saved_alpha = float(payload["config"]["kl_shrinkage"])
        for episode in payload["per_step"]:
            moments = episode.get("moments", [])
            if [r["step"] for r in moments] != episode["steps"]:
                raise ValueError(f"{source}: moments and valid KL steps do not align")
            for i, record in enumerate(moments):
                measurement_count += 1
                f, r = kl_at_alpha(record, saved_alpha, sigma)
                saved = [episode["kl_forward"][i], episode["kl_reverse"][i]]
                if not np.allclose([f, r], saved, rtol=1e-10, atol=1e-10):
                    raise ValueError(f"{source}: saved moments do not reproduce KL at step {record['step']}")
                reproduction_errors.append(float(np.max(np.abs(np.asarray([f, r]) - saved))))
                mean_distances.append(float(np.linalg.norm(
                    np.asarray(record["reference"]["mean"]) - record["degraded"]["mean"])))
                for name in diagnostics:
                    raw = np.asarray(record[name]["covariance_raw"])
                    eig = np.linalg.eigvalsh(raw)
                    diagnostics[name]["rank"].append(int(np.linalg.matrix_rank(raw)))
                    diagnostics[name]["ess"].append(float(record[name]["ess"]))
                    diagnostics[name]["min_eigenvalue"].append(float(eig[0]))
                    diagnostics[name]["max_eigenvalue"].append(float(eig[-1]))
            for alpha in alphas:
                values = [kl_at_alpha(record, alpha, sigma)[0] for record in moments]
                if values:
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite sensitivity KL in {source}, alpha={alpha}")
                    episode_values[alpha].append(float(np.mean(values)))
    curve = []
    for alpha, values in episode_values.items():
        curve.append({"alpha": alpha, "episode_kl": values,
                      "mean": float(np.mean(values)) if values else None,
                      "se": float(np.std(values, ddof=1) / np.sqrt(len(values)))
                      if len(values) > 1 else None})
    return {"label": cell["label"], "family_id": cell["family_id"],
            "geometry": cell["geometry"], "null": cell["null"],
            "source_paths": cell["source_paths"], "recorded_files": source_count,
            "files_without_moments": skipped_files, "measurements": measurement_count,
            "curve": curve, "mean_action_distance": summarize(mean_distances),
            "saved_kl_reproduction_abs_error": summarize(reproduction_errors),
            "diagnostics": {name: {k: summarize(v) for k, v in data.items()}
                            for name, data in diagnostics.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0001, 0.001, 0.01, 0.1])
    args = parser.parse_args()
    alphas = sorted(set(args.alphas))
    if not alphas or any(not np.isfinite(a) or not 0 < a <= 1 for a in alphas):
        parser.error("all alphas must be finite and in (0, 1]")
    cells, _ = merge_dir(args.results_dir)
    reports = [analyze_cell(cell, alphas) for cell in cells.values()]
    reports = [report for report in reports if report["measurements"]]
    if not reports:
        raise ValueError("No valid raw moments found; run a worker with --record_kl_moments")
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = {"purpose": "Fixed-particle shrinkage sensitivity; not a new control experiment",
              "weighting": "episode", "direction": "reference || degraded",
              "cells": reports}
    (args.outdir / "shrinkage_sensitivity.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    for family in sorted({r["family_id"] for r in reports}):
        group = [r for r in reports if r["family_id"] == family]
        fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
        for report in group:
            curve = report["curve"]
            label = report["label"].replace(report["geometry"] + "_", "")
            ax.plot([r["alpha"] for r in curve], [r["mean"] for r in curve],
                    "o--" if report["null"] else "o-", label=label)
        ax.set(xscale="log", yscale="log", xlabel="Covariance shrinkage alpha",
               ylabel="Episode-balanced forward KL (nats)",
               title=f"{group[0]['geometry']}: fixed-particle sensitivity\nSame recorded actions and weights at every alpha")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)
        for suffix in ("png", "pdf"):
            fig.savefig(args.outdir / f"shrinkage_{group[0]['geometry']}_{family}.{suffix}", dpi=170)
        plt.close(fig)
    for r in reports:
        print(r["label"], "measurements=", r["measurements"],
              "KL=", [(v["alpha"], v["mean"]) for v in r["curve"]])


if __name__ == "__main__":
    main()
