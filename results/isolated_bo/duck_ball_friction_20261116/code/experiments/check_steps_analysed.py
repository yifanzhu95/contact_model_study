#!/usr/bin/env python3
"""Would fewer analysed steps per episode do? Recompute each cell's metrics from every k-th one.

``process_batch_results.py`` plans every analysed step three times, so its cost
grows with ``--n-steps-analysed`` (``N_STEPS_ANALYSED`` in
``hpc/process_batch_results.slurm``). Episodes differ far more than the steps
within one do, though, so the episodes, not the steps, set most of a cell's
uncertainty. This script takes a batch that is already analysed and, with
no GPU, keeps every k-th analysed step of each episode (the analysed steps are
evenly spaced, so that is about what ``--n-steps-analysed N/k`` would give) and
recomputes each cell's metrics from them::

    python experiments/check_steps_analysed.py results/episode_batches_final_123
    python experiments/check_steps_analysed.py <batch dir> --every 2 3 4 --metrics kl_used_opt regret

For every cell, metric and k it reports:

* ``mean``: the cell's mean over the steps kept, pooled, as
  ``final_results.csv`` reports it;
* ``ep_half``: the half-width of the 95% interval of the cell's mean taken
  over its *episodes* (a t-interval over the episodes' means), the honest
  uncertainty when steps are correlated, and ``step_half``, the step-level
  one ``final_results.csv`` gives, for comparison;
* ``shift``: how far ``mean`` moved from k = 1, in units of k = 1's
  ``ep_half``;
* ``null_p95``: what ``|shift|`` would be by chance alone. Each episode
  instead keeps a random set of as many steps as every k-th gives,
  ``--null-draws`` times; ``null_p95`` is the 95th percentile of those
  ``|shift|`` values, and ``null_frac_ge`` is the fraction of draws at least
  as far as the real ``|shift|`` (a p-value);
* ``width_ratio``: ``ep_half`` over k = 1's;
* ``lag1``: the median over the episodes of the lag-1 autocorrelation of the
  steps kept: near 1, neighbouring steps still say much the same thing.

The printed table gives, per metric and k, the largest ``|shift|`` over the
cells, how many cells' ``|shift|`` passes their ``null_p95``, and the median
``width_ratio`` and ``lag1``; the full table goes to ``--out`` (default
``<batch dir>/steps_analysed_check.csv``).

**Reading it.** Dropping steps does two things:

1. It costs precision: ``width_ratio``. Whether that is worth half the GPU
   time is a judgement call. On the Nov 2026 batch, k = 2 widened the
   intervals by 3.5-9%.
2. It could bias the means, if the steps kept differ systematically from
   the steps dropped. Random subsampling cannot do that, so compare
   ``|shift|`` with ``null_p95``, not with a fixed threshold: dropping half
   the steps at random already moves a mean by about 0.1 half-widths, and
   by up to ~0.65 in 1 cell in 20. With no bias, about 5% of cells pass
   their ``null_p95`` by chance. Many more than that means every k-th step
   is not a fair sample: keep ``N_STEPS_ANALYSED``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import process_batch_results as pbr  # noqa: E402
import run_episode_batches as batches  # noqa: E402

DEFAULT_METRICS = ("kl_used_opt", "kl_opt_used", "regret", "fsim_obj_pos_err", "fsim_obj_rot_err",
                   "eval_used_vs_opt_obj_pos_err")


def _half(lo, hi) -> float:
    return float("nan") if lo is None or hi is None else (hi - lo) / 2


def lag1(values: np.ndarray) -> float:
    """Lag-1 autocorrelation of an episode's steps, over the consecutive pairs both finite; NaN below 3."""
    v = np.asarray(values, dtype=float)
    a, b = v[:-1], v[1:]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def subsampled(episodes: dict[str, np.ndarray], k: int) -> dict:
    """One metric's summary over every ``k``-th analysed step of each episode."""
    kept = {i: np.asarray(v, dtype=float).ravel()[::k] for i, v in episodes.items()}
    pooled = np.concatenate(list(kept.values())) if kept else np.empty(0)
    ep_means = [float(np.mean(v[np.isfinite(v)])) for v in kept.values() if np.isfinite(v).any()]
    mean, step_lo, step_hi = batches.meanInterval(pooled)
    _, ep_lo, ep_hi = batches.meanInterval(ep_means)
    lags = [x for x in (lag1(v) for v in kept.values()) if np.isfinite(x)]
    return {"k": k, "n_steps": int(np.isfinite(pooled).sum()), "n_episodes": len(ep_means),
            "mean": mean, "ep_half": _half(ep_lo, ep_hi), "step_half": _half(step_lo, step_hi),
            "lag1": float(np.median(lags)) if lags else float("nan")}


def nullShifts(episodes: dict[str, np.ndarray], k: int, base_mean: float, base_half: float,
               rng: np.random.Generator, n_draws: int) -> np.ndarray:
    """``|shift|`` when each episode keeps a random set of as many steps as every ``k``-th gives, ``n_draws`` times.

    What the shift of every k-th step looks like by chance alone: these subsets are unbiased by construction.
    """
    arrays = [np.asarray(v, dtype=float).ravel() for v in episodes.values()]
    sizes = [len(v[::k]) for v in arrays]
    out = np.empty(n_draws)
    for d in range(n_draws):
        kept = np.concatenate([v[rng.choice(len(v), size=s, replace=False)] for v, s in zip(arrays, sizes)])
        kept = kept[np.isfinite(kept)]
        out[d] = abs(kept.mean() - base_mean) / base_half if len(kept) else np.nan
    return out


def checkCell(per_episode: dict[str, dict[str, np.ndarray]], metrics, ks, n_draws: int = 200,
              seed: int = 0) -> list[dict]:
    """Rows for one cell: each metric at k = 1 and at each of ``ks``, with its shift, its shift's
    chance distribution (``nullShifts``) and the width ratio."""
    rng = np.random.default_rng(seed)
    rows = []
    for metric in metrics:
        episodes = {i: m[metric] for i, m in per_episode.items() if metric in m}
        if not episodes:
            continue
        base = subsampled(episodes, 1)
        ok_base = base["mean"] is not None and base["ep_half"] > 0
        for k in [1, *ks]:
            r = base if k == 1 else subsampled(episodes, k)
            ok = ok_base and r["mean"] is not None
            shift = (r["mean"] - base["mean"]) / base["ep_half"] if ok else float("nan")
            null_p95 = null_ge = float("nan")
            if ok and k > 1 and n_draws > 0:
                null = nullShifts(episodes, k, base["mean"], base["ep_half"], rng, n_draws)
                null = null[np.isfinite(null)]
                if len(null):
                    null_p95 = float(np.percentile(null, 95))
                    null_ge = float(np.mean(null >= abs(shift)))
            rows.append({"metric": metric, **r, "shift": shift, "null_p95": null_p95, "null_frac_ge": null_ge,
                         "width_ratio": r["ep_half"] / base["ep_half"] if base["ep_half"] > 0 else float("nan")})
    return rows


def checkBatch(batch_dir: Path, metrics=DEFAULT_METRICS, ks=(2, 4), n_draws: int = 200,
               seed: int = 0) -> list[dict]:
    """``checkCell`` of every cell of ``batch_dir`` whose analysis is saved (its ``analysis.npz``)."""
    rows = []
    for cell in pbr.listCells(batch_dir):
        npz = pbr._analysisPaths(batch_dir, cell["name"])["npz"]
        if npz.exists():
            rows += [{"cell": cell["index"], "name": cell["name"], **r}
                     for r in checkCell(pbr.perEpisodeMetrics(npz), metrics, ks, n_draws, seed + cell["index"])]
    return rows


def _nanmedian(values) -> float:
    v = [x for x in values if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def printSummary(rows: list[dict]) -> None:
    """Per metric and k: the largest |shift| over the cells, how many cells' |shift| passes their chance
    95th percentile, and the median width ratio and lag-1."""
    print(f"{'metric':<30} {'k':>3} {'steps/cell':>10} {'max |shift|':>12} {'past null p95':>14} "
          f"{'width ratio':>12} {'lag-1':>7}")
    for metric in dict.fromkeys(r["metric"] for r in rows):
        for k in dict.fromkeys(r["k"] for r in rows if r["metric"] == metric):
            sel = [r for r in rows if r["metric"] == metric and r["k"] == k]
            shifts = [abs(r["shift"]) for r in sel if np.isfinite(r["shift"])]
            tested = [r for r in sel if np.isfinite(r["shift"]) and np.isfinite(r["null_p95"])]
            past = f"{sum(abs(r['shift']) > r['null_p95'] for r in tested)}/{len(tested)}" if tested else "-"
            print(f"{metric:<30} {k:>3} {_nanmedian(r['n_steps'] for r in sel):>10.0f} "
                  f"{max(shifts) if shifts else float('nan'):>12.3f} {past:>14} "
                  f"{_nanmedian(r['width_ratio'] for r in sel):>12.3f} {_nanmedian(r['lag1'] for r in sel):>7.2f}")
    print("\npast null p95: cells whose |shift| is beyond what dropping steps at random gives 95% of the time; "
          "with no bias, about 1 in 20.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch_dir", type=Path, help="a batch results folder already analysed by process_batch_results.py")
    p.add_argument("--every", type=int, nargs="+", default=[2, 4], metavar="K",
                   help="keep every K-th analysed step (default 2 4); K = 1, all of them, is the baseline")
    p.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS),
                   help=f"per-step metrics to check (default {' '.join(DEFAULT_METRICS)})")
    p.add_argument("--out", type=Path, default=None, help="the full table; default <batch dir>/steps_analysed_check.csv")
    p.add_argument("--null-draws", type=int, default=200,
                   help="random subsets per cell, metric and k for the chance distribution of the shift (default "
                        "200; 0 skips it)")
    p.add_argument("--seed", type=int, default=0, help="seed of the random subsets (default 0)")
    args = p.parse_args(argv)
    if any(k < 2 for k in args.every):
        p.error("--every values must be >= 2")
    if args.null_draws < 0:
        p.error("--null-draws must be >= 0")
    batch_dir = args.batch_dir.resolve()
    if not batch_dir.is_dir():
        p.error(f"no such folder: {batch_dir}")
    rows = checkBatch(batch_dir, args.metrics, sorted(set(args.every)), args.null_draws, args.seed)
    if not rows:
        p.error(f"{batch_dir} has no analysed cells (analysis/<cell>.analysis.npz) with metrics {args.metrics}")
    out = args.out or batch_dir / "steps_analysed_check.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len({r['name'] for r in rows})} analysed cells of {batch_dir.name}\n")
    printSummary(rows)
    print(f"\nfull table -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
