#!/usr/bin/env python3
"""Plot the contact-parameter sweep (``contact_param_sweep.py`` output) as regret heatmaps and a ranking.

One figure per swept cell, read from ``sweep/<cell>.sweep.json``: the mean
regret over the analysed steps at each grid point, the first swept parameter
across and the second up. A model with a third parameter (M6's refinement) gets
one panel per value of it, all on one colour scale. Regret is a signed
difference of free energies (``F_A(U_B(θ)) - F_A(U_A)``), so the scale diverges
from zero: red where planning with θ cost more than the zero-gap plan, blue
where it scored better (sampling noise can do that at a step), gray near zero.
The scale runs from the lowest mean shown to the highest, each side of zero to
its own extreme; when every mean has one sign it runs from gray at the
smallest to the full colour at the largest.
The nominal point, the cell's own parameters, is outlined; the lowest mean is
marked with a dot. A grid point the sweep dropped is hatched. Each tick shows
the value and, except for a count (iterations, refinement), its multiple of
the nominal.

A second figure per cell, ``<cell>.rank.png``, ranks the grid points by mean
regret, lowest on the left, each with its 95% interval over the analysed
steps as an error bar and its parameter values as the tick; the nominal point
is drawn in ink and labelled.

Usage:
    python experiments/plot_contact_param_sweep.py <batch dir> [--out <dir>]
    python experiments/plot_contact_param_sweep.py <batch dir>/sweep/<cell>.sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: E402

# Diverging pair, blue <-> red, through a neutral gray at zero.
NEGATIVE, NEUTRAL, POSITIVE = "#2a78d6", "#f0efec", "#e34948"
REGRET_CMAP = LinearSegmentedColormap.from_list("regret", [NEGATIVE, NEUTRAL, POSITIVE])
# Its halves, for a scale whose values are all one sign.
POSITIVE_CMAP = LinearSegmentedColormap.from_list("regret_pos", [NEUTRAL, POSITIVE])
NEGATIVE_CMAP = LinearSegmentedColormap.from_list("regret_neg", [NEGATIVE, NEUTRAL])
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a66", "#fcfcfb"
#: Parameters swept as counts, whose ticks show the value alone (``contact_param_sweep.INT_PARAMS``).
WHOLE_NUMBER_PARAMS = {"xpbd_iterations", "refinement"}


def _tick(value: float, mult: float, show_mult: bool) -> str:
    text = f"{value:.3g}"
    return f"{text}\n×{mult:.3g}" if show_mult and np.isfinite(mult) else text


def colourScale(values: np.ndarray):
    """``(norm, cmap)`` spanning ``values``' range, gray at zero when the range crosses it."""
    lo, hi = (float(values.min()), float(values.max())) if values.size else (-1.0, 1.0)
    if lo < 0 < hi:
        return TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi), REGRET_CMAP
    if lo == hi:  # one value, or all equal: a range around it so the norm is defined
        lo, hi = lo - 0.5, hi + 0.5
    return Normalize(vmin=lo, vmax=hi), (NEGATIVE_CMAP if hi <= 0 else POSITIVE_CMAP)


def plotCell(rec: dict, out: Path) -> Path:
    """One cell's heatmap(s), saved to ``out``; returns the path."""
    names = rec["param_names"]
    if len(names) < 2:
        raise ValueError(f"{rec['name']}: a heatmap needs two swept parameters, the sweep has {names}")
    x_name, y_name, facet = names[0], names[1], (names[2] if len(names) > 2 else None)
    points = rec["points"]
    value = lambda p, n: p[f"p_{n}"]  # noqa: E731
    xs = sorted({value(p, x_name) for p in points} | {d[x_name] for d in rec["dropped"]})
    ys = sorted({value(p, y_name) for p in points} | {d[y_name] for d in rec["dropped"]})
    fs = sorted({value(p, facet) for p in points}) if facet else [None]
    nominal = rec["nominal"]
    mult = lambda n, v: v / nominal[n] if nominal[n] else np.nan  # noqa: E731
    by_mult = {n: n not in WHOLE_NUMBER_PARAMS for n in names}

    # The grids of mean regret, NaN where a point is missing (dropped).
    grids = {f: np.full((len(ys), len(xs)), np.nan) for f in fs}
    for p in points:
        f = value(p, facet) if facet else None
        if p["regret_mean"] is not None:
            grids[f][ys.index(value(p, y_name)), xs.index(value(p, x_name))] = p["regret_mean"]
    finite = np.concatenate([g[np.isfinite(g)] for g in grids.values()])
    norm, cmap = colourScale(finite)
    best = min((p for p in points if p["regret_mean"] is not None), key=lambda p: p["regret_mean"])

    # Panels grow with the grid so a finer sweep keeps its tick labels apart.
    n, w, h = len(fs), max(3.6, 0.6 * len(xs)), max(3.4, 0.5 * len(ys))
    fig, axes = plt.subplots(1, n, figsize=(1.0 + w * n, 1.0 + h), squeeze=False, facecolor=SURFACE,
                             constrained_layout=True)
    for ax, f in zip(axes[0], fs):
        g = grids[f]
        ax.set_facecolor(SURFACE)
        # Cells as patches with a 2px surface gap, so neighbours never merge.
        for i in range(len(ys)):
            for j in range(len(xs)):
                v = g[i, j]
                kw = (dict(facecolor=cmap(norm(v))) if np.isfinite(v)
                      else dict(facecolor=SURFACE, hatch="///", edgecolor=MUTED))
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2,
                                           **{"edgecolor": SURFACE, **kw}))
        here = lambda p: f is None or value(p, facet) == f  # noqa: E731
        nom = points[rec["nominal_index"]]
        if here(nom):
            j, i = xs.index(value(nom, x_name)), ys.index(value(nom, y_name))
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=INK, linewidth=2))
            ax.annotate("nominal", (j, i + 0.32), ha="center", va="center", fontsize=7, color=INK)
        if here(best):
            j, i = xs.index(value(best, x_name)), ys.index(value(best, y_name))
            ax.plot(j, i, "o", ms=8, mfc=INK, mec=SURFACE, mew=2)
            ax.annotate(f"lowest\n{best['regret_mean']:.3g}", (j, i - 0.3), ha="center", va="center",
                        fontsize=7, color=INK)
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_ylim(-0.5, len(ys) - 0.5)
        ax.set_xticks(range(len(xs)), [_tick(v, mult(x_name, v), by_mult[x_name]) for v in xs], fontsize=8,
                      color=MUTED)
        ax.set_yticks(range(len(ys)), [_tick(v, mult(y_name, v), by_mult[y_name]) for v in ys], fontsize=8,
                      color=MUTED)
        ax.set_xlabel(x_name, color=INK)
        ax.set_ylabel(y_name, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        if facet:
            ax.set_title(f"{facet} = {f:g}" + (" (nominal)" if f == nominal[facet] else ""), fontsize=10,
                         color=INK)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes[0].tolist(), shrink=0.85, pad=0.02)
    cb.set_label("mean regret  F_A(U_B(θ)) − F_A(U_A)", color=INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    eps = rec["episodes"]
    fig.suptitle(f"{rec['name']}\n{rec['rollout_model']} rollout vs {rec['eval_sim']} eval, "
                 f"{len(eps)} episode{'s' if len(eps) != 1 else ''}, {rec['n_steps_analyzed']} steps",
                 color=INK, fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def plotRank(rec: dict, out: Path) -> Path:
    """The grid points ranked by mean regret, with 95% error bars, saved to ``out``; returns the path."""
    names = rec["param_names"]
    points = sorted((p for p in rec["points"] if p["regret_mean"] is not None), key=lambda p: p["regret_mean"])
    x = np.arange(1, len(points) + 1)
    mean = np.array([p["regret_mean"] for p in points])
    err = np.array([[p["regret_mean"] - p["regret_mean_lo"], p["regret_mean_hi"] - p["regret_mean"]]
                    for p in points]).T
    nominal = np.array([p["is_nominal"] for p in points])

    fig, ax = plt.subplots(figsize=(max(6.0, 1.5 + 0.13 * len(points)), 4.6), facecolor=SURFACE,
                           constrained_layout=True)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8, zorder=0)
    for sel, colour in ((~nominal, NEGATIVE), (nominal, INK)):
        ax.errorbar(x[sel], mean[sel], yerr=err[:, sel], fmt="o", ms=4.5, color=colour, ecolor=colour,
                    elinewidth=1.2, capsize=0, mec=SURFACE, mew=1, zorder=3)
    for i in np.where(nominal)[0]:
        ax.annotate(f"nominal\n{mean[i]:.3g}", (x[i], mean[i] + err[1, i]), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xlim(0.3, len(points) + 0.7)
    lo, hi = min(0.0, float((mean - err[0]).min())), float((mean + err[1]).max())
    ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.set_xticks(x, [" / ".join(f"{p[f'p_{n}']:.3g}" for n in names) for p in points], rotation=90,
                  fontsize=6, color=MUTED)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.tick_params(length=0)
    ax.set_xlabel(f"grid point ({' / '.join(names)}), lowest mean regret first", color=INK)
    ax.set_ylabel("mean regret  F_A(U_B(θ)) − F_A(U_A)\nwith 95% interval", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    eps = rec["episodes"]
    ax.set_title(f"{rec['name']}: grid points ranked by regret\n{rec['rollout_model']} rollout vs "
                 f"{rec['eval_sim']} eval, {len(eps)} episode{'s' if len(eps) != 1 else ''}, "
                 f"{rec['n_steps_analyzed']} steps", color=INK, fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="a batch folder (every sweep/*.sweep.json) or one sweep.json")
    p.add_argument("--out", type=Path, default=None,
                   help="the folder for the PNGs (default: <batch>/sweep/plots, or beside the json)")
    args = p.parse_args(argv)
    if args.source.is_dir():
        files = sorted((args.source / "sweep").glob("*.sweep.json"))
        out_dir = args.out or args.source / "sweep" / "plots"
    else:
        files = [args.source]
        out_dir = args.out or args.source.parent / "plots"
    if not files:
        p.error(f"no sweep/*.sweep.json under {args.source}")
    for f in files:
        rec = json.loads(f.read_text())
        print(plotCell(rec, out_dir / f"{rec['name']}.regret.png"))
        print(plotRank(rec, out_dir / f"{rec['name']}.rank.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
