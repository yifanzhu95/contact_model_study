"""plot_kl_divergence_dir.py

Directory plotter for the planner-approximation-quality sweep — reads a whole
directory of per-cell JSONs written by experiments/hpc/kl_divergence_eval.slurm
(results/kl_divergence_eval_<id>/<model>_n<samples>_i<iters>[_null].json) and
turns them into the figure that sweep exists to produce: success rate vs. the
KL divergence between the reference planner's induced optimal control
distribution and the degraded planner's approximation of it.

Unlike the other sweeps in analysis/, this one is NOT a clustered bar chart of
M1-M4: every cell uses the same contact model, and the swept axis is optimizer
compute (n_samples x n_iterations). So the headline panel is a scatter, one
point per cell, and the model appears only in the title.

Panels
------
Top    — scatter of success rate (%) vs. KL, one point per non-null cell.
         Marker colour = n_samples, marker shape = n_iterations; x-error is the
         SE of the pooled per-step KL mean, y-error the binomial ±1 SE.
Bottom — the null-control check: real KL next to that config's own null KL
         (same settings, different noise seed = the estimator-noise floor).
         A config whose real bar is not clearly above its null bar is not
         reporting a meaningful difference; those cells are drawn hollow in the
         top panel and dropped entirely with --drop_below_null.

Merge rule: files sharing a label (replicate runs of the same cell) are pooled.
Success rate is episode-weighted with the whisker recomputed from the pooled
rate and total episode count (binomial SE = sqrt(p(1-p)/N)); KL mean/SD are
pooled over the per-step sample counts (SD via the pooled second moment, so it
is the spread of all steps together, not an average of SDs).

Caveat on the KL whisker: the per-step KL samples within an episode are
correlated, so sd/sqrt(n) understates the true uncertainty. It is drawn as a
spread indicator, not an inferential interval.

Usage:
    python analysis/plot_kl_divergence_dir.py                          # latest kl_divergence_eval_* dir
    python analysis/plot_kl_divergence_dir.py results/kl_divergence_eval_12345
    python analysis/plot_kl_divergence_dir.py --stat median --direction reverse
    python analysis/plot_kl_divergence_dir.py --drop_below_null
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Marker shape carries n_iterations, colour carries n_samples (assigned from a
# sequential map once the sweep's sample counts are known).
ITER_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}
_FALLBACK_MARKER = "P"

NULL_COLOR = "#BBBBBB"

_LABEL_RE = re.compile(r"^(M\d+)_n(\d+)_i(\d+)(_null)?$")


def _latest_dir() -> Path:
    dirs = sorted(d for d in RESULTS_DIR.glob("kl_divergence_eval_*") if d.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"No kl_divergence_eval_* directories found in {RESULTS_DIR}. "
            f"Pass a directory explicitly."
        )
    return dirs[-1]


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Loading + pooling
# ---------------------------------------------------------------------------
def _pool_stats(acc: dict) -> tuple[float | None, float | None, int]:
    """Collapse a {n, sum, sumsq} accumulator into (mean, sd, n).

    sumsq is accumulated as n*(sd^2 + mean^2) per file, so the pooled SD is the
    spread of every per-step sample together rather than a mean of per-file SDs.
    """
    n = acc["n"]
    if n <= 0:
        return None, None, 0
    mean = acc["sum"] / n
    var  = max(acc["sumsq"] / n - mean * mean, 0.0)
    return mean, math.sqrt(var), n


def merge_dir(directory: Path) -> tuple[dict[str, dict], list[Path]]:
    """Pool every per-cell JSON in *directory* into one entry per label.

    Returns (cells, files) where cells[label] holds the fields the plotting
    needs: model, n_samples, n_iterations, null, task_name, n_episodes,
    success_rate, success_rate_se, mean_step_ms, the two KL directions and the
    ESS diagnostics.
    """
    files = sorted(p for p in directory.glob("*.json") if p.name != "meta.json")
    if not files:
        raise FileNotFoundError(f"No per-cell *.json files found in {directory}")

    acc: dict[str, dict] = {}
    for path in files:
        d = load(path)
        label = d.get("label")
        if label is None:
            continue
        m = _LABEL_RE.match(label)
        if m is None:
            print(f"  ! skipping {path.name}: label {label!r} is not "
                  f"<model>_n<samples>_i<iters>[_null]")
            continue

        cfg = d.get("config", {})
        agg = d.get("aggregate", {})
        a = acc.setdefault(label, {
            "model":        m.group(1),
            "n_samples":    int(m.group(2)),
            "n_iterations": int(m.group(3)),
            "null":         m.group(4) is not None,
            "task_name":    agg.get("task_name", d.get("task", "unknown")),
            "ref_n_samples":    cfg.get("ref_n_samples"),
            "ref_n_iterations": cfg.get("ref_n_iterations"),
            "n_episodes":   0,
            "n_success":    0.0,
            "step_ms":      0.0,
            "series":       {k: {"n": 0, "sum": 0.0, "sumsq": 0.0}
                             for k in ("forward", "reverse",
                                       "ess_ref", "ess_deg", "mu_dist")},
            "files":        0,
        })
        a["files"] += 1

        n_ep = int(agg.get("n_episodes", 0) or 0)
        a["n_episodes"] += n_ep
        a["n_success"]  += float(agg.get("success_rate", 0.0)) * n_ep
        a["step_ms"]    += float(agg.get("mean_step_ms", 0.0)) * n_ep

        sources = {
            "forward":  d.get("kl", {}).get("forward"),
            "reverse":  d.get("kl", {}).get("reverse"),
            "ess_ref":  d.get("diagnostics", {}).get("ess_ref"),
            "ess_deg":  d.get("diagnostics", {}).get("ess_deg"),
            "mu_dist":  d.get("diagnostics", {}).get("mu_dist"),
        }
        for key, s in sources.items():
            if not s or not s.get("n") or s.get("mean") is None:
                continue
            n, mean, sd = int(s["n"]), float(s["mean"]), float(s.get("sd") or 0.0)
            t = a["series"][key]
            t["n"]     += n
            t["sum"]   += mean * n
            t["sumsq"] += n * (sd * sd + mean * mean)

        # Medians cannot be pooled from summaries; recover them from the raw
        # per-step arrays when the cell kept them (it does by default).
        for r in d.get("per_step", []) or []:
            for key, field in (("forward", "kl_forward"), ("reverse", "kl_reverse")):
                vals = r.get(field) or []
                a.setdefault("raw", {}).setdefault(key, []).extend(
                    float(v) for v in vals if np.isfinite(v)
                )

    cells: dict[str, dict] = {}
    for label, a in acc.items():
        n_ep = a["n_episodes"]
        w    = n_ep if n_ep else 1
        sr   = a["n_success"] / w
        cell = {
            "label":        label,
            "model":        a["model"],
            "n_samples":    a["n_samples"],
            "n_iterations": a["n_iterations"],
            "null":         a["null"],
            "task_name":    a["task_name"],
            "ref_n_samples":    a["ref_n_samples"],
            "ref_n_iterations": a["ref_n_iterations"],
            "n_files":      a["files"],
            "n_episodes":   n_ep,
            "success_rate": sr,
            "success_rate_se":
                math.sqrt(max(sr * (1.0 - sr), 0.0) / n_ep) if n_ep > 1 else 0.0,
            "mean_step_ms": a["step_ms"] / w,
        }
        for key in a["series"]:
            mean, sd, n = _pool_stats(a["series"][key])
            cell[key] = {"mean": mean, "sd": sd, "n": n}
        for key, vals in (a.get("raw") or {}).items():
            cell[key]["median"] = float(np.median(vals)) if vals else None
        cells[label] = cell

    return cells, files


def kl_value(cell: dict, direction: str, stat: str) -> tuple[float | None, float]:
    """(value, ±error) for a cell's KL under the chosen direction and statistic.

    The error is the SE of the pooled per-step mean (sd/sqrt(n)); it is zero for
    the median, which has no comparable closed form here.
    """
    s = cell.get(direction) or {}
    if stat == "median":
        return s.get("median"), 0.0
    mean, sd, n = s.get("mean"), s.get("sd") or 0.0, s.get("n") or 0
    return mean, (sd / math.sqrt(n) if n > 1 else 0.0)


def pair_with_null(cells: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Split into (real cells, null cells) and attach each real cell's null
    partner — the same (model, n_samples, n_iterations) run with --null_control.
    """
    reals = [c for c in cells.values() if not c["null"]]
    nulls = [c for c in cells.values() if c["null"]]
    by_key = {(c["model"], c["n_samples"], c["n_iterations"]): c for c in nulls}
    for c in reals:
        c["null_cell"] = by_key.get((c["model"], c["n_samples"], c["n_iterations"]))
    reals.sort(key=lambda c: (c["n_iterations"], c["n_samples"]))
    nulls.sort(key=lambda c: (c["n_iterations"], c["n_samples"]))
    return reals, nulls


def above_null(cell: dict, direction: str, stat: str) -> bool | None:
    """True/False if the cell clears its own estimator-noise floor, None if it
    has no null partner. "Clears" = real KL exceeds the null KL by more than the
    two errors combined; with --stat median (no error bars) it is a plain
    comparison.
    """
    null = cell.get("null_cell")
    if null is None:
        return None
    v,  e  = kl_value(cell, direction, stat)
    nv, ne = kl_value(null, direction, stat)
    if v is None or nv is None:
        return None
    return v - e > nv + ne


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _sample_colors(n_values: list[int]) -> dict[int, tuple]:
    cmap = plt.get_cmap("viridis")
    if len(n_values) == 1:
        return {n_values[0]: cmap(0.35)}
    return {n: cmap(0.12 + 0.76 * i / (len(n_values) - 1))
            for i, n in enumerate(n_values)}


def _scatter_panel(ax, reals, direction, stat, colors, drop_below_null):
    """Success rate vs. KL, one marker per cell. Cells that do not clear their
    null floor are drawn hollow (or omitted when *drop_below_null*)."""
    plotted = 0
    for c in reals:
        v, e = kl_value(c, direction, stat)
        if v is None:
            continue
        ok = above_null(c, direction, stat)
        if ok is False and drop_below_null:
            continue

        color  = colors[c["n_samples"]]
        marker = ITER_MARKERS.get(c["n_iterations"], _FALLBACK_MARKER)
        # Hollow = indistinguishable from this config's own estimator noise.
        face   = color if ok is not False else "none"

        ax.errorbar(
            v, c["success_rate"] * 100.0,
            xerr=e if e > 0 else None,
            yerr=c["success_rate_se"] * 100.0,
            fmt="none", ecolor="#666666", elinewidth=1.0,
            capsize=3.0, capthick=1.0, zorder=3,
        )
        ax.plot(v, c["success_rate"] * 100.0, marker=marker, markersize=9,
                markerfacecolor=face, markeredgecolor=color,
                markeredgewidth=1.8, linestyle="none", zorder=4)
        ax.annotate(
            f"n={c['n_samples']}, i={c['n_iterations']}",
            (v, c["success_rate"] * 100.0),
            textcoords="offset points", xytext=(8, 6),
            fontsize=8, color="#333333",
        )
        plotted += 1

    # Log x: KL spans orders of magnitude across the sweep, but a sweep that
    # lands inside one decade would otherwise show a single labelled tick, so
    # label the 2/3/5 minors too.
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(2.0, 3.0, 5.0)))
    ax.xaxis.set_minor_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:g}")
    )
    ax.tick_params(axis="x", which="minor", labelsize=8)
    ax.set_xlabel(f"KL divergence — {direction} ({stat} over measured steps)",
                  fontsize=11)
    ax.set_ylabel("Success rate  (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linewidth=0.4, color="#cccccc", zorder=0)

    # Two legends: colour = n_samples, shape = n_iterations, plus the hollow
    # marker's meaning if any cell was flagged.
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=colors[n], markeredgecolor=colors[n],
                   label=str(n))
        for n in sorted(colors)
    ]
    leg1 = ax.legend(handles=handles, title="n_samples", title_fontsize=9,
                     fontsize=9, loc="lower left", framealpha=0.9)
    leg1.get_frame().set_linewidth(0.5)
    ax.add_artist(leg1)

    iters = sorted({c["n_iterations"] for c in reals})
    shape_handles = [
        plt.Line2D([], [], marker=ITER_MARKERS.get(i, _FALLBACK_MARKER),
                   linestyle="none", markersize=8, markerfacecolor="#555555",
                   markeredgecolor="#555555", label=str(i))
        for i in iters
    ]
    if any(above_null(c, direction, stat) is False for c in reals):
        shape_handles.append(
            plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                       markerfacecolor="none", markeredgecolor="#555555",
                       markeredgewidth=1.8, label="below null floor")
        )
    leg2 = ax.legend(handles=shape_handles, title="n_iterations",
                     title_fontsize=9, fontsize=9, loc="lower right",
                     framealpha=0.9)
    leg2.get_frame().set_linewidth(0.5)
    return plotted


def _null_panel(ax, reals, direction, stat, colors):
    """Real KL vs. that config's null-control KL, one cluster per config."""
    centres = np.arange(len(reals))
    bar_w   = 0.38

    for i, c in enumerate(reals):
        v, e   = kl_value(c, direction, stat)
        null   = c.get("null_cell")
        nv, ne = kl_value(null, direction, stat) if null else (None, 0.0)
        color  = colors[c["n_samples"]]

        if v is not None:
            ax.bar(centres[i] - bar_w / 2, v, width=bar_w, color=color,
                   label="_nolegend_", zorder=3)
            if e > 0:
                ax.errorbar(centres[i] - bar_w / 2, v, yerr=e, fmt="none",
                            ecolor="black", elinewidth=1.0, capsize=3.0,
                            capthick=1.0, zorder=4)
        if nv is not None:
            ax.bar(centres[i] + bar_w / 2, nv, width=bar_w, color=NULL_COLOR,
                   hatch="//", edgecolor="#888888", label="_nolegend_", zorder=3)
            if ne > 0:
                ax.errorbar(centres[i] + bar_w / 2, nv, yerr=ne, fmt="none",
                            ecolor="black", elinewidth=1.0, capsize=3.0,
                            capthick=1.0, zorder=4)
        elif v is not None:
            ax.annotate("no null", (centres[i] + bar_w / 2, 0),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8, rotation=90, color="#888888")

    ax.set_yscale("log")
    ax.set_ylabel(f"KL — {direction} ({stat})", fontsize=11)
    ax.set_xticks(centres)
    ax.set_xticklabels([f"n={c['n_samples']}\ni={c['n_iterations']}"
                        for c in reals], fontsize=9)
    ax.set_xlabel("Degraded-planner compute", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, color="#cccccc", zorder=0)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   markerfacecolor="#4C72B0", markeredgecolor="#4C72B0",
                   label="vs. reference planner"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   markerfacecolor=NULL_COLOR, markeredgecolor="#888888",
                   label="null control (noise floor)"),
    ]
    legend = ax.legend(handles=handles, fontsize=9, loc="upper right",
                       framealpha=0.9)
    legend.get_frame().set_linewidth(0.5)


def plot(reals, direction: str, stat: str, title: str, out_path: Path,
         drop_below_null: bool):
    have_null = any(c.get("null_cell") for c in reals)
    n_rows    = 2 if have_null else 1

    # Constrained layout rather than tight_layout: the scatter panel carries two
    # legends and a two-line title, which tight_layout cannot place.
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(max(8, len(reals) * 1.3), 9 if have_null else 6),
        gridspec_kw={"height_ratios": [3, 2]} if have_null else None,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)

    colors = _sample_colors(sorted({c["n_samples"] for c in reals}))

    plotted = _scatter_panel(axes[0], reals, direction, stat, colors,
                             drop_below_null)
    axes[0].set_title(title, fontsize=12, fontweight="bold", pad=10)
    if not plotted:
        raise ValueError(
            "Nothing left to plot — every cell was dropped. Re-run without "
            "--drop_below_null to see the cells and their null floors."
        )

    if have_null:
        _null_panel(axes[1], reals, direction, stat, colors)

    # Format follows --out's extension (the default path is .pdf).
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
def _print_table(reals, direction, stat):
    print(f"\n  {'cell':<18} {'eps':>4} {'succ':>7} {'KL':>12} {'±SE':>9} "
          f"{'null KL':>12} {'ESS deg':>8} {'ESS ref':>8}  verdict")
    for c in reals:
        v, e   = kl_value(c, direction, stat)
        null   = c.get("null_cell")
        nv, _  = kl_value(null, direction, stat) if null else (None, 0.0)
        ok     = above_null(c, direction, stat)
        verdict = {True: "above floor", False: "BELOW FLOOR", None: "no null cell"}[ok]

        ess_d = (c.get("ess_deg") or {}).get("mean")
        ess_r = (c.get("ess_ref") or {}).get("mean")
        # ESS near 1 means one particle carries all the weight; ESS near N means
        # the weights are uniform and the induced distribution has collapsed
        # back onto the proposal. Either end makes the covariance uninformative.
        flags = []
        if ess_d is not None and ess_d < 2.0:
            flags.append("ESS_deg~1")
        if ess_d is not None and ess_d > 0.9 * c["n_samples"]:
            flags.append("ESS_deg~N")
        if flags:
            verdict += "  [" + ", ".join(flags) + "]"

        print(f"  {c['label']:<18} {c['n_episodes']:>4d} "
              f"{c['success_rate']*100:>6.1f}% "
              f"{v if v is not None else float('nan'):>12.4g} "
              f"{e:>9.3g} "
              f"{(nv if nv is not None else float('nan')):>12.4g} "
              f"{(ess_d if ess_d is not None else float('nan')):>8.2f} "
              f"{(ess_r if ess_r is not None else float('nan')):>8.2f}  {verdict}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a directory of KL-divergence sweep JSONs: success "
                    "rate vs. KL, with the null-control floor."
    )
    parser.add_argument(
        "results_dir", nargs="?", type=Path,
        help="Directory of per-cell JSONs. Defaults to latest "
             "kl_divergence_eval_* dir in results/",
    )
    parser.add_argument(
        "--direction", choices=["forward", "reverse", "auto"], default="auto",
        help="Which KL direction to plot. 'auto' uses the headline_direction "
             "recorded by the sweep (default forward = KL(reference||degraded)).",
    )
    parser.add_argument(
        "--stat", choices=["mean", "median"], default="mean",
        help="Aggregation over the measured steps. The per-step KL is "
             "heavy-tailed across contact vs. free-flight states, so the median "
             "is often the more honest summary.",
    )
    parser.add_argument(
        "--drop_below_null", action="store_true",
        help="Omit cells whose KL does not clear their own null-control floor, "
             "instead of drawing them hollow.",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PDF path. Defaults to <dir>/<dir name>_plot.pdf")
    args = parser.parse_args()

    directory = args.results_dir or _latest_dir()
    print(f"Loading directory: {directory}")

    cells, files = merge_dir(directory)
    print(f"  Merged {len(files)} file(s) into {len(cells)} cell(s)")

    reals, nulls = pair_with_null(cells)
    if not reals:
        raise ValueError(
            "No non-null cells found. The sweep writes both, but a run of only "
            "--null_control cells has no real KL to plot."
        )

    direction = args.direction
    if direction == "auto":
        meta = directory / "meta.json"
        direction = (load(meta).get("kl_direction", "forward")
                     if meta.exists() else "forward")

    print(f"  Direction: {direction} ({args.stat})")
    print(f"  Real cells: {len(reals)}   null cells: {len(nulls)}")
    if not nulls:
        print("  ! No null-control cells — the estimator-noise floor is unknown, "
              "so no cell can be verified as reporting a real difference.")
    _print_table(reals, direction, args.stat)

    task_name = reals[0]["task_name"]
    model     = reals[0]["model"]
    ref_ns    = reals[0].get("ref_n_samples")
    ref_ni    = reals[0].get("ref_n_iterations")
    title = f"Success rate vs. planner KL divergence — {task_name} ({model})"
    if ref_ns:
        title += f"\nreference planner: {ref_ns} samples x {ref_ni} iterations"

    out_path = args.out or directory / f"{directory.name}_plot.pdf"
    plot(reals, direction, args.stat, title, out_path, args.drop_below_null)


if __name__ == "__main__":
    main()
