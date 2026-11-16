#!/usr/bin/env python3
"""Closed-loop trajectory cost against planning horizon (in seconds), one line per control period.

One panel per model, one line per control period (16, 24, ..., 128 ms, every
8 ms), from the ``horizon_closed_loop.py`` runs in DIR. Models and periods
without runs are left out. Lines are dotted (and markers hollow) where planning
is not real-time: the mean Plan() time exceeds the control period.

Sync (default), M1-M6, the ``cl_`` runs:
  cl_{M}_onM5_T0.1_meancost_dt{P}_goal{F}_seeds{N}.npz   the period sweep: N seeds from the initial
                                                          pose, toward the goal showing face F
  Without one, the earlier driver-start runs: cl_{M}_..._dt{P}_S{S}K{K}.npz (the
  fewest repeats first), and at 64 ms cl_M5_..._W20_every2_S8K6.npz for M5 and
  cl_{M}_..._W20_all.npz for the others, cut to the same start times (episodes
  0-1, steps 0, 6, 12, 18) and its first 6 repeats.
  Plan times: plan_time_solo.json in DIR (bench_plan_time.py, one process alone
  on the GPU) for M1-M4, M6; M5's from its own runs (at 64 ms even h=1 takes
  ~150 ms solo, so every M5 point is over budget).

``--async``, M1-M4 and M6, the ``acl_`` runs (horizon_closed_loop.py --async):
  acl_{M}_onM5_T0.1_meancost_dt{P}_goal{F}_seeds{N}.npz, or the earlier _S{S}K{K}.npz
  Planning took sim time in these runs, so a planner slower than its control
  period fell behind and played later rows of older plans. Plan times are the
  mean Plan() time measured in the run itself.

Metrics, the cost of a closed-loop trajectory: per start state (seed) the mean over its
repeats (the trajectory's cost with one), then the mean over start states, drawn with a
band of one standard error either side:
  running   the time-averaged running cost: sum over steps 1..W-1 / (W - 1)
  terminal  the terminal cost at the end of the window
  whole     running time-average * (W64 - 1)/W64 + terminal / W64, W64 the sweep's window at
            64 ms (16 steps): the terminal weighted as at 64 ms at every period

The standard error is of each point on its own. The seeds are the same at every horizon
and period, so differences between points are paired and better resolved than the bands
suggest. A run in which a trajectory reached the goal is reported when it loads: the
window is meant to end before any does.

The window is round(1 s / period) control steps (0.96-1.04 s). The horizons plotted are
the period sweep's (``period_sweep_jobs.horizons``: log-spaced, one control step to
1.28 s), at their length in seconds, when a run has them all (an earlier run with every
horizon is cut to them); otherwise every horizon in the run.

``--tag T`` reads the runs named with T appended (run_period_sweep.sh's TAG), e.g. the
``--brownian-sigma`` variant; the subtitle gives sigma as the runs set it.

``--heatmap`` plots each model as a heatmap instead: control frequency across, horizon up
(log; ``--linear`` for linear), the mean as colour and printed in each cell tall enough (with its standard
error where there is room). A red dotted staircase marks the
Pareto frontier, taken as the real-time (async: keeps-up) limit: in each column, the cells
below it have a mean plan time within the control period.

``--frontiers`` puts every model's real-time frontier from the heatmaps on one plot, one
hue per model. Each column's segment is drawn at an intensity set by the whole-trajectory
cost at that period's longest real-time horizon, where the segment sits: stronger is lower cost,
on one scale across models. Lines that would overlap are nudged apart by a few points.
A period's real-time limit is its longest run of real-time horizons from one step up: plan
times are measured, so one just over the period can sit between two under it.

Writes OUT_PREFIX_{running,whole,terminal}.png, OUT_PREFIX_heat_{...}.png with --heatmap,
or OUT_PREFIX_frontiers.png with --frontiers.

Usage:
    python experiments/plot_period_sweep.py results/horizon_kl_sweep/cl_period_onM5_T0.1_meancost
    python experiments/plot_period_sweep.py --async results/horizon_kl_sweep/acl_period_onM5_T0.1_meancost
    python experiments/plot_period_sweep.py --async --tag _bsig results/horizon_kl_sweep/acl_period_bsig
    python experiments/plot_period_sweep.py --heatmap --tag _bsig results/horizon_kl_sweep/cl_period_bsig
"""
import argparse
import collections
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.collections  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms  # noqa: E402
import numpy as np  # noqa: E402

import period_sweep_jobs as psj  # noqa: E402


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("prefix", help="output prefix: writes PREFIX_{running,whole,terminal}.png")
ap.add_argument("dir", nargs="?", type=Path,
                default=Path(__file__).resolve().parents[1] / "results" / "horizon_kl_sweep")
ap.add_argument("--async", dest="async_", action="store_true",
                help="plot the async (acl_) runs: planning time spent as sim time")
ap.add_argument("--tag", default="", help="read the runs named with this appended, e.g. _bsig")
ap.add_argument("--frontiers", action="store_true",
                help="plot every model's real-time frontier on one plane, intensity set by its cost")
ap.add_argument("--heatmap", action="store_true",
                help="plot each model as a heatmap over control frequency x horizon, with the real-time frontier")
ap.add_argument("--linear", action="store_true", help="with --heatmap: a linear horizon axis instead of log")
opts = ap.parse_args()
W64 = round(psj.WINDOW_MS / 64)                 # the whole-trajectory metric's terminal weight is 1 / W64
TAG = opts.tag
D = opts.dir
ASYNC = opts.async_
MODELS = ["M1", "M2", "M3", "M4", "M6"] if ASYNC else ["M1", "M2", "M3", "M4", "M5", "M6"]
PERIODS = list(range(16, 129, 8))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# ordinal: control period, one blue ramp light (fast) -> dark (slow), with a marker each. The
# reference blue ramp (steps 250 -> 700) resampled at 8 even OKLCH lightness steps (dL >= 0.06;
# the light end clears 2:1 on the surface), interpolated to one colour per period.
_RAMP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "period", ["#86b6ef", "#65a2ea", "#438de6", "#2b79d7", "#2366ba", "#1a549e", "#114383", "#0c3366"])
PCOLOR = {p: matplotlib.colors.to_hex(_RAMP(i / (len(PERIODS) - 1))) for i, p in enumerate(PERIODS)}
PMARK = dict(zip(PERIODS, ["o", "s", "v", "^", "P", "X", "h", "D", "<", ">", "p", "*", "d", "H", "8"]))
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})

SOLO = None if ASYNC else json.loads((D / "plan_time_solo.json").read_text())


def planMs(m, p, h, d):
    """Mean plan time in ms of model m at period p ms and horizon h: solo, or the run's own."""
    if ASYNC or m == "M5":
        return 1e3 * float(np.mean(d["plan_times"][str(h)]))
    if f"{m}/{p}/{h}" not in SOLO:
        print(f"  no solo plan time for {m}/{p}/{h} in plan_time_solo.json: drawn as not real-time")
        return np.nan
    return 1e3 * SOLO[f"{m}/{p}/{h}"]["mean"]


def runFile(m, p):
    """(json path or None, cut to the period sweep's start states and repeats).

    The seeds run of the current sweep; else the earlier driver-start runs, the fewest repeats first.
    """
    base = f"{'acl' if ASYNC else 'cl'}_{m}_onM5_T0.1_meancost_dt{p}"
    hits = [f for f in D.glob(f"{base}_*seeds*{TAG}.json") if re.fullmatch(rf".*_seeds\d+{re.escape(TAG)}", f.stem)]
    hits += sorted((f for f in D.glob(f"{base}_S*{TAG}.json") if re.fullmatch(rf".*_S\d+K\d+{re.escape(TAG)}", f.stem)),
                   key=lambda f: int(re.search(r"_S\d+K(\d+)", f.stem).group(1)))
    if hits:
        return hits[0], False
    if p == 64 and not ASYNC:                                     # the earlier runs, from before 64 ms was in the sweep
        if m == "M5":
            return D / f"cl_M5_onM5_T0.1_meancost_W20_every2_S8K6{TAG}.json", False
        return D / f"cl_{m}_onM5_T0.1_meancost_W20_all{TAG}.json", True
    return None, False


def load(m, p):
    """(horizon seconds, run (S, nh, K, W), term (S, nh, K), lam, label, real-time mask, plan ms, sigma,
    window seconds, goal face or None) or None."""
    f, cut = runFile(m, p)
    if f is None or not f.exists() or not f.with_suffix(".npz").exists():
        return None
    d = json.loads(f.read_text())
    z = np.load(f.with_suffix(".npz"))
    H = np.array(d["horizons"])
    dt_ms = round(d["control_dt"] * 1e3)
    run, term = z["run"], z["term"]
    if "success_at" in z.files and (z["success_at"] >= 0).any():
        reached = {int(h): int((z["success_at"][:, i] >= 0).sum()) for i, h in enumerate(H) if (z["success_at"][:, i] >= 0).any()}
        print(f"  {f.name}: trajectories reached the goal within the window, by horizon: {reached}")
    if cut:
        st = [tuple(s) for s in d["starts"]]
        keep = [i for i, (e, s) in enumerate(st) if e in (0, 1) and s in (0, 6, 12, 18)]
        run, term = run[keep][:, :, :6], term[keep][:, :, :6]
    grid = psj.horizons(p)                       # the sweep's horizons, when the run has them all
    want = grid if set(grid) <= set(H.tolist()) else sorted(H.tolist())
    ix = [list(H).index(h) for h in want]
    plan_ms = np.array([planMs(m, p, h, d) for h in want])
    sig = (d.get("sigma_scaling", "none"), d.get("noise_sigma_ref", d["noise_sigma"]), d.get("sigma_ref_dt"))
    return (np.array(want) * dt_ms / 1e3, run[:, ix], term[:, ix], d["temperature"], f.name, plan_ms <= p,
            plan_ms, sig, d["window"] * dt_ms / 1e3, d.get("goal_face"))


def realTime(ok):
    """How many horizons, from the shortest, are real-time before the first that is not."""
    return len(ok) if ok.all() else int(np.argmin(ok))


def metrics(run, term):
    """Each metric per horizon: the mean over start states of the mean over repeats (with one
    repeat, the trajectory's cost), and its standard error as ``<metric>_se``."""
    W = run.shape[-1]
    r = run[..., :W - 1].sum(-1) / (W - 1)                         # (S, nh, K)
    out = {"S": run.shape[0], "K": run.shape[2]}
    for key, x in (("running", r), ("whole", r * (W64 - 1) / W64 + term / W64), ("terminal", term)):
        x = np.nanmean(x, -1)                                      # (S, nh)
        n = np.isfinite(x).sum(0)
        out[key] = np.nanmean(x, 0)
        out[f"{key}_se"] = np.nanstd(x, 0, ddof=1) / np.sqrt(n) if x.shape[0] > 1 else np.zeros(x.shape[1])
    return out


data, SIGMAS, WINDOWS, GOALS = {}, set(), set(), set()
for m in MODELS:
    for p in PERIODS:
        got = load(m, p)
        if got is not None:
            hs, run, term, lam, name, ok, plan_ms, sig, win, goal = got
            data[m, p] = (hs, metrics(run, term), name, ok, plan_ms)
            GOALS.add(goal)
            SIGMAS.add(sig)
            WINDOWS.add(round(win, 3))
MODELS = [m for m in MODELS if any((m, p) in data for p in PERIODS)]
# the usual start states and rollouts per start; panels note the runs that differ
S0, K0 = collections.Counter((v[1]["S"], v[1]["K"]) for v in data.values()).most_common(1)[0][0]
SEEDED = all("_seeds" in v[2] for v in data.values())
OVER = f"{S0} seeds" if SEEDED and K0 == 1 else f"{S0} start states (K={K0} rollouts each)"
if GOALS and None not in GOALS:
    OVER += f", goal: face {' or '.join(sorted(GOALS))} up"
PERIODS = [p for p in PERIODS if any((m, p) in data for m in MODELS)]
print("loaded:", ", ".join(f"{m}@{p}ms" for m, p in data))
print("whole-trajectory cost per horizon (s), * = over budget, (mean plan time):")
for (m, p), (hs, mt, _, ok, plan_ms) in sorted(data.items()):
    print(f"  {m} {p:3d} ms  " + "  ".join(f"{h:.3g}s:{mt['whole'][i]:.1f}{'' if ok[i] else '*'}({plan_ms[i]:.0f}ms)"
                                          for i, h in enumerate(hs)))

if ASYNC:
    KEEPS_UP = ["keeps up: mean plan time ≤ control period", "runs out of time: mean plan time > period"]
    TIMES = "plan times measured in the async runs"
    HEAD, SUB = "Async MPPI", "planning time spent as sim time (tape executor); "
else:
    KEEPS_UP = ["real-time: mean plan time ≤ control period", "not real-time: mean plan time > period"]
    TIMES = "plan times measured solo"
    HEAD, SUB = "MPPI", ""
SIG = []
for scaling, s0, ref in sorted(SIGMAS, key=str):
    SIG.append(f"σ={s0:g}·√(period / {ref * 1e3:g} ms)" if scaling == "brownian" else f"σ={s0:g}")
SIG = " or ".join(SIG) or "σ=?"
WIN = (f"{min(WINDOWS):g}" if len(WINDOWS) == 1 else
       f"≈{np.median(sorted(WINDOWS)):.3g} s ({min(WINDOWS):g}–{max(WINDOWS):g}")
WIN = WIN + (" s" if len(WINDOWS) == 1 else " s)")

METRICS = (
    ("running", "Running cost\n[cost per step]",
     f"Time-averaged running cost over the {WIN} window"),
    ("whole", "Whole-trajectory cost\n[cost per step]",
     f"Whole trajectory, terminal weighted as at 64 ms: running avg × {W64 - 1}/{W64} + terminal / {W64}"),
    ("terminal", "Terminal cost\n[cost, once]", f"Terminal cost at the end of the {WIN} window"))


def panels(panel_h, header, **gridspec):
    """``(fig, axes, H)``: one panel per model, in two columns (one, full width, for a single model),
    each ``panel_h`` inches tall, under ``header`` inches for the title and legend; H is the figure
    height in inches."""
    ncols = min(2, len(MODELS))
    nrows = -(-len(MODELS) // ncols)
    H = panel_h * nrows + header
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, H), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw={**gridspec, "top": 1 - header / H})
    return fig, axes, H


def linePlots():
    for key, ylabel, title in METRICS:
        fig, axes, H = panels(4, 1.6, hspace=0.32, wspace=0.08)
        for ax, m in zip(axes.ravel(), MODELS):
            for p in PERIODS:
                if (m, p) not in data:
                    continue
                hs, mt, _, ok, _ = data[m, p]
                y, se = mt[key], mt[f"{key}_se"]
                ms = 5.5 if len(hs) <= 12 else 3.5
                ax.fill_between(hs, y - se, y + se, color=PCOLOR[p], alpha=0.07, lw=0, zorder=1)
                for i in range(len(hs) - 1):      # solid between two real-time points, dotted otherwise
                    rt = ok[i] and ok[i + 1]
                    ax.plot(hs[i:i + 2], y[i:i + 2], color=PCOLOR[p], lw=2 if rt else 1.6,
                            ls="-" if rt else (0, (1.2, 1.6)), solid_capstyle="round", zorder=2)
                ax.plot(hs[ok], y[ok], ls="none", color=PCOLOR[p], marker=PMARK[p], ms=ms, markeredgecolor=SURF,
                        markeredgewidth=0.8, zorder=3)
                ax.plot(hs[~ok], y[~ok], ls="none", marker=PMARK[p], ms=ms, markerfacecolor=SURF,
                        markeredgecolor=PCOLOR[p], markeredgewidth=1.1, zorder=3)
                ax.plot([], [], color=PCOLOR[p], lw=2, marker=PMARK[p], ms=5.5, markeredgecolor=SURF,
                        markeredgewidth=1.2, label=f"{p} ms ({1e3 / p:.3g} Hz)")
            notes = [f"{p} ms: S={data[m, p][1]['S']}, K={data[m, p][1]['K']}" for p in PERIODS
                     if (m, p) in data and (data[m, p][1]["S"], data[m, p][1]["K"]) != (S0, K0)]
            missing = [f"{p}" for p in PERIODS if (m, p) not in data]
            if missing:
                notes.append(f"no {', '.join(missing)} ms yet")
            ax.set_title(m + ("   (" + "; ".join(notes) + ")" if notes else ""), fontsize=11, color=INK, loc="left")
            ax.axvline(0.384, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=0)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks([0, 0.256, 0.512, 0.768, 1.024, 1.28])
            ax.set_xticklabels(["0", "0.256", "0.512", "0.768", "1.024", "1.28"])
        if axes.shape[1] == 2 and len(MODELS) % 2:  # the panel above the empty last one carries the x axis
            axes[-1, -1].set_visible(False)
            if axes.shape[0] > 1:
                axes[-2, 1].xaxis.set_tick_params(labelbottom=True)
                axes[-2, 1].set_xlabel("Planning horizon  [s]")
        else:
            axes[-1, -1].set_xlabel("Planning horizon  [s]")
        axes[-1, 0].set_xlabel("Planning horizon  [s]")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
            ax.yaxis.set_tick_params(labelleft=True)
        hl = {}
        for ax in axes.ravel():
            for h_, l_ in zip(*ax.get_legend_handles_labels()):
                hl.setdefault(l_, h_)
        order = sorted(hl, key=lambda l_: int(l_.split()[0]))
        style = [matplotlib.lines.Line2D([], [], color=INK2, lw=2, marker="o", ms=5.5, markeredgecolor=SURF),
                 matplotlib.lines.Line2D([], [], color=INK2, lw=1.6, ls=(0, (1.2, 1.6)), marker="o", ms=5.5,
                                         markerfacecolor=SURF, markeredgecolor=INK2)]
        fig.legend([hl[l_] for l_ in order] + style, order + KEEPS_UP,
                   frameon=False, fontsize=9, loc="upper center", ncol=-(-(len(PERIODS) + 2) // 3),
                   bbox_to_anchor=(0.5, 1 - 0.45 / H),
                   title=f"control period (mean ± 1 standard error over {OVER}, unless noted); {TIMES}",
                   title_fontsize=9)
        axes[0, 0].text(0.384, 0.97, " nominal 0.384 s", color=INK2, fontsize=8,
                        transform=axes[0, 0].get_xaxis_transform(), va="top")
        fig.suptitle(f"{HEAD} in closed loop on M5 for {WIN}, by control period: {title.lower()}\n"
                     f"{SUB}cube, N=256, T=0.1, {SIG} per control step, cost averaged over horizon; "
                     f"mean ± standard error over {OVER}", fontsize=10.5, color=INK, y=1)
        out = f"{opts.prefix}_{key}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(out)


# sequential, one hue: low cost light -> high cost dark (the period ramp, extended at both ends)
COST_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "cost", ["#e3eefc", "#86b6ef", "#2b79d7", "#114383", "#0a2447"])
FRONTIER = "#e03131"                            # the real-time frontier, apart from the blue ramp


#: The heatmaps' horizon axis (s): its limits, and the axis transform (log, or linear with --linear).
H_LIM = (0, 1.36) if opts.linear else (0.0075, 1.75)
_fwd, _inv = (lambda x: np.asarray(x, float), lambda x: x) if opts.linear else (np.log, np.exp)


def cellEdges(hs):
    """Row edges of one period's column on the horizon axis: halfway (on the axis) between horizons."""
    lh = _fwd(hs)
    mid = (lh[1:] + lh[:-1]) / 2
    e = np.r_[2 * lh[0] - mid[0], mid, 2 * lh[-1] - mid[-1]]
    return _inv(np.maximum(e, 0) if opts.linear else e)


def axisFraction(lo, hi):
    """How much of the horizon axis the span lo..hi covers."""
    return float((_fwd(hi) - _fwd(lo)) / (_fwd(H_LIM[1]) - _fwd(H_LIM[0])))


def heatMaps():
    cols = PERIODS[::-1]                        # control frequency rising left to right
    for key, ylabel, title in METRICS:
        vals = np.concatenate([data[m, p][1][key] for m in MODELS for p in cols if (m, p) in data])
        norm = matplotlib.colors.Normalize(np.nanmin(vals), np.nanmax(vals))
        fig, axes, H = panels(6.5, 1.3, hspace=0.25, wspace=0.06)
        ax_in = axes[0, 0].get_position().height * H     # a panel's height, inches
        for ax, m in zip(axes.ravel(), MODELS):
            fx, fy = [], []
            for j, p in enumerate(cols):
                if (m, p) not in data:
                    continue
                hs, mt, _, ok, _ = data[m, p]
                y, se, e = mt[key], mt[f"{key}_se"], cellEdges(hs)
                ax.pcolormesh([j - 0.5, j + 0.5], e, y[:, None], cmap=COST_CMAP, norm=norm, edgecolors="face",
                              linewidth=0.3)
                for i, h in enumerate(hs):
                    tall = axisFraction(e[i], e[i + 1]) * ax_in      # the cell's height, inches
                    if tall < 0.1:                   # too thin for a label
                        continue
                    lum = np.dot(COST_CMAP(norm(y[i]))[:3], [0.2126, 0.7152, 0.0722])
                    label = f"{y[i]:.1f}\n±{se[i]:.1f}" if tall >= 0.19 else f"{y[i]:.1f}"
                    ax.text(j, _inv((_fwd(e[i]) + _fwd(e[i + 1])) / 2), label, ha="center", va="center", fontsize=6,
                            linespacing=0.9,
                            color=SURF if lum < 0.4 else INK)
                if fx and fx[-1] != j - 0.5:         # a missing column breaks the line
                    fx, fy = fx + [np.nan], fy + [np.nan]
                fx += [j - 0.5, j + 0.5]
                fy += [e[realTime(ok)]] * 2
            ax.plot(fx, fy, color=FRONTIER, lw=2.4, ls=(0, (0.1, 1.6)), dash_capstyle="round", zorder=5)
            ax.set_title(m, fontsize=11, color=INK, loc="left")
            if opts.linear:
                ax.set_yticks([0, 0.128, 0.256, 0.384, 0.512, 0.64, 0.768, 0.896, 1.024, 1.152, 1.28])
            else:
                ax.set_yscale("log")
                ax.set_yticks([0.016, 0.032, 0.064, 0.128, 0.256, 0.384, 0.768, 1.28])
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([f"{1e3 / p:.3g}\n{p}" for p in cols], fontsize=6.5)
            ax.set_xlim(-0.5, len(cols) - 0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=0)
        ax.set_ylim(*H_LIM)
        for ax in axes[:, 0]:
            ax.set_ylabel("Planning horizon  [s]")
            ax.yaxis.set_tick_params(labelleft=True)
        xlabel = "Control frequency  [Hz]  (period [ms])"
        axes[-1, 0].set_xlabel(xlabel)
        if axes.shape[1] == 2 and len(MODELS) % 2:  # the empty last slot holds the colour bar
            axes[-1, -1].set_visible(False)
            if axes.shape[0] > 1:
                axes[-2, 1].xaxis.set_tick_params(labelbottom=True)
                axes[-2, 1].set_xlabel(xlabel)
            box = axes[-1, -1].get_position()
            cax = fig.add_axes([box.x0 + 0.1 * box.width, box.y0 + 0.62 * box.height, 0.8 * box.width,
                                0.06 * box.height])
            cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm, COST_CMAP), cax=cax, orientation="horizontal")
        else:
            axes[-1, -1].set_xlabel(xlabel)
            cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm, COST_CMAP), ax=axes, shrink=0.5, pad=0.02)
        cb.set_label(ylabel.replace("\n", "  "), color=INK)
        cb.outline.set_visible(False)
        fig.legend([matplotlib.lines.Line2D([], [], color=FRONTIER, lw=2.4, ls=(0, (0.1, 1.6)),
                                            dash_capstyle="round")],
                   [f"Pareto frontier ({KEEPS_UP[0].split(':')[0].replace(' ', '-')} limit): "
                    "cells below have mean plan time ≤ control period"],
                   frameon=False, fontsize=9, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1 - 0.75 / H))
        fig.suptitle(f"{HEAD} in closed loop on M5 for {WIN}, by control period and horizon: {title.lower()}\n"
                     f"{SUB}cube, N=256, T=0.1, {SIG} per control step, cost averaged over horizon\n"
                     f"mean over {OVER} (± 1 standard error where it fits); {TIMES}", fontsize=10.5, color=INK,
                     y=1)
        out = f"{opts.prefix}_heat_{key}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(out)


# categorical, fixed per model: the reference order's first five slots for the five models plotted
MCOLOR = dict(zip(["M1", "M2", "M3", "M4", "M6", "M5"],
                  ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MMARK = dict(zip(["M1", "M2", "M3", "M4", "M6", "M5"], ["o", "s", "^", "D", "v", "P"]))


def frontierPlot():
    """Every model's real-time frontier on one frequency x horizon plane, intensity from whole cost."""
    cols = PERIODS[::-1]
    fr = {}                                     # model -> [(column, longest real-time horizon, its cost)]
    for m in MODELS:
        fr[m] = []
        for j, p in enumerate(cols):
            if (m, p) in data:
                hs, mt, _, ok, _ = data[m, p]
                n = realTime(ok)
                if n:
                    fr[m].append((j, hs[n - 1], mt["whole"][n - 1]))
    f = np.array([v for m in MODELS for _, _, v in fr[m]])
    lo, hi = np.nanmin(f), np.nanmax(f)
    surf = np.array(matplotlib.colors.to_rgb(SURF))

    def shade(m, v):                            # 0.2 of the hue at the highest cost, full hue at the lowest
        a = 0.2 + 0.8 * (hi - v) / (hi - lo)
        return tuple(surf + a * (np.array(matplotlib.colors.to_rgb(MCOLOR[m])) - surf))

    fig, ax = plt.subplots(figsize=(10, 6.4))
    for k, m in enumerate(MODELS):
        tr = matplotlib.transforms.offset_copy(ax.transData, fig=fig, y=3.2 * (k - (len(MODELS) - 1) / 2),
                                               units="points")
        segs, colors = [], []
        for i, (j, y, v) in enumerate(fr[m]):
            segs.append([(j - 0.5, y), (j + 0.5, y)])
            colors.append(shade(m, v))
            if i + 1 < len(fr[m]) and fr[m][i + 1][0] == j + 1:  # riser to the next column
                y2, v2 = fr[m][i + 1][1:]
                segs.append([(j + 0.5, y), (j + 0.5, y2)])
                colors.append(shade(m, (v + v2) / 2))
        ax.add_collection(matplotlib.collections.LineCollection(segs, colors=colors, lw=3, capstyle="round",
                                                                transform=tr, zorder=2 + k))
        for j, y, v in fr[m]:
            ax.plot(j, y, marker=MMARK[m], ms=6.5, color=shade(m, v), markeredgecolor=SURF, markeredgewidth=1,
                    transform=tr, zorder=10 + k)
    ax.set_yscale("log")
    ax.set_yticks([0.016, 0.032, 0.064, 0.128, 0.256, 0.384, 0.768, 1.28])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0.05, 1.6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{1e3 / p:.3g}\n{p}" for p in cols], fontsize=8)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("Control frequency  [Hz]  (period [ms])")
    ax.set_ylabel("Longest real-time planning horizon  [s]" if not ASYNC else
                  "Longest planning horizon that keeps up  [s]")
    handles = [matplotlib.lines.Line2D([], [], color=MCOLOR[m], lw=3, marker=MMARK[m], ms=6.5,
                                       markeredgecolor=SURF) for m in MODELS]
    ax.legend(handles, MODELS, frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="model", title_fontsize=9.5, alignment="left")
    cax = fig.add_axes([0.925, 0.14, 0.018, 0.4])
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "intensity", [tuple(surf + 0.2 * (np.array(matplotlib.colors.to_rgb(INK2)) - surf)), INK2])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(lo, hi), ramp.reversed()),
                      cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("whole-trajectory cost on the frontier\n(stronger line = lower cost)", color=INK, fontsize=9)
    fig.subplots_adjust(right=0.88)
    fig.suptitle(f"{HEAD} in closed loop on M5: each model's {KEEPS_UP[0].split(':')[0].replace(' ', '-')} "
                 "frontier, shaded by the whole-trajectory cost at the frontier horizon\n"
                 f"{SUB}cube, N=256, T=0.1, {SIG} per control step\n{WIN} window; {TIMES}",
                 fontsize=10.5, color=INK, y=0.99)
    out = f"{opts.prefix}_frontiers.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(out)


if opts.frontiers:
    frontierPlot()
else:
    heatMaps() if opts.heatmap else linePlots()
