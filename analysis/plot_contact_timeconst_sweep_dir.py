"""plot_contact_timeconst_sweep_dir.py

Two-panel clustered bar chart for the contact-time-constant sweep — the
per-cell JSONs written by experiments/hpc/contact_timeconst_eval.slurm
(results/contact_timeconst_eval_<id>/<model>_tc<microseconds>.json).

Top panel    — success rate (%) with ±1 SE whiskers.
Bottom panel — mean MPPI step time (ms) with ±1 SD whiskers.

Each cluster on the x-axis is one contact time constant (solref timeconst, in
ms); bars within a cluster are one per contact model, so the figure stays
correct if the sweep is later widened past its default single model (M2).
Smaller timeconst = stiffer contact.

Two reference lines are drawn from the sweep's meta.json:
  * the MuJoCo/XML default of 20 ms (M2's un-overridden value), and
  * the 2·rollout_dt integrator stability floor, below which contact rings or
    diverges — cells there are expected to look bad, and that is the result.

Merge rule matches plot_cntrl_freq_sweep_dir.py: each label's stats are pooled
across files weighted by that file's n_episodes, the success-rate whisker is
recomputed as a binomial SE sqrt(p(1-p)/N) from the pooled rate and total
episode count, and step-time mean/SD are episode-weighted averages.

Drawing helpers and the model colour/label maps are reused from
plot_cntrl_freq_sweep.py so the figures look like one family.

Usage:
    python analysis/plot_contact_timeconst_sweep_dir.py                          # latest contact_timeconst_eval_* dir
    python analysis/plot_contact_timeconst_sweep_dir.py results/contact_timeconst_eval_12345
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Reuse the loader + drawing helpers from the control-frequency plotter (same dir).
sys.path.insert(0, str(Path(__file__).parent))
import plot_cntrl_freq_sweep as base  # noqa: E402
import sweep_io  # noqa: E402

RESULTS_DIR = Path(__file__).parent.parent / "results"

_LABEL_RE = re.compile(r"^(M\d+)_tc(\d+)$")

# MuJoCo's compiler default solref timeconst, in ms — M2's value when nothing
# overrides it. Drawn as a reference line so "stiffer/softer than stock" reads
# straight off the axis.
DEFAULT_TIMECONST_MS = 20.0


def _latest_dir() -> Path:
    dirs = sorted(d for d in RESULTS_DIR.glob("contact_timeconst_eval_*") if d.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"No contact_timeconst_eval_* directories found in {RESULTS_DIR}. "
            f"Pass a directory explicitly."
        )
    return dirs[-1]


def _load_rollout_dt(directory: Path, override: float | None) -> float | None:
    """Return rollout_dt (s) from meta.json, or the override, or None."""
    if override is not None:
        return override
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        print(f"  Warning: no meta.json in {directory}; the stability-floor "
              f"marker will be omitted (pass --rollout_dt to draw it)")
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    dt = meta.get("rollout_dt")
    return float(dt) if dt is not None else None


def merge_dir(directory: Path) -> tuple[list[dict], list[Path]]:
    """Pool every *.json (excluding meta.json) in *directory* into one record
    per model_label.

    Returns records with the fields parse_records reads: model_label,
    task_name, n_episodes, success_rate, success_rate_se,
    mean_step_ms, std_step_ms.
    """
    files = sorted(p for p in directory.glob("*.json") if p.name != "meta.json")
    if not files:
        raise FileNotFoundError(f"No *.json files found in {directory}")

    # label -> episode-weighted accumulators
    acc: dict[str, dict] = {}
    for path in files:
        for r in sweep_io.load_aggregates(path):
            label = r.get("model_label")
            if label is None:
                continue
            n = int(r.get("n_episodes", 0) or 0)
            a = acc.setdefault(label, {
                "n": 0, "sr": 0.0, "ms": 0.0, "sd": 0.0,
                "task_name": r.get("task_name", "unknown"),
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
            "n_episodes":      n_tot,
            "success_rate":    sr,
            "success_rate_se": se,
            "mean_step_ms":    a["ms"] / w,
            "std_step_ms":     a["sd"] / w,
        })

    return merged, files


def parse_records(records: list[dict]):
    """Return (models, timeconst_values_ms, data).

    data[model][timeconst_ms] = (success_rate, success_rate_se,
                                 mean_step_ms, std_step_ms)
    """
    data: dict[str, dict[float, tuple]] = {}

    for r in records:
        m = _LABEL_RE.match(r["model_label"])
        if m is None:
            continue
        model = m.group(1)
        # Labels carry microseconds to stay integral; plot in ms.
        tc_ms = int(m.group(2)) / 1000.0
        data.setdefault(model, {})[tc_ms] = (
            float(r["success_rate"]),
            float(r["success_rate_se"]),
            float(r.get("mean_step_ms", 0.0)),
            float(r.get("std_step_ms",  0.0)),
        )

    models    = sorted(data.keys())
    tc_values = sorted({t for mdict in data.values() for t in mdict})
    return models, tc_values, data


def _tc_label(tc_ms: float) -> str:
    if tc_ms == int(tc_ms):
        return f"{int(tc_ms)}"
    return f"{tc_ms:.4g}"


def _marker_position(tc_values: list[float], target: float) -> float | None:
    """Map a time constant in ms onto the categorical x-axis.

    Clusters are evenly spaced regardless of their actual values, so a
    reference line at e.g. 8 ms has to be interpolated between the two
    neighbouring clusters. Log-space interpolation matches the 2x-spaced grid.
    Returns None when the target falls outside the swept range.
    """
    if not tc_values or target <= 0:
        return None
    if target < tc_values[0] or target > tc_values[-1]:
        return None
    logs = np.log2(tc_values)
    return float(np.interp(math.log2(target), logs, np.arange(len(tc_values))))


def plot(models, tc_values, data, title: str, rollout_dt: float | None,
         out_path: Path):
    n_groups  = len(tc_values)
    n_models  = len(models)
    bar_w     = 0.7 / n_models
    group_gap = 1.0

    fig, (ax_succ, ax_time) = plt.subplots(
        2, 1,
        figsize=(max(8, n_groups * 1.4), 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
    )

    group_centres = np.arange(n_groups) * group_gap
    offsets       = (np.arange(n_models) - (n_models - 1) / 2) * bar_w

    # ── Top panel: success rate ──────────────────────────────────────
    base._clustered_bars(ax_succ, models, tc_values, data,
                         value_idx=0, error_idx=1,
                         group_centres=group_centres, offsets=offsets,
                         bar_w=bar_w, scale=100.0, add_legend=True)

    ax_succ.set_ylabel("Success rate  (%)", fontsize=11)
    ax_succ.set_ylim(0, 115)
    ax_succ.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_succ.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax_succ.spines[["top", "right"]].set_visible(False)
    ax_succ.grid(axis="y", linewidth=0.4, color="#cccccc", zorder=0)

    # Reference markers: the stock MuJoCo default, and the stability floor.
    for ax in (ax_succ, ax_time):
        x = _marker_position(tc_values, DEFAULT_TIMECONST_MS)
        if x is not None:
            ax.axvline(x, color="#666666", linestyle="--", linewidth=1.0, zorder=1)
        if rollout_dt is not None:
            xf = _marker_position(tc_values, 2.0 * rollout_dt * 1e3)
            if xf is not None:
                ax.axvspan(-0.5, xf, color="#c44e52", alpha=0.07, zorder=0)
                ax.axvline(xf, color="#C44E52", linestyle=":", linewidth=1.2, zorder=1)

    # Reference-line labels run vertically alongside their line: the lines land
    # at data-dependent x positions and would otherwise collide with the legend
    # or the tallest bars.
    x_def = _marker_position(tc_values, DEFAULT_TIMECONST_MS)
    if x_def is not None:
        ax_succ.text(
            x_def - 0.04, 50, f"MuJoCo default ({_tc_label(DEFAULT_TIMECONST_MS)} ms)",
            rotation=90, ha="right", va="center", fontsize=7.5, color="#666666",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )
    if rollout_dt is not None:
        x_floor = _marker_position(tc_values, 2.0 * rollout_dt * 1e3)
        if x_floor is not None:
            ax_succ.text(
                x_floor + 0.04, 50,
                f"2·rollout_dt = {2*rollout_dt*1e3:.3g} ms (unstable below)",
                rotation=90, ha="left", va="center", fontsize=7, color="#C44E52",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            )
        ax_succ.annotate(
            f"rollout_dt = {rollout_dt*1e3:.3g} ms",
            xy=(1.0, 0.02), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=7.5, color="#888888",
        )

    legend = ax_succ.legend(
        title="Contact model", title_fontsize=9, fontsize=9,
        framealpha=0.9, loc="upper left",
    )
    legend.get_frame().set_linewidth(0.5)

    # ── Bottom panel: mean step time ─────────────────────────────────
    base._clustered_bars(ax_time, models, tc_values, data,
                         value_idx=2, error_idx=3,
                         group_centres=group_centres, offsets=offsets,
                         bar_w=bar_w, scale=1.0, add_legend=False)

    ax_time.set_ylabel("Mean step time  (ms)", fontsize=11)
    ax_time.set_ylim(bottom=0)
    ax_time.spines[["top", "right"]].set_visible(False)
    ax_time.grid(axis="y", linewidth=0.4, color="#cccccc", zorder=0)

    ax_time.set_xlim(-0.5, n_groups - 0.5)
    ax_time.set_xticks(group_centres)
    ax_time.set_xticklabels([_tc_label(t) for t in tc_values], fontsize=10)
    ax_time.set_xlabel(
        "Contact time constant, solref[0]  (ms)\n(smaller = stiffer contact)",
        fontsize=10, labelpad=8,
    )

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a directory of contact-time-constant sweep JSONs (merged)."
    )
    parser.add_argument(
        "results_dir", nargs="?", type=Path,
        help="Directory of sweep JSONs. Defaults to latest "
             "contact_timeconst_eval_* dir in results/",
    )
    parser.add_argument(
        "--rollout_dt", type=float, default=None,
        help="Rollout timestep in seconds, used to draw the 2*rollout_dt "
             "stability floor. Defaults to reading rollout_dt from the "
             "directory's meta.json.",
    )
    args = parser.parse_args()

    directory = args.results_dir or _latest_dir()
    print(f"Loading directory: {directory}")

    rollout_dt = _load_rollout_dt(directory, args.rollout_dt)

    records, files = merge_dir(directory)
    print(f"  Merged {len(files)} file(s)")

    models, tc_values, data = parse_records(records)
    if not models:
        raise ValueError("No parseable records found (expected labels like M2_tc20000).")

    print(f"  Models      : {models}")
    print(f"  Time consts : {[_tc_label(t) + ' ms' for t in tc_values]}")
    if rollout_dt is not None:
        print(f"  rollout_dt  : {rollout_dt*1e3:.3g} ms  "
              f"(stability floor {2*rollout_dt*1e3:.3g} ms)")

    task_name = records[0].get("task_name", "unknown") if records else "unknown"
    title     = f"Success rate vs. contact time constant — {task_name}"

    out_path = directory / f"{directory.name}_plot.pdf"
    plot(models, tc_values, data, title, rollout_dt, out_path)


if __name__ == "__main__":
    main()
