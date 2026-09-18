#!/usr/bin/env python3
"""Plot task success rate against one-control-step forward error.

The input directory must contain per-cell JSON files written by
``analysis/run_forward_error_cells.py``.  Replicate files with the same source
label are pooled before plotting, following the structure of the project's
KL-divergence plotter.  The default x error bar is the standard error of the
pooled mean; ``--xerr sd`` displays the sample spread instead.

Examples
--------
python analysis/plot_forward_error_dir.py results/forward_error_eval_12345
python analysis/plot_forward_error_dir.py RESULTS --xerr sd --out plot.pdf
python analysis/plot_forward_error_dir.py RESULTS --x-stat median --xerr none
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODEL_COLORS = {
    "M1": "#4C72B0",
    "M2": "#55A868",
    "M3": "#C44E52",
    "M4": "#8172B2",
}


def _latest_dir() -> Path:
    directories = sorted(
        path for path in RESULTS_DIR.glob("forward_error_*") if path.is_dir()
    )
    if not directories:
        raise FileNotFoundError(
            f"No forward_error_* directories found in {RESULTS_DIR}; "
            "pass a results directory explicitly"
        )
    return directories[-1]


def _load_cells(
    directory: Path, aggregation: str
) -> tuple[list[dict[str, Any]], list[Path]]:
    files = sorted(directory.glob("*_forward_error.json"))
    if not files:
        raise FileNotFoundError(
            f"No *_forward_error.json files found in {directory}"
        )

    accumulators: dict[str, dict[str, Any]] = {}
    for path in files:
        with path.open() as stream:
            result = json.load(stream)
        if result.get("schema") not in {
            "contact-study.forward-error.cell.v1",
            "contact-study.forward-error.cell.v2",
        }:
            print(f"  ! skipping {path.name}: unrecognized schema")
            continue

        source = result["source_cell"]
        label = str(source.get("label") or path.stem)
        acc = accumulators.setdefault(
            label,
            {
                "label": label,
                "task": source["task"],
                "model": source["model"],
                "model_label": source["model_label"],
                "geometry": source["geometry"],
                "n_files": 0,
                "n_episodes": 0,
                "n_success": 0,
                "raw_error": [],
                "raw_episode_error": [],
            },
        )
        identity = (source["task"], source["model"], source["geometry"])
        expected = (acc["task"], acc["model"], acc["geometry"])
        if identity != expected:
            raise ValueError(
                f"label {label!r} identifies inconsistent cells: "
                f"expected {expected}, got {identity} in {path}"
            )

        acc["n_files"] += 1
        acc["n_episodes"] += int(source["n_episodes"])
        acc["n_success"] += int(source["n_success"])
        acc["raw_error"].extend(
            float(sample["primary_error"]["unweighted_sum_m_plus_rad"])
            for sample in result.get("samples", [])
        )
        if result.get("per_episode"):
            acc["raw_episode_error"].extend(
                float(
                    episode["forward_error"][
                        "primary_unweighted_m_plus_rad"
                    ]["mean"]
                )
                for episode in result["per_episode"]
            )
        else:
            # Backward-compatible fallback for files produced before the
            # per-episode summary was added.
            grouped: dict[int, list[float]] = {}
            for sample in result.get("samples", []):
                grouped.setdefault(int(sample["episode_index"]), []).append(
                    float(sample["primary_error"]["unweighted_sum_m_plus_rad"])
                )
            acc["raw_episode_error"].extend(
                sum(values) / len(values) for values in grouped.values()
            )

    cells: list[dict[str, Any]] = []
    for acc in accumulators.values():
        n_episodes = acc["n_episodes"]
        if not acc["raw_error"] or n_episodes <= 0:
            continue
        success_rate = acc["n_success"] / n_episodes
        # A deliberately limited smoke test can select transitions from only a
        # subset of the source episodes.  The success rate still belongs to all
        # source episodes, while an episode-balanced forward-error estimate can
        # only use episodes that contain at least one selected transition.
        # Full analyses normally have one error mean per source episode.
        n_error_episodes = len(acc["raw_episode_error"])
        if n_error_episodes <= 0:
            continue
        raw = sorted(
            acc["raw_error"]
            if aggregation == "step"
            else acc["raw_episode_error"]
        )
        n_error = len(raw)
        mean = sum(raw) / n_error
        variance = max(sum(value * value for value in raw) / n_error - mean * mean, 0.0)
        sd = math.sqrt(variance)
        cells.append(
            {
                **acc,
                "error_mean": mean,
                "error_sd": sd,
                "error_se": sd / math.sqrt(n_error),
                "error_median": _percentile(raw, 50.0),
                "error_p95": _percentile(raw, 95.0),
                "aggregation": aggregation,
                "n_error": n_error,
                "n_error_episodes": n_error_episodes,
                "n_step_samples": len(acc["raw_error"]),
                "success_rate": success_rate,
                "success_rate_se": math.sqrt(
                    max(success_rate * (1.0 - success_rate), 0.0) / n_episodes
                ),
            }
        )
    if not cells:
        raise ValueError(f"No usable forward-error cells found in {directory}")
    cells.sort(key=lambda cell: (cell["model"], cell["label"]))
    return cells, files


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Linear percentile matching NumPy's default, without importing NumPy."""
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _x_value(cell: dict[str, Any], stat: str) -> float:
    return float(cell[f"error_{stat}"])


def _x_error(cell: dict[str, Any], stat: str, error_kind: str) -> float:
    if stat != "mean" or error_kind == "none":
        return 0.0
    return float(cell[f"error_{error_kind}"])


def _short_label(cell: dict[str, Any]) -> str:
    return f"{cell['model']} ({cell['geometry']})"


def _print_table(
    cells: list[dict[str, Any]], stat: str, xerr: str, aggregation: str
) -> None:
    unit_label = "steps" if aggregation == "step" else "episodes"
    print(
        f"\n  {'cell':<25} {'episodes':>8} {'success':>9} "
        f"{'FE':>12} {'x error':>12} {unit_label:>8}"
    )
    for cell in cells:
        value = _x_value(cell, stat)
        error = _x_error(cell, stat, xerr)
        print(
            f"  {_short_label(cell):<25} {cell['n_episodes']:>8d} "
            f"{100.0 * cell['success_rate']:>8.1f}% {value:>12.6g} "
            f"{error:>12.3g} {cell['n_error']:>8d}"
        )


def plot(
    cells: list[dict[str, Any]],
    *,
    stat: str,
    xerr: str,
    log_x: bool,
    title: str,
    output_path: Path,
    aggregation: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.7), layout="constrained")
    one_geometry = len({cell["geometry"] for cell in cells}) == 1
    same_y_occurrence: dict[float, int] = {}
    for cell in cells:
        x = _x_value(cell, stat)
        xe = _x_error(cell, stat, xerr)
        y = 100.0 * cell["success_rate"]
        ye = 100.0 * cell["success_rate_se"]
        color = MODEL_COLORS.get(cell["model"], "#666666")
        ax.errorbar(
            x,
            y,
            xerr=xe if xe > 0.0 else None,
            yerr=ye if ye > 0.0 else None,
            fmt="o",
            markersize=8,
            color=color,
            ecolor="#666666",
            elinewidth=1.0,
            capsize=3.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )
        y_key = round(y, 8)
        occurrence = same_y_occurrence.get(y_key, 0)
        same_y_occurrence[y_key] = occurrence + 1
        if y >= 80.0:
            y_offsets = (8, -20, -36, -52)
        elif y <= 20.0:
            y_offsets = (8, 24, 40, 56)
        else:
            y_offsets = (8, -20, 24, -36)
        annotation = cell["model"] if one_geometry else _short_label(cell)
        ax.annotate(
            annotation,
            (x, y),
            textcoords="offset points",
            xytext=(7, y_offsets[min(occurrence, len(y_offsets) - 1)]),
            fontsize=9,
            color="#333333",
        )

    if log_x:
        if any(_x_value(cell, stat) <= 0.0 for cell in cells):
            raise ValueError("--log-x requires every plotted error to be positive")
        ax.set_xscale("log")
    error_description = {
        "se": "± SE" if stat == "mean" else "",
        "sd": "± SD" if stat == "mean" else "",
        "none": "",
    }[xerr]
    suffix = f" {error_description}" if error_description else ""
    aggregation_label = (
        "step-weighted" if aggregation == "step" else "episode-balanced"
    )
    ax.set_xlabel(
        "One-control-step object forward error\n"
        f"position L2 [m] + SO(3) angle [rad] "
        f"({aggregation_label} {stat}{suffix})",
        fontsize=11,
    )
    ax.set_ylabel("Episode success rate", fontsize=11)
    ax.set_ylim(-5.0, 105.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(linewidth=0.45, color="#D0D0D0", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        help="directory containing *_forward_error.json files",
    )
    parser.add_argument(
        "--aggregation",
        choices=("step", "episode"),
        default="step",
        help=(
            "pool sampled transitions (step), or average within each episode "
            "before pooling episode means (episode)"
        ),
    )
    parser.add_argument(
        "--x-stat",
        choices=("mean", "median", "p95"),
        default="mean",
        help="per-cell forward-error summary shown on the x axis",
    )
    parser.add_argument(
        "--xerr",
        choices=("se", "sd", "none"),
        default="se",
        help="mean-error whisker; ignored for median and p95",
    )
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    directory = (args.results_dir or _latest_dir()).resolve()
    cells, files = _load_cells(directory, args.aggregation)
    print(f"Loaded {len(files)} file(s), pooled into {len(cells)} cell(s)")
    _print_table(cells, args.x_stat, args.xerr, args.aggregation)

    if any(cell["n_episodes"] < 2 for cell in cells):
        print(
            "\n  ! At least one cell has fewer than two episodes. Its binomial "
            "success-rate SE is zero at 0% or 100%; treat this as a pipeline "
            "smoke test, not statistical evidence."
        )
    if args.aggregation == "step" and any(cell["n_error"] < 10 for cell in cells):
        print(
            "  ! At least one cell has fewer than ten forward-error samples; "
            "the x summary and whisker are preliminary."
        )
    if args.aggregation == "episode" and any(cell["n_error"] < 5 for cell in cells):
        print(
            "  ! At least one cell has fewer than five episode means; "
            "the episode-balanced summary and whisker are preliminary."
        )
    if any(
        cell["n_error_episodes"] != cell["n_episodes"] for cell in cells
    ):
        print(
            "  ! At least one cell has sampled errors for only a subset of its "
            "source episodes. The success rate uses every source episode, but "
            "the episode-balanced x value uses only sampled episodes."
        )

    task_names = sorted({cell["task"] for cell in cells})
    title = "Success rate vs. one-step forward error — " + ", ".join(task_names)
    geometries = sorted({cell["geometry"] for cell in cells})
    if len(geometries) == 1:
        title += f" — {geometries[0]}"
    output_path = (
        args.out.resolve()
        if args.out is not None
        else directory / f"{directory.name}_error_vs_success.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot(
        cells,
        stat=args.x_stat,
        xerr=args.xerr,
        log_x=args.log_x,
        title=title,
        output_path=output_path,
        aggregation=args.aggregation,
    )


if __name__ == "__main__":
    main()
