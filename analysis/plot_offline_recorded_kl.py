"""Create a validated two-panel KL-vs-success plot from offline episode outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))

from offline_recorded_kl_common import (
    SCHEMA,
    array_digest,
    evenly_spaced_unique_steps,
    independent_gaussian_kl,
    load_cells_manifest,
    output_filename,
    require,
    sha256_bytes,
    sha256_file,
)

COLORS = {"M1": "#2864A5", "M2": "#DD8B26", "M3": "#27917E", "M4": "#9458AC"}
ITERATION_MARKERS = {1: "o", 2: "D"}
SAMPLE_SIZES = {16: 75, 64: 135, 256: 225}
LABEL_LAYOUT = {
    "M1": (6, 8, "left"),
    "M2": (6, -11, "left"),
    "M3": (6, 8, "left"),
    "M4": (-6, -11, "right"),
}


def wilson(successes, total):
    proportion, z = successes / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    width = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return 100 * (center - width), 100 * (center + width)


def configure_plotting():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def load_source_once(source_dir, cell):
    path = Path(source_dir) / cell["file"]
    payload = path.read_bytes()
    checksum = sha256_bytes(payload)
    require(checksum == cell["sha256"], f"Source hash mismatch: {path}")
    return {"source": json.loads(payload), "checksum": checksum}


def validate_episode(path, cell, episode_index, source_bundle, batch):
    record = json.loads(Path(path).read_text())
    require(record["schema"] == SCHEMA and record["completed"] is True,
            f"Incomplete or unknown episode output: {path}")
    require(record["source"] == cell["file"] and record["episode"] == episode_index,
            f"Source/episode mismatch: {path}")
    require(record["source_sha256"] == cell["sha256"] == source_bundle["checksum"],
            f"Source provenance mismatch: {path}")
    require(record["analysis_script_sha256"] == batch["worker_sha256"],
            f"Worker version mismatch: {path}")
    common_key = "analysis/offline_recorded_kl_common.py"
    require(record["analysis_support_sha256"] == batch["analysis_files_sha256"][common_key],
            f"Shared analysis-code version mismatch: {path}")
    require(
        record["scientific_runtime_provenance"]
        == batch["scientific_runtime_provenance_by_source"][cell["file"]],
        f"Scientific code/scene provenance mismatch: {path}",
    )
    source = source_bundle["source"]
    require(record["model"] == source["model"] == cell["model"], "Model mismatch")
    require(record["acting_config"] == source["planner_kwargs"] == cell["config"],
            "Acting configuration mismatch")
    require(len(source["episodes"]) == cell["n"], "Source episode total mismatch")
    require(sum(bool(ep["success"]) for ep in source["episodes"]) == cell["success"],
            "Source success total mismatch")
    episode = source["episodes"][episode_index]
    require(record["success"] == episode["success"] and record["end_reason"] == episode["end_reason"],
            "Recorded outcome changed")
    require(record["source_outcome_unchanged"] is True and record["closed_loop_rerun"] is False,
            "Output is not a read-only offline comparison")
    require(record["reference_cross_state_history"] == "none", "Reference retained cross-state mean history")
    preset = batch["reference_preset"]
    require(record["shrinkage"] == preset["shrinkage"], "Shrinkage mismatch")
    for key in ("n_samples", "temperature", "convergence_tol", "max_iterations"):
        require(record["reference_config"][key] == preset[key], f"Reference {key} mismatch")
    require(record["reference_config"]["resample_interval"] == 1, "No fresh per-state noise")
    require(record["reference_config"]["resample_per_iteration"] is False,
            "Noise changed inside convergence iterations")

    trajectory = episode["trajectory"]
    context, states, acting = trajectory["context"], trajectory["steps"], trajectory["planner_dist"]
    require(record["selection"]["max_measurements"] == preset["max_measurements"],
            "Measurement-cap mismatch")
    require(record["selection"]["selection_uses_outcome_or_kl"] is False,
            "Measurement selection is not outcome-blind")
    selected = evenly_spaced_unique_steps(acting["step"], preset["max_measurements"])
    require(record["selected_steps"] == selected, "Evenly spaced membership mismatch")
    require([row["step"] for row in record["measurements"]] == selected,
            "Measurement membership/order mismatch")
    state_lookup = {int(step): index for index, step in enumerate(states["step"])}
    acting_lookup = {int(step): index for index, step in enumerate(acting["step"])}
    goal = np.asarray(record["goal"], dtype=np.float32)
    previous_final = None
    measurements = []
    noise_keys = []
    invalid = 0
    for measurement_index, row in enumerate(record["measurements"]):
        step = row["step"]
        require(row["measurement_index"] == measurement_index, "Measurement index mismatch")
        require(row["input_preservation_passed"] is True, "Reference changed its input")
        require(row["reference_zero_reset_passed"] is True and row["reference_zero_reset_max_abs"] == 0,
                "Per-state zero reset failed")
        require(np.count_nonzero(row["reference_mean_after_reset"]) == 0,
                "Saved post-reset mean is nonzero")
        before_reset = np.asarray(row["reference_mean_before_reset"])
        if measurement_index == 0:
            require(np.count_nonzero(before_reset) == 0 and not row["reference_previous_mean_discarded"],
                    "First state reset audit is inconsistent")
        else:
            np.testing.assert_array_equal(before_reset, previous_final)
            require(row["reference_previous_mean_discarded"] is True,
                    "Previous state mean was not discarded")
        previous_final = np.asarray(row["reference_mean_after_solve"])
        require(row["reference_fresh_noise_draw_passed"] is True, "Fresh-noise audit failed")
        require(row["reference_resample_count_after"] == row["reference_resample_count_before"] + 1,
                "Expected exactly one fresh noise block")
        noise_keys.append(row["reference_noise_key"])

        state_index = state_lookup[step]
        inputs = {
            "qpos": np.asarray(states["qpos"][state_index], dtype=np.float64),
            "qvel": np.asarray(states["qvel"][state_index], dtype=np.float64),
            "ctrl": np.asarray(context["u0"] if step == 0 else states["ctrl"][state_lookup[step - 1]],
                               dtype=np.float64),
        }
        for name, value in inputs.items():
            require(row["input_hashes"][name] == array_digest(value),
                    f"Physical input hash mismatch: {name}, step {step}")
        require(row["input_hashes"]["goal"] == array_digest(goal), "Goal hash mismatch")

        iterations = row["reference_iterations"]
        residuals = row["reference_update_squared_l2"]
        converged = bool(residuals and residuals[-1] < preset["convergence_tol"])
        require(row["reference_converged"] == converged, "Convergence flag mismatch")
        if row["status"] != "valid_numerical_measurement":
            require(row["status"] in {"invalid_recorded_acting_distribution", "invalid_reference_solve"},
                    "Unknown invalid status")
            if row["status"] == "invalid_recorded_acting_distribution":
                require(acting["degenerate"][acting_lookup[step]], "Acting distribution not actually degenerate")
            invalid += 1
            continue
        require(2 <= iterations <= preset["max_iterations"], "Valid solve has invalid iteration count")
        require(len(residuals) == iterations - 1, "Residual/iteration count mismatch")
        if iterations < preset["max_iterations"]:
            require(converged, "Reference stopped early without convergence")
        require(row["inclusion_all_valid"] is True and row["inclusion_converged_only"] == converged,
                "Panel-inclusion flags mismatch")
        acting_index = acting_lookup[step]
        require(not acting["degenerate"][acting_index], "Degenerate acting distribution labelled valid")
        acting_mean = np.asarray(row["acting_mean"], dtype=np.float64)
        acting_cov_raw = np.asarray(row["acting_covariance_raw"], dtype=np.float64)
        np.testing.assert_array_equal(acting_mean, np.asarray(acting["mu"][acting_index], dtype=np.float64))
        np.testing.assert_array_equal(acting_cov_raw, np.asarray(acting["cov"][acting_index], dtype=np.float64))
        require(row["acting_mean_sha256"] == array_digest(acting_mean), "Acting mean digest mismatch")
        require(row["acting_covariance_sha256"] == array_digest(acting_cov_raw),
                "Acting covariance digest mismatch")
        reference_mean = np.asarray(row["reference_mean"], dtype=np.float64)
        reference_cov_raw = np.asarray(row["reference_covariance_raw"], dtype=np.float64)
        target = context["noise_sigma"] ** 2 * np.eye(context["nu"])
        alpha = preset["shrinkage"]
        reference_cov = (1 - alpha) * 0.5 * (reference_cov_raw + reference_cov_raw.T) + alpha * target
        acting_cov = (1 - alpha) * 0.5 * (acting_cov_raw + acting_cov_raw.T) + alpha * target
        np.linalg.cholesky(reference_cov)
        np.linalg.cholesky(acting_cov)
        forward = independent_gaussian_kl(reference_mean, reference_cov, acting_mean, acting_cov)
        reverse = independent_gaussian_kl(acting_mean, acting_cov, reference_mean, reference_cov)
        require(np.isfinite([forward, reverse]).all() and min(forward, reverse) >= -1e-8,
                "Independent KL is invalid")
        np.testing.assert_allclose(forward, row["kl_ref_to_acting"], rtol=1e-9, atol=1e-7)
        np.testing.assert_allclose(reverse, row["kl_acting_to_ref"], rtol=1e-9, atol=1e-7)
        np.testing.assert_allclose(forward, row["kl_independent_audit_ref_to_acting"], rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(reverse, row["kl_independent_audit_acting_to_ref"], rtol=1e-12, atol=1e-10)
        require(row["kl_independent_audit_passed"] is True, "Worker KL audit is false")
        measurements.append({
            "step": step,
            "kl": max(0.0, forward),
            "converged": converged,
            "iterations": iterations,
            "final_residual": residuals[-1],
        })
    require(len(noise_keys) == len(set(noise_keys)), "Noise key reused within episode")
    valid = len(measurements)
    converged_count = sum(item["converged"] for item in measurements)
    summary = record["summary"]
    require(summary["selected_measurements"] == len(selected), "Selected count summary mismatch")
    require(summary["valid_numerical_measurements"] == valid, "Valid count summary mismatch")
    require(summary["converged_valid_measurements"] == converged_count,
            "Converged count summary mismatch")
    require(summary["nonconverged_valid_measurements"] == valid - converged_count,
            "Nonconverged count summary mismatch")
    require(summary["invalid_measurements"] == invalid, "Invalid count summary mismatch")
    all_values = [item["kl"] for item in measurements]
    converged_values = [item["kl"] for item in measurements if item["converged"]]
    return {
        "episode": episode_index,
        "success": bool(episode["success"]),
        "end_reason": episode["end_reason"],
        "selected_measurements": len(selected),
        "valid_measurements": valid,
        "invalid_measurements": invalid,
        "converged_measurements": converged_count,
        "nonconverged_measurements": valid - converged_count,
        "mean_kl_all_valid": float(np.mean(all_values)) if all_values else None,
        "mean_kl_converged_only": float(np.mean(converged_values)) if converged_values else None,
        "measurements": measurements,
    }


def common_completed_prefix(batch_dir, cells):
    for episode_index in range(min(cell["n"] for cell in cells)):
        for cell in cells:
            path = Path(batch_dir) / output_filename(cell["file"], episode_index)
            if not path.exists():
                return episode_index
            try:
                if json.loads(path.read_text()).get("completed") is not True:
                    return episode_index
            except (OSError, json.JSONDecodeError):
                return episode_index
    return min(cell["n"] for cell in cells)


def aggregate_config(episodes):
    def panel(key):
        values = np.asarray([ep[key] for ep in episodes if ep[key] is not None], dtype=float)
        require(not len(values) or (np.isfinite(values).all() and np.all(values >= 0)),
                f"Invalid episode means for {key}")
        return {
            "episodes": len(values),
            "episodes_excluded": len(episodes) - len(values),
            "mean_kl": float(values.mean()) if len(values) else None,
            "sd_episode_means": float(values.std(ddof=0)) if len(values) else None,
            "episode_means": values.tolist(),
        }
    return {
        "episodes": len(episodes),
        "selected_measurements": sum(ep["selected_measurements"] for ep in episodes),
        "valid_measurements": sum(ep["valid_measurements"] for ep in episodes),
        "invalid_measurements": sum(ep["invalid_measurements"] for ep in episodes),
        "converged_measurements": sum(ep["converged_measurements"] for ep in episodes),
        "nonconverged_measurements": sum(ep["nonconverged_measurements"] for ep in episodes),
        "all_valid": panel("mean_kl_all_valid"),
        "converged_only": panel("mean_kl_converged_only"),
    }


def draw_panel(axis, rows, key, title, coverage):
    plotted_values = []
    for row in rows:
        panel = row[key]
        if panel["mean_kl"] is None:
            continue
        plotted_values.extend(panel["episode_means"])
        plotted_values.append(panel["mean_kl"])
        config, model = row["acting_config"], row["model"]
        marker = ITERATION_MARKERS.get(config["n_iterations"], "P")
        size = SAMPLE_SIZES.get(config["n_samples"], 120)
        success = 100 * row["source_success"] / row["source_episodes"]
        low, high = wilson(row["source_success"], row["source_episodes"])
        if len(panel["episode_means"]) > 1:
            axis.scatter(panel["episode_means"], [success] * len(panel["episode_means"]),
                         marker=marker, s=max(18, size * 0.18), color=COLORS[model],
                         alpha=0.18, linewidths=0, zorder=1)
        axis.errorbar(panel["mean_kl"], success,
                      yerr=np.maximum(0, [[success - low], [high - success]]),
                      fmt="none", ecolor=COLORS[model], capsize=4, alpha=0.8, zorder=2)
        axis.scatter([panel["mean_kl"]], [success], marker=marker, s=size,
                     color=COLORS[model], edgecolor="white", linewidth=1.2, zorder=3)
        dx, dy, alignment = LABEL_LAYOUT.get(model, (6, 8, "left"))
        axis.annotate(f"{config['n_samples']}x{config['n_iterations']}, T={config['temperature']:g}",
                      (panel["mean_kl"], success), xytext=(dx, dy), textcoords="offset points",
                      ha=alignment, fontsize=7.3, color=COLORS[model],
                      bbox={"boxstyle": "round,pad=.15", "fc": "white", "ec": "none", "alpha": .72})
    positive = [value for value in plotted_values if value > 0]
    linear_threshold = min(positive) / 10 if positive else 1.0
    axis.set_xscale("symlog", linthresh=linear_threshold, linscale=0.6)
    axis.set(ylim=(-7, 109),
             xlabel="Episode-balanced mean KL(reference || recorded acting), nats [symmetric-log; zero supported]",
             ylabel="Recorded task success rate (%)")
    axis.set_title(title, fontweight="bold")
    axis.grid(True, which="major", alpha=.22)
    axis.grid(True, which="minor", axis="x", alpha=.08)
    axis.margins(x=.16)
    axis.text(.02, .02, coverage, transform=axis.transAxes, va="bottom", fontsize=8.5,
              bbox={"boxstyle": "round,pad=.35", "fc": "#F4F7FA", "ec": "#CCD6E0", "alpha": .95})


def build(source_dir, manifest_path, batch_dir, outdir, episodes_per_cell=None):
    manifest_document, cells = load_cells_manifest(manifest_path)
    batch = json.loads((Path(batch_dir) / "execution_manifest.json").read_text())
    require(batch["input_manifest_sha256"] == sha256_file(manifest_path),
            "Batch and plotting input manifests differ")
    require(Path(batch["source_directory"]).resolve() == Path(source_dir).resolve(),
            "Batch and plotting source directories differ")
    require(batch["protocol"] == "independent zero-mean restart at every selected recorded state",
            "Batch protocol is not per-state zero restart")
    available = common_completed_prefix(batch_dir, cells)
    count = available if episodes_per_cell is None else episodes_per_cell
    require(1 <= count <= available,
            f"Requested {count} episodes per cell, but common completed prefix is {available}")
    rows = []
    for cell in cells:
        source_bundle = load_source_once(source_dir, cell)
        episodes = [
            validate_episode(Path(batch_dir) / output_filename(cell["file"], index),
                             cell, index, source_bundle, batch)
            for index in range(count)
        ]
        aggregate = aggregate_config(episodes)
        rows.append({
            "source": cell["file"], "source_sha256": cell["sha256"],
            "model": cell["model"], "acting_config": cell["config"],
            "source_episodes": cell["n"], "source_success": cell["success"],
            **aggregate, "episode_summaries": episodes,
        })
        del source_bundle
    outdir = Path(outdir)
    if outdir.exists():
        raise FileExistsError(f"Refusing to overwrite plot export: {outdir}")
    outdir.mkdir(parents=True)
    total_episodes = len(cells) * count
    selected = sum(row["selected_measurements"] for row in rows)
    valid = sum(row["valid_measurements"] for row in rows)
    invalid = sum(row["invalid_measurements"] for row in rows)
    converged = sum(row["converged_measurements"] for row in rows)
    nonconverged = sum(row["nonconverged_measurements"] for row in rows)
    all_episodes = sum(row["all_valid"]["episodes"] for row in rows)
    converged_episodes = sum(row["converged_only"]["episodes"] for row in rows)

    configure_plotting()
    figure, axes = plt.subplots(1, 2, figsize=(18, 8.7))
    draw_panel(axes[0], rows, "all_valid", "A  All finite valid KL measurements",
               f"KL episodes retained: {all_episodes}/{total_episodes}\n"
               f"Measurements: {valid:,}/{selected:,} valid\n"
               f"Nonconverged retained: {nonconverged:,}/{valid:,}")
    draw_panel(axes[1], rows, "converged_only", "B  Converged-reference measurements only",
               f"KL episodes retained: {converged_episodes}/{total_episodes}\n"
               f"Measurements retained: {converged:,}/{valid:,}\n"
               f"Nonconverged excluded: {nonconverged:,}")
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=9,
                      markerfacecolor=color, markeredgecolor="white", label=model)
               for model, color in COLORS.items()]
    handles += [Line2D([0], [0], marker=marker, linestyle="none", markersize=8,
                       markerfacecolor="#555", markeredgecolor="white",
                       label=f"acting iterations = {iterations}")
                for iterations, marker in ITERATION_MARKERS.items()]
    handles += [Line2D([0], [0], marker="o", linestyle="none",
                       markersize=math.sqrt(size) * .72, markerfacecolor="none",
                       markeredgecolor="#555", label=f"acting samples = {samples}")
                for samples, size in SAMPLE_SIZES.items()]
    figure.legend(handles=handles, loc="lower center", ncol=9, frameon=False,
                  bbox_to_anchor=(.5, .065), fontsize=9)
    scope = "all recorded episodes" if count == min(cell["n"] for cell in cells) else f"balanced prefix: first {count} episode(s) per configuration"
    preset = batch["reference_preset"]
    figure.suptitle(
        "Per-state zero-restart reference KL vs recorded task success\n"
        f"{scope} | {len(cells)} configurations | {total_episodes} KL episodes | SR uses every original outcome\n"
        f"Reference: N={preset['n_samples']}, T={preset['temperature']:g}, cap={preset['max_iterations']}, "
        f"squared-L2 threshold={preset['convergence_tol']:g} | up to {preset['max_measurements']} states/episode",
        fontsize=13, y=.965)
    figure.text(.5, .022,
                "Large marks: mean of per-episode means; faint marks: individual episode means. Vertical: Wilson 95%. "
                "Color=model, shape=acting iterations, size=acting samples; acting temperature is labeled.",
                ha="center", fontsize=9, color="#3F4C5A")
    figure.subplots_adjust(left=.065, right=.985, bottom=.19, top=.80, wspace=.17)
    figure.savefig(outdir / "KL_vs_SR_all_valid_and_converged_only.png", dpi=200)
    figure.savefig(outdir / "KL_vs_SR_all_valid_and_converged_only.svg")
    plt.close(figure)

    package = {
        "schema": "offline_recorded_kl_plot_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plot_script_sha256": sha256_file(Path(__file__)),
        "analysis_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ANALYSIS.parent, text=True
        ).strip(),
        "analysis_worktree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ANALYSIS.parent, text=True,
        ).strip()),
        "source_directory": str(Path(source_dir).resolve()),
        "input_manifest": str(Path(manifest_path).resolve()),
        "batch_directory": str(Path(batch_dir).resolve()),
        "batch_status_at_export": batch["status"],
        "available_common_completed_prefix": available,
        "episodes_per_configuration": count,
        "balanced_prefix": True,
        "configurations": len(cells),
        "kl_episodes": total_episodes,
        "selected_measurements": selected,
        "valid_measurements": valid,
        "invalid_measurements": invalid,
        "converged_measurements": converged,
        "nonconverged_measurements": nonconverged,
        "all_valid_episodes_retained": all_episodes,
        "converged_only_episodes_retained": converged_episodes,
        "aggregation": {
            "all_valid": "mean valid KL within episode, then equal-weight mean of eligible episode means",
            "converged_only": "mean converged KL within episode, then equal-weight mean of eligible episode means",
            "measurement_pooled_mean_used": False,
        },
        "reference_preset": preset,
        "kl_direction": "KL(reference || recorded acting)",
        "y_interval": "Wilson 95% using every source outcome per configuration",
        "manifest_metadata": {key: value for key, value in manifest_document.items() if key != "cells"},
        "validation": [
            "source/output/worker SHA-256 and scientific workflow digest",
            "resolved scene assets and original outcomes/acting configurations",
            "common completed prefix", "outcome-blind evenly spaced membership",
            "exact per-state zero mean and previous-mean discard", "fresh-noise counters",
            "physical state/control/goal hashes", "unchanged recorded acting moments",
            "positive-definite regularized covariance", "both KL directions independently recalculated",
            "convergence and panel-inclusion flags",
        ],
        "limitations": [
            "The higher-compute reference is not certified globally optimal.",
            "Converged-only filtering may preferentially remove difficult states or episodes.",
            "Acting temperature and visited trajectory vary between configurations.",
            "The reconstructed goal is not explicitly serialized in the recorded log.",
            "Gaussian KL summarizes the first action, not a complete trajectory distribution.",
        ],
        "cells": rows,
    }
    (outdir / "plot_data_and_validation.json").write_text(
        json.dumps(package, indent=2, allow_nan=False) + "\n"
    )
    (outdir / "README.md").write_text(
        f"""# Offline recorded-log KL vs success

This export uses the same completed prefix of {count} episode(s) from each of {len(cells)} configurations. It contains {total_episodes} KL episodes and {valid} finite valid measurements; {invalid} selected measurements were numerically invalid. Success rates use every original source outcome and no closed-loop task is rerun.

Each selected state independently restarts the higher-compute reference at zero mean. Panel A retains nonconverged but finite KL values. Panel B excludes them; it contains {converged} measurements and {converged_episodes} eligible episode means. Filtering can remove difficult states non-randomly, so the two panels are complementary sensitivity views.

Aggregation first averages measurements within each episode, then gives each eligible episode equal weight. Large markers show these configuration-level means; faint marks show episode means. Vertical intervals are Wilson 95% success intervals. The x-axis is symmetric-logarithmic so exact zero KL remains visible; KL is in nats. Population SD of episode means is saved in the JSON; no horizontal confidence interval is claimed.

Reference settings are recorded in `plot_data_and_validation.json`. The higher-compute reference is not a proof of global optimality. Acting temperatures and visited trajectories vary across configurations, so this descriptive figure does not establish a causal KL-success relationship.
"""
    )
    archive = outdir.with_suffix(".zip")
    pending = archive.with_suffix(".zip.pending")
    with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(outdir.iterdir()):
            if path.is_file():
                bundle.write(path, path.name)
    with zipfile.ZipFile(pending) as bundle:
        require(bundle.testzip() is None, "ZIP integrity test failed")
        for name in bundle.namelist():
            require(bundle.read(name) == (outdir / name).read_bytes(), f"ZIP byte mismatch: {name}")
    pending.replace(archive)
    print(json.dumps({"output": str(outdir), "archive": str(archive),
                      "episodes_per_configuration": count, "valid_measurements": valid,
                      "converged_measurements": converged}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--episodes-per-cell", type=int)
    args = parser.parse_args(argv)
    build(args.source_dir, args.manifest, args.batch_dir, args.outdir, args.episodes_per_cell)


if __name__ == "__main__":
    main()
