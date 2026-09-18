"""Focused tests for the repository-integrated recorded-log KL workflow."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
REPO = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))

import offline_recorded_kl_common as common
import build_offline_recorded_kl_manifest as manifest_builder
import plot_offline_recorded_kl as plotter
import run_offline_recorded_kl_batch as batch
import run_offline_recorded_kl_episode as episode_worker


def test_default_protocol_is_centralized():
    assert common.DEFAULT_MAX_MEASUREMENTS == 10
    assert common.DEFAULT_REFERENCE_SAMPLES == 4096
    assert common.DEFAULT_REFERENCE_TEMPERATURE == 50
    assert common.DEFAULT_CONVERGENCE_TOL == pytest.approx(1e-3)
    assert common.DEFAULT_MAX_ITERATIONS == 25
    assert common.DEFAULT_SHRINKAGE == pytest.approx(0.001)


def test_even_selection_is_unique_outcome_blind_and_includes_endpoints():
    steps = list(range(0, 4000, 10))
    assert common.evenly_spaced_unique_steps(steps, 10) == [
        0, 440, 890, 1330, 1770, 2220, 2660, 3100, 3550, 3990
    ]
    assert common.evenly_spaced_unique_steps([0, 10, 20], 10) == [0, 10, 20]


@pytest.mark.parametrize("steps", [[], [0, 0, 10], [10, 0]])
def test_bad_recorded_step_sequences_are_rejected(steps):
    with pytest.raises(ValueError):
        common.evenly_spaced_unique_steps(steps, 10)


def test_independent_kl_identity_and_diagonal_case():
    mean = np.array([0.2, -0.7])
    covariance = np.diag([0.3, 1.4])
    assert common.independent_gaussian_kl(mean, covariance, mean, covariance) == pytest.approx(0, abs=1e-12)
    mean_q = np.array([1.0, 2.0])
    covariance_q = np.diag([2.0, 3.0])
    expected = 0.5 * (
        np.trace(np.linalg.solve(covariance_q, covariance))
        + (mean_q - mean) @ np.linalg.solve(covariance_q, mean_q - mean)
        - 2
        + np.log(np.linalg.det(covariance_q) / np.linalg.det(covariance))
    )
    assert common.independent_gaussian_kl(mean, covariance, mean_q, covariance_q) == pytest.approx(expected)


def test_manifest_validation_and_output_name(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cells": [{
        "file": "cell_00000.json", "sha256": "abc", "model": "M1",
        "geometry": "cube_high_high",
        "config": {}, "n": 30, "success": 4,
    }]}))
    _, cells = common.load_cells_manifest(manifest)
    assert cells[0]["file"] == "cell_00000.json"
    assert common.output_filename(cells[0]["file"], 7) == "cell_00000_ep_007_zero_restart_kl.json"


def test_manifest_builder_makes_project_cell_directory_directly_runnable(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    context = {
        "goal_difficulty": 1, "goal_switch_steps": [], "warm_start": False,
        "time_constrained": False, "shrinkage": 0, "action_source": "mean",
        "planner": "mppi", "resample_per_iteration": False,
        "rollout_dt": 0.004, "q0": [0], "v0": [0], "u0": [0],
        "horizon": 5, "substeps": 2, "nu": 1,
        "ctrl_relative_to_qpos": False, "noise_sigma": 0.1,
    }
    trajectory = {
        "context": context,
        "steps": {"step": [0], "qpos": [[0]], "qvel": [[0]], "ctrl": [[0]]},
        "planner_dist": {"step": [0], "mu": [[0]], "cov": [[[1]]],
                         "ess": [1], "degenerate": [False]},
    }
    cell = {
        "driver": "sync", "task": "grasp_reorient", "model": "M3",
        "geometry": "cube_high_high", "planner": "mppi",
        "planner_kwargs": {"n_samples": 64, "n_iterations": 1,
                           "temperature": 12.5},
        "seed": 0, "combo_index": 0, "full_weights": {},
        "n_episodes": 2, "n_success": 1,
        "episodes": [
            {"success": True, "end_reason": "success", "trajectory": trajectory},
            {"success": False, "end_reason": "timeout", "trajectory": trajectory},
        ],
    }
    (source_dir / "cell_00000.json").write_text(json.dumps(cell))
    output = tmp_path / "manifest.json"
    manifest = manifest_builder.build_manifest(source_dir, output)
    _, loaded = common.load_cells_manifest(output)
    assert manifest["schema"] == manifest_builder.SCHEMA
    assert loaded == manifest["cells"]
    assert loaded[0]["n"] == 2
    assert loaded[0]["success"] == 1
    assert loaded[0]["sha256"] == common.sha256_file(source_dir / "cell_00000.json")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        manifest_builder.build_manifest(source_dir, output)


def test_manifest_builder_rejects_unsupported_or_incomplete_cells(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    incomplete = {
        "driver": "sync", "task": "cart_pole", "model": "M1",
        "geometry": "accurate", "planner": "mppi", "planner_kwargs": {},
        "seed": 0, "combo_index": 0, "full_weights": {},
        "n_episodes": 0, "n_success": 0, "episodes": [],
    }
    (source_dir / "cell_00000.json").write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="supported grasp_reorient"):
        manifest_builder.build_manifest(source_dir, tmp_path / "manifest.json")


def test_scientific_provenance_covers_runtime_backend_and_resolved_scene_assets():
    provenance = common.scientific_runtime_provenance(
        REPO, "cube_high_high", "M1"
    )
    local = provenance["repository_files_sha256"]
    for required in (
        "contact_study/contact_models/api.py",
        "contact_study/contact_models/config.py",
        "contact_study/contact_models/xpbd_backend.py",
        "contact_study/drivers/run_eval_episode.py",
        "contact_study/tasks/base.py",
        "contact_study/tasks/config.py",
    ):
        assert required in local
    scene = provenance["resolved_scene_assets"]
    assert scene["entrypoint"] == "scenes/leap/env_leap_rollout_cube_high_high.xml"
    assert scene["resolved_file_count"] > 1
    assert "scenes/leap/leap_right_hand_high.xml" in scene["resolved_files_sha256"]
    assert scene["resolved_bundle_sha256"]
    assert provenance["external_backend"]["python_file_count"] > 1
    assert provenance["external_backend"]["python_tree_sha256"]
    assert provenance["workflow_sha256"]


def test_batch_parser_uses_default_protocol(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    args = batch.build_parser().parse_args([
        "--source-dir", str(source), "--manifest", str(manifest),
        "--outdir", str(tmp_path / "out"),
    ])
    assert batch.expected_settings(args) == {
        "n_samples": 4096,
        "temperature": 50.0,
        "convergence_tol": 0.001,
        "max_iterations": 25,
        "shrinkage": 0.001,
        "max_measurements": 10,
    }


def test_completed_episode_cache_is_reused_only_for_exact_protocol(tmp_path):
    item = {"cell": {"file": "cell_00000.json", "sha256": "source", "model": "M1"},
            "episode": 2}
    settings = {"n_samples": 4096, "temperature": 50.0, "convergence_tol": .001,
                "max_iterations": 25, "shrinkage": .001, "max_measurements": 10}
    record = {
        "completed": True, "schema": common.SCHEMA, "source": "cell_00000.json",
        "source_sha256": "source", "episode": 2, "analysis_script_sha256": "worker",
        "analysis_support_sha256": "common",
        "scientific_runtime_provenance": {"workflow_sha256": "runtime"},
        "shrinkage": .001,
        "reference_cross_state_history": "none", "source_outcome_unchanged": True,
        "closed_loop_rerun": False,
        "reference_config": {"n_samples": 4096, "temperature": 50.0,
                             "convergence_tol": .001, "max_iterations": 25},
        "selection": {"max_measurements": 10},
    }
    provenance = {
        "worker_sha256": "worker", "common_sha256": "common",
        "scientific_by_source": {
            "cell_00000.json": {"workflow_sha256": "runtime"}
        },
    }
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(record))
    assert batch.validate_cached(path, item, settings, provenance)
    record["reference_config"]["max_iterations"] = 10
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        batch.validate_cached(path, item, settings, provenance)


def test_common_prefix_never_mixes_unbalanced_episode_indices(tmp_path):
    cells = [{"file": "cell_00000.json", "n": 30}, {"file": "cell_00001.json", "n": 30}]
    for episode in range(2):
        for cell in cells:
            path = tmp_path / common.output_filename(cell["file"], episode)
            path.write_text(json.dumps({"completed": True}))
    # A later completion in just one cell must not enter the prefix.
    (tmp_path / common.output_filename("cell_00000.json", 2)).write_text(json.dumps({"completed": True}))
    assert plotter.common_completed_prefix(tmp_path, cells) == 2


def test_episode_balanced_aggregation_is_not_measurement_pooled():
    episodes = [
        {"selected_measurements": 10, "valid_measurements": 10, "invalid_measurements": 0,
         "converged_measurements": 10, "nonconverged_measurements": 0,
         "mean_kl_all_valid": 2.0, "mean_kl_converged_only": 1.0},
        {"selected_measurements": 3, "valid_measurements": 3, "invalid_measurements": 0,
         "converged_measurements": 0, "nonconverged_measurements": 3,
         "mean_kl_all_valid": 8.0, "mean_kl_converged_only": None},
    ]
    result = plotter.aggregate_config(episodes)
    assert result["all_valid"]["mean_kl"] == 5.0
    assert result["all_valid"]["sd_episode_means"] == 3.0
    assert result["converged_only"]["mean_kl"] == 1.0
    assert result["converged_only"]["episodes"] == 1
    assert result["valid_measurements"] == 13


def test_exact_zero_kl_is_valid_and_uses_zero_safe_axis():
    episodes = [{
        "selected_measurements": 1, "valid_measurements": 1,
        "invalid_measurements": 0, "converged_measurements": 1,
        "nonconverged_measurements": 0, "mean_kl_all_valid": 0.0,
        "mean_kl_converged_only": 0.0,
    }]
    aggregate = plotter.aggregate_config(episodes)
    assert aggregate["all_valid"]["mean_kl"] == 0.0
    figure, axis = plotter.plt.subplots()
    plotter.draw_panel(axis, [{
        "all_valid": aggregate["all_valid"], "model": "M1",
        "acting_config": {"n_iterations": 1, "n_samples": 16, "temperature": 50},
        "source_success": 1, "source_episodes": 2,
    }], "all_valid", "zero", "coverage")
    assert axis.get_xscale() == "symlog"
    figure.canvas.draw()
    plotter.plt.close(figure)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("argument", ["--temperature", "--convergence-tol"])
def test_nonfinite_reference_float_is_rejected_by_both_clis(tmp_path, value, argument):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_json = source_dir / "cell.json"
    source_json.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    episode_parser = episode_worker.build_parser()
    episode_args = episode_parser.parse_args([
        "--source-json", str(source_json), "--episode", "0",
        "--outdir", str(tmp_path / "episode"), f"{argument}={value}",
    ])
    with pytest.raises(SystemExit):
        episode_worker.validate_args(episode_args, episode_parser)

    batch_parser = batch.build_parser()
    batch_args = batch_parser.parse_args([
        "--source-dir", str(source_dir), "--manifest", str(manifest),
        "--outdir", str(tmp_path / "batch"), f"{argument}={value}",
    ])
    with pytest.raises(SystemExit):
        batch.validate_args(batch_args, batch_parser)
