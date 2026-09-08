"""CPU-only safeguards against mixing incompatible KL experiment results."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from analysis import plot_kl_divergence_dir as plot_module
from analysis.plot_kl_divergence_dir import (
    above_null,
    family_output_path,
    group_families,
    kl_se_available,
    kl_value,
    main,
    merge_dir,
    null_unavailable_reason,
    pair_with_null,
)


def _record(*, geometry="cube_high_high", seed=0, null=False,
            n_samples=16, ref_samples=4096, value=2.0, **config_changes):
    object_name = geometry.rsplit("_", 2)[0]
    config = {
        "geometry": geometry, "object": object_name,
        "n_samples": n_samples, "n_iterations": 1,
        "ref_n_samples": n_samples if null else ref_samples,
        "ref_n_iterations": 1 if null else 4,
        "comparison_ref_n_samples": ref_samples,
        "comparison_ref_n_iterations": 4,
        "null_control": null, "seed": seed, "n_episodes": 1,
        "temperature": 1.0, "goal_difficulty": 1,
        "eval_sim": "pinocchio", "eval_dt": 0.0005,
        "seed_protocol": "common_random_numbers_v1",
        "kl_direction": "forward", "kl_every": 20,
    }
    config.update(config_changes)
    return {
        "schema_version": 2,
        "label": f"{geometry}_M3_n{n_samples}_i1" + ("_null" if null else ""),
        "model": "M3", "task": "grasp_reorient",
        "geometry": geometry, "object": object_name, "config": config,
        "aggregate": {"task_name": "grasp_reorient", "n_episodes": 1,
                      "success_rate": 1.0, "mean_step_ms": 20.0},
        "kl": {"forward": {"mean": value, "sd": 0.0, "n": 2},
               "reverse": {"mean": value / 2, "sd": 0.0, "n": 2}},
        "per_step": [{"kl_forward": [value, value],
                      "kl_reverse": [value / 2, value / 2]}],
        "episodes": [{"episode_index": 0, "environment_seed": seed * 100 + 7}],
    }


def _write(directory: Path, name: str, data: dict):
    (directory / name).write_text(json.dumps(data), encoding="utf-8")


def test_objects_have_separate_families_and_own_nulls(tmp_path):
    for geometry in ("cube_high_high", "duck_high_high"):
        for null in (False, True):
            _write(tmp_path, f"{geometry}_{null}.json",
                   _record(geometry=geometry, null=null))
    cells, files = merge_dir(tmp_path)
    assert len(files) == 4
    assert len(group_families(cells)) == 2
    reals, nulls = pair_with_null(cells)
    assert len(reals) == len(nulls) == 2
    assert all(c["null_cell"]["geometry"] == c["geometry"] for c in reals)


def test_same_object_different_geometry_is_not_merged_or_null_paired(tmp_path):
    _write(tmp_path, "high_real.json", _record(geometry="cube_high_high"))
    _write(tmp_path, "low_null.json", _record(geometry="cube_low_high", null=True))
    cells, _ = merge_dir(tmp_path)
    assert len(group_families(cells)) == 2
    reals, _ = pair_with_null(cells)
    assert reals[0]["null_cell"] is None


@pytest.mark.parametrize("changed", [
    {"temperature": 2.0},
    {"goal_difficulty": 2},
    {"ref_samples": 2048},
    {"eval_dt": 0.001},
    {"contact_config": {"solver_iterations": 50}},
    {"future_scientific_parameter": "different"},
])
def test_scientific_setting_changes_never_merge_or_pair(tmp_path, changed):
    _write(tmp_path, "real_a.json", _record())
    _write(tmp_path, "real_b.json", _record(seed=1, **changed))
    _write(tmp_path, "null_b.json", _record(null=True, **changed))
    cells, _ = merge_dir(tmp_path)
    assert len(cells) == 3
    assert len(group_families(cells)) == 2
    reals, _ = pair_with_null(cells)
    assert sorted(c["null_cell"] is not None for c in reals) == [False, True]


def test_compatible_distinct_seeds_pool_and_compute_budgets_share_a_plot(tmp_path):
    _write(tmp_path, "seed0.json", _record(seed=0, value=2.0))
    _write(tmp_path, "seed1.json", _record(seed=1, value=6.0))
    _write(tmp_path, "n256.json", _record(seed=0, n_samples=256))
    cells, _ = merge_dir(tmp_path)
    assert len(cells) == 2
    assert len(group_families(cells)) == 1
    pooled = cells["cube_high_high_M3_n16_i1"]
    assert pooled["n_episodes"] == 2
    assert pooled["n_files"] == 2
    assert kl_value(pooled, "forward", "mean") == (4.0, 2.0)


def test_optional_moment_recording_does_not_change_scientific_family(tmp_path):
    _write(tmp_path, "without.json", _record(seed=0, record_kl_moments=False))
    _write(tmp_path, "with.json", _record(seed=1, record_kl_moments=True))
    cells, _ = merge_dir(tmp_path)
    assert len(cells) == 1
    assert next(iter(cells.values()))["n_episodes"] == 2


def test_terminal_scoring_protocols_remain_separate(tmp_path):
    _write(tmp_path, "before.json", _record(seed=0))
    _write(tmp_path, "after.json", _record(seed=1, kl_protocol="first_action_v2_final_state_check"))
    cells, _ = merge_dir(tmp_path)
    assert len(group_families(cells)) == 2


def test_copied_payload_is_not_extra_evidence(tmp_path):
    record = _record()
    _write(tmp_path, "original.json", record)
    _write(tmp_path, "copied_under_another_name.json", record)
    with pytest.raises(ValueError, match="Duplicate result payload"):
        merge_dir(tmp_path)


def test_sanitized_invalid_measurements_are_skipped_not_zeroed(tmp_path):
    record = _record()
    record["per_step"][0]["kl_forward"] = [None, 2.0, float("nan"), 4.0]
    record["per_step"][0]["kl_reverse"] = [None]
    _write(tmp_path, "invalid_measurements.json", record)
    cells, _ = merge_dir(tmp_path)
    cell = next(iter(cells.values()))
    assert kl_value(cell, "forward", "mean") == (3.0, 0.0)
    assert kl_value(cell, "reverse", "mean") == (None, 0.0)


@pytest.mark.parametrize("real_means,null_means", [
    ([100.0], [1.0]),
    ([100.0], [1.0, 1.0]),
    ([100.0, 100.0], [1.0]),
])
def test_single_valid_episode_cannot_pass_mean_null_diagnostic(real_means, null_means):
    real = {"episode_series": {"forward": {"mean": real_means}}}
    null = {"episode_series": {"forward": {"mean": null_means}}}
    real["null_cell"] = null
    assert above_null(real, "forward", "mean") is None
    assert null_unavailable_reason(real, "forward", "mean") == (
        "insufficient replication (<2 valid episodes)"
    )
    assert null_unavailable_reason(null, "forward", "mean") == "no compatible null cell"


def test_single_episode_table_reports_na_and_insufficient_replication(tmp_path, capsys):
    _write(tmp_path, "real.json", _record(value=100.0))
    _write(tmp_path, "null.json", _record(null=True, value=1.0))
    cells, _ = merge_dir(tmp_path)
    reals, _ = pair_with_null(cells)
    plot_module._print_table(reals, "forward", "mean", "episode")
    output = capsys.readouterr().out
    assert "n/a" in output
    assert "insufficient replication" in output
    assert "above null diagnostic" not in output
    assert kl_value(reals[0], "forward", "mean") == (100.0, 0.0)
    assert not kl_se_available(reals[0], "forward", "mean")


def test_median_se_is_unavailable_even_with_multiple_episodes():
    cell = {"episode_series": {"forward": {"median": [1.0, 2.0, 3.0]}}}
    assert kl_value(cell, "forward", "median") == (2.0, 0.0)
    assert not kl_se_available(cell, "forward", "median")


def test_legend_distinguishes_missing_null_from_insufficient_replication(tmp_path):
    _write(tmp_path, "real16.json", _record())
    _write(tmp_path, "null16.json", _record(null=True))
    _write(tmp_path, "real256.json", _record(n_samples=256))
    cells, _ = merge_dir(tmp_path)
    reals, _ = pair_with_null(cells)
    fig, ax = plot_module.plt.subplots()
    try:
        plot_module._scatter_panel(
            ax, reals, "forward", "mean", "episode", "wilson",
            plot_module._sample_colors([16, 256]), False,
        )
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert "null diagnostic: insufficient episodes" in labels
        assert "no compatible null cell" in labels
    finally:
        plot_module.plt.close(fig)


def test_same_seed_numerically_different_repeat_is_not_independent(tmp_path):
    _write(tmp_path, "first.json", _record(seed=0, value=2.0))
    _write(tmp_path, "repeat.json", _record(seed=0, value=2.1))
    with pytest.raises(ValueError, match="Duplicate episode seed identity"):
        merge_dir(tmp_path)


def test_same_environment_seed_with_changed_root_seed_is_also_rejected(tmp_path):
    first, second = _record(seed=0), _record(seed=1)
    second["episodes"][0]["environment_seed"] = first["episodes"][0]["environment_seed"]
    _write(tmp_path, "first.json", first)
    _write(tmp_path, "second.json", second)
    with pytest.raises(ValueError, match="Duplicate episode seed identity"):
        merge_dir(tmp_path)


def _legacy(record):
    data = copy.deepcopy(record)
    data.pop("schema_version")
    data.pop("geometry")
    data.pop("object")
    cfg = data["config"]
    for key in ("geometry", "object", "comparison_ref_n_samples",
                "comparison_ref_n_iterations"):
        cfg.pop(key)
    data["label"] = f"M3_n{cfg['n_samples']}_i1" + ("_null" if cfg["null_control"] else "")
    return data


def test_legacy_unknown_geometry_stays_separate_from_new_schema(tmp_path):
    _write(tmp_path, "old_real.json", _legacy(_record()))
    _write(tmp_path, "old_null.json", _legacy(_record(null=True)))
    _write(tmp_path, "new_real.json", _record())
    cells, _ = merge_dir(tmp_path)
    assert len(group_families(cells)) == 2
    reals, _ = pair_with_null(cells)
    old = next(c for c in reals if c["geometry"] == "unknown (legacy)")
    new = next(c for c in reals if c["geometry"] == "cube_high_high")
    assert old["null_cell"] is not None
    assert new["null_cell"] is None
    assert "M3_n16_i1" in cells  # backward-compatible dictionary key


def test_ambiguous_legacy_null_reference_budget_is_not_guessed(tmp_path):
    _write(tmp_path, "old_real_a.json", _legacy(_record(ref_samples=4096)))
    _write(tmp_path, "old_real_b.json", _legacy(_record(ref_samples=1024)))
    _write(tmp_path, "old_null.json", _legacy(_record(null=True)))
    cells, _ = merge_dir(tmp_path)
    reals, _ = pair_with_null(cells)
    assert len(reals) == 2
    assert all(c["null_cell"] is None for c in reals)


def test_schema_v2_requires_requested_reference_budget(tmp_path):
    data = _record()
    data["config"].pop("comparison_ref_n_samples")
    _write(tmp_path, "incomplete.json", data)
    with pytest.raises(ValueError, match="requires comparison_ref"):
        merge_dir(tmp_path)


def test_label_geometry_conflict_is_rejected(tmp_path):
    data = _record()
    data["label"] = "duck_high_high_M3_n16_i1"
    _write(tmp_path, "bad.json", data)
    with pytest.raises(ValueError, match="label disagrees with recorded geometry"):
        merge_dir(tmp_path)


def test_multi_family_outputs_have_unique_suffixes(tmp_path):
    cube = {"geometry": "cube_high_high", "family_id": "abc123"}
    duck = {"geometry": "duck_high_high", "family_id": "def456"}
    base = tmp_path / "plot.png"
    assert family_output_path(base, cube, False) == base
    assert family_output_path(base, cube, True).name == "plot_cube_high_high_abc123.png"
    assert family_output_path(base, duck, True).name == "plot_duck_high_high_def456.png"


def test_cli_writes_separate_figures_and_supports_geometry_selection(tmp_path, monkeypatch):
    for geometry in ("cube_high_high", "duck_high_high"):
        _write(tmp_path, f"{geometry}.json", _record(geometry=geometry))
    monkeypatch.setattr("sys.argv", ["plot_kl_divergence_dir.py", str(tmp_path),
                                    "--out", str(tmp_path / "multi.png")])
    main()
    assert len(list(tmp_path.glob("multi_*.png"))) == 2
    monkeypatch.setattr("sys.argv", ["plot_kl_divergence_dir.py", str(tmp_path),
                                    "--geometry", "duck_high_high",
                                    "--out", str(tmp_path / "duck_only.png")])
    main()
    assert (tmp_path / "duck_only.png").is_file()


def test_auto_direction_uses_each_family_not_shared_legacy_meta(tmp_path, monkeypatch):
    _write(tmp_path, "forward.json", _record(kl_direction="forward"))
    _write(tmp_path, "reverse.json", _record(kl_direction="reverse"))
    _write(tmp_path, "meta.json", {"kl_direction": "reverse"})
    calls = []

    def capture_plot(reals, direction, *args):
        calls.append((reals[0]["family_settings"]["config"]["kl_direction"], direction))

    monkeypatch.setattr(plot_module, "plot", capture_plot)
    monkeypatch.setattr("sys.argv", ["plot_kl_divergence_dir.py", str(tmp_path)])
    main()
    assert sorted(calls) == [("forward", "forward"), ("reverse", "reverse")]
