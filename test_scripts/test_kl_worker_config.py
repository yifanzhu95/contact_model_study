"""CPU checks for KL geometry selection and non-overwriting result identities."""

import pytest

from experiments.hpc.run_kl_divergence_cell import (
    build_parser, initialize_reference, reference_init_mode,
    resolve_kl_geometry, result_filename, validate_kl_args,
)


@pytest.mark.parametrize("obj", ["cube", "duck", "ball", "spam", "tomato"])
def test_object_shorthand_uses_high_high(obj):
    assert resolve_kl_geometry("grasp_reorient", obj) == f"{obj}_high_high"
    assert resolve_kl_geometry("grasp_reorient", f"{obj}_high_high") == f"{obj}_high_high"


@pytest.mark.parametrize("value", ["accurate", "linearized", "convex_hull",
                                 "primitive_union", "cube_low_high", "duck_high_low", ""])
def test_rejects_ambiguous_or_lower_fidelity_geometry(value):
    with pytest.raises(ValueError):
        resolve_kl_geometry("grasp_reorient", value)


def test_default_geometry_is_explicit():
    assert build_parser().parse_args([]).geometry == "cube_high_high"


def test_reference_defaults_match_per_state_zero_protocol():
    args = build_parser().parse_args([])
    assert reference_init_mode(args) == "zero"
    assert args.ref_n_samples == 4096
    assert args.ref_temperature == pytest.approx(50.0)
    assert args.temperature == pytest.approx(1.0)
    assert args.ref_convergence_tol == pytest.approx(1e-3)
    assert args.ref_max_iterations == 25


def test_legacy_reference_initializations_are_explicit():
    parser = build_parser()
    assert reference_init_mode(parser.parse_args(
        ["--reference_init", "degraded_pre_solve"]
    )) == "degraded_pre_solve"
    assert reference_init_mode(parser.parse_args(
        ["--reference_init", "persistent"]
    )) == "persistent"
    # Old command-line spellings remain aliases, but are no longer defaults.
    assert reference_init_mode(parser.parse_args(["--sync_reference_mean"])) == \
        "degraded_pre_solve"
    assert reference_init_mode(parser.parse_args(["--no-sync_reference_mean"])) == \
        "persistent"
    # Optional null diagnostics force a same-proposal comparison regardless of
    # the real-reference zero-start default.
    assert reference_init_mode(parser.parse_args(["--null_control"])) == \
        "degraded_pre_solve"


def test_reference_initialization_actions():
    class Buffer:
        def __init__(self):
            self.value = None
        def assign(self, value):
            self.value = value.copy()

    class Reference:
        def __init__(self):
            self.U_wp = Buffer()
            self.reset_count = 0
        def reset(self):
            self.reset_count += 1
            self.U_wp.value = "zero"

    reference = Reference()
    initialize_reference(reference, "zero")
    assert reference.reset_count == 1
    assert reference.U_wp.value == "zero"

    import numpy as np
    mean = np.ones((2, 3))
    initialize_reference(reference, "degraded_pre_solve", mean)
    np.testing.assert_array_equal(reference.U_wp.value, mean)
    assert reference.reset_count == 1

    initialize_reference(reference, "persistent")
    np.testing.assert_array_equal(reference.U_wp.value, mean)
    assert reference.reset_count == 1


def test_result_name_preserves_reruns_and_config_changes():
    payload = {"label": "duck_high_high_M3_n16_i1", "run_id": "run_a",
               "config": {"seed": 0, "temperature": 1.0}}
    first = result_filename(payload)
    assert first.startswith("duck_high_high_M3_n16_i1_")
    payload["run_id"] = "run_b"
    assert result_filename(payload) != first
    payload["run_id"] = "run_a"
    payload["config"]["temperature"] = 0.5
    assert result_filename(payload) != first


@pytest.mark.parametrize("field,value", [
    ("temperature", 0), ("ref_temperature", 0),
    ("noise_sigma", float("nan")),
    ("step_time", float("inf")), ("max_steps", 0),
    ("eval_substeps", 0), ("settle", -1), ("delta", -1),
    ("n_samples", 0), ("kl_shrinkage", 0),
])
def test_invalid_inputs_fail_before_gpu_work(field, value):
    args = build_parser().parse_args([])
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_kl_args(args)


def test_drake_rejected_in_all_object_kl_workflow():
    args = build_parser().parse_args(["--eval_sim", "drake"])
    with pytest.raises(ValueError, match="hand-only"):
        validate_kl_args(args)


def test_non_grasp_geometry_is_not_rewritten():
    assert resolve_kl_geometry("cart_pole", "accurate") == "accurate"
