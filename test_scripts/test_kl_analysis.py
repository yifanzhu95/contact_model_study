"""CPU-only checks for the KL-divergence analysis helpers."""

from __future__ import annotations

import numpy as np

from analysis.plot_kl_divergence_dir import above_null, kl_value, success_error
from contact_study.evaluation.distributions import (
    gaussian_kl,
    weighted_moments_from_particles,
)


def test_gaussian_kl_identity_and_mean_shift():
    eye = np.eye(2)
    zero = np.zeros(2)

    assert gaussian_kl(zero, eye, zero, eye) == 0.0
    assert np.isclose(gaussian_kl(zero, eye, np.array([1.0, 0.0]), eye), 0.5)


def test_weighted_moments_and_effective_sample_size():
    particles = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    weights = np.array([0.5, 0.25, 0.25])

    mu, cov, ess = weighted_moments_from_particles(
        particles, weights, shrinkage=0.1, sigma=0.5
    )

    assert np.allclose(mu, [0.5, 0.5])
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0.0
    assert np.isclose(ess, 1.0 / np.sum(weights**2))


def test_episode_weighting_uses_episodes_as_independent_units():
    cell = {
        "forward": {"mean": 10.0, "sd": 4.0, "n": 100},
        "episode_series": {
            "forward": {"mean": [1.0, 3.0], "median": [0.8, 2.8]}
        },
    }

    value, error = kl_value(cell, "forward", "mean", "episode")
    assert value == 2.0
    assert np.isclose(error, 1.0)

    legacy_value, legacy_error = kl_value(cell, "forward", "mean", "step")
    assert legacy_value == 10.0
    assert np.isclose(legacy_error, 0.4)


def test_null_floor_uses_selected_weighting():
    real = {
        "forward": {"mean": 5.0, "sd": 1.0, "n": 100},
        "episode_series": {"forward": {"mean": [4.0, 6.0], "median": []}},
    }
    null = {
        "forward": {"mean": 1.0, "sd": 1.0, "n": 100},
        "episode_series": {"forward": {"mean": [0.5, 1.5], "median": []}},
    }
    real["null_cell"] = null

    assert above_null(real, "forward", "mean", "episode") is True
    assert above_null(real, "forward", "mean", "step") is True


def test_wilson_success_interval_does_not_collapse_at_zero_success():
    cell = {"success_rate": 0.0, "success_rate_se": 0.0, "n_episodes": 3}

    lower, upper = success_error(cell, "wilson")
    assert lower == 0.0
    assert upper > 0.5

    assert success_error(cell, "se") == (0.0, 0.0)
