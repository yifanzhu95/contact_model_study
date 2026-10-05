"""Utils/PlannerKLDiv.py: Gaussian KL between planners and/or Gaussian actions."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ROLLOUT_CUBE
from ContactModelStudy.Utils.PlannerKLDiv import CalcPlannerKLDiv, GaussianKL, PlannerGaussian


def _spd(rng, d):
    A = rng.normal(size=(d, d))
    return A @ A.T + d * np.eye(d)


# -- the formula -------------------------------------------------------------------
@pytest.mark.legacy
def test_gaussian_kl_matches_the_old_implementation():
    from contact_study.evaluation.distributions import gaussian_kl
    rng = np.random.default_rng(0)
    for d in (1, 3, 16):
        for _ in range(5):
            mu0, mu1 = rng.normal(size=d), rng.normal(size=d)
            S0, S1 = _spd(rng, d), _spd(rng, d)
            assert np.isclose(GaussianKL(mu0, S0, mu1, S1), gaussian_kl(mu0, S0, mu1, S1), rtol=1e-10)


def test_closed_form_cases():
    I = np.eye(3)
    assert GaussianKL(np.zeros(3), I, np.zeros(3), I) == pytest.approx(0.0, abs=1e-12)
    # Mean shift only: ||dmu||^2 / (2 sigma^2).
    assert GaussianKL(np.zeros(3), 4 * I, np.array([2.0, 0, 0]), 4 * I) == pytest.approx(0.5)
    # 1-D variance ratio: 0.5 (r - 1 - ln r), r = s_p^2 / s_q^2.
    r = 4.0
    assert GaussianKL([0.0], [[4.0]], [0.0], [[1.0]]) == pytest.approx(0.5 * (r - 1 - np.log(r)))
    # Not symmetric.
    assert GaussianKL([0.0], [[4.0]], [0.0], [[1.0]]) != pytest.approx(GaussianKL([0.0], [[1.0]], [0.0], [[4.0]]))
    assert GaussianKL([0.0], [[1.0]], [0.0], [[-1.0]]) == float("inf")


def test_actions_scalar_vector_and_matrix_sigma():
    u = np.array([0.1, -0.2, 0.3])
    a = CalcPlannerKLDiv(None, None, (u, 0.1), (u, np.full(3, 0.1)))
    b = CalcPlannerKLDiv(None, None, (u, np.full(3, 0.1)), (u, np.eye(3) * 0.01))
    assert a == pytest.approx(0.0, abs=1e-9) and b == pytest.approx(0.0, abs=1e-9)
    shifted = CalcPlannerKLDiv(None, None, (u, 0.1), (u + [0.1, 0, 0], 0.1))
    assert shifted == pytest.approx(0.5, rel=1e-6)           # (0.1 / 0.1)^2 / 2
    with pytest.raises(ValueError):
        CalcPlannerKLDiv(None, None, (u, 0.1), (np.zeros(4), 0.1))
    with pytest.raises(ValueError):
        CalcPlannerKLDiv(None, None, (u, np.ones(2)), (u, 0.1))
    with pytest.raises(TypeError):
        CalcPlannerKLDiv(None, None, u, (u, 0.1))
    with pytest.raises(ValueError):
        CalcPlannerKLDiv(None, None, (u, 0.1), (u, 0.1), shrinkage=2.0)


def test_zero_sigma_is_floored_not_infinite():
    u = np.zeros(2)
    assert np.isfinite(CalcPlannerKLDiv(None, None, (u, 0.0), (u, 0.1)))


# -- planners ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim():
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco, VectorizedMujocoConfig
    return VectorizedMujoco(ROLLOUT_CUBE, VectorizedMujocoConfig(horizon=6, substeps=4), N=256)


def _mppi(sim, task, **kw):
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config
    return MPPI(sim, task, MPPI_Config(**{"noise_sigma": 0.1, "temperature": 10.0, "seed": 0, **kw}))


@pytest.mark.gpu
@pytest.mark.legacy
def test_first_action_moments_match_the_old_weighted_moments(sim, cube_tasks, cube_initial):
    from contact_study.evaluation.distributions import weighted_moments_from_particles
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0])
    p.Plan(q0, v0, u=u0)
    mu, cov = p.FirstActionDistribution()
    V0 = p.V_wp.numpy()[:, 0, :].astype(np.float64)
    w = p.w_wp.numpy().astype(np.float64)
    mu_old, cov_old, _ = weighted_moments_from_particles(V0, w, shrinkage=0.0, sigma=0.1)
    assert np.allclose(cov, cov_old, atol=1e-12)
    # The mean is in command space: the returned action (pos_relative: q + U0).
    assert np.allclose(mu - q0[:16], mu_old, atol=1e-5)
    assert np.allclose(mu, p.last_action_seq[0], atol=1e-6)


@pytest.mark.gpu
def test_planner_state_is_restored(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], warm_start=True)
    p.Plan(q0, v0, u=u0)
    before = p.SaveState()
    CalcPlannerKLDiv(q0, v0, p, (u0, 0.1), u=u0)
    after = p.SaveState()
    assert np.array_equal(before["U"], after["U"])
    assert before["resample_count"] == after["resample_count"] and before["noise_seed"] == after["noise_seed"]


@pytest.mark.gpu
def test_planner_against_its_own_reported_action(sim, cube_tasks, cube_initial):
    """KL(planner || its own (action, sigma)) is small: same mean, and the diag
    sigma is the planner's own per-actuator spread. Not zero: the planner's
    covariance has off-diagonal terms the diagonal Gaussian lacks."""
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], return_uncertainty=True)
    a, sigma = p.Plan(q0, v0, u=u0)
    mu, cov = PlannerGaussian(p, q0, v0, u=u0, shrinkage=0.0, restore=False)
    # The same plan described two ways agrees on the mean and the per-actuator spread.
    assert np.allclose(np.sqrt(np.diag(cov)), p.last_action_uncertainty, rtol=1e-3, atol=1e-6)
    kl = CalcPlannerKLDiv(q0, v0, (mu, cov), (mu, np.sqrt(np.diag(cov))), u=u0)
    assert 0.0 <= kl < 2.0


@pytest.mark.gpu
def test_identical_planners_are_close_and_temperature_moves_them_apart(sim, cube_tasks, cube_initial):
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco, VectorizedMujocoConfig
    q0, v0, u0 = cube_initial
    task = cube_tasks[0]
    sim2 = VectorizedMujoco(ROLLOUT_CUBE, VectorizedMujocoConfig(horizon=6, substeps=4), N=256)
    a, b = _mppi(sim, task), _mppi(sim2, task)                       # same config and seed
    same = CalcPlannerKLDiv(q0, v0, a, b, u=u0)
    c = _mppi(sim2, task, temperature=0.5)
    different = CalcPlannerKLDiv(q0, v0, a, c, u=u0)
    assert np.isfinite(same) and np.isfinite(different)
    assert 0.0 <= same < different
    # Direction matters.
    assert CalcPlannerKLDiv(q0, v0, c, a, u=u0) != pytest.approx(different, rel=1e-3)


@pytest.mark.gpu
def test_failed_plan_gives_nan(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0])
    p._updateParams = lambda: False
    assert np.isnan(CalcPlannerKLDiv(q0, v0, p, (u0, 0.1), u=u0))
