"""MPPI: control-mode kernels against the old planner, graph capture, action uncertainty."""

from __future__ import annotations

import importlib
import time

import numpy as np
import pytest

from conftest import ROLLOUT_CUBE

pytestmark = pytest.mark.gpu

SB = importlib.import_module("ContactModelStudy.SamplingBasedPlanners.SamplingBasedPlannerBase")
SIGMA = 0.1


@pytest.fixture(scope="module")
def sim():
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco, VectorizedMujocoConfig
    return VectorizedMujoco(ROLLOUT_CUBE, VectorizedMujocoConfig(horizon=8, substeps=4), N=256)


def _mppi(sim, task, **kw):
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config
    return MPPI(sim, task, MPPI_Config(**{"noise_sigma": SIGMA, "seed": 0, **kw}))


# -- control modes against the old kernels -----------------------------------
@pytest.mark.legacy
def test_pos_relative_matches_old_kernel(cube_initial):
    import warp as wp
    import contact_study.planners.base as oldpb
    q0 = cube_initial[0]
    rng = np.random.default_rng(0)
    NN, H, NU = 8, 4, 16
    V = wp.array(rng.normal(0, 0.05, (NN, H, NU)).astype(np.float32), dtype=wp.float32, device="cuda")
    q = wp.array((np.tile(q0, (NN, 1)) + rng.normal(0, 0.1, (NN, 23))).astype(np.float32),
                 dtype=wp.float32, device="cuda")
    for t in range(H):
        a = wp.zeros((NN, NU), dtype=wp.float32, device="cuda")
        b = wp.zeros((NN, NU), dtype=wp.float32, device="cuda")
        wp.launch(oldpb._assign_ctrl_relative_kernel, dim=(NN, NU), inputs=[V, t, q, 0, a])
        wp.launch(SB._pos_relative_kernel, dim=(NN, NU), inputs=[V, t, q, 0], outputs=[b])
        assert np.array_equal(a.numpy(), b.numpy())


@pytest.mark.legacy
def test_ctrl_relative_matches_old_kernel(cube_initial):
    import warp as wp
    import contact_study.planners.base as oldpb
    u0 = cube_initial[2]
    rng = np.random.default_rng(0)
    NN, H, NU = 8, 4, 16
    V = wp.array(rng.normal(0, 0.05, (NN, H, NU)).astype(np.float32), dtype=wp.float32, device="cuda")
    ctrl = wp.array(np.tile(u0, (NN, 1)).astype(np.float32), dtype=wp.float32, device="cuda")
    A = wp.zeros((NN, H, NU), dtype=wp.float32, device="cuda")
    wp.launch(SB._cumsum_commands_kernel, dim=(NN, NU),
              inputs=[V, wp.array(u0.astype(np.float32), dtype=wp.float32, device="cuda"), H], outputs=[A])
    for t in range(H):
        wp.launch(oldpb._assign_ctrl_kernel, dim=(NN, NU), inputs=[V, t, ctrl])
        assert np.array_equal(ctrl.numpy(), A.numpy()[:, t])


# -- graph capture -----------------------------------------------------------
def test_graph_capture_is_much_faster(cube_tasks, cube_initial):
    import warp as wp
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco, VectorizedMujocoConfig
    ro, _ = cube_tasks
    q0, v0, _ = cube_initial
    times = {}
    for use_graph in (False, True):
        s = VectorizedMujoco(ro.getModelPath(), VectorizedMujocoConfig(substeps=4, horizon=8), N=256)
        m = _mppi(s, ro, noise_sigma=0.05, use_graph=use_graph)
        m.Plan(q0, v0)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(8):
            m.Plan(q0, v0)
        wp.synchronize()
        times[use_graph] = (time.perf_counter() - t0) / 8
    assert times[False] / times[True] > 10


# -- action uncertainty --------------------------------------------------------
def test_default_returns_a_bare_action(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], temperature=10.0)
    a = p.Plan(q0, v0, u=u0)
    assert isinstance(a, np.ndarray) and a.shape == (16,) and p.last_action_uncertainty is None


def test_uncertainty_is_the_weighted_spread_of_first_actions(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], temperature=10.0, return_uncertainty=True)
    act, unc = p.Plan(q0, v0, u=u0)
    assert act.shape == unc.shape == (16,)
    w = p.w_wp.numpy().astype(float)
    V0 = p.V_wp.numpy()[:, 0, :].astype(float)
    U0 = p.U_wp.numpy()[0].astype(float)
    assert np.allclose(unc, np.sqrt((w[:, None] * (V0 - U0) ** 2).sum(0)), rtol=1e-4, atol=1e-6)
    assert np.array_equal(p.last_action_uncertainty, unc)
    assert np.allclose(act, q0[:16] + U0, atol=1e-5)          # pos_relative: q_robot + U0


def test_uncertainty_limits(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    hi = _mppi(sim, cube_tasks[0], temperature=1e9, return_uncertainty=True).Plan(q0, v0, u=u0)[1]
    lo = _mppi(sim, cube_tasks[0], temperature=1e-6, return_uncertainty=True).Plan(q0, v0, u=u0)[1]
    assert abs(hi.mean() - SIGMA) < 0.02          # uniform weights -> noise_sigma
    assert lo.max() < 1e-3                        # one dominant sample -> ~0


@pytest.mark.parametrize("kw", [dict(warm_start="shift"), dict(warm_start="hold"),
                                dict(control_mode="absolute"),
                                dict(control_mode="ctrl_relative")])
def test_uncertainty_in_every_mode(sim, cube_tasks, cube_initial, kw):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], return_uncertainty=True, **kw)
    if kw.get("control_mode") == "absolute":
        p.Reset(mean=u0)
    for _ in range(3):
        _, unc = p.Plan(q0, v0, u=u0)
    assert np.isfinite(unc).all() and unc.shape == (16,)


# -- warm start ------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["pos_relative", "absolute"])
def test_warm_start_none_starts_every_plan_from_the_reset_seed(sim, cube_tasks, cube_initial, mode):
    q0, v0, u0 = cube_initial
    seed = u0 if mode == "absolute" else None
    p = _mppi(sim, cube_tasks[0], temperature=10.0, control_mode=mode)
    p.Reset(seed)
    p.Plan(q0, v0, u=u0)
    second = p.Plan(q0, v0, u=u0)
    # A fresh planner on the same noise draw: the first plan left nothing behind.
    # (Repeat plans differ by ~2e-5 on the GPU; a carried-over mean, by ~1e-2.)
    fresh = _mppi(sim, cube_tasks[0], temperature=10.0, control_mode=mode)
    fresh.Reset(seed)
    fresh._resample_count = 1
    assert np.allclose(second, fresh.Plan(q0, v0, u=u0), atol=1e-4)


@pytest.mark.parametrize("warm_start, rows", [("hold", 0), ("shift", 1)])
def test_warm_start_carries_the_last_solution(sim, cube_tasks, cube_initial, warm_start, rows):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], temperature=10.0, warm_start=warm_start)
    p.Reset()
    p.Plan(q0, v0, u=u0)
    solved = p.last_action_seq - q0[:16]          # pos_relative: the mean, before any shift
    expect = np.zeros_like(solved)
    expect[:len(solved) - rows] = solved[rows:]
    assert np.allclose(p.U_wp.numpy(), expect, atol=1e-5)


@pytest.mark.parametrize("warm_start", ["hold", "shift"])
def test_a_plan_replays_from_its_recorded_start(sim, cube_tasks, cube_initial, warm_start):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], temperature=10.0, warm_start=warm_start)
    p.Reset()
    p.Plan(q0, v0, u=u0)
    second = p.Plan(q0, v0, u=u0)
    start = p.last_plan_start
    assert start["resample_count"] == 1 and np.any(start["U"] != 0)     # warm: the first plan's mean
    fresh = _mppi(sim, cube_tasks[0], temperature=10.0, warm_start=warm_start)
    fresh.LoadState(start)
    assert np.allclose(fresh.Plan(q0, v0, u=start["u_prev"]), second, atol=1e-4)


def test_warm_start_must_be_a_known_choice():
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI_Config
    assert MPPI_Config().warm_start == "none"
    with pytest.raises(ValueError, match="warm_start"):
        MPPI_Config(warm_start=True)


def test_reset_seed_travels_with_the_state(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], control_mode="absolute")
    p.Reset(u0)
    state = p.SaveState()
    p.Reset(None)
    p.LoadState(state)
    p.Plan(q0, v0, u=u0)
    assert np.allclose(p._reset_mean, u0)


def test_failed_plan_gives_zero_action_and_nan_uncertainty(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], return_uncertainty=True)
    p._updateParams = lambda: False
    a, unc = p.Plan(q0, v0, u=u0)
    assert np.all(a == 0) and np.isnan(unc).all()


def test_planner_without_the_hook_refuses_the_flag(sim, cube_tasks):
    class NoUncertainty(SB.SamplingBasedPlannerBase):
        def _buildSamples(self):
            pass

        def _updateParams(self):
            return True

    with pytest.raises(NotImplementedError):
        NoUncertainty(sim, cube_tasks[0], SB.SamplingBasedPlannerConfig(return_uncertainty=True))
    NoUncertainty(sim, cube_tasks[0], SB.SamplingBasedPlannerConfig())


# -- planning on the eval scene ------------------------------------------------
def test_plan_on_eval_scene(cube_tasks, cube_initial):
    from conftest import EVAL_CUBE
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco, VectorizedMujocoConfig
    ro, ev = cube_tasks
    q0, v0, u0 = cube_initial
    eval_sim = VectorizedMujoco(EVAL_CUBE, VectorizedMujocoConfig(timestep=ev.timestep, horizon=4, substeps=4), N=32)
    p = _mppi(eval_sim, ev, plan_on_eval_scene=True)
    assert np.isfinite(p.Plan(q0, v0, u=u0)).all()


def test_plan_on_eval_scene_must_match_the_task_role(sim, cube_tasks):
    ro, ev = cube_tasks
    with pytest.raises(ValueError, match="role is 'rollout'"):
        _mppi(sim, ro, plan_on_eval_scene=True)
    with pytest.raises(ValueError, match="role is 'eval'"):
        _mppi(sim, ev)


# -- free-energy convergence stop ---------------------------------------------
def test_convergence_config_is_validated():
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI_Config
    with pytest.raises(ValueError, match="convergence_ratio"):
        MPPI_Config(convergence_ratio=0.0)
    with pytest.raises(ValueError, match="max_iterations"):
        MPPI_Config(convergence_ratio=2.0, max_iterations=1)
    with pytest.raises(ValueError, match="adaptive_temp"):
        MPPI_Config(convergence_ratio=2.0, adaptive_temp=True)


def test_without_a_ratio_the_plan_runs_n_iterations(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], temperature=10.0, n_iterations=3)
    p.Plan(q0, v0, u=u0)
    assert p.last_n_iterations == 3 and p.last_converged is None
    assert p.last_free_energy is None and p.last_free_energy_drop is None


def _scripted_free_energy(p, values):
    """Make ``p`` report the free energies ``values`` in turn, each with se 1."""
    it = iter(values)
    p._freeEnergy = lambda: (next(it), 1.0)


def test_a_drop_within_the_noise_stops_the_plan(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], temperature=10.0, convergence_ratio=2.0, max_iterations=10)
    _scripted_free_energy(p, [100.0, 90.0, 80.0, 79.0, 78.0])   # se of a drop: sqrt(2)
    p.Plan(q0, v0, u=u0)
    assert p.last_n_iterations == 4 and p.last_converged is True and p.last_plan_ok
    assert p.last_free_energy == 79.0 and p.last_free_energy_drop == 1.0
    assert p.last_free_energy_drop_se == pytest.approx(np.sqrt(2))


def test_a_rise_stops_the_plan(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], temperature=10.0, convergence_ratio=1e-30, max_iterations=10)
    _scripted_free_energy(p, [100.0, 90.0, 95.0, 0.0])
    p.Plan(q0, v0, u=u0)
    assert p.last_n_iterations == 3 and p.last_converged is True


def test_a_steady_descent_runs_to_the_cap(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], temperature=10.0, convergence_ratio=2.0, max_iterations=4)
    _scripted_free_energy(p, [100.0, 90.0, 80.0, 70.0])
    p.Plan(q0, v0, u=u0)
    assert p.last_n_iterations == 4 and p.last_converged is False and p.last_plan_ok


def test_the_free_energy_is_the_soft_minimum_of_the_sampled_costs(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    lam = 0.1
    p = _mppi(sim, cube_tasks[0], temperature=lam, convergence_ratio=1e30, max_iterations=2)
    p.Plan(q0, v0, u=u0)
    J = p.costs_wp.numpy().astype(float)              # the last update's folded costs
    expect = J.min() - lam * np.log(np.mean(np.exp(-(J - J.min()) / lam)))
    assert p.last_free_energy == pytest.approx(expect, rel=1e-6)
    assert J.min() <= p.last_free_energy <= J.mean()
    # The delta-method standard error agrees with a bootstrap over the same costs.
    f, se = p._freeEnergy()
    rng = np.random.default_rng(0)
    boot = [(lambda b: b.min() - lam * np.log(np.mean(np.exp(-(b - b.min()) / lam))))(rng.choice(J, J.size))
            for _ in range(2000)]
    assert se == pytest.approx(np.std(boot), rel=0.2)


# -- noise resampling between iterations --------------------------------------
@pytest.mark.parametrize("resample", [True, False])
def test_the_first_draw_and_noise_index_do_not_depend_on_resampling(sim, cube_tasks, cube_initial, resample):
    q0, v0, u0 = cube_initial
    one = _mppi(sim, cube_tasks[0], temperature=10.0)
    one.Plan(q0, v0, u=u0)
    p = _mppi(sim, cube_tasks[0], temperature=10.0, n_iterations=3, resample_noise=resample)
    p.Plan(q0, v0, u=u0)
    assert p._resample_count == one._resample_count == 1
    assert np.array_equal(p.eps_wp.numpy(), one.eps_wp.numpy()) != resample


def test_a_resampled_plan_replays_from_its_recorded_start(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], temperature=10.0, n_iterations=3)
    p.Plan(q0, v0, u=u0)
    second = p.Plan(q0, v0, u=u0)
    fresh = _mppi(sim, cube_tasks[0], temperature=10.0, n_iterations=3)
    fresh.LoadState(p.last_plan_start)
    assert np.allclose(fresh.Plan(q0, v0, u=u0), second, atol=1e-4)


# -- noise centering ----------------------------------------------------------
@pytest.mark.parametrize("center", [True, False])
def test_centered_noise_sums_to_zero_over_samples(sim, cube_tasks, cube_initial, center):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], temperature=10.0, center_noise=center)
    p.Plan(q0, v0, u=u0)
    col_mean = np.abs(p.eps_wp.numpy().astype(float).mean(axis=0)).max()
    assert (col_mean < 1e-6) == center


@pytest.mark.parametrize("center", [True, False])
def test_uniform_weights_leave_only_an_uncentered_mean_drifting(sim, cube_tasks, cube_initial, center):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], temperature=1e9, center_noise=center)
    p.Plan(q0, v0, u=u0)
    # The mean starts at zero. Uncentered, it moves by the noise's sample mean,
    # about SIGMA / sqrt(N) per entry; centered, it stays put.
    assert (np.abs(p.U_wp.numpy()).max() < 1e-6) == center


# -- scoring a given mean ------------------------------------------------------
@pytest.mark.parametrize("warm_start", ["none", "shift"])
def test_last_mean_is_the_plan_before_the_warm_start_shift(sim, cube_tasks, cube_initial, warm_start):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], temperature=10.0, warm_start=warm_start)
    p.Plan(q0, v0, u=u0)
    assert np.allclose(p.last_action_seq, q0[:16] + p.last_mean, atol=1e-5)    # pos_relative tape
    assert np.array_equal(p.last_mean, p.U_wp.numpy()) == (warm_start == "none")
    p._updateParams = lambda: False
    p.Plan(q0, v0, u=u0)
    assert p.last_mean is None                                                  # a failed plan


def test_free_energy_of_a_mean_is_the_soft_minimum_of_its_samples(sim, cube_tasks, cube_initial):
    q0, v0, u0 = cube_initial
    lam = 0.1
    p = _mppi(sim, cube_tasks[0], temperature=lam)
    p.Plan(q0, v0, u=u0)
    U, state = p.last_mean, p.SaveState()
    f, se = p.FreeEnergyOf(q0, v0, U, u=u0)
    J = p.costs_wp.numpy().astype(float)
    assert f == pytest.approx(J.min() - lam * np.log(np.mean(np.exp(-(J - J.min()) / lam))), rel=1e-6)
    assert se > 0
    # LoadState chooses the noise, so the same mean scores the same, up to MJWarp's run-to-run
    # nondeterminism in contact-rich rollouts (about one standard error here).
    p.LoadState(state)
    assert abs(p.FreeEnergyOf(q0, v0, U, u=u0)[0] - f) < 3 * se
    with pytest.raises(ValueError, match="shape"):
        p.FreeEnergyOf(q0, v0, U[:-1], u=u0)
