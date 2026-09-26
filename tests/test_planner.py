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


@pytest.mark.parametrize("kw", [dict(warm_start=True), dict(control_mode="absolute"),
                                dict(control_mode="ctrl_relative")])
def test_uncertainty_in_every_mode(sim, cube_tasks, cube_initial, kw):
    q0, v0, u0 = cube_initial
    p = _mppi(sim, cube_tasks[0], return_uncertainty=True, **kw)
    if kw.get("control_mode") == "absolute":
        p.Reset(mean=u0)
    for _ in range(3):
        _, unc = p.Plan(q0, v0, u=u0)
    assert np.isfinite(unc).all() and unc.shape == (16,)


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
