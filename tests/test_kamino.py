"""Simulators/Kamino.py (M5): config, MuJoCo layout, conventions, planning, eval.

Needs Newton 1.6 (the ``contact_kamino`` env); everything but the config test
skips without it, so that one runs in ``contact_modeling`` too.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from conftest import EVAL_CUBE, ROLLOUT_CUBE
from ContactModelStudy.Simulators.Kamino import KaminoConfig


def test_config_validation():
    KaminoConfig()
    for bad in (dict(dynamics_storage="banded"), dict(sparse_linear_solver="LU"), dict(padmm_tolerance=0.0),
                dict(padmm_max_iterations=0), dict(padmm_rho0=-1.0), dict(padmm_penalty_update="x"),
                dict(contact_gap=-0.1)):
        with pytest.raises(ValueError):
            KaminoConfig(**bad)


# Everything below needs Newton and a GPU.
needs_newton = pytest.mark.skipif(importlib.util.find_spec("newton") is None,
                                  reason="needs Newton (contact_kamino env)")


def _sim(N=1, **kw):
    from ContactModelStudy.Simulators.Kamino import Kamino
    return Kamino(ROLLOUT_CUBE, KaminoConfig(**{"timestep": 0.004, "dynamics_storage": "dense", **kw}), N=N)


@pytest.fixture(scope="module")
def one():
    return _sim()


@pytest.mark.gpu
@needs_newton
def test_state_roundtrip_and_sites_match_mujoco(one, cube_initial):
    import mujoco
    q0, _, _ = cube_initial
    q = q0.copy()
    c, s = np.cos(0.4), np.sin(0.4)
    q[19:23] = [c, 0.0, s, 0.0]                          # a rotated object
    v = np.zeros(one.nv)
    v[16:22] = [0.1, -0.05, 0.02, 0.5, -0.3, 0.2]       # MuJoCo: angular velocity in the body frame
    one.SetState(q, v)
    qq, vv = one.GetState()
    assert qq.shape == (1, one.nq) and vv.shape == (1, one.nv)
    assert np.abs(qq[0] - q).max() < 1e-5 and np.abs(vv[0] - v).max() < 1e-5
    d = mujoco.MjData(one.mjm)
    d.qpos[:] = q
    mujoco.mj_forward(one.mjm, d)
    assert np.abs(one.DeviceState().site_xpos.numpy()[0] - d.site_xpos).max() < 1e-5


@pytest.mark.gpu
@needs_newton
def test_free_joint_velocity_convention_matches_mujoco(cube_initial):
    """A spinning, falling cube with contacts off: Kamino and MuJoCo agree (body-frame angular velocity)."""
    import mujoco
    sim = _sim(collision_detection=False, timestep=0.002)
    mjm = mujoco.MjModel.from_xml_path(ROLLOUT_CUBE)
    mjm.opt.timestep = 0.002
    mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(mjm)
    q = mjm.qpos0.copy()
    q[16:19] = [0.1, 0.0, 0.5]
    q[19:23] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    v = np.zeros(mjm.nv)
    v[16:22] = [0.3, -0.2, 0.1, 2.0, 0.0, 0.0]
    d.qpos[:], d.qvel[:] = q, v
    for _ in range(20):
        mujoco.mj_step(mjm, d)
    sim.SetState(q, v)
    sim.SetControl(q[:16])
    sim.Step(20)
    qq, vv = sim.GetState()
    assert np.linalg.norm(qq[0, 16:19] - d.qpos[16:19]) < 1e-4
    assert abs(np.dot(qq[0, 19:23], d.qpos[19:23])) > 1 - 1e-5
    assert np.allclose(vv[0, 19:22], d.qvel[19:22], atol=1e-3)


@pytest.mark.gpu
@needs_newton
def test_holds_the_grasp_and_tasks_read_it(one, cube_tasks, cube_initial):
    ro, _ = cube_tasks
    q0, v0, u0 = cube_initial
    one.SetState(q0, v0)
    one.SetControl(u0)
    one.Step(250)                                       # 1 s
    q, v = one.GetState()
    assert np.isfinite(q).all() and np.isfinite(v).all()
    assert not ro.isFailure(one).numpy()[0]
    assert q[0, 18] > 0.07                              # the cube is still in the palm
    assert np.isfinite(ro.calcCosts(one).numpy()).all()
    assert one.Diagnostics()["converged"].all()


@pytest.mark.gpu
@needs_newton
def test_mppi_graph_matches_eager(cube_tasks, cube_initial):
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config
    ro, _ = cube_tasks
    q0, v0, u0 = cube_initial
    out = {}
    for graph in (True, False):
        sim = _sim(N=4, dynamics_storage="sparse", horizon=2, substeps=2)
        p = MPPI(sim, ro, MPPI_Config(seed=0, noise_sigma=0.1, temperature=10.0, use_graph=graph))
        a = p.Plan(q0, v0, u=u0)
        assert np.isfinite(a).all() and not p._graph_failed
        out[graph] = p.costs_wp.numpy() + p.terminal_costs_wp.numpy()
    assert np.allclose(out[True], out[False], rtol=1e-4)


@pytest.mark.gpu
@needs_newton
def test_preset_and_eval_sim(cube_initial):
    from ContactModelStudy.Simulators.SingleWorld import SingleWorld
    from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
    from ContactModelStudy.Utils.EvalSimulators import evalSimConfig, isGpuEvalSim, makeEvalSim
    assert type(GetContactModelSim("kamino", ROLLOUT_CUBE, N=1, timestep=0.004)).__name__ == "Kamino"
    sim = makeEvalSim("M5", ROLLOUT_CUBE, 0.004)
    assert isinstance(sim, SingleWorld) and sim.class_name == "Kamino" and isGpuEvalSim("M5")
    assert evalSimConfig("M5", 0.004) == ("Kamino", sim.config)
    q0, v0, u0 = cube_initial
    sim.SetState(q0, v0)
    sim.SetControl(u0)
    t0 = sim.time
    sim.Step(5)
    assert sim.GetState()[0].shape == (sim.nq,) and sim.time == pytest.approx(t0 + 5 * 0.004)


@pytest.mark.slow
@pytest.mark.gpu
@needs_newton
def test_eval_scene_falls_back_to_the_unfused_solver_when_it_must():
    """The eval scene's fused CR solve wants more shared memory than an Ada GPU has."""
    from ContactModelStudy.Simulators.Kamino import Kamino
    sim = Kamino(EVAL_CUBE, KaminoConfig(timestep=0.0005), N=1)
    assert sim.linear_solver in ("CRF", "CR")          # CR on GPUs with ~100 KB of shared memory per block
    sim.Step(2)
    assert np.isfinite(sim.GetState()[0]).all()


@pytest.mark.slow
@pytest.mark.gpu
@needs_newton
def test_driver_plans_with_m5(tmp_path):
    from ContactModelStudy.Drivers import run_episodes as drv
    from ContactModelStudy.Utils.EpisodeRecorder import EpisodeReplayer
    out = tmp_path / "m5.json"
    assert drv.main(["--timestep", "0.004", "--eval-steps-per-rollout-step", "1", "--substeps", "2",
                     "--horizon", "2", "--n-samples", "4", "--settle", "0", "--rollout-model", "M5",
                     "--eval-sim", "M2", "--steps", "2", "--no-video", "--results", str(out)]) == 0
    rp = EpisodeReplayer(out)
    assert rp.configs["rollout_simulator"]["class"] == "Kamino" and len(rp[0]) == 2
    assert json.loads(out.read_text())["configs"]["rollout_simulator"]["config"]["padmm_tolerance"] == 5e-4
