"""MJWarp, ComFree and XPBD simulators: agreement with the old backends, graph capture.

MJWarp is not bit-deterministic run to run (atomics), so each comparison with
the old backend is judged against the old backend's own run-to-run spread.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ROLLOUT_CUBE, matches_within_noise

pytestmark = pytest.mark.gpu

N, T, DT = 16, 25, 0.004


@pytest.fixture(scope="module")
def controls(cube_initial):
    _, _, u0 = cube_initial
    rng = np.random.default_rng(0)
    return (u0 + rng.normal(0, 0.3, (T, N, u0.size))).astype(np.float32)


def _common():
    # The old study wrote Newton / 25 iterations / 1e-6 onto every model.
    return dict(timestep=DT, horizon=2, solver="Newton", iterations=25, tolerance=1e-6,
                nconmax=None, njmax=None)


def _run_new(sim, q0, U):
    sim.SetState(q0)
    for t in range(T):
        sim.SetControl(U[t])
        sim.Step_GPU(1)
    return sim.GetState()[0]


def _run_old(cfg, q0, U):
    import mujoco
    from contact_study.contact_models import api
    mjm = mujoco.MjModel.from_xml_path(ROLLOUT_CUBE)
    mjm.opt.timestep = DT
    m = api.put_model(mjm, cfg)
    d = api.make_data(mjm, m, nworld=N)
    d.qpos.assign(np.tile(q0, (N, 1)).astype(np.float32))
    d.qvel.zero_()
    api.forward(m, d)
    for t in range(T):
        d.ctrl.assign(U[t])
        api.step(m, d)
    return d.qpos.numpy()


def _make(kind, **extra):
    from ContactModelStudy.Simulators.ComFree import ComFree, ComFreeConfig
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco, VectorizedMujocoConfig
    from ContactModelStudy.Simulators.XPBD import XPBD, XPBDConfig
    cls, cfg = {"mujoco": (VectorizedMujoco, VectorizedMujocoConfig), "comfree": (ComFree, ComFreeConfig),
                "xpbd": (XPBD, XPBDConfig)}[kind]
    return cls(ROLLOUT_CUBE, cfg(**_common(), **extra), N=N)


def _old_cfg(kind, **kw):
    from contact_study.contact_models.config import ComfreeParams, ContactModelConfig, XPBDParams
    if kind == "mujoco":
        return ContactModelConfig.M2()
    if kind == "comfree":
        c = ContactModelConfig.M3()
        c.comfree = ComfreeParams(**kw)
        return c
    c = ContactModelConfig.M4()
    c.xpbd = XPBDParams(**kw)
    return c


@pytest.mark.legacy
@pytest.mark.parametrize("kind, new_kw, old_kw", [
    ("mujoco", {}, {}),
    ("comfree", {}, {}),
    ("comfree", dict(stiffness=0.5, damping=0.01), dict(stiffness=0.5, damping=0.01)),
    ("xpbd", {}, {}),
])
def test_matches_old_backend_within_its_own_noise(kind, new_kw, old_kw, cube_initial, controls):
    q0 = cube_initial[0]
    olds = [_run_old(_old_cfg(kind, **old_kw), q0, controls) for _ in range(3)]
    assert matches_within_noise(_run_new(_make(kind, **new_kw), q0, controls), olds)


def test_backends_really_differ(cube_initial, controls):
    q0 = cube_initial[0]
    mj, cf, xp = (_run_new(_make(k), q0, controls) for k in ("mujoco", "comfree", "xpbd"))
    cf2 = _run_new(_make("comfree", stiffness=0.5, damping=0.01), q0, controls)
    assert np.abs(mj - cf).max() > 1e-3
    assert np.abs(cf - xp).max() > 1e-3
    assert np.abs(cf - cf2).max() > 1e-3


def test_comfree_parameters_reach_the_device_model():
    from ContactModelStudy.Simulators.ComFree import ComFree, ComFreeConfig
    s = ComFree(ROLLOUT_CUBE, ComFreeConfig(stiffness=0.37, damping=0.002, solimp_d=0.95, timestep=DT), N=2)
    assert np.isclose(s.m.comfree_stiffness.numpy()[0], 0.37)
    assert np.isclose(s.m.comfree_damping.numpy()[0], 0.002)
    assert np.allclose(s.mjm.geom_solimp[:, :2], 0.95)


def test_types():
    from ContactModelStudy.Simulators.ComFree import ComFree, ComFreeConfig
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujocoConfig
    from ContactModelStudy.Simulators.VectorizedSimulator import VectorizedSimulator
    from ContactModelStudy.Simulators.XPBD import XPBD, XPBDConfig
    for c in (ComFree, XPBD):
        assert issubclass(c, VectorizedSimulator)
    for c in (ComFreeConfig, XPBDConfig):
        assert issubclass(c, VectorizedMujocoConfig)


@pytest.mark.parametrize("module, cls, kwargs", [
    ("ComFree", "ComFreeConfig", dict(stiffness=-1)),
    ("ComFree", "ComFreeConfig", dict(damping=-1)),
    ("ComFree", "ComFreeConfig", dict(cone="elliptic")),
    ("XPBD", "XPBDConfig", dict(xpbd_substeps=0)),
    ("XPBD", "XPBDConfig", dict(xpbd_iterations=0)),
    ("XPBD", "XPBDConfig", dict(relaxation=0)),
    ("XPBD", "XPBDConfig", dict(vmax_depenetration=0)),
])
def test_config_validation(module, cls, kwargs):
    import importlib
    C = getattr(importlib.import_module(f"ContactModelStudy.Simulators.{module}"), cls)
    with pytest.raises(ValueError):
        C(**kwargs)


@pytest.mark.parametrize("kind, extra", [("mujoco", {}), ("comfree", {}), ("xpbd", {})])
def test_captured_graph_reproduces_eager_rollout(kind, extra, cube_initial):
    import warp as wp
    from ContactModelStudy.Simulators.ComFree import ComFree, ComFreeConfig
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco, VectorizedMujocoConfig
    from ContactModelStudy.Simulators.XPBD import XPBD, XPBDConfig
    q0, _, u0 = cube_initial
    H, sub, NW = 8, 4, 64
    cls, cfg = {"mujoco": (VectorizedMujoco, VectorizedMujocoConfig), "comfree": (ComFree, ComFreeConfig),
                "xpbd": (XPBD, XPBDConfig)}[kind]
    sim = cls(ROLLOUT_CUBE, cfg(timestep=DT, horizon=H, substeps=sub, **extra), N=NW)
    rng = np.random.default_rng(1)
    Uwp = wp.array((u0 + rng.normal(0, 0.2, (NW, H, u0.size))).astype(np.float32), device="cuda")
    qd = wp.array(q0.astype(np.float32), device="cuda")
    vd = wp.zeros(sim.nv, dtype=float, device="cuda")

    def body():
        sim.BroadcastState(qd, vd)
        sim.SetControlSequence(Uwp)
        sim.Step_GPU(H * sub)

    eager = []
    for _ in range(3):
        body()
        wp.synchronize()
        eager.append(sim.GetState()[0])
    with wp.ScopedCapture() as cap:
        body()
    wp.capture_launch(cap.graph)
    wp.synchronize()
    assert matches_within_noise(sim.GetState()[0], eager)


def test_xpbd_substeps_stay_stable_holding_the_grasp(cube_initial):
    from ContactModelStudy.Simulators.XPBD import XPBD, XPBDConfig
    q0, _, u0 = cube_initial
    for ns in (1, 2, 3, 4):
        s = XPBD(ROLLOUT_CUBE, XPBDConfig(timestep=0.002, xpbd_substeps=ns), N=4)
        s.SetState(q0)
        s.SetControl(u0)
        s.Step_GPU(250)
        q = s.GetState()[0]
        assert np.isfinite(q).all() and q[0, 18] > 0.05, f"xpbd_substeps={ns}"
