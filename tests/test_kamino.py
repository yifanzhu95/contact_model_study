"""Simulators/Kamino.py (M5): config, MuJoCo layout, conventions, planning, eval.

Needs Newton 1.6 (the ``contact_kamino`` env); everything but the config test
skips without it, so that one runs in ``contact_modeling`` too.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from conftest import EVAL_CUBE, ROLLOUT_CUBE, SCENES
from ContactModelStudy.Simulators.Kamino import KaminoConfig


def test_config_validation():
    c = KaminoConfig()
    assert c.dynamics_storage == "dense" and c.max_contacts_per_world == 128 and c.convex_meshes
    # The anti-sticking tuning: a grasp converges only with a small penalty and a full warm start.
    assert c.padmm_rho0 == 0.1 and c.padmm_warmstart_scale == 1.0
    assert c.contact_stabilization == 0.05 and c.joint_stabilization == 0.1
    assert c.dynamics_solver == "padmm" and c.contact_gap == 0.0
    KaminoConfig(max_contacts_per_world=None)
    KaminoConfig(dynamics_storage="sparse", padmm_penalty_update="balanced")
    for bad in (dict(dynamics_storage="banded"), dict(sparse_linear_solver="LU"), dict(padmm_tolerance=0.0),
                dict(padmm_max_iterations=0), dict(padmm_rho0=-1.0), dict(padmm_penalty_update="x"),
                dict(padmm_penalty_update="balanced"), dict(padmm_warmstart_scale=1.5),
                dict(dynamics_solver="pgs"), dict(integrator="rk4"), dict(dvi_max_iterations=0),
                dict(dvi_sweeps=1.5), dict(dvi_tolerance=-1.0), dict(contact_stabilization=-0.1),
                dict(joint_stabilization=2.0), dict(limit_stabilization=float("nan")),
                dict(contact_penetration_margin=-1e-6),
                dict(contact_gap=-0.1), dict(max_contacts_per_world=0), dict(max_contacts_per_world=1.5)):
        with pytest.raises(ValueError):
            KaminoConfig(**bad)


# Everything below needs Newton and a GPU.
needs_newton = pytest.mark.skipif(importlib.util.find_spec("newton") is None,
                                  reason="needs Newton (contact_kamino env)")


def _sim(N=1, **kw):
    from ContactModelStudy.Simulators.Kamino import Kamino
    return Kamino(ROLLOUT_CUBE, KaminoConfig(**{"timestep": 0.004, **kw}), N=N)


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
    d = one.Diagnostics()
    assert d["converged"].all() and d["contact_capacity"] == 128
    assert 0 < d["contacts"].max() < 256


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


@pytest.mark.gpu
@needs_newton
@pytest.mark.parametrize("scene", ["env_leap_rollout_duck_low_high.xml", "env_leap_eval_duck.xml"])
def test_duck_scenes_load_and_hold(scene):
    """The duck's meshes resolve through the included meshdir, and the hulls hold the grasp."""
    from ContactModelStudy.Simulators.Kamino import Kamino
    from ContactModelStudy.Tasks.DuckReorient import DuckReorient
    from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig
    from ContactModelStudy.Tasks.TaskBase import TaskRole
    q0, v0, u0 = DuckReorient(LeapReorientConfig(role=TaskRole.ROLLOUT)).getInitialState()
    dt = 0.0005 if "eval" in scene else 0.004
    sim = Kamino(str(SCENES / scene), KaminoConfig(timestep=dt), N=2)
    sim.SetState(q0, v0)
    sim.SetControl(u0)
    sim.Step(int(0.25 / dt))
    q, _ = sim.GetState()
    d = sim.Diagnostics()
    assert np.isfinite(q).all() and (q[:, 18] > 0.07).all()
    assert d["converged"].all()


@pytest.mark.gpu
@needs_newton
def test_meshes_collide_as_convex_hulls_by_default():
    """As in MuJoCo: Newton's triangle-mesh path puts several times more points on the duck's hulls."""
    import newton as nt
    from ContactModelStudy.Simulators.Kamino import Kamino
    from ContactModelStudy.Tasks.DuckReorient import DuckReorient
    from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig
    from ContactModelStudy.Tasks.TaskBase import TaskRole
    q0, v0, u0 = DuckReorient(LeapReorientConfig(role=TaskRole.ROLLOUT)).getInitialState()
    peaks = {}
    for convex in (True, False):
        sim = Kamino(str(SCENES / "env_leap_eval_duck.xml"), KaminoConfig(timestep=0.0005, convex_meshes=convex), N=1)
        types = sim.model.shape_type.numpy()
        colliding = (sim.model.shape_flags.numpy() & int(nt.ShapeFlags.COLLIDE_SHAPES)) != 0
        assert (types[colliding] == int(nt.GeoType.MESH)).any() != convex
        if convex:                                          # hulls, at most the MJCF's maxhullvert=30 vertices
            hulls = [m for m, t in zip(sim.model.shape_source, types) if t == int(nt.GeoType.CONVEX_MESH)]
            assert hulls and max(len(m.vertices) for m in hulls) <= 64
        sim.SetState(q0, v0)
        sim.SetControl(u0)
        sim.Diagnostics()                                   # reset the peak
        sim.Step(500)                                       # 0.25 s: the duck settles into the palm
        peaks[convex] = int(sim.Diagnostics()["contacts"].max())
    assert 0 < peaks[True] < peaks[False]


@pytest.mark.gpu
@needs_newton
def test_a_squeezed_grasp_converges_and_a_sunk_fingertip_comes_out(cube_initial):
    """The anti-sticking tuning, on the eval scene at its 0.5 ms step.

    With ``rho0=1`` and a 0.9 warm start nearly every squeezed-grasp step ran to
    the iteration cap, so the fingers drifted off their joints, and with
    Newton's 0.01 contact stabilization a fingertip 3 mm inside the cube was
    still about 1.5 mm in after 50 ms.
    """
    import mujoco
    from ContactModelStudy.Simulators.Kamino import Kamino
    sim = Kamino(EVAL_CUBE, KaminoConfig(timestep=0.0005), N=1)
    q0, v0, u0 = cube_initial
    sim.SetState(q0, v0)
    sim.SetControl(u0)
    sim.Step(400)                                       # 0.2 s: the cube settles into the palm
    flex = [a for a in range(sim.nu) if sim.mjm.actuator(a).name.split("_")[1] in ("mcp", "pip", "dip", "ipl")]
    q, _ = sim.GetState()
    u = q[0, :16].copy()
    u[flex] += 0.3                                      # squeeze
    sim.SetControl(u)
    converged, iterations = [], []
    for _ in range(20):
        sim.Step(10)
        d = sim.Diagnostics()
        converged.append(bool(d["converged"][0]))
        iterations.append(int(d["iterations"][0]))
    assert np.mean(converged) >= 0.8 and np.mean(iterations) < 60

    # The links stay on their joints: fingertip sites from Kamino's body poses
    # agree with forward kinematics of its qpos (1 mm apart, and growing, with
    # the old settings).
    mjm, md = sim.mjm, mujoco.MjData(sim.mjm)
    md.qpos[:] = sim.GetState()[0][0]
    mujoco.mj_kinematics(mjm, md)
    tips = [mjm.site(n).id for n in ("if_tip", "mf_tip", "rf_tip", "th_tip")]
    assert np.abs(sim.DeviceState().site_xpos.numpy()[0][tips] - md.site_xpos[tips]).max() < 3e-4

    # Curl the index finger 3 mm into the cube and keep commanding that pose.
    index = [mjm.joint(n).qposadr[0] for n in ("if_mcp", "if_pip", "if_dip")]

    def penetration(qpos):
        md.qpos[:] = qpos
        mujoco.mj_forward(mjm, md)
        return -min([0.0] + [c.dist for c in md.contact[:md.ncon]])

    q, _ = sim.GetState()
    q = q[0].astype(np.float64)
    while penetration(q) < 3e-3:
        q[index] += 0.005
    sim.SetState(q)
    u = sim.GetControl()[0]
    u[index] = q[index]
    sim.SetControl(u)
    sim.Step(100)                                       # 50 ms
    assert penetration(sim.GetState()[0][0].astype(np.float64)) < 1e-3


@pytest.mark.gpu
@needs_newton
def test_a_full_contact_buffer_warns(cube_initial):
    q0, v0, u0 = cube_initial
    sim = _sim(max_contacts_per_world=2)
    sim.SetState(q0, v0)
    sim.SetControl(u0)
    sim.Step(2)
    with pytest.warns(RuntimeWarning, match="contact capacity of 2"):
        sim.GetState()


@pytest.mark.slow
@pytest.mark.gpu
@needs_newton
def test_eval_scene_falls_back_to_the_unfused_solver_when_it_must():
    """Uncapped and sparse, the eval scene's fused CR solve wants more shared memory than an Ada GPU has."""
    from ContactModelStudy.Simulators.Kamino import Kamino
    sim = Kamino(EVAL_CUBE, KaminoConfig(timestep=0.0005, dynamics_storage="sparse", max_contacts_per_world=None),
                 N=1)
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
