"""Pinocchio and Drake eval simulators: MJCF inheritance, regression, cross-simulator agreement."""

from __future__ import annotations

import types

import mujoco
import numpy as np
import pytest

from conftest import EVAL_CUBE, curl_controls
from ContactModelStudy.Simulators.Simulator import Simulator
from ContactModelStudy.Utils.MjcfModelInfo import MjcfModelInfo

pin = pytest.importorskip("pinocchio")
DT = 0.002

# The old PinocchioSimulator's solver settings, for like-for-like regression.
OLD_PIN_SETTINGS = dict(anderson_capacity=20, baumgarte_kp=10.0, baumgarte_kd=0.5)


@pytest.fixture(scope="module")
def info():
    return MjcfModelInfo.fromXml(EVAL_CUBE)


@pytest.fixture(scope="module")
def pinocchio_sim():
    from ContactModelStudy.Simulators.Pinocchio import Pinocchio, PinocchioConfig
    return Pinocchio(EVAL_CUBE, PinocchioConfig(timestep=DT))


@pytest.fixture(scope="module")
def drake_sim():
    pytest.importorskip("pydrake")
    from ContactModelStudy.Simulators.Drake import Drake, DrakeConfig
    return Drake(EVAL_CUBE, DrakeConfig(timestep=DT))


# -- MjcfModelInfo -----------------------------------------------------------
def test_mjcf_model_info(info):
    assert (info.nq, info.nv, info.nu) == (23, 22, 16)
    assert len(info.joints) == 17 and info.joints[-1].type == "free"
    a = info.actuators[0]
    assert (a.kp, a.kv) == (3.0, 0.01) and a.force_range == (-np.inf, np.inf)
    assert len(info.allowed_pairs) == 1754


def test_allowed_pairs_contain_every_pair_mujoco_reports(info):
    q0 = mujoco.MjModel.from_xml_path(EVAL_CUBE)
    q0.geom_margin[:] = 10.0                # provoke contacts on as many pairs as possible
    d = mujoco.MjData(q0)
    mujoco.mj_forward(q0, d)
    seen = {tuple(sorted((int(c.geom1), int(c.geom2)))) for c in d.contact[:d.ncon]}
    assert seen and seen <= info.allowed_pairs


# -- Pinocchio -----------------------------------------------------------------
def test_pinocchio_inherits_from_the_mjcf(pinocchio_sim, info):
    s, m = pinocchio_sim, pinocchio_sim.model
    for j, jid in s._joints:
        iv, iq = m.joints[jid].idx_v, m.joints[jid].idx_q
        assert np.allclose(m.armature[iv:iv + j.nv], j.armature), j.name
        assert np.allclose(s._damping[iv:iv + j.nv], j.damping), j.name
        if j.limited and j.type != "free":
            assert np.allclose([m.lowerPositionLimit[iq], m.upperPositionLimit[iq]], j.range), j.name
    assert np.allclose(s._kp, info.mjm.actuator_gainprm[:, 0])
    assert np.allclose(s._kv, -info.mjm.actuator_biasprm[:, 2])
    assert np.allclose(s._ctrl_lo, info.mjm.actuator_ctrlrange[:, 0])


def test_pinocchio_collision_pairs_and_friction(pinocchio_sim, info):
    s = pinocchio_sim
    pairs = {tuple(sorted((s._pin_to_mj_geom[p.first], s._pin_to_mj_geom[p.second])))
             for p in s.collision_model.collisionPairs}
    assert pairs == info.allowed_pairs
    for k, p in enumerate(s.collision_model.collisionPairs):
        a, b = s._pin_to_mj_geom[p.first], s._pin_to_mj_geom[p.second]
        assert np.isclose(s._pair_friction[k], max(info.geoms[a].friction, info.geoms[b].friction))
    for k, g in enumerate(s._pin_to_mj_geom):
        if g is not None:
            assert s.collision_model.geometryObjects[k].name == info.geoms[g].name


def test_pinocchio_state_roundtrip(pinocchio_sim, cube_initial):
    q0 = cube_initial[0]
    vt = np.linspace(-1, 1, 22) * 0.1
    pinocchio_sim.SetState(q0, vt)
    q, v = pinocchio_sim.GetState()
    assert np.allclose(q, q0, atol=1e-12) and np.allclose(v, vt, atol=1e-12)


@pytest.mark.parametrize("kwargs", [dict(admm_max_iterations=0), dict(absolute_tolerance=0),
                                    dict(admm_update_rule="x"), dict(baumgarte_kp=-1),
                                    dict(max_contacts_per_pair=0)])
def test_pinocchio_config_validation(kwargs):
    from ContactModelStudy.Simulators.Pinocchio import PinocchioConfig
    with pytest.raises(ValueError):
        PinocchioConfig(**kwargs)


def _old_pinocchio():
    import contact_study.contact_models.pinocchio_sim as old
    mjm = mujoco.MjModel.from_xml_path(EVAL_CUBE)
    jc = [old.PinocchioJointChannel(mjm.joint(j).name, int(mjm.jnt_qposadr[j]), int(mjm.jnt_dofadr[j]))
          for j in range(16)]
    names = [mjm.joint(int(mjm.actuator(a).trnid[0])).name for a in range(16)]
    pid = old.PinocchioPdActuation(ctrl_joint_names=names, kp=3.0, armature=0.001, kd=0.31)
    return old.PinocchioSimulator(
        EVAL_CUBE, types.SimpleNamespace(timestep=DT, cam_fps=30), 23, 22, pid, jc,
        [old.PinocchioFreeBodyChannel("obj_joint", 16, 16)], old.PinocchioContactConfig(),
        old.PinocchioJointConstraintConfig(enforce_limits=True), render=False)


@pytest.mark.legacy
def test_pinocchio_matches_the_old_simulator(cube_initial):
    """Same solver settings, same controls: identical to rounding, then chaotic.

    ADMM hits its iteration cap on a few percent of steps, so a difference of
    1e-16 grows. The whole-run gap is held to the old simulator's own drift
    when its start state is nudged by 1e-15.
    """
    from ContactModelStudy.Simulators.Pinocchio import Pinocchio, PinocchioConfig
    q0, _, u0 = cube_initial
    U = curl_controls(mujoco.MjModel.from_xml_path(EVAL_CUBE), u0, 0.6, 600)
    new = Pinocchio(EVAL_CUBE, PinocchioConfig(timestep=DT, **OLD_PIN_SETTINGS))
    old, old2 = _old_pinocchio(), _old_pinocchio()
    new.SetState(q0)
    old.reset(q0, np.zeros(22))
    qp = q0.copy(); qp[0] += 1e-15
    old2.reset(qp, np.zeros(22))
    err, self_drift = [], []
    for u in U:
        new.SetControl(u); old.apply_control(u); old2.apply_control(u)
        new.Step(1); old.step(1); old2.step(1)
        qo = old.get_state().qpos
        err.append(np.abs(new.GetState()[0] - qo).max())
        self_drift.append(np.abs(old2.get_state().qpos - qo).max())
    err = np.array(err)
    assert np.isfinite(err).all()
    assert err[:100].max() < 1e-8
    assert err.max() <= 3 * max(max(self_drift), 1e-6)


# -- Drake -------------------------------------------------------------------
def test_drake_inherits_from_the_mjcf(drake_sim, info):
    pl = drake_sim.plant
    for j in info.joints:
        if j.type == "free":
            continue
        joint = pl.GetJointByName(j.name)
        assert np.allclose(joint.default_damping_vector(), j.damping), j.name
        if j.limited:
            assert np.allclose([joint.position_lower_limits()[0], joint.position_upper_limits()[0]], j.range)
    for a in info.actuators:
        act = pl.GetJointActuatorByName(a.name)
        g = act.get_controller_gains()
        assert np.isclose(g.p, a.kp) and np.isclose(g.d, a.kv), a.name
        assert np.isclose(act.default_rotor_inertia(), info.joint(a.joint).armature[0])
        assert act.effort_limit() == (a.force_range[1] if np.isfinite(a.force_range[1]) else np.inf)


def test_drake_collision_candidates_and_friction(drake_sim, pinocchio_sim, info):
    insp = drake_sim.scene_graph.model_inspector()
    cand = {tuple(sorted((drake_sim._geom_to_mj[a], drake_sim._geom_to_mj[b])))
            for a, b in insp.GetCollisionCandidates()}
    assert cand == info.allowed_pairs
    for g, m in drake_sim._geom_to_mj.items():
        cf = insp.GetProximityProperties(g).GetProperty("material", "coulomb_friction")
        assert np.isclose(cf.static_friction(), info.geoms[m].friction)
    p = pinocchio_sim
    pin_pairs = {tuple(sorted((p._pin_to_mj_geom[c.first], p._pin_to_mj_geom[c.second])))
                 for c in p.collision_model.collisionPairs}
    assert pin_pairs == cand


def test_drake_state_roundtrip(drake_sim, cube_initial):
    q0 = cube_initial[0]
    vt = np.linspace(-1, 1, 22) * 0.1
    drake_sim.SetState(q0, vt)
    q, v = drake_sim.GetState()
    assert np.allclose(q, q0, atol=1e-12) and np.allclose(v, vt, atol=1e-12)


@pytest.mark.parametrize("kwargs", [dict(contact_model="x"), dict(discrete_approximation="tamsi"),
                                    dict(penetration_allowance=0), dict(stiction_tolerance=-1)])
def test_drake_config_validation(kwargs):
    pytest.importorskip("pydrake")
    from ContactModelStudy.Simulators.Drake import DrakeConfig
    with pytest.raises(ValueError):
        DrakeConfig(**kwargs)


# -- all three together --------------------------------------------------------
@pytest.fixture(scope="module")
def three_sims(pinocchio_sim, drake_sim):
    from ContactModelStudy.Simulators.Mujoco import Mujoco, MujocoConfig
    return {"mujoco": Mujoco(EVAL_CUBE, MujocoConfig(timestep=DT)),
            "pinocchio": pinocchio_sim, "drake": drake_sim}


def test_all_are_simulators_with_mujoco_dimensions(three_sims):
    for s in three_sims.values():
        assert isinstance(s, Simulator) and (s.nq, s.nv, s.nu) == (23, 22, 16)


@pytest.mark.parametrize("name", ["mujoco", "pinocchio", "drake"])
def test_cube_stays_in_hand_for_two_seconds(three_sims, name, cube_initial):
    q0, _, u0 = cube_initial
    s = three_sims[name]
    s.SetState(q0)
    s.SetControl(u0)
    s.Step(1000)
    q = s.GetState()[0]
    assert np.isfinite(q).all() and q[18] > 0.05


@pytest.mark.parametrize("name", ["pinocchio", "drake"])
def test_hand_tracks_mujoco_in_free_space(three_sims, name, cube_initial):
    """25% curl with the cube parked away from the hand: no contacts, so the servo,
    inertia and damping mapping is all that is compared."""
    q0, _, u0 = cube_initial
    mjm = mujoco.MjModel.from_xml_path(EVAL_CUBE)
    qf = q0.copy()
    qf[16:19] = [0.4, 0.4, 0.05]
    qf[19:23] = [1, 0, 0, 0]
    qf[:16] = np.clip(qf[:16], mjm.jnt_range[:16, 0], mjm.jnt_range[:16, 1])
    U = curl_controls(mjm, u0, 0.25, 1000)
    traj = {}
    for key in ("mujoco", name):
        s = three_sims[key]
        s.SetState(qf)
        out = []
        for u in U:
            s.SetControl(u)
            s.Step(1)
            out.append(s.GetState()[0][:16])
        traj[key] = np.array(out)
    assert np.abs(traj[name] - traj["mujoco"]).max() < 0.01
