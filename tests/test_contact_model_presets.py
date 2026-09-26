"""M1-M4 presets: each reproduces the old study's model."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ROLLOUT_CUBE, matches_within_noise
from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim, _presetConfig

N, T, DT = 16, 25, 0.004


def _opt(v):
    return float(v.numpy()[0]) if hasattr(v, "numpy") else float(v)


def _old(name, q0, U):
    import mujoco
    from contact_study.contact_models import api
    from contact_study.contact_models.config import ContactModelConfig
    mjm = mujoco.MjModel.from_xml_path(ROLLOUT_CUBE)
    mjm.opt.timestep = DT
    m = api.put_model(mjm, getattr(ContactModelConfig, name)())
    d = api.make_data(mjm, m, nworld=N)
    d.qpos.assign(np.tile(q0, (N, 1)).astype(np.float32))
    d.qvel.zero_()
    api.forward(m, d)
    for t in range(T):
        d.ctrl.assign(U[t])
        api.step(m, d)
    return m, d.qpos.numpy()


@pytest.mark.gpu
@pytest.mark.legacy
@pytest.mark.parametrize("name", ["M1", "M2", "M3", "M4"])
def test_preset_matches_old_study(name, cube_initial):
    from contact_study.contact_models.config import ContactModelConfig
    q0, _, u0 = cube_initial
    U = (u0 + np.random.default_rng(0).normal(0, 0.3, (T, N, 16))).astype(np.float32)
    m_old, qa = _old(name, q0, U)
    olds = [qa] + [_old(name, q0, U)[1] for _ in range(2)]
    sim = GetContactModelSim(name, ROLLOUT_CUBE, N=N, timestep=DT, horizon=2, nconmax=None, njmax=None)
    inner_old = getattr(m_old, "_m", m_old)
    for k in ("iterations", "tolerance", "solver", "cone"):
        assert _opt(getattr(sim.m.opt, k)) == _opt(getattr(inner_old.opt, k)), k
    assert np.array_equal(sim.m.geom_solref.numpy(), inner_old.geom_solref.numpy())
    assert np.array_equal(sim.m.geom_solimp.numpy(), inner_old.geom_solimp.numpy())
    if name == "M3":
        assert np.allclose(sim.m.comfree_stiffness.numpy(), m_old.comfree_stiffness.numpy())
        assert np.allclose(sim.m.comfree_damping.numpy(), m_old.comfree_damping.numpy())
    if name == "M4":
        p, c = ContactModelConfig.M4().xpbd, sim.config
        assert (c.xpbd_substeps, c.xpbd_iterations, c.relaxation, c.vmax_depenetration) == \
            (p.substeps, p.iterations, p.relaxation, p.vmax_depenetration)
    sim.SetState(q0)
    for t in range(T):
        sim.SetControl(U[t])
        sim.Step_GPU(1)
    assert matches_within_noise(sim.GetState()[0], olds)


@pytest.mark.gpu
def test_aliases_and_case():
    assert type(GetContactModelSim("comfree", ROLLOUT_CUBE, N=2, horizon=1)).__name__ == "ComFree"
    assert _presetConfig("m3").stiffness == 0.2


def test_m1_timeconst_follows_the_timestep():
    assert np.isclose(_presetConfig("M1", timestep=0.001).solref_timeconst, 0.002)
    assert np.isclose(_presetConfig("M1", timestep=0.004).solref_timeconst, 0.008)


def test_overrides_win_over_the_preset():
    assert _presetConfig("M3", stiffness=0.5).stiffness == 0.5
    assert _presetConfig("M1", solref_timeconst=0.05).solref_timeconst == 0.05


def test_run_shape_passes_through():
    c = _presetConfig("M4", substeps=4, horizon=8, nconmax=5000)
    assert (c.substeps, c.horizon, c.nconmax) == (4, 8, 5000)
    c = _presetConfig("M3", timestep=0.004, ctrl_time_step=0.016, time_horizon=0.128)
    assert c.resolved_horizon == 8


def test_unknown_model_name():
    with pytest.raises(ValueError):
        _presetConfig("M5")


def test_override_that_is_not_a_field_of_that_model():
    with pytest.raises(TypeError, match="stiffness"):
        _presetConfig("M2", stiffness=1.0)
