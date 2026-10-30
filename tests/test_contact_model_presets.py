"""M1-M4 presets: each reproduces the old study's model."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ROLLOUT_CUBE, matches_within_noise
from ContactModelStudy.Drivers.run_episodes import TASKS
from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Utils.ContactModelPresets import (CONTACT_MODELS, GetContactModelSim, _presetConfig,
                                                         getContactModel)

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
    assert _presetConfig("m3").stiffness == CONTACT_MODELS["M3"].params["stiffness"]


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
        _presetConfig("M9")


def test_override_that_is_not_a_field_of_that_model():
    with pytest.raises(TypeError, match="stiffness"):
        _presetConfig("M2", stiffness=1.0)


# -- per-task tuning -------------------------------------------------------------
class _TunedCube(CubeReorient):
    CONTACT_PARAMS = {"M3": {"stiffness": 0.3, "damping": 0.004}, "M1": {"solref_timeconst": 0.05}}


class _RetunedCube(_TunedCube):
    CONTACT_PARAMS = {"M3": {"damping": 0.02}}


def test_task_tuning_wins_over_the_preset():
    c = _presetConfig("M3", _TunedCube())
    assert (c.stiffness, c.damping) == (0.3, 0.004)


def test_override_wins_over_the_task():
    c = _presetConfig("M3", _TunedCube(), stiffness=0.9)
    assert (c.stiffness, c.damping) == (0.9, 0.004)


def test_a_model_the_task_does_not_list_runs_its_preset():
    assert _presetConfig("M4", _TunedCube()) == _presetConfig("M4")


def test_aliases_find_the_tuning():
    assert _TunedCube().contactParams("comfree") == {"stiffness": 0.3, "damping": 0.004}
    assert getContactModel("comfree") is CONTACT_MODELS["M3"]


def test_tuning_merges_down_the_class_hierarchy():
    assert _RetunedCube().contactParams("M3") == {"stiffness": 0.3, "damping": 0.02}


def test_task_timeconst_stops_m1_deriving_it():
    assert _presetConfig("M1", _TunedCube(), timestep=0.004).solref_timeconst == 0.05


def test_a_bad_field_in_a_task_table_names_the_task():
    class Bad(CubeReorient):
        CONTACT_PARAMS = {"M2": {"stiffness": 1.0}}
    with pytest.raises(TypeError, match="Bad.contactParams"):
        _presetConfig("M2", Bad())


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_every_task_table_builds(task_name):
    task = TASKS[task_name]()
    for model in CONTACT_MODELS:
        _presetConfig(model, task)
