"""Simulator configs: the control-step and horizon pairs, contact parameters, buffer sizes."""

from __future__ import annotations

import warnings

import mujoco
import numpy as np
import pytest

from conftest import EVAL_CUBE, ROLLOUT_CUBE
from ContactModelStudy.Simulators.Mujoco import Mujoco, MujocoConfig
from ContactModelStudy.Simulators.Simulator import SimulatorConfig
from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujocoConfig

CONTACT_FIELDS = ["cone", "solver", "iterations", "tolerance", "solimp_d", "solimp_width",
                  "solimp_midpoint", "solimp_power", "solref_timeconst", "solref_dampratio"]


# -- control step: substeps or ctrl_time_step --------------------------------
def test_neither_substeps_field_means_one_step():
    c = SimulatorConfig(timestep=0.004)
    assert c.resolved_substeps == 1 and c.substeps is None and c.ctrl_time_step is None


def test_substeps_used_as_given():
    assert SimulatorConfig(timestep=0.004, substeps=4).resolved_substeps == 4


@pytest.mark.parametrize("ctrl, expected", [(0.016, 4), (0.018, 4), (0.032, 8)])
def test_ctrl_time_step_floors_to_whole_steps(ctrl, expected):
    # 0.032 / 0.004 is 7.999999 in binary; the tolerance keeps it at 8.
    c = SimulatorConfig(timestep=0.004, ctrl_time_step=ctrl)
    assert c.resolved_substeps == expected
    assert np.isclose(c.control_timestep, expected * 0.004)


def test_ctrl_time_step_shorter_than_a_step_clamps_with_one_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = SimulatorConfig(timestep=0.004, ctrl_time_step=0.001)
    assert c.resolved_substeps == 1
    assert len(w) == 1 and "ctrl_time_step" in str(w[0].message)


# -- horizon: horizon or time_horizon ----------------------------------------
def test_time_horizon_counts_control_steps():
    v = VectorizedMujocoConfig(timestep=0.004, substeps=4, time_horizon=0.25)
    assert v.resolved_horizon == 15 and np.isclose(v.horizon_duration, 0.24)


def test_both_durations_together():
    v = VectorizedMujocoConfig(timestep=0.004, ctrl_time_step=0.032, time_horizon=0.256)
    assert (v.resolved_substeps, v.resolved_horizon) == (8, 8)


def test_neither_horizon_field_means_one():
    assert VectorizedMujocoConfig().resolved_horizon == 1


@pytest.mark.parametrize("cls, kwargs", [
    (SimulatorConfig, dict(substeps=2, ctrl_time_step=0.01)),
    (MujocoConfig, dict(substeps=2, ctrl_time_step=0.01)),
    (VectorizedMujocoConfig, dict(horizon=8, time_horizon=0.2)),
    (SimulatorConfig, dict(substeps=0)),
    (SimulatorConfig, dict(ctrl_time_step=0.0)),
    (VectorizedMujocoConfig, dict(time_horizon=-1.0)),
    (VectorizedMujocoConfig, dict(horizon=0)),
])
def test_schedule_validation(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_both_of_a_pair_names_the_fields():
    with pytest.raises(ValueError, match="set substeps or ctrl_time_step, not both"):
        SimulatorConfig(substeps=2, ctrl_time_step=0.01)


def test_config_rebuilds_from_its_own_fields():
    v = VectorizedMujocoConfig(timestep=0.004, ctrl_time_step=0.032, time_horizon=0.256)
    rebuilt = VectorizedMujocoConfig(**{k: getattr(v, k) for k in v.__dataclass_fields__})
    assert rebuilt.resolved_horizon == v.resolved_horizon


# -- contact parameters ------------------------------------------------------
@pytest.mark.parametrize("cls", [MujocoConfig, VectorizedMujocoConfig])
def test_configs_carry_all_contact_fields(cls):
    names = {f for f in cls.__dataclass_fields__}
    assert set(CONTACT_FIELDS) <= names and {"nconmax", "njmax"} <= names


def test_none_contact_fields_leave_the_scene_alone():
    # Explicit, since the config's own defaults may set some (cone does).
    raw = mujoco.MjModel.from_xml_path(EVAL_CUBE)
    m = Mujoco(EVAL_CUBE, MujocoConfig(**{f: None for f in CONTACT_FIELDS})).mjm
    assert np.array_equal(raw.geom_solref, m.geom_solref)
    assert np.array_equal(raw.geom_solimp, m.geom_solimp)
    assert (raw.opt.solver, raw.opt.iterations, raw.opt.tolerance, raw.opt.cone) == \
        (m.opt.solver, m.opt.iterations, m.opt.tolerance, m.opt.cone)


def test_every_contact_field_reaches_the_cpu_model():
    vals = dict(cone="elliptic", solver="CG", iterations=37, tolerance=3e-7, solimp_d=0.95,
                solimp_width=0.003, solimp_midpoint=0.4, solimp_power=3.0,
                solref_timeconst=0.01, solref_dampratio=0.8)
    m = Mujoco(EVAL_CUBE, MujocoConfig(**vals)).mjm
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert m.opt.solver == mujoco.mjtSolver.mjSOL_CG and m.opt.iterations == 37
    assert np.isclose(m.opt.tolerance, 3e-7)
    assert np.all(m.geom_solimp[:, :2] == 0.95)          # solimp_d sets dmin and dmax
    assert np.all(m.geom_solimp[:, 2:] == [0.003, 0.4, 3.0])
    assert np.all(m.geom_solref == [0.01, 0.8])


@pytest.mark.gpu
def test_every_contact_field_reaches_the_gpu_model():
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco
    vals = dict(solver="CG", iterations=37, solimp_d=0.95, solimp_width=0.003,
                solref_timeconst=0.01, solref_dampratio=0.8)
    g = VectorizedMujoco(ROLLOUT_CUBE, VectorizedMujocoConfig(horizon=2, **vals), N=2)
    gi = g.m.geom_solimp.numpy().reshape(-1, 5)[-g.mjm.ngeom:]
    gr = g.m.geom_solref.numpy().reshape(-1, 2)[-g.mjm.ngeom:]
    assert g.mjm.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
    assert g.mjm.opt.solver == mujoco.mjtSolver.mjSOL_CG and g.mjm.opt.iterations == 37
    assert np.allclose(gi[:, :2], 0.95) and np.allclose(gi[:, 2], 0.003)
    assert np.allclose(gr, [0.01, 0.8])


@pytest.mark.legacy
def test_m1_hard_contact_preset_rebuilt_bit_identically():
    from contact_study.contact_models.api import _apply_hard_contact_preset
    from contact_study.contact_models.config import ContactModelConfig
    dt = 0.002
    old = mujoco.MjModel.from_xml_path(EVAL_CUBE)
    old.opt.timestep = dt
    _apply_hard_contact_preset(old, ContactModelConfig.M1().mujoco)
    new = Mujoco(EVAL_CUBE, MujocoConfig(
        timestep=dt, solimp_d=0.9999, solimp_width=1e-4, solimp_midpoint=0.5, solimp_power=2.0,
        solref_timeconst=2 * dt, solref_dampratio=1.0)).mjm
    assert np.array_equal(old.geom_solimp, new.geom_solimp)
    assert np.array_equal(old.geom_solref, new.geom_solref)


@pytest.mark.parametrize("cls, kwargs", [
    (MujocoConfig, dict(solimp_d=1.5)), (MujocoConfig, dict(solver="Bogus")),
    (MujocoConfig, dict(iterations=0)), (MujocoConfig, dict(solref_dampratio=0.0)),
    (VectorizedMujocoConfig, dict(cone="elliptic")), (VectorizedMujocoConfig, dict(solimp_d=2.0)),
])
def test_contact_param_validation(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_solref_timeconst_below_two_dt_warns_but_applies():
    with pytest.warns(RuntimeWarning):
        m = Mujoco(EVAL_CUBE, MujocoConfig(timestep=0.002, solref_timeconst=0.001)).mjm
    assert np.all(m.geom_solref[:, 0] == 0.001)


# -- nconmax / njmax ---------------------------------------------------------
def test_buffer_sizes_reach_the_cpu_model():
    a = Mujoco(EVAL_CUBE, MujocoConfig(nconmax=64, njmax=300))
    assert (a.mjm.nconmax, a.mjm.njmax) == (64, 300)
    b = Mujoco(EVAL_CUBE, MujocoConfig(nconmax=None, njmax=300))
    assert (b.mjm.nconmax, b.mjm.njmax) == (-1, 300)


def test_none_buffer_sizes_keep_the_scene_default():
    m = Mujoco(EVAL_CUBE, MujocoConfig(nconmax=None, njmax=None)).mjm
    assert (m.nconmax, m.njmax) == (-1, -1)


def test_buffer_sizes_change_nothing_else():
    d0 = Mujoco(EVAL_CUBE, MujocoConfig(nconmax=None, njmax=None))
    a = Mujoco(EVAL_CUBE, MujocoConfig(nconmax=64, njmax=300))
    for f in ("geom_size", "body_mass", "dof_damping", "actuator_gainprm"):
        assert np.array_equal(getattr(d0.mjm, f), getattr(a.mjm, f))
    q0 = d0.GetState()[0]
    for s in (d0, a):
        s.SetState(q0)
        s.Step(200)
    assert np.allclose(d0.GetState()[0], a.GetState()[0])


def test_inline_xml_with_buffer_sizes():
    xml = "<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>"
    assert Mujoco(xml, MujocoConfig(njmax=50)).mjm.njmax == 50


@pytest.mark.gpu
def test_buffer_sizes_are_per_world_on_the_gpu():
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco
    v = VectorizedMujoco(ROLLOUT_CUBE, VectorizedMujocoConfig(nconmax=4000, njmax=20000, horizon=2), N=4)
    assert v.d.naconmax == 4000 * 4 and v.d.efc.J.shape[1] == 20000


@pytest.mark.parametrize("cls", [MujocoConfig, VectorizedMujocoConfig])
@pytest.mark.parametrize("kwargs", [dict(nconmax=0), dict(njmax=-5)])
def test_buffer_size_validation(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)
