"""The study's four rollout contact models, M1-M4, as ready-made simulators.

Every number that defines a contact model is in the table below, copied from
the old study's ``contact_study/contact_models/config.py``. Build one with
``GetContactModelSim``::

    sim = GetContactModelSim("M3", task.getModelPath(), N=256,
                             timestep=task.timestep, substeps=4, horizon=8)

========  =============================  =====================================
Model     Simulator                      What makes it that model
========  =============================  =====================================
M1        ``VectorizedMujoco`` (MJWarp)  Stiff limit of MuJoCo's soft contact:
                                         impedance ~1, solref timeconst 2*dt,
                                         Newton run to 200 its / 1e-10.
M2        ``VectorizedMujoco`` (MJWarp)  MuJoCo's default soft contact (the
                                         scene's own solref/solimp).
M3        ``ComFree``                    Complementarity-free contact (Jin 2024).
M4        ``XPBD``                       XPBD relaxation on MJWarp's rows.
========  =============================  =====================================

All four use pyramidal friction cones, the only kind MJWarp has, so the
paper's elliptic "MuJoCo default" M2 is reported as pyramidal.

M2-M4 carry the old study's solver settings (Newton, 25 iterations, 1e-6),
which it wrote onto every model. The leap scenes themselves declare 100
iterations and 1e-8, so a plain ``VectorizedMujoco`` on those scenes is not
quite M2. ComFree and XPBD do not use MuJoCo's solver, so for them these three
settings are inert, but they are kept so that each model is exactly the old
one.
"""

from __future__ import annotations

import dataclasses

from ContactModelStudy.Simulators.VectorizedSimulator import VectorizedSimulator

# The old MujocoSolverParams defaults, which the old study applied to every model.
_OLD_SOLVER = {"cone": "pyramidal", "solver": "Newton", "iterations": 25, "tolerance": 1e-6}

#: M1's contact time constant, as a multiple of the timestep. Resolved when the
#: simulator is built, because the timestep is only known then; never below
#: 2 * dt, the stability floor of the semi-implicit integrator.
M1_SOLREF_TIMECONST_MULT = 2.0

#: Each model: which simulator runs it, and the config fields that define it.
CONTACT_MODELS: dict[str, dict] = {
    "M1": {
        "simulator": "mujoco",
        "description": "stiff-limit pyramidal MJWarp",
        "params": {
            "cone": "pyramidal", "solver": "Newton", "iterations": 200, "tolerance": 1e-10,
            # solimp d -> 1 drives the constraint regularizer toward zero.
            "solimp_d": 0.9999, "solimp_width": 1e-4, "solimp_midpoint": 0.5,
            "solimp_power": 2.0,
            # solref_timeconst is filled in from M1_SOLREF_TIMECONST_MULT.
            "solref_dampratio": 1.0,
        },
    },
    "M2": {
        "simulator": "mujoco",
        "description": "MJWarp default soft contact (pyramidal)",
        "params": dict(_OLD_SOLVER),
    },
    "M3": {
        "simulator": "comfree",
        "description": "complementarity-free contact (Jin 2024)",
        "params": {**_OLD_SOLVER, "stiffness": 0.2, "damping": 0.001},
    },
    "M4": {
        "simulator": "xpbd",
        "description": "XPBD relaxation on MJWarp's constraint rows",
        "params": {**_OLD_SOLVER, "xpbd_substeps": 1, "xpbd_iterations": 2,
                   "relaxation": 0.1, "vmax_depenetration": 1.0},
    },
}

#: Other accepted names: the old study's backend names.
ALIASES = {"mujoco_hard": "M1", "mujoco_soft": "M2", "comfree": "M3", "xpbd": "M4"}


def _simulatorClasses(kind: str):
    """``(simulator class, config class)`` for a kind, imported on demand."""
    if kind == "mujoco":
        from ContactModelStudy.Simulators.VectorizedMujoco import (
            VectorizedMujoco, VectorizedMujocoConfig)
        return VectorizedMujoco, VectorizedMujocoConfig
    if kind == "comfree":
        from ContactModelStudy.Simulators.ComFree import ComFree, ComFreeConfig
        return ComFree, ComFreeConfig
    from ContactModelStudy.Simulators.XPBD import XPBD, XPBDConfig
    return XPBD, XPBDConfig


def _resolveName(name: str) -> str:
    key = name.strip()
    key = ALIASES.get(key.lower(), key.upper())
    if key not in CONTACT_MODELS:
        raise ValueError(f"unknown contact model {name!r}; choose from "
                         f"{sorted(CONTACT_MODELS)} or {sorted(ALIASES)}")
    return key


def _presetConfig(name: str, **overrides):
    """The config ``GetContactModelSim`` would build, without building the simulator."""
    key = _resolveName(name)
    preset = CONTACT_MODELS[key]
    _, config_cls = _simulatorClasses(preset["simulator"])
    known = {f.name for f in dataclasses.fields(config_cls)}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"{config_cls.__name__} has no field(s) {sorted(unknown)}")
    fields = {**preset["params"], **overrides}
    if key == "M1" and "solref_timeconst" not in overrides:
        dt = fields.get("timestep", config_cls().timestep)
        fields["solref_timeconst"] = max(M1_SOLREF_TIMECONST_MULT * dt, 2.0 * dt)
    return config_cls(**fields)


def GetContactModelSim(name: str, xml: str, N: int = 1, **overrides) -> VectorizedSimulator:
    """Build a rollout simulator running the named contact model.

    Args:
        name: ``"M1"``-``"M4"`` (case-insensitive), or an old backend name:
            ``"mujoco_hard"``, ``"mujoco_soft"``, ``"comfree"``, ``"xpbd"``.
        xml: Path to the rollout MJCF.
        N: Number of parallel worlds.
        **overrides: Config fields to set on top of the preset. Normally the
            run's shape: ``timestep``, ``substeps``, ``horizon``, ``device``,
            ``nconmax``, ``njmax``. A preset parameter can be overridden too,
            for a sweep (for example ``stiffness=0.5`` on M3). An explicit value
            always wins over the preset.

    Returns:
        A ``VectorizedMujoco``, ``ComFree`` or ``XPBD``, whichever runs the model.

    Raises:
        ValueError: If ``name`` is not a known model.
        TypeError: If an override is not a field of that model's config.
    """
    config = _presetConfig(name, **overrides)
    sim_cls, _ = _simulatorClasses(CONTACT_MODELS[_resolveName(name)]["simulator"])
    return sim_cls(xml, config, N=N)
