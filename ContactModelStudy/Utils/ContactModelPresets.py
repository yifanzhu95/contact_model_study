"""The study's contact models, M1-M5, as ready-made simulators.

Each model is a ``ContactModel`` in ``CONTACT_MODELS``, carrying the numbers
that define it, copied from the old study's
``contact_study/contact_models/config.py``. Build one with
``GetContactModelSim``, passing the task so its own tuning for the model
(``TaskBase.CONTACT_PARAMS``) is applied::

    sim = GetContactModelSim("M3", task.getModelPath(), N=256, task=task,
                             timestep=task.timestep, substeps=4, horizon=8)

The preset is the model's default; a task overrides it where the model works
better fitted to that task (ComFree's stiffness and damping, notably), and an
explicit keyword overrides both.

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
M5        ``Kamino`` (Newton)            Full NCP, solved by PADMM (Newton's
                                         SolverKamino); the reference model.
                                         Needs Newton 1.6 (the contact_kamino
                                         env), and is the slowest model.
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
from dataclasses import dataclass, field
from typing import Any, Mapping

from ContactModelStudy.Simulators.VectorizedSimulator import VectorizedSimulator

# The old MujocoSolverParams defaults, which the old study applied to every model.
_OLD_SOLVER = {"cone": "pyramidal", "solver": "Newton", "iterations": 25, "tolerance": 1e-6}

#: M1's contact time constant, as a multiple of the timestep. Resolved when the
#: simulator is built, because the timestep is only known then; never below
#: 2 * dt, the stability floor of the semi-implicit integrator.
M1_SOLREF_TIMECONST_MULT = 2.0


def _simulatorClasses(kind: str):
    """``(simulator class, config class)`` for a kind, imported on demand."""
    if kind == "mujoco":
        from ContactModelStudy.Simulators.VectorizedMujoco import (
            VectorizedMujoco, VectorizedMujocoConfig)
        return VectorizedMujoco, VectorizedMujocoConfig
    if kind == "comfree":
        from ContactModelStudy.Simulators.ComFree import ComFree, ComFreeConfig
        return ComFree, ComFreeConfig
    if kind == "kamino":
        from ContactModelStudy.Simulators.Kamino import Kamino, KaminoConfig
        return Kamino, KaminoConfig
    from ContactModelStudy.Simulators.XPBD import XPBD, XPBDConfig
    return XPBD, XPBDConfig


@dataclass(frozen=True)
class ContactModel:
    """One contact model of the study: which simulator runs it, and the settings that define it.

    A model's config is built in three layers, each overriding the one before:

    1. ``params``, the model's own preset;
    2. the task's tuning for this model, ``task.contactParams(name)``, so a
       model can be fitted to each task (ComFree's stiffness, say) without
       changing it for every other task;
    3. explicit overrides from the call site: the run's shape (``timestep``,
       ``horizon``, ...) and any sweep over a model parameter.

    Attributes:
        name: The model's canonical name, ``"M1"``-``"M5"``.
        simulator: Which simulator runs it: ``"mujoco"``, ``"comfree"``,
            ``"xpbd"`` or ``"kamino"``.
        description: One line, for printing.
        params: The config fields that make this model what it is.
    """

    name: str
    simulator: str
    description: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def simulatorClass(self) -> type:
        """The simulator class that runs this model, imported on first use."""
        return _simulatorClasses(self.simulator)[0]

    @property
    def configClass(self) -> type:
        """The config class of ``simulatorClass``."""
        return _simulatorClasses(self.simulator)[1]

    def makeConfig(self, task=None, **overrides):
        """The config this model runs with, for ``task`` if given.

        Args:
            task: A task whose tuning for this model (``contactParams``) is laid
                over the preset. ``None`` uses the preset alone.
            **overrides: Config fields set on top of both.

        Raises:
            TypeError: If the task's tuning or an override names a field this
                model's config does not have.
        """
        tuned = dict(task.contactParams(self.name)) if task is not None else {}
        self._checkFields(tuned, f"{type(task).__name__}.contactParams({self.name!r})")
        self._checkFields(overrides, "override")
        fields = {**self.params, **tuned, **overrides}
        self._resolve(fields, explicit=set(tuned) | set(overrides))
        return self.configClass(**fields)

    def makeSim(self, xml, N: int = 1, task=None, **overrides) -> VectorizedSimulator:
        """Build this model's simulator on ``xml`` with ``N`` worlds; see ``makeConfig``."""
        return self.simulatorClass(xml, self.makeConfig(task, **overrides), N=N)

    def _resolve(self, fields: dict, explicit: set) -> None:
        """Fill in fields that depend on others, in place. None here.

        Args:
            fields: The merged fields, about to become the config.
            explicit: The names a task or the call site set, which a derived
                value must not replace.
        """
        return None

    def _checkFields(self, fields: Mapping, source: str) -> None:
        known = {f.name for f in dataclasses.fields(self.configClass)}
        unknown = set(fields) - known
        if unknown:
            raise TypeError(f"{self.configClass.__name__} has no field(s) {sorted(unknown)} "
                            f"(from {source})")


@dataclass(frozen=True)
class StiffMujocoModel(ContactModel):
    """M1: MuJoCo's soft contact at its stiff limit, with a time constant tied to the timestep."""

    def _resolve(self, fields: dict, explicit: set) -> None:
        if "solref_timeconst" in explicit:
            return
        dt = fields.get("timestep", self.configClass().timestep)
        fields["solref_timeconst"] = max(M1_SOLREF_TIMECONST_MULT * dt, 2.0 * dt)


#: Every model, by canonical name.
CONTACT_MODELS: dict[str, ContactModel] = {m.name: m for m in (
    StiffMujocoModel(
        "M1", "mujoco", "stiff-limit pyramidal MJWarp",
        {
            "cone": "pyramidal", "solver": "Newton", "iterations": 200, "tolerance": 1e-10,
            # solimp d -> 1 drives the constraint regularizer toward zero.
            "solimp_d": 0.9999, "solimp_width": 1e-4, "solimp_midpoint": 0.5,
            "solimp_power": 2.0,
            # solref_timeconst is filled in from M1_SOLREF_TIMECONST_MULT.
            "solref_dampratio": 1.0,
        },
    ),
    ContactModel("M2", "mujoco", "MJWarp default soft contact (pyramidal)", dict(_OLD_SOLVER)),
    # The old study used stiffness 0.2, damping 0.001. A task tuned to other
    # values sets them in its CONTACT_PARAMS rather than here.
    ContactModel("M3", "comfree", "complementarity-free contact (Jin 2024)",
                 {**_OLD_SOLVER, "stiffness": 0.75, "damping": 0.01}),
    ContactModel("M4", "xpbd", "XPBD relaxation on MJWarp's constraint rows",
                 {**_OLD_SOLVER, "xpbd_substeps": 1, "xpbd_iterations": 2,
                  "relaxation": 0.1, "vmax_depenetration": 1.0}),
    # M5's solver settings are KaminoConfig's defaults (PADMM with rho0=0.1 and a
    # full warm start, dense dynamics, a 128-contact buffer per world; tuned
    # against objects sticking in a grasp); nothing physical is set here: Kamino
    # reads it from the MJCF, and combines two geoms' friction by MuJoCo's max
    # rule (KaminoConfig.friction_mix_mode), as M1-M4 do.
    ContactModel("M5", "kamino", "Kamino full-NCP PADMM (Newton SolverKamino)"),
)}

#: Other accepted names: the old study's backend names.
ALIASES = {"mujoco_hard": "M1", "mujoco_soft": "M2", "comfree": "M3", "xpbd": "M4", "kamino": "M5"}


def _resolveName(name: str) -> str:
    key = name.strip()
    key = ALIASES.get(key.lower(), key.upper())
    if key not in CONTACT_MODELS:
        raise ValueError(f"unknown contact model {name!r}; choose from "
                         f"{sorted(CONTACT_MODELS)} or {sorted(ALIASES)}")
    return key


def getContactModel(name: str) -> ContactModel:
    """The named model: ``"M1"``-``"M5"`` (case-insensitive) or an old backend name.

    Raises:
        ValueError: If ``name`` is not a known model.
    """
    return CONTACT_MODELS[_resolveName(name)]


def _presetConfig(name: str, task=None, **overrides):
    """The config ``GetContactModelSim`` would build, without building the simulator."""
    return getContactModel(name).makeConfig(task, **overrides)


def GetContactModelSim(name: str, xml: str, N: int = 1, task=None, **overrides) -> VectorizedSimulator:
    """Build a rollout simulator running the named contact model.

    Args:
        name: ``"M1"``-``"M5"`` (case-insensitive), or an old backend name:
            ``"mujoco_hard"``, ``"mujoco_soft"``, ``"comfree"``, ``"xpbd"``,
            ``"kamino"``.
        xml: Path to the rollout MJCF.
        N: Number of parallel worlds.
        task: The task the simulator runs, whose tuning for this model
            (``TaskBase.contactParams``) goes on top of the preset. ``None``
            uses the preset alone.
        **overrides: Config fields to set on top of the preset and the task.
            Normally the run's shape: ``timestep``, ``substeps``, ``horizon``,
            ``device``, ``nconmax``, ``njmax``. A preset parameter can be
            overridden too, for a sweep (for example ``stiffness=0.5`` on M3).
            An explicit value always wins.

    Returns:
        A ``VectorizedMujoco``, ``ComFree``, ``XPBD`` or ``Kamino``, whichever
        runs the model.

    Raises:
        ValueError: If ``name`` is not a known model.
        TypeError: If the task's tuning or an override is not a field of that
            model's config.
    """
    return getContactModel(name).makeSim(xml, N=N, task=task, **overrides)
