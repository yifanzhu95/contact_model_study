"""Standard CPU MuJoCo, wrapped as a ``Simulator``.

This is the reference "real" environment for the study: one ``MjModel`` and one
``MjData``, stepped by ``mj_step``. Because the interface's index layout *is*
MuJoCo's, every method here is an identity map — no remapping, unlike the
Pinocchio and Drake wrappers.

Rendering lives in ``ContactModelStudy.Renderers``, not here. A renderer reads
the public ``mjm`` and ``mjd`` handles below; the simulator itself neither owns
a renderer nor captures frames.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import mujoco

from ContactModelStudy.Simulators.Simulator import Simulator, SimulatorConfig

_SOLVERS = {
    "PGS": mujoco.mjtSolver.mjSOL_PGS,
    "CG": mujoco.mjtSolver.mjSOL_CG,
    "Newton": mujoco.mjtSolver.mjSOL_NEWTON,
}
_CONES = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
}


@dataclass
class MujocoConfig(SimulatorConfig):
    """Physics parameters for the MuJoCo simulators.

    ``timestep``, ``substeps`` and ``gravity`` from ``SimulatorConfig``, plus
    solver and contact overrides. Every override defaults to ``None``, meaning
    "keep what the XML declares", so a plain ``MujocoConfig()`` runs the scene
    exactly as its XML declares and a config changes only what it names.
    ``Mujoco.applyContactParams`` writes them into a model.

    MuJoCo's soft-contact model, in brief: each contact row is a regularized
    constraint whose *impedance* ``d`` (from solimp) sets how much of the
    violation is resisted — ``d -> 1`` is hard, ``d -> 0`` soft — and whose
    *reference* (from solref) is a mass-spring-damper pulling the violation back
    to zero with time constant ``timeconst`` and damping ratio ``dampratio``.

    Attributes:
        cone: Friction cone, ``"pyramidal"`` or ``"elliptic"``.
        solver: Constraint solver, ``"PGS"``, ``"CG"`` or ``"Newton"``.
        iterations: Solver iteration cap.
        tolerance: Solver convergence tolerance.
        solimp_d: Contact impedance, in (0, 1). Written to both ``dmin`` and
            ``dmax``, flattening the impedance curve so every contact gets this
            value whatever its penetration — as the old M1 hard-contact preset
            did. Near 1 is stiff.
        solimp_width: Penetration depth (m) over which impedance would ramp
            from ``dmin`` to ``dmax``. Only matters when they differ, i.e. when
            ``solimp_d`` is not set and the XML's own ``dmin``/``dmax`` are used.
        solimp_midpoint: Where along that width the ramp's sigmoid is centred,
            in (0, 1).
        solimp_power: Sharpness of the ramp, >= 1.
        solref_timeconst: Time constant (s) of the reference spring-damper —
            how fast penetration is corrected. Smaller is stiffer. Below
            ``2 * timestep`` the semi-implicit integrator rings or diverges;
            that draws a warning but is applied as asked, so a sweep probing
            the stiff limit gets the cell it requested.
        solref_dampratio: Damping ratio of that spring-damper; 1 is critical.
        nconmax: Maximum number of contacts. ``None`` keeps the scene's own
            ``<size nconmax>``, which by default means MuJoCo sizes its memory
            arena dynamically. When set, the model is compiled with it, and
            contacts beyond it are dropped with a warning.
        njmax: Maximum number of constraint rows, the same way. ``None`` keeps
            the scene's own ``<size njmax>``.

    The six solimp/solref fields are written to every geom's parameters and to
    every explicit ``<pair>``'s, which between them cover every contact MuJoCo
    can generate. Each is applied independently, so setting only
    ``solref_timeconst`` keeps the XML's damping ratio. Joint limits and
    equality constraints have solref/solimp of their own and are not touched.

    For instance, the old study's M1 "stiff-limit" contact::

        MujocoConfig(timestep=dt, solimp_d=0.9999, solimp_width=1e-4,
                     solimp_midpoint=0.5, solimp_power=2.0,
                     solref_timeconst=2 * dt, solref_dampratio=1.0)
    """

    cone: Optional[str] = "elliptic"
    solver: Optional[str] = None
    iterations: Optional[int] = None
    tolerance: Optional[float] = None
    solimp_d: Optional[float] = None
    solimp_width: Optional[float] = None
    solimp_midpoint: Optional[float] = None
    solimp_power: Optional[float] = None
    solref_timeconst: Optional[float] = None
    solref_dampratio: Optional[float] = None
    nconmax: Optional[int] = 100#
    njmax: Optional[int] = 300#

    def __post_init__(self) -> None:
        super().__post_init__()
        MujocoConfig.validateContactParams(self)

    @staticmethod
    def validateContactParams(cfg) -> None:
        """Reject solver/contact values MuJoCo would silently clamp or misread.

        A static method taking the config explicitly, so ``VectorizedMujocoConfig``
        — which carries the same twelve fields — is checked by this same code
        rather than a copy of it.
        """
        def bad(name, why):
            raise ValueError(f"{name} {why}, got {getattr(cfg, name)!r}")

        if cfg.cone is not None and cfg.cone not in _CONES:
            bad("cone", f"must be one of {tuple(_CONES)}")
        if cfg.solver is not None and cfg.solver not in _SOLVERS:
            bad("solver", f"must be one of {tuple(_SOLVERS)}")
        if cfg.iterations is not None and cfg.iterations < 1:
            bad("iterations", "must be >= 1")
        if cfg.tolerance is not None and cfg.tolerance <= 0.0:
            bad("tolerance", "must be positive")
        if cfg.solimp_d is not None and not 0.0 < cfg.solimp_d < 1.0:
            bad("solimp_d", "must be in (0, 1)")
        if cfg.solimp_width is not None and cfg.solimp_width <= 0.0:
            bad("solimp_width", "must be positive")
        if cfg.solimp_midpoint is not None and not 0.0 < cfg.solimp_midpoint < 1.0:
            bad("solimp_midpoint", "must be in (0, 1)")
        if cfg.solimp_power is not None and cfg.solimp_power < 1.0:
            bad("solimp_power", "must be >= 1")
        # Positive only: MuJoCo reads a *negative* solref as direct stiffness
        # and damping instead, a different parameterization entirely.
        if cfg.solref_timeconst is not None and cfg.solref_timeconst <= 0.0:
            bad("solref_timeconst", "must be positive (a time constant in seconds)")
        if cfg.solref_dampratio is not None and cfg.solref_dampratio <= 0.0:
            bad("solref_dampratio", "must be positive")
        if cfg.nconmax is not None and cfg.nconmax < 1:
            bad("nconmax", "must be >= 1")
        if cfg.njmax is not None and cfg.njmax < 1:
            bad("njmax", "must be >= 1")


class Mujoco(Simulator):
    """Single-world MuJoCo simulator.

    Attributes:
        mjm: The compiled ``mujoco.MjModel``. Public so renderers and tasks can
            read model constants (body ids, actuator ranges) without a getter
            per field.
        mjd: The live ``mujoco.MjData``. Public for the same reason; treat it as
            read-only from the outside — writes should go through ``SetState``
            and ``SetControl`` so derived quantities stay consistent.
    """

    def __init__(self, xml: str | Path, sim_config: SimulatorConfig | None = None):
        """Compile a model and allocate its data.

        The config's ``timestep`` and ``gravity`` overwrite whatever the XML
        declared, so a scene file and a sweep config cannot disagree about the
        physics. With a ``MujocoConfig``, its solver and contact overrides are
        applied too; every field it leaves ``None`` — and everything a plain
        ``SimulatorConfig`` has no field for — stays as the XML has it.

        Args:
            xml: Path to an MJCF file, or the MJCF document itself.
            sim_config: Physics parameters. Defaults to ``MujocoConfig()``.

        Raises:
            FileNotFoundError: If ``xml`` looks like a path and does not exist.
            ValueError: If MuJoCo cannot compile the model.
        """
        super().__init__(xml, sim_config if sim_config is not None else MujocoConfig())

        # Compile from the path when there is one: MuJoCo resolves <mesh>,
        # <texture> and <include> references relative to the XML's own
        # directory, which it cannot do for a string. Inline XML that
        # references external assets will fail to compile — pass a path for
        # scenes with meshes.
        self.mjm = self._compile()

        self.mjm.opt.timestep = self.config.timestep
        self.mjm.opt.gravity[:] = self.config.gravity
        # After the timestep, which the solref stability check reads. A plain
        # SimulatorConfig carries no overrides, so there is nothing to apply.
        if isinstance(self.config, MujocoConfig):
            self.applyContactParams(self.mjm, self.config)

        self.mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, self.mjd)

    def _compile(self) -> mujoco.MjModel:
        """Compile the model, with the config's buffer sizes when it sets any.

        ``nconmax`` and ``njmax`` are fixed at compile time (they size MuJoCo's
        memory arena and are read-only on a compiled model), so when either is
        set the scene goes through ``MjSpec`` and they are written before
        compiling. Otherwise it is compiled directly.
        """
        cfg = self.config
        nconmax = getattr(cfg, "nconmax", None)
        njmax = getattr(cfg, "njmax", None)
        if nconmax is None and njmax is None:
            if self.model_path is not None:
                return mujoco.MjModel.from_xml_path(str(self.model_path))
            return mujoco.MjModel.from_xml_string(self.model_xml)
        spec = (mujoco.MjSpec.from_file(str(self.model_path)) if self.model_path is not None
                else mujoco.MjSpec.from_string(self.model_xml))
        if nconmax is not None:
            spec.nconmax = int(nconmax)
        if njmax is not None:
            spec.njmax = int(njmax)
        return spec.compile()

    # -- configuration -------------------------------------------------------
    @staticmethod
    def applyContactParams(mjm: mujoco.MjModel, cfg) -> None:
        """Write a config's solver and contact overrides into ``mjm``, in place.

        ``None`` fields are skipped, leaving the XML's values. Call after
        ``mjm.opt.timestep`` is final: the ``solref_timeconst`` stability check
        compares against it.

        Static, taking the model and config explicitly, so ``VectorizedMujoco``
        applies its (identically named) fields with this same code before
        uploading its model — one implementation, so the CPU and GPU simulators
        cannot come to mean different things by the same setting.
        """
        if cfg.cone is not None:
            mjm.opt.cone = _CONES[cfg.cone]
        if cfg.solver is not None:
            mjm.opt.solver = _SOLVERS[cfg.solver]
        if cfg.iterations is not None:
            mjm.opt.iterations = int(cfg.iterations)
        if cfg.tolerance is not None:
            mjm.opt.tolerance = float(cfg.tolerance)

        # (column, value) pairs for the (n, 5) solimp and (n, 2) solref tables.
        solimp = []
        if cfg.solimp_d is not None:
            solimp += [(0, cfg.solimp_d), (1, cfg.solimp_d)]      # dmin == dmax
        if cfg.solimp_width is not None:
            solimp.append((2, cfg.solimp_width))
        if cfg.solimp_midpoint is not None:
            solimp.append((3, cfg.solimp_midpoint))
        if cfg.solimp_power is not None:
            solimp.append((4, cfg.solimp_power))
        solref = []
        if cfg.solref_timeconst is not None:
            dt = float(mjm.opt.timestep)
            if cfg.solref_timeconst < 2.0 * dt:
                warnings.warn(
                    f"solref_timeconst={cfg.solref_timeconst:g}s is below the 2*dt="
                    f"{2.0 * dt:g}s stability floor for the semi-implicit integrator; "
                    f"contact may ring or diverge. Applying it as requested.",
                    RuntimeWarning, stacklevel=3,
                )
            solref.append((0, cfg.solref_timeconst))
        if cfg.solref_dampratio is not None:
            solref.append((1, cfg.solref_dampratio))

        for col, val in solimp:
            mjm.geom_solimp[:, col] = val
            if mjm.npair > 0:
                mjm.pair_solimp[:, col] = val
        for col, val in solref:
            mjm.geom_solref[:, col] = val
            if mjm.npair > 0:
                mjm.pair_solref[:, col] = val

    # -- state ---------------------------------------------------------------
    def SetState(self, q: np.ndarray, q_dot: np.ndarray | None = None) -> None:
        """Overwrite qpos/qvel and refresh derived quantities.

        Args:
            q: Positions, shape ``(nq,)``.
            q_dot: Velocities, shape ``(nv,)``. ``None`` zeroes them.

        Runs ``mj_forward`` afterwards, so contacts, site positions and sensor
        readings are consistent with the new state before anything reads them.
        """
        self.mjd.qpos[:] = self._check(q, self.nq, "q")
        self.mjd.qvel[:] = (
            np.zeros(self.nv) if q_dot is None else self._check(q_dot, self.nv, "q_dot")
        )
        mujoco.mj_forward(self.mjm, self.mjd)

    def GetState(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(qpos, qvel)`` as copies, shapes ``(nq,)`` and ``(nv,)``."""
        return self.mjd.qpos.copy(), self.mjd.qvel.copy()

    # -- control -------------------------------------------------------------
    def SetControl(self, u: np.ndarray) -> None:
        """Set the actuator command, shape ``(nu,)``.

        MuJoCo clamps this to each actuator's ``ctrlrange`` at step time when
        the actuator is ``ctrllimited``, so ``GetControl`` can report a value
        the solver will not actually apply. Clamp before calling if the caller
        needs the two to agree.
        """
        self.mjd.ctrl[:] = self._check(u, self.nu, "u")

    def GetControl(self) -> np.ndarray:
        """Return the current actuator command as a copy, shape ``(nu,)``."""
        return self.mjd.ctrl.copy()

    # -- stepping ------------------------------------------------------------
    def Step(self, steps: int = 1) -> None:
        """Advance by ``steps`` calls to ``mj_step``, holding the control fixed."""
        for _ in range(steps):
            mujoco.mj_step(self.mjm, self.mjd)

    # -- dimensions ----------------------------------------------------------
    @property
    def nq(self) -> int:
        return self.mjm.nq

    @property
    def nv(self) -> int:
        return self.mjm.nv

    @property
    def nu(self) -> int:
        return self.mjm.nu

    @property
    def timestep(self) -> float:
        """The timestep MuJoCo is actually integrating with."""
        return float(self.mjm.opt.timestep)

    @property
    def time(self) -> float:
        """Simulated time in seconds, as MuJoCo counts it.

        Advanced by ``Step``. ``SetState`` deliberately does not rewind it —
        teleporting the state mid-episode is a perturbation, not a reset — so a
        driver that reuses one simulator across episodes should track episode
        time itself.
        """
        return float(self.mjd.time)

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _check(x: np.ndarray, dim: int, name: str) -> np.ndarray:
        """Validate a 1-D array's length, so a mismatch names the offending arg.

        Assigning a wrong-length array into an MjData field raises a ValueError
        that mentions only the buffer shape, which is hard to trace back to the
        call site in a rollout loop.
        """
        a = np.asarray(x, dtype=float).ravel()
        if a.shape != (dim,):
            raise ValueError(f"{name} must have shape ({dim},), got {np.shape(x)}")
        return a
