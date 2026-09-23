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

from pathlib import Path

import numpy as np
import mujoco

from ContactModelStudy.Simulators.Simulator import Simulator, SimulatorConfig


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
        physics. Everything else — integrator, solver, friction cone, contact
        solref/solimp — is left exactly as the XML has it.

        Args:
            xml: Path to an MJCF file, or the MJCF document itself.
            sim_config: Physics parameters. Defaults to ``SimulatorConfig()``.

        Raises:
            FileNotFoundError: If ``xml`` looks like a path and does not exist.
            ValueError: If MuJoCo cannot compile the model.
        """
        super().__init__(xml, sim_config)

        # Compile from the path when there is one: MuJoCo resolves <mesh>,
        # <texture> and <include> references relative to the XML's own
        # directory, which it cannot do for a string. Inline XML that
        # references external assets will fail to compile — pass a path for
        # scenes with meshes.
        if self.model_path is not None:
            self.mjm = mujoco.MjModel.from_xml_path(str(self.model_path))
        else:
            self.mjm = mujoco.MjModel.from_xml_string(self.model_xml)

        self.mjm.opt.timestep = self.config.timestep
        self.mjm.opt.gravity[:] = self.config.gravity

        self.mjd = mujoco.MjData(self.mjm)
        mujoco.mj_forward(self.mjm, self.mjd)

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
