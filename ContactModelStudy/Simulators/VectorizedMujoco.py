"""MuJoCo Warp (MJWarp), wrapped as a ``VectorizedSimulator``.

``N`` worlds of one model, stepped together on the GPU. This is the rollout
engine sampling-based planners run against: one world per sampled control
sequence.

Controls stay on the device. ``SetControlSequence`` uploads the whole ``(N, H,
nu)`` block once, and each ``Step_GPU`` writes that step's slice into ``d.ctrl``
with a kernel — no host round-trip per step, which is the difference between a
rollout that is GPU-bound and one that is PCIe-bound.

MJWarp limitation: only pyramidal friction cones are implemented on the GPU.
The cone is forced to pyramidal at upload; see ``VectorizedMujocoConfig.cone``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import mujoco
import warp as wp

from ContactModelStudy.Simulators.VectorizedSimulator import (
    DeviceState,
    VectorizedSimulator,
    VectorizedSimulatorConfig,
)

_SOLVERS = {
    "PGS": mujoco.mjtSolver.mjSOL_PGS,
    "CG": mujoco.mjtSolver.mjSOL_CG,
    "Newton": mujoco.mjtSolver.mjSOL_NEWTON,
}


@wp.kernel
def _assign_ctrl_kernel(
    U: wp.array3d(dtype=float),     # (N, H, nu)
    t: int,
    ctrl: wp.array2d(dtype=float),  # (N, nu)  [out]
):
    """Copy the t-th slice of the control sequence into d.ctrl, on the device."""
    n, u = wp.tid()
    ctrl[n, u] = U[n, t, u]


@wp.kernel
def _broadcast_kernel(
    src: wp.array(dtype=float),     # (n,)
    dst: wp.array2d(dtype=float),   # (N, n)  [out]
):
    """Copy one state vector into every world."""
    n, i = wp.tid()
    dst[n, i] = src[i]


@dataclass
class VectorizedMujocoConfig(VectorizedSimulatorConfig):
    """Physics parameters for the MJWarp backend.

    Inherits ``timestep``, ``substeps``, ``gravity``, ``horizon`` and ``device``.

    The solver fields default to ``None``, meaning "leave whatever the XML
    declares" — the same policy as the CPU ``Mujoco`` wrapper, so the two agree
    about a scene unless told otherwise. Only ``timestep``, ``gravity`` and the
    cone are written unconditionally.

    Attributes:
        cone: Friction cone. MJWarp implements only ``"pyramidal"`` on the GPU,
            so this is written to the model at upload and anything else is
            rejected. Named explicitly rather than assumed because it is a real
            difference from reference MuJoCo: a scene authored with an elliptic
            cone does not run unchanged here, and results should be reported as
            pyramidal.
        solver: ``"PGS"``, ``"CG"`` or ``"Newton"``. ``None`` keeps the XML's.
        iterations: Solver iteration cap. ``None`` keeps the XML's.
        tolerance: Solver convergence tolerance. ``None`` keeps the XML's.
        nconmax: Contact buffer capacity across all worlds. ``None`` lets MJWarp
            size it. Too small silently drops contacts in a contact-rich scene.
        njmax: Constraint buffer capacity across all worlds. ``None`` lets
            MJWarp size it.
    """

    cone: str = "pyramidal"
    solver: Optional[str] = None
    iterations: Optional[int] = None
    tolerance: Optional[float] = None
    nconmax: Optional[int] = None
    njmax: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.cone != "pyramidal":
            raise ValueError(
                f"MJWarp only implements pyramidal friction cones on the GPU; got "
                f"cone={self.cone!r}. Run an elliptic-cone scene on the CPU "
                f"Mujoco simulator instead."
            )
        if self.solver is not None and self.solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {tuple(_SOLVERS)}, got {self.solver!r}")
        if self.iterations is not None and self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.tolerance is not None and self.tolerance <= 0.0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")


class VectorizedMujoco(VectorizedSimulator):
    """``N`` parallel MuJoCo worlds on the GPU, via MJWarp.

    Attributes:
        mjm: The host ``MjModel`` the worlds were compiled from. Useful for
            model constants (actuator ranges, body ids) and for handing to a
            renderer.
        m: The device-side MJWarp model, shared by all ``N`` worlds.
        d: The device-side MJWarp data. ``d.qpos``/``d.qvel``/``d.ctrl`` are
            ``(N, nq)``/``(N, nv)``/``(N, nu)`` float32 warp arrays.
    """

    def __init__(
        self,
        xml: str | Path,
        sim_config: VectorizedMujocoConfig | None = None,
        N: int = 1,
    ):
        """Compile a model, upload it, and allocate ``N`` worlds on the device.

        Args:
            xml: Path to an MJCF file, or the MJCF document itself. Prefer a
                path: MuJoCo resolves meshes and includes relative to the file.
            sim_config: Physics parameters. Defaults to
                ``VectorizedMujocoConfig()``.
            N: Number of parallel worlds.

        Raises:
            ImportError: If MJWarp is not installed.
            FileNotFoundError: If ``xml`` looks like a path and does not exist.
        """
        super().__init__(xml, sim_config if sim_config is not None else VectorizedMujocoConfig(), N)
        mjw = _mujoco_warp()
        cfg = self.config

        if self.model_path is not None:
            self.mjm = mujoco.MjModel.from_xml_path(str(self.model_path))
        else:
            self.mjm = mujoco.MjModel.from_xml_string(self.model_xml)

        self.mjm.opt.timestep = cfg.timestep
        self.mjm.opt.gravity[:] = cfg.gravity
        # Not optional: MJWarp has no elliptic cone. Written even when the XML
        # already says pyramidal, so the uploaded model cannot disagree.
        self.mjm.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        if cfg.solver is not None:
            self.mjm.opt.solver = _SOLVERS[cfg.solver]
        if cfg.iterations is not None:
            self.mjm.opt.iterations = cfg.iterations
        if cfg.tolerance is not None:
            self.mjm.opt.tolerance = cfg.tolerance

        self.m = mjw.put_model(self.mjm)
        kwargs = {}
        if cfg.nconmax is not None:
            kwargs["nconmax"] = cfg.nconmax
        if cfg.njmax is not None:
            kwargs["njmax"] = cfg.njmax
        self.d = mjw.make_data(self.mjm, nworld=self.N, **kwargs)

        # Control sequences live here, uploaded once by SetControlSequence and
        # indexed per step by _assign_ctrl_kernel. Allocated once at
        # construction because N and H are both fixed then — a planner that
        # re-plans every control step would otherwise allocate this on every
        # call.
        self._U_wp = wp.zeros(
            (self.N, self.horizon, self.mjm.nu), dtype=wp.float32, device=cfg.device
        )
        self._sequence_active = False
        # Control index currently written into d.ctrl; -1 means "none yet".
        self._applied_index = -1
        self._mjw = mjw

    # -- control sequences ---------------------------------------------------
    def SetControlSequence(self, U_n) -> None:
        """Upload an ``(N, H, nu)`` control sequence to the device.

        Overrides the base to keep the sequence on the GPU: a host array is
        validated, uploaded into the preallocated buffer, and dropped.

        A warp array of the right shape is copied device-to-device instead.
        That is the path a GPU planner takes — its sampled controls are already
        on the device, and round-tripping them through the host every plan would
        cost more than the rollout.
        """
        if isinstance(U_n, wp.array):
            want = (self.N, self.horizon, self.mjm.nu)
            if tuple(U_n.shape) != want:
                raise ValueError(
                    f"U_n warp array must have shape {want}, got {tuple(U_n.shape)}"
                )
            wp.copy(self._U_wp, U_n)
        else:
            self._U_wp.assign(self._validate_control_sequence(U_n).astype(np.float32))
        self._sequence_active = True
        self._sequence_index = 0
        self._applied_index = -1

    def ClearControlSequence(self) -> None:
        """Drop the active sequence; stepping reverts to the held ``SetControl``."""
        self._sequence_active = False
        self._sequence_index = 0
        self._applied_index = -1

    @property
    def _has_control_sequence(self) -> bool:
        return self._sequence_active

    # -- stepping ------------------------------------------------------------
    def Step_GPU(self, steps: int = 1) -> None:
        """Advance all ``N`` worlds by ``steps`` timesteps on the device.

        Nothing crosses the PCIe bus here. When a sequence is active each step's
        controls are written by a kernel; otherwise the held control stands.

        Safe to capture into a CUDA graph for a fixed number of steps: the
        sequence index is resolved in Python at capture time, so each captured
        step carries its own constant index — which is what an unrolled rollout
        wants anyway.
        """
        for _ in range(steps):
            t = self._advance_sequence()
            # Re-assign only when the control index moves on, so a control step
            # with substeps>1 costs one launch rather than one per substep. The
            # decision is made in Python, so a captured graph records exactly
            # the launches it needs and no more.
            if t is not None and t != self._applied_index:
                wp.launch(
                    _assign_ctrl_kernel,
                    dim=(self.N, self.mjm.nu),
                    inputs=[self._U_wp, t, self.d.ctrl],
                )
                self._applied_index = t
            self._mjw.step(self.m, self.d)

    # -- state ---------------------------------------------------------------
    def SetState(self, q: np.ndarray, q_dot: np.ndarray | None = None) -> None:
        """Overwrite every world's state and refresh derived quantities.

        Args:
            q: Positions, ``(N, nq)`` or ``(nq,)`` to seed all worlds alike —
                the usual case, since a plan starts every rollout from the one
                measured state.
            q_dot: Velocities, ``(N, nv)`` or ``(nv,)``. ``None`` zeroes them.

        Runs MJWarp's ``forward`` afterwards, so contacts and site poses match
        the new state, mirroring ``Mujoco.SetState``'s ``mj_forward``.
        """
        self.d.qpos.assign(self._to_worlds(q, self.nq, "q").astype(np.float32))
        qvel = (
            np.zeros((self.N, self.nv), dtype=np.float32)
            if q_dot is None
            else self._to_worlds(q_dot, self.nv, "q_dot").astype(np.float32)
        )
        self.d.qvel.assign(qvel)
        self._mjw.forward(self.m, self.d)

    def GetState(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(q, q_dot)`` for all worlds, ``(N, nq)`` and ``(N, nv)``.

        A device-to-host copy, so this is the expensive call on the rollout
        path — a planner that scores costs on the GPU should not need it.

        The arrays are float32: MJWarp is single precision throughout. They are
        deliberately not widened to float64, which would imply a precision the
        simulator does not have and hide real divergence from the CPU
        ``Mujoco`` simulator.
        """
        return self.d.qpos.numpy(), self.d.qvel.numpy()

    # -- control -------------------------------------------------------------
    def SetControl(self, u) -> None:
        """Set the held control for every world, ``(N, nu)`` or ``(nu,)``.

        A host array is validated and uploaded. An ``(N, nu)`` warp array is
        copied device-to-device instead — kernel work only, so it can sit inside
        a captured CUDA graph. That is how a planner applies commands it had to
        compute on the device mid-rollout, from the state the rollout reached.

        Does not clear an active sequence — the sequence still wins on the next
        step. Call ``ClearControlSequence`` first to hand control back to this.
        """
        if isinstance(u, wp.array):
            want = (self.N, self.nu)
            if tuple(u.shape) != want:
                raise ValueError(f"u warp array must have shape {want}, got {tuple(u.shape)}")
            wp.copy(self.d.ctrl, u)
            return
        self.d.ctrl.assign(self._to_worlds(u, self.nu, "u").astype(np.float32))

    def GetControl(self) -> np.ndarray:
        """Return the current controls for all worlds, ``(N, nu)`` float32."""
        return self.d.ctrl.numpy()

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
        """The timestep the uploaded model integrates with."""
        return float(self.mjm.opt.timestep)

    @property
    def time(self) -> np.ndarray:
        """Simulated time per world, ``(N,)``. Worlds stepped together agree."""
        return self.d.time.numpy()

    # -- device state --------------------------------------------------------
    def BroadcastState(self, q, q_dot=None, u=None) -> None:
        """Seed every world from device arrays; see ``VectorizedSimulator``."""
        wp.launch(_broadcast_kernel, dim=(self.N, self.nq),
                  inputs=[q], outputs=[self.d.qpos])
        if q_dot is None:
            self.d.qvel.zero_()
        else:
            wp.launch(_broadcast_kernel, dim=(self.N, self.nv),
                      inputs=[q_dot], outputs=[self.d.qvel])
        if u is not None:
            wp.launch(_broadcast_kernel, dim=(self.N, self.nu),
                      inputs=[u], outputs=[self.d.ctrl])

    def DeviceState(self) -> DeviceState:
        """The live MJWarp arrays, for a cost function that runs on the device."""
        return DeviceState(
            qpos=self.d.qpos, qvel=self.d.qvel, ctrl=self.d.ctrl,
            site_xpos=self.d.site_xpos,
        )

    # -- lifecycle -----------------------------------------------------------
    def Close(self) -> None:
        """Drop the device allocations.

        Warp frees device memory when the last reference goes, so this releases
        the model, the per-world data and the control buffer rather than
        waiting for the simulator itself to be collected.
        """
        self.d = None
        self.m = None
        self._U_wp = None


def _mujoco_warp():
    """Import MJWarp, with a message that names the package if it is missing.

    Imported lazily and through comfree_warp's vendored copy, which is what the
    study installs — there is no standalone ``mujoco_warp`` distribution pinned
    in ``pyproject.toml``.
    """
    try:
        import comfree_warp.mujoco_warp as mjw
    except ImportError as exc:
        raise ImportError(
            "VectorizedMujoco needs MJWarp, provided by the comfree_warp package "
            "(see pyproject.toml). Install it, or use the CPU Mujoco simulator."
        ) from exc
    return mjw
