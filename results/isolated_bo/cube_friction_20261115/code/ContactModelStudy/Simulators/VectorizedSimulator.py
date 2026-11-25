"""Base interface for vectorized (parallel, GPU-backed) simulators.

A ``VectorizedSimulator`` holds ``N`` independent copies — *worlds* — of one
model and advances all of them in lockstep on the GPU. This is what sampling-
based planners roll out against: one world per sampled control sequence.

It is a ``Simulator``, so everything in that interface still applies, with one
systematic change: every per-world quantity grows a leading world axis.

    ============  ==================  ==========================
    Quantity      ``Simulator``       ``VectorizedSimulator``
    ============  ==================  ==========================
    ``q``         ``(nq,)``           ``(N, nq)``
    ``q_dot``     ``(nv,)``           ``(N, nv)``
    ``u``         ``(nu,)``           ``(N, nu)``
    ============  ==================  ==========================

Setters also accept the un-batched shape and broadcast it to all worlds, which
is the normal way an episode starts: every world is seeded with the same
measured state, then driven apart by different controls.

Rollouts stay on the device. ``Step_GPU`` never copies state back to the host;
``GetState`` is the explicit (and expensive) transfer, so a planner that scores
rollouts with an on-device cost function never pays for one.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from typing import Optional

from ContactModelStudy.Simulators.Simulator import (
    Simulator,
    SimulatorConfig,
    _exclusive,
    _floorSteps,
)


@dataclass
class VectorizedSimulatorConfig(SimulatorConfig):
    """Physics parameters for a vectorized simulator.

    Inherits ``timestep``, the control step (``substeps`` or
    ``ctrl_time_step``) and ``gravity`` from ``SimulatorConfig`` and adds the
    two things a parallel backend needs that a single-world one does not: where
    the worlds live, and how long the control sequences driving them are.

    The horizon can be given two ways, like the control step: as a count of
    control steps (``horizon``) or as a duration (``time_horizon``). Set at
    most one; setting both raises. Setting neither means a horizon of 1.
    ``resolved_horizon`` is the count actually used.

    The world count ``N`` is deliberately *not* here — it is a ``__init__``
    argument, because it determines how much device memory gets allocated and
    so belongs at construction rather than in a config that gets copied around.

    Attributes:
        horizon: Length ``H``, in control steps, of the control sequences passed
            to ``SetControlSequence``. Fixed at construction so device buffers
            are allocated once; a sequence of any other length is rejected.
        time_horizon: Planning horizon in seconds, the alternative to
            ``horizon``. Rounded *down* to whole control steps (of
            ``control_timestep``, so after the control step is resolved), and
            at least one.
        device: Compute device for the worlds, e.g. ``"cuda"`` or ``"cpu"``.
    """

    horizon: Optional[int] = None
    time_horizon: Optional[float] = None
    device: str = "cuda"

    def __post_init__(self) -> None:
        super().__post_init__()
        _exclusive(self, "horizon", "time_horizon")
        if self.horizon is not None and self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.time_horizon is not None and self.time_horizon <= 0.0:
            raise ValueError(f"time_horizon must be positive, got {self.time_horizon}")
        self._resolveHorizon(warn=True)

    def _resolveHorizon(self, warn: bool = False) -> int:
        if self.time_horizon is not None:
            return _floorSteps(self.time_horizon, self.control_timestep, "time_horizon", warn)
        return 1 if self.horizon is None else int(self.horizon)

    @property
    def resolved_horizon(self) -> int:
        """Control steps in a plan: ``horizon``, or ``time_horizon`` in control steps."""
        return self._resolveHorizon()

    @property
    def horizon_duration(self) -> float:
        """Seconds of simulated time a plan covers (``resolved_horizon * control_timestep``)."""
        return self.resolved_horizon * self.control_timestep


@dataclass
class DeviceState:
    """A vectorized simulator's live device arrays.

    Backend-typed (warp arrays for the MJWarp backends), deliberately not
    converted to numpy: the whole point is to hand a device-side cost function
    something it can read without a transfer.

    Attributes:
        qpos: ``(N, nq)`` positions.
        qvel: ``(N, nv)`` velocities.
        ctrl: ``(N, nu)`` current controls.
        site_xpos: ``(N, nsite)`` site world positions, or ``None`` for a
            backend with no sites. Cost functions with a contact or end-effector
            term need these, and they cannot be derived from ``qpos`` without
            forward kinematics.
    """

    qpos: object
    qvel: object
    ctrl: object
    site_xpos: object | None = None


class VectorizedSimulator(Simulator):
    """Base class for all parallel simulators in the study.

    Subclasses implement the same abstract members as ``Simulator`` — with the
    batched shapes above — plus ``Step_GPU``. They are expected to call
    ``super().__init__(xml, sim_config, N)`` first.

    Note on the constructor signature: the refactor plan writes
    ``init(time_step, sim_config, N)``. The timestep already lives on
    ``sim_config``, so this takes ``(xml, sim_config, N)`` instead, which keeps
    the model source in the same position as ``Simulator.__init__``.
    """

    def __init__(
        self,
        xml: str | Path,
        sim_config: VectorizedSimulatorConfig | None = None,
        N: int = 1,
    ):
        """Load a model and instantiate ``N`` copies of it on the device.

        Args:
            xml: Path to an MJCF/URDF file, or the XML document itself. Parsed
                once; the ``N`` worlds share a single model and differ only in
                their state and controls.
            sim_config: Physics parameters. Defaults to
                ``VectorizedSimulatorConfig()``.
            N: Number of parallel worlds. Must be >= 1.

        Raises:
            ValueError: If ``N < 1``.
            FileNotFoundError: If ``xml`` looks like a path and does not exist.
        """
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        super().__init__(xml, sim_config if sim_config is not None else VectorizedSimulatorConfig())
        self._n_worlds = int(N)
        # Set by SetControlSequence; None means "no sequence active", in which
        # case stepping holds whatever SetControl last wrote.
        self._control_sequence: np.ndarray | None = None
        self._sequence_index: int = 0

    # -- shape ---------------------------------------------------------------
    @property
    def n_worlds(self) -> int:
        """Number of parallel worlds ``N``."""
        return self._n_worlds

    @property
    def N(self) -> int:
        """Alias for ``n_worlds``, matching the refactor plan's naming."""
        return self._n_worlds

    @property
    def horizon(self) -> int:
        """Control-sequence length ``H``, resolved from the config."""
        return self.config.resolved_horizon

    def _to_worlds(self, x: np.ndarray, dim: int, name: str) -> np.ndarray:
        """Coerce ``x`` to ``(N, dim)``, broadcasting an un-batched input.

        Accepts ``(dim,)`` — the same value for every world — or ``(N, dim)``.
        Anything else is a shape bug in the caller and raises rather than
        silently broadcasting into the wrong axis, which is a mistake that
        otherwise surfaces much later as a rollout that quietly did nothing.
        """
        a = np.asarray(x, dtype=float)
        if a.shape == (dim,):
            return np.broadcast_to(a, (self._n_worlds, dim)).copy()
        if a.shape == (self._n_worlds, dim):
            return a
        raise ValueError(
            f"{name} must have shape ({dim},) or ({self._n_worlds}, {dim}), got {a.shape}"
        )

    # -- control sequences ---------------------------------------------------
    def _validate_control_sequence(self, U_n: np.ndarray) -> np.ndarray:
        """Coerce a control sequence to ``(N, H, nu)``, broadcasting ``(H, nu)``.

        Separate from ``SetControlSequence`` so a backend that stores the
        sequence on the device can reuse the shape checking without also
        inheriting a host-side copy it would never read.
        """
        a = np.asarray(U_n, dtype=float)
        H, nu = self.horizon, self.nu
        if a.shape == (H, nu):
            return np.broadcast_to(a, (self._n_worlds, H, nu)).copy()
        if a.shape != (self._n_worlds, H, nu):
            raise ValueError(
                f"U_n must have shape ({self._n_worlds}, {H}, {nu}) or ({H}, {nu}), "
                f"got {a.shape}"
            )
        return a

    def SetControlSequence(self, U_n: np.ndarray) -> None:
        """Set the ``H``-step control sequence driving each world.

        Args:
            U_n: Controls of shape ``(N, H, nu)`` — one sequence per world, the
                layout sampling planners already produce — or ``(H, nu)`` to
                drive every world with the same sequence.

        The sequence cursor resets to 0, so the next ``H`` steps replay it from
        the start. Stepping past the end holds the final control (zero-order
        hold) rather than wrapping or erroring: a rollout that overruns its
        horizon should coast, not silently restart its plan.

        A GPU backend overrides this to upload the sequence to the device once,
        and then indexes it with ``_advance_sequence`` rather than reading the
        host array back every step.

        Raises:
            ValueError: If the shape or horizon does not match.
        """
        self._control_sequence = self._validate_control_sequence(U_n)
        self._sequence_index = 0

    def ClearControlSequence(self) -> None:
        """Drop the active sequence; stepping reverts to the held ``SetControl``."""
        self._control_sequence = None
        self._sequence_index = 0

    def _advance_sequence(self) -> int | None:
        """Return this step's index into the sequence and advance the cursor.

        ``None`` means no sequence is active and the held control stands. The
        index is clamped to ``H - 1``, so stepping past the horizon holds the
        last control.

        This is the part a device-resident backend needs: an index it can pass
        to a kernel, with no host array involved. The cursor advances purely
        from Python state, so a fixed-length rollout still captures into a CUDA
        graph — each step bakes in its own constant index, which is exactly what
        an unrolled rollout wants.
        """
        if not self._has_control_sequence:
            return None
        # The cursor counts PHYSICS steps; the sequence is indexed by CONTROL
        # steps, so each entry is held for `substeps` of them. That is what
        # makes a control sequence a zero-order hold at the control rate rather
        # than a new command every integration step.
        i = min(self._sequence_index // self.config.resolved_substeps, self.horizon - 1)
        self._sequence_index += 1
        return i

    @property
    def _has_control_sequence(self) -> bool:
        """Whether a sequence is active. Overridden when it lives on the device."""
        return self._control_sequence is not None

    def _next_control(self) -> np.ndarray | None:
        """Return this step's ``(N, nu)`` controls and advance the cursor.

        The host-side convenience: a backend that steps on the CPU calls this
        once per step from inside ``Step_GPU``. Returns ``None`` when no
        sequence is active, meaning the held control stands.
        """
        i = self._advance_sequence()
        if i is None or self._control_sequence is None:
            return None
        return self._control_sequence[:, i, :]

    # -- stepping ------------------------------------------------------------
    @abc.abstractmethod
    def Step_GPU(self, steps: int = 1) -> None:
        """Advance all ``N`` worlds by ``steps`` timesteps on the device.

        No host transfer happens here — neither of state nor of controls. When a
        control sequence is active the implementation pulls each step's controls
        from ``_next_control``; otherwise the control held by ``SetControl``
        applies to every step.
        """
        ...

    def Step(self, steps: int = 1) -> None:
        """Advance all worlds by ``steps`` timesteps.

        A vectorized simulator has no separate host-side integrator, so this is
        ``Step_GPU``. It exists so code written against the ``Simulator``
        interface — a driver, a renderer — runs unchanged on a vectorized
        backend.
        """
        self.Step_GPU(steps)

    # -- batched state and control ------------------------------------------
    @abc.abstractmethod
    def SetState(self, q: np.ndarray, q_dot: np.ndarray | None = None) -> None:
        """Overwrite the state of every world.

        Args:
            q: Positions, ``(N, nq)``, or ``(nq,)`` to seed all worlds alike.
            q_dot: Velocities, ``(N, nv)`` or ``(nv,)``. ``None`` zeroes them.

        Implementations pass both through ``_to_worlds`` and are expected to
        refresh derived quantities, as in ``Simulator.SetState``.
        """
        ...

    @abc.abstractmethod
    def GetState(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(q, q_dot)`` for all worlds, shapes ``(N, nq)``/``(N, nv)``.

        This is a device-to-host copy, so it is the expensive call on the
        rollout path. Planners that score on the GPU should not need it.
        """
        ...

    @abc.abstractmethod
    def SetControl(self, u: np.ndarray) -> None:
        """Set the held control for every world.

        Args:
            u: Absolute actuator commands, ``(N, nu)``, or ``(nu,)`` to apply
                the same command everywhere. A backend may also accept a device
                array of shape ``(N, nu)``, copied without a host round-trip.

        This is the single-control path; for a rollout use
        ``SetControlSequence``. Setting a control does not clear an active
        sequence — the sequence still wins on the next step.
        """
        ...

    @abc.abstractmethod
    def GetControl(self) -> np.ndarray:
        """Return the current controls for all worlds as a copy, ``(N, nu)``."""
        ...

    # -- device state --------------------------------------------------------
    def BroadcastState(self, q, q_dot=None, u=None) -> None:
        """Seed every world from device-resident ``(nq,)``/``(nv,)``/``(nu,)`` arrays.

        The capturable twin of ``SetState``. ``SetState`` takes host arrays, so
        it necessarily contains a host-to-device copy, and a CUDA graph cannot
        record one — which is what stops a planner capturing its whole rollout.
        This does the same job with nothing but kernel launches, reading a start
        state the caller has already staged on the device.

        Args:
            q: ``(nq,)`` device array of positions, broadcast to all worlds.
            q_dot: ``(nv,)`` device array of velocities. ``None`` zeroes them.
            u: ``(nu,)`` device array of controls, or ``None`` to leave the
                current controls alone. Part of a task's initial state on a
                position-actuated system, where ``ctrl = 0`` is not "at rest"
                but "drive every joint to zero".

        Deliberately does *not* run ``forward``: the first ``Step_GPU`` that
        follows recomputes everything derived anyway, and skipping it keeps the
        reset to two kernel launches.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement BroadcastState(); it is "
            f"needed to capture a rollout into a CUDA graph."
        )

    def DeviceState(self) -> "DeviceState":
        """Return the backend's live device arrays, without copying to the host.

        This is what a planner needs: ``GetState`` crosses PCIe, and a rollout
        that pays that every step is not a GPU rollout. A cost function running on
        the device reads these directly.

        The arrays are the simulator's own live buffers — they change under the
        caller on the next ``Step_GPU``, and must not be written to.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose its device state. Implement "
            f"DeviceState() to use it with a GPU planner."
        )

    def __repr__(self) -> str:
        source = self.model_path.name if self.model_path is not None else "<inline XML>"
        return (
            f"{type(self).__name__}(model={source!r}, N={self._n_worlds}, "
            f"H={self.horizon}, dt={self.timestep}, device={self.config.device!r})"
        )
