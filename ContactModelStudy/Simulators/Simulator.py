"""Base simulator interface for the contact-model study.

Every simulator in this package (MuJoCo, MuJoCo Warp, ComFree, XPBD, Pinocchio,
Drake) is a thin wrapper that exposes the same six operations: set/get state,
set/get control, and step. Planners, drivers and renderers are written against
this interface alone, so swapping the contact model under an experiment is a
one-line change.

Two contracts make that work across backends whose internal joint orderings
differ:

  * ``GetState`` returns ``(q, q_dot)`` in the **MuJoCo** qpos/qvel index
    layout, whatever the backend stores internally.
  * ``SetControl`` takes a **MuJoCo-ordered, absolute** actuator command. Any
    delta/rate accumulation is the caller's business.

Each subclass remaps internally; callers never see backend-specific indices.
"""

from __future__ import annotations

import abc
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Recognized as an XML document rather than a path when a model source string
# starts with one of these (after stripping leading whitespace).
_XML_PREFIXES = ("<?xml", "<mujoco", "<robot", "<sdf", "<!--")

# Slack when a duration is floored to whole steps, so a duration that is an
# exact multiple of the step in decimal (0.032 s of 4 ms steps) is not lost to
# binary rounding (0.032 / 0.004 = 7.999999999999999).
_RATIO_TOL = 1e-9


def _floorSteps(duration: float, unit: float, name: str, warn: bool) -> int:
    """Whole ``unit`` steps in ``duration``, rounded down, and at least 1.

    Rounding down keeps the realized duration at or under the requested one,
    as the old planner's ``resolve_schedule`` did. A duration shorter than one
    step is clamped to one step, with a warning when ``warn`` is set.
    """
    n = int(math.floor(duration / unit + _RATIO_TOL))
    if n < 1:
        if warn:
            warnings.warn(f"{name}={duration:g}s is shorter than one step of {unit:g}s; "
                          f"using 1 step ({unit:g}s).", stacklevel=4)
        return 1
    return n


def _exclusive(cfg, a: str, b: str) -> None:
    """Raise unless at most one of the two fields ``a`` and ``b`` is set."""
    if getattr(cfg, a) is not None and getattr(cfg, b) is not None:
        raise ValueError(f"set {a} or {b}, not both (got {a}={getattr(cfg, a)!r}, "
                         f"{b}={getattr(cfg, b)!r})")


@dataclass
class SimulatorConfig:
    """Physics parameters shared by every simulator backend.

    Deliberately minimal: only the parameters every backend can honor. Anything
    specific to one backend's integrator or contact solver (MuJoCo's cone and
    solver, ComFree's stiffness, XPBD's iteration count) belongs on that
    backend's own config, so this object stays reusable across M1-M4.

    The control step can be given two ways: as a step count (``substeps``) or
    as a duration (``ctrl_time_step``). Set at most one; setting both raises.
    Setting neither means one step per control step. Both fields keep what was
    asked for; ``resolved_substeps`` is the count actually used.

    Attributes:
        timestep: Integration timestep in seconds. The *fine* step; one call to
            ``Step()`` advances exactly this much.
        substeps: Fine steps per control step. Only drivers and planners use
            this, to turn a control period into a ``Step`` count; ``Step``
            itself is unaware of it.
        ctrl_time_step: Control-step duration in seconds, the alternative to
            ``substeps``. Rounded *down* to whole timesteps, so the realized
            control step never exceeds it, and at least one timestep.
        gravity: Gravity vector in world coordinates (m/s^2).
    """

    timestep: float = 0.002
    substeps: Optional[int] = None
    ctrl_time_step: Optional[float] = None
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    def __post_init__(self) -> None:
        if self.timestep <= 0.0:
            raise ValueError(f"timestep must be positive, got {self.timestep}")
        _exclusive(self, "substeps", "ctrl_time_step")
        if self.substeps is not None and self.substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {self.substeps}")
        if self.ctrl_time_step is not None and self.ctrl_time_step <= 0.0:
            raise ValueError(f"ctrl_time_step must be positive, got {self.ctrl_time_step}")
        if len(self.gravity) != 3:
            raise ValueError(f"gravity must have 3 components, got {len(self.gravity)}")
        self.gravity = tuple(float(g) for g in self.gravity)
        self._resolveSubsteps(warn=True)        # warns once, here, if it clamps

    def _resolveSubsteps(self, warn: bool = False) -> int:
        if self.ctrl_time_step is not None:
            return _floorSteps(self.ctrl_time_step, self.timestep, "ctrl_time_step", warn)
        return 1 if self.substeps is None else int(self.substeps)

    @property
    def resolved_substeps(self) -> int:
        """Fine steps per control step: ``substeps``, or ``ctrl_time_step`` in steps."""
        return self._resolveSubsteps()

    @property
    def control_timestep(self) -> float:
        """Seconds of simulated time in one control step (``timestep * resolved_substeps``)."""
        return self.timestep * self.resolved_substeps


class Simulator(abc.ABC):
    """Base class for all (non-vectorized) simulators in the study.

    Subclasses implement the abstract members below and are expected to call
    ``super().__init__(xml, sim_config)`` first, which resolves the model source
    and stashes the config on ``self.config``.

    Subclasses must NOT re-read ``xml``; use ``self.model_xml`` (the XML text)
    or ``self.model_path`` (the file it came from, ``None`` for inline XML)
    instead, so a model is loaded from exactly one place.
    """

    def __init__(self, xml: str | Path, sim_config: SimulatorConfig | None = None):
        """Load a model and prepare the backend.

        Args:
            xml: Either a path to an MJCF/URDF file or the XML document itself
                as a string. Which one it is is detected from the content, not
                from an extension, so in-memory scenes need no temp file.
            sim_config: Physics parameters. Defaults to ``SimulatorConfig()``.

        Raises:
            FileNotFoundError: If ``xml`` looks like a path and does not exist.
        """
        self.config = sim_config if sim_config is not None else SimulatorConfig()
        self.model_path, self.model_xml = self._resolve_model_source(xml)

    # -- model source --------------------------------------------------------
    @staticmethod
    def _resolve_model_source(xml: str | Path) -> tuple[Path | None, str]:
        """Normalize a model source into ``(path_or_None, xml_text)``.

        A ``Path``, or a string whose first non-whitespace characters are not an
        XML opening tag, is treated as a filename; anything else is treated as
        an inline XML document. Inline XML that came from a file still reports
        that file in ``model_path`` so backends that can only load from disk
        (Drake's parser, some URDF paths) have something to hand them, and so
        relative mesh references resolve against the right directory.
        """
        if isinstance(xml, Path) or not xml.lstrip().startswith(_XML_PREFIXES):
            path = Path(xml).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Model file not found: {path}")
            return path, path.read_text()
        return None, str(xml)

    # -- state ---------------------------------------------------------------
    @abc.abstractmethod
    def SetState(self, q: np.ndarray, q_dot: np.ndarray | None = None) -> None:
        """Overwrite the current state.

        Args:
            q: Generalized positions, shape ``(nq,)``, in the MuJoCo qpos layout.
            q_dot: Generalized velocities, shape ``(nv,)``. ``None`` zeroes them,
                which is the common case when resetting an episode.

        Implementations are expected to bring derived quantities (forward
        kinematics, contact sets) up to date so that a ``GetState`` immediately
        afterwards, or a renderer reading the backend, sees a consistent world.
        """
        ...

    @abc.abstractmethod
    def GetState(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(q, q_dot)`` as copies, in the MuJoCo qpos/qvel layout.

        Copies, not views: callers store these in rollout buffers, and a view
        into live backend memory would silently mutate under them on the next
        ``Step``.
        """
        ...

    # -- control -------------------------------------------------------------
    @abc.abstractmethod
    def SetControl(self, u: np.ndarray) -> None:
        """Set the current control.

        Args:
            u: Absolute actuator command, shape ``(nu,)``, in the MuJoCo
                actuator order and MuJoCo semantics (position targets for
                position actuators, torques for motors).
        """
        ...

    @abc.abstractmethod
    def GetControl(self) -> np.ndarray:
        """Return the current control as a copy, shape ``(nu,)``."""
        ...

    # -- stepping ------------------------------------------------------------
    @abc.abstractmethod
    def Step(self, steps: int = 1) -> None:
        """Advance the simulation by ``steps`` timesteps of ``config.timestep``.

        The control set by ``SetControl`` is held constant across all of them —
        this is a zero-order hold, not a re-sample per step.
        """
        ...

    # -- dimensions ----------------------------------------------------------
    @property
    @abc.abstractmethod
    def nq(self) -> int:
        """Number of generalized position coordinates."""
        ...

    @property
    @abc.abstractmethod
    def nv(self) -> int:
        """Number of generalized velocity coordinates (degrees of freedom)."""
        ...

    @property
    @abc.abstractmethod
    def nu(self) -> int:
        """Number of actuators."""
        ...

    @property
    def timestep(self) -> float:
        """Integration timestep in seconds.

        Reads back from the config. A backend that cannot honor the requested
        timestep exactly (a fixed-step external integrator, say) overrides this
        to report what it actually runs at, since drivers convert wall-clock
        episode lengths into step counts with it.
        """
        return self.config.timestep

    # -- lifecycle -----------------------------------------------------------
    def Close(self) -> None:
        """Release backend resources. No-op unless a subclass holds any.

        Defined here so drivers can unconditionally close whatever simulator
        they were handed; GPU-backed subclasses free device memory in it.
        """
        return None

    def __enter__(self) -> "Simulator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.Close()

    def __repr__(self) -> str:
        source = self.model_path.name if self.model_path is not None else "<inline XML>"
        return f"{type(self).__name__}(model={source!r}, dt={self.timestep})"
