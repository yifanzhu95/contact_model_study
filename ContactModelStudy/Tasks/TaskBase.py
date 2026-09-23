"""Base class for every task in the study.

A task is the problem statement: which scene to load, where the system starts,
and what a trajectory through it costs. It owns no simulator and no renderer —
those are handed the task's model path and its states — so the same task object
can score a CPU rollout, a GPU batch, or a recorded episode.

Costs come in two flavours with deliberately identical semantics:

  * ``calcCosts`` on the host, for analysis, evaluation and tests;
  * ``calcCosts_GPU`` on the device, for the rollout inner loop.

A task is responsible for the two agreeing. They are tested against each other
per task, because a planner optimizing one while the study reports the other is
the kind of bug that produces a plausible, wrong paper.
"""

from __future__ import annotations

import abc
import enum
from pathlib import Path

import numpy as np


class TaskRole(str, enum.Enum):
    """Which scene a task should load.

    The study runs two models of the same physical setup: a high-fidelity
    ``EVAL`` scene standing in for reality, and a cheaper ``ROLLOUT`` scene the
    planner predicts with. The gap between them is the thing being measured, so
    which one a task hands out is an explicit choice, never a default that
    quietly picks the wrong one.
    """

    EVAL = "eval"
    ROLLOUT = "rollout"


class TaskBase(abc.ABC):
    """Base class for all tasks.

    Subclasses implement ``getModelPath``, ``getInitialState`` and the two cost
    functions.
    """

    def __init__(self, role: TaskRole | str = TaskRole.EVAL):
        """Create a task bound to one scene role.

        Args:
            role: ``TaskRole.EVAL`` or ``TaskRole.ROLLOUT``, or the equivalent
                string. Decides which scene ``getModelPath`` returns.
        """
        self.role = TaskRole(role)

    # -- scene ---------------------------------------------------------------
    @abc.abstractmethod
    def getModelPath(self) -> str:
        """Return the path to this task's MJCF model, for its role.

        A string rather than a ``Path``, per the plan, and because this is what
        gets handed to ``mujoco.MjModel.from_xml_path`` and to renderers.
        """
        ...

    # -- initial conditions --------------------------------------------------
    @abc.abstractmethod
    def getInitialState(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(q0, q_dot0, u0)`` — where an episode starts.

        Not in the refactor plan's function list, but every driver needs it and
        every task has it, so it belongs on the interface rather than being
        reached for through a subclass-specific attribute. The control is part
        of the initial condition because a position-actuated system left at
        ``ctrl = 0`` collapses before the planner's first command lands.
        """
        ...

    # -- costs ---------------------------------------------------------------
    @abc.abstractmethod
    def calcCosts(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        u: np.ndarray | None = None,
        terminal: bool = False,
    ) -> np.ndarray:
        """Cost of states ``(q, q_dot)`` under controls ``u``, on the host.

        Args:
            q: Positions, ``(..., nq)``. Leading axes are free: ``(nq,)`` for a
                single state, ``(T, nq)`` for a trajectory, ``(N, T, nq)`` for a
                batch of them.
            q_dot: Velocities, ``(..., nv)``, with leading axes matching ``q``.
            u: Controls, ``(..., nu)``. Optional because not every task's cost
                depends on the command.
            terminal: Score with the terminal cost rather than the running one.
                Terminal cost *replaces* the running cost; it is not added.

        Returns:
            Costs shaped like the leading axes of ``q`` — a scalar array for a
            single state, ``(T,)`` for a trajectory, ``(N, T)`` for a batch.
        """
        ...

    @abc.abstractmethod
    def calcCosts_GPU(self, q, q_dot, u=None, terminal: bool = False, out=None):
        """Cost of a batch of states on the device.

        Args:
            q: ``(N, nq)`` warp array of positions, one row per world.
            q_dot: ``(N, nv)`` warp array of velocities.
            u: ``(N, nu)`` warp array of controls, or ``None``.
            terminal: As ``calcCosts``.
            out: ``(N,)`` warp array to write into. ``None`` allocates one;
                pass a preallocated buffer on the rollout path, where allocating
                per step would show up in the profile.

        Returns:
            A ``(N,)`` warp array of costs.

        Must agree with ``calcCosts`` to float32 precision on the same inputs.
        """
        ...

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _resolve_path(path: str | Path) -> str:
        """Check a scene file exists and return it as a string.

        Failing here names the missing scene; failing inside MuJoCo's parser
        reports a less obvious error further from the cause.
        """
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Task scene not found: {p}")
        return str(p)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(role={self.role.value!r})"
