"""What an episode driver needs from a planner, whatever produces the command.

``run_episodes`` hands a planner the measured state each control step and
applies the absolute actuator command it returns. A sampling-based planner
rolls candidate futures out on a vectorized simulator to choose that command; a
learned policy maps the state to it directly. Both are planners here, so one
driver, one recorder and one eval simulator score them alike.
"""

from __future__ import annotations

import abc

import numpy as np


class PlannerBase(abc.ABC):
    """Base class for everything ``run_episodes`` can run.

    Attributes:
        task: The task the planner works towards; its ``timestep`` times
            ``substeps`` is one control step.
        substeps: Simulator steps of ``task.timestep`` per control step.
        config: The planner's configuration, recorded with each episode.
        sim: The simulator the planner predicts with, or ``None`` for one that
            predicts nothing.
        last_plan_ok: Whether the last ``Plan`` produced a usable command.
    """

    name = "Planner"

    task = None
    substeps: int = 1
    config = None
    sim = None
    last_plan_ok: bool = False

    @abc.abstractmethod
    def Plan(self, q: np.ndarray, q_dot: np.ndarray,
             u: np.ndarray | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return the next absolute actuator command, ``(nu,)``, for the measured state.

        Args:
            q: Measured positions, ``(nq,)``.
            q_dot: Measured velocities, ``(nv,)``.
            u: The command currently applied, ``(nu,)``.

        Returns:
            The command, or ``(command, uncertainty)`` for a planner configured
            to report one.
        """

    @abc.abstractmethod
    def Reset(self, mean: np.ndarray | None = None) -> None:
        """Forget the previous episode or goal; called before an episode's first ``Plan``."""

    def Close(self) -> None:
        """Release anything the planner holds; nothing by default."""
