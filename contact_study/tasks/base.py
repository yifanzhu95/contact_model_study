"""Base task interface and task registry.

Each task defines:
  - The XML scene (geometry-variant-aware)
  - An initial state distribution
  - A cost/reward function for the MPC planner
  - A success criterion for evaluation
  - Contact complexity metadata for the taxonomy
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from contact_study.contact_models.config import GeometryVariant

SCENES_DIR = Path(__file__).parents[3] / "scenes"


class ContactComplexity(enum.IntEnum):
    """Qualitative contact complexity level, used to sort tasks in results."""
    LOW    = 1   # pushing: ≤2 contacts, quasi-static
    MEDIUM = 2   # grasp-reorient: ~4 contacts, dynamic
    HIGH   = 3   # peg-in-hole assembly: tight clearance, multi-contact


@dataclass
class TaskSpec:
    name:              str
    complexity:        ContactComplexity
    xml_path_template: str    # format string with {geometry} placeholder
    max_steps:         int
    success_threshold: float  # task-specific (e.g., position error in meters)


class BaseTask(abc.ABC):
    """Abstract base class for all manipulation tasks."""

    def __init__(self, geometry: GeometryVariant = GeometryVariant.ACCURATE):
        self.geometry = geometry
        self._mjm: mujoco.MjModel | None = None
        self._mjd: mujoco.MjData  | None = None

    @property
    @abc.abstractmethod
    def spec(self) -> TaskSpec: ...

    def load(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Load the MuJoCo model for this task and geometry variant."""
        xml_path = self.spec.xml_path_template.format(geometry=self.geometry.value)
        full_path = SCENES_DIR / xml_path
        self._mjm = mujoco.MjModel.from_xml_path(str(full_path))
        self._mjd = mujoco.MjData(self._mjm)
        return self._mjm, self._mjd

    @property
    def mjm(self) -> mujoco.MjModel:
        assert self._mjm is not None, "Call load() first."
        return self._mjm

    @property
    def mjd(self) -> mujoco.MjData:
        assert self._mjd is not None, "Call load() first."
        return self._mjd

    @abc.abstractmethod
    def sample_initial_state(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Return (qpos, qvel) for a random initial state."""
        ...

    @abc.abstractmethod
    def cost_fn(
        self,
        qpos,     # Warp array (nworld, nq)
        qvel,     # Warp array (nworld, nv)
        ctrl,     # Warp array (nworld, nu)
        terminal: bool,
    ) -> np.ndarray:
        """Return (nworld,) cost array. Called inside the MPC rollout."""
        ...

    @abc.abstractmethod
    def is_success(self, mjd: mujoco.MjData) -> bool:
        """Check whether the current state satisfies the task goal."""
        ...

    def evaluate_episode(
        self,
        mjm: mujoco.MjModel,
        plan_fn: Callable[[mujoco.MjData], np.ndarray],
        max_steps: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Run one closed-loop episode and return result dict.

        Args:
            mjm:      Host model.
            plan_fn:  Callable(mjd) -> ctrl array.
            max_steps: Override task max_steps.
            rng:      RNG for initial state sampling.

        Returns:
            dict with keys: success (bool), steps_to_success (int | None),
            final_cost (float), trajectory (list of qpos copies).
        """
        rng = rng or np.random.default_rng()
        mjd = mujoco.MjData(mjm)
        T   = max_steps or self.spec.max_steps

        q0, v0 = self.sample_initial_state(rng)
        mjd.qpos[:] = q0
        mjd.qvel[:] = v0
        mujoco.mj_forward(mjm, mjd)

        trajectory = []
        steps_to_success = None

        for t in range(T):
            ctrl = plan_fn(mjd)
            mjd.ctrl[:] = ctrl
            mujoco.mj_step(mjm, mjd)
            trajectory.append(mjd.qpos.copy())

            if self.is_success(mjd) and steps_to_success is None:
                steps_to_success = t + 1

        return {
            "success":          steps_to_success is not None,
            "steps_to_success": steps_to_success,
            "final_cost":       float(np.linalg.norm(mjd.qpos - q0)),
            "trajectory":       trajectory,
        }


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseTask]] = {}

def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def get_task(name: str, geometry: GeometryVariant = GeometryVariant.ACCURATE) -> BaseTask:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](geometry=geometry)

def list_tasks() -> list[str]:
    return list(_REGISTRY.keys())
