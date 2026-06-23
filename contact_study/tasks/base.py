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
from pathlib import Path
from typing import Callable, Any

import mujoco
import numpy as np
import warp as wp

from contact_study.contact_models.config import GeometryVariant
# ContactComplexity / TaskSpec now live in config.py (to avoid a base<->config
# import cycle) and are re-exported here for backward compatibility.
from contact_study.tasks.config import (
    ContactComplexity,
    TaskSpec,
    TaskConfig,
    TaskRole,
    EvalSimulatorKind,
)

SCENES_DIR = Path(__file__).parents[2] / "scenes"


class BaseTask(abc.ABC):
    """Abstract base class for all manipulation tasks."""

    def __init__(
        self,
        geometry: GeometryVariant = GeometryVariant.ACCURATE,
        role: TaskRole = TaskRole.ROLLOUT,
    ):
        self.geometry = geometry
        self.role = role
        self._mjm: mujoco.MjModel | None = None
        self._mjd: mujoco.MjData  | None = None

        # Editable per-task configuration. Tasks migrated to TaskConfig set this
        # in their constructor; legacy tasks (push, peg_in_hole) leave it None
        # and instead override the `spec` property to return a TaskSpec.
        self.config: TaskConfig | None = None

        # Task-specific metadata extracted at load time
        self.goal_vector: np.ndarray | None = None
        self.index_vector: np.ndarray | None = None
        self.goal_vector_wp: wp.array | None = None
        self.index_vector_wp: wp.array | None = None
        self.weights_wp: wp.array | None = None

    @property
    def spec(self):
        """Task spec/config. TaskConfig is a superset of TaskSpec and exposes
        the same attribute names, so `task.spec.X` works for either. Legacy
        tasks override this property to return a TaskSpec instead."""
        if self.config is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set self.config or override spec."
            )
        return self.config

    def load(self, full_path: str | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Load the MuJoCo model for this task and geometry variant."""
        if full_path is None:
            xml_path = self.spec.xml_path_template.format(geometry=self.geometry.value)
            full_path = SCENES_DIR / xml_path
        self._mjm = mujoco.MjModel.from_xml_path(str(full_path))
        self._mjd = mujoco.MjData(self._mjm)

        # Post-load initialization to extract task-specific indices/goals
        self.initialize_task()

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
    def initialize_task(self):
        """Extract task-specific info (joint indices, goal poses) from MjModel.
        
        Should populate self.goal_vector and self.index_vector as both
        NumPy and Warp arrays.
        """
        ...

    @abc.abstractmethod
    def get_inital_state(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return (qpos, qvel, ctrl) for a random initial state."""
        ...

    @property
    @abc.abstractmethod
    def cost_fn_wp(self) -> wp.func:
        """Return the Warp cost function for this task."""
        ...

    @property
    def cost_goal_wp(self) -> wp.array:
        """Return the goal array for the cost function."""
        return self.goal_vector_wp

    @property
    def cost_idx_wp(self) -> wp.array:
        """Return the index array for the cost function."""
        return self.index_vector_wp

    @property
    def cost_weights_wp(self) -> wp.array:
        """Return the weights array for the cost function."""
        return self.weights_wp

    @abc.abstractmethod
    def is_success(self, mjd: mujoco.MjData) -> bool:
        """Check whether the current state satisfies the task goal."""
        ...

    def has_failed(self, mjd: mujoco.MjData) -> bool:
        """Check whether the episode has failed (e.g. object fell). Override per task."""
        return False

    def make_eval_simulator(self, video_path: str | None = None, render: bool = True):
        """Build the eval/"real" simulator selected by self.config.eval_sim.

        The MuJoCo branch is generic. The Drake branch needs a task-specific
        joint-channel/actuation map, so tasks that support Drake eval override
        this method (delegating MuJoCo back to super()).
        """
        if self.config is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no config; cannot build an eval simulator."
            )
        if self.config.eval_sim == EvalSimulatorKind.MUJOCO:
            from contact_study.sim.mujoco_sim import MujocoSimulator
            return MujocoSimulator(self.mjm, self.config, render=render)
        raise NotImplementedError(
            f"Drake eval for {type(self).__name__} requires a task-specific channel "
            f"map; override make_eval_simulator()."
        )

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

        q0, v0, u0 = self.sample_initial_state(rng)
        mjd.qpos[:] = q0
        mjd.qvel[:] = v0
        if u0 is not None:
            mjd.ctrl[:] = u0
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

def get_task(
    name: str,
    geometry: GeometryVariant = GeometryVariant.ACCURATE,
    role: TaskRole = TaskRole.ROLLOUT,
) -> BaseTask:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](geometry=geometry, role=role)

def list_tasks() -> list[str]:
    return list(_REGISTRY.keys())
