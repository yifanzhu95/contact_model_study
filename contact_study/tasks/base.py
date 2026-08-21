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

# ContactComplexity / TaskSpec now live in config.py (to avoid a base<->config
# import cycle) and are re-exported here for backward compatibility.
from contact_study.tasks.config import (
    ContactComplexity,
    TaskSpec,
    TaskConfig,
    TaskRole,
    EvalSimulatorKind,
    SceneVariant,
    DEFAULT_SCENE_VARIANT,
)

SCENES_DIR = Path(__file__).parents[2] / "scenes"


class BaseTask(abc.ABC):
    """Abstract base class for all manipulation tasks."""

    def __init__(
        self,
        geometry: str = DEFAULT_SCENE_VARIANT,
        role: TaskRole = TaskRole.ROLLOUT,
    ):
        # `geometry` stays the raw string the CLI passed (so callers that label
        # output with it are unaffected); scene_variant is the parsed form that
        # resolve_scene_path fills templates from.
        self.geometry = geometry
        self.scene_variant = SceneVariant.parse(geometry)
        self.role = role
        self._mjm: mujoco.MjModel | None = None
        self._mjd: mujoco.MjData  | None = None

        # Editable per-task configuration. Tasks migrated to TaskConfig set this
        # in their constructor; legacy tasks (push, peg_in_hole) leave it None
        # and instead override the `spec` property (TaskSpec) themselves.
        self.config: TaskConfig | None = None

        # Task-specific metadata extracted at load time
        self.goal_vector: np.ndarray | None = None
        self.index_vector: np.ndarray | None = None
        self.goal_vector_wp: wp.array | None = None
        self.index_vector_wp: wp.array | None = None
        self.weights_wp: wp.array | None = None

    def resolve_scene_path(self) -> Path:
        """Resolve which MJCF this instance loads, honouring self.role.

        Precedence:
          1. the role-specific template — eval_xml_template for EVAL,
             rollout_xml_template for ROLLOUT — when the task defines one;
          2. xml_path_template, the role-agnostic fallback that keeps every
             single-scene task (balance_stick, actuator_test, and the legacy
             push/peg_in_hole templates) working unchanged;
          3. error.

        Templates are filled by SceneVariant.format, so one with no
        placeholders passes through untouched.
        """
        cfg = self.config
        if cfg is None:
            raise ValueError(
                f"{type(self).__name__} has no TaskConfig; override load()."
            )

        tmpl = None
        if self.role is TaskRole.EVAL:
            tmpl = cfg.eval_xml_template
        elif self.role is TaskRole.ROLLOUT:
            tmpl = cfg.rollout_xml_template
        if tmpl is None:
            tmpl = cfg.xml_path_template
        if tmpl is None:
            raise ValueError(
                f"{type(self).__name__} defines no scene template for "
                f"role={self.role.value} (set rollout_xml_template / "
                f"eval_xml_template, or xml_path_template for a single scene)."
            )

        path = SCENES_DIR / self.scene_variant.format(tmpl)
        if not path.exists():
            siblings = sorted(p.name for p in path.parent.glob("*.xml"))
            raise FileNotFoundError(
                f"No scene for variant {self.scene_variant.raw!r} "
                f"role={self.role.value}: {path}\n"
                f"  template: {tmpl}\n"
                f"  present:  {siblings}"
            )
        return path

    def _publish_eval_model_paths(self) -> None:
        """Record the resolved eval scene as the model path for the eval
        simulators that re-parse the MJCF from disk.

        setdefault, so a task that pins an explicit path keeps it, and DRAKE is
        never touched — its hand is a URDF that predates the scene-variant
        naming convention and has no per-object form.
        """
        cfg = self.config
        if cfg is None or not cfg.eval_xml_template:
            return
        eval_path = str(SCENES_DIR / self.scene_variant.format(cfg.eval_xml_template))
        for kind in (EvalSimulatorKind.MUJOCO, EvalSimulatorKind.PINOCCHIO):
            cfg.eval_model_paths.setdefault(kind, eval_path)
        if cfg.rollout_model_path is None and cfg.rollout_xml_template:
            cfg.rollout_model_path = str(
                SCENES_DIR / self.scene_variant.format(cfg.rollout_xml_template)
            )

    def load(self, full_path: str | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Load the MuJoCo model for this task's role and scene variant.

        Tasks that set self.config (TaskConfig) use it directly here. Legacy
        tasks (push, peg_in_hole) leave self.config None and override `load()`
        or `spec` themselves — see those modules.
        """
        if full_path is None:
            full_path = self.resolve_scene_path()
        self._mjm = mujoco.MjModel.from_xml_path(str(full_path))
        self._mjd = mujoco.MjData(self._mjm)

        # Must run before initialize_task / make_eval_simulator: the eval-sim
        # builders read config.eval_model_paths immediately after load().
        self._publish_eval_model_paths()

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

    def goal_errors(self, mjd: mujoco.MjData) -> dict[str, float]:
        """Per-criterion distance to the goal, keyed like config.success_thresholds.

        The keys must match success_thresholds, so a criterion is satisfied iff
        err[k] < thr[k] — that is exactly the is_success test, and it lets a
        search scalarize the error by normalizing each term against its own
        threshold. Returns {} for tasks with no meaningful continuous goal
        metric (callers must treat an empty dict as "not available").
        """
        return {}

    def has_failed(self, mjd: mujoco.MjData) -> bool:
        """Check whether the episode has failed (e.g. object fell). Override per task."""
        return False

    def make_eval_simulator(self, video_path: str | None = None, render: bool = True,
                            use_mp4: bool = True):
        """Build the eval/"real" simulator selected by self.config.eval_sim.

        The MuJoCo branch is generic. The Drake branch needs a task-specific
        joint-channel/actuation map, so tasks that support Drake eval override
        this method (delegating MuJoCo back to super()).

        use_mp4 selects the video container (.mp4 when True, .gif otherwise); it
        overrides the extension of video_path / the path passed to save_video.
        """
        if self.config is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no config; cannot build an eval simulator."
            )
        if self.config.eval_sim == EvalSimulatorKind.MUJOCO:
            from contact_study.sim.mujoco_sim import MujocoSimulator
            return MujocoSimulator(self.mjm, self.config, render=render, use_mp4=use_mp4)
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
        T   = max_steps or self.config.max_steps

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
    geometry: str = DEFAULT_SCENE_VARIANT,
    role: TaskRole = TaskRole.ROLLOUT,
) -> BaseTask:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](geometry=geometry, role=role)

def list_tasks() -> list[str]:
    return list(_REGISTRY.keys())
