"""Task configuration objects.

`TaskConfig` is the editable, per-task configuration: construct it, tweak any
field, then hand it to a task constructor. It is a superset of the legacy
`TaskSpec` (it carries the same field names — name/complexity/max_steps/
cost_weights/success_thresholds/xml_path_template) plus the new fields needed
to drive a selectable eval simulator: which simulator backs the "real"
environment, the eval model path, the camera pose used for rendering, and
dynamics/control limits.

Tasks migrated off `TaskSpec` (cart_pole, grasp_reorient) set `self.config` and
read it directly — no `spec` property involved. The two not yet migrated
(push, peg_in_hole) still override `spec` themselves and read from it; generic
callers spanning all task types use `task.config or task.spec`.

`ContactComplexity` and `TaskSpec` live here (rather than in base.py) so that
`base.py` can import them without a circular dependency; they are re-exported
from `base.py` for backward compatibility.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class ContactComplexity(enum.IntEnum):
    """Qualitative contact complexity level, used to sort tasks in results."""
    LOW    = 1   # pushing: ≤2 contacts, quasi-static
    MEDIUM = 2   # grasp-reorient: ~4 contacts, dynamic
    HIGH   = 3   # peg-in-hole assembly: tight clearance, multi-contact


class TaskRole(enum.Enum):
    """Which side of the planner/real split a task instance represents.

    ROLLOUT — owns the planning MuJoCo model + cost arrays consumed by the
              MPPIController for batched GPU rollouts.
    EVAL     — owns the high-fidelity "real" environment (an EvalSimulator,
              either MuJoCo or Drake) that the main loop actually steps.
    """
    ROLLOUT = "rollout"
    EVAL    = "eval"


class EvalSimulatorKind(str, enum.Enum):
    """Which simulator backs the eval/"real" environment."""
    MUJOCO = "mujoco"
    DRAKE  = "drake"


@dataclass
class TaskSpec:
    """Legacy task spec. Retained for the tasks not migrated to TaskConfig
    (push, peg_in_hole) and re-exported from base.py for import compatibility."""
    name:               str
    complexity:         ContactComplexity
    xml_path_template:  str    # format string with {geometry} placeholder
    max_steps:          int
    success_thresholds: dict   # e.g. {"pos": 0.05, "quat": 0.05, "vel": 0.1}
    cost_weights:       dict


@dataclass
class TaskConfig:
    """Editable per-task configuration. Initialize → edit → pass to the task.

    The first block mirrors `TaskSpec` (same attribute names) so a task can
    expose `self.config` as `task.spec` and keep every `task.spec.X` caller
    working unchanged.
    """

    # --- identity / legacy TaskSpec-compatible fields ----------------------
    name:               str
    complexity:         ContactComplexity
    max_steps:          int
    cost_weights:       dict = field(default_factory=dict)
    success_thresholds: dict = field(default_factory=dict)
    # Legacy MuJoCo geometry-variant template (None for URDF-sourced tasks).
    xml_path_template:  str | None = None

    # --- ROLLOUT (planning) model source -----------------------------------
    # Path to the MJCF the GPU rollouts plan with (precompiled, loaded as-is).
    rollout_model_path: str | None = None
    rollout_is_urdf:    bool       = False
    # Precompiled MJCF path passed to MjModel.from_xml_path by tasks (e.g.
    # cart_pole) that load a static rollout model instead of using
    # xml_path_template.
    mjcf_out_path:      str | None = None

    # --- EVAL ("real") simulator selection ---------------------------------
    eval_sim:        EvalSimulatorKind = EvalSimulatorKind.MUJOCO
    # Model for the eval simulator (SDF/URDF for Drake). Falls back to the
    # rollout model when None and eval_sim is MUJOCO.
    eval_model_path: str | None = None

    # --- camera (shared by Drake's VideoWriter and MuJoCo's Renderer) ------
    # world_from_camera rotation (3x3) and camera position, world frame.
    cam_pos:    tuple[float, float, float] = (0.0, -2.5, 0.25)
    cam_rotmat: tuple[tuple[float, float, float], ...] = (
        (1.0,  0.0, 0.0),
        (0.0,  0.0, 1.0),
        (0.0, -1.0, 0.0),
    )
    cam_fps:    float = 30.0

    # --- dynamics / control ------------------------------------------------
    # Eval ("real") simulator timestep (s) — the fine, high-fidelity step the
    # EvalSimulator integrates at. The rollout/planning model runs coarser:
    #   rollout_dt = timestep * eval_substeps_per_rollout
    # i.e. the eval sim takes `eval_substeps_per_rollout` steps for each single
    # rollout step. The driver infers rollout_dt from these and stamps it onto
    # the planning model. Control frequency stays a separate knob (MPPI substeps:
    #   control_dt = mppi_substeps * rollout_dt).
    timestep:                  float = 0.002
    eval_substeps_per_rollout: int   = 10
    # Absolute actuator command range (e.g. hand joint-target ctrlrange).
    control_limits: tuple[float, float] | None = None
    # Cart force clip for cart_pole (absolute, applied after delta integration).
    force_limits:   tuple[float, float] | None = None

    # --- task-specific knobs ----------------------------------------------
    difficulty: int = 1   # grasp_reorient goal difficulty
