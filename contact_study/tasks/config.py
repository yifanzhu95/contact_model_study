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
    MUJOCO    = "mujoco"
    DRAKE     = "drake"
    PINOCCHIO = "pinocchio"


# ---------------------------------------------------------------------------
# Scene variants
# ---------------------------------------------------------------------------
# A scene variant names three things at once: WHICH object is manipulated, at
# WHAT collision fidelity the planner's hand is modelled, and at what fidelity
# the object is. Scene files are then found by convention, so adding an object
# means dropping in XMLs rather than editing code:
#
#   rollout:  scenes/leap/env_leap_rollout_{obj}_{hand_acc}_{obj_acc}.xml
#   eval:     scenes/leap/env_leap_eval_{obj}.xml
#
# The eval scene carries no accuracy suffix on purpose — it IS the reference
# fidelity. Only the planner's model is degraded, which is the axis the study
# actually varies.
#
# Object names may NOT contain "_" (parse() splits on it).
DEFAULT_OBJECT   = "cube"
DEFAULT_HAND_ACC = "low"
DEFAULT_OBJ_ACC  = "high"
DEFAULT_SCENE_VARIANT = f"{DEFAULT_OBJECT}_{DEFAULT_HAND_ACC}_{DEFAULT_OBJ_ACC}"

# Values the retired GeometryVariant enum accepted. Every one maps to the
# default variant, which resolves to exactly the scene files that were loaded
# before scene variants existed. Kept so the experiments/hpc/*.slurm scripts
# that still pass `--geometry accurate` keep working untouched; delete this
# table once those are updated.
_LEGACY_GEOMETRY_ALIASES = frozenset(
    {"accurate", "convex_hull", "primitive_union", "linearized"}
)


@dataclass(frozen=True)
class SceneVariant:
    """Parsed scene-variant selector, built once per task instance from the
    `geometry` string that flows in from the CLI.

    The string stays a plain `str` everywhere outside the task layer, so every
    driver/experiment signature that merely forwards it is unaffected.
    """
    obj:      str = DEFAULT_OBJECT
    hand_acc: str = DEFAULT_HAND_ACC
    obj_acc:  str = DEFAULT_OBJ_ACC
    raw:      str = DEFAULT_SCENE_VARIANT   # what the caller actually typed

    @classmethod
    def parse(cls, s: str | None) -> "SceneVariant":
        """Accepted forms:
            None / ""            -> defaults
            a legacy enum name   -> defaults (raw preserved, so output labels
                                    built from the raw string keep working)
            "duck"               -> that object at the default fidelities
            "duck_low_high"      -> object, hand accuracy, object accuracy
        """
        if s is None:
            return cls()
        s = str(s).strip()
        if not s or s in _LEGACY_GEOMETRY_ALIASES:
            return cls(raw=s or DEFAULT_SCENE_VARIANT)

        parts = s.split("_")
        if len(parts) == 1:
            return cls(obj=parts[0], raw=s)
        if len(parts) == 3:
            return cls(obj=parts[0], hand_acc=parts[1], obj_acc=parts[2], raw=s)
        raise ValueError(
            f"Bad scene variant {s!r}. Expected '<object>' (default fidelities), "
            f"'<object>_<hand_acc>_<obj_acc>' (e.g. duck_low_high), or a legacy "
            f"geometry name ({sorted(_LEGACY_GEOMETRY_ALIASES)}). "
            f"Object names may not contain '_'."
        )

    def format(self, template: str) -> str:
        """Fill a scene template.

        Supplies every parsed field plus `geometry` (the raw string), so the
        role-specific templates and the legacy `{geometry}` templates that
        push/peg_in_hole still use are both served by one call. A template with
        no placeholders passes through untouched.
        """
        return template.format(
            obj=self.obj, hand_acc=self.hand_acc, obj_acc=self.obj_acc,
            geometry=self.raw,
        )

    def __str__(self) -> str:
        return self.raw


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
    # Legacy role-agnostic scene template (None for URDF-sourced tasks). Still
    # the fallback for every single-scene task; see BaseTask.resolve_scene_path.
    xml_path_template:  str | None = None

    # --- scene templates, filled by SceneVariant.format --------------------
    # Role-specific MJCF templates. When set, they take precedence over
    # xml_path_template for that role; leave both None on single-scene tasks.
    #   rollout_xml_template = "leap/env_leap_rollout_{obj}_{hand_acc}_{obj_acc}.xml"
    #   eval_xml_template    = "leap/env_leap_eval_{obj}.xml"
    # The eval scene carries no accuracy suffix: it IS the reference fidelity.
    rollout_xml_template: str | None = None
    eval_xml_template:    str | None = None

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
    # Per-simulator eval model paths: maps an EvalSimulatorKind to the model
    # file that simulator loads (e.g. Drake -> URDF, MuJoCo/Pinocchio -> MJCF).
    # Look up with the active eval_sim; falls back to the rollout model when the
    # active eval_sim has no entry and eval_sim is MUJOCO.
    eval_model_paths: dict[EvalSimulatorKind, str] = field(default_factory=dict)

    # --- camera (shared by Drake's VideoWriter and MuJoCo's Renderer) ------
    # world_from_camera rotation (3x3) and camera position, world frame.
    cam_pos:    tuple[float, float, float] = (0.0, -2.5, 0.25)
    cam_rotmat: tuple[tuple[float, float, float], ...] = (
        (1.0,  0.0, 0.0),
        (0.0,  0.0, 1.0),
        (0.0, -1.0, 0.0),
    )
    cam_fps:    float = 30.0
    cam_width:  int   = 640
    cam_height: int   = 480

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
