"""Small, backend-independent contracts. Defaults are development settings, not a study."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Protocol

import torch


MJWARP_NATIVE_PROFILE = "native_mjwarp_v1"


@dataclass(frozen=True)
class EnvConfig:
    num_envs: int = 2
    contact_preset: str = "M2"
    physics_backend: str = "mujoco_cpu_diagnostic"
    device: str = "cpu"
    physics_dt: float = 0.004
    control_dt: float = 0.016
    episode_steps: int = 32
    cost_reference: float = 100.0
    target_rate: float = 1.0
    goal_difficulty: int = 8
    seed: int = 0
    numerics_profile: str | None = None
    randomization: dict | None = None
    task_name: str = "cube_reorient"
    scene_role: str = "eval"
    hand_acc: str = "high"
    obj_acc: str = "high"
    cost_weights: dict | None = None

    def __post_init__(self):
        from ContactModelStudy.Tasks.Registry import TASK_MODULES
        if not isinstance(self.task_name, str) or self.task_name not in TASK_MODULES:
            raise ValueError(f"task_name must be one of {tuple(TASK_MODULES)}")
        if self.scene_role not in ("eval", "rollout") or self.hand_acc not in ("low", "med", "high"):
            raise ValueError("scene_role must be eval/rollout and hand_acc low/med/high")
        if not isinstance(self.obj_acc, str) or not self.obj_acc or not self.obj_acc.replace("_", "").isalnum():
            raise ValueError("obj_acc must name an existing scene accuracy")
        if self.cost_weights is not None:
            from ContactModelStudy.Tasks.LeapReorient import COST_WEIGHT_KEYS
            if (not isinstance(self.cost_weights, dict) or set(self.cost_weights)-set(COST_WEIGHT_KEYS)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or v < 0 for v in self.cost_weights.values())):
                raise ValueError("cost_weights must contain known finite nonnegative weights")
            object.__setattr__(self, "cost_weights", {k: float(v) for k, v in self.cost_weights.items()})
        if self.randomization is not None:
            from ContactModelStudy.RL.Randomization import reserved_spec
            object.__setattr__(self, "randomization", reserved_spec(self.randomization))
        for key in ("num_envs", "episode_steps", "goal_difficulty", "seed"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        if self.num_envs < 1 or self.episode_steps < 1 or self.seed < 0:
            raise ValueError("num_envs/episode_steps must be positive; seed must be nonnegative")
        if not 0 <= self.goal_difficulty <= 9:
            raise ValueError("goal_difficulty must be in 0..9")
        for key in ("physics_dt", "control_dt", "cost_reference", "target_rate"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        ratio = self.control_dt / self.physics_dt
        if ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
            raise ValueError("control_dt must be an integer multiple of physics_dt")
        allowed_presets = {"mujoco_cpu_diagnostic": ("M1", "M2"), "mjwarp": ("M1", "M2"),
                           "comfree": ("M3",), "xpbd": ("M4",), "kamino": ("M5",),
                           "mujoco_eval": ("native",), "pinocchio_eval": ("native",), "drake_eval": ("native",)}
        if self.physics_backend not in allowed_presets:
            raise ValueError(f"Implemented backends: {tuple(allowed_presets)}; no fallback")
        cpu = self.physics_backend in ("mujoco_cpu_diagnostic", "mujoco_eval", "pinocchio_eval", "drake_eval")
        if ((cpu and self.device != "cpu") or (not cpu and self.device != "cuda:0")):
            raise ValueError("CPU backends require cpu; current GPU adapters require cuda:0")
        allowed = allowed_presets[self.physics_backend]
        if self.contact_preset not in allowed:
            raise ValueError(f"{self.physics_backend} implements {allowed}; no model substitution")
        if self.physics_backend == "mjwarp" and self.numerics_profile is None:
            object.__setattr__(self, "numerics_profile", MJWARP_NATIVE_PROFILE)
        if self.numerics_profile is not None:
            if self.physics_backend != "mjwarp" or self.numerics_profile not in (
                    MJWARP_NATIVE_PROFILE, "dense_newton_ordered_changes_v1", "dense_newton_ordered_rows_gpu_v1"):
                raise ValueError("Explicit numerics_profile requires a declared MJWarp profile")

    @property
    def physics_steps(self) -> int:
        return round(self.control_dt / self.physics_dt)

    def to_dict(self):
        value = asdict(self)
        # Omit inapplicable options; MJWarp profiles are always explicit in new runs.
        if self.numerics_profile is None:
            value.pop("numerics_profile")
        if self.randomization is None:
            value.pop("randomization")
        # Default cube artifacts retain their original serialized contract.
        for key, default in (("task_name", "cube_reorient"), ("scene_role", "eval"),
                             ("hand_acc", "high"), ("obj_acc", "high"), ("cost_weights", None)):
            if value[key] == default:
                value.pop(key)
        return value


@dataclass(frozen=True)
class Frame:
    """Owned canonical snapshots, not views into mutable simulation arrays.

    qpos/qvel use the existing project's MuJoCo order. Sites and object twist
    use world axes. The twist is linear then angular, at the object origin.
    """
    qpos: torch.Tensor
    qvel: torch.Tensor
    sites: torch.Tensor
    object_twist: torch.Tensor


class PhysicsBatch(Protocol):
    num_envs: int

    def reset_worlds(self, ids, cases) -> None: ...
    def step(self, target: torch.Tensor, steps: int) -> None: ...
    def read_frame(self) -> Frame: ...
    def describe(self) -> dict: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class Transition:
    next_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    reasons: tuple[str, ...]
    diagnostics: dict[str, torch.Tensor]

    @property
    def done_ids(self):
        return torch.nonzero((self.terminated | self.truncated).flatten()).flatten()


class NumericalTransitionError(RuntimeError):
    """Invalid physics must not be inserted into the learner's buffer."""

    def __init__(self, message: str, frame: Frame):
        super().__init__(message)
        self.frame = frame
