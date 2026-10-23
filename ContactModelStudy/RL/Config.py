"""What a training run is: the environment it trains in and the PPO settings it trains with.

``EnvConfig`` is also the contract a trained actor carries. ``run_episodes``
reads it back from ``actor.pt`` to run the policy at the control step, episode
length and settle time it was trained for.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

#: Contact models a training environment can run on, as for ``--rollout-model``.
TRAIN_MODELS = ("M1", "M2", "M3", "M4", "M5")


@dataclass(frozen=True)
class EnvConfig:
    """One training environment: ``num_envs`` worlds of one task on one contact model.

    Attributes:
        task_name: ``cube_reorient``, ``duck_reorient`` or ``ball_reorient``.
        rollout_model: Contact model the policy trains in, M1-M5.
        hand_acc, obj_acc: Rollout-scene fidelity, as ``run_episodes``'
            ``--hand-acc`` / ``--obj-acc``.
        num_envs: Parallel worlds.
        physics_dt: Simulator timestep (s).
        control_dt: Control step (s); a whole number of ``physics_dt``.
        episode_steps: Control steps before the deadline ends an episode.
        settle_s: Seconds the initial grasp is held after a reset before the
            episode starts, as ``run_episodes --settle``.
        target_rate: Largest change of an actuator target per second.
        cost_reference: Running cost that maps to the largest per-step penalty.
        goal_difficulty: The task's goal sampler, 0-9.
        seed: Goal sampling and network initialization seed.
        cost_weights: Overrides of the task's named cost weights.
        device: CUDA device for the simulator and the networks.
    """

    task_name: str = "cube_reorient"
    rollout_model: str = "M2"
    hand_acc: str = "high"
    obj_acc: str = "high"
    num_envs: int = 256
    physics_dt: float = 0.004
    control_dt: float = 0.064
    episode_steps: int = 150
    settle_s: float = 1.0
    target_rate: float = 1.0
    cost_reference: float = 100.0
    goal_difficulty: int = 8
    seed: int = 0
    cost_weights: dict | None = None
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        from ContactModelStudy.Tasks.Registry import TASK_MODULES

        if self.task_name not in TASK_MODULES:
            raise ValueError(f"task_name must be one of {tuple(TASK_MODULES)}, got {self.task_name!r}")
        if self.rollout_model not in TRAIN_MODELS:
            raise ValueError(f"rollout_model must be one of {TRAIN_MODELS}, got {self.rollout_model!r}")
        for key in ("num_envs", "episode_steps", "goal_difficulty", "seed"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer, got {value!r}")
        if self.num_envs < 2:
            raise ValueError("num_envs must be at least 2; skrl's vectorized loop needs a batch")
        if self.episode_steps < 1 or self.seed < 0 or not 0 <= self.goal_difficulty <= 9:
            raise ValueError("episode_steps must be positive, seed nonnegative and goal_difficulty 0-9")
        for key in ("physics_dt", "control_dt", "target_rate", "cost_reference"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(self.settle_s) or self.settle_s < 0:
            raise ValueError("settle_s must be finite and nonnegative")
        if not math.isclose(self.substeps * self.physics_dt, self.control_dt, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"control_dt {self.control_dt} is not a whole number of physics_dt {self.physics_dt}")
        if not str(self.device).startswith("cuda"):
            raise ValueError("Training runs on a CUDA device")

    @property
    def substeps(self) -> int:
        """Physics steps per control step."""
        return max(1, round(self.control_dt / self.physics_dt))

    @property
    def settle_steps(self) -> int:
        """Physics steps of the settle phase."""
        return int(round(self.settle_s / self.physics_dt))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PPOConfig:
    """skrl PPO settings. A rollout is ``num_envs * rollout_steps`` transitions."""

    rollout_steps: int = 32
    learning_epochs: int = 5
    mini_batches: int = 4
    learning_rate: float = 3e-4
    hidden_sizes: tuple[int, ...] = (256, 128)
    ratio_clip: float = 0.2
    gae_lambda: float = 0.95
    entropy_loss_scale: float = 0.0
    initial_log_std: float = -1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        counts = (self.rollout_steps, self.learning_epochs, self.mini_batches, *self.hidden_sizes)
        if not self.hidden_sizes or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in counts):
            raise ValueError("rollout_steps, learning_epochs, mini_batches and hidden_sizes must be positive integers")
        if not 0 < self.learning_rate < math.inf or not 0 < self.ratio_clip < 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid learning rate, ratio clip or GAE lambda")
        if not math.isfinite(self.entropy_loss_scale) or self.entropy_loss_scale < 0:
            raise ValueError("entropy_loss_scale must be finite and nonnegative")
        if not -5 <= self.initial_log_std <= 1:
            raise ValueError("initial_log_std must be in [-5, 1]")

    def to_dict(self) -> dict:
        return asdict(self)
