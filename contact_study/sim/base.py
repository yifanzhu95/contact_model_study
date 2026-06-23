"""Eval-simulator abstraction.

The eval/"real" environment is pluggable: MuJoCo (single-world MjData) or Drake
(pydrake MultibodyPlant). Both implement the same small interface so the main
loop in `contact_study/drivers/run_eval_episode.py` is simulator-agnostic.

Two contracts make this work across simulators with different internal joint
orderings:

  * `get_state()` returns qpos/qvel in the **MuJoCo** index layout, so the
    driver can mirror the state straight into the planning MjData.
  * `apply_control()` takes a **MuJoCo-ordered, MuJoCo-semantics** command
    (absolute — the driver owns any delta/rate accumulation).

Each simulator remaps internally. The driver never sees simulator-specific
indices.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class EvalState:
    """Simulator state in the MuJoCo qpos/qvel index layout."""
    qpos: np.ndarray
    qvel: np.ndarray


class EvalSimulator(abc.ABC):
    """Interface for a high-fidelity eval/"real" environment."""

    @abc.abstractmethod
    def reset(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Reset to an initial state (MuJoCo-ordered)."""
        ...

    @abc.abstractmethod
    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Overwrite the current state (MuJoCo-ordered)."""
        ...

    @abc.abstractmethod
    def get_state(self) -> EvalState:
        """Read the current state, returned in the MuJoCo index layout."""
        ...

    @abc.abstractmethod
    def apply_control(self, ctrl: np.ndarray) -> None:
        """Apply an absolute, MuJoCo-ordered control command."""
        ...

    @abc.abstractmethod
    def step(self, n_substeps: int = 1) -> None:
        """Advance the simulation by `n_substeps` control-timestep increments."""
        ...

    @abc.abstractmethod
    def render(self) -> None:
        """Capture one video frame (no-op if rendering is disabled)."""
        ...

    @abc.abstractmethod
    def save_video(self, path: str) -> None:
        """Write any captured frames to `path`."""
        ...

    @property
    @abc.abstractmethod
    def timestep(self) -> float:
        """Control/integration timestep in seconds."""
        ...


def camera_pose_from_config(config) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_world_camera 3x3, position 3) from a TaskConfig.

    Single source of truth for the camera so Drake's VideoWriter and MuJoCo's
    Renderer frame the scene identically.
    """
    R = np.asarray(config.cam_rotmat, dtype=float)
    p = np.asarray(config.cam_pos, dtype=float)
    return R, p
