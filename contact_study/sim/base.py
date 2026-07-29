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
import math
from dataclasses import dataclass
from pathlib import Path

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

    def render(self) -> None:
        """Deprecated no-op, kept so older call sites still work.

        Frames are captured on the SIM clock from inside step() (see FrameClock),
        not on the caller's control clock — a driver that calls render() once per
        control step would otherwise write a video whose playback speed depends on
        the control frequency.
        """
        return None

    @abc.abstractmethod
    def save_video(self, path: str) -> str | None:
        """Write any captured frames to `path`.

        Returns the path actually written, which may differ from `path`: the
        container (.mp4/.gif) is chosen by the simulator's `use_mp4` flag, so the
        extension is normalized here. Returns None when there is nothing to write.
        """
        ...

    @property
    @abc.abstractmethod
    def timestep(self) -> float:
        """Control/integration timestep in seconds."""
        ...


class FrameClock:
    """Sim-time video frame scheduler.

    advance() is called once per FINE sim substep and returns True on the substep
    whose sim time lands closest to the next 1/fps deadline. Capturing there (rather
    than once per control step, as the drivers used to via render()) makes the stored
    frames evenly spaced in SIM time, so a video written at `fps` plays back in real
    time no matter what the control frequency is — and the frame count stays ~fps per
    simulated second instead of scaling with the control rate.
    """

    def __init__(self, fps: float):
        self.frame_dt = 1.0 / fps if fps and fps > 0 else 0.0
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.next_t = 0.0

    def advance(self, dt: float) -> bool:
        """Advance the clock by one substep; True if this substep should be captured."""
        self.t += dt
        if self.frame_dt <= 0.0:
            return False
        # Round, don't floor: fire on the substep whose end time is nearest the
        # deadline, i.e. as soon as we are within half a substep of it.
        if self.t + 0.5 * dt < self.next_t - 1e-12:
            return False
        self.next_t += self.frame_dt
        if self.next_t <= self.t:
            # frame_dt < dt (fps faster than the sim can resolve): drop the
            # deadlines we cannot hit rather than emitting duplicate frames.
            self.next_t += math.ceil((self.t - self.next_t) / self.frame_dt) * self.frame_dt
        return True


def resolve_video_path(path, use_mp4: bool) -> str:
    """Force `path`'s extension to match the requested container (.mp4 or .gif)."""
    return str(Path(path).with_suffix(".mp4" if use_mp4 else ".gif"))


def camera_pose_from_config(config) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_world_camera 3x3, position 3) from a TaskConfig.

    Single source of truth for the camera so Drake's VideoWriter and MuJoCo's
    Renderer frame the scene identically.
    """
    R = np.asarray(config.cam_rotmat, dtype=float)
    p = np.asarray(config.cam_pos, dtype=float)
    return R, p
