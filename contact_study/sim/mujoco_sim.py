"""MuJoCo-backed eval simulator (single world).

This is the fallback "real" environment when a task's eval_sim is MUJOCO. It is
a thin wrapper over a single MjModel/MjData with the EvalSimulator interface.
Because it *is* MuJoCo, get_state/apply_control are identity index maps.
"""

from __future__ import annotations

import numpy as np
import mujoco

from contact_study.sim.base import EvalSimulator, EvalState, camera_pose_from_config


class MujocoSimulator(EvalSimulator):
    def __init__(
        self,
        mjm: mujoco.MjModel,
        config,
        render: bool = True,
        camera_name: str | None = None,
        height: int = 480,
        width: int = 640,
    ):
        self.mjm = mjm
        self.mjd = mujoco.MjData(mjm)
        self._config = config
        self._camera_name = camera_name
        self._frames: list[np.ndarray] = []

        self._renderer = (
            mujoco.Renderer(mjm, height=height, width=width) if render else None
        )
        self._cam = self._build_camera()

    # -- camera --------------------------------------------------------------
    def _build_camera(self):
        """Free MjvCamera framed to match TaskConfig's camera pose.

        Uses the same world<-camera rotation/position as Drake's VideoWriter so
        the two simulators frame the scene the same way. The camera's viewing
        direction is the +Z column of the world<-camera rotation (Drake camera
        optical-axis convention)."""
        if self._camera_name is not None:
            return self._camera_name  # let the Renderer resolve a named camera

        R, eye = camera_pose_from_config(self._config)
        forward = R[:, 2]
        dist = float(np.linalg.norm(eye)) or 1.0
        lookat = eye + forward * dist

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = dist
        cam.azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])))
        cam.elevation = float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0))))
        return cam

    # -- EvalSimulator interface --------------------------------------------
    def reset(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.set_state(qpos, qvel)

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.mjd.qpos[:] = qpos
        self.mjd.qvel[:] = qvel
        mujoco.mj_forward(self.mjm, self.mjd)

    def get_state(self) -> EvalState:
        return EvalState(self.mjd.qpos.copy(), self.mjd.qvel.copy())

    def apply_control(self, ctrl: np.ndarray) -> None:
        self.mjd.ctrl[:] = ctrl

    def step(self, n_substeps: int = 1) -> None:
        for _ in range(n_substeps):
            mujoco.mj_step(self.mjm, self.mjd)

    def render(self) -> None:
        if self._renderer is None:
            return
        self._renderer.update_scene(self.mjd, camera=self._cam)
        self._frames.append(self._renderer.render())

    def save_video(self, path: str) -> None:
        if not self._frames:
            return
        import mediapy as media
        media.write_video(path, self._frames, fps=int(self._config.cam_fps))

    @property
    def timestep(self) -> float:
        return float(self.mjm.opt.timestep)
