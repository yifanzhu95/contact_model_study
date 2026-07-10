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
        height: int | None = None,
        width: int | None = None,
        hard_contact: bool = True,
    ):
        height = height if height is not None else config.cam_height
        width = width if width is not None else config.cam_width

        self.mjm = mjm
        # The eval sim runs at the (fine) eval timestep, independent of whatever
        # the loaded XML declared. rollout_dt = config.timestep * substeps is
        # applied to the planning model separately (see the driver).
        self.mjm.opt.timestep = float(config.timestep)

        # Default the ground-truth eval sim to the M1 "hard_contact" preset:
        # stiffen every contact row's solref/solimp toward the hard-constraint
        # limit so the reference environment behaves rigidly. Only the contact
        # rows are hardened here — cone/solver/iterations stay at reference
        # (CPU) MuJoCo defaults, which (unlike MJWarp) support the elliptic
        # cone. Must run after opt.timestep is set: the preset derives its
        # solref timeconst from mjm.opt.timestep.
        if hard_contact:
            from contact_study.contact_models.api import _apply_hard_contact_preset
            from contact_study.contact_models.config import ContactModelConfig
            _apply_hard_contact_preset(self.mjm, ContactModelConfig.M1().mujoco)
        self.mjd = mujoco.MjData(mjm)
        self._config = config
        self._camera_name = camera_name
        self._frames: list[np.ndarray] = []

        self._renderer = (
            mujoco.Renderer(mjm, height=height, width=width) if render else None
        )
        self._cam = self._build_camera()

        # Capture frames at cam_fps, not every render() call — render() is
        # called once per (fine) sim step, and at cam_fps=30 with a small
        # timestep that's far more frames than the video needs, so storing all
        # of them blows up memory on long episodes. Mirrors Drake's VideoWriter,
        # which already captures on its own fps clock during AdvanceTo.
        self._t = 0.0
        self._frame_dt = 1.0 / config.cam_fps if config.cam_fps > 0 else 0.0
        self._next_frame_t = 0.0

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
        self._t = 0.0
        self._next_frame_t = 0.0
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
        self._t += n_substeps * self.mjm.opt.timestep

    def render(self) -> None:
        if self._renderer is None:
            return
        if self._frame_dt > 0 and self._t + 1e-9 < self._next_frame_t:
            return
        self._next_frame_t += self._frame_dt
        self._renderer.update_scene(self.mjd, camera=self._cam)
        self._frames.append(self._renderer.render())

    def save_video(self, path: str) -> None:
        if not self._frames:
            return
        import mediapy as media
        # mediapy defaults to the h264 codec and does not infer "gif" from the
        # extension, so the ffmpeg GIF muxer rejects an h264-encoded .gif. Select
        # the codec from the extension (Drake's eval path writes gifs natively).
        kwargs = {"codec": "gif"} if str(path).lower().endswith(".gif") else {}
        media.write_video(path, self._frames, fps=int(self._config.cam_fps), **kwargs)

    @property
    def timestep(self) -> float:
        return float(self.mjm.opt.timestep)
