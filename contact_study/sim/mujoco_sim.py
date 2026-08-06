"""MuJoCo-backed eval simulator (single world).

This is the fallback "real" environment when a task's eval_sim is MUJOCO. It is
a thin wrapper over a single MjModel/MjData with the EvalSimulator interface.
Because it *is* MuJoCo, get_state/apply_control are identity index maps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from contact_study.sim.base import (
    EvalSimulator, EvalState, FrameClock, camera_pose_from_config, resolve_video_path,
)


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
        use_mp4: bool = True,
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
        self._goal_mocap_id = self._resolve_goal_mocap_id()
        self._frames: list[np.ndarray] = []
        # Output container for save_video: .mp4 (default) or .gif. The flag wins
        # over whatever extension the caller passes.
        self._use_mp4 = bool(use_mp4)

        self._renderer = (
            mujoco.Renderer(mjm, height=height, width=width) if render else None
        )
        self._cam = self._build_camera()

        # Frames are captured from inside step(), on the SIM clock: one frame per
        # 1/cam_fps of simulated time, taken at the fine substep nearest each
        # deadline. That keeps playback real-time regardless of how many fine steps
        # the caller advances per control step, and bounds the frame count to
        # cam_fps per simulated second. Mirrors Drake's VideoWriter, which already
        # captures on its own fps clock during AdvanceTo.
        self._clock = FrameClock(config.cam_fps)

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
        self._clock.reset()
        self._frames = []
        self.set_state(qpos, qvel)

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.mjd.qpos[:] = qpos
        self.mjd.qvel[:] = qvel
        mujoco.mj_forward(self.mjm, self.mjd)

    def get_state(self) -> EvalState:
        return EvalState(self.mjd.qpos.copy(), self.mjd.qvel.copy())

    def apply_control(self, ctrl: np.ndarray) -> None:
        self.mjd.ctrl[:] = ctrl

    # -- goal marker -----------------------------------------------------
    def _resolve_goal_mocap_id(self) -> int | None:
        """Find the on-screen goal-marker mocap body's id, if this scene has one.

        Mirrors the body-name fallback in grasp_reorient.py's _update_goal
        ("goal", then the legacy "obj_target" name); None for scenes without
        either (e.g. cart_pole), making set_goal_quat a no-op there."""
        for body_name in ("goal", "obj_target"):
            body_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                mocap_id = self.mjm.body_mocapid[body_id]
                if mocap_id >= 0:
                    return mocap_id
        return None

    def set_goal_quat(self, quat_wxyz: np.ndarray) -> None:
        """Re-orient the goal marker to match a newly sampled goal.

        This simulator has its own MjData, separate from the rollout task's
        (which is what grasp_reorient.sample_new_goal writes into), so the
        driver mirrors new goals here explicitly — see _update_goal's
        set_goal_quat call. No-op when this scene has no goal marker."""
        if self._goal_mocap_id is None:
            return
        self.mjd.mocap_quat[self._goal_mocap_id] = quat_wxyz

    def step(self, n_substeps: int = 1) -> None:
        dt = self.mjm.opt.timestep
        for _ in range(n_substeps):
            mujoco.mj_step(self.mjm, self.mjd)
            if self._clock.advance(dt) and self._renderer is not None:
                self._capture()

    def _capture(self) -> None:
        self._renderer.update_scene(self.mjd, camera=self._cam)
        self._frames.append(self._renderer.render())

    def save_video(self, path: str) -> str | None:
        if not self._frames:
            return None
        import mediapy as media
        out = resolve_video_path(path, self._use_mp4)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        # mediapy defaults to the h264 codec and does not infer "gif" from the
        # extension, so the ffmpeg GIF muxer rejects an h264-encoded .gif. Select
        # the codec from the extension (Drake's eval path writes gifs natively).
        kwargs = {"codec": "gif"} if out.lower().endswith(".gif") else {}
        # The frames are cam_fps apart in sim time, so writing at cam_fps replays
        # the episode in real time.
        media.write_video(out, self._frames, fps=float(self._config.cam_fps), **kwargs)
        return out

    @property
    def timestep(self) -> float:
        return float(self.mjm.opt.timestep)
