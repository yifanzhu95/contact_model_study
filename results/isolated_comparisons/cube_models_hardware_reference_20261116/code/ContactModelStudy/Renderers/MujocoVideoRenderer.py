"""Offscreen MuJoCo renderer that accumulates frames into a video.

Wraps ``mujoco.Renderer``. Each ``RenderState(q)`` places the scene at ``q`` and
appends one rendered frame; ``Save`` writes them out.

Frames are held in memory until then — at the 640x480 default that is about
0.9 MB each, so a 30 fps minute is roughly 1.6 GB. ``Reset`` between episodes,
or a lower resolution, is the fix; this deliberately does not stream to disk,
because writing incrementally would give up the single-pass encode that keeps
playback smooth.

Offscreen rendering needs a GL context. On a headless node set ``MUJOCO_GL``
(``egl`` with a GPU, ``osmesa`` without) before the first render.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from ContactModelStudy.Renderers.RendererBase import (
    VideoRendererBase,
    VideoRendererBaseConfig,
)

# Containers mediapy can mux. Anything else is rejected up front, rather than
# after an episode's worth of frames has already been rendered.
_CONTAINERS = (".mp4", ".gif")


class MujocoVideoRenderer(VideoRendererBase):
    """Renders a task's scene offscreen with MuJoCo and writes a video.

    Attributes:
        mjm: The ``MjModel`` this renderer draws with. Its own, compiled from
            the task's MJCF — not the simulator's, so the renderer stays usable
            with backends that have no ``MjModel`` at all. Camera overrides from
            the config are written into it at construction.
        mjd: The ``MjData`` used to place the scene. Holds only positions;
            nothing here is ever stepped.
    """

    CONFIG_CLASS = VideoRendererBaseConfig

    def __init__(self, task, config: VideoRendererBaseConfig | None = None):
        """Compile the task's scene and open an offscreen renderer.

        Args:
            task: A task exposing ``getModelPath()``, or a path to an MJCF file.
            config: Video parameters. ``None`` asks the task for its own.

        Raises:
            FileNotFoundError: If the task's model file does not exist.
            ValueError: If ``cam_name`` names a camera the scene does not
                define, or ``output_path`` has an unsupported extension.
        """
        super().__init__(task, config)

        if self.config.output_path is not None:
            self._checkContainer(self.config.output_path)

        self.mjm = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.mjd = mujoco.MjData(self.mjm)
        self._camera = self._setupCamera()
        self._renderer = mujoco.Renderer(
            self.mjm, height=self.config.height, width=self.config.width
        )
        self._frames: list[np.ndarray] = []
        self._closed = False

    # -- camera --------------------------------------------------------------
    def _setupCamera(self):
        """Resolve the config's camera into an ``update_scene`` argument.

        Three cases, in order:

        * a named camera — its model entry is patched in place with whichever
          of ``cam_pos``/``cam_quat``/``cam_fovy`` are set, which is exact;
        * no name but some override — a free ``MjvCamera`` is synthesized from
          the requested pose, since a camera cannot be added to a compiled
          model. ``MjvCamera`` is parameterized by lookat/azimuth/elevation
          rather than pose, so the conversion is done here;
        * neither — ``-1``, MuJoCo's default free camera framed on the model.
        """
        cfg = self.config

        if cfg.cam_name is not None:
            cam_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, cfg.cam_name)
            if cam_id < 0:
                available = [
                    mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, i)
                    for i in range(self.mjm.ncam)
                ]
                raise ValueError(
                    f"camera {cfg.cam_name!r} is not defined in {self.model_path.name}; "
                    f"available: {available or 'none (use cam_name=None for the free camera)'}"
                )
            if cfg.cam_pos is not None:
                self.mjm.cam_pos[cam_id] = cfg.cam_pos
            if cfg.cam_quat is not None:
                self.mjm.cam_quat[cam_id] = cfg.cam_quat
            if cfg.cam_fovy is not None:
                self.mjm.cam_fovy[cam_id] = cfg.cam_fovy
            return cfg.cam_name

        if not cfg.has_camera_override:
            return -1
        return self._freeCamera(cfg)

    def _freeCamera(self, cfg):
        """Build a free ``MjvCamera`` at the requested pose.

        MuJoCo's camera convention: the camera looks along its own -Z axis with
        +Y up, so the view direction is the negated third column of the
        rotation matrix. ``MjvCamera`` orbits a lookat point, so a point one
        ``distance`` along that direction is used as the target.
        """
        pos = cfg.cam_pos if cfg.cam_pos is not None else np.zeros(3)
        if cfg.cam_quat is not None:
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, np.asarray(cfg.cam_quat, dtype=float))
            forward = -mat.reshape(3, 3)[:, 2]
        else:
            forward = np.array([0.0, 1.0, 0.0])   # look along +Y from the front
        if cfg.cam_fovy is not None:
            self.mjm.vis.global_.fovy = cfg.cam_fovy

        distance = float(np.linalg.norm(pos)) or 1.0
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = pos + forward * distance
        cam.distance = distance
        cam.azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])))
        cam.elevation = float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0))))
        return cam

    @staticmethod
    def _checkContainer(path: str | Path) -> Path:
        """Normalize an output path, defaulting a missing extension to .mp4."""
        p = Path(path)
        if p.suffix == "":
            p = p.with_suffix(".mp4")
        if p.suffix.lower() not in _CONTAINERS:
            raise ValueError(
                f"output must be one of {_CONTAINERS}, got {p.suffix!r} in {str(path)!r}"
            )
        return p

    # -- RendererBase interface ---------------------------------------------
    def RenderState(self, q: np.ndarray) -> None:
        """Place the scene at ``q`` and append one frame.

        Args:
            q: Positions, shape ``(nq,)``, in the MuJoCo qpos layout.

        Runs ``mj_forward`` so geom and site poses match ``q`` before the scene
        is drawn. Velocities are left at zero — nothing visible depends on them.
        """
        a = np.asarray(q, dtype=float).ravel()
        if a.shape != (self.mjm.nq,):
            raise ValueError(f"q must have shape ({self.mjm.nq},), got {np.shape(q)}")
        if self._closed:
            raise RuntimeError("RenderState called on a closed renderer")

        self.mjd.qpos[:] = a
        mujoco.mj_forward(self.mjm, self.mjd)
        self._renderer.update_scene(self.mjd, camera=self._camera)
        self._frames.append(self._renderer.render())

    def Close(self) -> None:
        """Write the video if an ``output_path`` is configured, then free the context.

        Idempotent: closing twice is a no-op, so a driver can close in a
        ``finally`` without tracking whether it already has.
        """
        if self._closed:
            return
        if self.config.output_path is not None:
            self.Save(self.config.output_path)
        self._renderer.close()
        self._closed = True

    # -- VideoRendererBase interface ----------------------------------------
    def Save(self, path: str | Path | None = None) -> str | None:
        """Write the captured frames.

        Args:
            path: Destination. ``None`` uses ``config.output_path``. The
                extension selects the container (``.mp4`` or ``.gif``); a
                missing one becomes ``.mp4``.

        Returns:
            The path actually written — which may differ from ``path`` once the
            extension is normalized — or ``None`` when there is nothing to
            write. An empty video is not an error: an episode that ended before
            the first frame deadline legitimately captured nothing. (The plan
            writes this as returning None; the path is returned as well because
            callers log it, and ignoring a return value costs nothing.)

        Writing at ``config.fps`` replays in real time, since that is the rate
        the frames were meant to be captured at.
        """
        target = path if path is not None else self.config.output_path
        if target is None:
            raise ValueError("no path given and config.output_path is None")
        if not self._frames:
            return None

        import mediapy as media

        out = self._checkContainer(target)
        out.parent.mkdir(parents=True, exist_ok=True)
        # mediapy defaults to h264 and does not infer the codec from the
        # extension, so the GIF muxer would reject an h264-encoded .gif.
        kwargs = {"codec": "gif"} if out.suffix.lower() == ".gif" else {}
        media.write_video(str(out), self._frames, fps=float(self.config.fps), **kwargs)
        return str(out)

    def Reset(self) -> None:
        """Drop the captured frames, keeping the renderer open for another episode."""
        self._frames = []

    @property
    def frame_count(self) -> int:
        """Frames captured since construction or the last ``Reset``."""
        return len(self._frames)
