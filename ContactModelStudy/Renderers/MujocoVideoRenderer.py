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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

from ContactModelStudy.Renderers.RendererBase import RendererBase, RendererBaseConfig

# Containers mediapy can mux. Anything else is rejected up front, rather than
# after an episode's worth of frames has already been rendered.
_CONTAINERS = (".mp4", ".gif")


@dataclass
class MujocoVideoRendererConfig(RendererBaseConfig):
    """Video-output parameters.

    Inherits ``width``, ``height``, ``fps`` and ``camera`` from
    ``RendererBaseConfig``, and adds where the video goes.

    Attributes:
        output_path: File to write on ``Close``. ``None`` keeps the frames in
            memory for an explicit ``Save(path)``. The extension picks the
            container (``.mp4`` or ``.gif``); a path without one gets ``.mp4``.
    """

    output_path: str | None = None


class MujocoVideoRenderer(RendererBase):
    """Renders a task's scene offscreen with MuJoCo and writes a video.

    Attributes:
        mjm: The ``MjModel`` this renderer draws with. Its own, compiled from
            the task's MJCF — not the simulator's, so the renderer stays
            usable with backends that have no ``MjModel`` at all.
        mjd: The ``MjData`` used to place the scene. Holds only positions;
            nothing here is ever stepped.
    """

    def __init__(self, task, config: MujocoVideoRendererConfig | None = None):
        """Compile the task's scene and open an offscreen renderer.

        Args:
            task: A task exposing ``getModelPath()``, or a path to an MJCF file.
            config: Video parameters. Defaults to
                ``MujocoVideoRendererConfig()``.

        Raises:
            FileNotFoundError: If the task's model file does not exist.
            ValueError: If ``config.camera`` names a camera the scene does not
                define, or ``output_path`` has an unsupported extension.
        """
        super().__init__(task, config if config is not None else MujocoVideoRendererConfig())

        if self.config.output_path is not None:
            self._check_container(self.config.output_path)

        self.mjm = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.mjd = mujoco.MjData(self.mjm)
        self._camera = self._resolve_camera(self.config.camera)
        self._renderer = mujoco.Renderer(
            self.mjm, height=self.config.height, width=self.config.width
        )
        self._frames: list[np.ndarray] = []
        self._closed = False

    # -- setup ---------------------------------------------------------------
    def _resolve_camera(self, name: str | None) -> str | int:
        """Return a camera argument for ``update_scene``: a name, or -1 for free.

        Checks the name against the model now rather than letting
        ``update_scene`` fail on the first frame, and names the cameras the
        scene does define — MuJoCo's own error does not.
        """
        if name is None:
            return -1
        if mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
            available = [
                mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, i)
                for i in range(self.mjm.ncam)
            ]
            raise ValueError(
                f"camera {name!r} is not defined in {self.model_path.name}; "
                f"available: {available or 'none (use camera=None for the free camera)'}"
            )
        return name

    @staticmethod
    def _check_container(path: str | Path) -> Path:
        """Normalize an output path, defaulting a missing extension to .mp4."""
        p = Path(path)
        if p.suffix == "":
            p = p.with_suffix(".mp4")
        if p.suffix.lower() not in _CONTAINERS:
            raise ValueError(
                f"output must be one of {_CONTAINERS}, got {p.suffix!r} in {path!r}"
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

    # -- video ---------------------------------------------------------------
    def Save(self, path: str | Path | None = None) -> str | None:
        """Write the captured frames to ``path``.

        Args:
            path: Destination. Defaults to ``config.output_path``. The
                extension selects the container; a missing one becomes ``.mp4``.

        Returns:
            The path actually written — which may differ from ``path`` once the
            extension is normalized — or ``None`` when there is nothing to
            write. An empty video is not an error: an episode that ended before
            the first frame deadline legitimately captured nothing.

        Writing at ``config.fps`` replays in real time, since that is the rate
        the frames were meant to be captured at.
        """
        target = path if path is not None else self.config.output_path
        if target is None:
            raise ValueError("no path given and config.output_path is None")
        if not self._frames:
            return None

        import mediapy as media

        out = self._check_container(target)
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
