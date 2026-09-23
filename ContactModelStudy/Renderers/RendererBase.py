"""Base renderer interface for the contact-model study.

A renderer draws a *state*, not a simulator. It owns its own model and its own
scene, built from the task's MJCF, and is handed positions to draw:

    renderer.RenderState(q)

That keeps the simulators free of any rendering code — a `Simulator` has no
renderer, captures no frames and takes no camera arguments — and means the same
renderer can visualize any backend, including ones with no graphics of their
own (Pinocchio, XPBD) and one world sampled out of a vectorized batch.

Only positions are passed because only positions move geometry. Velocities
affect nothing the camera can see.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RendererBaseConfig:
    """Camera and output parameters shared by every renderer.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second of *simulated* time. This is what a video is
            written at, and what ``steps_per_frame`` converts into a capture
            cadence, so playback runs in real time.
        camera: Name of a camera defined in the scene's MJCF (the leap scenes
            define ``"demo-cam"``). ``None`` uses the viewer's default free
            camera.
    """

    width: int = 640
    height: int = 480
    fps: float = 30.0
    camera: str | None = None

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError(f"width/height must be >= 1, got {self.width}x{self.height}")
        if self.fps <= 0.0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    def steps_per_frame(self, timestep: float) -> int:
        """How many simulator steps of ``timestep`` to run between frames.

        A renderer draws whenever it is called, so pacing is the caller's job::

            every = cfg.steps_per_frame(sim.timestep)
            for i in range(n_steps):
                sim.Step()
                if i % every == 0:
                    renderer.RenderState(sim.GetState()[0])

        Capturing on this cadence — rather than once per control step — is what
        makes playback speed independent of the control frequency, and bounds
        the frame count to ``fps`` per simulated second. Returns at least 1,
        so an ``fps`` faster than the simulator can resolve captures every step
        instead of asking for frames that do not exist.
        """
        if timestep <= 0.0:
            raise ValueError(f"timestep must be positive, got {timestep}")
        return max(1, round(1.0 / (self.fps * timestep)))


class RendererBase(abc.ABC):
    """Base class for all renderers.

    Subclasses implement ``RenderState`` and ``Close`` and are expected to call
    ``super().__init__(task, config)`` first, which resolves the task's model
    path onto ``self.model_path``.
    """

    def __init__(self, task, config: RendererBaseConfig | None = None):
        """Prepare a renderer for a task's scene.

        Args:
            task: A task exposing ``getModelPath() -> str``, per ``TaskBase``.
                A path to an MJCF file is also accepted, which is what makes
                this usable before the Tasks port lands and convenient for
                rendering a scene that has no task attached.
            config: Camera and output parameters. Defaults to
                ``RendererBaseConfig()``.

        Raises:
            FileNotFoundError: If the resolved model path does not exist.
        """
        self.task = task
        self.config = config if config is not None else RendererBaseConfig()
        self.model_path = self._resolve_task_model(task)

    @staticmethod
    def _resolve_task_model(task) -> Path:
        """Return the MJCF path for ``task``, which may be a task or a path.

        Duck-typed on ``getModelPath`` rather than isinstance-checked against
        ``TaskBase``, so ``Renderers`` does not have to import ``Tasks`` — the
        two are independent halves of the package and a renderer only ever
        needs the one string.
        """
        source = task.getModelPath() if hasattr(task, "getModelPath") else task
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Task model file not found: {path}")
        return path

    @abc.abstractmethod
    def RenderState(self, q: np.ndarray) -> None:
        """Draw the state ``q``.

        Args:
            q: Generalized positions, shape ``(nq,)``, in the MuJoCo qpos
                layout — exactly the first element of ``Simulator.GetState()``.

        What "draw" means is the subclass's business: appending a frame to a
        video buffer, or pushing the state to an on-screen viewer.
        """
        ...

    @abc.abstractmethod
    def Close(self) -> None:
        """End the render and release resources.

        Writes out any pending output and frees the graphics context. Safe to
        call more than once, so a driver can close in a ``finally`` without
        tracking whether it already did.
        """
        ...

    def __enter__(self) -> "RendererBase":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.Close()

    def __repr__(self) -> str:
        c = self.config
        return (
            f"{type(self).__name__}(model={self.model_path.name!r}, "
            f"{c.width}x{c.height}@{c.fps}fps, camera={c.camera!r})"
        )
