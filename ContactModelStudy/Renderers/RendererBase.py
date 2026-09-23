"""Base renderer interfaces for the contact-model study.

A renderer draws a *state*, not a simulator. It owns its own model, built from
the task's MJCF, and is handed positions to draw::

    renderer.RenderState(q)

That keeps the simulators free of any rendering code — a ``Simulator`` has no
renderer, captures no frames and takes no camera arguments — and means the same
renderer can visualize any backend, including ones with no graphics of their own
(Pinocchio, XPBD) and one world sampled out of a vectorized batch.

Only positions are passed because only positions move geometry. Velocities
affect nothing the camera can see.

Four classes, in two layers:

  * ``RendererBase`` / ``RendererBaseConfig`` — anything that draws a state;
  * ``VideoRendererBase`` / ``VideoRendererBaseConfig`` — those that accumulate
    frames and write a file.

An interactive viewer subclasses the first pair; an offscreen recorder the
second.
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RendererBaseConfig:
    """Camera parameters shared by every renderer.

    Every camera field defaults to ``None``, meaning "use what the scene's MJCF
    declares". That way a config can override one aspect of a camera — nudge the
    field of view, say — without having to restate a pose that the scene already
    gets right.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second. For a video renderer this is the target rate
            the frames are captured and written at; for an interactive viewer
            it is an upper bound on how often the window is refreshed.
        cam_name: Name of a camera defined in the scene's MJCF (the leap scenes
            define ``"demo-cam"``). ``None`` uses the scene's default view — a
            free camera framed on the model.
        cam_pos: ``(3,)`` camera position in world coordinates, or ``None`` to
            keep the scene's.
        cam_quat: ``(4,)`` camera orientation as a wxyz quaternion, or ``None``
            to keep the scene's. MuJoCo's camera convention: the camera looks
            along its own -Z axis with +Y up.
        cam_fovy: Vertical field of view in degrees, or ``None`` to keep the
            scene's.
    """

    width: int = 640
    height: int = 480
    fps: float = 30.0
    cam_name: str | None = None
    cam_pos: np.ndarray | None = None
    cam_quat: np.ndarray | None = None
    cam_fovy: float | None = None

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError(f"width/height must be >= 1, got {self.width}x{self.height}")
        if self.fps <= 0.0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        self.cam_pos = self._vector(self.cam_pos, 3, "cam_pos")
        self.cam_quat = self._vector(self.cam_quat, 4, "cam_quat")
        if self.cam_fovy is not None:
            self.cam_fovy = float(self.cam_fovy)
            if not 0.0 < self.cam_fovy < 180.0:
                raise ValueError(f"cam_fovy must be in (0, 180) degrees, got {self.cam_fovy}")

    @staticmethod
    def _vector(value, n: int, name: str) -> np.ndarray | None:
        """Coerce an optional camera vector to ``(n,)`` floats."""
        if value is None:
            return None
        a = np.asarray(value, dtype=float).ravel()
        if a.shape != (n,):
            raise ValueError(f"{name} must have {n} elements, got {a.shape[0]}")
        return a

    @property
    def has_camera_override(self) -> bool:
        """Whether any camera pose field departs from the scene's own."""
        return (self.cam_pos is not None
                or self.cam_quat is not None
                or self.cam_fovy is not None)


@dataclass
class VideoRendererBaseConfig(RendererBaseConfig):
    """Camera parameters plus where the video goes.

    Attributes:
        output_path: File to write when ``Save`` or ``Close`` is called without
            a path. ``None`` keeps the frames in memory until an explicit
            ``Save(path)``.
    """

    output_path: str | None = None


class RendererBase(abc.ABC):
    """Base class for all renderers.

    Subclasses implement ``RenderState`` and ``Close``, and set ``CONFIG_CLASS``
    if they need a richer config than ``RendererBaseConfig``.
    """

    #: Config type this renderer expects. A config of a different type is
    #: up-converted field-by-field, so a task can hand out one plain
    #: RendererBaseConfig and have it work for any renderer.
    CONFIG_CLASS: type = RendererBaseConfig

    def __init__(self, task, config: RendererBaseConfig | None = None):
        """Prepare a renderer for a task's scene.

        Args:
            task: A task exposing ``getModelPath() -> str``. A path to an MJCF
                file is also accepted, which is convenient for rendering a scene
                that has no task attached.
            config: Camera parameters, used exactly as given — the caller is
                expected to have run ``task.alignRendererConfigWithTask`` on it
                already, and may have overridden the task's choices since, so
                they are not re-applied here. ``None`` builds a default config
                and has the task align that, so the task's camera is still used
                when the caller has no settings of its own.

        Raises:
            FileNotFoundError: If the resolved model path does not exist.
        """
        self.task = task
        self.config = self._resolveConfig(task, config)
        self.model_path = self._resolveTaskModel(task)

    # -- construction helpers ------------------------------------------------
    @classmethod
    def _resolveConfig(cls, task, config: RendererBaseConfig | None) -> RendererBaseConfig:
        """Get a config of this renderer's type.

        With none given, a default one is built and handed to the task's
        ``alignRendererConfigWithTask``, if it has one — a bare MJCF path does
        not, and simply gets the defaults.
        """
        if config is None:
            config = cls.CONFIG_CLASS()
            align = getattr(task, "alignRendererConfigWithTask", None)
            if align is not None:
                align(config)
        return cls._coerceConfig(config)

    @classmethod
    def _coerceConfig(cls, config: RendererBaseConfig) -> RendererBaseConfig:
        """Up-convert a base config to this renderer's config type.

        A caller may build a plain ``RendererBaseConfig`` and hand it to a video
        renderer, which also needs an output path. Rather than making every
        caller know which renderer will consume the config, the shared fields
        are copied across and the extra ones take their defaults.
        """
        if isinstance(config, cls.CONFIG_CLASS):
            return config
        if not isinstance(config, RendererBaseConfig):
            raise TypeError(
                f"config must be a RendererBaseConfig, got {type(config).__name__}"
            )
        shared = {f.name for f in dataclasses.fields(cls.CONFIG_CLASS)}
        values = {k: v for k, v in dataclasses.asdict(config).items() if k in shared}
        return cls.CONFIG_CLASS(**values)

    @staticmethod
    def _resolveTaskModel(task) -> Path:
        """Return the MJCF path for ``task``, which may be a task or a path.

        Duck-typed on ``getModelPath`` rather than isinstance-checked against
        ``TaskBase``, so ``Renderers`` does not have to import ``Tasks`` — the
        two are independent halves of the package and a renderer only ever needs
        the one string.
        """
        source = task.getModelPath() if hasattr(task, "getModelPath") else task
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Task model file not found: {path}")
        return path

    # -- interface -----------------------------------------------------------
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
            f"{c.width}x{c.height}@{c.fps}fps, cam={c.cam_name!r})"
        )


class VideoRendererBase(RendererBase):
    """Base class for renderers that accumulate frames and write a video."""

    CONFIG_CLASS: type = VideoRendererBaseConfig

    def getStepsPerFrame(self) -> int:
        """Simulator steps between captures, to best respect ``config.fps``.

        Reads the task's timestep, so a renderer configured for 30 fps captures
        at 30 fps of *simulated* time whatever rate the caller happens to drive
        the loop at. Capturing on this cadence — rather than once per control
        step — is what makes playback speed independent of the control
        frequency, and bounds the frame count to ``fps`` per simulated second::

            every = renderer.getStepsPerFrame()
            for i in range(n_steps):
                sim.Step()
                if i % every == 0:
                    renderer.RenderState(sim.GetState()[0])

        Returns at least 1, so an ``fps`` faster than the simulator can resolve
        captures every step instead of asking for frames that do not exist.

        The unit is one step of the *task's* timestep. A caller advancing the
        simulator several steps at a time divides by that stride.

        Raises:
            AttributeError: If the task does not expose a ``timestep``.
        """
        timestep = getattr(self.task, "timestep", None)
        if timestep is None:
            raise AttributeError(
                f"{type(self.task).__name__} has no timestep, so frames cannot be "
                f"scheduled; give the task one or capture on your own cadence."
            )
        timestep = float(timestep)
        if timestep <= 0.0:
            raise ValueError(f"task timestep must be positive, got {timestep}")
        return max(1, round(1.0 / (self.config.fps * timestep)))

    @abc.abstractmethod
    def Save(self, path: str | Path | None = None):
        """Write the captured frames.

        Args:
            path: Destination. ``None`` uses ``config.output_path``.
        """
        ...

    @abc.abstractmethod
    def Reset(self) -> None:
        """Drop the captured frames, keeping the renderer open for another episode."""
        ...
