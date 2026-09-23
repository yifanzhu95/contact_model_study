"""Base class for every task in the study.

A task is the problem statement: which scene to load, where the system starts,
what a trajectory through it costs, and what counts as winning or losing. It
owns no simulator and no renderer — it is handed a simulator and reads what it
needs from it — so the same task object scores a GPU rollout batch or judges a
CPU evaluation episode.

The cost exists only on the device. ``calcCosts`` takes a vectorized simulator
and scores every world in place; there is deliberately no host twin to keep in
sync with it. Episode evaluation on a CPU simulator uses ``isSuccess`` and
``isFailure`` instead, which accept either kind of simulator.
"""

from __future__ import annotations

import abc
import enum
from pathlib import Path

import numpy as np


class TaskRole(str, enum.Enum):
    """Which scene a task should load.

    The study runs two models of the same physical setup: a high-fidelity
    ``EVAL`` scene standing in for reality, and a cheaper ``ROLLOUT`` scene the
    planner predicts with. The gap between them is the thing being measured, so
    which one a task hands out is an explicit choice, never a default that
    quietly picks the wrong one.
    """

    EVAL = "eval"
    ROLLOUT = "rollout"


class TaskBase(abc.ABC):
    """Base class for all tasks.

    Subclasses implement ``getModelPath``, ``getInitialState``, ``calcCosts``,
    ``isFailure``, ``isSuccess`` and the three goal functions (``sampleNewGoal``,
    ``setGoal``, ``setRendererToGoal``), and override
    ``alignRendererConfigWithTask`` when their scenes have a camera worth using.
    ``setSimToInitialState`` is provided here, on top of ``getInitialState``.
    """

    def __init__(self, role: TaskRole | str = TaskRole.EVAL, timestep: float = 0.002):
        """Create a task bound to one scene role.

        Args:
            role: ``TaskRole.EVAL`` or ``TaskRole.ROLLOUT``, or the equivalent
                string. Decides which scene ``getModelPath`` returns.
            timestep: The physics timestep this task is posed at, in seconds.
                The task carries it because things other than a simulator need
                it — a video renderer schedules frames against it — and because
                a task's costs and initial state are only meaningful at the
                rate they were tuned for. A simulator is still free to run at a
                different one; this is the task's declared rate, not a
                constraint on the caller.
        """
        self.role = TaskRole(role)
        if timestep <= 0.0:
            raise ValueError(f"timestep must be positive, got {timestep}")
        self.timestep = float(timestep)
        # Device copies of the initial state, uploaded on first use and kept, so
        # later calls to setSimToInitialState allocate nothing and can be
        # recorded into a CUDA graph. Keyed by device.
        self._initial_state_buffers: dict = {}

    # -- scene ---------------------------------------------------------------
    @abc.abstractmethod
    def getModelPath(self) -> str:
        """Return the path to this task's MJCF model, for its role.

        A string rather than a ``Path``, per the plan, and because this is what
        gets handed to ``mujoco.MjModel.from_xml_path`` and to renderers.
        """
        ...

    def alignRendererConfigWithTask(self, config) -> None:
        """Fill in the renderer settings this task needs, in place.

        Called on a config *before* the renderer is built from it, so the camera
        a scene was authored around travels with the task rather than being
        restated at every call site::

            cfg = VideoRendererBaseConfig(width=640, height=480)
            task.alignRendererConfigWithTask(cfg)
            renderer = MujocoVideoRenderer(task, cfg)

        The task sets only the fields it has an opinion about and leaves the rest
        — resolution, fps, output path — as the caller made them. To override one
        of the task's choices, set it *after* aligning.

        The default has no opinions and changes nothing. Override it in a task
        whose scenes define a camera worth using.

        Args:
            config: A ``RendererBaseConfig`` or subclass, modified in place.
        """
        return None

    # -- initial conditions --------------------------------------------------
    @abc.abstractmethod
    def getInitialState(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(q0, q_dot0, u0)`` — where an episode starts.

        Not in the refactor plan's function list, but every driver needs it and
        every task has it, so it belongs on the interface rather than being
        reached for through a subclass-specific attribute. The control is part
        of the initial condition because a position-actuated system left at
        ``ctrl = 0`` collapses before the planner's first command lands.
        """
        ...

    def setSimToInitialState(self, sim) -> None:
        """Put ``sim`` in this task's initial state — positions, velocities, control.

        Args:
            sim: A ``Simulator`` or a ``VectorizedSimulator``.

        A single simulator gets ``SetState`` and ``SetControl``. A vectorized one
        gets ``BroadcastState`` from device-resident copies of the initial
        state, so the call is nothing but kernel launches and can be recorded
        into a CUDA graph. Those copies are uploaded on the first call for a
        given device; make one call before capturing, so the recording does not
        contain the upload.

        Implemented once, here, on top of ``getInitialState`` — a subclass
        defines where it starts, not how a simulator is put there.
        """
        from ContactModelStudy.Simulators.VectorizedSimulator import VectorizedSimulator

        self._onInitialState()
        q0, v0, u0 = self.getInitialState()
        if not isinstance(sim, VectorizedSimulator):
            sim.SetState(q0, v0)
            sim.SetControl(u0)
            return

        import warp as wp

        device = str(sim.config.device)
        bufs = self._initial_state_buffers.get(device)
        if bufs is None:
            bufs = tuple(
                wp.array(np.asarray(a, dtype=np.float32), dtype=wp.float32, device=device)
                for a in (q0, v0, u0)
            )
            self._initial_state_buffers[device] = bufs
        sim.BroadcastState(*bufs)

    def _onInitialState(self) -> None:
        """Hook for task-side state an episode reset should clear. No-op here.

        Must stay host-only — no device writes — so ``setSimToInitialState``
        remains capturable.
        """
        return None

    # -- goal ----------------------------------------------------------------
    #
    # Choosing a goal and applying it are separate steps, so one sampled goal
    # can be handed to several task instances — the planner's rollout task and
    # the evaluator's eval task must chase the same goal:
    #
    #     g = eval_task.sampleNewGoal()
    #     eval_task.setGoal(g)
    #     rollout_task.setGoal(g)
    #
    # The goal lives in the task. The cost and the success test read it from
    # there; setRendererToGoal only brings a renderer's *depiction* of it into
    # line. No simulator needs telling: nothing a simulator computes depends on
    # the goal.

    @abc.abstractmethod
    def sampleNewGoal(self, current_goal: np.ndarray | None = None) -> np.ndarray:
        """Draw a new goal, relative to the current one, and return it.

        Args:
            current_goal: The goal to sample from. ``None`` uses the task's own
                reference — the goal last adopted with ``setGoal``, or the
                starting goal right after ``setSimToInitialState``, when the
                object is back where it began.

        Does not adopt the result — call ``setGoal`` for that. Consumes
        randomness but leaves the task's goal, and everything that reads it,
        alone.
        """
        ...

    @abc.abstractmethod
    def setGoal(self, goal: np.ndarray) -> None:
        """Make ``goal`` the task's goal.

        The cost and the success test read it from then on. A task must update
        any device copy of the goal *in place*, never by reallocating, so a
        planner that has already captured its rollout into a CUDA graph picks
        the new goal up without re-capturing.
        """
        ...

    @abc.abstractmethod
    def setRendererToGoal(self, renderer) -> None:
        """Bring ``renderer``'s depiction of the goal in line, so frames show it.

        A renderer owns its own copy of the scene, separate from any
        simulator's, so this is what makes the goal visible in a video.
        """
        ...

    # -- cost ----------------------------------------------------------------
    @abc.abstractmethod
    def calcCosts(self, sim, terminal: bool = False, out=None):
        """Score every world of a vectorized simulator, on the device.

        Reads whatever the cost needs — state, actuation, site positions —
        straight from ``sim.DeviceState()``, so nothing crosses to the host.
        That is what lets a planner capture its whole rollout, cost included,
        into one CUDA graph.

        Args:
            sim: A ``VectorizedSimulator``. There is deliberately no CPU path:
                the cost is defined once, on the device, and never reproduced.
            terminal: Score with the terminal cost rather than the running one.
                Terminal cost *replaces* the running cost; it is not added.
            out: ``(N,)`` device array to write into. ``None`` uses a buffer the
                task keeps per world count — which is reused, and overwritten,
                by the next call. Pass one explicitly on the rollout path.

        Returns:
            The ``(N,)`` device array of per-world costs — ``out`` if given.
            ``.numpy()`` turns it into a numpy array. Subclasses that need more
            inputs to avoid recomputing something may widen this signature.
        """
        ...

    # -- outcome -------------------------------------------------------------
    @abc.abstractmethod
    def isFailure(self, sim, out=None):
        """Whether the simulator has failed the task.

        Args:
            sim: A ``Simulator`` or a ``VectorizedSimulator``.
            out: For a vectorized simulator, an ``(N,)`` boolean device array to
                write into; ``None`` uses a reused per-task buffer.

        Returns:
            A plain ``bool`` for a single simulator. For a vectorized one, an
            ``(N,)`` boolean device array — computed with kernel launches only,
            so it can be recorded into a CUDA graph.
        """
        ...

    @abc.abstractmethod
    def isSuccess(self, sim, out=None):
        """Whether the simulator has achieved the task.

        Same argument and return conventions as ``isFailure``.
        """
        ...

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _resolve_path(path: str | Path) -> str:
        """Check a scene file exists and return it as a string.

        Failing here names the missing scene; failing inside MuJoCo's parser
        reports a less obvious error further from the cause.
        """
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Task scene not found: {p}")
        return str(p)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(role={self.role.value!r})"
