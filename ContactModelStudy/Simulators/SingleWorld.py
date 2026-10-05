"""A one-world ``VectorizedSimulator`` presented as a plain ``Simulator``.

What lets a GPU contact model (M1-M4) be the *eval* simulator: the episode loop,
the tasks' success tests, the recorder and the renderer all speak the
``Simulator`` interface, with un-batched shapes. ``SingleWorld`` wraps a
vectorized simulator built with ``N=1`` and strips the world axis on the way
out. Because it is not a ``VectorizedSimulator``, a task judges it on the host
(``isSuccess`` returns a bool), exactly as it judges CPU MuJoCo.

Stepping replays CUDA graphs. An eval control step is tens of fine physics
steps, each dozens of kernel launches, and launching them one by one costs more
than the physics. ``Step(n)`` splits ``n`` into power-of-two blocks (at most
``MAX_BLOCK``), captures each block size once, and replays it after. So only a
handful of graphs ever exist, whatever step counts the settle, the control step
and the renderer's frame deadlines ask for.
"""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

from ContactModelStudy.Simulators.Simulator import Simulator
from ContactModelStudy.Simulators.VectorizedSimulator import VectorizedSimulator

#: Largest block of steps captured into one graph.
MAX_BLOCK = 256


class SingleWorld(Simulator):
    """``Simulator`` view of a ``VectorizedSimulator`` with one world.

    Attributes:
        inner: The wrapped simulator. Its config is this one's ``config``.
        class_name: The wrapped simulator's class, which is what a recorder
            reports (see ``EpisodeRecorder._className``).
    """

    def __init__(self, vec_sim: VectorizedSimulator, use_graph: bool = True):
        """Wrap ``vec_sim``.

        Args:
            vec_sim: A vectorized simulator built with ``N=1``.
            use_graph: Replay steps from CUDA graphs. Off steps eagerly, which
                is slower but identical; for debugging.

        Raises:
            ValueError: If ``vec_sim`` has more than one world.
        """
        if vec_sim.N != 1:
            raise ValueError(f"SingleWorld needs a simulator with N=1, got N={vec_sim.N}")
        # No Simulator.__init__: the model is already loaded, so share its source.
        self.inner = vec_sim
        self.config = vec_sim.config
        self.model_path, self.model_xml = vec_sim.model_path, vec_sim.model_xml
        self.class_name = type(vec_sim).__name__
        self.use_graph = use_graph
        self._graphs: dict[int, object] = {}
        self._graph_failed = False
        vec_sim.ClearControlSequence()

    # -- state ---------------------------------------------------------------
    def SetState(self, q: np.ndarray, q_dot: np.ndarray | None = None) -> None:
        self.inner.SetState(q, q_dot)

    def GetState(self) -> tuple[np.ndarray, np.ndarray]:
        q, q_dot = self.inner.GetState()
        return np.array(q[0], dtype=float), np.array(q_dot[0], dtype=float)

    def SetControl(self, u: np.ndarray) -> None:
        self.inner.SetControl(u)

    def GetControl(self) -> np.ndarray:
        return np.array(self.inner.GetControl()[0], dtype=float)

    # -- stepping ------------------------------------------------------------
    def Step(self, steps: int = 1) -> None:
        """Advance ``steps`` physics steps, replaying captured graphs."""
        steps = int(steps)
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")
        if not self.use_graph or self._graph_failed:
            self.inner.Step_GPU(steps)
            return
        while steps > 0:
            block = min(MAX_BLOCK, 1 << (steps.bit_length() - 1))
            self._stepBlock(block)
            steps -= block

    def _stepBlock(self, k: int) -> None:
        graph = self._graphs.get(k)
        if graph is not None:
            wp.capture_launch(graph)
            return
        # The first block of this size is the real step, taken eagerly; it also
        # compiles kernels and lets MJWarp allocate. Capture records the same
        # launches without running them, for every later block of this size.
        self.inner.Step_GPU(k)
        wp.synchronize()
        try:
            with wp.ScopedCapture() as capture:
                self.inner.Step_GPU(k)
            self._graphs[k] = capture.graph
        except Exception as exc:  # noqa: BLE001 - any capture failure degrades the same way
            self._graph_failed = True
            warnings.warn(f"CUDA graph capture failed ({exc}); {self.class_name} eval steps "
                          f"fall back to eager launches (slower, identical).", RuntimeWarning, stacklevel=3)

    # -- dimensions ----------------------------------------------------------
    @property
    def nq(self) -> int:
        return self.inner.nq

    @property
    def nv(self) -> int:
        return self.inner.nv

    @property
    def nu(self) -> int:
        return self.inner.nu

    @property
    def timestep(self) -> float:
        return self.inner.timestep

    def Close(self) -> None:
        self._graphs.clear()
        self.inner.Close()

    def __repr__(self) -> str:
        return f"SingleWorld({self.inner!r})"
