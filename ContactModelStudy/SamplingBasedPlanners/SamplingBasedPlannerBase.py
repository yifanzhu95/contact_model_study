"""Shared engine for sampling-based predictive control.

Every planner here runs the same loop::

    x0   <- measured state
    for i = 1..N:  U^(i) ~ pi_theta(U);  J^(i) <- J(U^(i), x0)
    theta <- update(U^(1:N), J^(1:N))
    u     <- action(theta)

The N candidate trajectories are rolled out in parallel as the N worlds of a
``VectorizedSimulator``. A concrete planner supplies only two hooks —
``_buildSamples`` (draw ``V`` from ``theta``) and ``_updateParams`` (fold the
costs back into ``theta``) — so comparing planners holds the rollout, the cost
and the control parameterization fixed.

The big structural change from the old ``contact_study.planners.base``: the
planner no longer owns the physics. It used to call ``api.put_model`` and
``api.make_data`` itself and hold ``self.m``/``self.d``; now it is handed a
simulator and a task and drives them through their public interfaces. Swapping
the contact model under a planner is a matter of constructing a different
simulator.

Everything stays on the device between plans. The sampled controls are uploaded
device-to-device, the costs are evaluated by the task's GPU kernel against the
simulator's own arrays, and only a handful of scalars and the ``(H, nu)`` mean
ever cross to the host.
"""

from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass

import numpy as np
import warp as wp

from ContactModelStudy.Simulators.VectorizedSimulator import VectorizedSimulator

_CONTROL_MODES = ("absolute", "relative")


@dataclass
class SamplingBasedPlannerConfig:
    """Configuration shared by every sampling-based planner.

    Deliberately small. ``N``, ``H``, ``nu``, ``substeps`` and the device are
    *not* here — they are read off the simulator the planner is given, because
    duplicating them invites a config and a simulator that disagree about the
    shape of the same buffer.

    Attributes:
        noise_sigma: Standard deviation of the Gaussian perturbations.
        n_iterations: Optimizer iterations per ``Plan`` call.
        warm_start: Shift the mean one control step forward after each plan, so
            the next call starts from the tail of the last solution.
        control_mode: How a sample becomes an actuator command.
            ``"relative"`` (the study's default) treats the sequence as a delta
            on the measured robot joint positions: ``ctrl = q_robot + U``.
            ``"absolute"`` sends ``U`` straight through. See ``Plan`` for how
            this differs from the old per-step formulation.
        delta_range: ``(low, high)`` clip applied to every sampled value before
            it becomes a command. Either side may be ``None`` to leave it
            unbounded. Off by default: under ``"relative"`` the value *is* the
            position-servo error, so clipping it caps how hard the hand can
            grip.
        seed: Seed for the noise. ``None`` draws one from fresh entropy.
        use_graph: Capture the rollout into a CUDA graph. This is most of the
            planner's speed — a rollout is thousands of small kernel launches,
            and replaying one recorded graph removes nearly all of that
            overhead. Falls back to eager launches, with a warning, if the
            capture cannot be made.
        debug: Print per-plan diagnostics.
    """

    noise_sigma: float = 0.01
    n_iterations: int = 1
    warm_start: bool = False
    control_mode: str = "relative"
    delta_range: tuple[float | None, float | None] = (None, None)
    seed: int | None = None
    use_graph: bool = True
    debug: bool = False

    def __post_init__(self) -> None:
        if self.noise_sigma <= 0.0:
            raise ValueError(f"noise_sigma must be positive, got {self.noise_sigma}")
        if self.n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {self.n_iterations}")
        if self.control_mode not in _CONTROL_MODES:
            raise ValueError(
                f"control_mode must be one of {_CONTROL_MODES}, got {self.control_mode!r}"
            )
        low, high = self.delta_range
        if low is not None and high is not None and low > high:
            raise ValueError(f"delta_range low > high: {self.delta_range}")


class SamplingBasedPlannerBase(abc.ABC):
    """Base class for all sampling-based planners.

    Subclasses implement ``_buildSamples`` and ``_updateParams``.

    Attributes:
        U_wp: ``(H, nu)`` mean control sequence — the distribution parameter
            ``theta`` that ``Plan`` optimizes.
        V_wp: ``(N, H, nu)`` sampled sequences drawn from it.
        A_wp: ``(N, H, nu)`` absolute commands handed to the simulator.
        costs_wp: ``(N,)`` running-cost sums.
        terminal_costs_wp: ``(N,)`` terminal costs, kept separate so the running
            sum can be horizon-normalized without also scaling a single-step
            terminal cost.
    """

    name = "SamplingBasedPlanner"

    def __init__(
        self,
        simulator: VectorizedSimulator,
        task,
        config: SamplingBasedPlannerConfig | None = None,
    ):
        """Bind a planner to a simulator and a task.

        Args:
            simulator: The rollout engine. Its world count is the sample count
                and its horizon is the planning horizon — the planner never
                allocates worlds of its own.
            task: Supplies ``calcCosts_GPU``, and (for ``"relative"`` control)
                the robot-joint start address via its ``indices`` vector.
            config: Planner parameters. Defaults to
                ``SamplingBasedPlannerConfig()``.

        Raises:
            ValueError: If the simulator's horizon is 1 while warm starting,
                which would shift the whole sequence off the end every plan.
        """
        self.sim = simulator
        self.task = task
        self.config = config if config is not None else SamplingBasedPlannerConfig()

        # Shape comes from the simulator, never from the config.
        self.N = simulator.N
        self.horizon = simulator.horizon
        self.nu = simulator.nu
        self.nq = simulator.nq
        self.nv = simulator.nv
        self.substeps = simulator.config.substeps
        self.device = simulator.config.device

        if self.config.warm_start and self.horizon < 2:
            raise ValueError(
                f"warm_start needs a horizon of at least 2, got {self.horizon}"
            )

        # Where the robot's joints start in qpos, for the relative control mode.
        # Slot 2 of the task's index vector, the same convention the old planner
        # used; 0 (robot joints lead qpos) when the task does not publish one.
        self.robot_qpos_adr = 0
        idx = getattr(task, "indices", None)
        if idx is not None and len(idx) > 2:
            self.robot_qpos_adr = int(idx[2])

        self._rng = np.random.default_rng(self.config.seed)
        self._noise_seed = int(self._rng.integers(0, 2**31 - 1))
        self._resample_count = 0

        # Diagnostics a driver or an evaluation pass can read after a plan.
        self.last_action_seq: np.ndarray | None = None
        self.last_plan_ok: bool = False

        self._setupArrays()

    # -- setup ---------------------------------------------------------------
    def _setupArrays(self) -> None:
        """Allocate every device buffer once, at construction."""
        N, H, nu = self.N, self.horizon, self.nu
        dev = self.device

        self.U_wp = wp.zeros((H, nu), dtype=wp.float32, device=dev)
        self.U_shift_wp = wp.zeros((H, nu), dtype=wp.float32, device=dev)
        self.V_wp = wp.zeros((N, H, nu), dtype=wp.float32, device=dev)
        self.A_wp = wp.zeros((N, H, nu), dtype=wp.float32, device=dev)
        self.eps_wp = wp.zeros((N, H, nu), dtype=wp.float32, device=dev)

        self.costs_wp = wp.zeros(N, dtype=wp.float32, device=dev)
        self.terminal_costs_wp = wp.zeros(N, dtype=wp.float32, device=dev)
        self._step_cost_wp = wp.zeros(N, dtype=wp.float32, device=dev)

        # Per-actuator clip bounds. A None side becomes +/-inf so the clamp is a
        # no-op there; when both are None the kernel skips clamping entirely.
        low, high = self.config.delta_range
        rng = np.empty((nu, 2), dtype=np.float32)
        rng[:, 0] = -np.inf if low is None else low
        rng[:, 1] = np.inf if high is None else high
        self._clip_wp = wp.array(rng, dtype=wp.float32, device=dev)
        self._has_clip = not (low is None and high is None)

        # Added to every sample to form the absolute command. Stays zero in
        # "absolute" mode; refilled from the measured state each plan in
        # "relative" mode.
        self._offset_wp = wp.zeros(nu, dtype=wp.float32, device=dev)

        # The rollout's start state, staged on the device. The host-to-device
        # copy into these happens once per plan, outside the graph; the graph
        # itself broadcasts from them.
        self._q0_wp = wp.zeros(self.nq, dtype=wp.float32, device=dev)
        self._v0_wp = wp.zeros(self.nv, dtype=wp.float32, device=dev)
        self._rollout_graph = None
        self._graph_failed = False

    # -- planner-specific hooks ---------------------------------------------
    @abc.abstractmethod
    def _buildSamples(self) -> None:
        """Fill ``V_wp`` with ``N`` sequences drawn from the current ``theta``.

        Reads ``U_wp`` and ``eps_wp``; must apply ``delta_range`` clipping, so
        that whatever the update does with ``V`` inherits the bound.
        """
        ...

    @abc.abstractmethod
    def _updateParams(self) -> bool:
        """Fold ``costs_wp``/``terminal_costs_wp`` into ``theta``.

        Returns:
            ``False`` when the update could not be made — every rollout cost was
            NaN, say — which makes ``Plan`` return a zero action rather than one
            derived from a poisoned distribution.
        """
        ...

    def _resetParams(self) -> None:
        """Restore any adaptive state. No-op unless a subclass has some."""
        return None

    # -- the loop ------------------------------------------------------------
    def Plan(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """Optimize from the measured state and return the next action.

        Args:
            q: Measured positions, ``(nq,)``.
            q_dot: Measured velocities, ``(nv,)``.

        Returns:
            ``(nu,)`` absolute actuator command — what to hand
            ``Simulator.SetControl``, already including the relative-mode
            offset, so a driver never has to know which parameterization the
            planner used.

        Note on ``"relative"`` mode: the offset is the robot's *measured joint
        positions at plan time*, held constant across the horizon. The old
        planner instead re-read each world's current qpos at every rollout step
        (``_assign_ctrl_relative_kernel``). The two agree exactly at the first
        control step — which is the action actually applied — and diverge over
        the horizon, where the old form tracks each world's drifting pose and
        this one does not. Expressing the per-step version would mean reaching
        past ``SetControlSequence`` into the backend's own arrays, which is the
        coupling this refactor removes.
        """
        q = np.asarray(q, dtype=float).ravel()
        q_dot = np.asarray(q_dot, dtype=float).ravel()
        if q.shape != (self.nq,):
            raise ValueError(f"q must have shape ({self.nq},), got {q.shape}")
        if q_dot.shape != (self.nv,):
            raise ValueError(f"q_dot must have shape ({self.nv},), got {q_dot.shape}")

        self._setOffset(q)
        self._sampleNoise()
        self.last_plan_ok = False

        for _ in range(self.config.n_iterations):
            self._buildSamples()
            self._buildCommands()
            self._rollout(q, q_dot)
            if not self._updateParams():
                # The update may have written NaN into theta; clear it so the
                # next call starts clean, and publish a zero tape — a driver
                # replaying the previous one would be acting on a failed solve.
                self.U_wp.zero_()
                self.last_action_seq = np.zeros((self.horizon, self.nu), dtype=np.float32)
                return np.zeros(self.nu, dtype=np.float32)

        self.last_plan_ok = True
        return self._extractAction()

    def _setOffset(self, q: np.ndarray) -> None:
        """Set the command offset from the measured state for this plan."""
        if self.config.control_mode == "absolute":
            return
        a = self.robot_qpos_adr
        self._offset_wp.assign(q[a:a + self.nu].astype(np.float32))

    def _sampleNoise(self) -> None:
        """Redraw the whole ``(N, H, nu)`` perturbation block on the device.

        The seed advances every call, so no two draws in this planner's lifetime
        replay the same block, and a fixed ``config.seed`` reproduces the whole
        episode.
        """
        wp.launch(
            _sample_noise_kernel,
            dim=(self.N, self.horizon, self.nu),
            inputs=[self._noise_seed + self._resample_count, self.config.noise_sigma],
            outputs=[self.eps_wp],
        )
        self._resample_count += 1

    def _buildCommands(self) -> None:
        """``A = V + offset``: turn sampled values into absolute commands."""
        wp.launch(
            _add_offset_kernel,
            dim=(self.N, self.horizon, self.nu),
            inputs=[self.V_wp, self._offset_wp],
            outputs=[self.A_wp],
        )

    def _rollout(self, q: np.ndarray, q_dot: np.ndarray) -> None:
        """Roll all ``N`` sampled sequences out in parallel and score them.

        Stages the start state on the device, then replays the captured rollout
        graph — or runs it eagerly when capture is off or unavailable.
        """
        self._q0_wp.assign(q.astype(np.float32))
        self._v0_wp.assign(q_dot.astype(np.float32))
        self.sim.SetControlSequence(self.A_wp)

        if not self.config.use_graph or self._graph_failed:
            self._rolloutBody()
            return
        if self._rollout_graph is None:
            self._captureRollout()
        wp.capture_launch(self._rollout_graph)

    def _rolloutBody(self) -> None:
        """The rollout itself: reset, unroll, accumulate. Kernel launches only.

        Nothing here may touch the host — no ``.assign`` from numpy, no
        ``.numpy()``, no synchronize — because this is the body that gets
        recorded into a CUDA graph.

        The cost of the final control step goes to ``terminal_costs_wp`` instead
        of the running sum, not in addition to it, matching the old
        ``_launch_accumulate_costs(terminal=...)``, where the terminal cost
        *replaces* that step's running cost.
        """
        self.sim.BroadcastState(self._q0_wp, self._v0_wp)
        self.costs_wp.zero_()
        self.terminal_costs_wp.zero_()

        state = self.sim.DeviceState()
        for t in range(self.horizon):
            self.sim.Step_GPU(self.substeps)
            terminal = t == self.horizon - 1
            self.task.calcCosts_GPU(
                state.qpos, state.qvel, state.ctrl,
                terminal=terminal, out=self._step_cost_wp,
                site_xpos=state.site_xpos, device=self.device,
            )
            wp.launch(
                _accumulate_kernel, dim=self.N,
                inputs=[self._step_cost_wp],
                outputs=[self.terminal_costs_wp if terminal else self.costs_wp],
            )

    def _captureRollout(self) -> None:
        """Record one rollout into a CUDA graph, after a warm-up pass.

        The warm-up matters: the first call compiles kernels and lets MJWarp
        make whatever internal allocations it needs, neither of which belongs in
        a recording. The control sequence is re-set afterwards so the capture
        starts from cursor 0 and bakes in control indices 0..H-1 — which is what
        an unrolled rollout wants, and why the captured graph stays correct when
        replayed against new contents of the same buffers.

        A capture failure is not fatal: it disables graphs for this planner and
        warns, rather than taking the run down.
        """
        self._rolloutBody()
        wp.synchronize()
        self.sim.SetControlSequence(self.A_wp)
        try:
            with wp.ScopedCapture() as capture:
                self._rolloutBody()
            self._rollout_graph = capture.graph
        except Exception as exc:  # noqa: BLE001 - any capture failure degrades the same way
            self._graph_failed = True
            warnings.warn(
                f"CUDA graph capture failed ({exc}); falling back to eager "
                f"launches. Rollouts will be slower but identical.",
                RuntimeWarning, stacklevel=2,
            )
            self._rolloutBody()

    def _foldCosts(self, running_scale: float = 1.0, total_scale: float = 1.0) -> None:
        """Add the terminal cost into the running sum and rescale, in place."""
        wp.launch(
            _combine_costs_kernel, dim=self.N,
            inputs=[self.costs_wp, self.terminal_costs_wp, running_scale, total_scale],
        )

    def _extractAction(self) -> np.ndarray:
        """Read row 0 of the mean, apply the warm-start shift, return the action.

        The whole ``(H, nu)`` mean comes across — a few hundred bytes, one
        transfer either way — and is published as ``last_action_seq`` *before*
        the shift, so a driver that plays a tape out over a latency window gets
        a sequence aligned with the state it was planned from.
        """
        seq = self.U_wp.numpy().copy()
        offset = self._offset_wp.numpy()
        self.last_action_seq = seq + offset
        action = self.last_action_seq[0].copy()

        if self.config.warm_start:
            wp.launch(
                _shift_kernel, dim=(self.horizon, self.nu),
                inputs=[self.U_wp, self.horizon, 1],
                outputs=[self.U_shift_wp],
            )
            self.U_wp, self.U_shift_wp = self.U_shift_wp, self.U_wp
        return action

    # -- episode boundaries --------------------------------------------------
    def Reset(self, mean: np.ndarray | None = None) -> None:
        """Reset the distribution. Call at the start of an episode or new goal.

        Args:
            mean: ``(nu,)`` command to seed every row of the mean sequence with.
                ``None`` zeroes it.

        Zero is the right seed under ``"relative"`` control — the sequence is a
        delta, so a zero mean means "hold the measured pose". Under
        ``"absolute"`` it is not: a zero mean commands every joint to zero, and
        on a grasping hand that opens it and drops the object on the first plan.
        Seed it with the task's initial control there.
        """
        if mean is None:
            self.U_wp.zero_()
        else:
            m = np.asarray(mean, dtype=np.float32).ravel()
            if m.shape != (self.nu,):
                raise ValueError(f"mean must have shape ({self.nu},), got {m.shape}")
            self.U_wp.assign(np.tile(m, (self.horizon, 1)))
        self.last_action_seq = None
        self.last_plan_ok = False
        self._resetParams()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(N={self.N}, H={self.horizon}, nu={self.nu}, "
            f"substeps={self.substeps}, mode={self.config.control_mode!r})"
        )


# ---------------------------------------------------------------------------
# Kernels shared by every sampling-based planner
# ---------------------------------------------------------------------------

@wp.kernel
def _sample_noise_kernel(seed: int, sigma: float, eps: wp.array3d(dtype=float)):
    """Redraw the perturbation block in place.

    Each element gets its own RNG stream keyed by (seed, flat index), so a draw
    is reproducible for a given seed and decorrelated across seeds.
    """
    n, h, u = wp.tid()
    tid = (n * eps.shape[1] + h) * eps.shape[2] + u
    state = wp.rand_init(seed, tid)
    eps[n, h, u] = sigma * wp.randn(state)


@wp.kernel
def _add_offset_kernel(
    V: wp.array3d(dtype=float),       # (N, H, nu)
    offset: wp.array(dtype=float),    # (nu,)
    out: wp.array3d(dtype=float),     # (N, H, nu)  [out]
):
    """Absolute command from a sampled value: out = V + offset."""
    n, h, u = wp.tid()
    out[n, h, u] = V[n, h, u] + offset[u]


@wp.kernel
def _accumulate_kernel(src: wp.array(dtype=float), dst: wp.array(dtype=float)):
    """dst += src, one element per sample."""
    n = wp.tid()
    dst[n] = dst[n] + src[n]


@wp.kernel
def _combine_costs_kernel(
    running: wp.array(dtype=float),    # (N,)  [in/out] becomes the total
    terminal: wp.array(dtype=float),   # (N,)
    running_scale: float,
    total_scale: float,
):
    """Fold the terminal cost into the running sum and rescale, in place."""
    n = wp.tid()
    running[n] = (running[n] + terminal[n]) * running_scale * total_scale


@wp.kernel
def _shift_kernel(
    src: wp.array2d(dtype=float),   # (H, nu)
    H: int,
    shift: int,
    dst: wp.array2d(dtype=float),   # (H, nu)  [out]
):
    """Warm start: dst[h] = src[h + shift], zeroing the rows that fall off."""
    h, u = wp.tid()
    s = h + shift
    if s < H:
        dst[h, u] = src[s, u]
    else:
        dst[h, u] = float(0.0)
