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

_CONTROL_MODES = ("absolute", "ctrl_relative", "pos_relative")


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
        control_mode: How a planned value ``U_t`` becomes the command at
            rollout control step ``t``:

            * ``"absolute"`` — ``ctrl_t = U_t``. Planning happens directly in
              the actuators' command space.
            * ``"ctrl_relative"`` — ``ctrl_t = ctrl_{t-1} + U_t``, starting from
              the control currently applied. The deltas accumulate over the
              horizon (the old planner's legacy ``ctrl += delta`` mode).
            * ``"pos_relative"`` (default) — ``ctrl_t = q_t + U_t``, where
              ``q_t`` is each world's robot joint position *at that step*,
              re-read every control step: a bounded servo relative to wherever
              the world has got to (the old planner's default,
              ``_assign_ctrl_relative_kernel``).
        delta_range: ``(low, high)`` clip applied to every sampled value before
            it becomes a command. Either side may be ``None`` to leave it
            unbounded. Off by default: under ``"pos_relative"`` the value *is*
            the position-servo error, so clipping it caps how hard the hand can
            grip. Under ``"absolute"`` it bounds the command itself; under
            ``"ctrl_relative"``, the per-step change in it.
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
    control_mode: str = "pos_relative"
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
        A_wp: ``(N, H, nu)`` commands for ``"ctrl_relative"``, precomputed
            from ``V`` and handed to the simulator as a sequence.
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
            task: Supplies ``calcCosts``, and (for ``"pos_relative"`` control)
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

        # Where the robot's joints start in qpos, for the pos_relative mode.
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

        # ctrl_relative: the control applied when planning starts, which the
        # accumulated deltas are added to.
        self._u_prev_wp = wp.zeros(nu, dtype=wp.float32, device=dev)
        # pos_relative: one control step's commands, computed on the device from
        # the rollout's current positions and handed to the simulator.
        self._ctrl_step_wp = wp.zeros((N, nu), dtype=wp.float32, device=dev)
        # Host copies of the two command bases, for forming the returned action.
        self._u_prev: np.ndarray | None = None
        self._q_robot = np.zeros(nu)

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
    def Plan(self, q: np.ndarray, q_dot: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        """Optimize from the measured state and return the next action.

        Args:
            q: Measured positions, ``(nq,)``.
            q_dot: Measured velocities, ``(nv,)``.
            u: The control currently applied, ``(nu,)``. Only
                ``"ctrl_relative"`` uses it — its deltas start from it. ``None``
                falls back to the action this planner returned last, which is
                right as long as that is what was applied; pass it whenever it
                may not be, such as at the start of an episode. Not in the
                plan's signature; ``ctrl_relative`` cannot work without it.

        Returns:
            ``(nu,)`` actuator command — what to hand
            ``Simulator.SetControl``. Already resolved for the control mode, so
            a driver never has to know which parameterization the planner used:
            ``U_0``, ``u + U_0`` or ``q_robot + U_0``.

        Raises:
            ValueError: On a bad shape, or under ``"ctrl_relative"`` when the
                current control is unknown — no ``u`` and no previous action.
        """
        q = np.asarray(q, dtype=float).ravel()
        q_dot = np.asarray(q_dot, dtype=float).ravel()
        if q.shape != (self.nq,):
            raise ValueError(f"q must have shape ({self.nq},), got {q.shape}")
        if q_dot.shape != (self.nv,):
            raise ValueError(f"q_dot must have shape ({self.nv},), got {q_dot.shape}")

        self._setCommandBase(q, u)
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

    def _setCommandBase(self, q: np.ndarray, u: np.ndarray | None) -> None:
        """Record what this plan's commands are relative to.

        ``pos_relative`` needs the measured joint positions — only for the
        returned action and tape; the rollout re-reads each world's own. And
        ``ctrl_relative`` needs the control currently applied, uploaded for the
        rollout's accumulation.
        """
        a = self.robot_qpos_adr
        self._q_robot = q[a:a + self.nu].copy()
        if self.config.control_mode != "ctrl_relative":
            return
        if u is not None:
            u = np.asarray(u, dtype=float).ravel()
            if u.shape != (self.nu,):
                raise ValueError(f"u must have shape ({self.nu},), got {u.shape}")
            self._u_prev = u.copy()
        if self._u_prev is None:
            raise ValueError(
                "ctrl_relative needs the control currently applied: pass u= to "
                "Plan (there is no previous action to fall back on)."
            )
        self._u_prev_wp.assign(self._u_prev.astype(np.float32))

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
        """Turn sampled values into commands, where that can be done up front.

        ``absolute`` needs nothing — the samples *are* the commands.
        ``ctrl_relative`` accumulates them onto the current control:
        ``A_t = u + sum_{k<=t} V_k``. ``pos_relative`` cannot be precomputed —
        each step's command depends on where the rollout has got to — so it is
        formed inside the rollout, step by step.
        """
        if self.config.control_mode == "ctrl_relative":
            wp.launch(
                _cumsum_commands_kernel,
                dim=(self.N, self.nu),
                inputs=[self.V_wp, self._u_prev_wp, self.horizon],
                outputs=[self.A_wp],
            )

    def _armControls(self) -> None:
        """Hand the simulator this rollout's commands, as the mode requires.

        Precomputable modes go in as a sequence the simulator steps through.
        ``pos_relative`` clears any sequence instead, since the rollout body
        sets each step's command itself.
        """
        mode = self.config.control_mode
        if mode == "absolute":
            self.sim.SetControlSequence(self.V_wp)
        elif mode == "ctrl_relative":
            self.sim.SetControlSequence(self.A_wp)
        else:
            self.sim.ClearControlSequence()

    def _rollout(self, q: np.ndarray, q_dot: np.ndarray) -> None:
        """Roll all ``N`` sampled sequences out in parallel and score them.

        Stages the start state on the device, then replays the captured rollout
        graph — or runs it eagerly when capture is off or unavailable.
        """
        self._q0_wp.assign(q.astype(np.float32))
        self._v0_wp.assign(q_dot.astype(np.float32))
        self._armControls()

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
        pos_relative = self.config.control_mode == "pos_relative"

        for t in range(self.horizon):
            if pos_relative:
                # ctrl_t = q_t + V_t, from each world's joint positions as they
                # stand at the start of this control step.
                wp.launch(
                    _pos_relative_kernel, dim=(self.N, self.nu),
                    inputs=[self.V_wp, t, state.qpos, self.robot_qpos_adr],
                    outputs=[self._ctrl_step_wp],
                )
                self.sim.SetControl(self._ctrl_step_wp)
            self.sim.Step_GPU(self.substeps)
            terminal = t == self.horizon - 1
            # The task reads the simulator's device state itself; `out` keeps
            # the call allocation-free, which is what lets it be captured.
            self.task.calcCosts(self.sim, terminal=terminal, out=self._step_cost_wp)
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
        self._armControls()
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
        seq = self.U_wp.numpy().astype(float)
        mode = self.config.control_mode
        if mode == "absolute":
            tape = seq
        elif mode == "ctrl_relative":
            tape = self._u_prev + np.cumsum(seq, axis=0)
        else:
            # Row 0 is exact — the rollout's first step reads the measured q.
            # Later rows assume the robot holds q; the rollouts re-read it.
            tape = self._q_robot + seq
        self.last_action_seq = tape.astype(np.float32)
        action = self.last_action_seq[0].copy()
        # The next ctrl_relative plan starts from this command, unless told
        # otherwise with Plan(u=...).
        self._u_prev = action.astype(float)

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

        Zero is the right seed under both relative modes — the sequence is a
        delta, so a zero mean means "hold the pose" (``pos_relative``) or "hold
        the command" (``ctrl_relative``). Under ``"absolute"`` it is not: a zero
        mean commands every joint to zero, and on a grasping hand that opens it
        and drops the object on the first plan. Seed it with the task's initial
        control there.
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
def _cumsum_commands_kernel(
    V: wp.array3d(dtype=float),       # (N, H, nu)  per-step deltas
    u_prev: wp.array(dtype=float),    # (nu,)       control applied at plan time
    H: int,
    out: wp.array3d(dtype=float),     # (N, H, nu)  [out]
):
    """ctrl_relative commands: out[t] = u_prev + sum_{k<=t} V[k].

    One thread per (world, actuator), running the sum along the horizon in
    order — the same additions, in the same order, as applying ctrl += V[t]
    one step at a time.
    """
    n, u = wp.tid()
    acc = u_prev[u]
    for t in range(H):
        acc = acc + V[n, t, u]
        out[n, t, u] = acc


@wp.kernel
def _pos_relative_kernel(
    V: wp.array3d(dtype=float),       # (N, H, nu)
    t: int,
    qpos: wp.array2d(dtype=float),    # (N, nq)  each world's current positions
    robot_adr: int,                   # robot-joint start index in qpos
    out: wp.array2d(dtype=float),     # (N, nu)  [out]
):
    """pos_relative command for step t: out = q_t + V[t], per world."""
    n, u = wp.tid()
    out[n, u] = qpos[n, robot_adr + u] + V[n, t, u]


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
