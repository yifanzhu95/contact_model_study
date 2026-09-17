"""Shared engine for sampling-based predictive control.

Every planner in this package implements the same loop:

    while planning:
        x0 <- estimate current state
        for i = 1..N:  U^(i) ~ pi_theta(U);  J^(i) <- J(U^(i), x0)
        theta <- update_params(U^(1:N), J^(1:N))
        u(t)  <- get_action(theta, t)

The N candidate trajectories are rolled out in parallel on the GPU as N worlds
of one batched MuJoCo-Warp environment (nworld = n_samples), with the physics
step and the state reset captured in CUDA graphs so no host-side data reset
happens inside the loop.

`SamplingPlanner` owns everything that is planner-agnostic — schedule
quantization, Warp array allocation, graph capture, the rollout unroll (full
graph / per-step graph / wall-clock-budgeted), cost accumulation, the control
parameterization and the warm-start shift. A concrete planner supplies only:

    _build_samples()          fill V_wp (N, H, nu) from the current theta
    _update_params(n_eff)     fold costs_wp/terminal_costs_wp into theta
    _reset_params()           restore adaptive state on reset()

MPPI (softmax-weighted mean), CEM (elite refit) and the predictive sampler
(greedy best-of-N) differ *only* in those three hooks, so planner comparisons
hold the rollout, the cost and the control parameterization fixed.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
import math
import time
import warnings

import numpy as np
import warp as wp
import mujoco

from contact_study.contact_models import api
from contact_study.contact_models.config import ContactModelConfig


# Step counts used (with a warning) when a schedule quantity is specified
# neither in time nor in steps. These are the historical MPPIConfig defaults.
_FALLBACK_STEP_HORIZON  = 50
_FALLBACK_STEP_SUBSTEPS = 1

# Slack on the steps-per-duration ratio before flooring, so a duration that is
# an exact multiple of rollout_dt in exact arithmetic (e.g. 0.1 / 0.02) is not
# knocked down a step by binary floating-point representation error.
_RATIO_TOL = 1e-6


@dataclass
class PlannerConfig:
    """Configuration shared by every sampling-based planner.

    Concrete planners subclass this and add their own knobs (MPPI's
    `temperature`, CEM's `elite_frac`, ...). Every field here has a default, so
    subclass fields may also have defaults in any order.
    """
    n_samples:       int   = 1024   # N: number of candidate trajectories
    # --- rollout schedule: time-based (preferred) or step-based -------------
    # The schedule is quantized against the planning model's rollout timestep
    # (mjm.opt.timestep) by PlannerConfig.resolve_schedule, which the planner
    # calls at construction. When a time-based field is set it wins and the
    # corresponding step-based field is ignored; the quantization always rounds
    # DOWN, so the realized durations never exceed the requested ones.
    #   step_time    -> step_substeps = floor(step_time / rollout_dt)
    #   time_horizon -> step_horizon  = floor(time_horizon / control_dt),
    #                   where control_dt = step_substeps * rollout_dt
    # When both the time-based and the step-based field of a quantity are None
    # the fallback above is used and a warning is issued.
    time_horizon:    float | None = 0.25   # planning horizon (seconds)
    step_time:       float | None = 0.032  # control-step duration (seconds)
    step_horizon:    int   | None = None   # H: planning horizon (control steps)
    step_substeps:   int   | None = None   # physics steps per control step
    noise_sigma:     float = 0.01   # action noise std dev
    n_iterations:    int   = 1      # optimizer iterations per plan() call
    # Warm start: shift the action sequence one step forward between plans.
    # Default False to match irisim_warp, which keeps the running mean and does
    # not roll it forward (the shift is commented out there).
    warm_start:      bool  = False
    # Rollout control parameterization. True (default, irisim_warp-style): each
    # rollout step commands ctrl = current measured robot joint qpos + delta,
    # re-reading the joint every step (a bounded servo relative to the current
    # pose). False (legacy): accumulate the deltas onto the commanded target
    # across the horizon (ctrl += delta).
    ctrl_relative_to_qpos: bool = True
    nconmax:         int   = 200
    njmax:           int   = 500
    debug:           bool  = True
    # Per-step delta clip (low, high). Either side may be None to leave that
    # side unconstrained (e.g. (None, 0.1) clips only the upper bound,
    # (None, None) — the default — disables clipping entirely).
    #
    # Off by default because the clamp is NOT a velocity limit: with
    # ctrl_relative_to_qpos the command is ctrl = measured qpos + delta, so
    # delta IS the position-servo error, and capping it caps the joint torque
    # the hand can develop — i.e. how hard it can grip. Every driver that
    # exposes --delta already defaults it to None (run_eval_episode.py,
    # run_bayes_opt.py, the hpc cells), so this makes the dataclass default
    # agree with them: a config built without an explicit delta_range now
    # behaves the same as one built by those drivers with the flag omitted.
    # Set it explicitly to re-enable the clamp.
    delta_range:     tuple[float | None, float | None] = (None, None)
    use_full_graph:  bool  = True   # True=single mega CUDA graph, False=step+reset graphs
    seed:            int | None = None  # seed for deterministic noise sampling
    # plan() steps between noise resamples: 1=every step, None=sample once at
    # construction and reuse for the whole episode.
    resample_interval: int | None = 1
    # Redraw the noise block before every optimizer iteration after the first,
    # instead of reusing one block for the whole plan() call. Off (default) means
    # the iterations inside one plan() re-center the SAME perturbations on the
    # updated mean, which is what makes the update a fixed-point map (see MPPI's
    # convergence_tol). Requires resample_interval (the GPU resampler); a no-op
    # unless a plan() actually runs more than one iteration.
    resample_per_iteration: bool = False
    # Stop rollouts once plan_budget_ms of wall-clock has elapsed (capped at
    # horizon) instead of always unrolling the full horizon.
    time_constrained: bool  = False
    plan_budget_ms:   float | None = None   # required when time_constrained

    def __post_init__(self):
        for name in ("time_horizon", "step_time"):
            val = getattr(self, name)
            if val is not None and val <= 0.0:
                raise ValueError(f"{name} must be > 0 or None, got {val}")
        for name in ("step_horizon", "step_substeps"):
            val = getattr(self, name)
            if val is not None and val < 1:
                raise ValueError(f"{name} must be >= 1 or None, got {val}")
        delta_low, delta_high = self.delta_range
        if delta_low is not None and delta_high is not None and delta_low >= delta_high:
            raise ValueError(
                f"delta_range low must be < high when both are set, got {self.delta_range}"
            )
        if self.resample_interval is not None and self.resample_interval < 1:
            raise ValueError(
                f"resample_interval must be >= 1 or None, got {self.resample_interval}"
            )
        if self.resample_per_iteration and self.resample_interval is None:
            raise ValueError(
                "resample_per_iteration=True requires resample_interval to be set: "
                "resample_interval=None draws the block once on the host via "
                "_draw_static_noise, and redrawing it on the GPU would silently replace "
                "planner-shaped noise (MPPI's spline, CEM's unit-variance draw) with "
                "plain Gaussian noise."
            )
        if self.n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {self.n_iterations}")
        if self.time_constrained:
            if self.plan_budget_ms is None or self.plan_budget_ms <= 0:
                raise ValueError(
                    "time_constrained=True requires plan_budget_ms > 0, got "
                    f"{self.plan_budget_ms}"
                )
            if self.use_full_graph:
                raise ValueError(
                    "time_constrained=True requires use_full_graph=False: the full-graph path "
                    "bakes the entire horizon unroll into one replayed CUDA graph and cannot "
                    "stop early."
                )

    def resolve_schedule(self, rollout_dt: float) -> tuple[int, int]:
        """Quantize the rollout schedule against the planning model's timestep.

        Returns (horizon, substeps) in whole steps, both >= 1. `rollout_dt` is
        the planning model's step (mjm.opt.timestep, which the driver stamps as
        eval_dt * eval_substeps_per_rollout).

        Substeps are resolved first: they define the control step the horizon is
        counted in (control_dt = substeps * rollout_dt). Both conversions floor,
        so substeps * rollout_dt <= step_time and horizon * control_dt <=
        time_horizon — the realized schedule is the closest one that stays at or
        under the requested durations. A requested duration shorter than one
        step of its unit clamps to 1 with a warning, since a zero-length rollout
        has nothing to plan with.
        """
        if rollout_dt <= 0.0:
            raise ValueError(f"rollout_dt must be > 0, got {rollout_dt}")

        def _floor_steps(duration: float, unit: float, name: str) -> int:
            n = int(math.floor(duration / unit + _RATIO_TOL))
            if n < 1:
                warnings.warn(
                    f"{type(self).__name__}.{name}={duration:g}s is shorter than one step of "
                    f"{unit:g}s; clamping to 1 step ({unit:g}s).",
                    stacklevel=3,
                )
                return 1
            return n

        if self.step_time is not None:
            substeps = _floor_steps(self.step_time, rollout_dt, "step_time")
        elif self.step_substeps is not None:
            substeps = int(self.step_substeps)
        else:
            warnings.warn(
                f"{type(self).__name__}: neither step_time nor step_substeps is set; falling "
                f"back to step_substeps={_FALLBACK_STEP_SUBSTEPS}.",
                stacklevel=2,
            )
            substeps = _FALLBACK_STEP_SUBSTEPS

        control_dt = substeps * rollout_dt
        if self.time_horizon is not None:
            horizon = _floor_steps(self.time_horizon, control_dt, "time_horizon")
        elif self.step_horizon is not None:
            horizon = int(self.step_horizon)
        else:
            warnings.warn(
                f"{type(self).__name__}: neither time_horizon nor step_horizon is set; falling "
                f"back to step_horizon={_FALLBACK_STEP_HORIZON}.",
                stacklevel=2,
            )
            horizon = _FALLBACK_STEP_HORIZON

        return horizon, substeps


# ---------------------------------------------------------------------------
# Module-level Warp kernels shared by all planners
# ---------------------------------------------------------------------------

@wp.kernel
def _broadcast_1d_to_2d_kernel(
    src: wp.array(dtype=float),
    dst: wp.array2d(dtype=float)
):
    n, i = wp.tid()
    dst[n, i] = src[i]


@wp.kernel
def _sample_gaussian_noise_kernel(
    seed:  int,
    sigma: float,
    eps:   wp.array3d(dtype=float),   # (N, H, nu)  [out]
):
    """Resample the noise block in place on the GPU.

    Each element gets its own RNG stream keyed by (seed, flat index), so the
    draw is reproducible for a given seed and decorrelated across seeds. Written
    in place: eps is read by the sample-building kernels outside any CUDA graph,
    so in-place writes are safe but the buffer must never be swapped.
    """
    n, h, u = wp.tid()
    tid   = (n * eps.shape[1] + h) * eps.shape[2] + u
    state = wp.rand_init(seed, tid)
    eps[n, h, u] = sigma * wp.randn(state)


@wp.kernel
def _assign_ctrl_kernel(
    V:    wp.array3d(dtype=float),   # (N, H, nu)
    t:    int,                       # timestep index
    ctrl: wp.array2d(dtype=float),   # (N, nu)  [out]
):
    """Copy the t-th slice of V into d.ctrl without a Python-side round-trip."""
    n, u = wp.tid()
    ctrl[n, u] += V[n, t, u]


@wp.kernel
def _assign_ctrl_relative_kernel(
    V:         wp.array3d(dtype=float),   # (N, H, nu)
    t:         int,                       # timestep index
    qpos:      wp.array2d(dtype=float),   # (N, nq)
    robot_adr: int,                       # robot-joint start index in qpos
    ctrl:      wp.array2d(dtype=float),   # (N, nu)  [out]
):
    """irisim_warp-style servo control: ctrl = current robot joint qpos + delta.

    Re-reads the measured joint from qpos each rollout step so the command is a
    bounded servo relative to the current pose, rather than accumulating the
    deltas onto the running command (see _assign_ctrl_kernel)."""
    n, u = wp.tid()
    ctrl[n, u] = qpos[n, robot_adr + u] + V[n, t, u]


def _make_accumulate_kernel(cost_fn_wp: wp.func):
    @wp.kernel
    def _kernel(
        qpos:      wp.array2d(dtype=float),
        qvel:      wp.array2d(dtype=float),
        ctrl:      wp.array2d(dtype=float),
        site_xpos: wp.array2d(dtype=wp.vec3),
        site_xmat: wp.array2d(dtype=wp.mat33),
        terminal:  bool,
        goal:      wp.array(dtype=float),
        indices:   wp.array(dtype=int),
        weights:   wp.array(dtype=float),
        costs_out: wp.array(dtype=float),
    ):
        w = wp.tid()
        costs_out[w] += cost_fn_wp(
            qpos[w], qvel[w], ctrl[w], site_xpos[w], site_xmat[w],
            terminal, goal, indices, weights
        )
    return _kernel


@wp.kernel
def _combine_costs_kernel(
    running:       wp.array(dtype=float),   # (N,)  running-cost sum [in/out: total]
    terminal:      wp.array(dtype=float),   # (N,)  terminal cost
    running_scale: float,                   # 1/H  (or 1.0 to disable)
    total_scale:   float,                   # 1/N  (or 1.0 to disable)
):
    """Fold the separately-accumulated terminal cost into the running-cost sum.

    The terminal cost is added to the running-cost sum and the combined total is
    scaled by running_scale (1/H when mean_cost_over_horizon is set) to a
    per-step mean over the whole trajectory — matching irisim_warp's
    `total_cost / horizon`, which likewise averages every step including the
    terminal one. total_scale (1/N) is applied on top for sample invariance.
    Written in place into `running`, which downstream kernels read as the
    per-sample trajectory cost.
    """
    n = wp.tid()
    running[n] = (running[n] + terminal[n]) * running_scale * total_scale


@wp.kernel
def _find_min_kernel(
    costs:   wp.array(dtype=float),
    min_val: wp.array(dtype=float),
):
    n = wp.tid()
    wp.atomic_min(min_val, 0, costs[n])


@wp.kernel
def _shift_2d_kernel(
    src:   wp.array2d(dtype=float),  # (H, nu)
    dst:   wp.array2d(dtype=float),  # (H, nu)  [out]
    H:     int,
    shift: int,                      # rows to roll forward (1 = classic warm start)
    fill:  float,                    # value written into the rows that fall off the end
):
    """Warm-start shift: dst[h] = src[h+shift], with the tail set to `fill`.

    shift=1 is the classic one-step warm start. The async driver passes the
    number of control steps that elapsed during the solve, so the mean carried
    into the next plan() is aligned with that plan's t0 rather than with a t0
    that is already in the past.

    Written into a separate buffer (not in place) so adjacent rows can't race
    on the read-before-write; the caller swaps src/dst afterward."""
    h, u = wp.tid()
    if h + shift < H:
        dst[h, u] = src[h + shift, u]
    else:
        dst[h, u] = fill


# ---------------------------------------------------------------------------
# Planner base
# ---------------------------------------------------------------------------

class SamplingPlanner(abc.ABC):
    """Sampling-based predictive controller backed by a contact model.

    Subclasses implement _build_samples / _update_params / _reset_params; see
    the module docstring.
    """

    # Short tag used in debug output.
    name: str = "planner"

    # Costs at or above this value mean wp.atomic_min was never written
    # (all rollouts produced NaN costs).
    _COST_SENTINEL = 1e9

    def __init__(
        self,
        task,
        cfg:         ContactModelConfig,
        planner_cfg: PlannerConfig,
        rng:         np.random.Generator | None = None,
    ):
        # The (ROLLOUT) task is a thin adapter supplying the planning model and
        # the cost arrays: the rollout env and CUDA graphs below only ever read
        # self.mjm / self.cost_fn_wp_func / self.goal_wp / self.indices_wp /
        # self.weights_wp.
        self.task = task
        self.mjm = task.mjm
        self.cfg = cfg
        self.pc  = planner_cfg
        self.rng = rng or np.random.default_rng()

        # Rollout schedule in whole steps, quantized against the planning
        # model's timestep (the driver stamps rollout_dt onto mjm before
        # constructing the planner). self.horizon / self.substeps are the
        # authoritative values from here on — read them, not the config fields,
        # which may hold durations instead of step counts. They are written back
        # onto the config so callers that only hold it (e.g. logging a run's
        # parameters) see the resolved integers.
        self.rollout_dt = float(self.mjm.opt.timestep)
        self.horizon, self.substeps = self.pc.resolve_schedule(self.rollout_dt)
        self.control_dt = self.substeps * self.rollout_dt
        self.pc.step_horizon  = self.horizon
        self.pc.step_substeps = self.substeps

        self.nu = self.mjm.nu
        self.nq = self.mjm.nq
        self.nv = self.mjm.nv

        self.cost_fn_wp_func = task.cost_fn_wp
        self.goal_wp = task.cost_goal_wp
        self.indices_wp = task.cost_idx_wp
        self.weights_wp = task.cost_weights_wp
        self._accumulate_costs_kernel = _make_accumulate_kernel(self.cost_fn_wp_func)

        # Robot-joint start address in qpos for the relative (servo) control
        # parameterization (ctrl_relative_to_qpos). Taken from the task's cost
        # index vector (slot 2 = robot_qpos_adr) when available, else 0 (robot
        # joints lead qpos).
        self.robot_qpos_adr = 0
        if self.indices_wp is not None:
            idx_np = self.indices_wp.numpy()
            if idx_np.shape[0] > 2:
                self.robot_qpos_adr = int(idx_np[2])

        # Noise-resampling bookkeeping. _resample_count only ever advances (it
        # keys the RNG seed), so no two resamples in this planner's lifetime
        # can replay the same noise block.
        self._plan_count     = 0
        self._resample_count = 0
        # Rollout steps actually taken by the last plan() — < horizon only when
        # the time-constrained path truncated.
        self.last_n_steps    = self.horizon
        # Optimizer iterations the last plan() actually ran — below the cap only
        # when the convergence test fired (MPPI's convergence_tol).
        self.last_n_iterations = 0

        # The full (H, nu) mean action sequence produced by the last plan(),
        # captured BEFORE the warm-start shift. Synchronous drivers only need
        # row 0 (which plan() returns); the async driver plays the whole tape
        # out over the latency window. None until the first plan().
        self.last_action_seq: np.ndarray | None = None
        # Whether the last plan() completed a real parameter update. False after
        # an all-NaN solve, which zeroes U_wp/last_action_seq but leaves the
        # planner-specific particle state (MPPI's w_wp, CEM's elite_idx_wp)
        # holding the PREVIOUS call's values — so anything reading that state
        # (evaluation/distributions.py) must check this first or it will log a
        # stale distribution as if it were this step's.
        self.last_plan_ok:    bool = False
        # Rows the warm-start shift rolls forward per plan(). 1 is the classic
        # one-step warm start; the async driver raises it to the number of
        # control steps the solve actually consumed.
        self.shift_steps     = 1

        self._setup_warp_arrays()
        self._setup_warp_backend()

    # -- setup --------------------------------------------------------------

    def _setup_warp_arrays(self):
        """Allocate Warp arrays on GPU once (avoids runtime host copies)."""
        N, H, nu = self.pc.n_samples, self.horizon, self.nu

        # Pre-allocate reset buffers
        self.qpos_reset = wp.empty(self.nq, dtype=wp.float32, device="cuda")
        self.qvel_reset = wp.empty(self.nv, dtype=wp.float32, device="cuda")
        self.ctrl_reset = wp.empty(self.nu, dtype=wp.float32, device="cuda")

        # theta's mean action sequence, the sampled trajectories drawn from it,
        # and a scratch buffer for the on-GPU warm-start shift (swapped with U_wp).
        self.U_wp = wp.zeros((H, nu), dtype=wp.float32, device="cuda")
        self.U_shift_wp = wp.zeros((H, nu), dtype=wp.float32, device="cuda")
        self.V_wp = wp.zeros((N, H, nu), dtype=wp.float32, device="cuda")
        # Running (per-step) cost sum and the terminal cost are accumulated into
        # separate buffers so the running sum can be horizon-normalized without
        # dividing the single-step terminal cost; _fold_costs folds them.
        self.costs_wp = wp.zeros(N, dtype=wp.float32, device="cuda")
        self.terminal_costs_wp = wp.zeros(N, dtype=wp.float32, device="cuda")

        self.min_cost_wp = wp.zeros(1, dtype=wp.float32, device="cuda")
        self._big_float_np = np.array([1e30], dtype=np.float32)

        # Per-step delta limits. A None side is replaced with +/-inf so the clamp
        # kernel's single wp.clamp(val, low, high) call becomes a no-op on
        # that side; if both sides are None, clamping is skipped entirely via
        # has_limits.
        delta_low, delta_high = self.pc.delta_range
        delta_range_np = np.empty((self.nu, 2), dtype=np.float32)
        delta_range_np[:, 0] = delta_low if delta_low is not None else -np.inf
        delta_range_np[:, 1] = delta_high if delta_high is not None else np.inf

        has_limits = not (delta_low is None and delta_high is None)
        self._ctrl_range_wp = wp.array(delta_range_np, dtype=wp.float32, device="cuda")
        self._has_limits_wp = wp.array(np.full(self.nu, has_limits, dtype=bool),
                                       dtype=wp.bool, device="cuda")

        # Deterministic noise: use seed from config if provided, else fall back
        # to the shared rng.
        noise_rng = np.random.default_rng(self.pc.seed) if self.pc.seed is not None else self.rng

        if self.pc.resample_interval is not None:
            # Resampling path: the block is drawn on the GPU by
            # _maybe_resample_noise, so only an int base seed is taken from
            # noise_rng here. The first plan() call always resamples, so these
            # zeros are never actually rolled out.
            self._noise_seed = int(noise_rng.integers(0, 2**31 - 1))
            self._static_eps_wp = wp.zeros((N, H, nu), dtype=wp.float32, device="cuda")
        else:
            self._static_eps_wp = wp.array(
                self._draw_static_noise(noise_rng, N, H, nu),
                dtype=wp.float32, device="cuda",
            )

    def _setup_warp_backend(self):
        """Initialize the batched MuJoCo backends and CUDA graphs."""
        self.m = api.put_model(self.mjm, self.cfg)
        self.d = api.make_data(
            self.mjm, self.m,
            nworld=self.pc.n_samples,
            nconmax=self.pc.nconmax,
            njmax=self.pc.njmax,
        )
        if self.pc.use_full_graph:
            self.rollout_graph = self.create_rollout_graph()
        else:
            self.reset_graph = self.create_reset_graph()
            self.step_graph = self.create_step_graph()

    def create_reset_graph(self):
        """Create a graph to broadcast environment states and zero costs across N worlds."""
        with wp.ScopedCapture() as capture:
            self._launch_reset()
        return capture.graph

    def create_step_graph(self):
        """Capture a single physics step.

        When use_full_graph=False, substeps are handled by launching this
        graph substeps-times in the Python loop rather than baking them into
        the graph.  This keeps the graph small regardless of substep count.
        """
        with wp.ScopedCapture() as capture:
            api.step(self.m, self.d)
        return capture.graph

    def create_rollout_graph(self):
        """Captures the reset AND the entire H-step unroll into a single CUDA graph."""
        with wp.ScopedCapture() as capture:
            self._launch_reset()
            for t in range(self.horizon):
                terminal = (t == self.horizon - 1)
                self._launch_assign_ctrl(t)
                for _ in range(self.substeps):
                    api.step(self.m, self.d)
                self._launch_accumulate_costs(terminal)
        return capture.graph

    # -- rollout primitives -------------------------------------------------

    def _launch_reset(self):
        """Broadcast the start state across the N worlds and zero the costs."""
        wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nq),
                  inputs=[self.qpos_reset, self.d.qpos])
        wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nv),
                  inputs=[self.qvel_reset, self.d.qvel])
        wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nu),
                  inputs=[self.ctrl_reset, self.d.ctrl])
        self.costs_wp.zero_()
        self.terminal_costs_wp.zero_()

    def _launch_assign_ctrl(self, t: int):
        """Write the rollout controls for horizon step t into d.ctrl.

        ctrl_relative_to_qpos selects the parameterization (see PlannerConfig):
        True re-reads the current robot joint qpos each step (irisim_warp servo),
        False accumulates the delta onto the running command.
        """
        if self.pc.ctrl_relative_to_qpos:
            wp.launch(
                _assign_ctrl_relative_kernel,
                dim=(self.pc.n_samples, self.nu),
                inputs=[self.V_wp, t, self.d.qpos, self.robot_qpos_adr, self.d.ctrl],
            )
        else:
            wp.launch(
                _assign_ctrl_kernel,
                dim=(self.pc.n_samples, self.nu),
                inputs=[self.V_wp, t, self.d.ctrl],
            )

    def _launch_accumulate_costs(self, terminal: bool):
        """Accumulate one step's cost across the N worlds.

        The terminal cost goes to its own buffer so it can be excluded from the
        horizon normalization in _fold_costs."""
        wp.launch(
            self._accumulate_costs_kernel,
            dim=self.pc.n_samples,
            inputs=[
                self.d.qpos, self.d.qvel, self.d.ctrl, self.d.site_xpos, self.d.site_xmat,
                terminal, self.goal_wp, self.indices_wp, self.weights_wp
            ],
            outputs=[self.terminal_costs_wp if terminal else self.costs_wp],
        )

    # -- noise --------------------------------------------------------------

    def _draw_static_noise(self, noise_rng, N: int, H: int, nu: int) -> np.ndarray:
        """Host-side noise block for the no-resampling path (resample_interval=None).

        Default: i.i.d. Gaussian at the configured sigma. Overridden by planners
        that shape their noise differently (MPPI's spline noise, CEM's
        unit-variance draw)."""
        return noise_rng.normal(
            loc=0.0, scale=self.pc.noise_sigma, size=(N, H, nu)
        ).astype(np.float32)

    @property
    def _noise_sample_sigma(self) -> float:
        """Scale used by the GPU resampler. CEM overrides this with 1.0 and
        applies its per-(h, u) sigma when building samples."""
        return self.pc.noise_sigma

    def _sample_noise_block(self):
        """Redraw the whole (N, H, nu) noise block on the GPU."""
        wp.launch(
            _sample_gaussian_noise_kernel,
            dim=(self.pc.n_samples, self.horizon, self.nu),
            inputs=[self._noise_seed + self._resample_count, self._noise_sample_sigma],
            outputs=[self._static_eps_wp],
        )

    def _maybe_resample_noise(self):
        """Redraw the noise block every `resample_interval` plan() calls (no-op if None)."""
        k = self.pc.resample_interval
        if k is not None and self._plan_count % k == 0:
            self._sample_noise_block()
            self._resample_count += 1
        self._plan_count += 1

    def _resample_between_iterations(self):
        """Redraw the noise block before an optimizer iteration after the first.

        Only called when resample_per_iteration is set. Iteration 0 always uses
        whatever block the per-plan resample_interval cadence selected in
        _maybe_resample_noise, so the default (flag off) path is untouched.
        _resample_count keeps advancing, so no two blocks in this planner's
        lifetime replay the same noise.
        """
        self._sample_noise_block()
        self._resample_count += 1

    # -- cost folding -------------------------------------------------------

    def _fold_costs(self, running_scale: float = 1.0, total_scale: float = 1.0):
        """Fold the terminal cost into the running sum; costs_wp holds the total."""
        wp.launch(
            _combine_costs_kernel, dim=self.pc.n_samples,
            inputs=[self.costs_wp, self.terminal_costs_wp, running_scale, total_scale],
        )

    def _min_cost(self) -> float:
        """Minimum folded trajectory cost (device reduce + one scalar sync).

        Returns >= _COST_SENTINEL when every cost was NaN — wp.atomic_min never
        fired, so the sentinel survives."""
        self.min_cost_wp.assign(self._big_float_np)
        wp.launch(_find_min_kernel, dim=self.pc.n_samples,
                  inputs=[self.costs_wp, self.min_cost_wp])
        wp.synchronize()
        return float(self.min_cost_wp.numpy()[0])

    # -- rollout ------------------------------------------------------------

    def _rollout(self) -> int:
        """Roll the N sampled trajectories out in parallel; return steps taken.

        Dispatches on time_constrained, then use_full_graph. The return value is
        the horizon except on the time-constrained path, which stops early.
        """
        if self.pc.time_constrained:
            n_eff = self._rollout_time_constrained()
        elif self.pc.use_full_graph:
            n_eff = self._rollout_full_graph()
        else:
            n_eff = self._rollout_step_graphs()
        self.last_n_steps = n_eff
        return n_eff

    def _rollout_full_graph(self) -> int:
        """Single mega CUDA graph: reset + full H-step unroll captured together."""
        wp.capture_launch(self.rollout_graph)
        return self.horizon

    def _rollout_step_graphs(self) -> int:
        """Separate reset + step graphs: reset once, then loop over H steps."""
        H = self.horizon
        wp.capture_launch(self.reset_graph)
        for t in range(H):
            self._launch_assign_ctrl(t)
            # Launch the one-step graph substeps times — keeps the graph small
            # regardless of substep count (unlike the full-graph path which
            # bakes all substeps into a single captured graph).
            for _ in range(self.substeps):
                wp.capture_launch(self.step_graph)
            self._launch_accumulate_costs(terminal=(t == H - 1))
        return H

    def _rollout_time_constrained(self) -> int:
        """Step rollouts against a wall-clock budget, capped at the horizon.

        Unrolls like _rollout_step_graphs but stops as soon as plan_budget_ms has
        elapsed, then finishes the current step so the action is computed from the
        steps that did run. The deadline is only tested *after* a completed step,
        so a too-small budget degrades to a 1-step horizon rather than no action.
        """
        H = self.horizon
        budget_s = self.pc.plan_budget_ms * 1e-3
        wp.capture_launch(self.reset_graph)

        deadline = time.perf_counter() + budget_s
        n_eff = 0
        for t in range(H):
            self._launch_assign_ctrl(t)
            for _ in range(self.substeps):
                wp.capture_launch(self.step_graph)

            # Warp launches are async, so without this sync the clock would
            # measure CPU enqueue time rather than real GPU progress and the
            # loop would run the full horizon regardless of the budget.
            wp.synchronize()

            # Decide `terminal` before accumulating: a terminal cost replaces
            # the running cost rather than adding to it, so each step needs
            # exactly one launch carrying the correct flag.
            last  = (t == H - 1) or (time.perf_counter() >= deadline)
            n_eff = t + 1
            self._launch_accumulate_costs(terminal=last)
            if last:
                break

        if self.pc.debug:
            print(f"  [{self.name}] time-constrained: {n_eff}/{H} steps in "
                  f"{self.pc.plan_budget_ms:.1f} ms budget")
        return n_eff

    # -- action extraction --------------------------------------------------

    def _extract_action(self) -> np.ndarray:
        # The whole (H, nu) mean crosses device→host — a few hundred bytes, and
        # no more expensive than the row-0 copy it replaces, since both pay one
        # transfer. Stash it pre-shift so async drivers can play the tape out;
        # the warm-start shift itself stays on the GPU. U_wp is never baked into
        # a captured graph, so swapping the buffer reference is safe.
        seq = self.U_wp.numpy().copy()
        self.last_action_seq = seq
        # An owned copy, not a view into seq: callers have always been free to
        # mutate the returned action, and must not reach the published tape.
        action_np = seq[0].copy()
        if self.pc.warm_start:
            wp.launch(
                _shift_2d_kernel,
                dim=(self.horizon, self.nu),
                inputs=[self.U_wp, self.U_shift_wp, self.horizon,
                        self.shift_steps, 0.0],
            )
            self.U_wp, self.U_shift_wp = self.U_shift_wp, self.U_wp
        return action_np

    # -- public API ---------------------------------------------------------

    def reset(self):
        """Clear the action sequence (call at the start of a new episode/goal)."""
        self.U_wp.zero_()
        self.last_action_seq = None
        # Restart the resample cadence so a fresh goal begins on fresh noise.
        # _resample_count deliberately keeps advancing (it keys the seed).
        self._plan_count = 0
        self._reset_params()

    def plan(self, mjd: mujoco.MjData) -> np.ndarray:
        """Run the sampling-based optimizer from state `mjd`, return the first action.

        One call = one iteration of the outer loop in the module docstring:
        sample N control sequences from theta, evaluate them in parallel on the
        GPU, update theta, and read the action off it.
        """
        self._maybe_resample_noise()

        # x0 <- estimated current state, broadcast into all N worlds by the
        # reset (captured inside the rollout graph on the full-graph path).
        self.qpos_reset.assign(mjd.qpos)
        self.qvel_reset.assign(mjd.qvel)
        self.ctrl_reset.assign(mjd.ctrl)

        self._begin_plan()
        self.last_plan_ok = False
        self.last_n_iterations = 0
        # tol=None is the fixed-iteration path: the loop runs `cap` times and
        # never pays the per-iteration device->host read of U[0].
        tol, cap = self._converge_tol, self._iteration_cap()
        u_prev = None
        for i in range(cap):
            # Iteration 0 keeps whatever block the per-plan resample_interval
            # cadence selected above, so the flag-off path is untouched.
            if i and self.pc.resample_per_iteration:
                self._resample_between_iterations()
            self._build_samples()             # U^(i) ~ pi_theta(U)
            n_eff = self._rollout()           # J^(i) <- J(U^(i), x0)
            if not self._update_params(n_eff):
                # Every rollout produced a NaN cost; the update may have written
                # NaN into theta — reset it so the next call starts clean. The
                # published tape is zeroed alongside it: a driver replaying the
                # previous tape here would be acting on a solve that failed.
                self.U_wp.zero_()
                self.last_action_seq = np.zeros((self.horizon, self.nu), dtype=np.float32)
                return np.zeros(self.nu, dtype=np.float32)
            self.last_n_iterations = i + 1
            if tol is None:
                continue
            # Row 0 is the action this call will return. Read self.U_wp fresh
            # every iteration rather than caching the array: _extract_action
            # swaps U_wp/U_shift_wp under warm_start, so a held reference would
            # alias the wrong buffer.
            u_now = self.U_wp.numpy()[0].copy()
            if u_prev is not None:
                d = u_now - u_prev
                if float(d @ d) < tol:
                    break
            # u_prev is None on i == 0, so the test always compares two
            # consecutive updates — the convergence path runs >= 2 iterations.
            u_prev = u_now

        if self.pc.debug and tol is not None:
            print(f"  [{self.name}] converged in {self.last_n_iterations}/{cap} iterations")

        self.last_plan_ok = True
        return self._extract_action()         # u(t) <- get_action(theta, t)

    # -- planner-specific hooks --------------------------------------------

    def _iteration_cap(self) -> int:
        """Upper bound on optimizer iterations for one plan() call.

        Planners that terminate on convergence rather than on a fixed count
        (MPPI's convergence_tol) return their cap here instead."""
        return self.pc.n_iterations

    @property
    def _converge_tol(self) -> float | None:
        """Squared-L2 tolerance on the change in the returned action between
        successive iterations, or None to run a fixed iteration count.

        None also means plan() never pays the extra device->host read of U[0].
        Only MPPI sets this (MPPIConfig.convergence_tol)."""
        return None

    def _begin_plan(self):
        """Hook run once per plan() call, before the first iteration.

        For planners whose distribution narrows as it converges (CEM), this is
        where the exploration is re-opened, so the n_iterations inside one call
        anneal from a known scale instead of inheriting a collapsed one from the
        previous control step."""
        return

    @abc.abstractmethod
    def _build_samples(self):
        """Fill V_wp (N, H, nu) with N control sequences drawn from theta.

        Samples must be clamped to delta_range (via _ctrl_range_wp /
        _has_limits_wp) so the action derived from theta stays bounded."""
        ...

    @abc.abstractmethod
    def _update_params(self, n_eff: int) -> bool:
        """Update theta from V_wp and the costs; False if the update is degenerate.

        costs_wp / terminal_costs_wp hold the per-sample running and terminal
        costs; call _fold_costs() first. n_eff caps the update at the rollout
        steps actually simulated (time-constrained path) — rows beyond it never
        influenced a cost, so they should be left untouched. Returning False
        means no valid rollout existed (all costs NaN) and plan() should emit a
        zero action."""
        ...

    def _reset_params(self):
        """Restore any adaptive state on reset() (temperature, sigma, ...)."""
        return
