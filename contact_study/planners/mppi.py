"""Model Predictive Path Integral (MPPI) controller.

All rollouts are executed in parallel on GPU via the batched step()
interface (nworld = N samples). The physics step and state resets are
encapsulated in CUDA graphs, eliminating slow CPU-side data resets
during the MPPI loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import warnings

from scipy.interpolate import CubicSpline
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
class MPPIConfig:
    n_samples:       int   = 1024   # N: number of candidate trajectories
    # --- rollout schedule: time-based (preferred) or step-based -------------
    # The schedule is quantized against the planning model's rollout timestep
    # (mjm.opt.timestep) by MPPIConfig.resolve_schedule, which the controller
    # calls at construction. When a time-based field is set it wins and the
    # corresponding step-based field is ignored; the quantization always rounds
    # DOWN, so the realized durations never exceed the requested ones.
    #   step_time    -> step_substeps = floor(step_time / rollout_dt)
    #   time_horizon -> step_horizon  = floor(time_horizon / control_dt),
    #                   where control_dt = step_substeps * rollout_dt
    # When both the time-based and the step-based field of a quantity are None
    # the fallback above is used and a warning is issued.
    time_horizon:    float | None = 0.25   # planning horizon (seconds)
    step_time:       float | None = 0.032   # control-step duration (seconds)
    step_horizon:    int   | None = None   # H: planning horizon (control steps)
    step_substeps:   int   | None = None   # physics steps per control step
    temperature:     float = 1.0    # lambda: MPPI temperature (inverse temp)
    noise_sigma:     float = 0.01   # action noise std dev
    n_iterations:    int   = 1      # number of MPPI update iterations per call
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
    adaptive_temp:   bool  = False
    adp_temp_params: tuple[float, float, float, float] = (10.0, 5.0, 0.9, 1.1)
    use_spline_noise:bool  = False   # toggle between spline and Gaussian noise
    n_spline_points: int   = 5      # control points for spline-smoothed noise
    debug:           bool  = True
    delta_range:     tuple[float, float] = (-0.1, 0.1)
    use_full_graph:  bool  = True   # True=single mega CUDA graph, False=step+reset graphs
    seed:            int | None = None  # seed for deterministic noise sampling
    # plan() steps between noise resamples: 1=every step, None=sample once at
    # construction and reuse for the whole episode.
    resample_interval: int | None = 1
    # Stop rollouts once plan_budget_ms of wall-clock has elapsed (capped at
    # horizon) instead of always unrolling the full horizon.
    time_constrained: bool  = False
    plan_budget_ms:   float | None = None   # required when time_constrained
    # Average the total trajectory cost (running + terminal) across the horizon
    # length before the weight computation, so `temperature` is invariant to
    # horizon. Unlike the old normalize_cost_by_horizon (which divided only the
    # running sum and left the terminal cost un-normalized), the terminal cost is
    # included in the mean here, matching irisim_warp's `total_cost / horizon`.
    mean_cost_over_horizon: bool = True
    # Normalize the total trajectory cost (running + terminal) by the number of
    # samples before the weight computation, so `temperature` is invariant to
    # n_samples.
    normalize_cost_by_samples: bool = False

    def __post_init__(self):
        for name in ("time_horizon", "step_time"):
            val = getattr(self, name)
            if val is not None and val <= 0.0:
                raise ValueError(f"{name} must be > 0 or None, got {val}")
        for name in ("step_horizon", "step_substeps"):
            val = getattr(self, name)
            if val is not None and val < 1:
                raise ValueError(f"{name} must be >= 1 or None, got {val}")
        if self.resample_interval is not None and self.resample_interval < 1:
            raise ValueError(
                f"resample_interval must be >= 1 or None, got {self.resample_interval}"
            )
        if self.use_spline_noise and self.resample_interval is not None:
            raise ValueError(
                "use_spline_noise is incompatible with resample_interval: spline noise is "
                "built host-side with CubicSpline and has no GPU sampler, so resampling it "
                "every step would defeat the purpose. Use Gaussian noise to resample."
            )
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
                    f"MPPIConfig.{name}={duration:g}s is shorter than one step of "
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
                "MPPIConfig: neither step_time nor step_substeps is set; falling back to "
                f"step_substeps={_FALLBACK_STEP_SUBSTEPS}.",
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
                "MPPIConfig: neither time_horizon nor step_horizon is set; falling back to "
                f"step_horizon={_FALLBACK_STEP_HORIZON}.",
                stacklevel=2,
            )
            horizon = _FALLBACK_STEP_HORIZON

        return horizon, substeps


# ---------------------------------------------------------------------------
# Module-level Warp Kernels
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
    """Resample the MPPI noise block in place on the GPU.

    Each element gets its own RNG stream keyed by (seed, flat index), so the
    draw is reproducible for a given seed and decorrelated across seeds. Written
    in place: eps is read by _add_noise_and_clip_kernel outside any CUDA graph,
    so in-place writes are safe but the buffer must never be swapped.
    """
    n, h, u = wp.tid()
    tid   = (n * eps.shape[1] + h) * eps.shape[2] + u
    state = wp.rand_init(seed, tid)
    eps[n, h, u] = sigma * wp.randn(state)

@wp.kernel
def _add_noise_and_clip_kernel(
    U_mean:     wp.array2d(dtype=float),   # (H, nu)
    eps:        wp.array3d(dtype=float),   # (N, H, nu)
    ctrl_range: wp.array2d(dtype=float),   # (nu, 2)
    has_limits: wp.array(dtype=bool),      # (nu,)
    V_out:      wp.array3d(dtype=float),   # (N, H, nu)  [out]
):
    """Add noise to the mean action sequence and clip to actuator limits."""
    n, h, u = wp.tid()
    val = U_mean[h, u] + eps[n, h, u]
    if has_limits[u]:
        val = wp.clamp(val, ctrl_range[u, 0], ctrl_range[u, 1])
    V_out[n, h, u] = val

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


# ---------------------------------------------------------------------------
# GPU weight-computation kernels (keep weight calc entirely on device)
# ---------------------------------------------------------------------------

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
def _compute_weights_kernel(
    costs:   wp.array(dtype=float),
    beta:    wp.array(dtype=float),   # 1-element: min cost
    lam:     float,
    w_out:   wp.array(dtype=float),
):
    n = wp.tid()
    c = costs[n]
    if wp.isnan(c):
        # NaN rollout (sim blew up): contribute zero weight so valid rollouts
        # still drive the update.  wp.atomic_min already ignores NaN, so beta
        # already reflects only the valid minimum.
        w_out[n] = float(0.0)
    else:
        w_out[n] = wp.exp(-(c - beta[0]) / lam)

@wp.kernel
def _sum_reduce_kernel(
    arr:   wp.array(dtype=float),
    total: wp.array(dtype=float),   # 1-element accumulator
):
    n = wp.tid()
    wp.atomic_add(total, 0, arr[n])

@wp.kernel
def _normalize_kernel(
    arr:   wp.array(dtype=float),
    total: wp.array(dtype=float),   # 1-element sum (before adding eps)
):
    n = wp.tid()
    arr[n] = arr[n] / (total[0] + float(1e-8))

@wp.kernel
def _debug_stats_kernel(
    w:     wp.array(dtype=float),   # (N,) NORMALIZED weights (sum_n w[n] = 1)
    costs: wp.array(dtype=float),   # (N,) folded per-sample trajectory cost
    out:   wp.array(dtype=float),   # 3-element: [sum w^2, sum cost, n_valid]
):
    """Debug-only reductions, both accumulated in a single pass over the samples.

    out[0] = sum_n w[n]^2       -> effective sample size ESS = 1 / out[0]
    out[1] = sum of non-NaN costs
    out[2] = count of non-NaN costs   -> average cost = out[1] / out[2]

    NaN rollouts (sim blew up) are skipped in the cost sum so one bad sample
    can't poison the reported mean; their weight is already exactly zero
    (_compute_weights_kernel), so they contribute nothing to the w^2 term.
    Launched only when the debug flag is set — nothing here feeds the update.
    """
    n = wp.tid()
    wp.atomic_add(out, 0, w[n] * w[n])
    if wp.isnan(costs[n]):
        return
    wp.atomic_add(out, 1, costs[n])
    wp.atomic_add(out, 2, float(1.0))

@wp.kernel
def _weighted_mean_kernel(
    weights: wp.array(dtype=float),    # (N,)
    samples: wp.array3d(dtype=float),  # (N, H, nu)  clamped control samples V
    out:     wp.array2d(dtype=float),  # (H, nu)  [out]  new mean (replaced)
    N:       int,
):
    """Weighted mean of the sample trajectories: out[h,u] = sum_n w[n]*V[n,h,u].

    One thread per (h, u); inner loop over N avoids atomic collisions. With
    normalized weights (sum_n w[n] = 1) this is a convex combination, so when the
    samples V are clamped to delta_range the resulting mean is clamped too. Reads
    `samples` and writes a separate `out`, so it is safe to pass U itself as
    `out` (full replacement, no read-before-write hazard)."""
    h, u = wp.tid()
    val = float(0.0)
    for n in range(N):
        val = val + weights[n] * samples[n, h, u]
    out[h, u] = val

@wp.kernel
def _shift_U_kernel(
    src: wp.array2d(dtype=float),  # (H, nu)
    dst: wp.array2d(dtype=float),  # (H, nu)  [out]
    H:   int,
):
    """Warm-start shift: dst[h] = src[h+1], with the last row zeroed.

    Written into a separate buffer (not in place) so adjacent rows can't race
    on the read-before-write; the caller swaps src/dst afterward."""
    h, u = wp.tid()
    if h < H - 1:
        dst[h, u] = src[h + 1, u]
    else:
        dst[h, u] = 0.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MPPIController:
    """MPPI controller backed by a contact model."""

    def __init__(
        self,
        task,
        cfg:      ContactModelConfig,
        mppi_cfg: MPPIConfig,
        rng:      np.random.Generator | None = None,
    ):
        # The (ROLLOUT) task is a thin adapter supplying the planning model and
        # the cost arrays. The GPU rollout env, CUDA graphs and plan() below are
        # unchanged — they only ever read self.mjm / self.cost_fn_wp_func /
        # self.goal_wp / self.indices_wp / self.weights_wp (the same five objects
        # that used to be passed loosely), so behavior is identical.
        self.task = task
        self.mjm = task.mjm
        self.cfg = cfg
        self.pc  = mppi_cfg
        self.rng = rng or np.random.default_rng()
        self.lam = self.pc.temperature

        # Rollout schedule in whole steps, quantized against the planning
        # model's timestep (the driver stamps rollout_dt onto mjm before
        # constructing the controller). self.horizon / self.substeps are the
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
        # keys the RNG seed), so no two resamples in this controller's lifetime
        # can replay the same noise block.
        self._plan_count     = 0
        self._resample_count = 0
        # Rollout steps actually taken by the last plan() — < horizon only when
        # the time-constrained path truncated.
        self.last_n_steps    = self.horizon

        self._setup_warp_arrays()
        self._setup_warp_backend()

    def _setup_warp_arrays(self):
        """Allocate Warp arrays on GPU once (avoids runtime host copies)."""
        N, H, nu = self.pc.n_samples, self.horizon, self.nu

        # Pre-allocate reset buffers
        self.qpos_reset = wp.empty(self.nq, dtype=wp.float32, device="cuda")
        self.qvel_reset = wp.empty(self.nv, dtype=wp.float32, device="cuda")
        self.ctrl_reset = wp.empty(self.nu, dtype=wp.float32, device="cuda")

        self.U_wp = wp.zeros((H, nu), dtype=wp.float32, device="cuda")
        # Scratch buffer for the on-GPU warm-start shift (swapped with U_wp).
        self.U_shift_wp = wp.zeros((H, nu), dtype=wp.float32, device="cuda")
        self.V_wp = wp.zeros((N, H, nu), dtype=wp.float32, device="cuda")
        # Running (per-step) cost sum and the terminal cost are accumulated into
        # separate buffers so the running sum can be horizon-normalized without
        # dividing the single-step terminal cost; _gpu_weight_update folds them.
        self.costs_wp = wp.zeros(N, dtype=wp.float32, device="cuda")
        self.terminal_costs_wp = wp.zeros(N, dtype=wp.float32, device="cuda")

        # GPU arrays for weight computation (costs never transferred to CPU)
        self.w_wp        = wp.zeros(N,        dtype=wp.float32, device="cuda")
        self.min_cost_wp = wp.zeros(1,        dtype=wp.float32, device="cuda")
        self.eta_wp      = wp.zeros(1,        dtype=wp.float32, device="cuda")
        # [sum w^2, sum cost, n_valid] — written only on debug plans.
        self._dbg_wp     = wp.zeros(3,        dtype=wp.float32, device="cuda")
        self._dbg_ess      = float("nan")
        self._dbg_avg_cost = float("nan")
        self._big_float_np = np.array([1e30], dtype=np.float32)

        # Actuator limits
        delta_low, delta_high = self.pc.delta_range
        delta_range_np = np.empty((self.nu, 2), dtype=np.float32)
        delta_range_np[:, 0] = delta_low
        delta_range_np[:, 1] = delta_high

        self._ctrl_range_wp = wp.array(delta_range_np, dtype=wp.float32, device="cuda")
        self._has_limits_wp = wp.array(np.ones(self.nu, dtype=bool), dtype=wp.bool, device="cuda")

        # Deterministic noise: use seed from config if provided, else fall back to shared rng
        noise_rng = np.random.default_rng(self.pc.seed) if self.pc.seed is not None else self.rng

        if self.pc.resample_interval is not None:
            # Resampling path: the block is drawn on the GPU by _maybe_resample_noise,
            # so only an int base seed is taken from noise_rng here. The first plan()
            # call always resamples, so these zeros are never actually rolled out.
            self._noise_seed = int(noise_rng.integers(0, 2**31 - 1))
            self._static_eps_wp = wp.zeros((N, H, nu), dtype=wp.float32, device="cuda")
            return

        if self.pc.use_spline_noise:
            t_knots    = np.linspace(0, H - 1, self.pc.n_spline_points)
            t_dense    = np.arange(H)
            knot_noise = noise_rng.normal(
                0, self.pc.noise_sigma, (N, self.pc.n_spline_points, nu)
            ).astype(np.float32)
            static_eps_np = np.empty((N, H, nu), dtype=np.float32)
            for n in range(N):
                for j in range(nu):
                    static_eps_np[n, :, j] = CubicSpline(t_knots, knot_noise[n, :, j])(t_dense)
        else:
            static_eps_np = noise_rng.normal(
                loc=0.0, scale=self.pc.noise_sigma, size=(N, H, nu)
            ).astype(np.float32)

        self._static_eps_wp = wp.array(static_eps_np, dtype=wp.float32, device="cuda")

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
            wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nq), inputs=[self.qpos_reset, self.d.qpos])
            wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nv), inputs=[self.qvel_reset, self.d.qvel])
            wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nu), inputs=[self.ctrl_reset, self.d.ctrl])
            self.costs_wp.zero_()
            self.terminal_costs_wp.zero_()
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
            wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nq), inputs=[self.qpos_reset, self.d.qpos])
            wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nv), inputs=[self.qvel_reset, self.d.qvel])
            wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nu), inputs=[self.ctrl_reset, self.d.ctrl])
            self.costs_wp.zero_()
            self.terminal_costs_wp.zero_()

            for t in range(self.horizon):
                terminal = (t == self.horizon - 1)
                self._launch_assign_ctrl(t)
                for _ in range(self.substeps):
                    api.step(self.m, self.d)
                # Terminal cost goes to its own buffer so it can be excluded from
                # the horizon normalization in _gpu_weight_update.
                wp.launch(
                    self._accumulate_costs_kernel,
                    dim=self.pc.n_samples,
                    inputs=[
                        self.d.qpos, self.d.qvel, self.d.ctrl, self.d.site_xpos, self.d.site_xmat,
                        terminal, self.goal_wp, self.indices_wp, self.weights_wp
                    ],
                    outputs=[self.terminal_costs_wp if terminal else self.costs_wp],
                )

        return capture.graph

    def reset(self):
        """Clear the action sequence (call at the start of a new episode)."""
        self.U_wp.zero_()
        # Restart the resample cadence so a fresh goal begins on fresh noise.
        # _resample_count deliberately keeps advancing (it keys the seed).
        self._plan_count = 0

    def _launch_assign_ctrl(self, t: int):
        """Write the rollout controls for horizon step t into d.ctrl.

        ctrl_relative_to_qpos selects the parameterization (see MPPIConfig):
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

    def _maybe_resample_noise(self):
        """Redraw the noise block every `resample_interval` plan() calls (no-op if None)."""
        k = self.pc.resample_interval
        if k is not None and self._plan_count % k == 0:
            wp.launch(
                _sample_gaussian_noise_kernel,
                dim=(self.pc.n_samples, self.horizon, self.nu),
                inputs=[self._noise_seed + self._resample_count, self.pc.noise_sigma],
                outputs=[self._static_eps_wp],
            )
            self._resample_count += 1
        self._plan_count += 1

    def _gpu_weight_update(self, n_eff: int | None = None) -> tuple[float, float]:
        """Compute MPPI weights and U update entirely on GPU.

        Returns (eta, beta) as Python scalars — the only values transferred
        from device to host. eta is used for adaptive temperature; beta
        (min cost) is returned for optional debug logging.

        n_eff caps the update at the rollout steps that were actually simulated
        (time-constrained path). Rows beyond it never influenced a cost, so
        weighting their noise would inject variance from weights that never saw
        them; they are left untouched. None means the full horizon.
        """
        N, nu = self.pc.n_samples, self.nu
        H = n_eff if n_eff is not None else self.horizon

        # Fold the terminal cost into the running-cost sum and normalize. The
        # combined total (running + terminal) is divided by the horizon so the
        # MPPI temperature `lam` is invariant to horizon — a mean over the whole
        # trajectory, matching irisim_warp; the total is then divided by the
        # sample count for invariance to n_samples. Every sample in this update
        # ran the same number of steps, so this is a uniform rescale that leaves
        # the relative sample weighting unchanged; it also rescales the reported
        # min cost `beta`. costs_wp holds the folded total afterward.
        running_scale = 1.0
        if self.pc.mean_cost_over_horizon and self.horizon > 0:
            running_scale /= float(self.horizon)
        total_scale = 1.0
        if self.pc.normalize_cost_by_samples and N > 0:
            total_scale /= float(N)
        wp.launch(
            _combine_costs_kernel, dim=N,
            inputs=[self.costs_wp, self.terminal_costs_wp, running_scale, total_scale],
        )

        # Find minimum cost
        self.min_cost_wp.assign(self._big_float_np)
        wp.launch(_find_min_kernel, dim=N, inputs=[self.costs_wp, self.min_cost_wp])

        # Compute unnormalized weights: w[n] = exp(-(cost[n] - beta) / lam)
        wp.launch(_compute_weights_kernel, dim=N,
                  inputs=[self.costs_wp, self.min_cost_wp, self.lam, self.w_wp])

        # Sum weights for normalization and adaptive temperature
        self.eta_wp.zero_()
        wp.launch(_sum_reduce_kernel, dim=N, inputs=[self.w_wp, self.eta_wp])

        # Normalize weights in-place
        wp.launch(_normalize_kernel, dim=N, inputs=[self.w_wp, self.eta_wp])

        # Debug-only stats (ESS + mean cost). Skipped entirely unless debug is
        # on, and folded into the sync below so it costs no extra device→host
        # round-trip. Must come after the normalize kernel: ESS is defined on
        # the normalized weights.
        if self.pc.debug:
            self._dbg_wp.zero_()
            wp.launch(_debug_stats_kernel, dim=N,
                      inputs=[self.w_wp, self.costs_wp, self._dbg_wp])

        # Replace the mean with the weighted average of the CLAMPED samples:
        # U[h,u] = sum_n w[n] * V[n,h,u], where V = clamp(U + eps, delta_range).
        # Because every V is clamped to delta_range and the weights form a convex
        # combination (sum_n w[n] = 1), the new mean — and the returned action
        # U[0] — is intrinsically bounded to delta_range, so the clamp is always
        # in force. (The previous update, U += clamp(sum_n w[n]*eps), clamped only
        # the increment and let U integrate past delta_range across plans.)
        wp.launch(_weighted_mean_kernel, dim=(H, nu),
                  inputs=[self.w_wp, self.V_wp, self.U_wp, N])

        # Single sync: only scalars cross device→host boundary
        wp.synchronize()
        eta  = float(self.eta_wp.numpy()[0]) + 1e-8
        beta = float(self.min_cost_wp.numpy()[0])
        if self.pc.debug:
            sum_w2, sum_cost, n_valid = (float(v) for v in self._dbg_wp.numpy())
            self._dbg_ess      = 1.0 / (sum_w2 + 1e-8)
            self._dbg_avg_cost = sum_cost / n_valid if n_valid > 0 else float("nan")
        return eta, beta

    def _update_adaptive_temp(self, eta: float):
        if self.pc.adaptive_temp:
            if eta > self.pc.adp_temp_params[0]:
                self.lam = self.pc.adp_temp_params[2] * self.lam
            elif eta < self.pc.adp_temp_params[1]:
                self.lam = self.pc.adp_temp_params[3] * self.lam

    def _print_debug(self, beta: float, eta: float):
        """ESS and avg cost come from _debug_stats_kernel, which the last
        _gpu_weight_update only launched because this same debug flag is set."""
        print(
            f"min cost: {beta:.4f}  "
            f"avg cost: {self._dbg_avg_cost:.4f}  "
            f"eta: {eta:.4f}  "
            f"ESS: {self._dbg_ess:.1f}/{self.pc.n_samples}"
        )

    def _extract_action(self) -> np.ndarray:
        # Only the first action crosses device→host; the warm-start shift stays
        # on the GPU (no full (H,nu) round-trip). U_wp is never baked into a
        # captured graph, so swapping the buffer reference is safe.
        action_np = self.U_wp[0].numpy().copy()
        if self.pc.warm_start:
            H, nu = self.horizon, self.nu
            wp.launch(
                _shift_U_kernel,
                dim=(H, nu),
                inputs=[self.U_wp, self.U_shift_wp, H],
            )
            self.U_wp, self.U_shift_wp = self.U_shift_wp, self.U_wp
        return action_np

    # Costs at or above this value mean wp.atomic_min was never written (all NaN rollouts).
    _COST_SENTINEL = 1e9

    def _is_degenerate(self, beta: float, eta: float) -> bool:
        """True only when *all* rollout costs were NaN (beta never left the sentinel).

        Partial NaN is handled upstream in _compute_weights_kernel, which zeroes
        out NaN-cost weights so valid rollouts still drive the update.  This check
        fires only when no valid rollout exists at all.
        """
        return beta >= self._COST_SENTINEL or math.isnan(eta) or math.isinf(eta)

    def plan(self, mjd: mujoco.MjData) -> np.ndarray:
        """Run MPPI and return the first action.

        Dispatches on time_constrained, then use_full_graph.
        """
        self._maybe_resample_noise()
        if self.pc.time_constrained:
            return self._plan_time_constrained(mjd)
        if self.pc.use_full_graph:
            return self._plan_full_graph(mjd)
        else:
            return self._plan_step_graphs(mjd)

    def _plan_full_graph(self, mjd: mujoco.MjData) -> np.ndarray:
        """Single mega CUDA graph: reset + full H-step unroll captured together."""
        N, H = self.pc.n_samples, self.horizon

        self.qpos_reset.assign(mjd.qpos)
        self.qvel_reset.assign(mjd.qvel)
        self.ctrl_reset.assign(mjd.ctrl)

        for _ in range(self.pc.n_iterations):
            wp.launch(
                _add_noise_and_clip_kernel,
                dim=(N, H, self.nu),
                inputs=[self.U_wp, self._static_eps_wp, self._ctrl_range_wp, self._has_limits_wp],
                outputs=[self.V_wp],
            )
            wp.capture_launch(self.rollout_graph)

            eta, beta = self._gpu_weight_update()

            if self.pc.debug:
                self._print_debug(beta, eta)

            if self._is_degenerate(beta, eta):
                # All rollouts produced NaN costs; weight update may have written
                # NaN into U_wp — reset it so the next call starts clean.
                self.U_wp.zero_()
                if self.pc.debug:
                    print(f"  [MPPI] all rollouts NaN (beta={beta:.2e}, eta={eta}) — zero action")
                return np.zeros(self.nu, dtype=np.float32)

            self._update_adaptive_temp(eta)

        return self._extract_action()

    def _plan_step_graphs(self, mjd: mujoco.MjData) -> np.ndarray:
        """Separate reset + step graphs: reset once, then loop over H steps."""
        N, H = self.pc.n_samples, self.horizon

        self.qpos_reset.assign(mjd.qpos)
        self.qvel_reset.assign(mjd.qvel)
        self.ctrl_reset.assign(mjd.ctrl)

        for _ in range(self.pc.n_iterations):
            wp.launch(
                _add_noise_and_clip_kernel,
                dim=(N, H, self.nu),
                inputs=[self.U_wp, self._static_eps_wp, self._ctrl_range_wp, self._has_limits_wp],
                outputs=[self.V_wp],
            )
            wp.capture_launch(self.reset_graph)

            for t in range(H):
                terminal = (t == H - 1)
                self._launch_assign_ctrl(t)
                # Launch one-step graph substeps times — keeps the graph small
                # regardless of substep count (unlike the full-graph path which
                # bakes all substeps into a single captured graph).
                for _ in range(self.substeps):
                    wp.capture_launch(self.step_graph)
                # Terminal cost goes to its own buffer (see create_rollout_graph).
                wp.launch(
                    self._accumulate_costs_kernel,
                    dim=N,
                    inputs=[
                        self.d.qpos, self.d.qvel, self.d.ctrl, self.d.site_xpos, self.d.site_xmat,
                        terminal, self.goal_wp, self.indices_wp, self.weights_wp
                    ],
                    outputs=[self.terminal_costs_wp if terminal else self.costs_wp],
                )

            eta, beta = self._gpu_weight_update()

            if self.pc.debug:
                self._print_debug(beta, eta)

            if self._is_degenerate(beta, eta):
                self.U_wp.zero_()
                if self.pc.debug:
                    print(f"  [MPPI] all rollouts NaN (beta={beta:.2e}, eta={eta}) — zero action")
                return np.zeros(self.nu, dtype=np.float32)

            self._update_adaptive_temp(eta)

        return self._extract_action()

    def _plan_time_constrained(self, mjd: mujoco.MjData) -> np.ndarray:
        """Step rollouts against a wall-clock budget, capped at the horizon.

        Unrolls like _plan_step_graphs but stops as soon as plan_budget_ms has
        elapsed, then finishes the current step and computes the action from the
        steps that did run. The deadline is only tested *after* a completed step,
        so a too-small budget degrades to a 1-step horizon rather than no action.
        """
        N, H = self.pc.n_samples, self.horizon
        budget_s = self.pc.plan_budget_ms * 1e-3

        self.qpos_reset.assign(mjd.qpos)
        self.qvel_reset.assign(mjd.qvel)
        self.ctrl_reset.assign(mjd.ctrl)

        for _ in range(self.pc.n_iterations):
            wp.launch(
                _add_noise_and_clip_kernel,
                dim=(N, H, self.nu),
                inputs=[self.U_wp, self._static_eps_wp, self._ctrl_range_wp, self._has_limits_wp],
                outputs=[self.V_wp],
            )
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
                # The terminal step's cost goes to its own buffer so horizon
                # normalization in _gpu_weight_update skips it (see above).
                wp.launch(
                    self._accumulate_costs_kernel,
                    dim=N,
                    inputs=[
                        self.d.qpos, self.d.qvel, self.d.ctrl, self.d.site_xpos, self.d.site_xmat,
                        last, self.goal_wp, self.indices_wp, self.weights_wp
                    ],
                    outputs=[self.terminal_costs_wp if last else self.costs_wp],
                )
                if last:
                    break

            self.last_n_steps = n_eff
            eta, beta = self._gpu_weight_update(n_eff=n_eff)

            if self.pc.debug:
                print(f"  [MPPI] time-constrained: {n_eff}/{H} steps in "
                      f"{self.pc.plan_budget_ms:.1f} ms budget")
                self._print_debug(beta, eta)

            if self._is_degenerate(beta, eta):
                self.U_wp.zero_()
                if self.pc.debug:
                    print(f"  [MPPI] all rollouts NaN (beta={beta:.2e}, eta={eta}) — zero action")
                return np.zeros(self.nu, dtype=np.float32)

            self._update_adaptive_temp(eta)

        return self._extract_action()
