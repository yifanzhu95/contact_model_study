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

from scipy.interpolate import CubicSpline
import numpy as np
import warp as wp
import mujoco

from contact_study.contact_models import api
from contact_study.contact_models.config import ContactModelConfig


@dataclass
class MPPIConfig:
    n_samples:       int   = 1024   # N: number of candidate trajectories
    horizon:         int   = 50     # H: planning horizon (steps)
    temperature:     float = 1.0    # lambda: MPPI temperature (inverse temp)
    noise_sigma:     float = 0.01   # action noise std dev
    n_iterations:    int   = 1      # number of MPPI update iterations per call
    warm_start:      bool  = True   # shift action sequence one step forward
    nconmax:         int   = 200
    njmax:           int   = 500
    substeps:        int   = 1
    adaptive_temp:   bool  = False
    adp_temp_params: tuple[float, float, float, float] = (10.0, 5.0, 0.9, 1.1)
    use_spline_noise:bool  = False   # toggle between spline and Gaussian noise
    n_spline_points: int   = 5      # control points for spline-smoothed noise
    debug:           bool  = True
    delta_range:     tuple[float, float] = (-0.1, 0.1)
    use_full_graph:  bool  = True   # True=single mega CUDA graph, False=step+reset graphs
    seed:            int | None = None  # seed for deterministic noise sampling


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
def _weighted_sum_kernel(
    weights: wp.array(dtype=float),    # (N,)
    eps:     wp.array3d(dtype=float),  # (N, H, nu)
    dU:      wp.array2d(dtype=float),  # (H, nu)
    N:       int,
):
    """One thread per (h, u); inner loop over N avoids atomic collisions."""
    h, u = wp.tid()
    val = float(0.0)
    for n in range(N):
        val = val + weights[n] * eps[n, h, u]
    dU[h, u] = val

@wp.kernel
def _apply_dU_kernel(
    U:    wp.array2d(dtype=float),  # (H, nu)
    dU:   wp.array2d(dtype=float),  # (H, nu)
    low:  float,
    high: float,
):
    h, u = wp.tid()
    U[h, u] = U[h, u] + wp.clamp(dU[h, u], low, high)


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

        self.nu = self.mjm.nu
        self.nq = self.mjm.nq
        self.nv = self.mjm.nv

        self.cost_fn_wp_func = task.cost_fn_wp
        self.goal_wp = task.cost_goal_wp
        self.indices_wp = task.cost_idx_wp
        self.weights_wp = task.cost_weights_wp
        self._accumulate_costs_kernel = _make_accumulate_kernel(self.cost_fn_wp_func)

        self._setup_warp_arrays()
        self._setup_warp_backend()

    def _setup_warp_arrays(self):
        """Allocate Warp arrays on GPU once (avoids runtime host copies)."""
        N, H, nu = self.pc.n_samples, self.pc.horizon, self.nu

        # Pre-allocate reset buffers
        self.qpos_reset = wp.empty(self.nq, dtype=wp.float32, device="cuda")
        self.qvel_reset = wp.empty(self.nv, dtype=wp.float32, device="cuda")
        self.ctrl_reset = wp.empty(self.nu, dtype=wp.float32, device="cuda")

        self.U_wp = wp.zeros((H, nu), dtype=wp.float32, device="cuda")
        self.V_wp = wp.zeros((N, H, nu), dtype=wp.float32, device="cuda")
        self.costs_wp = wp.zeros(N, dtype=wp.float32, device="cuda")

        # GPU arrays for weight computation (costs never transferred to CPU)
        self.w_wp        = wp.zeros(N,        dtype=wp.float32, device="cuda")
        self.dU_wp       = wp.zeros((H, nu),  dtype=wp.float32, device="cuda")
        self.min_cost_wp = wp.zeros(1,        dtype=wp.float32, device="cuda")
        self.eta_wp      = wp.zeros(1,        dtype=wp.float32, device="cuda")
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

            for t in range(self.pc.horizon):
                terminal = (t == self.pc.horizon - 1)
                wp.launch(
                    _assign_ctrl_kernel,
                    dim=(self.pc.n_samples, self.nu),
                    inputs=[self.V_wp, t, self.d.ctrl],
                )
                for _ in range(self.pc.substeps):
                    api.step(self.m, self.d)
                wp.launch(
                    self._accumulate_costs_kernel,
                    dim=self.pc.n_samples,
                    inputs=[
                        self.d.qpos, self.d.qvel, self.d.ctrl, self.d.site_xpos, self.d.site_xmat,
                        terminal, self.goal_wp, self.indices_wp, self.weights_wp
                    ],
                    outputs=[self.costs_wp],
                )

        return capture.graph

    def reset(self):
        """Clear the action sequence (call at the start of a new episode)."""
        self.U_wp.zero_()

    def _gpu_weight_update(self) -> tuple[float, float]:
        """Compute MPPI weights and U update entirely on GPU.

        Returns (eta, beta) as Python scalars — the only values transferred
        from device to host. eta is used for adaptive temperature; beta
        (min cost) is returned for optional debug logging.
        """
        N, H, nu = self.pc.n_samples, self.pc.horizon, self.nu
        low, high = self.pc.delta_range

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

        # Compute weighted noise sum: dU[h,u] = sum_n w[n]*eps[n,h,u]
        self.dU_wp.zero_()
        wp.launch(_weighted_sum_kernel, dim=(H, nu),
                  inputs=[self.w_wp, self._static_eps_wp, self.dU_wp, N])

        # Clip dU and add to U, all on GPU
        wp.launch(_apply_dU_kernel, dim=(H, nu),
                  inputs=[self.U_wp, self.dU_wp, low, high])

        # Single sync: only scalars cross device→host boundary
        wp.synchronize()
        eta  = float(self.eta_wp.numpy()[0]) + 1e-8
        beta = float(self.min_cost_wp.numpy()[0])
        return eta, beta

    def _update_adaptive_temp(self, eta: float):
        if self.pc.adaptive_temp:
            if eta > self.pc.adp_temp_params[0]:
                self.lam = self.pc.adp_temp_params[2] * self.lam
            elif eta < self.pc.adp_temp_params[1]:
                self.lam = self.pc.adp_temp_params[3] * self.lam

    def _print_debug(self, mjd: mujoco.MjData, beta: float, eta: float):
        if self.indices_wp is not None and len(self.indices_wp) >= 1:
            idx      = self.indices_wp.numpy()
            obj_pos  = mjd.qpos[idx[0] : idx[0] + 3]
            obj_quat = mjd.qpos[idx[0] + 3 : idx[0] + 7]
            print(
                f"min cost: {beta:.4f}  "
                f"eta: {eta:.4f}  "
                f"lam: {self.lam:.6f}  "
                f"obj_pos: {obj_pos}  "
                f"obj_quat: {obj_quat}"
            )
        else:
            print(f"min cost: {beta:.4f}  eta: {eta:.4f}  lam: {self.lam:.6f}")

    def _extract_action(self) -> np.ndarray:
        action_np = self.U_wp[0].numpy().copy()
        if self.pc.warm_start:
            U_np      = self.U_wp.numpy()
            U_np[:-1] = U_np[1:]
            U_np[-1]  = 0.0
            self.U_wp.assign(U_np)
        return action_np

    # Costs at or above this value mean wp.atomic_min was never written (all NaN rollouts).
    _COST_SENTINEL = 1e29

    def _is_degenerate(self, beta: float, eta: float) -> bool:
        """True only when *all* rollout costs were NaN (beta never left the sentinel).

        Partial NaN is handled upstream in _compute_weights_kernel, which zeroes
        out NaN-cost weights so valid rollouts still drive the update.  This check
        fires only when no valid rollout exists at all.
        """
        return beta >= self._COST_SENTINEL or math.isnan(eta) or math.isinf(eta)

    def plan(self, mjd: mujoco.MjData) -> np.ndarray:
        """Run MPPI and return the first action. Dispatches based on use_full_graph."""
        if self.pc.use_full_graph:
            return self._plan_full_graph(mjd)
        else:
            return self._plan_step_graphs(mjd)

    def _plan_full_graph(self, mjd: mujoco.MjData) -> np.ndarray:
        """Single mega CUDA graph: reset + full H-step unroll captured together."""
        N, H = self.pc.n_samples, self.pc.horizon

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
                self._print_debug(mjd, beta, eta)

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
        N, H = self.pc.n_samples, self.pc.horizon

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
                wp.launch(
                    _assign_ctrl_kernel,
                    dim=(N, self.nu),
                    inputs=[self.V_wp, t, self.d.ctrl],
                )
                # Launch one-step graph substeps times — keeps the graph small
                # regardless of substep count (unlike the full-graph path which
                # bakes all substeps into a single captured graph).
                for _ in range(self.pc.substeps):
                    wp.capture_launch(self.step_graph)
                wp.launch(
                    self._accumulate_costs_kernel,
                    dim=N,
                    inputs=[
                        self.d.qpos, self.d.qvel, self.d.ctrl, self.d.site_xpos, self.d.site_xmat,
                        terminal, self.goal_wp, self.indices_wp, self.weights_wp
                    ],
                    outputs=[self.costs_wp],
                )

            eta, beta = self._gpu_weight_update()

            if self.pc.debug:
                self._print_debug(mjd, beta, eta)

            if self._is_degenerate(beta, eta):
                self.U_wp.zero_()
                if self.pc.debug:
                    print(f"  [MPPI] all rollouts NaN (beta={beta:.2e}, eta={eta}) — zero action")
                return np.zeros(self.nu, dtype=np.float32)

            self._update_adaptive_temp(eta)

        return self._extract_action()
