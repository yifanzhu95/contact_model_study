"""Model Predictive Path Integral (MPPI) controller.

All rollouts are executed in parallel on GPU via the batched step()
interface (nworld = N samples). The physics step and state resets are
encapsulated in CUDA graphs, eliminating slow CPU-side data resets
during the MPPI loop.

The rollout engine itself (schedule quantization, Warp arrays, graph capture,
the unroll, the control parameterization, the warm-start shift) lives in
`contact_study.planners.base.SamplingPlanner` and is shared with the CEM and
predictive-sampler planners. What is MPPI-specific and lives here: sampling
Gaussian perturbations of the running mean, and updating that mean with the
softmax(-cost/lambda)-weighted average of the sampled trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.interpolate import CubicSpline
import numpy as np
import warp as wp

from contact_study.contact_models.config import ContactModelConfig
from contact_study.planners.base import (
    PlannerConfig,
    SamplingPlanner,
    # Re-exported for backward compatibility with code that reached into this
    # module for the shared kernels/constants before they moved to base.py.
    _FALLBACK_STEP_HORIZON,    # noqa: F401
    _FALLBACK_STEP_SUBSTEPS,   # noqa: F401
    _RATIO_TOL,                # noqa: F401
    _broadcast_1d_to_2d_kernel,   # noqa: F401
    _sample_gaussian_noise_kernel,  # noqa: F401
    _assign_ctrl_kernel,          # noqa: F401
    _assign_ctrl_relative_kernel, # noqa: F401
    _make_accumulate_kernel,      # noqa: F401
    _combine_costs_kernel,        # noqa: F401
    _find_min_kernel,             # noqa: F401
    _shift_2d_kernel,             # noqa: F401
)


@dataclass
class MPPIConfig(PlannerConfig):
    """MPPI knobs on top of the shared PlannerConfig (schedule, noise, graphs)."""
    temperature:     float = 1.0    # lambda: MPPI temperature (inverse temp)
    adaptive_temp:   bool  = False
    adp_temp_params: tuple[float, float, float, float] = (10.0, 5.0, 0.9, 1.1)
    use_spline_noise:bool  = False   # toggle between spline and Gaussian noise
    n_spline_points: int   = 5      # control points for spline-smoothed noise
    # Average the total trajectory cost (running + terminal) across the horizon
    # length before the weight computation, so `temperature` is invariant to
    # horizon. Unlike the old normalize_cost_by_horizon (which divided only the
    # running sum and left the terminal cost un-normalized), the terminal cost is
    # included in the mean here, matching irisim_warp's `total_cost / horizon`.
    mean_cost_over_horizon: bool = False
    # Normalize the total trajectory cost (running + terminal) by the number of
    # samples before the weight computation, so `temperature` is invariant to
    # n_samples.
    normalize_cost_by_samples: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.use_spline_noise and self.resample_interval is not None:
            raise ValueError(
                "use_spline_noise is incompatible with resample_interval: spline noise is "
                "built host-side with CubicSpline and has no GPU sampler, so resampling it "
                "every step would defeat the purpose. Use Gaussian noise to resample."
            )


# ---------------------------------------------------------------------------
# MPPI-specific Warp kernels (the shared ones live in base.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MPPIController(SamplingPlanner):
    """MPPI controller backed by a contact model."""

    name = "MPPI"

    def __init__(
        self,
        task,
        cfg:      ContactModelConfig,
        mppi_cfg: MPPIConfig,
        rng:      np.random.Generator | None = None,
    ):
        super().__init__(task=task, cfg=cfg, planner_cfg=mppi_cfg, rng=rng)
        self.lam = self.pc.temperature

    def _setup_warp_arrays(self):
        super()._setup_warp_arrays()
        N = self.pc.n_samples
        # GPU arrays for weight computation (costs never transferred to CPU)
        self.w_wp   = wp.zeros(N, dtype=wp.float32, device="cuda")
        self.eta_wp = wp.zeros(1, dtype=wp.float32, device="cuda")
        # [sum w^2, sum cost, n_valid] — written only on debug plans.
        self._dbg_wp     = wp.zeros(3, dtype=wp.float32, device="cuda")
        self._dbg_ess      = float("nan")
        self._dbg_avg_cost = float("nan")

    def _draw_static_noise(self, noise_rng, N: int, H: int, nu: int) -> np.ndarray:
        """Spline-smoothed noise when configured; otherwise the base Gaussian block."""
        if not self.pc.use_spline_noise:
            return super()._draw_static_noise(noise_rng, N, H, nu)

        t_knots    = np.linspace(0, H - 1, self.pc.n_spline_points)
        t_dense    = np.arange(H)
        knot_noise = noise_rng.normal(
            0, self.pc.noise_sigma, (N, self.pc.n_spline_points, nu)
        ).astype(np.float32)
        static_eps_np = np.empty((N, H, nu), dtype=np.float32)
        for n in range(N):
            for j in range(nu):
                static_eps_np[n, :, j] = CubicSpline(t_knots, knot_noise[n, :, j])(t_dense)
        return static_eps_np

    # -- planner hooks ------------------------------------------------------

    def _build_samples(self):
        """V = clamp(U + eps): Gaussian perturbations of the running mean."""
        wp.launch(
            _add_noise_and_clip_kernel,
            dim=(self.pc.n_samples, self.horizon, self.nu),
            inputs=[self.U_wp, self._static_eps_wp, self._ctrl_range_wp, self._has_limits_wp],
            outputs=[self.V_wp],
        )

    def _update_params(self, n_eff: int) -> bool:
        eta, beta = self._gpu_weight_update(n_eff=n_eff)

        if self.pc.debug:
            self._print_debug(beta, eta)

        if self._is_degenerate(beta, eta):
            if self.pc.debug:
                print(f"  [MPPI] all rollouts NaN (beta={beta:.2e}, eta={eta}) — zero action")
            return False

        self._update_adaptive_temp(eta)
        return True

    def _reset_params(self):
        """Undo any adaptive-temperature drift so a new goal starts from lambda."""
        self.lam = self.pc.temperature

    # -- MPPI weight update -------------------------------------------------

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
        self._fold_costs(running_scale, total_scale)

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

    def _is_degenerate(self, beta: float, eta: float) -> bool:
        """True only when *all* rollout costs were NaN (beta never left the sentinel).

        Partial NaN is handled upstream in _compute_weights_kernel, which zeroes
        out NaN-cost weights so valid rollouts still drive the update.  This check
        fires only when no valid rollout exists at all.
        """
        return beta >= self._COST_SENTINEL or math.isnan(eta) or math.isinf(eta)
