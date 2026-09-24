"""Model Predictive Path Integral control.

MPPI's update is a softmax-weighted mean: every sampled sequence gets weight
``exp(-(J - beta) / lambda)``, and the new mean is their weighted average.
Low-cost samples dominate; ``lambda`` sets how sharply.

Everything else — sampling, the rollout, cost accumulation, the control
parameterization, the warm start — lives in ``SamplingBasedPlannerBase``. This
file is the two hooks and the weight kernels.

Ported from ``contact_study/planners/mppi.py``, with the same normalization
options and the same NaN handling, against the new simulator/task interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ContactModelStudy.SamplingBasedPlanners.SamplingBasedPlannerBase import (
    SamplingBasedPlannerBase,
    SamplingBasedPlannerConfig,
)

# Stands in for "no finite minimum yet". wp.atomic_min never fires on a NaN, so
# a block of all-NaN costs leaves this untouched — which is exactly the test for
# a degenerate solve.
_COST_SENTINEL = 1e30


@dataclass
class MPPI_Config(SamplingBasedPlannerConfig):
    """MPPI knobs on top of the shared planner config.

    Attributes:
        temperature: ``lambda``. Small values concentrate the weight on the best
            samples (greedy); large values average more broadly.
        adaptive_temp: Scale ``lambda`` between plans to hold the weight sum in
            a band, so the update stays informative when costs change scale.
        adp_temp_params: ``(eta_high, eta_low, scale_down, scale_up)``. When the
            weight sum exceeds ``eta_high``, ``lambda *= scale_down``; below
            ``eta_low``, ``lambda *= scale_up``.
        mean_cost_over_horizon: Divide the total trajectory cost by the horizon
            before weighting, making ``temperature`` invariant to horizon. The
            terminal cost is included in the mean.
        normalize_cost_by_samples: Divide the total by ``N``, making
            ``temperature`` invariant to the sample count.
    """

    temperature: float = 1.0
    adaptive_temp: bool = False
    adp_temp_params: tuple[float, float, float, float] = (10.0, 5.0, 0.9, 1.1)
    mean_cost_over_horizon: bool = False
    normalize_cost_by_samples: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


class MPPI(SamplingBasedPlannerBase):
    """MPPI planner over a vectorized simulator.

    Example::

        sim  = VectorizedMujoco(task.getModelPath(), cfg, N=1024)
        task = CubeReorient(LeapReorientConfig(role=TaskRole.ROLLOUT))
        mppi = MPPI(sim, task, MPPI_Config(temperature=1.0, noise_sigma=0.05))
        u    = mppi.Plan(q, q_dot)
    """

    name = "MPPI"

    def __init__(self, simulator, task, config: MPPI_Config | None = None):
        """See ``SamplingBasedPlannerBase.__init__``; ``config`` is an ``MPPI_Config``."""
        super().__init__(simulator, task, config if config is not None else MPPI_Config())
        self.lam = self.config.temperature

        dev = self.device
        self.w_wp = wp.zeros(self.N, dtype=wp.float32, device=dev)
        self.eta_wp = wp.zeros(1, dtype=wp.float32, device=dev)
        self.min_cost_wp = wp.zeros(1, dtype=wp.float32, device=dev)
        self._sentinel = np.array([_COST_SENTINEL], dtype=np.float32)
        # [sum w^2, sum cost, n_valid] — filled only when debug is on.
        self._dbg_wp = wp.zeros(3, dtype=wp.float32, device=dev)
        self._unc_wp = wp.zeros(self.nu, dtype=wp.float32, device=dev)
        self._dbg_ess = float("nan")
        self._dbg_avg_cost = float("nan")
        # Diagnostics from the last update.
        self.last_min_cost = float("nan")
        self.last_eta = float("nan")

    # -- hooks ---------------------------------------------------------------
    def _buildSamples(self) -> None:
        """``V = clip(U + eps)``: Gaussian perturbations of the running mean."""
        wp.launch(
            _add_noise_and_clip_kernel,
            dim=(self.N, self.horizon, self.nu),
            inputs=[self.U_wp, self.eps_wp, self._clip_wp, 1 if self._has_clip else 0],
            outputs=[self.V_wp],
        )

    def _updateParams(self) -> bool:
        """Replace the mean with the softmax-weighted average of the samples.

        Returns ``False`` only when *every* rollout cost was NaN, which leaves
        the minimum at its sentinel: there is no finite cost to weight against,
        so there is no update to make.
        """
        eta, beta = self._weightUpdate()
        self.last_eta, self.last_min_cost = eta, beta

        if self.config.debug:
            print(f"  [MPPI] min {beta:.4f}  avg {self._dbg_avg_cost:.4f}  "
                  f"eta {eta:.4f}  ESS {self._dbg_ess:.1f}/{self.N}  lam {self.lam:.4f}")

        if beta >= _COST_SENTINEL:
            if self.config.debug:
                print(f"  [MPPI] all rollouts NaN (beta={beta:.2e}) — zero action")
            return False

        self._updateTemperature(eta)
        return True

    def _resetParams(self) -> None:
        """Undo adaptive-temperature drift so a new episode starts from lambda."""
        self.lam = self.config.temperature

    def _actionUncertainty(self) -> np.ndarray:
        """Weighted standard deviation of the samples' first actions.

        ``sigma[u] = sqrt(sum_n w[n] * (V[n,0,u] - U[0,u])^2)``, with the
        normalized weights the last update used and ``U[0] = sum_n w[n] V[n,0]``
        the mean it produced. Every control mode turns ``V[n,0]`` into a command
        by adding the same offset to all samples, so this is also the spread of
        the commands themselves.

        Near zero when a few samples dominate the weights (a confident plan, or
        a temperature so low that one sample wins). Near ``noise_sigma`` when
        the weights are close to uniform (the costs could not tell the samples
        apart).
        """
        wp.launch(_weighted_std_first_kernel, dim=self.nu,
                  inputs=[self.w_wp, self.V_wp, self.U_wp, self.N],
                  outputs=[self._unc_wp])
        return self._unc_wp.numpy().copy()

    # -- the update ----------------------------------------------------------
    def _weightUpdate(self) -> tuple[float, float]:
        """Compute the weights and the new mean on the device.

        Returns:
            ``(eta, beta)`` — the weight sum and the minimum cost. These two
            scalars are the only values that cross to the host here.
        """
        N, nu, H = self.N, self.nu, self.horizon

        # Fold terminal into running, with the requested invariances. Both are
        # uniform rescales, so they leave the relative sample weighting alone
        # and only change what `temperature` means.
        running_scale = 1.0 / H if (self.config.mean_cost_over_horizon and H > 0) else 1.0
        total_scale = 1.0 / N if (self.config.normalize_cost_by_samples and N > 0) else 1.0
        self._foldCosts(running_scale, total_scale)

        # beta = min cost. atomic_min ignores NaN, so the sentinel survives a
        # block of all-NaN costs and _updateParams reads that as degenerate.
        self.min_cost_wp.assign(self._sentinel)
        wp.launch(_find_min_kernel, dim=N, inputs=[self.costs_wp, self.min_cost_wp])

        # w = exp(-(J - beta) / lambda), NaN rollouts weighted zero.
        wp.launch(_compute_weights_kernel, dim=N,
                  inputs=[self.costs_wp, self.min_cost_wp, self.lam],
                  outputs=[self.w_wp])
        self.eta_wp.zero_()
        wp.launch(_sum_reduce_kernel, dim=N, inputs=[self.w_wp], outputs=[self.eta_wp])
        wp.launch(_normalize_kernel, dim=N, inputs=[self.eta_wp], outputs=[self.w_wp])

        if self.config.debug:
            self._dbg_wp.zero_()
            wp.launch(_debug_stats_kernel, dim=N,
                      inputs=[self.w_wp, self.costs_wp], outputs=[self._dbg_wp])

        # U[h,u] = sum_n w[n] * V[n,h,u]. The weights are normalized, so this is
        # a convex combination of clipped samples — the new mean, and therefore
        # the returned action, inherits delta_range without a second clamp.
        wp.launch(_weighted_mean_kernel, dim=(H, nu),
                  inputs=[self.w_wp, self.V_wp, N], outputs=[self.U_wp])

        # One sync; only scalars come back.
        wp.synchronize()
        eta = float(self.eta_wp.numpy()[0]) + 1e-8
        beta = float(self.min_cost_wp.numpy()[0])
        if self.config.debug:
            sum_w2, sum_cost, n_valid = (float(v) for v in self._dbg_wp.numpy())
            self._dbg_ess = 1.0 / (sum_w2 + 1e-8)
            self._dbg_avg_cost = sum_cost / n_valid if n_valid > 0 else float("nan")
        return eta, beta

    def _updateTemperature(self, eta: float) -> None:
        """Nudge ``lambda`` to keep the weight sum inside its band."""
        if not self.config.adaptive_temp:
            return
        eta_high, eta_low, down, up = self.config.adp_temp_params
        if eta > eta_high:
            self.lam *= down
        elif eta < eta_low:
            self.lam *= up


# ---------------------------------------------------------------------------
# MPPI kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _add_noise_and_clip_kernel(
    U: wp.array2d(dtype=float),      # (H, nu)  mean
    eps: wp.array3d(dtype=float),    # (N, H, nu)
    clip: wp.array2d(dtype=float),   # (nu, 2)
    has_clip: int,
    V: wp.array3d(dtype=float),      # (N, H, nu)  [out]
):
    """V = clip(U + eps)."""
    n, h, u = wp.tid()
    val = U[h, u] + eps[n, h, u]
    if has_clip == 1:
        val = wp.clamp(val, clip[u, 0], clip[u, 1])
    V[n, h, u] = val


@wp.kernel
def _find_min_kernel(costs: wp.array(dtype=float), min_val: wp.array(dtype=float)):
    """Minimum cost. atomic_min never fires on NaN, so NaNs are skipped."""
    n = wp.tid()
    wp.atomic_min(min_val, 0, costs[n])


@wp.kernel
def _compute_weights_kernel(
    costs: wp.array(dtype=float),
    beta: wp.array(dtype=float),     # 1 element: min cost
    lam: float,
    w: wp.array(dtype=float),        # (N,)  [out]
):
    """w = exp(-(J - beta) / lambda).

    A NaN rollout (the sim blew up) gets exactly zero weight, so the valid
    samples still drive the update instead of the whole plan being lost.
    """
    n = wp.tid()
    c = costs[n]
    if wp.isnan(c):
        w[n] = float(0.0)
    else:
        w[n] = wp.exp(-(c - beta[0]) / lam)


@wp.kernel
def _sum_reduce_kernel(arr: wp.array(dtype=float), total: wp.array(dtype=float)):
    n = wp.tid()
    wp.atomic_add(total, 0, arr[n])


@wp.kernel
def _normalize_kernel(total: wp.array(dtype=float), arr: wp.array(dtype=float)):
    """In-place normalize by the pre-epsilon sum."""
    n = wp.tid()
    arr[n] = arr[n] / (total[0] + float(1e-8))


@wp.kernel
def _debug_stats_kernel(
    w: wp.array(dtype=float),        # normalized weights
    costs: wp.array(dtype=float),
    out: wp.array(dtype=float),      # [sum w^2, sum cost, n_valid]
):
    """Diagnostics in one pass: ESS = 1/out[0], mean cost = out[1]/out[2].

    NaN costs are skipped in the sum so one blown-up sample cannot poison the
    reported mean; their weight is already exactly zero.
    """
    n = wp.tid()
    wp.atomic_add(out, 0, w[n] * w[n])
    if wp.isnan(costs[n]):
        return
    wp.atomic_add(out, 1, costs[n])
    wp.atomic_add(out, 2, float(1.0))


@wp.kernel
def _weighted_mean_kernel(
    w: wp.array(dtype=float),        # (N,)
    V: wp.array3d(dtype=float),      # (N, H, nu)
    N: int,
    U: wp.array2d(dtype=float),      # (H, nu)  [out]
):
    """U[h,u] = sum_n w[n] * V[n,h,u].

    One thread per (h, u) with the sum over N inside, which avoids atomic
    collisions. Reads V and writes U, so a full replacement is safe.
    """
    h, u = wp.tid()
    val = float(0.0)
    for n in range(N):
        val = val + w[n] * V[n, h, u]
    U[h, u] = val


@wp.kernel
def _weighted_std_first_kernel(
    w: wp.array(dtype=float),        # (N,)  normalized weights
    V: wp.array3d(dtype=float),      # (N, H, nu)
    U: wp.array2d(dtype=float),      # (H, nu)  weighted mean of V
    N: int,
    out: wp.array(dtype=float),      # (nu,)  [out]
):
    """out[u] = sqrt(sum_n w[n] * (V[n,0,u] - U[0,u])^2): spread of the first step."""
    u = wp.tid()
    mean = U[0, u]
    var = float(0.0)
    for n in range(N):
        d = V[n, 0, u] - mean
        var = var + w[n] * d * d
    out[u] = wp.sqrt(var)
