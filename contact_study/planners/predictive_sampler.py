"""Predictive sampler — greedy best-of-N sampling-based MPC (MJPC-style).

The simplest instance of the sampling-based predictive control loop: perturb the
nominal control sequence N-1 times, roll all N candidates out in parallel on the
GPU, and take the single lowest-cost one as the new nominal.

    theta <- argmin_i J(U^(i), x0)

Sample 0 is the *unperturbed* nominal (`include_nominal`, on by default), so the
update can never move theta to a sequence that scored worse than the one it
started from — the greedy step is monotone with respect to the sampled costs.
With `n_iterations > 1` this becomes iterative greedy refinement inside a single
plan() call, each iteration re-centred on the current best.

Everything else — schedule, rollout, per-step delta / servo control
parameterization, delta_range clamp, warm-start shift — is inherited from
`SamplingPlanner` and is identical to MPPI's, so the only difference between the
two planners is the update rule (argmin here, softmax-weighted mean there).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from contact_study.contact_models.config import ContactModelConfig
from contact_study.planners.base import PlannerConfig, SamplingPlanner


@dataclass
class PredictiveSamplerConfig(PlannerConfig):
    """Predictive-sampler knobs on top of the shared PlannerConfig.

    The sampling scale is `noise_sigma` and the candidate count is `n_samples`,
    both inherited — this planner adds only the nominal-inclusion switch.
    """
    # Keep sample 0 unperturbed so argmin can always fall back on the current
    # nominal. Turn off to make the sampling identical to MPPI's (every sample
    # perturbed), at the cost of losing the monotonicity guarantee.
    include_nominal: bool = True


# ---------------------------------------------------------------------------
# Predictive-sampler Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _ps_build_samples_kernel(
    U_mean:     wp.array2d(dtype=float),   # (H, nu)  nominal
    eps:        wp.array3d(dtype=float),   # (N, H, nu)
    ctrl_range: wp.array2d(dtype=float),   # (nu, 2)
    has_limits: wp.array(dtype=bool),      # (nu,)
    n_nominal:  int,                       # leading samples left unperturbed
    V_out:      wp.array3d(dtype=float),   # (N, H, nu)  [out]
):
    """V[n] = clamp(U + eps[n]), except the first n_nominal rows, which are clamp(U).

    Clamping the nominal too (rather than copying it verbatim) keeps every row
    of V on the same footing: whichever one argmin picks becomes the new theta
    and is therefore already inside delta_range."""
    n, h, u = wp.tid()
    val = U_mean[h, u]
    if n >= n_nominal:
        val = val + eps[n, h, u]
    if has_limits[u]:
        val = wp.clamp(val, ctrl_range[u, 0], ctrl_range[u, 1])
    V_out[n, h, u] = val


@wp.kernel
def _argmin_index_kernel(
    costs:    wp.array(dtype=float),   # (N,) folded per-sample trajectory cost
    min_cost: wp.array(dtype=float),   # 1-element: the minimum, already reduced
    best_idx: wp.array(dtype=int),     # 1-element [out], pre-set to N
):
    """Recover the index of the minimum-cost sample.

    Ties resolve to the LOWEST index via atomic_min, so when a perturbation only
    matches the nominal's cost the nominal (sample 0) wins and theta stays put.
    NaN costs never compare equal, so they can never be selected."""
    n = wp.tid()
    if costs[n] == min_cost[0]:
        wp.atomic_min(best_idx, 0, n)


@wp.kernel
def _copy_best_kernel(
    samples:  wp.array3d(dtype=float),   # (N, H, nu)
    best_idx: wp.array(dtype=int),       # 1-element
    U_out:    wp.array2d(dtype=float),   # (H, nu)  [out]
):
    """theta <- the winning sample. Reads best_idx on the device, so the argmin
    never has to round-trip to the host."""
    h, u = wp.tid()
    U_out[h, u] = samples[best_idx[0], h, u]


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class PredictiveSampler(SamplingPlanner):
    """Greedy best-of-N predictive sampler backed by a contact model.

    Interface identical to MPPIController for easy swapping in experiments.
    """

    name = "PS"

    def __init__(
        self,
        task,
        cfg:    ContactModelConfig,
        ps_cfg: PredictiveSamplerConfig,
        rng:    np.random.Generator | None = None,
    ):
        super().__init__(task=task, cfg=cfg, planner_cfg=ps_cfg, rng=rng)

    def _setup_warp_arrays(self):
        super()._setup_warp_arrays()
        self.best_idx_wp = wp.zeros(1, dtype=wp.int32, device="cuda")
        # Sentinel for the atomic_min: any real sample index beats N.
        self._n_samples_np = np.array([self.pc.n_samples], dtype=np.int32)

    # -- planner hooks ------------------------------------------------------

    def _build_samples(self):
        n_nominal = 1 if self.pc.include_nominal else 0
        wp.launch(
            _ps_build_samples_kernel,
            dim=(self.pc.n_samples, self.horizon, self.nu),
            inputs=[self.U_wp, self._static_eps_wp, self._ctrl_range_wp,
                    self._has_limits_wp, n_nominal],
            outputs=[self.V_wp],
        )

    def _update_params(self, n_eff: int) -> bool:
        """theta <- argmin_i J^(i), entirely on the device."""
        N = self.pc.n_samples
        self._fold_costs()

        # Reduce to the minimum cost, then recover its index. _min_cost() carries
        # the one scalar sync of this update; it also tells us whether any valid
        # rollout exists (the sentinel survives only if every cost was NaN).
        beta = self._min_cost()
        if beta >= self._COST_SENTINEL or np.isnan(beta):
            if self.pc.debug:
                print(f"  [PS] all rollouts NaN (beta={beta:.2e}) — zero action")
            return False

        self.best_idx_wp.assign(self._n_samples_np)
        wp.launch(_argmin_index_kernel, dim=N,
                  inputs=[self.costs_wp, self.min_cost_wp, self.best_idx_wp])

        # Rows beyond n_eff never influenced a cost (time-constrained path), so
        # the nominal keeps its existing tail there.
        wp.launch(_copy_best_kernel, dim=(n_eff, self.nu),
                  inputs=[self.V_wp, self.best_idx_wp], outputs=[self.U_wp])

        if self.pc.debug:
            costs = self.costs_wp.numpy()
            valid = ~np.isnan(costs)
            nominal = f"  nominal: {float(costs[0]):.4f}" if self.pc.include_nominal else ""
            print(f"  [PS] min cost: {beta:.4f}  "
                  f"avg cost: {float(costs[valid].mean()):.4f}  "
                  f"best: {int(self.best_idx_wp.numpy()[0])}/{N}{nominal}")
        return True
