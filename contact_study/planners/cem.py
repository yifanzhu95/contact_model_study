"""Cross-Entropy Method (CEM) sampling-based MPC.

CEM maintains a Gaussian distribution over action sequences — a mean sequence
(the shared `U_wp`) and a *per-(step, actuator)* standard deviation — and
iteratively refits it to the elite (lowest-cost) sample set.

Everything except the sampling distribution and the refit comes from
`SamplingPlanner` (see contact_study/planners/base.py): the N candidates are
rolled out in parallel on the GPU as N worlds, with the same per-step delta /
servo control parameterization, delta_range clamp, schedule quantization and
warm-start shift as MPPI. Only the theta update differs — elite refit here,
softmax-weighted mean there.

The elite *selection* is the one host-side step: the N folded costs (N floats)
come back to the CPU for an argpartition, and only the k elite indices go back
to the device. The (N, H, nu) sample block never leaves the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from contact_study.contact_models.config import ContactModelConfig
from contact_study.planners.base import (
    PlannerConfig,
    SamplingPlanner,
    _shift_2d_kernel,
)


@dataclass
class CEMConfig(PlannerConfig):
    """CEM knobs on top of the shared PlannerConfig (schedule, noise, graphs).

    `noise_sigma` is the initial (and post-reset) standard deviation of every
    element of the action sequence; CEM adapts it from there.
    """
    # Elite set size. n_elites wins when set; otherwise k = elite_frac * n_samples.
    n_elites:     int | None = None
    elite_frac:   float = 0.1
    # Smoothing of the refit onto the running distribution:
    #   new = alpha * old + (1 - alpha) * elite_fit
    # so alpha=0 is a hard replacement and alpha->1 barely moves.
    alpha:        float = 0.1
    # Floor on the refit sigma, so a fully-agreeing elite set cannot drive the
    # scale to zero mid-optimization.
    min_sigma:    float = 1e-3
    # Re-open sigma to noise_sigma at the start of every plan() call. CEM's
    # variance only ever shrinks, so without this the distribution collapses
    # after a handful of control steps and never explores again for the rest of
    # the episode; with it, each plan() is a fresh anneal from a known scale,
    # warm-started on the running mean. Turn off for a single continuously
    # narrowing distribution across the whole episode.
    reopen_sigma_each_plan: bool = True
    # CEM is iterative by construction — a single iteration is just a
    # best-k-average of one Gaussian draw.
    n_iterations: int = 3

    def __post_init__(self):
        super().__post_init__()
        if self.n_elites is not None and self.n_elites < 1:
            raise ValueError(f"n_elites must be >= 1 or None, got {self.n_elites}")
        if self.n_elites is None and not (0.0 < self.elite_frac <= 1.0):
            raise ValueError(f"elite_frac must be in (0, 1], got {self.elite_frac}")
        if not (0.0 <= self.alpha < 1.0):
            raise ValueError(f"alpha must be in [0, 1), got {self.alpha}")
        if self.min_sigma < 0.0:
            raise ValueError(f"min_sigma must be >= 0, got {self.min_sigma}")


# ---------------------------------------------------------------------------
# CEM-specific Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _cem_build_samples_kernel(
    mu:         wp.array2d(dtype=float),   # (H, nu)
    sigma:      wp.array2d(dtype=float),   # (H, nu)
    z:          wp.array3d(dtype=float),   # (N, H, nu)  unit-variance normals
    ctrl_range: wp.array2d(dtype=float),   # (nu, 2)
    has_limits: wp.array(dtype=bool),      # (nu,)
    V_out:      wp.array3d(dtype=float),   # (N, H, nu)  [out]
):
    """V = clamp(mu + sigma * z), with sigma varying per (step, actuator).

    The scale is applied here rather than at draw time (as MPPI does) because
    CEM's sigma is a full (H, nu) array that changes every iteration, while the
    noise block holds plain unit-variance normals."""
    n, h, u = wp.tid()
    val = mu[h, u] + sigma[h, u] * z[n, h, u]
    if has_limits[u]:
        val = wp.clamp(val, ctrl_range[u, 0], ctrl_range[u, 1])
    V_out[n, h, u] = val


@wp.kernel
def _cem_refit_kernel(
    elite_idx: wp.array(dtype=int),       # (>=k,) sample indices of the elites
    samples:   wp.array3d(dtype=float),   # (N, H, nu)  clamped control samples V
    k:         int,                       # elites actually used
    alpha:     float,                     # smoothing toward the previous fit
    min_sigma: float,
    mu:        wp.array2d(dtype=float),   # (H, nu)  [in/out]
    sigma:     wp.array2d(dtype=float),   # (H, nu)  [in/out]
):
    """Refit the Gaussian to the elite samples, smoothed onto the running one.

    One thread per (h, u), looping over the k elites: no atomics, and each thread
    touches only its own mu/sigma element, so the in-place update is race-free.
    Every elite is a clamped sample, so the elite mean stays inside delta_range
    and the action read off mu is bounded exactly as MPPI's is.
    """
    h, u = wp.tid()

    total = float(0.0)
    for j in range(k):
        total = total + samples[elite_idx[j], h, u]
    mean = total / float(k)

    var = float(0.0)
    for j in range(k):
        d = samples[elite_idx[j], h, u] - mean
        var = var + d * d
    # Population std, matching numpy's default ddof=0; the epsilon keeps a
    # fully-collapsed elite set from producing an exactly-zero scale.
    std = wp.sqrt(var / float(k)) + float(1e-5)

    mu[h, u] = alpha * mu[h, u] + (1.0 - alpha) * mean
    sigma[h, u] = wp.max(alpha * sigma[h, u] + (1.0 - alpha) * std, min_sigma)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class CEMController(SamplingPlanner):
    """CEM controller backed by a contact model.

    Interface identical to MPPIController for easy swapping in experiments.
    """

    name = "CEM"

    def __init__(
        self,
        task,
        cfg:     ContactModelConfig,
        cem_cfg: CEMConfig,
        rng:     np.random.Generator | None = None,
    ):
        super().__init__(task=task, cfg=cfg, planner_cfg=cem_cfg, rng=rng)

    # -- setup --------------------------------------------------------------

    def _setup_warp_arrays(self):
        super()._setup_warp_arrays()
        N, H, nu = self.pc.n_samples, self.horizon, self.nu

        # Elite count, resolved once. n_elites wins over elite_frac; at least 1
        # elite and never more than the sample count.
        k = self.pc.n_elites if self.pc.n_elites is not None else round(self.pc.elite_frac * N)
        self.n_elites = int(min(max(k, 1), N))

        # theta = (mu, sigma). mu IS the base class's U_wp, so _extract_action
        # and the warm-start shift work unchanged.
        self.sigma_wp = wp.array(
            np.full((H, nu), self.pc.noise_sigma, dtype=np.float32),
            dtype=wp.float32, device="cuda",
        )
        self.sigma_shift_wp = wp.zeros((H, nu), dtype=wp.float32, device="cuda")
        # Sized N (not k) so a shrunken elite set — when some rollouts NaN'd —
        # can be uploaded without reallocating; the kernel reads only the first k.
        self.elite_idx_wp = wp.zeros(N, dtype=wp.int32, device="cuda")
        self._elite_idx_np = np.zeros(N, dtype=np.int32)
        # Elites the LAST refit actually used. Rows [k:] of elite_idx_wp are
        # padded with elite 0, so k cannot be recovered from the array — and a
        # reader that took the whole thing would double-count elite 0 (N-k)
        # times. Read by evaluation/distributions.py.
        self.last_n_elites = self.n_elites

    @property
    def mu_wp(self):
        """The distribution mean — an alias of the shared action sequence."""
        return self.U_wp

    # -- noise --------------------------------------------------------------
    # CEM scales by its per-(h, u) sigma when building samples, so the noise
    # block holds unit-variance normals rather than sigma-scaled ones.

    @property
    def _noise_sample_sigma(self) -> float:
        return 1.0

    def _draw_static_noise(self, noise_rng, N: int, H: int, nu: int) -> np.ndarray:
        return noise_rng.normal(loc=0.0, scale=1.0, size=(N, H, nu)).astype(np.float32)

    # -- planner hooks ------------------------------------------------------

    def _begin_plan(self):
        if self.pc.reopen_sigma_each_plan:
            self.sigma_wp.fill_(self.pc.noise_sigma)

    def _build_samples(self):
        wp.launch(
            _cem_build_samples_kernel,
            dim=(self.pc.n_samples, self.horizon, self.nu),
            inputs=[self.mu_wp, self.sigma_wp, self._static_eps_wp,
                    self._ctrl_range_wp, self._has_limits_wp],
            outputs=[self.V_wp],
        )

    def _update_params(self, n_eff: int) -> bool:
        """Refit (mu, sigma) to the k lowest-cost samples."""
        self._fold_costs()

        # The only device→host transfer of the update: N floats. numpy() syncs.
        costs = self.costs_wp.numpy()
        valid = ~np.isnan(costs)
        n_valid = int(valid.sum())
        if n_valid == 0:
            if self.pc.debug:
                print("  [CEM] all rollouts NaN — zero action")
            return False

        # NaN rollouts must never enter the elite set: one would poison the whole
        # (mu, sigma) refit. Push them to +inf and shrink k if there are not
        # enough valid samples left to fill it.
        finite_costs = np.where(valid, costs, np.inf)
        k = min(self.n_elites, n_valid)
        elites = np.argpartition(finite_costs, k - 1)[:k]

        self._elite_idx_np[:k] = elites.astype(np.int32)
        self._elite_idx_np[k:] = self._elite_idx_np[0]
        self.elite_idx_wp.assign(self._elite_idx_np)
        self.last_n_elites = k

        # Rows beyond n_eff never influenced a cost (time-constrained path), so
        # the refit leaves them untouched.
        wp.launch(
            _cem_refit_kernel,
            dim=(n_eff, self.nu),
            inputs=[self.elite_idx_wp, self.V_wp, k,
                    self.pc.alpha, self.pc.min_sigma],
            outputs=[self.mu_wp, self.sigma_wp],
        )

        if self.pc.debug:
            elite_costs = finite_costs[elites]
            print(f"  [CEM] min cost: {float(elite_costs.min()):.4f}  "
                  f"elite mean: {float(elite_costs.mean()):.4f}  "
                  f"avg cost: {float(costs[valid].mean()):.4f}  "
                  f"elites: {k}/{self.pc.n_samples}")
        return True

    def _reset_params(self):
        """Re-open the search: sigma back to the configured initial scale."""
        self.sigma_wp.fill_(self.pc.noise_sigma)

    # -- action extraction --------------------------------------------------

    def _extract_action(self) -> np.ndarray:
        """Read mu[0], then roll the whole distribution forward if warm-starting.

        The base shifts mu (and zeros its last row); sigma is shifted alongside
        it by the same shift_steps, with the newly-appended rows re-opened to the
        initial noise_sigma — a freshly-appended step has no elite evidence
        behind it."""
        action_np = super()._extract_action()
        if self.pc.warm_start:
            wp.launch(
                _shift_2d_kernel,
                dim=(self.horizon, self.nu),
                inputs=[self.sigma_wp, self.sigma_shift_wp, self.horizon,
                        self.shift_steps, self.pc.noise_sigma],
            )
            self.sigma_wp, self.sigma_shift_wp = self.sigma_shift_wp, self.sigma_wp
        return action_np
