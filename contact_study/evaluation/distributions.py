"""First-action induced distribution of a sampling planner, and the KL between two.

A sampling planner's plan() call leaves a particle representation of its action
distribution on the GPU: V_wp holds the clamped candidate sequences and the
planner-specific state says how they were weighted. Summarizing the FIRST action
of that cloud as a Gaussian (mu, Sigma) is what makes two planners comparable —
it is how experiments/hpc/run_kl_divergence_cell.py measures the information a
degraded contact model loses relative to a reference one, and it is what
evaluation/trajectory.py records per control step so the same comparison can be
made offline against a trajectory that has already been run.

`weighted_moments` and `gaussian_kl` moved here verbatim from the KL cell, so the
numbers in existing results/kl_divergence_eval_* directories stay comparable.
`planner_moments` generalizes the first to CEM and the predictive sampler, which
have no weight vector at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = [
    "PlannerMoments", "planner_moments",
    "weighted_moments", "weighted_moments_from_particles", "gaussian_kl",
]


# ---------------------------------------------------------------------------
# moments
# ---------------------------------------------------------------------------
def weighted_moments_from_particles(V0, w, shrinkage: float, sigma: float):
    """First-action weighted mean and covariance from an explicit particle set.

    V0 is (N, nu) — the first action of each candidate sequence — and w is the
    (N,) weight vector those particles were scored with. Split out of
    weighted_moments so CEM's elite set and the predictive sampler's single
    winner go through exactly the same arithmetic as MPPI's weighted cloud.

    Returns (mu, Sigma, ess). Sigma is already shrunk toward sigma^2 I.
    """
    V0 = np.asarray(V0, dtype=np.float64)
    w  = np.asarray(w,  dtype=np.float64)

    # Guard against a degenerate weight vector (all-NaN rollouts zero U and can
    # leave w unnormalized); fall back to uniform so the caller keeps running.
    s = w.sum()
    if not np.isfinite(s) or s <= 0.0:
        w = np.full(w.shape, 1.0 / w.size, dtype=np.float64)
    else:
        w = w / s

    mu = w @ V0                                   # (nu,)
    D  = V0 - mu                                  # (N, nu)
    Sigma = (w[:, None] * D).T @ D                # (nu, nu)

    # Symmetrize (guards against float asymmetry before Cholesky) and shrink
    # toward the proposal covariance.
    Sigma = 0.5 * (Sigma + Sigma.T)
    d = mu.size
    Sigma = (1.0 - shrinkage) * Sigma + shrinkage * (sigma ** 2) * np.eye(d)

    ess = 1.0 / float(np.sum(w ** 2))
    return mu, Sigma, ess


def weighted_moments(controller, shrinkage: float, sigma: float):
    """First-action weighted mean and covariance of an MPPI planner's induced q*.

    Reads the state the controller already leaves on the GPU after plan():
    w_wp holds the NORMALIZED weights of the final MPPI iteration and V_wp the
    clamped samples those weights scored, so (V, w) is a consistent particle
    representation whose weighted mean equals the planner's returned U[0].

    Returns (mu, Sigma, ess). Sigma is already shrunk toward sigma^2 I.
    """
    w  = controller.w_wp.numpy().astype(np.float64)          # (N,) sums to 1
    V0 = controller.V_wp.numpy()[:, 0, :].astype(np.float64)  # (N, nu)
    return weighted_moments_from_particles(V0, w, shrinkage, sigma)


def gaussian_kl(mu0, S0, mu1, S1) -> float:
    """KL( N(mu0,S0) || N(mu1,S1) ), computed via Cholesky for stability.

        KL = 0.5 [ tr(S1^-1 S0) + (mu1-mu0)^T S1^-1 (mu1-mu0)
                   - d + ln(det S1 / det S0) ]

    Returns +inf if S1 is not positive definite (should not happen with
    shrinkage on, but a NaN cost can still poison a covariance).
    """
    d = mu0.size
    try:
        L1 = np.linalg.cholesky(S1)
    except np.linalg.LinAlgError:
        return float("inf")

    # S1^-1 S0 and S1^-1 dmu without forming an explicit inverse.
    Z    = np.linalg.solve(L1, S0)                 # L1^-1 S0
    tr   = float(np.trace(np.linalg.solve(L1.T, Z)))
    dmu  = mu1 - mu0
    y    = np.linalg.solve(L1, dmu)
    maha = float(y @ y)

    sign0, logdet0 = np.linalg.slogdet(S0)
    if sign0 <= 0:
        return float("inf")
    logdet1 = 2.0 * float(np.sum(np.log(np.diag(L1))))

    return 0.5 * (tr + maha - d + logdet1 - logdet0)


# ---------------------------------------------------------------------------
# planner-agnostic entry point
# ---------------------------------------------------------------------------
@dataclass
class PlannerMoments:
    """First-action distribution of one plan() call, summarized as a Gaussian.

    kind         "mppi" | "cem" | "predictive_sampler" | "unknown"
    mu           (nu,) weighted mean of the first action, None if unavailable
    cov          (nu, nu) weighted covariance, symmetric, shrunk toward sigma^2 I
    ess          effective sample size of the weight vector
    n_particles  particles the moments were computed from: N for MPPI, the elite
                 count k for CEM, 1 for the predictive sampler
    degenerate   the distribution is a point mass (greedy planner), or the solve
                 failed and nothing could be read
    """
    kind:        str
    mu:          np.ndarray | None = None
    cov:         np.ndarray | None = None
    ess:         float | None      = None
    n_particles: int | None        = None
    degenerate:  bool              = False


def _kind(controller) -> str:
    """Planner family, by the state it leaves behind rather than by class."""
    if hasattr(controller, "w_wp"):          return "mppi"
    if hasattr(controller, "elite_idx_wp"):  return "cem"
    if hasattr(controller, "best_idx_wp"):   return "predictive_sampler"
    return "unknown"


def _first_actions(controller) -> np.ndarray:
    """(N, nu) first action of every candidate sequence, as float64.

    V_wp is (N, H, nu) and .numpy() copies the whole thing across the bus — at
    n_samples=4096, H=7, nu=16 that is 1.8 MB and a full device sync per call.
    Callers decimate with planner_dist_every; this is why they need to.
    """
    return controller.V_wp.numpy()[:, 0, :].astype(np.float64)


def planner_moments(controller, *, shrinkage: float = 0.0,
                    sigma: float | None = None) -> PlannerMoments:
    """First-action induced distribution for any of the three planners.

    Dispatches on the planner-specific state each one leaves behind, by attribute
    rather than by isinstance so a future subclass works unedited:

      MPPI   w_wp — the softmax weights of the final iteration. The canonical
             case; mu equals the returned U[0] by construction.
      CEM    elite_idx_wp — no weights exist, so the elite set IS the particle
             set: uniform 1/k over the k elites of the final iteration, which
             reproduces exactly what _cem_refit_kernel computes, with ess == k.
             Note CEM's mean_seq (= U_wp = mu_wp) is the SMOOTHED refit
             alpha*old + (1-alpha)*elite_mean, so mu != mean_seq[0] here — unlike
             MPPI, where they are equal.
      PS     best_idx_wp — greedy argmin, so the induced distribution is a delta
             at the winner: cov is the shrinkage floor and ess is 1. Reported as
             degenerate rather than faked as uniform-over-samples, which would
             ignore the costs entirely and describe the PROPOSAL instead of the
             planner's decision.

    NEVER raises: this runs inside a sweep's control loop, and an unrecognized
    planner or a failed solve must cost the run a null record, not a crash.

    Only the final iteration's particles are visible — V_wp is overwritten per
    iteration — so with n_iterations > 1 (CEM defaults to 3) these are the
    moments of the last refit, not of the whole plan() call.
    """
    kind = _kind(controller)

    # A solve whose every rollout NaN'd zeroes U_wp but leaves w_wp/elite_idx_wp
    # holding the previous call's values — reading them would log a stale
    # distribution as if it were this step's.
    if not getattr(controller, "last_plan_ok", True):
        return PlannerMoments(kind=kind, degenerate=True)

    try:
        sigma = float(controller.pc.noise_sigma if sigma is None else sigma)

        if hasattr(controller, "w_wp"):                       # MPPI
            V0 = _first_actions(controller)
            w  = controller.w_wp.numpy().astype(np.float64)
            mu, cov, ess = weighted_moments_from_particles(V0, w, shrinkage, sigma)
            return PlannerMoments("mppi", mu, cov, ess, n_particles=int(V0.shape[0]))

        if hasattr(controller, "elite_idx_wp"):               # CEM
            V0  = _first_actions(controller)
            k   = int(getattr(controller, "last_n_elites", controller.n_elites))
            k   = max(1, min(k, V0.shape[0]))
            idx = controller.elite_idx_wp.numpy()[:k].astype(int)
            w   = np.full(k, 1.0 / k, dtype=np.float64)
            mu, cov, ess = weighted_moments_from_particles(V0[idx], w, shrinkage, sigma)
            return PlannerMoments("cem", mu, cov, ess, n_particles=k)

        if hasattr(controller, "best_idx_wp"):                # predictive sampler
            V0 = _first_actions(controller)
            # Pre-set to the sentinel N by the atomic_min; clamp so a solve that
            # never wrote a real index cannot index out of bounds.
            b  = min(int(controller.best_idx_wp.numpy()[0]), V0.shape[0] - 1)
            mu = V0[max(b, 0)]
            cov = shrinkage * (sigma ** 2) * np.eye(mu.size)
            return PlannerMoments("predictive_sampler", mu, cov, 1.0,
                                  n_particles=1, degenerate=True)
    except Exception:
        return PlannerMoments(kind=kind, degenerate=True)

    return PlannerMoments("unknown", degenerate=True)
