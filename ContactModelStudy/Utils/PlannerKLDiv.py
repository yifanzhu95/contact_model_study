"""KL divergence between two planners' actions, or between a planner and a Gaussian action.

Each side is summarized as a Gaussian over the action executed now:

* a **planner** (any ``SamplingBasedPlannerBase`` that implements
  ``FirstActionDistribution``, such as MPPI) plans from the given state, and
  its weighted sample cloud's first action is moment-matched: the mean is the
  action ``Plan`` returns, the covariance the weighted spread of the samples.
  Comparing these *induced* distributions, rather than the proposals the
  planners sampled from, includes what the costs did: two planners with the
  same ``noise_sigma`` would otherwise differ only in their means.
* a **Gaussian action** ``(u, u_sigma)`` is ``N(u, diag(u_sigma^2))``, which is
  what ``Plan`` returns with ``return_uncertainty=True``. ``u_sigma`` may also be
  a scalar (the same std on every actuator) or a full ``(nu, nu)`` covariance.

The KL between the two Gaussians is then exact:

    KL(P || Q) = 1/2 [ tr(S_Q^-1 S_P) + (mu_Q - mu_P)^T S_Q^-1 (mu_Q - mu_P)
                       - nu + ln det S_Q - ln det S_P ]

It loses what a Gaussian cannot carry (several separated modes, say).

A weighted covariance from a few hundred samples in 16 dimensions can be close
to singular, and a sharply peaked weighting (low temperature) makes it more so.
A planner's covariance is therefore shrunk toward its proposal,
``(1 - shrinkage) * S + shrinkage * noise_sigma^2 * I``, as the old study's KL
experiments did (``shrinkage`` 1e-3 by default), and every covariance gets a
small variance floor. Report the shrinkage with any KL numbers: at a strongly
peaked weighting, it decides the result.

Ported from ``contact_study/evaluation/distributions.py``
(``weighted_moments_from_particles`` and ``gaussian_kl``).
"""

from __future__ import annotations

import numpy as np

#: Added to every covariance's diagonal so a zero std does not make KL infinite.
MIN_VARIANCE = 1e-12


def GaussianKL(mu_p, cov_p, mu_q, cov_q) -> float:
    """``KL(N(mu_p, cov_p) || N(mu_q, cov_q))``, through Cholesky factors.

    Returns ``inf`` when either covariance is not positive definite.
    """
    mu_p, mu_q = np.asarray(mu_p, float).ravel(), np.asarray(mu_q, float).ravel()
    cov_p, cov_q = np.asarray(cov_p, float), np.asarray(cov_q, float)
    d = mu_p.size
    try:
        Lq = np.linalg.cholesky(cov_q)
        Lp = np.linalg.cholesky(cov_p)
    except np.linalg.LinAlgError:
        return float("inf")
    # tr(S_Q^-1 S_P) = ||Lq^-1 Lp||_F^2, and the Mahalanobis term, with no explicit inverse.
    A = np.linalg.solve(Lq, Lp)
    y = np.linalg.solve(Lq, mu_q - mu_p)
    logdet_q = 2.0 * np.sum(np.log(np.diag(Lq)))
    logdet_p = 2.0 * np.sum(np.log(np.diag(Lp)))
    # KL >= 0; rounding can leave a hair below zero for identical inputs.
    return max(0.0, float(0.5 * (np.sum(A * A) + y @ y - d + logdet_q - logdet_p)))


def _gaussianFromAction(action, nu: int | None) -> tuple[np.ndarray, np.ndarray]:
    """``(u, u_sigma)`` -> ``(mu, cov)``; ``u_sigma`` a scalar, ``(nu,)`` stds or ``(nu, nu)`` covariance."""
    if not (isinstance(action, (tuple, list)) and len(action) == 2):
        raise TypeError("an action must be a (u, u_sigma) pair or a planner, got "
                        f"{type(action).__name__}")
    mu = np.asarray(action[0], float).ravel()
    s = np.asarray(action[1], float)
    if s.ndim == 0:
        cov = np.eye(mu.size) * float(s) ** 2
    elif s.shape == (mu.size,):
        cov = np.diag(s ** 2)
    elif s.shape == (mu.size, mu.size):
        cov = 0.5 * (s + s.T)
    else:
        raise ValueError(f"u_sigma must be a scalar, ({mu.size},) or ({mu.size}, {mu.size}); got {s.shape}")
    if nu is not None and mu.size != nu:
        raise ValueError(f"u has {mu.size} entries, the other side has {nu}")
    return mu, cov


def _gaussianFromPlanner(planner, q, q_dot, u, shrinkage: float, restore: bool):
    """Plan from ``(q, q_dot)`` and moment-match the first action.

    With ``restore``, the planner's per-episode state (its mean sequence and noise
    stream) is put back afterwards, so measuring a planner that is driving an
    episode does not change what it does next.
    """
    saved = planner.SaveState() if restore else None
    try:
        planner.Plan(q, q_dot, u=u)
        mu, cov = planner.FirstActionDistribution()
    finally:
        if saved is not None:
            planner.LoadState(saved)
    sigma2 = float(planner.config.noise_sigma) ** 2
    cov = (1.0 - shrinkage) * cov + shrinkage * sigma2 * np.eye(mu.size)
    return mu, cov


def _isPlanner(x) -> bool:
    return hasattr(x, "Plan") and hasattr(x, "FirstActionDistribution")


def PlannerGaussian(x, q=None, q_dot=None, u=None, shrinkage: float = 1e-3,
                    restore: bool = True, nu: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(mu, cov)`` for a planner (planned from ``(q, q_dot)``) or a ``(u, u_sigma)`` action."""
    if _isPlanner(x):
        if q is None or q_dot is None:
            raise ValueError("a planner needs the state (q, q_dot) to plan from")
        mu, cov = _gaussianFromPlanner(x, q, q_dot, u, shrinkage, restore)
    else:
        mu, cov = _gaussianFromAction(x, nu)
    return mu, cov + MIN_VARIANCE * np.eye(mu.size)


def CalcPlannerKLDiv(q, q_dot, Planner1, Planner2, u=None, shrinkage: float = 1e-3,
                     restore: bool = True) -> float:
    """``KL(P1 || P2)`` between two planners' actions at a state, or a planner and an action.

    Args:
        q, q_dot: The state to plan from, in MuJoCo layout. Unused when both
            sides are ``(u, u_sigma)`` actions.
        Planner1, Planner2: Each a sampling-based planner (it plans from the
            state) or a Gaussian action ``(u, u_sigma)``.
        u: The control currently applied, passed to each planner's ``Plan``
            (``ctrl_relative`` mode starts from it).
        shrinkage: Fraction of each planner's covariance replaced by its
            proposal, ``noise_sigma^2 * I``, in [0, 1]. Not applied to actions.
        restore: Put each planner's state back after planning, so the
            measurement leaves it as it was.

    Returns:
        ``KL(P1 || P2)`` in nats, ``>= 0``; ``inf`` if a covariance is
        degenerate, ``nan`` if a planner's plan failed (every rollout NaN).
        KL is not symmetric: swap the arguments for the reverse direction.

    Raises:
        ValueError: On mismatched action sizes or a bad ``shrinkage``.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
    try:
        mu1, cov1 = PlannerGaussian(Planner1, q, q_dot, u, shrinkage, restore)
        mu2, cov2 = PlannerGaussian(Planner2, q, q_dot, u, shrinkage, restore, nu=mu1.size)
    except RuntimeError:            # a plan with no finite rollout has no distribution
        return float("nan")
    if mu1.size != mu2.size:
        raise ValueError(f"action sizes differ: {mu1.size} vs {mu2.size}")
    return GaussianKL(mu1, cov1, mu2, cov2)
