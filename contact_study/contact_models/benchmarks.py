"""Benchmarking utilities.

Two measurements are needed per (task, Mk) pair:

  1. ck  = wall-clock seconds per forward rollout (used in Condition A)
  2. eps_k = approximation error vs ground-truth M* across a fixed set
             of test states (used to characterize the accuracy-speed tradeoff)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np
import warp as wp

from .config import ContactModelConfig
from . import api


@dataclass
class BenchmarkResult:
    label:       str
    mean_ms:     float           # mean wall-clock time per rollout (ms)
    std_ms:      float
    speedup:     float = 1.0     # relative to baseline (M2 by default)
    approx_err:  float = 0.0     # mean L2 error in qpos vs ground truth
    approx_err_std: float = 0.0


def measure_rollout_speed(
    mjm:        mujoco.MjModel,
    cfg:        ContactModelConfig,
    horizon:    int     = 50,
    n_worlds:   int     = 512,
    n_warmup:   int     = 5,
    n_trials:   int     = 20,
    ctrl_fn:    Callable | None = None,
) -> BenchmarkResult:
    """Measure mean wall-clock time per H-step rollout for one contact model.

    Args:
        mjm:      Host-side MuJoCo model.
        cfg:      Contact model config to benchmark.
        horizon:  Rollout length in steps.
        n_worlds: Batch size (parallel worlds on GPU).
        n_warmup: Number of warmup rollouts (excluded from timing).
        n_trials: Number of timed rollouts.
        ctrl_fn:  Optional callable(m, d) -> None that sets d.ctrl.
                  If None, zero control is used.

    Returns:
        BenchmarkResult with mean/std timing in milliseconds.
    """
    m = api.put_model(mjm, cfg)
    d = api.make_data(mjm, m, nworld=n_worlds)

    def _run_one():
        api.reset_data(mjm, m, d)
        for _ in range(horizon):
            if ctrl_fn is not None:
                ctrl_fn(m, d)
            api.step(m, d)
        wp.synchronize()

    # Warmup
    for _ in range(n_warmup):
        _run_one()

    # Timed trials
    times_ms = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _run_one()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)

    times_ms = np.array(times_ms)
    return BenchmarkResult(
        label    = cfg.label,
        mean_ms  = float(np.mean(times_ms)),
        std_ms   = float(np.std(times_ms)),
    )


def measure_approximation_error(
    mjm:           mujoco.MjModel,
    cfg_gt:        ContactModelConfig,
    cfg_approx:    ContactModelConfig,
    test_states:   np.ndarray,            # (N, nq+nv) array of initial states
    horizon:       int = 20,
    ctrl_sequences: np.ndarray | None = None,  # (N, H, nu) or None
) -> tuple[float, float]:
    """Measure mean L2 error in (qpos, qvel) after `horizon` steps.

    Runs the ground-truth model and the approximate model from the same
    initial states and returns the mean/std of ||qpos_gt - qpos_approx||_2.

    Args:
        mjm:            Host model.
        cfg_gt:         Ground-truth contact config (typically M2).
        cfg_approx:     Approximate contact config (Mk).
        test_states:    (N, nq+nv) array.
        horizon:        Rollout length.
        ctrl_sequences: Optional (N, H, nu) control sequences.
                        Zero control is used if None.

    Returns:
        (mean_err, std_err) over the N test states.
    """
    N = test_states.shape[0]
    nq = mjm.nq
    nv = mjm.nv

    m_gt  = api.put_model(mjm, cfg_gt)
    m_ap  = api.put_model(mjm, cfg_approx)
    mjd   = mujoco.MjData(mjm)

    errs = []
    for i in range(N):
        q0 = test_states[i, :nq]
        v0 = test_states[i, nq:nq+nv]

        def _rollout(m):
            mjd.qpos[:] = q0
            mjd.qvel[:] = v0
            d = api.put_data(mjm, mjd, m)
            for t in range(horizon):
                if ctrl_sequences is not None:
                    d.ctrl.assign(ctrl_sequences[i, t])
                api.step(m, d)
            wp.synchronize()
            api.get_data_into(mjm, m, d, mjd)
            return mjd.qpos.copy(), mjd.qvel.copy()

        qpos_gt,  qvel_gt  = _rollout(m_gt)
        qpos_ap, _qvel_ap  = _rollout(m_ap)

        err = np.linalg.norm(qpos_gt - qpos_ap)
        errs.append(err)

    errs = np.array(errs)
    return float(np.mean(errs)), float(np.std(errs))


def run_full_benchmark(
    mjm:         mujoco.MjModel,
    configs:     list[ContactModelConfig],
    baseline_cfg: ContactModelConfig | None = None,
    **kwargs,
) -> list[BenchmarkResult]:
    """Run speed benchmark for all configs and compute relative speedups.

    Args:
        mjm:          Host model.
        configs:      List of configs to benchmark.
        baseline_cfg: Config to use as speedup denominator. If None,
                      the first config in the list is used.
        **kwargs:     Passed to measure_rollout_speed.

    Returns:
        List of BenchmarkResult, one per config, with speedup filled in.
    """
    results = [measure_rollout_speed(mjm, cfg, **kwargs) for cfg in configs]

    if baseline_cfg is None:
        baseline_time = results[0].mean_ms
    else:
        baseline_time = measure_rollout_speed(mjm, baseline_cfg, **kwargs).mean_ms

    for r in results:
        r.speedup = baseline_time / r.mean_ms if r.mean_ms > 0 else float("inf")

    return results
