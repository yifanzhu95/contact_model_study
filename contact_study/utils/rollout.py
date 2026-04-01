"""Batched rollout utilities.

The core function `batch_rollout` executes N trajectories in parallel
using the nworld batching in MJWarp/comfree_warp, returning costs and
final states without any Python-level loop over worlds.

For the study's two experimental conditions:

  Condition A (fixed_budget_rollout):  given a wall-clock budget T,
    determine how many samples fit and run exactly one batch.

  Condition B (fixed_sample_rollout):  run exactly N samples,
    independent of wall-clock time.
"""

from __future__ import annotations

import time
from typing import Callable

import mujoco
import numpy as np
import warp as wp

from contact_study.contact_models import api
from contact_study.contact_models.config import ContactModelConfig


def batch_rollout(
    mjm:       mujoco.MjModel,
    m,                             # device model (any backend)
    d,                             # device data  (any backend)
    action_sequences: np.ndarray,  # (N, H, nu)
    cost_fn:   Callable,           # (qpos, qvel, ctrl, terminal) -> (N,) array
    initial_qpos: np.ndarray,      # (nq,) or (N, nq)
    initial_qvel: np.ndarray,      # (nv,) or (N, nv)
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out N action sequences in parallel.

    Returns:
        costs:      (N,) total cost per sequence.
        final_qpos: (N, nq) final configuration.
    """
    N, H, nu = action_sequences.shape
    nq = mjm.nq

    # Upload initial states (broadcast scalar if single state given)
    api.reset_data(mjm, m, d)
    if initial_qpos.ndim == 1:
        d.qpos.assign(np.tile(initial_qpos, (N, 1)).astype(np.float32))
        d.qvel.assign(np.tile(initial_qvel, (N, 1)).astype(np.float32))
    else:
        d.qpos.assign(initial_qpos.astype(np.float32))
        d.qvel.assign(initial_qvel.astype(np.float32))

    costs = np.zeros(N, dtype=np.float32)

    for t in range(H):
        d.ctrl.assign(action_sequences[:, t, :].astype(np.float32))
        api.step(m, d)
        terminal = (t == H - 1)
        step_cost = np.asarray(
            cost_fn(d.qpos, d.qvel, d.ctrl, terminal),
            dtype=np.float32,
        )
        costs += step_cost

    wp.synchronize()
    final_qpos = d.qpos.numpy().copy()
    return costs, final_qpos


def fixed_budget_rollout(
    mjm:            mujoco.MjModel,
    cfg:            ContactModelConfig,
    budget_seconds: float,
    horizon:        int,
    cost_fn:        Callable,
    initial_qpos:   np.ndarray,
    initial_qvel:   np.ndarray,
    noise_sigma:    float = 0.1,
    rng:            np.random.Generator | None = None,
    min_samples:    int = 16,
    max_samples:    int = 16384,
) -> dict:
    """Condition A: run as many samples as possible within `budget_seconds`.

    First estimates samples/second using a small probe rollout, then
    launches the maximum batch that fits in the budget.

    Returns:
        dict with keys: n_samples, costs (N,), final_qpos (N, nq),
        elapsed_seconds, samples_per_second.
    """
    rng = rng or np.random.default_rng()
    nu  = mjm.nu

    # --- probe: estimate throughput at a moderate batch size ---
    probe_n = min(256, max_samples)
    m_probe = api.put_model(mjm, cfg)
    d_probe = api.make_data(mjm, m_probe, nworld=probe_n)
    probe_seqs = rng.normal(0, noise_sigma, (probe_n, horizon, nu)).astype(np.float32)

    t0 = time.perf_counter()
    batch_rollout(mjm, m_probe, d_probe, probe_seqs, cost_fn, initial_qpos, initial_qvel)
    probe_elapsed = time.perf_counter() - t0

    samples_per_sec = probe_n / probe_elapsed
    n_samples = int(min(max_samples, max(min_samples, int(budget_seconds * samples_per_sec))))
    # Round down to power of 2 for GPU efficiency
    n_samples = max(min_samples, 2 ** int(np.log2(n_samples)))

    # --- main rollout ---
    m = api.put_model(mjm, cfg)
    d = api.make_data(mjm, m, nworld=n_samples)
    seqs = rng.normal(0, noise_sigma, (n_samples, horizon, nu)).astype(np.float32)

    t0 = time.perf_counter()
    costs, final_qpos = batch_rollout(mjm, m, d, seqs, cost_fn, initial_qpos, initial_qvel)
    elapsed = time.perf_counter() - t0

    return {
        "n_samples":         n_samples,
        "costs":             costs,
        "final_qpos":        final_qpos,
        "elapsed_seconds":   elapsed,
        "samples_per_second": n_samples / elapsed,
    }


def fixed_sample_rollout(
    mjm:          mujoco.MjModel,
    cfg:          ContactModelConfig,
    n_samples:    int,
    horizon:      int,
    cost_fn:      Callable,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
    noise_sigma:  float = 0.1,
    rng:          np.random.Generator | None = None,
) -> dict:
    """Condition B: run exactly n_samples rollouts.

    Returns:
        dict with keys: n_samples, costs (N,), final_qpos (N, nq),
        elapsed_seconds.
    """
    rng = rng or np.random.default_rng()
    nu  = mjm.nu

    m    = api.put_model(mjm, cfg)
    d    = api.make_data(mjm, m, nworld=n_samples)
    seqs = rng.normal(0, noise_sigma, (n_samples, horizon, nu)).astype(np.float32)

    t0 = time.perf_counter()
    costs, final_qpos = batch_rollout(mjm, m, d, seqs, cost_fn, initial_qpos, initial_qvel)
    elapsed = time.perf_counter() - t0

    return {
        "n_samples":       n_samples,
        "costs":           costs,
        "final_qpos":      final_qpos,
        "elapsed_seconds": elapsed,
    }
