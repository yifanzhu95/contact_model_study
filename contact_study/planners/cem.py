"""Cross-Entropy Method (CEM) sampling-based MPC.

CEM maintains a Gaussian distribution over action sequences and
iteratively refits it to the elite sample set.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from contact_study.contact_models import api
from contact_study.contact_models.config import ContactModelConfig


@dataclass
class CEMConfig:
    n_samples:    int   = 1024
    horizon:      int   = 30
    n_elites:     int   = 64       # keep top-k samples for refit
    n_iterations: int   = 3        # CEM iterations per call
    noise_sigma:  float = 0.3      # initial action noise
    alpha:        float = 0.1      # mean smoothing factor (warm-start)


class CEMController:
    """CEM controller backed by a contact model.

    Interface identical to MPPIController for easy swapping in experiments.
    """

    def __init__(
        self,
        mjm:      mujoco.MjModel,
        cfg:      ContactModelConfig,
        cem_cfg:  CEMConfig,
        cost_fn,
        rng:      np.random.Generator | None = None,
    ):
        self.mjm     = mjm
        self.cfg     = cfg
        self.cc      = cem_cfg
        self.cost_fn = cost_fn
        self.rng     = rng or np.random.default_rng()

        self.nu = mjm.nu
        self.nq = mjm.nq

        # Distribution parameters
        self.mu    = np.zeros((cem_cfg.horizon, mjm.nu), dtype=np.float32)
        self.sigma = np.ones((cem_cfg.horizon, mjm.nu), dtype=np.float32) * cem_cfg.noise_sigma

        self._ctrl_range = mjm.actuator_ctrlrange.copy()
        self._has_limits  = mjm.actuator_ctrllimited.astype(bool)

        self.m = api.put_model(mjm, cfg, rng=self.rng)
        self.d = api.make_data(mjm, self.m, nworld=cem_cfg.n_samples)

    def reset(self):
        self.mu[:] = 0.0
        self.sigma[:] = self.cc.noise_sigma

    def _clip(self, u):
        for i in range(self.nu):
            if self._has_limits[i]:
                u[..., i] = np.clip(u[..., i], self._ctrl_range[i, 0], self._ctrl_range[i, 1])
        return u

    def _set_batch_state(self, mjd: mujoco.MjData):
        api.reset_data(self.mjm, self.m, self.d)
        N = self.cc.n_samples
        self.d.qpos.assign(np.tile(mjd.qpos, (N, 1)).astype(np.float32))
        self.d.qvel.assign(np.tile(mjd.qvel, (N, 1)).astype(np.float32))

    def _rollout_costs(self, mjd: mujoco.MjData, sequences: np.ndarray) -> np.ndarray:
        """Roll out N sequences in parallel and return (N,) cost array."""
        N, H, _ = sequences.shape
        self._set_batch_state(mjd)
        costs = np.zeros(N, dtype=np.float32)
        for t in range(H):
            self.d.ctrl.assign(sequences[:, t, :])
            api.step(self.m, self.d)
            terminal = (t == H - 1)
            costs += np.asarray(
                self.cost_fn(self.d.qpos, self.d.qvel, self.d.ctrl, terminal),
                dtype=np.float32,
            )
        wp.synchronize()
        return costs

    def plan(self, mjd: mujoco.MjData) -> np.ndarray:
        """Run CEM and return first action of the optimal sequence."""
        N  = self.cc.n_samples
        H  = self.cc.horizon
        k  = self.cc.n_elites

        for _ in range(self.cc.n_iterations):
            # Sample N sequences from current distribution
            eps = self.rng.standard_normal((N, H, self.nu)).astype(np.float32)
            seqs = self.mu[None] + self.sigma[None] * eps
            seqs = self._clip(seqs)

            costs = self._rollout_costs(mjd, seqs)

            # Refit to elite set
            elite_idx = np.argsort(costs)[:k]
            elites    = seqs[elite_idx]          # (k, H, nu)

            new_mu    = elites.mean(axis=0)
            new_sigma = elites.std(axis=0) + 1e-5

            # Smoothed update
            a = self.cc.alpha
            self.mu    = a * self.mu    + (1 - a) * new_mu
            self.sigma = a * self.sigma + (1 - a) * new_sigma

        action = self.mu[0].copy()

        # Shift
        self.mu[:-1] = self.mu[1:]
        self.mu[-1]  = 0.0

        return action
