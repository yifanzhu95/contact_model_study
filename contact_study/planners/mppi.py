"""Model Predictive Path Integral (MPPI) controller.

All rollouts are executed in parallel on GPU via the batched step()
interface (nworld = N samples). The MPPI weight update and action
resampling runs on CPU/numpy after a wp.synchronize().

Reference: Williams et al. 2017 "Information Theoretic MPC for
           Model-Based Reinforcement Learning".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np
import warp as wp

from contact_study.contact_models import api
from contact_study.contact_models.config import ContactModelConfig


@dataclass
class MPPIConfig:
    n_samples:    int   = 1024        # N: number of candidate trajectories
    horizon:      int   = 30          # H: planning horizon (steps)
    temperature:  float = 0.1         # lambda: MPPI temperature
    noise_sigma:  float = 0.1         # action noise std dev
    n_iterations: int   = 1           # number of MPPI update iterations per call
    warm_start:   bool  = True        # shift action sequence one step forward


class MPPIController:
    """MPPI controller backed by a contact model.

    Args:
        mjm:       Host MuJoCo model. Already-perturbed/already-degraded;
                   this class does not apply physics noise or swap geometry.
        cfg:       Contact model config (selects backend).
        mppi_cfg:  MPPI hyperparameters.
        cost_fn:   Callable(qpos, qvel, ctrl, terminal) -> float array (nworld,)
        rng:       NumPy RNG for action sampling reproducibility.
    """

    def __init__(
        self,
        mjm:       mujoco.MjModel,
        cfg:       ContactModelConfig,
        mppi_cfg:  MPPIConfig,
        cost_fn:   Callable,
        rng:       np.random.Generator | None = None,
    ):
        self.mjm      = mjm
        self.cfg      = cfg
        self.pc       = mppi_cfg
        self.cost_fn  = cost_fn
        self.rng      = rng or np.random.default_rng()

        self.nu = mjm.nu
        self.nq = mjm.nq
        self.nv = mjm.nv

        # Action sequence mean: (H, nu)
        self.U = np.zeros((mppi_cfg.horizon, mjm.nu), dtype=np.float32)

        # Device-side model for batch rollouts
        self.m = api.put_model(mjm, cfg)
        self.d = api.make_data(mjm, self.m, nworld=mppi_cfg.n_samples)

        # Control limits from model
        self._ctrl_range = mjm.actuator_ctrlrange.copy()   # (nu, 2)
        self._has_limits  = mjm.actuator_ctrllimited.astype(bool)

    # ------------------------------------------------------------------

    def reset(self):
        """Clear the action sequence (e.g., at start of new episode)."""
        self.U[:] = 0.0

    def _clip_ctrl(self, u: np.ndarray) -> np.ndarray:
        """Clip controls to actuator limits where defined."""
        for i in range(self.nu):
            if self._has_limits[i]:
                u[..., i] = np.clip(u[..., i], self._ctrl_range[i, 0], self._ctrl_range[i, 1])
        return u

    def _set_batch_state(self, mjd: mujoco.MjData):
        """Upload current env state to all N parallel worlds."""
        api.reset_data(self.mjm, self.m, self.d)
        self.d.qpos.assign(
            np.tile(mjd.qpos, (self.pc.n_samples, 1)).astype(np.float32)
        )
        self.d.qvel.assign(
            np.tile(mjd.qvel, (self.pc.n_samples, 1)).astype(np.float32)
        )

    def plan(self, mjd: mujoco.MjData) -> np.ndarray:
        """Run MPPI and return the first action of the optimal sequence."""
        N = self.pc.n_samples
        H = self.pc.horizon
        lam = self.pc.temperature
        sigma = self.pc.noise_sigma

        for _ in range(self.pc.n_iterations):
            eps = self.rng.normal(0, sigma, (N, H, self.nu)).astype(np.float32)
            V   = self.U[None] + eps
            V   = self._clip_ctrl(V)

            self._set_batch_state(mjd)
            costs = np.zeros(N, dtype=np.float32)

            for t in range(H):
                self.d.ctrl.assign(V[:, t, :])
                api.step(self.m, self.d)
                terminal = (t == H - 1)
                step_costs = self.cost_fn(
                    self.d.qpos, self.d.qvel, self.d.ctrl, terminal
                )
                costs += np.asarray(step_costs, dtype=np.float32)

            wp.synchronize()

            beta = costs.min()
            w    = np.exp(-(costs - beta) / lam)
            w   /= w.sum() + 1e-8

            dU = np.einsum("n,nht->ht", w, eps)
            self.U += dU
            self.U  = self._clip_ctrl(self.U)

        action = self.U[0].copy()

        if self.pc.warm_start:
            self.U[:-1] = self.U[1:]
            self.U[-1]  = 0.0

        return action