"""Model Predictive Path Integral (MPPI) controller.

All rollouts are executed in parallel on GPU via the batched step()
interface (nworld = N samples). The MPPI weight update and action
resampling runs on CPU/numpy after a wp.synchronize().

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from scipy.interpolate import CubicSpline
import numpy as np
import warp as wp

from contact_study.contact_models import api
from contact_study.contact_models.config import ContactModelConfig

import time
import mujoco


@dataclass
class MPPIConfig:
    n_samples:    int   = 1024        # N: number of candidate trajectories
    horizon:      int   = 50          # H: planning horizon (steps)
    temperature:  float = 1.0         # lambda: MPPI temperature <- Should be Inverse Temp Technically
    noise_sigma:  float = 0.01         # action noise std dev
    n_iterations: int   = 1           # number of MPPI update iterations per call
    warm_start:   bool  = True        # shift action sequence one step forward
    nconmax:      int   = 200
    n_spline_points: int = 3          # number of control points for spline noise
    njmax:        int   = 500
    debug:        bool  = True


# ---------------------------------------------------------------------------
# Warp Kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _add_noise_and_clip_kernel(
    U_mean: wp.array2d(dtype=float),        # (H, nu)
    eps: wp.array3d(dtype=float),           # (N, H, nu)
    ctrl_range: wp.array2d(dtype=float),    # (nu, 2)
    has_limits: wp.array(dtype=bool),       # (nu,)
    # out
    V_out: wp.array3d(dtype=float),         # (N, H, nu)
):
    """Adds noise to the mean action sequence and clips to actuator limits."""
    n, h, u = wp.tid()

    val = U_mean[h, u] + eps[n, h, u]

    if has_limits[u]:
        val = wp.clamp(val, ctrl_range[u, 0], ctrl_range[u, 1])

    V_out[n, h, u] = val


# Remove the broken module-level _accumulate_costs_kernel entirely.

# Add this factory function at module level instead:
def _make_accumulate_kernel(cost_fn_wp: wp.func):
    """Factory that bakes a specific wp.func into a new kernel at definition time."""
    @wp.kernel
    def _kernel(
        qpos:      wp.array2d(dtype=float),
        qvel:      wp.array2d(dtype=float),
        ctrl:      wp.array2d(dtype=float),
        terminal:  bool,
        costs_out: wp.array(dtype=float),
    ):
        w = wp.tid()
        costs_out[w] += cost_fn_wp(qpos[w], qvel[w], ctrl[w], terminal)
    return _kernel

class MPPIController:
    """MPPI controller backed by a contact model.

    Args:
        mjm:       Host MuJoCo model. Already-perturbed/already-degraded;
                   this class does not apply physics noise or swap geometry.
        cfg:       Contact model config (selects backend).
        mppi_cfg:  MPPI hyperparameters. Reference: Williams et al. 2017 "Information Theoretic MPC for Model-Based Reinforcement Learning".
        cost_fn:   Callable(qpos, qvel, ctrl, terminal) -> float array (nworld,)
        rng:       NumPy RNG for action sampling reproducibility.
    """

    def __init__(
        self,
        mjm:       mujoco.MjModel,
        cfg:       ContactModelConfig,
        mppi_cfg:  MPPIConfig,
        cost_fn:   wp.func, # Expecting a warp.func for GPU cost calculation
        rng:       np.random.Generator | None = None,
    ):
        self.mjm      = mjm
        self.cfg      = cfg
        self.pc       = mppi_cfg
        self.cost_fn_wp = cost_fn # Store the warp.func directly
        self.rng      = rng or np.random.default_rng()

        self.nu = mjm.nu
        self.nq = mjm.nq
        self.nv = mjm.nv

        target_cost_fn = self.cost_fn_wp

        self._accumulate_costs_kernel = _make_accumulate_kernel(cost_fn)

        # Action sequence mean: (H, nu) on GPU
        self.U_wp = wp.zeros((mppi_cfg.horizon, mjm.nu), dtype=wp.float32, device="cuda")

        # Device-side model for batch rollouts
        self.m = api.put_model(mjm, cfg)
        self.d = api.make_data(mjm, self.m, nworld=mppi_cfg.n_samples, nconmax=mppi_cfg.nconmax, njmax=mppi_cfg.njmax)

        # Control limits from model
        self._ctrl_range_wp = wp.array(mjm.actuator_ctrlrange, dtype=wp.float32, device="cuda")
        self._has_limits_wp = wp.array(mjm.actuator_ctrllimited.astype(bool), dtype=wp.bool, device="cuda")

        # Buffer for candidate action sequences (N, H, nu) on GPU
        self.V_wp = wp.zeros((mppi_cfg.n_samples, mppi_cfg.horizon, mjm.nu), dtype=wp.float32, device="cuda")

        # Buffer for accumulating costs (N,) on GPU
        self.costs_wp = wp.zeros(mppi_cfg.n_samples, dtype=wp.float32, device="cuda")

    # ------------------------------------------------------------------

    def reset(self):
        """Clear the action sequence (e.g., at start of new episode)."""
        self.U_wp.zero_()

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

        # Define time points for spline control points and for the full horizon
        t_control_points = np.linspace(0, H - 1, self.pc.n_spline_points)
        t_full_horizon = np.arange(H)

        iter_start = time.perf_counter()

        for i in range(self.pc.n_iterations):
            # Generate Gaussian noise for the spline control points on CPU
            noise_control_points = self.rng.normal(0, sigma, (N, self.pc.n_spline_points, self.nu)).astype(np.float32)
            eps = np.zeros((N, H, self.nu), dtype=np.float32)
            # For each sample and each actuator, fit a spline and evaluate it over the horizon
            for i in range(N):
                for j in range(self.nu):
                    spl = CubicSpline(t_control_points, noise_control_points[i, :, j])
                    eps[i, :, j] = spl(t_full_horizon) # eps is (N, H, nu)
            
            # Transfer noise to GPU
            eps_wp = wp.array(eps, dtype=wp.float32, device="cuda")

            # Add noise to mean action sequence and clip on GPU
            wp.launch(
                _add_noise_and_clip_kernel,
                dim=(N, H, self.nu),
                inputs=[self.U_wp, eps_wp, self._ctrl_range_wp, self._has_limits_wp],
                outputs=[self.V_wp],
            )

            self._set_batch_state(mjd)
            self.costs_wp.zero_() # Reset costs on GPU

            for t in range(H):
                # Assign the t-th action for all N samples from V_wp (GPU) to d.ctrl (GPU)
                self.d.ctrl.assign(self.V_wp[:, t, :])
                api.step(self.m, self.d)
                terminal = (t == H - 1)
                
                # Compute and accumulate costs on GPU
                wp.launch(
                    self._accumulate_costs_kernel,
                    dim=N,
                    inputs=[self.d.qpos, self.d.qvel, self.d.ctrl, terminal],
                    outputs=[self.costs_wp],
                )
            
            wp.synchronize()
            costs_np = self.costs_wp.numpy() # Bring total costs to CPU for MPPI update

            beta = costs_np.min()
            w    = np.exp(-(costs_np - beta) / lam)
            eta = w.sum() + 1e-8
            w   /= eta#w.sum() + 1e-8

            dU_np = np.einsum("n,nht->ht", w, eps) # dU is (H, nu)
            self.U_wp.assign(self.U_wp.numpy() + dU_np) # Update mean action sequence on GPU
            if self.pc.debug:
                print("Avg. Cost:", costs_np.mean(), "Min. Cost:", beta, "Eta:", eta)

        action_np = self.U_wp[0].numpy().copy() # Get the first action from GPU

        if self.pc.warm_start:
            # Shift U_wp on GPU
            U_shifted_np = self.U_wp.numpy()
            U_shifted_np[:-1] = U_shifted_np[1:]
            U_shifted_np[-1] = 0.0
            self.U_wp.assign(U_shifted_np)

        return action_np