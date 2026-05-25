"""Model Predictive Path Integral (MPPI) controller.

All rollouts are executed in parallel on GPU via the batched step()
interface (nworld = N samples). The physics step and state resets are 
encapsulated in CUDA graphs, eliminating slow CPU-side data resets 
during the MPPI loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from scipy.interpolate import CubicSpline
import numpy as np
import warp as wp
import mujoco

from contact_study.contact_models import api
from contact_study.contact_models.config import ContactModelConfig


@dataclass
class MPPIConfig:
    n_samples:       int   = 1024   # N: number of candidate trajectories
    horizon:         int   = 50     # H: planning horizon (steps)
    temperature:     float = 1.0    # lambda: MPPI temperature (inverse temp)
    noise_sigma:     float = 0.01   # action noise std dev
    n_iterations:    int   = 1      # number of MPPI update iterations per call
    warm_start:      bool  = True   # shift action sequence one step forward
    nconmax:         int   = 200
    njmax:           int   = 500
    substeps:        int   = 1
    adaptive_temp:   bool  = True
    adp_temp_params: tuple[float, float, float, float] = (10.0, 5.0, 0.9, 1.1)
    use_spline_noise:bool  = True   # toggle between spline and Gaussian noise
    n_spline_points: int   = 3      # control points for spline-smoothed noise
    debug:           bool  = True
    delta_range:     tuple[float, float] = (-0.01, 0.01)


# ---------------------------------------------------------------------------
# Module-level Warp Kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _broadcast_1d_to_2d_kernel(
    src: wp.array(dtype=float),
    dst: wp.array2d(dtype=float)
):
    n, i = wp.tid()
    dst[n, i] = src[i]

@wp.kernel
def _add_noise_and_clip_kernel(
    U_mean:     wp.array2d(dtype=float),   # (H, nu)
    eps:        wp.array3d(dtype=float),   # (N, H, nu)
    ctrl_range: wp.array2d(dtype=float),   # (nu, 2)
    has_limits: wp.array(dtype=bool),      # (nu,)
    V_out:      wp.array3d(dtype=float),   # (N, H, nu)  [out]
):
    """Add noise to the mean action sequence and clip to actuator limits."""
    n, h, u = wp.tid()
    val = U_mean[h, u] + eps[n, h, u]
    if has_limits[u]:
        val = wp.clamp(val, ctrl_range[u, 0], ctrl_range[u, 1])
    V_out[n, h, u] = val

@wp.kernel
def _assign_ctrl_kernel(
    V:    wp.array3d(dtype=float),   # (N, H, nu)
    t:    int,                       # timestep index
    ctrl: wp.array2d(dtype=float),   # (N, nu)  [out]
):
    """Copy the t-th slice of V into d.ctrl without a Python-side round-trip."""
    n, u = wp.tid()
    ctrl[n, u] += V[n, t, u]

def _make_accumulate_kernel(cost_fn_wp: wp.func):
    @wp.kernel
    def _kernel(
        qpos:      wp.array2d(dtype=float),
        qvel:      wp.array2d(dtype=float),
        ctrl:      wp.array2d(dtype=float),
        site_xpos: wp.array2d(dtype=wp.vec3),
        terminal:  bool,
        goal:      wp.array(dtype=float),
        indices:   wp.array(dtype=int),
        weights:   wp.array(dtype=float),
        costs_out: wp.array(dtype=float),
    ):
        w = wp.tid()
        costs_out[w] += cost_fn_wp(
            qpos[w], qvel[w], ctrl[w], site_xpos[w], 
            terminal, goal, indices, weights
        )
    return _kernel

# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MPPIController:
    """MPPI controller backed by a contact model."""

    def __init__(
        self,
        mjm:      mujoco.MjModel,
        cfg:      ContactModelConfig,
        mppi_cfg: MPPIConfig,
        cost_fn:  wp.func,
        goals_wp: wp.array,
        idx_wp:   wp.array,
        weights_wp: wp.array,
        rng:      np.random.Generator | None = None,
    ):
        self.mjm = mjm
        self.cfg = cfg
        self.pc  = mppi_cfg
        self.rng = rng or np.random.default_rng()
        self.lam = self.pc.temperature

        self.nu = mjm.nu
        self.nq = mjm.nq
        self.nv = mjm.nv

        # Handle potential tuple from cost_fn_wp
        self.cost_fn_wp_func = cost_fn
        self.goal_wp = goals_wp
        self.indices_wp = idx_wp
        self.weights_wp = weights_wp
        self._accumulate_costs_kernel = _make_accumulate_kernel(self.cost_fn_wp_func)

        self._setup_warp_arrays()
        self._setup_warp_backend()

    def _setup_warp_arrays(self):
        """Allocate Warp arrays on GPU once (avoids runtime host copies)."""
        N, H, nu = self.pc.n_samples, self.pc.horizon, self.nu

        # Pre-allocate reset buffers
        self.qpos_reset = wp.empty(self.nq, dtype=wp.float32, device="cuda")
        self.qvel_reset = wp.empty(self.nv, dtype=wp.float32, device="cuda")
        self.ctrl_reset = wp.empty(self.nu, dtype=wp.float32, device="cuda")

        self.U_wp = wp.zeros((H, nu), dtype=wp.float32, device="cuda")
        self.V_wp = wp.zeros((N, H, nu), dtype=wp.float32, device="cuda")
        self.costs_wp = wp.zeros(N, dtype=wp.float32, device="cuda")

        # Actuator limits
        delta_low, delta_high = self.pc.delta_range
        delta_range_np = np.empty((self.nu, 2), dtype=np.float32)
        delta_range_np[:, 0] = delta_low
        delta_range_np[:, 1] = delta_high

        self._ctrl_range_wp = wp.array(delta_range_np, dtype=wp.float32, device="cuda")
        self._has_limits_wp = wp.array(np.ones(self.nu, dtype=bool), dtype=wp.bool, device="cuda")

        # Pre-sample static noise (reused every plan call)
        if self.pc.use_spline_noise:
            t_knots  = np.linspace(0, H - 1, self.pc.n_spline_points)
            t_dense  = np.arange(H)
            knot_noise = self.rng.normal(
                0, self.pc.noise_sigma, (N, self.pc.n_spline_points, nu)
            ).astype(np.float32)
            
            static_eps_np = np.empty((N, H, nu), dtype=np.float32)
            for n in range(N):
                for j in range(nu):
                    static_eps_np[n, :, j] = CubicSpline(t_knots, knot_noise[n, :, j])(t_dense)
        else:
            static_eps_np = self.rng.normal(
                loc=0.0, scale=self.pc.noise_sigma, size=(N, H, nu)
            ).astype(np.float32)

        self._static_eps_np = static_eps_np
        self._static_eps_wp = wp.array(static_eps_np, dtype=wp.float32, device="cuda")

    def _setup_warp_backend(self):
        """Initialize the batched MuJoCo backends and CUDA graphs."""
        self.m = api.put_model(self.mjm, self.cfg)
        self.d = api.make_data(
            self.mjm, self.m,
            nworld=self.pc.n_samples,
            nconmax=self.pc.nconmax,
            njmax=self.pc.njmax,
        )

        self.reset_graph = self.create_reset_graph()
        self.step_graph = self.create_step_graph()
        self.rollout_graph = self.create_rollout_graph()

    def create_reset_graph(self):
        """Create a graph to broadcast environment states and zero costs across N worlds."""
        with wp.ScopedCapture() as capture:
            # Broadcast state
            wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nq), inputs=[self.qpos_reset, self.d.qpos])
            wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nv), inputs=[self.qvel_reset, self.d.qvel])
            wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nu), inputs=[self.ctrl_reset, self.d.ctrl])
            
            # Zero out the costs
            self.costs_wp.zero_()
        return capture.graph

    def create_step_graph(self):
        """Create a graph to advance physics by 'substeps' per MPPI timestep."""
        with wp.ScopedCapture() as capture:
            for _ in range(self.pc.substeps):
                api.step(self.m, self.d)
        return capture.graph

    def create_rollout_graph(self):
        """Captures the reset AND the entire H-step unroll into a single CUDA graph."""
        with wp.ScopedCapture() as capture:
            # 1. Broadcast initial state across all N samples
            wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nq), inputs=[self.qpos_reset, self.d.qpos])
            wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nv), inputs=[self.qvel_reset, self.d.qvel])
            wp.launch(_broadcast_1d_to_2d_kernel, dim=(self.pc.n_samples, self.nu), inputs=[self.ctrl_reset, self.d.ctrl])
            
            # 2. Zero out costs
            self.costs_wp.zero_()

            # 3. Unroll the entire horizon at capture time
            for t in range(self.pc.horizon):
                terminal = (t == self.pc.horizon - 1)

                # Assign controls
                wp.launch(
                    _assign_ctrl_kernel,
                    dim=(self.pc.n_samples, self.nu),
                    inputs=[self.V_wp, t, self.d.ctrl],
                )

                # Advance physics
                for _ in range(self.pc.substeps):
                    api.step(self.m, self.d)

                # Accumulate costs dynamically
                wp.launch(
                    self._accumulate_costs_kernel,
                    dim=self.pc.n_samples,
                    inputs=[
                        self.d.qpos, self.d.qvel, self.d.ctrl, self.d.site_xpos,
                        terminal, self.goal_wp, self.indices_wp,
                        self.weights_wp
                    ],
                    outputs=[self.costs_wp],
                )

        return capture.graph

    def reset(self):
        """Clear the action sequence (call at the start of a new episode)."""
        self.U_wp.zero_()

    def plan_(self, mjd: mujoco.MjData) -> np.ndarray:
        """Run MPPI and return the first action of the optimal sequence."""
        N = self.pc.n_samples
        H = self.pc.horizon

        # 1. Zero-allocation host-to-device copy
        # .assign() pushes numpy arrays directly to the pre-allocated GPU buffer
        self.qpos_reset.assign(mjd.qpos)
        self.qvel_reset.assign(mjd.qvel)
        self.ctrl_reset.assign(mjd.ctrl)

        for iteration in range(self.pc.n_iterations):

            # 2. Add noise and clip control sequences
            wp.launch(
                _add_noise_and_clip_kernel,
                dim=(N, H, self.nu),
                inputs=[self.U_wp, self._static_eps_wp, self._ctrl_range_wp, self._has_limits_wp],
                outputs=[self.V_wp],
            )

            # 3. Fire the Mega-Graph: executes reset + (H * substeps) physics ticks instantly
            wp.capture_launch(self.rollout_graph)

            # 4. Synchronize once per iteration to fetch final accumulated costs
            wp.synchronize()
            costs_np = self.costs_wp.numpy()

            # 5. MPPI weight update (CPU)
            beta = costs_np.min()
            w    = np.exp(-(costs_np - beta) / self.lam)
            eta  = w.sum() + 1e-8
            w   /= eta

            if self.pc.adaptive_temp:
                if eta > self.pc.adp_temp_params[0]:
                    self.lam = self.pc.adp_temp_params[2] * self.lam
                elif eta < self.pc.adp_temp_params[1]:
                    self.lam = self.pc.adp_temp_params[3] * self.lam

            low, high = self.pc.delta_range
            dU = np.einsum("n,nht->ht", w, self._static_eps_np).clip(low, high)
            
            # Zero-allocation update
            new_U = self.U_wp.numpy() + dU
            self.U_wp.assign(new_U.astype(np.float32))

            if self.pc.debug:
                indices_cpu = self.indices_wp.numpy()
                obj_pos = mjd.qpos[indices_cpu[0] : indices_cpu[0] + 3]
                print(
                    f"avg cost: {costs_np.mean():.4f} "
                    f"min cost: {beta:.4f}  "
                    f"eta: {eta:.4f} "
                    f"lam: {self.lam:.6f}"
                )

        # ------------------------------------------------------------------
        # Extract and return the first action
        # ------------------------------------------------------------------
        action_np = self.U_wp[0].numpy().copy()

        if self.pc.warm_start:
            U_np       = self.U_wp.numpy()
            U_np[:-1]  = U_np[1:]
            U_np[-1]   = 0.0
            self.U_wp.assign(U_np)

        return action_np

    def plan(self, mjd: mujoco.MjData) -> np.ndarray:
        """
        Run MPPI and return the first action of the optimal sequence.
        This one uses the Reset and Step Graphs. 
        """
        N = self.pc.n_samples
        H = self.pc.horizon

        # 1. Transfer single state to device buffers once per planning step (avoids api.reset_data CPU overhead)
        wp.copy(self.qpos_reset, wp.array(mjd.qpos, dtype=wp.float32, device="cuda"))
        wp.copy(self.qvel_reset, wp.array(mjd.qvel, dtype=wp.float32, device="cuda"))
        wp.copy(self.ctrl_reset, wp.array(mjd.ctrl, dtype=wp.float32, device="cuda"))

        for iteration in range(self.pc.n_iterations):

            # 2. Add noise and clip control sequences
            wp.launch(
                _add_noise_and_clip_kernel,
                dim=(N, H, self.nu),
                inputs=[self.U_wp, self._static_eps_wp, self._ctrl_range_wp, self._has_limits_wp],
                outputs=[self.V_wp],
            )

            # 3. Reset states efficiently purely on GPU
            wp.capture_launch(self.reset_graph)

            # 4. Unroll simulation over horizon (structurally mirrored from your controller.py)
            for t in range(H):
                terminal = (t == H - 1)

                # Set controls for current timestep
                wp.launch(
                    _assign_ctrl_kernel,
                    dim=(N, self.nu),
                    inputs=[self.V_wp, t, self.d.ctrl],
                )

                # Step dynamics graph
                wp.capture_launch(self.step_graph)

                # Accumulate costs dynamically
                wp.launch(
                    self._accumulate_costs_kernel,
                    dim=N,
                    inputs=[
                        self.d.qpos, self.d.qvel, self.d.ctrl, self.d.site_xpos,
                        terminal, self.goal_wp, self.indices_wp,
                        self.weights_wp
                    ],
                    outputs=[self.costs_wp],
                )

            # 5. Only synchronize once per iteration to fetch costs
            wp.synchronize()
            costs_np = self.costs_wp.numpy()

            # 6. MPPI weight update (CPU)
            beta = costs_np.min()
            w    = np.exp(-(costs_np - beta) / self.lam)
            eta  = w.sum() + 1e-8
            w   /= eta

            if self.pc.adaptive_temp:
                if eta > self.pc.adp_temp_params[0]:
                    self.lam = self.pc.adp_temp_params[2]*self.lam
                elif eta < self.pc.adp_temp_params[1]:
                    self.lam = self.pc.adp_temp_params[3]*self.lam

            low, high = self.pc.delta_range
            dU = np.einsum("n,nht->ht", w, self._static_eps_np).clip(low, high)   # (H, nu)
            new_U = self.U_wp.numpy() + dU
            self.U_wp.assign(new_U.astype(np.float32))

            if self.pc.debug:
                indices_cpu = self.indices_wp.numpy()
                obj_pos = mjd.qpos[indices_cpu[0] : indices_cpu[0] + 3]
                obj_quat = mjd.qpos[indices_cpu[0]+ 3: indices_cpu[0] + 3 + 4]
                print(
                    f"avg cost: {costs_np.mean():.4f} +/- {costs_np.std():.4f} "
                    f"min cost: {beta:.4f}  "
                    f"eta: {eta:.4f} "
                    f"lam: {self.lam:.6f} "
                    f"obj_pos: {obj_pos} "
                    f"obj_quat: {obj_quat} "
                )

        # ------------------------------------------------------------------
        # Extract and return the first action
        # ------------------------------------------------------------------
        action_np = self.U_wp[0].numpy().copy()

        if self.pc.warm_start:
            U_np       = self.U_wp.numpy()
            U_np[:-1]  = U_np[1:]
            U_np[-1]   = 0.0
            self.U_wp.assign(U_np)

        return action_np