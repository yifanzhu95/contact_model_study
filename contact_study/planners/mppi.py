"""Model Predictive Path Integral (MPPI) controller.

All rollouts are executed in parallel on GPU via the batched step()
interface (nworld = N samples). The MPPI weight update and action
resampling runs on CPU/numpy after a wp.synchronize().

Model Predictive Path Integral (MPPI) controller.

All rollouts are executed in parallel on GPU via the batched step()
interface (nworld = N samples). The inner H-step rollout loop is recorded
as a CUDA graph during __init__ and replayed with a single
wp.capture_launch() call per MPPI iteration, eliminating all Python-level
orchestration overhead within the rollout. The MPPI weight update and
action resampling run on CPU/numpy after a single wp.synchronize() once
the full rollout graph has completed.

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
    n_spline_points: int   = 3      # control points for spline-smoothed noise
    njmax:           int   = 500
    debug:           bool  = True


# ---------------------------------------------------------------------------
# Module-level Warp Kernels
# ---------------------------------------------------------------------------

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
    t:    int,                        # timestep index — baked in at graph capture
    ctrl: wp.array2d(dtype=float),   # (N, nu)  [out]
):
    """Copy the t-th slice of V into d.ctrl without a Python-side round-trip."""
    n, u = wp.tid()
    ctrl[n, u] += V[n, t, u]


def _make_accumulate_kernel(cost_fn_wp: wp.func):
    """
    Factory that closes over a specific @wp.func at kernel definition time.

    Warp resolves @wp.func references statically when the @wp.kernel decorator
    runs. By accepting cost_fn_wp as a local argument here (rather than reading
    it from self), the compiler can see the concrete function object in the
    enclosing scope and link the call correctly.
    """
    @wp.kernel
    def _kernel(
        qpos:      wp.array2d(dtype=float),
        qvel:      wp.array2d(dtype=float),
        ctrl:      wp.array2d(dtype=float),
        terminal:  bool,
        goal:      wp.array(dtype=float),
        indices:   wp.array(dtype=int),
        costs_out: wp.array(dtype=float),   # (N,)  [in/out]
    ):
        w = wp.tid()
        costs_out[w] += cost_fn_wp(qpos[w], qvel[w], ctrl[w], terminal, goal, indices)

    return _kernel


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MPPIController:
    """MPPI controller backed by a contact model.

    Args:
        mjm:       Host MuJoCo model (already configured with desired geometry).
        cfg:       Contact model config (selects physics backend).
        mppi_cfg:  MPPI hyperparameters.
                   Reference: Williams et al. 2017, "Information Theoretic MPC
                   for Model-Based Reinforcement Learning".
        cost_fn:   A @wp.func with signature
                       (qpos, qvel, ctrl: wp.array(float), terminal: bool) -> float
                   called once per world per timestep on the GPU.
        rng:       NumPy RNG for reproducible noise sampling.
    """

    def __init__(
        self,
        mjm:      mujoco.MjModel,
        cfg:      ContactModelConfig,
        mppi_cfg: MPPIConfig,
        cost_fn:  wp.func,
        rng:      np.random.Generator | None = None,
        initial_ctrl_sequence: np.ndarray | None = None,
    ):
        self.mjm = mjm
        self.cfg = cfg
        self.pc  = mppi_cfg
        self.rng = rng or np.random.default_rng()

        self.nu = mjm.nu
        self.nq = mjm.nq
        self.nv = mjm.nv

        # Handle potential tuple from cost_fn_wp
        if isinstance(cost_fn, tuple):
            self.cost_fn_wp_func, self.goal_wp, self.indices_wp = cost_fn
        else:
            self.cost_fn_wp_func = cost_fn
            # Provide empty fallbacks if not provided
            self.goal_wp = wp.zeros(1, dtype=wp.float32, device="cuda")
            self.indices_wp = wp.zeros(1, dtype=wp.int32, device="cuda")

        # Build the cost-accumulation kernel with this task's cost function
        # baked in at compile time (see factory docstring above).
        self._accumulate_costs_kernel = _make_accumulate_kernel(self.cost_fn_wp_func)

        # ---- GPU buffers -------------------------------------------------
        N, H, nu = mppi_cfg.n_samples, mppi_cfg.horizon, mjm.nu

        # Mean action sequence: (H, nu)
        # if initial_ctrl_sequence is not None:
        #     # Tile the initial control sequence for the entire horizon
        #     self.U_wp = wp.array(
        #         np.tile(initial_ctrl_sequence, (H, 1)),
        #         dtype=wp.float32,
        #         device="cuda"
        #     )
        # else:
        self.U_wp = wp.zeros((H, nu), dtype=wp.float32, device="cuda")

        # Candidate perturbed sequences: (N, H, nu)
        self.V_wp = wp.zeros((N, H, nu), dtype=wp.float32, device="cuda")

        # Per-sample rollout costs: (N,)
        self.costs_wp = wp.zeros(N, dtype=wp.float32, device="cuda")

        # Actuator limits
        self._ctrl_range_wp = wp.array(
            mjm.actuator_ctrlrange, dtype=wp.float32, device="cuda"
        )
        self._has_limits_wp = wp.array(
            mjm.actuator_ctrllimited.astype(bool), dtype=wp.bool, device="cuda"
        )

        # ---- Batched physics model ---------------------------------------
        self.m = api.put_model(mjm, cfg)
        self.d = api.make_data(
            mjm, self.m,
            nworld=N,
            nconmax=mppi_cfg.nconmax,
            njmax=mppi_cfg.njmax,
        )

        # ---- CUDA graph for the inner rollout loop -----------------------
        # Captured once here; replayed every MPPI iteration in plan().
        self._rollout_graph = self._build_rollout_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_rollout_graph(self) -> wp.context.Graph:
        """Record the H-step rollout as a CUDA graph.

        The graph encodes, for each timestep t in [0, H):
            1. _assign_ctrl_kernel  — write V_wp[:, t, :] → d.ctrl   (GPU→GPU)
            2. api.step             — advance all N worlds one step    (GPU)
            3. _accumulate_costs_kernel — add step cost to costs_wp   (GPU)

        Because t is a compile-time integer argument baked into each kernel
        launch at capture time, no Python loop runs during graph replay.
        costs_wp is zeroed as the first node so the graph is fully
        self-contained per iteration.

        Note: api.step() must not perform host-device synchronisation
        internally for graph capture to succeed. Warp-based MuJoCo backends
        satisfy this requirement.
        """
        N  = self.pc.n_samples
        H  = self.pc.horizon
        nu = self.nu

        wp.capture_begin(device="cuda")

        # Zero costs at the start of every rollout.
        self.costs_wp.zero_()

        for t in range(H):
            terminal = (t == H - 1)

            # 1. Write the t-th action slice into d.ctrl — no Python round-trip.
            wp.launch(
                _assign_ctrl_kernel,
                dim=(N, nu),
                inputs=[self.V_wp, t, self.d.ctrl],
            )

            # 2. Advance physics for all N worlds.
            api.step(self.m, self.d)

            # 3. Accumulate per-world costs.
            wp.launch(
                self._accumulate_costs_kernel,
                dim=N,
                inputs=[self.d.qpos, self.d.qvel, self.d.ctrl, terminal, self.goal_wp, self.indices_wp],
                outputs=[self.costs_wp],
            )

        return wp.capture_end(device="cuda")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        """Clear the action sequence (call at the start of a new episode)."""
        self.U_wp.zero_()

    def plan(self, mjd: mujoco.MjData) -> np.ndarray:
        """Run MPPI and return the first action of the optimal sequence.

        CPU↔GPU transfer summary per call
        ----------------------------------
        INTO GPU  : eps (noise)  — once per iteration, before the graph
                    qpos / qvel  — once per iteration, in _set_batch_state
        OUT OF GPU: costs        — once per iteration, after the graph
                    U_wp[0]      — once total, after all iterations
        """
        N     = self.pc.n_samples
        H     = self.pc.horizon
        lam   = self.pc.temperature
        sigma = self.pc.noise_sigma

        t_knots  = np.linspace(0, H - 1, self.pc.n_spline_points)
        t_dense  = np.arange(H)

        for iteration in range(self.pc.n_iterations):

            # ----------------------------------------------------------
            # 1. Sample spline-smoothed noise (CPU)
            # ----------------------------------------------------------
            knot_noise = self.rng.normal(
                0, sigma, (N, self.pc.n_spline_points, self.nu)
            ).astype(np.float32)

            eps = np.empty((N, H, self.nu), dtype=np.float32)
            for n in range(N):
                for j in range(self.nu):
                    eps[n, :, j] = CubicSpline(t_knots, knot_noise[n, :, j])(t_dense)

            # ----------------------------------------------------------
            # 2. Build V = U_mean + eps, clipped — fully on GPU
            # ----------------------------------------------------------
            eps_wp = wp.array(eps, dtype=wp.float32, device="cuda")

            wp.launch(
                _add_noise_and_clip_kernel,
                dim=(N, H, self.nu),
                inputs=[self.U_wp, eps_wp, self._ctrl_range_wp, self._has_limits_wp],
                outputs=[self.V_wp],
            )

            # ----------------------------------------------------------
            # 3. Initialise all N worlds from the current environment state
            # ----------------------------------------------------------
            self._set_batch_state(mjd)

            # ----------------------------------------------------------
            # 4. Run the full H-step rollout — single GPU graph launch
            #    (costs_wp is zeroed inside the graph)
            # ----------------------------------------------------------
            wp.capture_launch(self._rollout_graph)

            # ----------------------------------------------------------
            # 5. Single sync + single transfer: bring costs to CPU
            # ----------------------------------------------------------
            wp.synchronize()
            costs_np = self.costs_wp.numpy()

            # ----------------------------------------------------------
            # 6. MPPI weight update (CPU)
            # ----------------------------------------------------------
            beta = costs_np.min()
            w    = np.exp(-(costs_np - beta) / lam)
            eta  = w.sum() + 1e-8
            w   /= eta

            dU = np.einsum("n,nht->ht", w, eps)   # (H, nu)
            self.U_wp.assign((self.U_wp.numpy() + dU).astype(np.float32))

            if self.pc.debug:
                print(
                    f"avg cost: {costs_np.mean():.4f} +/- {costs_np.std():.4f} "
                    f"min cost: {beta:.4f}  "
                    f"eta: {eta:.4f}"
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_batch_state(self, mjd: mujoco.MjData):
        """Upload the current environment state to all N parallel worlds."""
        #print("BEFORE")
        #print(self.d.qpos)
        api.reset_data(self.mjm, self.m, self.d)
        self.d.qpos.assign(
            np.tile(mjd.qpos, (self.pc.n_samples, 1)).astype(np.float32)
        )
        self.d.qvel.assign(
            np.tile(mjd.qvel, (self.pc.n_samples, 1)).astype(np.float32)
        )
        #print(self.d.qpos)