"""run_kl_divergence_cell.py

HPC worker for the online closed-loop planner-approximation-quality sweep — one degraded-planner
cell. Measures Gaussian KL between two planners' weighted first-action
distributions under the same contact model and recorded planner settings, then pairs
that with the success rate the degraded planner achieves on the eval sim.
Recorded-log replay uses the separate workflow documented in
analysis/README_offline_recorded_kl.md.

The idea
--------
For a fixed proposal, cost-weighted sampling targets a tilted distribution

    q*(V)  proportional to  exp(-S(V)/lambda) * p(V),     p = N(U, sigma^2 I)

where S is the trajectory cost and lambda is the temperature. MPPI never
represents q* explicitly: it draws N samples from p and computes
self-normalized weights w_n ~ exp(-(S_n - beta)/lambda). That weighted particle
set {(V_n, w_n)} approximates that distribution. These are finite-sample,
last-iteration distributions, not a proof of the globally optimal policy.
Multiple iterations can move the proposal centers differently.

Two particle sets cannot be compared with KL directly (disjoint support), so
each is moment-matched to a Gaussian on the FIRST action (the only one that is
ever executed):

    mu    = sum_n w_n V_n[0]                      (equals the planner's U[0])
    Sigma = sum_n w_n (V_n[0]-mu)(V_n[0]-mu)^T

and the analytic Gaussian KL is evaluated in closed form. Comparing the induced
distributions rather than just the proposals includes the effect of costs: the
proposals share sigma by construction, so a proposal-level KL would collapse to
||mu_ref - mu_deg||^2 / (2 sigma^2) and see only mean displacement. The
moment-matched version additionally sees concentration and shape, but loses
non-Gaussian structure such as multiple separated modes.

Why the first action only: it is executed now, and reduces the dimension from
H*nu to nu. The raw covariance rank is at most min(d, N-1), NOT bounded by ESS.
Low ESS can still make the covariance ill-conditioned; even first-action
covariance needs shrinkage and sensitivity checks at small sample counts.

What runs each step
-------------------
Two MPPIControllers share one rollout task, state, horizon, step_time and
noise_sigma. The acting temperature is a cell setting; the higher-compute
reference has its own fixed temperature (50 by default), so a cell with a
different acting temperature is not a pure compute-only comparison:

  * the DEGRADED planner plans every control step and drives the eval sim;
  * the higher-compute REFERENCE planner is a shadow that never controls. It runs
    only every --kl_every steps, because it is the expensive one.

At every measured state, the reference proposal restarts from N(0, sigma).
Its mean is carried only between optimizer iterations within that one plan()
call, then discarded before the next measured state. Under this per-state-zero
protocol, measurements are independent of a stale reference
mean and the reference never receives the degraded planner's saved mean.

For compatibility studies, --reference_init degraded_pre_solve reproduces the
older same-U0 design and --reference_init persistent lets the reference mean
evolve across measured states. Neither compatibility mode is the default.

Optional null control (--null_control)
--------------------------------------
The degraded planner is degraded by having fewer samples, and fewer samples
also make its moment estimates noisier — so measured KL rises with degradation
partly for statistical rather than substantive reasons. With --null_control the
reference planner is rebuilt with the DEGRADED planner's own compute and
temperature, and is seeded from the degraded pre-solve proposal (differing
only in noise seed within that measured solve). This remains a SEPARATE
closed-loop run and is disabled in the default SLURM array. It diagnoses
finite-sample and optimization variability, but its visited states may differ
from the real run. It also uses a different initialization protocol from the
default zero-start real reference. It is not a strictly matched noise floor or
significance test, and should not be subtracted from real KL as a bias correction.

Known approximation (not corrected here)
----------------------------------------
When clipping is enabled, V = clamp(U + eps, delta_range) has boundary masses
(a clipped, not truncated, Gaussian). Moment-matching is an approximation. It bites
harder than for a mean-only comparison, because clipped dimensions get
artificially small variance which feeds straight into the log-det term. Left
uncorrected deliberately; treat cells whose actions ride the delta clip with
suspicion.

Numerical safety
----------------
Sigma is shrunk toward the proposal covariance before inversion:

    Sigma <- (1-alpha) Sigma + alpha * sigma^2 I

with alpha = --kl_shrinkage. sigma^2 I is the isotropic proposal-scale target.
Finite samples or clipping need not have exactly that covariance. Shrinkage keeps the
log-det finite when the effective sample size collapses toward 1, which does
happen on this task even when the mean ESS is healthy.

    python run_kl_divergence_cell.py \
        --outdir results/kl_divergence_eval_run \
        --task grasp_reorient --model M3 --geometry cube_high_high \
        --n_samples 64 --n_iterations 1 \
        --ref_n_samples 4096 --ref_convergence_tol 1e-3 \
        --ref_max_iterations 25 --reference_init zero \
        --n_episodes 5 --kl_every 20
"""

from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from dataclasses import asdict
import hashlib
import json
import re
import time
import uuid
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.evaluation.metrics import (
    EpisodeResult, aggregate_episodes,
)
from contact_study.planners.mppi import MPPIConfig, MPPIController
from contact_study.tasks.base import get_task
from contact_study.tasks.config import EvalSimulatorKind, TaskRole

from contact_study.drivers.run_eval_episode import (
    apply_goal_difficulty, load_rollout_task, resolve_mppi_schedule,
    MODEL_FACTORIES,
)
from contact_study.tasks.config import SceneVariant
from contact_study.evaluation import json_io
from contact_study.evaluation.distributions import (
    gaussian_kl, weighted_moments, weighted_moments_from_particles,
)
from contact_study.evaluation.trajectory import (
    TrajectoryConfig, TrajectoryRecorder, add_cli_flags as add_record_flags,
)


REFERENCE_INIT_CHOICES = ("zero", "degraded_pre_solve", "persistent")


def optional_positive_float(value: str) -> float | None:
    """Parse a positive float, with ``none`` selecting fixed iterations."""
    if value.lower() in {"none", "off", "fixed"}:
        return None
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be positive and finite, or 'none'")
    return number


def reference_init_mode(args) -> str:
    """Resolve the reference-start policy, including the legacy CLI alias."""
    legacy = getattr(args, "sync_reference_mean", None)
    requested = (args.reference_init if legacy is None else
                 ("degraded_pre_solve" if legacy else "persistent"))
    # The optional equal-compute null is only meaningful as a within-state
    # stochastic replicate if both planners start from the same proposal.
    # It is deliberately not part of the default zero-start sweep.
    if getattr(args, "null_control", False):
        return "degraded_pre_solve"
    return requested


def initialize_reference(ref, mode: str, degraded_pre_solve_mean=None) -> None:
    """Prepare the shadow reference immediately before one measured solve."""
    if mode == "zero":
        # reset() zeros the whole proposal mean, resets adaptive parameters and
        # restarts the plan cadence. The monotonically increasing resample
        # counter is deliberately retained, so separate states get fresh noise.
        ref.reset()
    elif mode == "degraded_pre_solve":
        if degraded_pre_solve_mean is None:
            raise ValueError("degraded_pre_solve initialization requires a saved mean")
        ref.U_wp.assign(degraded_pre_solve_mean)
    elif mode != "persistent":
        raise ValueError(f"Unknown reference initialization mode: {mode!r}")


def resolve_kl_geometry(task: str, geometry: str) -> str:
    """Use explicit high/high geometry for the grasp-reorient compute study.

    Object-only selectors are convenient, but legacy fidelity aliases are
    rejected instead of silently resolving to cube_low_high. Other experiment
    drivers and the shared SceneVariant defaults are unchanged.
    """
    if task != "grasp_reorient":
        return geometry
    value = geometry.strip()
    if value in {"accurate", "convex_hull", "primitive_union", "linearized"}:
        raise ValueError("KL geometry must name an object at high_high; use "
                         "--geometry cube_high_high, not a legacy alias.")
    if re.fullmatch(r"[a-zA-Z0-9]+", value):
        value += "_high_high"
    variant = SceneVariant.parse(value)
    if (variant.hand_acc, variant.obj_acc) != ("high", "high"):
        raise ValueError("The KL compute study requires high_high geometry for "
                         "both planners. Example: --geometry duck_high_high")
    if not re.fullmatch(r"[a-zA-Z0-9]+", variant.obj):
        raise ValueError("Invalid object name in KL geometry.")
    return f"{variant.obj}_high_high"


def result_filename(out: dict) -> str:
    """Names include geometry, config identity and a unique run identifier.

    A repeated seed is not new independent evidence; the plotter checks that
    separately. UUIDs prevent reruns or concurrent jobs overwriting raw data.
    """
    encoded = json.dumps(out["config"], sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()[:12]
    label = re.sub(r"[^a-zA-Z0-9_.-]", "_", out["label"])
    return f"{label}_{digest}_seed{out['config']['seed']}_{out['run_id']}.json"


def validate_kl_args(args) -> None:
    """Reject invalid numerical settings before starting expensive rollouts."""
    for name in ("n_episodes", "kl_every", "n_samples", "ref_n_samples",
                 "n_iterations", "ref_n_iterations", "ref_max_iterations",
                 "nconmax", "njmax"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    for name in ("max_steps", "eval_substeps"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name} must be positive")
    for name in ("noise_sigma", "temperature", "ref_temperature",
                 "time_horizon", "step_time"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name} must be finite and positive")
    for name in ("settle", "delta"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError(f"--{name} must be finite and nonnegative")
    if not (0.0 < args.kl_shrinkage <= 1.0):
        raise ValueError("--kl_shrinkage must be in (0, 1]")
    if (args.ref_convergence_tol is not None
            and (not np.isfinite(args.ref_convergence_tol)
                 or args.ref_convergence_tol <= 0.0)):
        raise ValueError("--ref_convergence_tol must be positive and finite, or none")
    if args.ref_convergence_tol is not None and args.ref_max_iterations < 2:
        raise ValueError("--ref_max_iterations must be at least 2 with convergence")
    mode = reference_init_mode(args)
    if mode not in REFERENCE_INIT_CHOICES:
        raise ValueError(f"--reference_init must be one of {REFERENCE_INIT_CHOICES}")
    if args.task == "grasp_reorient" and args.eval_sim == "drake":
        raise ValueError("The all-object KL workflow supports Pinocchio or "
                         "MuJoCo evaluation. The legacy Drake hand-only asset "
                         "does not implement these per-object scenes.")


# ---------------------------------------------------------------------------
# One episode with two planners
# ---------------------------------------------------------------------------
def measure_distribution(controller, args, sigma):
    """Optionally retain raw moments of the exact candidates used for KL.

    No extra solve or random draw is made. Raw moments permit CPU-only
    shrinkage sensitivity on fixed weighted particles, without reconstructing
    a nearly singular covariance by subtracting its regularization later.
    """
    if not args.record_kl_moments:
        mu, cov, ess = weighted_moments(controller, args.kl_shrinkage, sigma)
        return mu, cov, ess, None
    particles = controller.V_wp.numpy()[:, 0, :]
    weights = controller.w_wp.numpy()
    mu, raw, ess = weighted_moments_from_particles(particles, weights, 0.0, sigma)
    cov = ((1.0 - args.kl_shrinkage) * raw
           + args.kl_shrinkage * sigma ** 2 * np.eye(mu.size))
    record = {
        "mean": json_io.compact(mu, precision=0),
        "covariance_raw": json_io.compact(raw, precision=0),
        "ess": ess,
        "n_particles": int(particles.shape[0]),
    }
    return mu, cov, ess, record


def run_kl_episode(args, contact_cfg, deg_cfg: MPPIConfig, ref_cfg: MPPIConfig,
                   rng, geometry, eval_sim, ep_idx: int):
    """Closed-loop episode driven by the degraded planner, shadowed by the
    reference planner. Returns (EpisodeResult, per-step KL record dict).

    Forked from contact_study.drivers.run_eval_episode — same eval/rollout
    split, same control parameterization and clipping — with a second
    controller and the KL bookkeeping added. Kept here rather than in
    drivers/ so this experimental path cannot destabilize the shared driver.
    """
    # ---- ROLLOUT task + both planners -------------------------------------
    rollout_task = get_task(args.task, geometry=geometry, role=TaskRole.ROLLOUT)
    if args.goal_difficulty is not None:
        apply_goal_difficulty(rollout_task, args.goal_difficulty)
    mjm, mjd = rollout_task.load()
    cfg = rollout_task.config

    eval_dt       = cfg.timestep
    eval_substeps = args.eval_substeps if args.eval_substeps is not None \
        else cfg.eval_substeps_per_rollout
    rollout_dt = eval_dt * eval_substeps
    mjm.opt.timestep = rollout_dt

    # Both controllers share the task (and therefore the cost weights and the
    # goal array). api.put_model saves/restores mjm, so building the second one
    # does not disturb the first.
    deg = MPPIController(task=rollout_task, cfg=contact_cfg, mppi_cfg=deg_cfg, rng=rng)
    ref = MPPIController(task=rollout_task, cfg=contact_cfg, mppi_cfg=ref_cfg, rng=rng)

    # The KL is only defined if both distributions live in the same space.
    if (deg.horizon, deg.nu, deg.substeps) != (ref.horizon, ref.nu, ref.substeps):
        raise ValueError(
            f"reference and degraded planners must share the schedule; got "
            f"deg=(H={deg.horizon}, nu={deg.nu}, substeps={deg.substeps}) "
            f"ref=(H={ref.horizon}, nu={ref.nu}, substeps={ref.substeps}). "
                "Sample/iteration budgets and MPPI temperature may differ, "
                "but the schedule and action dimension must match."
        )

    # ---- EVAL task + "real" simulator -------------------------------------
    eval_task = get_task(args.task, geometry=geometry, role=TaskRole.EVAL)
    if args.goal_difficulty is not None:
        apply_goal_difficulty(eval_task, args.goal_difficulty)
    eval_task.load()
    if eval_sim is not None:
        eval_task.config.eval_sim = eval_sim
    sim = eval_task.make_eval_simulator(video_path=None, render=False)

    # ---- initial state -----------------------------------------------------
    q0, v0, u0 = rollout_task.get_inital_state(rng)
    sim.reset(np.asarray(q0, dtype=float), np.asarray(v0, dtype=float))
    u = np.asarray(u0, dtype=float).copy()

    if args.settle > 0.0:
        for _ in range(int(args.settle / rollout_dt)):
            sim.apply_control(u)
            sim.step(eval_substeps)

    if hasattr(rollout_task, "sample_new_goal"):
        st = sim.get_state()
        mjd.qpos[:] = st.qpos
        mjd.qvel[:] = st.qvel
        mujoco.mj_forward(mjm, mjd)
        rollout_task.sample_new_goal(mjd, rng)

    if cfg.force_limits is not None:
        clip_lo, clip_hi = cfg.force_limits
    elif cfg.control_limits is not None:
        clip_lo, clip_hi = cfg.control_limits
    else:
        clip_lo = clip_hi = None

    eval_steps_per_control = deg.substeps * eval_substeps
    n_steps = args.max_steps if args.max_steps is not None else cfg.max_steps
    steps_to_success: int | None = None
    sigma = float(deg_cfg.noise_sigma)
    ref_init = reference_init_mode(args)

    # Records the DEGRADED planner — the one that actually controls. The
    # reference planner's shadow solve is not part of the episode.
    rec = TrajectoryRecorder(
        TrajectoryConfig.from_args(args), deg, driver="sync",
        control_dt=deg.control_dt, rollout_dt=rollout_dt, eval_dt=eval_dt,
        eval_substeps=eval_substeps, max_steps=n_steps,
        clip=(clip_lo, clip_hi), settle_seconds=args.settle,
        extra_context={
            "task":            args.task,
            "geometry":        geometry,
            "planner":         "mppi",
            "model_label":     contact_cfg.label,
            "nq":              int(mjm.nq),
            "nv":              int(mjm.nv),
            "q0":              np.asarray(q0, dtype=float),
            "v0":              np.asarray(v0, dtype=float),
            "u0":              np.asarray(u0, dtype=float),
            "kl_every":        int(args.kl_every),
            "kl_shrinkage":    float(args.kl_shrinkage),
            "ref_n_samples":   int(ref_cfg.n_samples),
            "ref_temperature": float(ref_cfg.temperature),
            "reference_initialization": ref_init,
            "ref_convergence_tol": ref_cfg.convergence_tol,
            "ref_max_iterations": int(ref_cfg.max_iterations),
        },
    )
    # Which of the loop's three exits was taken; see EpisodeResult.end_reason.
    end_reason    = "timeout"
    n_steps_taken = n_steps

    kl_fwd, kl_rev, ess_ref, ess_deg, mu_dist, kl_steps = [], [], [], [], [], []
    invalid_kl_steps: list[int] = []
    step_times: list[float] = []
    ref_plan_times: list[float] = []
    ref_iterations: list[int] = []
    ref_converged: list[bool | None] = []
    ref_residuals: list[float | None] = []
    ref_solve_records: list[dict] = []
    moment_records: list[dict] = []
    ep_start = time.perf_counter()

    for t in range(n_steps):
        st = sim.get_state()
        mjd.qpos[:] = st.qpos
        mjd.qvel[:] = st.qvel
        mjd.ctrl[:] = u
        mujoco.mj_forward(mjm, mjd)

        if rollout_task.is_success(mjd):
            if steps_to_success is None:
                steps_to_success = t
                if args.debug:
                    print(f"  [ep {ep_idx:02d}] first success at step {t}")
            end_reason, n_steps_taken = "success", t
            break
        if rollout_task.has_failed(mjd):
            end_reason, n_steps_taken = "failed", t
            break

        measure = (t % args.kl_every == 0)

        # The compatibility protocol needs the degraded proposal from before
        # its solve. The default zero protocol never reads or copies that mean.
        U0 = (deg.U_wp.numpy().copy()
              if measure and ref_init == "degraded_pre_solve" else None)

        # --- degraded planner: this is the one that controls ---------------
        plan_start = time.perf_counter()
        action = deg.plan(mjd)
        plan_ms = (time.perf_counter() - plan_start) * 1e3
        step_times.append(plan_ms)

        if measure:
            mu_d, S_d, e_d, moments_d = measure_distribution(deg, args, sigma)

            # --- reference planner: shadow solve at the same physical state --
            # Default protocol: start this individual measurement from
            # N(0, sigma), retain the updated mean only across optimizer
            # iterations inside ref.plan(), then discard it at the next reset.
            initialize_reference(ref, ref_init, U0)
            ref_start = time.perf_counter()
            ref.plan(mjd)
            ref_plan_times.append((time.perf_counter() - ref_start) * 1e3)
            mu_r, S_r, e_r, moments_r = measure_distribution(ref, args, sigma)
            n_ref_iter = int(ref.last_n_iterations)
            did_converge = ref.last_converged
            residual = ref.last_convergence_residual

            # Direction matters: each expectation is taken under its first
            # distribution. Both mean differences and covariance shape matter.
            f = gaussian_kl(mu_r, S_r, mu_d, S_d)
            r = gaussian_kl(mu_d, S_d, mu_r, S_r)
            valid = bool(deg.last_plan_ok and ref.last_plan_ok
                         and np.isfinite(f) and np.isfinite(r))
            ref_solve_records.append({
                "step": t,
                "valid_kl": valid,
                "n_iterations": n_ref_iter,
                "converged": did_converge,
                "convergence_residual": residual,
            })
            if valid:
                kl_fwd.append(f)
                kl_rev.append(r)
                ess_ref.append(e_r)
                ess_deg.append(e_d)
                mu_dist.append(float(np.linalg.norm(mu_r - mu_d)))
                kl_steps.append(t)
                ref_iterations.append(n_ref_iter)
                ref_converged.append(did_converge)
                ref_residuals.append(residual)
                if args.record_kl_moments:
                    moment_records.append({"step": t, "degraded": moments_d,
                                           "reference": moments_r})
            else:
                # Do not silently reuse a stale particle cloud after an
                # all-NaN planner solve. Keep the step number so a failed cell
                # is diagnosable instead of merely returning fewer samples.
                invalid_kl_steps.append(t)

        # --- turn the planned delta into the absolute command --------------
        if args.execute == "sample":
            # Draw a particle from the degraded planner's induced distribution
            # (weighted, not uniform) instead of using its mean.
            w = deg.w_wp.numpy().astype(np.float64)
            s = w.sum()
            w = w / s if (np.isfinite(s) and s > 0) else np.full(w.size, 1.0 / w.size)
            idx = int(rng.choice(w.size, p=w))
            action = deg.V_wp.numpy()[idx, 0, :].astype(np.float64)

        if deg.pc.ctrl_relative_to_qpos:
            adr = deg.robot_qpos_adr
            u = st.qpos[adr : adr + deg.nu] + action
        else:
            u = u + action
        if clip_lo is not None:
            u = np.clip(u, clip_lo, clip_hi)

        # `action` is whatever built `u` — under --execute sample that is the
        # drawn particle, not the planner's mean, so the replay identity holds
        # either way. action_source records which.
        rec.step(step=t, t=t * deg.control_dt, qpos=st.qpos, qvel=st.qvel,
                 action=action, ctrl=u, plan_ms=plan_ms,
                 action_source=args.execute)

        sim.apply_control(u)
        sim.step(eval_steps_per_control)

        if args.debug and measure and t % (args.kl_every * 10) == 0:
            print(f"  [ep {ep_idx:02d} | step {t:04d}]  "
                  f"KL_fwd={kl_fwd[-1] if kl_fwd else float('nan'):9.4f}  "
                  f"KL_rev={kl_rev[-1] if kl_rev else float('nan'):9.4f}  "
                  f"ESS ref={ess_ref[-1] if ess_ref else float('nan'):7.1f}/"
                  f"{ref_cfg.n_samples} deg={ess_deg[-1] if ess_deg else float('nan'):7.1f}/"
                  f"{deg_cfg.n_samples}")

    # Also score the state produced by the last allowed command. Without this
    # boundary check, success or a drop on that command is labelled timeout.
    if end_reason == "timeout":
        final_state = sim.get_state()
        mjd.qpos[:] = final_state.qpos
        mjd.qvel[:] = final_state.qvel
        mujoco.mj_forward(mjm, mjd)
        if rollout_task.is_success(mjd):
            steps_to_success = n_steps_taken
            end_reason = "success"
        elif rollout_task.has_failed(mjd):
            end_reason = "failed"

    elapsed  = time.perf_counter() - ep_start
    step_arr = np.asarray(step_times)
    final_qpos = sim.get_state().qpos

    result = EpisodeResult(
        task_name        = cfg.name,
        model_label      = contact_cfg.label,
        success          = steps_to_success is not None,
        steps_to_success = steps_to_success,
        final_cost       = float(np.linalg.norm(final_qpos - np.asarray(q0, dtype=float))),
        n_samples_used   = deg_cfg.n_samples,
        elapsed_seconds  = elapsed,
        mean_step_ms     = float(step_arr.mean()) if len(step_arr) else 0.0,
        std_step_ms      = float(step_arr.std())  if len(step_arr) else 0.0,
        **rollout_task.goal_spec(),
        time_out         = end_reason == "timeout",
        end_reason       = end_reason,
        n_steps_taken    = n_steps_taken,
        trajectory       = rec.finish(),
    )
    record = {
        "steps":     kl_steps,
        "kl_forward": kl_fwd,
        "kl_reverse": kl_rev,
        "ess_ref":   ess_ref,
        "ess_deg":   ess_deg,
        "mu_dist":   mu_dist,
        "reference_plan_ms": ref_plan_times,
        # These three arrays align one-for-one with steps/kl_forward/kl_reverse.
        # ``converged`` is None only for explicit fixed-iteration compatibility
        # runs, where no convergence test exists.
        "reference_n_iterations": ref_iterations,
        "reference_converged": ref_converged,
        "reference_convergence_residual": ref_residuals,
        # Includes invalid KL attempts as well, for complete diagnostics.
        "reference_solves": ref_solve_records,
        "invalid_steps": invalid_kl_steps,
    }
    if args.record_kl_moments:
        record["moments"] = moment_records
    return result, record


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------
def _stats(v: list[float]) -> dict:
    """Summary of a per-step series. Heavy-tailed across contact vs free-flight
    states, so quantiles are reported alongside the mean and the raw arrays are
    kept, letting the aggregation choice be revisited without re-running."""
    if not v:
        return {"n": 0, "mean": None, "sd": None,
                "median": None, "p25": None, "p75": None, "p95": None}
    a = np.asarray(v, dtype=np.float64)
    return {
        "n":      int(a.size),
        "mean":   float(a.mean()),
        "sd":     float(a.std()),
        "median": float(np.median(a)),
        "p25":    float(np.percentile(a, 25)),
        "p75":    float(np.percentile(a, 75)),
        "p95":    float(np.percentile(a, 95)),
    }


def run_cell(args):
    validate_kl_args(args)

    geometry = resolve_kl_geometry(args.task, args.geometry)
    eval_peek = get_task(args.task, geometry=geometry, role=TaskRole.EVAL)
    eval_sim = (eval_peek.config.eval_sim if args.eval_sim == "none"
                else EvalSimulatorKind(args.eval_sim))
    eval_scene = eval_peek.resolve_scene_path()
    contact_cfg = MODEL_FACTORIES[args.model]()

    # In null-control mode the "reference" is a second instance of the degraded
    # planner (different noise seed only). This independent closed-loop run is
    # a stochastic diagnostic, not an exactly state-matched noise floor.
    ref_n_samples = args.n_samples if args.null_control else args.ref_n_samples
    # Null keeps equal compute with the degraded planner. The real reference
    # uses convergence termination by default; its recorded iteration count is
    # the cap, not a promise that every solve runs that many iterations.
    ref_convergence_tol = None if args.null_control else args.ref_convergence_tol
    ref_n_iterations = (args.n_iterations if args.null_control else
                        (args.ref_max_iterations if ref_convergence_tol is not None
                         else args.ref_n_iterations))
    comparison_ref_iterations = (args.ref_max_iterations
                                 if args.ref_convergence_tol is not None
                                 else args.ref_n_iterations)
    ref_init = reference_init_mode(args)
    ref_temperature = args.temperature if args.null_control else args.ref_temperature

    label = f"{geometry}_{args.model}_n{args.n_samples}_i{args.n_iterations}"
    if args.null_control:
        label += "_null"

    peek = load_rollout_task(args.task, geometry)
    if args.goal_difficulty is not None:
        apply_goal_difficulty(peek, args.goal_difficulty)
    effective_difficulty = getattr(peek, "goal_difficulty", peek.config.difficulty)
    eval_substeps = (args.eval_substeps if args.eval_substeps is not None
                    else peek.config.eval_substeps_per_rollout)
    horizon, substeps, rollout_dt = resolve_mppi_schedule(
        MPPIConfig(time_horizon=args.time_horizon, step_time=args.step_time),
        peek.config, args.eval_substeps,
    )
    n_steps = args.max_steps if args.max_steps is not None else peek.config.max_steps

    print(f"[{label}]  nq={peek.mjm.nq} nv={peek.mjm.nv} nu={peek.mjm.nu}  "
          f"max_steps={n_steps}  n_episodes={args.n_episodes}")
    print(f"[{label}]  geometry={geometry} eval_sim={eval_sim.value} "
          f"goal_difficulty={effective_difficulty} "
          f"ccd_iterations={peek.mjm.opt.ccd_iterations} (unchanged)")
    print(f"[{label}]  degraded : n_samples={args.n_samples} "
          f"n_iterations={args.n_iterations} temperature={args.temperature:g}")
    ref_budget = (f"convergence_tol={ref_convergence_tol:g} "
                  f"max_iterations={ref_n_iterations}"
                  if ref_convergence_tol is not None
                  else f"fixed_iterations={ref_n_iterations}")
    print(f"[{label}]  reference: n_samples={ref_n_samples} {ref_budget} "
          f"temperature={ref_temperature:g}"
          + ("   (NULL CONTROL: same fixed compute, different seed)"
             if args.null_control else ""))
    print(f"[{label}]  schedule : horizon={horizon} substeps={substeps} "
          f"rollout_dt={rollout_dt*1e3:.3f}ms  (shared by both planners)")
    print(f"[{label}]  KL       : first action (d={peek.mjm.nu}), every {args.kl_every} steps, "
          f"shrinkage={args.kl_shrinkage:g}, headline={args.kl_direction}, "
          f"reference_init={ref_init}, execute={args.execute}")

    delta_range = (-args.delta, args.delta) if args.delta is not None else (None, None)

    def make_cfg(n_samples, n_iterations, seed, *, temperature,
                 convergence_tol=None, max_iterations=10):
        return MPPIConfig(
            n_samples      = n_samples,
            n_iterations   = n_iterations,
            time_horizon   = args.time_horizon,
            step_time      = args.step_time,
            temperature    = temperature,
            noise_sigma    = args.noise_sigma,
            warm_start     = False,
            resample_interval = 1,
            use_full_graph = args.use_full_graph,
            delta_range    = delta_range,
            nconmax        = args.nconmax,
            njmax          = args.njmax,
            seed           = seed,
            convergence_tol = convergence_tol,
            max_iterations = max_iterations,
            debug          = False,   # per-plan MPPI spam would swamp two planners
        )

    # Common-random-number protocol. Every sweep cell receives the same
    # environment/goal seed for episode k, and a real/null pair receives the
    # same degraded-planner seed. Only the shadow reference gets its own stream.
    # This keeps task difficulty and the executed controller paired across the
    # sweep instead of confounding KL with a different randomly sampled goal.
    root_seed = (int(args.seed) if args.seed is not None else
                 int(np.random.SeedSequence().generate_state(1)[0]))

    def child_seed(stream: int, ep: int, *extra: int) -> int:
        return int(np.random.SeedSequence(
            [root_seed, stream, ep, *(int(x) for x in extra)]
        ).generate_state(1)[0])

    episodes, records, seed_records = [], [], []
    for ep in range(args.n_episodes):
        env_seed = child_seed(0, ep)
        deg_seed = child_seed(1, ep)
        # The real reference is held fixed across degraded-compute cells. The
        # null reference uses a separate stream but is keyed by its own compute
        # so it remains an independent, matched replicate of the degraded one.
        ref_seed = (child_seed(3, ep, args.n_samples, args.n_iterations)
                    if args.null_control else child_seed(2, ep))
        rng = np.random.default_rng(env_seed)

        deg_cfg = make_cfg(
            args.n_samples, args.n_iterations, deg_seed,
            temperature=args.temperature,
        )
        ref_cfg = make_cfg(
            ref_n_samples, ref_n_iterations, ref_seed,
            temperature=ref_temperature,
            convergence_tol=ref_convergence_tol,
            max_iterations=ref_n_iterations,
        )

        result, rec = run_kl_episode(
            args, contact_cfg, deg_cfg, ref_cfg, rng, geometry, eval_sim, ep,
        )
        episodes.append(result)
        records.append(rec)
        seed_records.append({
            "episode_index": ep,
            "environment_seed": env_seed,
            "degraded_planner_seed": deg_seed,
            "reference_planner_seed": ref_seed,
        })

        tick = "✓" if result.success else "✗"
        kf, kr = _stats(rec["kl_forward"]), _stats(rec["kl_reverse"])
        ref_ms = _stats(rec["reference_plan_ms"])["mean"]
        print(f"    ep {ep:02d}  {tick}  "
              f"KL_fwd={kf['mean'] if kf['mean'] is not None else float('nan'):9.4f}  "
              f"KL_rev={kr['mean'] if kr['mean'] is not None else float('nan'):9.4f}  "
              f"n_kl={kf['n']:4d}  deg_step={result.mean_step_ms:.1f}ms  "
              f"ref_plan={(ref_ms if ref_ms is not None else float('nan')):.1f}ms")

    agg = aggregate_episodes(episodes, args.task, label)

    # Preserve legacy pooled-step summaries. The plotter's default instead
    # reconstructs episode-balanced means from per-episode/per-step records.
    all_fwd = [x for r in records for x in r["kl_forward"]]
    all_rev = [x for r in records for x in r["kl_reverse"]]
    all_er  = [x for r in records for x in r["ess_ref"]]
    all_ed  = [x for r in records for x in r["ess_deg"]]
    all_md  = [x for r in records for x in r["mu_dist"]]
    all_ref_ms = [x for r in records for x in r["reference_plan_ms"]]
    all_ref_iterations = [x for r in records for x in r["reference_n_iterations"]]
    all_ref_converged = [x for r in records for x in r["reference_converged"]]
    all_ref_residuals = [x for r in records
                         for x in r["reference_convergence_residual"] if x is not None]
    n_invalid = sum(len(r["invalid_steps"]) for r in records)

    # Same finite KL estimates, two reporting rules requested for the
    # convergence-based reference: all valid measurements, and only those whose
    # reference solve met the tolerance before/at the cap.
    converged_fwd = [value for r in records
                     for value, ok in zip(r["kl_forward"], r["reference_converged"])
                     if ok is True]
    converged_rev = [value for r in records
                     for value, ok in zip(r["kl_reverse"], r["reference_converged"])
                     if ok is True]
    n_converged = sum(ok is True for ok in all_ref_converged)
    n_not_converged = sum(ok is False for ok in all_ref_converged)

    headline = all_fwd if args.kl_direction == "forward" else all_rev
    headline_stats = _stats(headline)
    headline_mean = (headline_stats["mean"] if headline_stats["mean"] is not None
                     else float("nan"))
    headline_median = (headline_stats["median"] if headline_stats["median"] is not None
                       else float("nan"))
    ess_ref_mean = _stats(all_er)["mean"]
    ess_deg_mean = _stats(all_ed)["mean"]
    ref_plan_mean = _stats(all_ref_ms)["mean"]
    headline_field = "kl_forward" if args.kl_direction == "forward" else "kl_reverse"
    episode_headline = [float(np.mean(r[headline_field])) for r in records if r[headline_field]]
    episode_headline_mean = float(np.mean(episode_headline)) if episode_headline else float("nan")
    print(f"  → success={agg.success_rate*100:.1f}%  "
          f"KL_{args.kl_direction}_episode_mean={episode_headline_mean:.4f} "
          f"(valid_episodes={len(episode_headline)}; "
          f"pooled_step_mean={headline_mean:.4f}, "
          f"pooled_step_median={headline_median:.4f}, n_steps={len(headline)})  "
          f"ESS ref={(ess_ref_mean if ess_ref_mean is not None else float('nan')):.1f}/"
          f"{ref_n_samples} deg={(ess_deg_mean if ess_deg_mean is not None else float('nan')):.1f}/"
          f"{args.n_samples} ref_plan="
          f"{(ref_plan_mean if ref_plan_mean is not None else float('nan')):.1f}ms "
          f"invalid_KL={n_invalid}")

    out = {
        "schema_version": 2,
        "run_id": uuid.uuid4().hex,
        "label": label,
        "task":  args.task,
        "model": args.model,
        "geometry": geometry,
        "object": peek.scene_variant.obj,
        "config": {
            "kl_protocol": f"first_action_v3_reference_{ref_init}_final_state_check",
            "record_kl_moments": args.record_kl_moments,
            "geometry": geometry,
            "object": peek.scene_variant.obj,
            "planner": "mppi",
            "eval_sim": eval_sim.value,
            "eval_dt": float(peek.config.timestep),
            "eval_substeps": int(eval_substeps),
            "settle": args.settle,
            "contact_config": asdict(contact_cfg),
            "cost_weights": dict(peek.config.cost_weights),
            "success_thresholds": dict(peek.config.success_thresholds),
            "ccd_iterations": int(peek.mjm.opt.ccd_iterations),
            "ccd_tolerance": float(peek.mjm.opt.ccd_tolerance),
            "nconmax": args.nconmax,
            "njmax": args.njmax,
            "use_full_graph": args.use_full_graph,
            "warm_start": False,
            "resample_interval": 1,
            "control_dt": float(substeps * rollout_dt),
            "realized_time_horizon": float(horizon * substeps * rollout_dt),
            "n_samples":        args.n_samples,
            "n_iterations":     args.n_iterations,
            "ref_n_samples":    ref_n_samples,
            "ref_n_iterations": ref_n_iterations,
            "ref_iteration_mode": ("convergence" if ref_convergence_tol is not None
                                   else "fixed"),
            "ref_convergence_tol": ref_convergence_tol,
            "ref_max_iterations": ref_n_iterations,
            "ref_temperature": ref_temperature,
            # Retain the intended high-compute comparison configuration as
            # provenance even for an explicitly requested null. The null's
            # distinct initialization protocol keeps it scientifically
            # separate from the default real cell.
            "comparison_ref_n_samples": args.ref_n_samples,
            "comparison_ref_n_iterations": comparison_ref_iterations,
            "comparison_ref_iteration_mode": (
                "convergence" if args.ref_convergence_tol is not None else "fixed"
            ),
            "comparison_ref_convergence_tol": args.ref_convergence_tol,
            "comparison_ref_max_iterations": comparison_ref_iterations,
            "comparison_ref_temperature": args.ref_temperature,
            "null_control":     args.null_control,
            "kl_every":         args.kl_every,
            "kl_shrinkage":     args.kl_shrinkage,
            "kl_direction":     args.kl_direction,
            "reference_initialization": ref_init,
            "execute":          args.execute,
            "time_horizon":     args.time_horizon,
            "step_time":        args.step_time,
            "temperature":      args.temperature,
            "noise_sigma":      args.noise_sigma,
            "delta":            args.delta,
            "horizon":          horizon,
            "action_dim":       int(peek.mjm.nu),
            "substeps":         substeps,
            "rollout_dt":       rollout_dt,
            "max_steps":        n_steps,
            "n_episodes":       args.n_episodes,
            "seed":             root_seed,
            "seed_protocol":    "common_random_numbers_v1",
            "goal_difficulty":  effective_difficulty,
        },
        "scene_files": {
            "rollout": peek.resolve_scene_path().name,
            "eval": eval_scene.name,
        },
        # AggregatedResult carries the success-rate side, in the same schema the
        # rest of the study's sweeps use.
        "aggregate": agg.to_dict(),
        "kl": {
            "headline_direction": args.kl_direction,
            "forward": _stats(all_fwd),
            "reverse": _stats(all_rev),
        },
        "kl_converged_only": {
            "definition": "valid KL measurements whose reference solve met its tolerance",
            "forward": _stats(converged_fwd),
            "reverse": _stats(converged_rev),
        },
        "diagnostics": {
            "ess_ref": _stats(all_er),
            "ess_deg": _stats(all_ed),
            "mu_dist": _stats(all_md),
            "reference_plan_ms": _stats(all_ref_ms),
            "reference_n_iterations": _stats(all_ref_iterations),
            "reference_convergence_residual": _stats(all_ref_residuals),
            "n_reference_converged": n_converged,
            "n_reference_not_converged": n_not_converged,
            "reference_convergence_rate": (
                n_converged / (n_converged + n_not_converged)
                if n_converged + n_not_converged else None
            ),
            "n_invalid_kl": n_invalid,
        },
        # The full EpisodeResult (end_reason / time_out / trajectory included),
        # with this cell's KL summary folded in beside it.
        "episodes": [
            {
                **e.to_dict(),
                **seeds,
                "kl_forward": _stats(r["kl_forward"]),
                "kl_reverse": _stats(r["kl_reverse"]),
                "kl_converged_only_forward": _stats([
                    value for value, ok in zip(
                        r["kl_forward"], r["reference_converged"]
                    ) if ok is True
                ]),
                "kl_converged_only_reverse": _stats([
                    value for value, ok in zip(
                        r["kl_reverse"], r["reference_converged"]
                    ) if ok is True
                ]),
                "reference_plan_ms": _stats(r["reference_plan_ms"]),
                "reference_n_iterations": _stats(r["reference_n_iterations"]),
                "n_reference_converged": sum(
                    ok is True for ok in r["reference_converged"]
                ),
                "n_reference_not_converged": sum(
                    ok is False for ok in r["reference_converged"]
                ),
            }
            for e, r, seeds in zip(episodes, records, seed_records)
        ],
        # Raw per-step series (a few hundred floats per episode) so the
        # aggregation choice can be changed without re-running the sweep.
        "per_step": records,
    }
    return out, label


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HPC worker: run one KL-divergence vs success-rate cell."
    )
    p.add_argument("--task",  type=str, default="grasp_reorient")
    p.add_argument("--model", type=str, default="M3", choices=list(MODEL_FACTORIES),
                   help="Contact model. Both planners use the SAME one — this "
                        "experiment varies optimizer quality, not model fidelity.")
    p.add_argument("--outdir", type=str, default="results/kl_divergence_eval_run")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--goal_difficulty", type=int, default=1,
                   help="Task goal difficulty (default 1, matching the KL grid).")

    # --- the swept axis: degraded-planner compute --------------------------
    p.add_argument("--n_samples",    type=int, default=64,
                   help="Degraded planner sample count (swept).")
    p.add_argument("--n_iterations", type=int, default=1,
                   help="Degraded planner MPPI iterations per plan (swept).")
    # --- the higher-compute reference planner -------------------------------
    p.add_argument("--ref_n_samples",    type=int, default=4096,
                   help="Reference planner sample count. Ignored with --null_control.")
    p.add_argument("--ref_n_iterations", type=int, default=25,
                   help="Fixed reference iterations when --ref_convergence_tol=none. "
                        "Ignored by the default convergence-based reference and "
                        "with --null_control.")
    p.add_argument("--ref_convergence_tol", type=optional_positive_float,
                   default=1e-3, metavar="FLOAT|none",
                   help="Squared-L2 first-action convergence tolerance for the "
                        "reference (default 1e-3). Use 'none' for explicit "
                        "fixed-iteration compatibility mode.")
    p.add_argument("--ref_max_iterations", type=int, default=25,
                   help="Maximum reference iterations under convergence "
                        "termination (default 25).")
    p.add_argument("--null_control", action=argparse.BooleanOptionalAction, default=False,
                   help="Rebuild the reference planner with the DEGRADED settings "
                        "(different seed only) in an independent diagnostic run.")

    # --- KL estimator knobs -------------------------------------------------
    p.add_argument("--kl_every", type=int, default=20,
                   help="Run the (expensive) reference planner every N control "
                        "steps. The dominant cost knob: the reference planner is "
                        "a shadow and does not need to run every step.")
    p.add_argument("--kl_shrinkage", type=float, default=1e-3,
                   help="Shrink each covariance toward sigma^2*I by this factor "
                        "before inversion, keeping the log-det finite when the "
                        "effective sample size collapses.")
    p.add_argument("--record_kl_moments", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Retain both planners' unregularized first-action means "
                        "and covariances at valid KL steps for CPU sensitivity "
                        "analysis. No extra plans; adds storage, default off.")
    p.add_argument("--kl_direction", type=str, default="forward",
                   choices=["forward", "reverse"],
                   help="Which direction is the headline number. forward = "
                        "KL(reference||degraded) (information lost by the "
                        "approximation); reverse swaps the arguments. BOTH "
                        "are always computed and stored — this only picks the "
                        "one printed and marked as headline.")
    p.add_argument("--reference_init", choices=REFERENCE_INIT_CHOICES, default="zero",
                   help="Reference proposal at each measured state. 'zero' "
                        "(default) restarts from N(0,sigma); 'degraded_pre_solve' "
                        "reproduces the older same-U0 protocol; 'persistent' "
                        "retains the reference mean across measured states.")
    # Backward-compatible aliases for commands created before reference_init.
    # New commands should use --reference_init explicitly.
    p.add_argument("--sync_reference_mean", action=argparse.BooleanOptionalAction,
                   default=None, help=argparse.SUPPRESS)
    p.add_argument("--execute", type=str, default="mean", choices=["mean", "sample"],
                   help="What the degraded planner executes. 'mean' matches every "
                        "other sweep in the study. 'sample' draws a particle from "
                        "its weighted particle distribution. This changes the "
                        "execution policy and is not pooled with mean execution.")

    # --- shared MPPI / eval knobs (must match across the two planners) ------
    p.add_argument("--time_horizon", type=float, default=0.352)
    p.add_argument("--step_time",    type=float, default=0.064)
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--ref_temperature", type=float, default=50.0,
                   help="Reference MPPI temperature (default 50), independent "
                        "of the acting planner's --temperature. Null diagnostics "
                        "use the acting temperature to preserve equal settings.")
    p.add_argument("--noise_sigma",  type=float, default=0.025)
    p.add_argument("--delta",        type=float, default=None,
                   help="Per-step MPPI delta clip magnitude (action units); "
                        "default disables the clamp, matching run_eval_episode.py.")
    p.add_argument("--max_steps",    type=int,   default=None,
                   help="Override the task's max_steps (cost control).")
    p.add_argument("--eval_substeps", type=int,  default=None)
    p.add_argument("--eval_sim",     type=str,   default="none",
                   choices=["none", "mujoco", "drake", "pinocchio"])
    p.add_argument("--settle",       type=float, default=1.0)
    p.add_argument("--geometry", type=str, default="cube_high_high",
                   help="For grasp_reorient: '<object>' or '<object>_high_high' "
                        "(cube, duck, ball, spam, tomato). Object-only names "
                        "select high_high; lower fidelity and legacy aliases "
                        "are rejected. The per-object eval scene stays fixed.")
    p.add_argument("--nconmax",      type=int,   default=200)
    p.add_argument("--njmax",        type=int,   default=500)
    p.add_argument("--use_full_graph",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed",  type=int, default=None)
    add_record_flags(p)
    p.add_argument("--debug", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    # Reject misleading geometry before initializing GPU resources.
    args.geometry = resolve_kl_geometry(args.task, args.geometry)
    validate_kl_args(args)
    wp.init()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out, _ = run_cell(args)
    output_path = outdir / result_filename(out)
    # One self-describing file per run: a shared meta.json would be overwritten
    # by parallel array jobs with different objects or settings. Keep config
    # floats at full precision for scientific grouping; trajectory arrays still
    # use the recorder's own compact precision.
    json_io.dump(out, output_path, precision=0)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
