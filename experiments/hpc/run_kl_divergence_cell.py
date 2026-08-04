"""run_kl_divergence_cell.py

HPC worker for the planner-approximation-quality sweep — one degraded-planner
cell. Measures the KL divergence between the contact model's *induced optimal
control distribution* and a compute-degraded approximation of it, then pairs
that with the success rate the degraded planner achieves on the eval sim.

The idea
--------
MPPI's target is not its Gaussian proposal but the optimal control distribution

    q*(V)  proportional to  exp(-S(V)/lambda) * p(V),     p = N(U, sigma^2 I)

where S is the trajectory cost and lambda is the temperature. MPPI never
represents q* explicitly: it draws N samples from p and computes
self-normalized weights w_n ~ exp(-(S_n - beta)/lambda). That weighted particle
set {(V_n, w_n)} IS an importance-sampling approximation of q*, and both
planners carry one every control step.

Two particle sets cannot be compared with KL directly (disjoint support), so
each is moment-matched to a Gaussian on the FIRST action (the only one that is
ever executed):

    mu    = sum_n w_n V_n[0]                      (equals the planner's U[0])
    Sigma = sum_n w_n (V_n[0]-mu)(V_n[0]-mu)^T

and the analytic Gaussian KL is evaluated in closed form. Comparing the induced
distributions rather than the proposals is what makes this a real KL: the
proposals share sigma by construction, so a proposal-level KL would collapse to
||mu_ref - mu_deg||^2 / (2 sigma^2) and see only mean displacement. The
moment-matched version additionally sees *concentration* (a planner whose
weights are near-uniform has learned nothing from its rollouts and its Sigma
relaxes back to sigma^2 I) and *shape*.

Why the first action only: for the full horizon d = H*nu, but the weighted
covariance has rank at most the effective sample size, so a full-horizon
covariance is badly rank-deficient. On the first action d = nu, which is
well-conditioned at realistic ESS.

What runs each step
-------------------
Two MPPIControllers share one rollout task, state, horizon, step_time and
noise_sigma, and differ only in compute (n_samples, n_iterations):

  * the DEGRADED planner plans every control step and drives the eval sim;
  * the REFERENCE ("optimal") planner is a shadow that never controls. It runs
    only every --kl_every steps, because it is the expensive one.

Before each reference solve, the reference planner's mean is seeded from the
degraded planner's mean as it was BEFORE that step's solve, so both
distributions are conditioned on an identical (state, U0) and the KL isolates
optimizer quality. This also makes the reference planner immune to --kl_every:
its own U never goes stale, because it is overwritten every time it runs.
Pass --no-sync_reference_mean to let the two means evolve independently
instead, which measures accumulated policy divergence rather than
instantaneous approximation error.

Null control (--null_control)
-----------------------------
The degraded planner is degraded by having fewer samples, and fewer samples
also make its moment estimates noisier — so measured KL rises with degradation
partly for statistical rather than substantive reasons. With --null_control the
reference planner is rebuilt with the DEGRADED planner's own settings (differing
only in noise seed), so the cell measures the estimator-noise floor at that
sample count. Run every config both ways and compare; a config whose real KL is
not clearly above its own null floor is not reporting a meaningful difference.

Known approximation (not corrected here)
----------------------------------------
V = clamp(U + eps, delta_range), so the particles come from a TRUNCATED
Gaussian and moment-matching to an untruncated one is an approximation. It bites
harder than for a mean-only comparison, because clipped dimensions get
artificially small variance which feeds straight into the log-det term. Left
uncorrected deliberately; treat cells whose actions ride the delta clip with
suspicion.

Numerical safety
----------------
Sigma is shrunk toward the proposal covariance before inversion:

    Sigma <- (1-alpha) Sigma + alpha * sigma^2 I

with alpha = --kl_shrinkage. sigma^2 I is the natural prior (it is exactly what
Sigma relaxes to when the weights go uniform). Shrinkage is what keeps the
log-det finite when the effective sample size collapses toward 1, which does
happen on this task even when the mean ESS is healthy.

    python run_kl_divergence_cell.py \
        --outdir results/kl_divergence_eval_run \
        --task grasp_reorient --model M2 \
        --n_samples 64 --n_iterations 1 \
        --ref_n_samples 4096 --ref_n_iterations 4 \
        --n_episodes 5 --kl_every 20
"""

from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.contact_models.config import GeometryVariant
from contact_study.evaluation.metrics import (
    EpisodeResult, aggregate_episodes,
)
from contact_study.planners.mppi import MPPIConfig, MPPIController
from contact_study.tasks.base import get_task
from contact_study.tasks.config import EvalSimulatorKind, TaskRole

from contact_study.drivers.run_eval_episode import (
    load_rollout_task, resolve_mppi_schedule, MODEL_FACTORIES,
)


# ---------------------------------------------------------------------------
# KL machinery
# ---------------------------------------------------------------------------
def weighted_moments(controller, shrinkage: float, sigma: float):
    """First-action weighted mean and covariance of a planner's induced q*.

    Reads the state the controller already leaves on the GPU after plan():
    w_wp holds the NORMALIZED weights of the final MPPI iteration and V_wp the
    clamped samples those weights scored, so (V, w) is a consistent particle
    representation whose weighted mean equals the planner's returned U[0].

    Returns (mu, Sigma, ess). Sigma is already shrunk toward sigma^2 I.
    """
    w  = controller.w_wp.numpy().astype(np.float64)          # (N,) sums to 1
    V0 = controller.V_wp.numpy()[:, 0, :].astype(np.float64)  # (N, nu)

    # Guard against a degenerate weight vector (all-NaN rollouts zero U and can
    # leave w unnormalized); fall back to uniform so the cell keeps running.
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
# One episode with two planners
# ---------------------------------------------------------------------------
def run_kl_episode(args, contact_cfg, deg_cfg: MPPIConfig, ref_cfg: MPPIConfig,
                   rng, geometry, eval_sim, ep_idx: int):
    """Closed-loop episode driven by the degraded planner, shadowed by the
    reference planner. Returns (EpisodeResult, per-step KL record dict).

    Forked from contact_study.drivers.run_eval_episode — same eval/rollout
    split, same control parameterization and clipping — with a second
    controller and the KL bookkeeping added. Kept here rather than in
    drivers/ so this experimental path cannot destabilize the shared driver
    (run_eval_episode_record_controls.py is the existing precedent for a
    forked variant).
    """
    # ---- ROLLOUT task + both planners -------------------------------------
    rollout_task = get_task(args.task, geometry=geometry, role=TaskRole.ROLLOUT)
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
            f"Only n_samples / n_iterations may differ."
        )

    # ---- EVAL task + "real" simulator -------------------------------------
    eval_task = get_task(args.task, geometry=geometry, role=TaskRole.EVAL)
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

    kl_fwd, kl_rev, ess_ref, ess_deg, mu_dist, kl_steps = [], [], [], [], [], []
    step_times: list[float] = []
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
            break
        if rollout_task.has_failed(mjd):
            break

        measure = (t % args.kl_every == 0)

        # Snapshot the degraded planner's mean BEFORE its solve, so the
        # reference planner can be conditioned on the same starting point
        # rather than on the degraded planner's own output.
        U0 = deg.U_wp.numpy().copy() if (measure and args.sync_reference_mean) else None

        # --- degraded planner: this is the one that controls ---------------
        plan_start = time.perf_counter()
        action = deg.plan(mjd)
        step_times.append((time.perf_counter() - plan_start) * 1e3)

        if measure:
            mu_d, S_d, e_d = weighted_moments(deg, args.kl_shrinkage, sigma)

            # --- reference planner: shadow solve at the same (state, U0) ---
            if U0 is not None:
                ref.U_wp.assign(U0)
            ref.plan(mjd)
            mu_r, S_r, e_r = weighted_moments(ref, args.kl_shrinkage, sigma)

            # Forward = KL(reference || degraded): information lost by using the
            # degraded planner in place of the reference. Reverse swaps them and
            # punishes the degraded planner for being over-broad instead.
            f = gaussian_kl(mu_r, S_r, mu_d, S_d)
            r = gaussian_kl(mu_d, S_d, mu_r, S_r)
            if np.isfinite(f) and np.isfinite(r):
                kl_fwd.append(f)
                kl_rev.append(r)
                ess_ref.append(e_r)
                ess_deg.append(e_d)
                mu_dist.append(float(np.linalg.norm(mu_r - mu_d)))
                kl_steps.append(t)

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

        sim.apply_control(u)
        sim.step(eval_steps_per_control)

        if args.debug and measure and t % (args.kl_every * 10) == 0:
            print(f"  [ep {ep_idx:02d} | step {t:04d}]  "
                  f"KL_fwd={kl_fwd[-1] if kl_fwd else float('nan'):9.4f}  "
                  f"KL_rev={kl_rev[-1] if kl_rev else float('nan'):9.4f}  "
                  f"ESS ref={ess_ref[-1] if ess_ref else float('nan'):7.1f}/"
                  f"{ref_cfg.n_samples} deg={ess_deg[-1] if ess_deg else float('nan'):7.1f}/"
                  f"{deg_cfg.n_samples}")

    elapsed  = time.perf_counter() - ep_start
    step_arr = np.asarray(step_times)
    final_qpos = sim.get_state().qpos

    result = EpisodeResult(
        task_name        = cfg.name,
        model_label      = contact_cfg.label,
        condition        = "B",
        success          = steps_to_success is not None,
        steps_to_success = steps_to_success,
        final_cost       = float(np.linalg.norm(final_qpos - np.asarray(q0, dtype=float))),
        n_samples_used   = deg_cfg.n_samples,
        elapsed_seconds  = elapsed,
        mean_step_ms     = float(step_arr.mean()) if len(step_arr) else 0.0,
        std_step_ms      = float(step_arr.std())  if len(step_arr) else 0.0,
    )
    record = {
        "steps":     kl_steps,
        "kl_forward": kl_fwd,
        "kl_reverse": kl_rev,
        "ess_ref":   ess_ref,
        "ess_deg":   ess_deg,
        "mu_dist":   mu_dist,
    }
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
    geometry = GeometryVariant(args.geometry)
    eval_sim = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)
    contact_cfg = MODEL_FACTORIES[args.model]()

    # In null-control mode the "reference" is a second instance of the degraded
    # planner (different noise seed only), so the cell measures the
    # estimator-noise floor at this sample count rather than a real difference.
    ref_n_samples   = args.n_samples   if args.null_control else args.ref_n_samples
    ref_n_iterations = args.n_iterations if args.null_control else args.ref_n_iterations

    label = f"{args.model}_n{args.n_samples}_i{args.n_iterations}"
    if args.null_control:
        label += "_null"

    peek = load_rollout_task(args.task, geometry)
    horizon, substeps, rollout_dt = resolve_mppi_schedule(
        MPPIConfig(time_horizon=args.time_horizon, step_time=args.step_time),
        peek.config, args.eval_substeps,
    )
    n_steps = args.max_steps if args.max_steps is not None else peek.config.max_steps

    print(f"[{label}]  nq={peek.mjm.nq} nv={peek.mjm.nv} nu={peek.mjm.nu}  "
          f"max_steps={n_steps}  n_episodes={args.n_episodes}")
    print(f"[{label}]  degraded : n_samples={args.n_samples} n_iterations={args.n_iterations}")
    print(f"[{label}]  reference: n_samples={ref_n_samples} n_iterations={ref_n_iterations}"
          + ("   (NULL CONTROL: same settings, different seed)" if args.null_control else ""))
    print(f"[{label}]  schedule : horizon={horizon} substeps={substeps} "
          f"rollout_dt={rollout_dt*1e3:.3f}ms  (shared by both planners)")
    print(f"[{label}]  KL       : first action (d={peek.mjm.nu}), every {args.kl_every} steps, "
          f"shrinkage={args.kl_shrinkage:g}, headline={args.kl_direction}, "
          f"sync_reference_mean={args.sync_reference_mean}, execute={args.execute}")

    def make_cfg(n_samples, n_iterations, seed):
        return MPPIConfig(
            n_samples      = n_samples,
            n_iterations   = n_iterations,
            time_horizon   = args.time_horizon,
            step_time      = args.step_time,
            temperature    = args.temperature,
            noise_sigma    = args.noise_sigma,
            warm_start     = False,
            resample_interval = 1,
            use_full_graph = args.use_full_graph,
            delta_range    = (-args.delta, args.delta),
            nconmax        = args.nconmax,
            njmax          = args.njmax,
            seed           = seed,
            debug          = False,   # per-plan MPPI spam would swamp two planners
        )

    model_ord = list(MODEL_FACTORIES).index(args.model)
    seed_seq  = np.random.SeedSequence(
        [s for s in (args.seed, args.n_samples, args.n_iterations,
                     int(args.null_control), model_ord) if s is not None] or None
    )
    episode_seeds = seed_seq.spawn(args.n_episodes)

    episodes, records = [], []
    for ep in range(args.n_episodes):
        ep_seed = int(episode_seeds[ep].generate_state(1)[0])
        rng     = np.random.default_rng(episode_seeds[ep])

        deg_cfg = make_cfg(args.n_samples, args.n_iterations, ep_seed)
        # Offset keeps the two noise streams independent — essential in null
        # mode, where an identical seed would make the KL identically zero.
        ref_cfg = make_cfg(ref_n_samples, ref_n_iterations,
                           (ep_seed + 0x5EED) % (2**31 - 1))

        result, rec = run_kl_episode(
            args, contact_cfg, deg_cfg, ref_cfg, rng, geometry, eval_sim, ep,
        )
        episodes.append(result)
        records.append(rec)

        tick = "✓" if result.success else "✗"
        kf, kr = _stats(rec["kl_forward"]), _stats(rec["kl_reverse"])
        print(f"    ep {ep:02d}  {tick}  "
              f"KL_fwd={kf['mean'] if kf['mean'] is not None else float('nan'):9.4f}  "
              f"KL_rev={kr['mean'] if kr['mean'] is not None else float('nan'):9.4f}  "
              f"n_kl={kf['n']:4d}  step={result.mean_step_ms:.1f}ms")

    agg = aggregate_episodes(episodes, args.task, label, "B")

    # Pool every measured step across all episodes for the cell-level KL. The
    # scatter point uses this; per-episode and per-step values are kept below.
    all_fwd = [x for r in records for x in r["kl_forward"]]
    all_rev = [x for r in records for x in r["kl_reverse"]]
    all_er  = [x for r in records for x in r["ess_ref"]]
    all_ed  = [x for r in records for x in r["ess_deg"]]
    all_md  = [x for r in records for x in r["mu_dist"]]

    headline = all_fwd if args.kl_direction == "forward" else all_rev
    print(f"  → success={agg.success_rate*100:.1f}%  "
          f"KL_{args.kl_direction}={_stats(headline)['mean']:.4f} "
          f"(median {_stats(headline)['median']:.4f}, n={len(headline)})  "
          f"ESS ref={_stats(all_er)['mean']:.1f}/{ref_n_samples} "
          f"deg={_stats(all_ed)['mean']:.1f}/{args.n_samples}")

    out = {
        "label": label,
        "task":  args.task,
        "model": args.model,
        "config": {
            "n_samples":        args.n_samples,
            "n_iterations":     args.n_iterations,
            "ref_n_samples":    ref_n_samples,
            "ref_n_iterations": ref_n_iterations,
            "null_control":     args.null_control,
            "kl_every":         args.kl_every,
            "kl_shrinkage":     args.kl_shrinkage,
            "kl_direction":     args.kl_direction,
            "sync_reference_mean": args.sync_reference_mean,
            "execute":          args.execute,
            "time_horizon":     args.time_horizon,
            "step_time":        args.step_time,
            "temperature":      args.temperature,
            "noise_sigma":      args.noise_sigma,
            "delta":            args.delta,
            "horizon":          horizon,
            "substeps":         substeps,
            "rollout_dt":       rollout_dt,
            "max_steps":        n_steps,
            "n_episodes":       args.n_episodes,
            "seed":             args.seed,
        },
        # AggregatedResult carries the success-rate side, in the same schema the
        # rest of the study's sweeps use.
        "aggregate": agg.to_dict(),
        "kl": {
            "headline_direction": args.kl_direction,
            "forward": _stats(all_fwd),
            "reverse": _stats(all_rev),
        },
        "diagnostics": {
            "ess_ref": _stats(all_er),
            "ess_deg": _stats(all_ed),
            "mu_dist": _stats(all_md),
        },
        "episodes": [
            {
                "success":          e.success,
                "steps_to_success": e.steps_to_success,
                "mean_step_ms":     e.mean_step_ms,
                "kl_forward":       _stats(r["kl_forward"]),
                "kl_reverse":       _stats(r["kl_reverse"]),
            }
            for e, r in zip(episodes, records)
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
    p.add_argument("--model", type=str, default="M2", choices=list(MODEL_FACTORIES),
                   help="Contact model. Both planners use the SAME one — this "
                        "experiment varies optimizer quality, not model fidelity.")
    p.add_argument("--outdir", type=str, default="results/kl_divergence_eval_run")
    p.add_argument("--n_episodes", type=int, default=10)

    # --- the swept axis: degraded-planner compute --------------------------
    p.add_argument("--n_samples",    type=int, default=64,
                   help="Degraded planner sample count (swept).")
    p.add_argument("--n_iterations", type=int, default=1,
                   help="Degraded planner MPPI iterations per plan (swept).")
    # --- the reference ("optimal") planner ---------------------------------
    p.add_argument("--ref_n_samples",    type=int, default=4096,
                   help="Reference planner sample count. Ignored with --null_control.")
    p.add_argument("--ref_n_iterations", type=int, default=4,
                   help="Reference planner iterations. Ignored with --null_control.")
    p.add_argument("--null_control", action=argparse.BooleanOptionalAction, default=False,
                   help="Rebuild the reference planner with the DEGRADED settings "
                        "(different seed only) to measure this cell's "
                        "estimator-noise floor.")

    # --- KL estimator knobs -------------------------------------------------
    p.add_argument("--kl_every", type=int, default=20,
                   help="Run the (expensive) reference planner every N control "
                        "steps. The dominant cost knob: the reference planner is "
                        "a shadow and does not need to run every step.")
    p.add_argument("--kl_shrinkage", type=float, default=1e-3,
                   help="Shrink each covariance toward sigma^2*I by this factor "
                        "before inversion, keeping the log-det finite when the "
                        "effective sample size collapses.")
    p.add_argument("--kl_direction", type=str, default="forward",
                   choices=["forward", "reverse"],
                   help="Which direction is the headline number. forward = "
                        "KL(reference||degraded) (information lost by the "
                        "approximation); reverse punishes over-broadness. BOTH "
                        "are always computed and stored — this only picks the "
                        "one printed and marked as headline.")
    p.add_argument("--sync_reference_mean",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Seed the reference planner's mean from the degraded "
                        "planner's pre-solve mean, so both are conditioned on the "
                        "same (state, U0) and the KL is instantaneous "
                        "approximation error. Disable to let the means evolve "
                        "independently (accumulated policy divergence).")
    p.add_argument("--execute", type=str, default="mean", choices=["mean", "sample"],
                   help="What the degraded planner executes. 'mean' matches every "
                        "other sweep in the study. 'sample' draws a particle from "
                        "its induced distribution, which is closer to the "
                        "'execute the distribution' framing but injects a full "
                        "sigma of extra noise into every action and will depress "
                        "the success-rate axis for reasons unrelated to KL.")

    # --- shared MPPI / eval knobs (must match across the two planners) ------
    p.add_argument("--time_horizon", type=float, default=0.256)
    p.add_argument("--step_time",    type=float, default=0.032)
    p.add_argument("--temperature",  type=float, default=0.01)
    p.add_argument("--noise_sigma",  type=float, default=0.02)
    p.add_argument("--delta",        type=float, default=0.1)
    p.add_argument("--max_steps",    type=int,   default=None,
                   help="Override the task's max_steps (cost control).")
    p.add_argument("--eval_substeps", type=int,  default=None)
    p.add_argument("--eval_sim",     type=str,   default="none",
                   choices=["none", "mujoco", "drake", "pinocchio"])
    p.add_argument("--settle",       type=float, default=1.0)
    p.add_argument("--geometry",     type=str,   default="accurate",
                   choices=[g.value for g in GeometryVariant])
    p.add_argument("--nconmax",      type=int,   default=200)
    p.add_argument("--njmax",        type=int,   default=500)
    p.add_argument("--use_full_graph",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed",  type=int, default=None)
    p.add_argument("--debug", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    wp.init()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out, label = run_cell(args)
    with open(outdir / f"{label}.json", "w") as f:
        json.dump(out, f, indent=2)

    meta_path = outdir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "task":             args.task,
            "model":            args.model,
            "ref_n_samples":    args.ref_n_samples,
            "ref_n_iterations": args.ref_n_iterations,
            "kl_every":         args.kl_every,
            "kl_shrinkage":     args.kl_shrinkage,
            "kl_direction":     args.kl_direction,
            "noise_sigma":      args.noise_sigma,
            "step_time":        args.step_time,
            "time_horizon":     args.time_horizon,
        }, f, indent=2)


if __name__ == "__main__":
    main()
