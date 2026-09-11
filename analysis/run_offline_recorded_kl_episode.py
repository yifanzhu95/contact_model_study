"""Compute per-state-zero reference KL for one episode from a recorded cell.

The acting moments and task outcome are read unchanged from the log. At each
selected recorded planner state, the higher-compute MPPI reference mean is
reset to zero, a fresh Gaussian perturbation block is drawn, and that block is
held fixed while the mean iterates to the stopping threshold or iteration cap.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np
import warp as wp

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ANALYSIS))

import contact_study.tasks  # noqa: F401 - registers task implementations
from contact_study.drivers.run_eval_episode import (
    MODEL_FACTORIES,
    apply_cost_weight_overrides,
    apply_goal_difficulty,
)
from contact_study.evaluation.distributions import gaussian_kl, weighted_moments
from contact_study.planners.mppi import MPPIConfig, MPPIController
from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole
from offline_recorded_kl_common import (
    DEFAULT_CONVERGENCE_TOL,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_MEASUREMENTS,
    DEFAULT_REFERENCE_SAMPLES,
    DEFAULT_REFERENCE_TEMPERATURE,
    DEFAULT_SHRINKAGE,
    SCHEMA,
    array_digest,
    atomic_json,
    evenly_spaced_unique_steps,
    independent_gaussian_kl,
    output_filename,
    resolved_grasp_reorient_rollout_scene,
    scientific_runtime_provenance,
    sha256_bytes,
)


class TracedReference(MPPIController):
    """Record consecutive first-action updates without changing optimization."""

    def _begin_plan(self):
        self.observed_updates = []
        super()._begin_plan()

    def _update_params(self, n_eff):
        valid = super()._update_params(n_eff)
        if valid:
            self.observed_updates.append(self.U_wp.numpy()[0].copy())
        return valid


def reconstruct_task(source, context, states, episode_child):
    if source["task"] != "grasp_reorient":
        raise ValueError("Recorded-goal reconstruction currently supports grasp_reorient only")
    task = get_task(source["task"], geometry=source["geometry"], role=TaskRole.ROLLOUT)
    apply_goal_difficulty(task, context["goal_difficulty"])
    model, data = task.load()
    apply_cost_weight_overrides(task, source["full_weights"])
    model.opt.timestep = context["rollout_dt"]
    rng = np.random.default_rng(episode_child)
    q0, v0, u0 = task.get_inital_state(rng)
    for name, reconstructed, logged in (
        ("q0", q0, context["q0"]),
        ("v0", v0, context["v0"]),
        ("u0", u0, context["u0"]),
    ):
        np.testing.assert_allclose(reconstructed, logged, rtol=1e-6, atol=1e-7,
                                   err_msg=f"Initial-state reconstruction failed: {name}")
    data.qpos[:] = states["qpos"][0]
    data.qvel[:] = states["qvel"][0]
    data.ctrl[:] = context["u0"]
    mujoco.mj_forward(model, data)
    return task, model, data, rng


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--max-measurements", type=int, default=DEFAULT_MAX_MEASUREMENTS)
    parser.add_argument("--ref-samples", type=int, default=DEFAULT_REFERENCE_SAMPLES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_REFERENCE_TEMPERATURE)
    parser.add_argument("--convergence-tol", type=float, default=DEFAULT_CONVERGENCE_TOL)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--shrinkage", type=float, default=DEFAULT_SHRINKAGE)
    return parser


def validate_args(args, parser):
    if not args.source_json.is_file():
        parser.error(f"source-json is not a file: {args.source_json}")
    if args.episode < 0:
        parser.error("episode must be non-negative")
    if args.max_measurements < 1 or args.ref_samples < 1:
        parser.error("max-measurements and ref-samples must be positive")
    if not np.isfinite(args.temperature) or not np.isfinite(args.convergence_tol):
        parser.error("temperature and convergence-tol must be finite")
    if args.temperature <= 0 or args.convergence_tol <= 0:
        parser.error("temperature and convergence-tol must be positive")
    if args.max_iterations < 2:
        parser.error("max-iterations must be at least two")
    if not 0 < args.shrinkage <= 1:
        parser.error("shrinkage must lie in (0, 1]")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    payload = args.source_json.read_bytes()
    source_sha256 = sha256_bytes(payload)
    if args.expected_source_sha256 and source_sha256 != args.expected_source_sha256:
        raise ValueError("Source SHA-256 differs from the input manifest")
    source = json.loads(payload)
    if len(source["episodes"]) != source["n_episodes"]:
        raise ValueError("Recorded episode list does not match n_episodes")
    if source["driver"] != "sync":
        raise ValueError("Offline state reconstruction currently supports synchronous logs only")
    if not 0 <= args.episode < source["n_episodes"]:
        raise ValueError("Episode index is outside the recorded cell")
    episode = source["episodes"][args.episode]
    trajectory = episode["trajectory"]
    context = trajectory["context"]
    states = trajectory["steps"]
    acting = trajectory["planner_dist"]
    if not states["step"] or int(states["step"][0]) != 0:
        raise ValueError("Recorded physical-state sequence must begin at step zero")
    if context["goal_difficulty"] != 1 or context["goal_switch_steps"]:
        raise ValueError("Goal reconstruction requires one fixed difficulty-1 goal")
    if context["warm_start"] or context["time_constrained"] or context["shrinkage"] != 0:
        raise ValueError("Unexpected source planning protocol")
    if context["action_source"] != "mean" or context["planner"] != "mppi":
        raise ValueError("Offline Gaussian KL requires recorded MPPI mean-action distributions")
    if context["resample_per_iteration"]:
        raise ValueError("Convergence requires fixed perturbations within a solve")

    selected_steps = evenly_spaced_unique_steps(acting["step"], args.max_measurements)
    acting_lookup = {int(step): index for index, step in enumerate(acting["step"])}
    state_lookup = {int(step): index for index, step in enumerate(states["step"])}
    if len(acting_lookup) != len(acting["step"]) or len(state_lookup) != len(states["step"]):
        raise ValueError("Duplicate recorded step number")
    if any(step not in state_lookup for step in selected_steps):
        raise ValueError("Selected planner row has no matching physical state")

    args.outdir.mkdir(parents=True, exist_ok=True)
    output = args.outdir / output_filename(args.source_json, args.episode)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite episode output: {output}")

    episode_child = np.random.SeedSequence(
        [source["seed"], source["combo_index"]]
    ).spawn(source["n_episodes"])[args.episode]
    reference_seed = int(
        np.random.SeedSequence(
            [source["seed"], source["combo_index"], args.episode, 73191]
        ).generate_state(1)[0]
    )
    wp.init()
    task, model, mjd, goal_rng = reconstruct_task(source, context, states, episode_child)
    expected_scene = resolved_grasp_reorient_rollout_scene(REPO, source["geometry"])
    if task.resolve_scene_path().resolve() != expected_scene:
        raise ValueError("Task scene resolution differs from the provenance resolver")
    scientific_provenance = scientific_runtime_provenance(
        REPO, source["geometry"], source["model"]
    )
    reference_config = MPPIConfig(
        **{
            **source["planner_kwargs"],
            "seed": reference_seed,
            "debug": False,
            "n_samples": args.ref_samples,
            "temperature": args.temperature,
            "convergence_tol": args.convergence_tol,
            "max_iterations": args.max_iterations,
            "warm_start": False,
            "ctrl_relative_to_qpos": context["ctrl_relative_to_qpos"],
            "resample_interval": 1,
            "resample_per_iteration": False,
            "time_constrained": False,
        }
    )
    reference = TracedReference(task, MODEL_FACTORIES[source["model"]](), reference_config)
    task.sample_new_goal(mjd, goal_rng)
    if (reference.horizon, reference.substeps, reference.nu) != (
        context["horizon"], context["substeps"], context["nu"]
    ):
        raise ValueError("Reference and recorded acting schedules differ")
    goal = reference.goal_wp.numpy().copy()

    record = {
        "schema": SCHEMA,
        "source": args.source_json.name,
        "source_path": str(args.source_json.resolve()),
        "source_sha256": source_sha256,
        "episode": args.episode,
        "success": bool(episode["success"]),
        "end_reason": episode["end_reason"],
        "model": source["model"],
        "geometry": source["geometry"],
        "acting_config": source["planner_kwargs"],
        "protocol": "independent zero-mean reference restart at every selected recorded state",
        "reference_initialization": "N(0, noise_sigma^2 I) at every selected state",
        "reference_cross_state_history": "none",
        "reference_noise_policy": "fresh block per state; fixed block across iterations within that solve",
        "selection": {
            "method": "up to max_measurements unique rows evenly spaced over recorded planner states; endpoints included",
            "max_measurements": args.max_measurements,
            "available_recorded_planner_states": len(acting["step"]),
            "selected_count": len(selected_steps),
            "selection_uses_outcome_or_kl": False,
        },
        "selected_steps": selected_steps,
        "reference_config": asdict(reference_config),
        "reference_seed": reference_seed,
        "goal": goal,
        "goal_source": "reconstructed from the recorded row/episode seed flow; not explicitly stored",
        "shrinkage": args.shrinkage,
        "direction": "KL(reference || recorded acting)",
        "convergence_definition": "squared L2 first-action mean change between iterations < convergence_tol",
        "source_outcome_unchanged": True,
        "closed_loop_rerun": False,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_support_sha256": hashlib.sha256(
            (ANALYSIS / "offline_recorded_kl_common.py").read_bytes()
        ).hexdigest(),
        "analysis_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "analysis_worktree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO, text=True,
        ).strip()),
        "scientific_runtime_provenance": scientific_provenance,
        "measurements": [],
        "completed": False,
    }

    started = time.perf_counter()
    previous_final_mean = np.zeros_like(reference.U_wp.numpy())
    for measurement_index, step in enumerate(selected_steps):
        dist_index = acting_lookup[step]
        state_index = state_lookup[step]
        if step and step - 1 not in state_lookup:
            raise ValueError(f"Previous control is absent for step {step}")

        mean_before_reset = reference.U_wp.numpy().copy()
        if measurement_index:
            np.testing.assert_array_equal(mean_before_reset, previous_final_mean)
        reference.reset()
        mean_after_reset = reference.U_wp.numpy().copy()
        np.testing.assert_array_equal(mean_after_reset, np.zeros_like(mean_after_reset))
        if reference.last_action_seq is not None or reference._plan_count != 0:
            raise AssertionError("Reference reset retained plan-local history")

        mjd.qpos[:] = states["qpos"][state_index]
        mjd.qvel[:] = states["qvel"][state_index]
        mjd.ctrl[:] = context["u0"] if step == 0 else states["ctrl"][state_lookup[step - 1]]
        mujoco.mj_forward(model, mjd)
        np.testing.assert_array_equal(reference.goal_wp.numpy(), goal)
        inputs = {name: getattr(mjd, name).copy() for name in ("qpos", "qvel", "ctrl")}
        inputs["goal"] = goal.copy()
        resample_before = reference._resample_count
        noise_key = int(reference._noise_seed + resample_before)

        plan_started = time.perf_counter()
        reference.plan(mjd)
        plan_ms = 1000 * (time.perf_counter() - plan_started)
        mean_after_solve = reference.U_wp.numpy().copy()
        previous_final_mean = mean_after_solve
        for name, before in inputs.items():
            after = reference.goal_wp.numpy() if name == "goal" else getattr(mjd, name)
            np.testing.assert_array_equal(after, before)
        if reference._plan_count != 1 or reference._resample_count != resample_before + 1:
            raise AssertionError("Each state must run one fresh-noise plan call")

        updates = reference.observed_updates
        if len(updates) != reference.last_n_iterations:
            raise AssertionError("Traced update count differs from planner iteration count")
        residuals = [float((new - old) @ (new - old)) for old, new in zip(updates[:-1], updates[1:])]
        converged = bool(residuals and residuals[-1] < args.convergence_tol)
        if reference.last_plan_ok and reference.last_n_iterations < args.max_iterations and not converged:
            raise AssertionError("Reference stopped early without satisfying convergence")

        acting_mean = np.asarray(acting["mu"][dist_index], dtype=np.float64)
        acting_cov_raw = np.asarray(acting["cov"][dist_index], dtype=np.float64)
        entry = {
            "measurement_index": measurement_index,
            "step": step,
            "acting_source_row": dist_index,
            "input_hashes": {name: array_digest(value) for name, value in inputs.items()},
            "input_preservation_passed": True,
            "reference_mean_before_reset": mean_before_reset,
            "reference_mean_after_reset": mean_after_reset,
            "reference_zero_reset_max_abs": float(np.max(np.abs(mean_after_reset))),
            "reference_zero_reset_passed": True,
            "reference_previous_mean_discarded": bool(measurement_index),
            "reference_mean_after_solve": mean_after_solve,
            "reference_noise_key": noise_key,
            "reference_resample_count_before": resample_before,
            "reference_resample_count_after": reference._resample_count,
            "reference_fresh_noise_draw_passed": True,
            "reference_plan_ms": plan_ms,
            "reference_iterations": reference.last_n_iterations,
            "reference_update_squared_l2": residuals,
            "reference_converged": converged,
            "reference_reached_iteration_cap": bool(
                reference.last_n_iterations == args.max_iterations and not converged
            ),
            "acting_moments_source": "copied unchanged from recorded planner_dist row",
            "acting_mean": acting_mean,
            "acting_covariance_raw": acting_cov_raw,
            "acting_ess": acting["ess"][dist_index],
            "acting_mean_sha256": array_digest(acting_mean),
            "acting_covariance_sha256": array_digest(acting_cov_raw),
        }
        if acting["degenerate"][dist_index]:
            entry.update(status="invalid_recorded_acting_distribution",
                         reason="Recorded acting distribution is degenerate")
        elif not reference.last_plan_ok:
            entry.update(status="invalid_reference_solve",
                         reason="Reference rollouts did not produce a current distribution")
        else:
            reference_mean, reference_cov_raw, reference_ess = weighted_moments(
                reference, 0, context["noise_sigma"]
            )
            target = context["noise_sigma"] ** 2 * np.eye(context["nu"])
            reference_cov = ((1 - args.shrinkage) * 0.5 *
                             (reference_cov_raw + reference_cov_raw.T) + args.shrinkage * target)
            acting_cov = ((1 - args.shrinkage) * 0.5 *
                          (acting_cov_raw + acting_cov_raw.T) + args.shrinkage * target)
            np.linalg.cholesky(reference_cov)
            np.linalg.cholesky(acting_cov)
            forward = gaussian_kl(reference_mean, reference_cov, acting_mean, acting_cov)
            reverse = gaussian_kl(acting_mean, acting_cov, reference_mean, reference_cov)
            audit_forward = independent_gaussian_kl(reference_mean, reference_cov, acting_mean, acting_cov)
            audit_reverse = independent_gaussian_kl(acting_mean, acting_cov, reference_mean, reference_cov)
            values = np.asarray([forward, reverse, audit_forward, audit_reverse])
            if not np.isfinite(values).all() or values.min() < -1e-8:
                raise ValueError(f"Invalid KL at step {step}: {values}")
            np.testing.assert_allclose(forward, audit_forward, rtol=1e-9, atol=1e-7)
            np.testing.assert_allclose(reverse, audit_reverse, rtol=1e-9, atol=1e-7)
            entry.update(
                status="valid_numerical_measurement",
                inclusion_all_valid=True,
                inclusion_converged_only=converged,
                kl_ref_to_acting=max(0.0, float(forward)),
                kl_acting_to_ref=max(0.0, float(reverse)),
                kl_independent_audit_ref_to_acting=max(0.0, float(audit_forward)),
                kl_independent_audit_acting_to_ref=max(0.0, float(audit_reverse)),
                kl_independent_audit_passed=True,
                reference_mean=reference_mean,
                reference_covariance_raw=reference_cov_raw,
                reference_ess=reference_ess,
                regularized_reference_min_eigenvalue=float(np.linalg.eigvalsh(reference_cov).min()),
                regularized_acting_min_eigenvalue=float(np.linalg.eigvalsh(acting_cov).min()),
            )
        record["measurements"].append(entry)
        record["wall_seconds"] = time.perf_counter() - started
        atomic_json(output, record)
        print(
            f"{args.source_json.name} ep={args.episode} step={step} status={entry['status']} "
            f"iterations={entry['reference_iterations']} converged={converged} ms={plan_ms:.0f}",
            flush=True,
        )

    valid = [row for row in record["measurements"] if row["status"] == "valid_numerical_measurement"]
    record["summary"] = {
        "selected_measurements": len(record["measurements"]),
        "valid_numerical_measurements": len(valid),
        "converged_valid_measurements": sum(row["reference_converged"] for row in valid),
        "nonconverged_valid_measurements": sum(not row["reference_converged"] for row in valid),
        "invalid_measurements": len(record["measurements"]) - len(valid),
    }
    record["completed"] = True
    record["wall_seconds"] = time.perf_counter() - started
    atomic_json(output, record)
    print("DONE", output, record["wall_seconds"], flush=True)
    del reference, task, model, mjd
    gc.collect()


if __name__ == "__main__":
    main()
