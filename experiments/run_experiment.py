"""run_experiment.py

Master experiment runner with the full simulation + planning. Executes the full study grid:

  tasks × contact_models × conditions × n_episodes

and writes results to results/experiment_{timestamp}.json.

Usage:
    python experiments/run_experiment.py \
        --tasks push grasp_reorient peg_in_hole \
        --models M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 \
        --conditions A B \
        --n_episodes 20 \
        --budget_seconds 0.1 \
        --n_samples_b 1024
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import mujoco
import numpy as np

# Ensure tasks are registered
import contact_study.tasks.tasks  # noqa: F401

from contact_study.contact_models.config import ContactModelConfig, GeometryVariant
from contact_study.contact_models.benchmarks import (
    measure_rollout_speed,
    measure_approximation_error,
)
from contact_study.evaluation.metrics import (
    EpisodeResult,
    AggregatedResult,
    aggregate_episodes,
    save_results,
)
from contact_study.planners.mppi import MPPIController, MPPIConfig
from contact_study.tasks.base import get_task
from contact_study.utils.rollout import fixed_budget_rollout, fixed_sample_rollout

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Config table: name -> factory
# ---------------------------------------------------------------------------

MODEL_FACTORIES = {
    "M1":  ContactModelConfig.M1,
    "M2":  ContactModelConfig.M2,
    "M3":  ContactModelConfig.M3,
    "M4":  ContactModelConfig.M4,
    "M4d": lambda: ContactModelConfig.M4(damping_friction=True),
    "M5":  lambda: ContactModelConfig.M5(GeometryVariant.CONVEX_HULL),
    "M5p": lambda: ContactModelConfig.M5(GeometryVariant.PRIMITIVE_UNION),
    "M6":  lambda: ContactModelConfig.M6(GeometryVariant.CONVEX_HULL),
    "M7":  ContactModelConfig.M7,
    "M8":  ContactModelConfig.M8,
    "M9":  ContactModelConfig.M9,
    "M10": ContactModelConfig.M10,
}


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_one_episode(
    mjm:            mujoco.MjModel,
    cfg:            ContactModelConfig,
    task,
    condition:      str,
    budget_seconds: float,
    n_samples_b:    int,
    horizon:        int,
    rng:            np.random.Generator,
) -> EpisodeResult:
    """Run one closed-loop episode under Condition A or B."""
    mppi_cfg = MPPIConfig(
        n_samples  = n_samples_b,
        horizon    = horizon,
        warm_start = True,
    )
    controller = MPPIController(
        mjm      = mjm,
        cfg      = cfg,
        mppi_cfg = mppi_cfg,
        cost_fn  = task.cost_fn,
        rng      = rng,
    )

    mjd = mujoco.MjData(mjm)
    q0, v0 = task.sample_initial_state(rng)
    mjd.qpos[:] = q0
    mjd.qvel[:] = v0
    mujoco.mj_forward(mjm, mjd)

    steps_to_success = None
    episode_start    = time.perf_counter()

    for t in range(task.spec.max_steps):
        if condition == "A":
            # Condition A: MPPI with dynamic sample count inside fixed budget
            result = fixed_budget_rollout(
                mjm            = mjm,
                cfg            = cfg,
                budget_seconds = budget_seconds,
                horizon        = horizon,
                cost_fn        = task.cost_fn,
                initial_qpos   = mjd.qpos,
                initial_qvel   = mjd.qvel,
                rng            = rng,
            )
            # Pick best action from the rollout
            best_idx = int(np.argmin(result["costs"]))
            ctrl     = result["final_qpos"][best_idx][:mjm.nu]  # placeholder
            n_used   = result["n_samples"]
        else:
            # Condition B: standard MPPI with fixed N
            ctrl   = controller.plan(mjd)
            n_used = n_samples_b

        mjd.ctrl[:] = ctrl
        mujoco.mj_step(mjm, mjd)

        if task.is_success(mjd) and steps_to_success is None:
            steps_to_success = t + 1

    elapsed = time.perf_counter() - episode_start

    return EpisodeResult(
        task_name        = task.spec.name,
        model_label      = cfg.label,
        condition        = condition,
        success          = steps_to_success is not None,
        steps_to_success = steps_to_success,
        final_cost       = float(np.linalg.norm(mjd.qpos - q0)),
        n_samples_used   = n_used,
        elapsed_seconds  = elapsed,
    )


# ---------------------------------------------------------------------------
# Full study
# ---------------------------------------------------------------------------

def run_study(
    task_names:     list[str],
    model_names:    list[str],
    conditions:     list[str],
    n_episodes:     int,
    budget_seconds: float,
    n_samples_b:    int,
    horizon:        int,
    seed:           int,
    baseline_model: str = "M2",
) -> list[AggregatedResult]:

    rng    = np.random.default_rng(seed)
    aggregated: list[AggregatedResult] = []
    all_cfgs = {name: MODEL_FACTORIES[name]() for name in model_names}

    # Pre-compute speed and accuracy metrics against baseline
    print("=== Pre-computing speed / accuracy metrics ===")
    speed_cache: dict[str, float] = {}
    error_cache: dict[str, float] = {}

    for task_name in task_names:
        task = get_task(task_name)
        mjm, _ = task.load()

        baseline_cfg = all_cfgs[baseline_model]
        baseline_r   = measure_rollout_speed(mjm, baseline_cfg)
        baseline_time = baseline_r.mean_ms

        test_states = np.stack([
            np.concatenate(task.sample_initial_state(rng))
            for _ in range(20)
        ])

        for name, cfg in all_cfgs.items():
            key = f"{task_name}/{name}"

            speed_r = measure_rollout_speed(mjm, cfg)
            speed_cache[key] = baseline_time / speed_r.mean_ms

            mean_err, _ = measure_approximation_error(
                mjm, baseline_cfg, cfg, test_states, horizon=horizon
            )
            error_cache[key] = mean_err
            print(f"  {key}: speedup={speed_cache[key]:.2f}x  err={error_cache[key]:.4f}")

    # Main experiment loop
    print("\n=== Running episodes ===")
    for task_name in task_names:
        task = get_task(task_name)
        mjm, _ = task.load()

        for model_name in model_names:
            cfg = all_cfgs[model_name]
            key = f"{task_name}/{model_name}"

            for condition in conditions:
                print(f"  {task_name} | {model_name} | Condition {condition} | {n_episodes} eps")
                episodes = []
                for ep in range(n_episodes):
                    result = run_one_episode(
                        mjm            = mjm,
                        cfg            = cfg,
                        task           = task,
                        condition      = condition,
                        budget_seconds = budget_seconds,
                        n_samples_b    = n_samples_b,
                        horizon        = horizon,
                        rng            = rng,
                    )
                    episodes.append(result)

                agg = aggregate_episodes(episodes, task_name, cfg.label, condition)
                agg.speedup_vs_baseline    = speed_cache.get(key, 1.0)
                agg.approx_err_vs_baseline = error_cache.get(key, 0.0)
                aggregated.append(agg)

    return aggregated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",    nargs="+", default=["push", "grasp_reorient", "peg_in_hole"])
    parser.add_argument("--models",   nargs="+", default=["M1","M2","M3","M4","M5","M7","M9"])
    parser.add_argument("--conditions", nargs="+", default=["A","B"])
    parser.add_argument("--n_episodes",     type=int,   default=20)
    parser.add_argument("--budget_seconds", type=float, default=0.1)
    parser.add_argument("--n_samples_b",    type=int,   default=1024)
    parser.add_argument("--horizon",        type=int,   default=30)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--output",         type=str,   default=None)
    args = parser.parse_args()

    results = run_study(
        task_names     = args.tasks,
        model_names    = args.models,
        conditions     = args.conditions,
        n_episodes     = args.n_episodes,
        budget_seconds = args.budget_seconds,
        n_samples_b    = args.n_samples_b,
        horizon        = args.horizon,
        seed           = args.seed,
    )

    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or str(RESULTS_DIR / f"experiment_{ts}.json")
    save_results(results, out)


if __name__ == "__main__":
    main()
