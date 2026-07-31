"""run_experiment.py

Master experiment runner. Executes the full study grid:

  tasks × contact_models × conditions × n_episodes

for ONE choice of (geometry variant, physics noise level). To sweep
over geometry or noise, invoke this script multiple times with different
--geometry / --friction_sigma / --mass_sigma flags, or wrap it in an
outer loop. Geometry and physics noise are deliberately NOT part of
ContactModelConfig — they live on orthogonal axes and are applied here
at load time.

NOTE: Geometry variants (--geometry) are accepted and recorded in output
labels but are not yet implemented — the simulation always uses the
'accurate' variant regardless of the flag value.

Usage:
    # Clean baseline (condition B, warm-started MPPI)
    python experiments/run_experiment.py \
        --tasks push grasp_reorient peg_in_hole \
        --models M1 M2 M3 M4 \
        --conditions B \
        --n_episodes 20

    # Both conditions, all models
    python experiments/run_experiment.py \
        --conditions A B \
        --n_samples 1024 --horizon 50

    # Physics noise ablation
    python experiments/run_experiment.py \
        --models M4 \
        --friction_sigma 0.2 --mass_sigma 0.1
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

# Ensure tasks are registered
import contact_study.tasks  # noqa: F401

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
from contact_study.utils.physics_noise import (
    PhysicsNoiseParams,
    apply_physics_noise,
)
from contact_study.utils.rollout import fixed_budget_rollout, fixed_sample_rollout

RESULTS_DIR = "results"#Path(__file__).parent.parent / "results"

wp.init()


# ---------------------------------------------------------------------------
# Contact model factory table — M1..M4
# ---------------------------------------------------------------------------

MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
}

# ---------------------------------------------------------------------------
# Helper: load a task model with optional physics noise applied
# ---------------------------------------------------------------------------

def load_mjm_for_study(
    task_name: str,
    geometry:  GeometryVariant,
    noise:     PhysicsNoiseParams,
    rng:       np.random.Generator,
) -> tuple[mujoco.MjModel, object]:
    task = get_task(task_name, geometry=geometry)
    mjm, _ = task.load()
    mjm = apply_physics_noise(mjm, noise, rng)
    task._mjm = mjm
    return mjm, task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_settled_keyframe(mjd: mujoco.MjData) -> None:
    """Print qpos/qvel/ctrl after settling in MuJoCo XML keyframe format."""
    def _rows(arr: np.ndarray, per_line: int = 4) -> str:
        lines = []
        for i in range(0, len(arr), per_line):
            lines.append("      " + "  ".join(f"{v:.8g}" for v in arr[i:i+per_line]))
        return "\n".join(lines)

    print('<key name="settled"')
    print(f'      qpos="\n{_rows(mjd.qpos)}"\n')
    print(f'      qvel="\n{_rows(mjd.qvel)}"\n')
    print(f'      ctrl="\n{_rows(mjd.ctrl)}"\n')
    print('      />')


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_one_episode(
    mjm:            mujoco.MjModel,
    cfg:            ContactModelConfig,
    task,
    condition:      str,
    budget_seconds: float,
    n_samples:      int,
    horizon:        int,
    seed:           int,
    rng:            np.random.Generator,
    settle_seconds: float = 1.0,
    use_full_graph: bool  = True,
    nconmax:        int   = 200,
    njmax:          int   = 500,
    debug:          bool  = True,
) -> EpisodeResult:
    """Run one closed-loop episode under Condition A or B."""
    task_cfg = task.config or task.spec

    mppi_cfg = MPPIConfig(
        n_samples      = n_samples,
        step_horizon   = horizon,
        temperature    = 0.01,
        noise_sigma    = 0.001,
        warm_start     = True,
        use_full_graph = use_full_graph,
        nconmax        = nconmax,
        njmax          = njmax,
        seed           = seed,
        debug          = debug,
    )

    controller = MPPIController(
        task       = task,
        cfg        = cfg,
        mppi_cfg   = mppi_cfg,
        rng        = rng,
    )

    mjd = mujoco.MjData(mjm)
    q0, v0, u0 = task.get_inital_state(rng)
    mjd.qpos[:] = q0
    mjd.qvel[:] = v0
    if u0 is not None:
        mjd.ctrl[:] = u0
    mujoco.mj_forward(mjm, mjd)

    settle_steps = int(settle_seconds / mjm.opt.timestep)
    for _ in range(settle_steps):
        mujoco.mj_step(mjm, mjd)

    if hasattr(task, "sample_new_goal"):
        task.sample_new_goal(mjd, rng)

    #_print_settled_keyframe(mjd)

    steps_to_success = None
    episode_start    = time.perf_counter()
    substeps         = controller.substeps
    n_used           = n_samples

    for t in range(task_cfg.max_steps):
        if condition == "A":
            result = fixed_budget_rollout(
                mjm            = mjm,
                cfg            = cfg,
                budget_seconds = budget_seconds,
                horizon        = horizon,
                initial_qpos   = mjd.qpos,
                initial_qvel   = mjd.qvel,
                rng            = rng,
            )
            best_idx = int(np.argmin(result["costs"]))
            ctrl     = result["final_qpos"][best_idx][:mjm.nu]
            n_used   = result["n_samples"]
        else:
            ctrl = controller.plan(mjd)

        mjd.ctrl[:] += ctrl
        for _ in range(substeps):
            mujoco.mj_step(mjm, mjd)

        if task.is_success(mjd) and steps_to_success is None:
            steps_to_success = t + 1
            print("Episode Success!")
            break

        if task.has_failed(mjd):
            print("Episode Failed!")
            break

    elapsed = time.perf_counter() - episode_start

    return EpisodeResult(
        task_name        = task_cfg.name,
        model_label      = cfg.label,
        condition        = condition,
        success          = steps_to_success is not None,
        steps_to_success = steps_to_success,
        final_cost       = float(np.linalg.norm(mjd.qpos - q0)), # THIS IS WRONG
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
    n_samples:      int,
    horizon:        int,
    seed:           int,
    geometry:       GeometryVariant,
    noise:          PhysicsNoiseParams,
    settle_seconds: float = 1.0,
    use_full_graph: bool  = True,
    nconmax:        int   = 200,
    njmax:          int   = 500,
    baseline_model: str   = "M2",
) -> list[AggregatedResult]:

    rng = np.random.default_rng(seed)
    aggregated: list[AggregatedResult] = []
    all_cfgs = {name: MODEL_FACTORIES[name]() for name in model_names}

    # Geometry variants not yet implemented — always simulate with ACCURATE.
    # The requested geometry value is still recorded in cell_tag so output
    # labels remain self-describing for when it is added.
    active_geometry = GeometryVariant.ACCURATE

    cell_tag_parts = []
    if geometry != GeometryVariant.ACCURATE:
        cell_tag_parts.append(geometry.value)
    if any(getattr(noise, f) > 0.0
           for f in ("mass_sigma", "inertia_sigma", "friction_sigma", "com_sigma")):
        cell_tag_parts.append(
            f"noise_m{noise.mass_sigma}_f{noise.friction_sigma}"
            f"_i{noise.inertia_sigma}_c{noise.com_sigma}"
        )
    cell_tag = "__" + "__".join(cell_tag_parts) if cell_tag_parts else ""

    print("=== Pre-computing speed / accuracy metrics ===")
    speed_cache: dict[str, float] = {}
    error_cache: dict[str, float] = {}

    for task_name in task_names:
        mjm, task = load_mjm_for_study(task_name, active_geometry, noise, rng)

        baseline_cfg  = all_cfgs[baseline_model]
        baseline_r    = measure_rollout_speed(mjm, baseline_cfg, nconmax=nconmax, njmax=njmax)
        baseline_time = baseline_r.mean_ms

        test_states = np.stack([
            np.concatenate(task.get_inital_state(rng)[:2])
            for _ in range(20)
        ]) #this is useless change

        for name, cfg in all_cfgs.items():
            key = f"{task_name}/{name}"

            speed_r = measure_rollout_speed(mjm, cfg, nconmax=nconmax, njmax=njmax)
            speed_cache[key] = baseline_time / speed_r.mean_ms

            mean_err, _ = measure_approximation_error(
                mjm, baseline_cfg, cfg, test_states, horizon=horizon,
                nconmax=nconmax, njmax=njmax,
            )
            error_cache[key] = mean_err
            print(f"  {key}: speedup={speed_cache[key]:.2f}x  err={error_cache[key]:.4f}")

    print("\n=== Running episodes ===")
    for task_name in task_names:
        mjm, task = load_mjm_for_study(task_name, active_geometry, noise, rng)

        for model_name in model_names:
            cfg = all_cfgs[model_name]
            key = f"{task_name}/{model_name}"

            for condition in conditions:
                print(f"  {task_name} | {model_name}{cell_tag} | Cond {condition} | {n_episodes} eps")
                episodes = []
                for ep in range(n_episodes):
                    print("Starting New Episode")
                    result = run_one_episode(
                        mjm            = mjm,
                        cfg            = cfg,
                        task           = task,
                        condition      = condition,
                        budget_seconds = budget_seconds,
                        n_samples      = n_samples,
                        horizon        = horizon,
                        seed           = seed,
                        rng            = rng,
                        settle_seconds = settle_seconds,
                        use_full_graph = use_full_graph,
                        nconmax        = nconmax,
                        njmax          = njmax,
                    )
                    result.model_label = cfg.label + cell_tag
                    episodes.append(result)

                agg = aggregate_episodes(
                    episodes, task_name, cfg.label + cell_tag, condition
                )
                agg.speedup_vs_baseline    = speed_cache.get(key, 1.0)
                agg.approx_err_vs_baseline = error_cache.get(key, 0.0)
                aggregated.append(agg)

    return aggregated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Master MPPI experiment runner — tasks × models × conditions."
    )
    parser.add_argument("--tasks",    nargs="+",
                        default=["grasp_reorient"],
                        help="Task names to evaluate")
    parser.add_argument("--models",   nargs="+",
                        default=["M1", "M2", "M3", "M4"],
                        choices=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--conditions", nargs="+", default=["B"],
                        choices=["A", "B"],
                        help="A=fixed_budget_rollout  B=warm-started MPPI")
    parser.add_argument("--n_episodes",     type=int,   default=40)
    parser.add_argument("--budget_seconds", type=float, default=0.1,
                        help="Per-step time budget for Condition A")
    parser.add_argument("--n_samples",      type=int,   default=1024)
    parser.add_argument("--horizon",        type=int,   default=50)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--settle",         type=float, default=1.0,
                        help="Seconds to allow physics to settle before planning starts")
    parser.add_argument("--use_full_graph", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a single mega CUDA graph (default) or separate step/reset graphs")
    parser.add_argument("--nconmax",        type=int,   default=200,
                        help="Max contacts per world for the Warp backend")
    parser.add_argument("--njmax",          type=int,   default=500,
                        help="Max constraint rows per world for the Warp backend")
    parser.add_argument("--output",         type=str,   default=None,
                        help="Path for results JSON (auto-timestamped if omitted)")

    # --- Orthogonal ablation axes ---
    # NOTE: geometry variants are not yet implemented.
    # --geometry is accepted and recorded in output labels but the simulation
    # always uses 'accurate'. This argument will be wired up when geometry
    # loading is added to the task XML paths.
    parser.add_argument("--geometry", type=str, default="accurate",
                        choices=[g.value for g in GeometryVariant],
                        help="(not yet implemented) Geometry variant — recorded in output labels only")
    parser.add_argument("--mass_sigma",     type=float, default=0.0)
    parser.add_argument("--inertia_sigma",  type=float, default=0.0)
    parser.add_argument("--friction_sigma", type=float, default=0.0)
    parser.add_argument("--com_sigma",      type=float, default=0.0)

    args = parser.parse_args()

    noise    = PhysicsNoiseParams(
        mass_sigma     = args.mass_sigma,
        inertia_sigma  = args.inertia_sigma,
        friction_sigma = args.friction_sigma,
        com_sigma      = args.com_sigma,
    )
    geometry = GeometryVariant(args.geometry)

    if geometry != GeometryVariant.ACCURATE:
        print(f"[WARNING] --geometry={geometry.value} requested but geometry variants "
              f"are not yet implemented. Falling back to 'accurate'.")

    results = run_study(
        task_names     = args.tasks,
        model_names    = args.models,
        conditions     = args.conditions,
        n_episodes     = args.n_episodes,
        budget_seconds = args.budget_seconds,
        n_samples      = args.n_samples,
        horizon        = args.horizon,
        seed           = args.seed,
        geometry       = geometry,
        noise          = noise,
        settle_seconds = args.settle,
        use_full_graph = args.use_full_graph,
        nconmax        = args.nconmax,
        njmax          = args.njmax,
    )

    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or str(RESULTS_DIR / f"experiment_{ts}.json")
    save_results(results, out)


if __name__ == "__main__":
    main()
