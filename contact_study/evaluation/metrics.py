"""Evaluation metrics and result aggregation.

Primary metrics:
  - success_rate:     fraction of episodes where is_success() triggers
  - steps_to_success: mean steps (conditioned on success)
  - planning_cost:    mean terminal cost across episodes

Secondary metrics (for the study's main figures):
  - accuracy_speed_frontier: (approx_err, speedup) pairs for Pareto analysis
  - condition_a_vs_b:        performance delta between fixed-budget and
                             fixed-sample conditions for each Mk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import json
import numpy as np

from contact_study.contact_models.config import ContactModelConfig
from contact_study.tasks.base import ContactComplexity


@dataclass
class EpisodeResult:
    task_name:         str
    model_label:       str
    condition:         str        # "A" (fixed budget) | "B" (fixed sample)
    success:           bool
    steps_to_success:  int | None
    final_cost:        float
    n_samples_used:    int
    elapsed_seconds:   float
    mean_step_ms:      float = 0.0
    std_step_ms:       float = 0.0
    # Which sampling planner produced the episode ("mppi" | "cem" |
    # "predictive_sampler"). Defaulted so results written before the planner
    # became selectable still load via EpisodeResult(**json_dict).
    planner:           str   = "mppi"
    # BaseTask.goal_errors() evaluated on the episode's final state: the actual
    # per-criterion distance to the task goal ({"pos": .., "quat": .., "vel": ..}
    # for grasp_reorient), keyed like TaskConfig.success_thresholds. None when
    # the task has no continuous goal metric. Distinct from final_cost, which is
    # ||q_final - q_0|| — displacement from the START pose, not goal error.
    # Defaulted so older result JSON still loads via EpisodeResult(**json_dict).
    final_goal_errs:   dict[str, float] | None = None

    # --- asynchronous driver telemetry ------------------------------------
    # Filled only by contact_study/drivers/run_async_eval_episode.py, where the
    # eval sim keeps running while the planner solves. All defaulted so results
    # from the synchronous driver (and every result written before this driver
    # existed) still load via EpisodeResult(**json_dict).
    #   n_plans              plan() calls completed during the episode
    #   mean/std_latency_ms  realized planning latency AS CHARGED (after any
    #                        --plan_latency_ms override / --latency_scale),
    #                        i.e. the sim time the loop actually spent planning.
    #                        mean_step_ms keeps holding the raw measured cost.
    #   mean_staleness_ms    mean age of a tape row at the moment it is applied,
    #                        measured from the start of the solve that made it
    #   missed_ticks         executor ticks that replayed an already-consumed
    #                        tape row because no fresh plan had landed
    #   tape_exhausted_ticks executor ticks that ran off the end of the tape and
    #                        clamped to its last row (planner slower than its
    #                        own horizon)
    #   sim_seconds          simulated duration of the episode
    n_plans:              int   = 0
    mean_latency_ms:      float = 0.0
    std_latency_ms:       float = 0.0
    mean_staleness_ms:    float = 0.0
    missed_ticks:         int   = 0
    tape_exhausted_ticks: int   = 0
    sim_seconds:          float = 0.0


@dataclass
class AggregatedResult:
    """Summary statistics over multiple episodes for one (task, Mk, condition) cell."""
    task_name:              str
    model_label:            str
    condition:              str
    n_episodes:             int

    success_rate:           float
    success_rate_se:        float    # standard error

    mean_steps_to_success:  float | None  # None if no successes
    mean_final_cost:        float
    std_final_cost:         float

    mean_n_samples:         float    # avg samples per planning cycle (Condition A varies)
    mean_elapsed:           float    # avg wall-clock per episode
    mean_step_ms:           float = 0.0   # avg per-control-step planning time
    std_step_ms:            float = 0.0

    # Speed / accuracy metadata (filled by run_full_study)
    speedup_vs_baseline:    float = 1.0
    approx_err_vs_baseline: float = 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


import dataclasses


def aggregate_episodes(
    episodes: list[EpisodeResult],
    task_name: str,
    model_label: str,
    condition: str,
) -> AggregatedResult:
    """Compute summary statistics from a list of EpisodeResult."""
    n = len(episodes)
    successes = [e.success for e in episodes]
    sr = float(np.mean(successes))
    se = float(np.std(successes) / np.sqrt(n)) if n > 1 else 0.0

    success_steps = [e.steps_to_success for e in episodes if e.steps_to_success is not None]
    mean_sts = float(np.mean(success_steps)) if success_steps else None

    costs    = [e.final_cost for e in episodes]
    samples  = [e.n_samples_used for e in episodes]
    elapsed  = [e.elapsed_seconds for e in episodes]

    step_ms     = [e.mean_step_ms for e in episodes]
    all_step_sd = [e.std_step_ms  for e in episodes]

    return AggregatedResult(
        task_name             = task_name,
        model_label           = model_label,
        condition             = condition,
        n_episodes            = n,
        success_rate          = sr,
        success_rate_se       = se,
        mean_steps_to_success = mean_sts,
        mean_final_cost       = float(np.mean(costs)),
        std_final_cost        = float(np.std(costs)),
        mean_n_samples        = float(np.mean(samples)),
        mean_elapsed          = float(np.mean(elapsed)),
        mean_step_ms          = float(np.mean(step_ms)),
        std_step_ms           = float(np.mean(all_step_sd)),
    )


def save_results(results: list[AggregatedResult], path: str | Path):
    """Serialize results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"Saved {len(results)} results to {path}")


def load_results(path: str | Path) -> list[AggregatedResult]:
    """Deserialize results from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [AggregatedResult(**d) for d in data]


def build_results_table(
    results: list[AggregatedResult],
    tasks: list[str],
    models: list[str],
    condition: str = "A",
    metric: str = "success_rate",
) -> np.ndarray:
    """Build a (len(models), len(tasks)) matrix for a given metric.

    Useful for generating the main results table in the paper.
    """
    mat = np.full((len(models), len(tasks)), np.nan)
    idx = {(r.model_label, r.task_name): r for r in results if r.condition == condition}
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            key = (m, t)
            if key in idx:
                mat[i, j] = getattr(idx[key], metric)
    return mat


def accuracy_speed_frontier(
    results:    list[AggregatedResult],
    task_name:  str,
    condition:  str = "A",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract (approx_err, speedup, label) for Pareto frontier plot."""
    subset = [r for r in results if r.task_name == task_name and r.condition == condition]
    errs     = np.array([r.approx_err_vs_baseline for r in subset])
    speedups = np.array([r.speedup_vs_baseline    for r in subset])
    labels   = [r.model_label for r in subset]
    return errs, speedups, labels
