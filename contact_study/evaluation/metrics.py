"""Evaluation metrics and result aggregation.

Primary metrics:
  - success_rate:     fraction of episodes where is_success() triggers
  - steps_to_success: mean steps (conditioned on success)
  - planning_cost:    mean terminal cost across episodes

Secondary metrics (for the study's main figures):
  - accuracy_speed_frontier: (approx_err, speedup) pairs for Pareto analysis
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import json
import numpy as np


def _from_dict(cls, d: dict, *, drop: Sequence[str] = ()):
    """Build a dataclass from a JSON dict, ignoring keys it no longer declares.

    Result files on disk predate several schema changes -- most recently the
    removal of `condition` -- so a bare cls(**d) raises TypeError on anything
    written before that. Unknown keys are dropped and missing keys fall back to
    the dataclass defaults, which is why every field added since the original
    schema carries one.
    """
    names = {f.name for f in dataclasses.fields(cls)} - set(drop)
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class EpisodeResult:
    task_name:         str
    model_label:       str
    success:           bool
    steps_to_success:  int | None
    final_cost:        float
    n_samples_used:    int
    elapsed_seconds:   float
    mean_step_ms:      float = 0.0
    std_step_ms:       float = 0.0
    # Distribution summaries of controller.plan() latency.  The mean alone is
    # easily distorted by occasional CUDA scheduling or contact-heavy spikes;
    # median and P95 distinguish the typical cost from the slow tail.
    median_step_ms:    float = 0.0
    p95_step_ms:       float = 0.0
    max_step_ms:       float = 0.0
    # Which sampling planner produced the episode ("mppi" | "cem" |
    # "predictive_sampler"). Defaulted so results written before the planner
    # became selectable still load via EpisodeResult.from_dict.
    planner:           str   = "mppi"
    # BaseTask.goal_errors() evaluated on the episode's final state: the actual
    # per-criterion distance to the task goal ({"pos": .., "quat": .., "vel": ..}
    # for grasp_reorient), keyed like TaskConfig.success_thresholds. None when
    # the task has no continuous goal metric. Distinct from final_cost, which is
    # ||q_final - q_0|| — displacement from the START pose, not goal error.
    final_goal_errs:   dict[str, float] | None = None

    # --- how the episode ended ---------------------------------------------
    # The control loop leaves by exactly one of three doors, and `success` alone
    # cannot tell them apart: a False success is "ran out of time" on one task
    # and "dropped the object through the floor" on another (grasp_reorient's
    # has_failed). end_reason names the door:
    #   "success"  is_success() fired and the loop broke out — or, in multi-goal
    #              mode, at least one success was recorded before the episode
    #              length ran out
    #   "failed"   has_failed() fired: the task hit a terminal bad state
    #   "timeout"  reached cfg.max_steps having never succeeded
    #   "unknown"  the record predates these fields
    # time_out is orthogonal, and answers only "did the loop run to max_steps":
    # in multi-goal mode (fin_ep_on_success=False) the episode always runs the
    # full length even after successes, so time_out stays True there while
    # end_reason reads "success".
    # n_steps_taken counts the control steps that actually executed (executor
    # TICKS in the async driver, missed ticks included). It equals
    # len(trajectory["steps"]["step"]) whenever a trajectory was recorded, and is
    # NOT steps_to_success, which is the index of the FIRST success. 0 on an old
    # record means "not recorded", not "failed immediately".
    time_out:          bool  = False
    end_reason:        str   = "unknown"
    n_steps_taken:     int   = 0

    # --- asynchronous driver telemetry ------------------------------------
    # Filled only by contact_study/drivers/run_async_eval_episode.py, where the
    # eval sim keeps running while the planner solves. All defaulted so results
    # from the synchronous driver (and every result written before this driver
    # existed) still load.
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
    median_latency_ms:    float = 0.0
    p95_latency_ms:       float = 0.0
    mean_staleness_ms:    float = 0.0
    missed_ticks:         int   = 0
    tape_exhausted_ticks: int   = 0
    sim_seconds:          float = 0.0

    # --- per-control-step recording ---------------------------------------
    # The trajectory + planner-distribution block built by
    # contact_study/evaluation/trajectory.py: enough to replay the applied
    # controls and to compute a KL divergence to a reference planner offline.
    # None when recording was disabled (--no-record_trajectory /
    # --no-record_planner_dist) or the record predates it. Megabytes when
    # populated — see evaluation/json_io.py, which is what keeps it from
    # exploding the file.
    trajectory:        dict | None = None

    # --- worker-process failure -------------------------------------------
    # Set only when the episode raised inside a pool worker (see
    # contact_study/drivers/episode_pool.py): "<ExcType>: <message>". The result
    # is otherwise a plain failure — success=False, final_goal_errs=None — so a
    # crashed episode costs its trial a point instead of killing the run.
    # end_reason reads "error" in that case, the fourth door alongside the three
    # documented above.
    error:             str | None = None

    @classmethod
    def from_dict(cls, d: dict, *, drop_trajectory: bool = False) -> "EpisodeResult":
        """Build from a JSON dict, ignoring keys this dataclass no longer declares.

        drop_trajectory skips the (large) trajectory block for callers that only
        want the summary statistics.
        """
        return _from_dict(cls, d, drop=("trajectory",) if drop_trajectory else ())

    def to_dict(self) -> dict:
        """Shallow field dict for JSON. Deliberately NOT dataclasses.asdict:
        that deep-copies, and `trajectory` is megabytes."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


@dataclass
class AggregatedResult:
    """Summary statistics over multiple episodes for one (task, Mk) cell."""
    task_name:              str
    model_label:            str
    n_episodes:             int

    success_rate:           float
    success_rate_se:        float    # standard error

    mean_steps_to_success:  float | None  # None if no successes
    mean_final_cost:        float
    std_final_cost:         float

    mean_n_samples:         float    # avg samples per planning cycle
    mean_elapsed:           float    # avg wall-clock per episode
    mean_step_ms:           float = 0.0   # avg per-control-step planning time
    std_step_ms:            float = 0.0

    # Speed / accuracy metadata (filled by run_full_study)
    speedup_vs_baseline:    float = 1.0
    approx_err_vs_baseline: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "AggregatedResult":
        """Build from a JSON dict, ignoring keys this dataclass no longer declares."""
        return _from_dict(cls, d)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def aggregate_episodes(
    episodes: list[EpisodeResult],
    task_name: str,
    model_label: str,
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


CELL_SCHEMA = 2


def cell_record(*, label: str, task: str, model: str,
                aggregate: AggregatedResult, episodes: list[EpisodeResult],
                config: dict | None = None, **extra) -> dict:
    """The standard per-cell result file: the aggregate row AND the episodes behind it.

    The sweep workers used to write a bare JSON list of AggregatedResult dicts via
    save_results, which discarded every per-episode field — including end_reason
    and the recorded trajectory. This is the shape run_kl_divergence_cell already
    writes, so every cell file in the repo now agrees on one layout.
    analysis/sweep_io.load_aggregates reads both this and the old list.
    """
    return {
        "schema":    CELL_SCHEMA,
        "label":     label,
        "task":      task,
        "model":     model,
        "config":    config or {},
        **extra,
        "aggregate": aggregate.to_dict(),
        "episodes":  [e.to_dict() for e in episodes],
    }


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
    return [AggregatedResult.from_dict(d) for d in data]


def build_results_table(
    results: list[AggregatedResult],
    tasks: list[str],
    models: list[str],
    metric: str = "success_rate",
) -> np.ndarray:
    """Build a (len(models), len(tasks)) matrix for a given metric.

    Useful for generating the main results table in the paper.
    """
    mat = np.full((len(models), len(tasks)), np.nan)
    idx = {(r.model_label, r.task_name): r for r in results}
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            key = (m, t)
            if key in idx:
                mat[i, j] = getattr(idx[key], metric)
    return mat


def accuracy_speed_frontier(
    results:    list[AggregatedResult],
    task_name:  str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract (approx_err, speedup, label) for Pareto frontier plot."""
    subset = [r for r in results if r.task_name == task_name]
    errs     = np.array([r.approx_err_vs_baseline for r in subset])
    speedups = np.array([r.speedup_vs_baseline    for r in subset])
    labels   = [r.model_label for r in subset]
    return errs, speedups, labels
