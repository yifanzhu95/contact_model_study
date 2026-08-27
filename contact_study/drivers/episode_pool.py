"""Run the episodes of one BO trial concurrently, one worker process each.

Why this exists
---------------
run_eval_episode alternates a GPU plan() with a CPU eval-sim advance, so one of
the two devices is always idle. Measured on grasp_reorient:

    eval_sim      plan (GPU)     eval (CPU)        plan share of wall
    mujoco        ~47 ms/step    ~5.8 ms/step      89%   (GPU-bound)
    pinocchio     ~47 ms/step    ~2-3 s/step       ~2%   (CPU-bound)

Fanning the episodes of a trial out across processes overlaps the two. In the
pinocchio regime — which is what `--eval_sim none` selects for grasp_reorient,
and therefore what experiments/hpc/bayes_opt.slurm runs — the GPU is idle 98% of
the time, so W workers give close to W-fold speedup before contention bites. In
the mujoco regime the GPU is already the bottleneck and the ceiling is ~1.1x;
the fan-out is harmless there, not helpful.

Why processes and not threads
-----------------------------
Each worker owns a complete planner (its own MJWarp Model/Data at
nworld=n_samples plus its own captured CUDA graphs, planners/base.py:489) and
its own eval simulator. None of that is thread-safe or shareable: Warp's capture
and stream state is per-thread, and the eval sims carry process-global
singletons (pinocchio_sim._PANDA_VIEWER, GraspReorientTask._active_eval_sim).
One episode at a time per process is exactly the invariant those singletons
already assume, so running episodes in separate processes satisfies them rather
than violating them.

SPAWN IS MANDATORY, NOT A PREFERENCE
------------------------------------
By the time the pool is built, run_bayes_opt.main() has already called wp.init()
and loaded a task that allocates wp.arrays on cuda. Forking a process that holds
a live CUDA context leaves the child's CUDA state undefined. The parent module
guards itself with `if __name__ == "__main__"`, which is what makes spawn safe.

Determinism
-----------
Episodes are pure functions of their SeedSequence, so the seed — not a
pre-built Generator — is what crosses the process boundary, and the Generator is
rebuilt identically inside the worker. Results are reassembled in ep_idx order.
A run with --n_workers 1 and one with --n_workers 4 therefore produce identical
objectives; tests/test_episode_pool_equivalence.py holds that line.
"""

from __future__ import annotations

import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Worker count
# ---------------------------------------------------------------------------

def default_worker_count(n_episodes: int) -> int:
    """Episodes, capped by the cores this process may ACTUALLY use.

    os.cpu_count() reports the machine, not the cgroup, so under
    `#SBATCH --cpus-per-task=4` on a 64-core node it would happily return 64 and
    oversubscribe the cell. sched_getaffinity respects the cgroup; SLURM_CPUS_PER_TASK
    is honored too for schedulers that set it without touching affinity.
    """
    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:          # not Linux
        n_cpu = os.cpu_count() or 1

    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            n_cpu = min(n_cpu, int(slurm))
        except ValueError:
            pass

    return max(1, min(int(n_episodes), n_cpu))


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------

def _worker_init(mujoco_gl: str) -> None:
    """Pay the imports once per worker instead of once per episode.

    MUJOCO_GL must be set BEFORE mujoco is imported, which is why it is passed
    in rather than inherited: a spawned child does inherit os.environ, but this
    keeps the ordering explicit and independent of import side effects.

    configure_headless_gl() is deliberately NOT called here — it re-execs the
    interpreter (utils/headless_gl.py:85), which inside a pool worker would be
    catastrophic. That path is CLI-only and this driver never renders.
    """
    os.environ["MUJOCO_GL"] = mujoco_gl

    import warp as wp
    wp.init()

    import contact_study.tasks                                      # noqa: F401
    from contact_study.drivers import run_eval_episode              # noqa: F401


def _failed_result(job: dict, exc: BaseException):
    """An EpisodeResult standing in for an episode that raised.

    Scored as a plain failure: success=False counts against the trial's success
    rate, and final_goal_errs=None drops it from the mean-goal-error term
    (normalized_goal_error returns None). One bad episode costs the trial a
    point instead of killing a 24-hour BO cell.
    """
    from contact_study.evaluation.metrics import EpisodeResult

    cfg = job.get("planner_cfg")
    contact_cfg = job.get("contact_cfg")
    return EpisodeResult(
        task_name       = job.get("task_name", "?"),
        model_label     = getattr(contact_cfg, "label", "?"),
        success         = False,
        steps_to_success= None,
        final_cost      = float("nan"),
        n_samples_used  = int(getattr(cfg, "n_samples", 0) or 0),
        elapsed_seconds = 0.0,
        planner         = job.get("planner", "mppi"),
        final_goal_errs = None,
        end_reason      = "error",
        error           = f"{type(exc).__name__}: {exc}",
    )


def _run_one(job: dict):
    """Run one episode. Returns an EpisodeResult, never raises.

    `job` is the full run_eval_episode kwarg dict except that `rng` is replaced
    by `seed_seq` (a np.random.SeedSequence), which pickles cleanly and rebuilds
    the exact same Generator here.
    """
    from contact_study.drivers.run_eval_episode import run_eval_episode

    job = dict(job)
    seed_seq = job.pop("seed_seq")
    try:
        return run_eval_episode(rng=np.random.default_rng(seed_seq), **job)
    except Exception as exc:                     # noqa: BLE001 — isolation is the point
        print(f"  ! episode {job.get('ep_idx')} raised; scoring it as a failure")
        traceback.print_exc()
        return _failed_result(job, exc)


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------

class EpisodePool:
    """A persistent spawn-based process pool for episode jobs.

    Persistent because standing a worker up costs a fresh interpreter, a CUDA
    context and the warp/mujoco/pinocchio imports; paying that per trial would
    eat the speedup it is meant to deliver.
    """

    def __init__(self, n_workers: int, mujoco_gl: str | None = None):
        self.n_workers = int(n_workers)
        self._mujoco_gl = mujoco_gl or os.environ.get("MUJOCO_GL", "egl")
        self._ex: ProcessPoolExecutor | None = None

    # -- lifecycle ----------------------------------------------------------

    def _executor(self) -> ProcessPoolExecutor:
        if self._ex is None:
            self._ex = ProcessPoolExecutor(
                max_workers = self.n_workers,
                # See the module docstring: the parent holds a live CUDA context.
                mp_context  = multiprocessing.get_context("spawn"),
                initializer = _worker_init,
                initargs    = (self._mujoco_gl,),
            )
        return self._ex

    def shutdown(self) -> None:
        """Tear the workers down. Safe to call twice, and on a pool never used."""
        if self._ex is not None:
            self._ex.shutdown(wait=True, cancel_futures=True)
            self._ex = None

    def __enter__(self) -> "EpisodePool":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    # -- the one operation --------------------------------------------------

    def map_episodes(
        self,
        jobs: Sequence[dict],
        on_result: Callable[[int, object], None] | None = None,
    ) -> list:
        """Run every job concurrently; return results in *submission* order.

        on_result(index, result) fires as each episode lands, out of order, so a
        20-minute pinocchio trial reports progress instead of going silent.

        _run_one swallows Python-level exceptions, so a BrokenProcessPool here
        means a hard crash — a segfault in coal, or the SLURM cgroup OOM-killing
        a worker. That poisons the executor permanently, so rebuild it and retry
        the trial once; a second break is real and propagates.
        """
        try:
            return self._submit(jobs, on_result)
        except BrokenProcessPool:
            print("  ! worker pool died (segfault or OOM-kill); rebuilding and "
                  "retrying this trial once")
            self.shutdown()
            return self._submit(jobs, on_result)

    def _submit(self, jobs: Sequence[dict], on_result) -> list:
        ex = self._executor()
        results: list = [None] * len(jobs)
        futures = {ex.submit(_run_one, job): i for i, job in enumerate(jobs)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            if on_result is not None:
                on_result(i, results[i])
        return results


def run_episodes_serially(
    jobs: Iterable[dict],
    on_result: Callable[[int, object], None] | None = None,
) -> list:
    """The --n_workers 1 path: same jobs, same _run_one, no processes at all.

    Routed through _run_one rather than calling run_eval_episode directly so the
    serial and parallel paths cannot drift apart.
    """
    results = []
    for i, job in enumerate(jobs):
        results.append(_run_one(job))
        if on_result is not None:
            on_result(i, results[i])
    return results
