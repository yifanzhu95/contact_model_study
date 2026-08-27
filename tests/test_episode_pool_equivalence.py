"""Fanning episodes out across worker processes must not change the objective.

run_bayes_opt runs a trial's episodes in parallel (drivers/episode_pool.py), one
worker process per episode, each owning its own planner and eval sim. Episodes
are pure functions of their SeedSequence, so --n_workers must be a pure
throughput knob: the same trial must score the same whether it ran in one
process or four.

Why cart_pole and not grasp_reorient — the same reason
tests/test_async_sync_equivalence.py gives: contact-rich tasks are not
reproducible run-to-run AT ALL, independent of this fan-out. The MPPI weight
normalization sums the sample weights with wp.atomic_add
(planners/mppi.py:117), and GPU float atomic addition is order-nondeterministic,
so eta lands a ULP or two apart between two runs of the *same* command. Over a
thousand contact-rich closed-loop steps that amplifies into entirely different
episodes. Verified: the untouched run_eval_episode driver at --seed 64 on
grasp_reorient returns final_cost 3.7937 on one run and 5.4632 on the next.
cart_pole with the MuJoCo eval sim has essentially no contacts and IS bit-exact,
which makes it the place to verify this.

The first three checks are pure bookkeeping and need no GPU. The last one needs
a CUDA device. The repo has no pytest — run directly:

    python tests/test_episode_pool_equivalence.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from contact_study.drivers.episode_pool import (          # noqa: E402
    EpisodePool, _failed_result, default_worker_count, run_episodes_serially,
)
from contact_study.drivers.run_bayes_opt import (          # noqa: E402
    BOObjective, build_parser, load_rollout_task, resolve_mppi_schedule,
)
from contact_study.planners.mppi import MPPIConfig         # noqa: E402


def cart_pole_argv(*models: str) -> list[str]:
    """The shared cart_pole setup, over whichever contact models are named."""
    return ["--models", *(models or ("M2",))] + CART_POLE_ARGV


CART_POLE_ARGV = [
    "--task", "cart_pole", "--eval_sim", "mujoco",
    "--no-record_trajectory", "--no-record_planner_dist",
    # The driver's default --opt_weights names grasp_reorient's weights.
    "--opt_weights", "angle:10:200", "pos:10:200",
    "--n_episodes", "2", "--n_samples", "128", "--settle", "0.0",
    "--seed", "64", "--bo_seed", "0",
]


def _objective(outdir: Path, pool, argv=None) -> BOObjective:
    args = build_parser().parse_args(argv if argv is not None else cart_pole_argv())
    task = load_rollout_task(args.task, args.geometry)
    schedule = resolve_mppi_schedule(
        MPPIConfig(time_horizon=args.time_horizon, step_time=args.step_time),
        task.config, args.eval_substeps,
    )
    return BOObjective(
        args, ["angle", "pos", "temperature"],
        dict(task.config.cost_weights), dict(task.config.success_thresholds),
        schedule, outdir, pool=pool,
    )


# ---------------------------------------------------------------------------

def test_job_is_picklable_and_carries_the_right_seed() -> None:
    """The job dict must survive a pickle round-trip and rebuild the same RNG.

    This is the load-bearing invariant behind the fan-out: the parent hands the
    worker a SeedSequence instead of a Generator, and the worker's
    default_rng(seed_seq) must be the Generator the serial path used.
    """
    import pickle

    with tempfile.TemporaryDirectory() as td:
        obj = _objective(Path(td), pool=None)
        job = obj._episode_job("M2", 1, {"angle": 50.0},
                               noise_sigma=0.1, temperature=3.0)

        # Everything crossing the process boundary must pickle.
        job2 = pickle.loads(pickle.dumps(job))

        assert job2["ep_idx"] == 1
        assert job2["cost_weight_overrides"] == {"angle": 50.0}
        assert "rng" not in job2, "the Generator must not cross; the seed does"

        # The rebuilt Generator matches what the old in-process loop built.
        want = np.random.default_rng(obj.episode_seeds[1]).random(8)
        got  = np.random.default_rng(job2["seed_seq"]).random(8)
        assert np.array_equal(want, got), "worker RNG diverges from the serial one"

        # The planner seed is derived in the parent and must be stable.
        assert job["planner_cfg"].seed == int(
            obj.episode_seeds[1].generate_state(1)[0]
        )
    print("ok  job pickles, and the worker rebuilds the identical Generator")


def test_worker_count_respects_the_cgroup(monkeypatch=None) -> None:
    """os.cpu_count() would report the whole node under --cpus-per-task."""
    import os

    assert default_worker_count(4) >= 1
    assert default_worker_count(1) == 1

    real = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        os.environ["SLURM_CPUS_PER_TASK"] = "2"
        assert default_worker_count(8) == 2, "SLURM_CPUS_PER_TASK must cap it"
    finally:
        if real is None:
            os.environ.pop("SLURM_CPUS_PER_TASK", None)
        else:
            os.environ["SLURM_CPUS_PER_TASK"] = real
    print("ok  worker count honors SLURM_CPUS_PER_TASK and sched_getaffinity")


def test_a_crashed_episode_scores_as_a_failure() -> None:
    """A worker that raises must cost its trial a point, not kill the run."""
    job = {"task_name": "cart_pole", "planner": "mppi",
           "contact_cfg": None, "planner_cfg": None, "ep_idx": 0}
    r = _failed_result(job, RuntimeError("coal segfaulted"))
    assert r.success is False
    assert r.end_reason == "error"
    assert r.final_goal_errs is None, "must drop out of the goal-error mean"
    assert "coal segfaulted" in r.error

    # run_episodes_serially routes through the same _run_one, so a raising job
    # comes back as a result rather than an exception.
    out = run_episodes_serially([{"seed_seq": np.random.SeedSequence(0),
                                  "task_name": "nope"}])
    assert len(out) == 1 and out[0].success is False
    print("ok  a crashed episode is scored as a failure, not raised")


def test_parallel_matches_serial() -> None:
    """The whole point: same trial, same score, any worker count. Needs CUDA."""
    x = [50.0, 80.0, 5.0]          # angle, pos, temperature

    with tempfile.TemporaryDirectory() as td:
        serial = Path(td) / "serial"
        par    = Path(td) / "par"
        serial.mkdir(); par.mkdir()

        j_serial = _objective(serial, pool=None)(x)
        pool = EpisodePool(2)
        try:
            j_par = _objective(par, pool=pool)(x)
        finally:
            pool.shutdown()

        assert j_serial == j_par, f"objective {j_serial} != {j_par}"

        def episodes(d: Path):
            cell = json.load(open(sorted(d.glob("cell_*.json"))[0]))
            return [(e["final_cost"], e["n_steps_taken"], e["end_reason"],
                     e["success"]) for e in cell["episodes"]]

        a, b = episodes(serial), episodes(par)
        assert a == b, f"per-episode results differ:\n  {a}\n  {b}"

    print(f"ok  serial and 2-worker agree bit-exactly (objective={j_serial:+.6f})")


def test_every_model_gets_the_identical_episodes() -> None:
    """A multi-model trial must seed by episode, never by (model, episode).

    If the models saw different initial states, a per-model score gap could not
    be attributed to the contact model — which is the whole point of scoring one
    weight vector across several. No GPU needed: this is job bookkeeping.
    """
    with tempfile.TemporaryDirectory() as td:
        obj = _objective(Path(td), pool=None, argv=cart_pole_argv("M1", "M2", "M3"))
        assert list(obj.contact_cfgs) == ["M1", "M2", "M3"]

        for ep in (0, 1):
            jobs = [obj._episode_job(m, ep, {}, 0.1, 3.0) for m in obj.contact_cfgs]
            seeds = {j["planner_cfg"].seed for j in jobs}
            draws = {np.random.default_rng(j["seed_seq"]).random(4).tobytes()
                     for j in jobs}
            assert len(seeds) == 1, f"ep {ep}: models got different planner seeds"
            assert len(draws) == 1, f"ep {ep}: models got different episode RNGs"

            # ...but the contact model itself must differ.
            labels = [j["contact_cfg"].label for j in jobs]
            assert len(set(labels)) == 3, f"models share a contact cfg: {labels}"

        # Episodes still differ from each other.
        a = obj._episode_job("M1", 0, {}, 0.1, 3.0)["planner_cfg"].seed
        b = obj._episode_job("M1", 1, {}, 0.1, 3.0)["planner_cfg"].seed
        assert a != b, "distinct episodes must not share a seed"
    print("ok  every model sees identical episodes; only the contact model varies")


def test_model_aggregation() -> None:
    """mean averages the per-model objectives; worst takes the max (minimized)."""
    from contact_study.evaluation.metrics import EpisodeResult

    def ep(success: bool):
        return EpisodeResult(task_name="t", model_label="m", success=success,
                             steps_to_success=None, final_cost=0.0,
                             n_samples_used=1, elapsed_seconds=0.0)

    with tempfile.TemporaryDirectory() as td:
        obj = _objective(Path(td), pool=None)
        good = obj._score([ep(True), ep(True)])      # objective -1.0
        bad  = obj._score([ep(False), ep(False)])    # objective  0.0
        assert good["objective"] == -1.0 and good["success_rate"] == 1.0
        assert bad["objective"] == 0.0 and bad["success_rate"] == 0.0

        scores = [good["objective"], bad["objective"]]
        assert float(np.mean(scores)) == -0.5, "mean should average the models"
        assert float(max(scores)) == 0.0, "worst should surface the failing model"
    print("ok  --model_agg mean averages, worst surfaces the failing model")


if __name__ == "__main__":
    test_job_is_picklable_and_carries_the_right_seed()
    test_worker_count_respects_the_cgroup()
    test_a_crashed_episode_scores_as_a_failure()
    test_every_model_gets_the_identical_episodes()
    test_model_aggregation()
    test_parallel_matches_serial()
    print("\nall checks passed")
