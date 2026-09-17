"""Pooled temperature x noise_sigma grid search — one cell per GPU.

A cell is one (contact model, object, n_iterations, n_samples): the SLURM script
owns those four axes and hands this script a single value of each, and this
script grids temperature x noise_sigma inside it.

The third sweep in this repo, and deliberately distinct from the other two:

  * experiments/hpc/param_search.slurm + run_param_cell.py — the SLURM script
    owns the grid and each array task evaluates ONE point, serially. No object
    axis, and the GPU idles between episodes.
  * contact_study/drivers/run_bayes_opt.py — the right execution model (an
    EpisodePool running many episodes at once on one GPU) but a GP search, so it
    never produces an even, readable surface over two knobs.

Here the SLURM script hands each array task one contact model, one object (via
--geometry), one --n_iterations and one --n_samples, and THIS script walks the
whole temperature x noise_sigma grid inside that cell, running its episodes
through the same pool run_bayes_opt uses. That suits these two knobs in
particular: the right MPPI temperature depends on the scale of the task cost,
which differs per contact model and per object, so the useful picture is one grid
per cell rather than a single global optimum. The iteration and sample counts are
cell axes rather than grid axes because they change the cost of an episode rather
than only its outcome: giving each its own GPU keeps one cell's wall clock
roughly constant instead of multiplying the inner grid.

--convergence_tol switches MPPI to its convergence-terminated mode (iterate until
the returned action settles, at most --max_iterations), which decides the
iteration count per plan() call; --n_iterations is then ignored.

Grid points are scored on the SAME episodes (--seed is constant, spawned once),
so neighbouring temperatures differ only in the knob. Points are ranked by
success rate, ties broken by mean steps-to-success.

Each point writes results/<run>/cell_<id>.json in run_param_cell.py's schema, so
experiments/hpc/combine_results.py merges a whole array unchanged. Cell ids are
offset by --array_index, which is what lets every array task share one output
directory. A point whose file already exists is skipped, so a requeued or
timed-out cell resumes where it left off.

Deliberately does NOT import from run_bayes_opt.py: that module imports skopt at
module scope, and this job has no business requiring scikit-optimize.

    python -m contact_study.drivers.run_temp_sigma_grid \
        --task grasp_reorient --geometry duck_high_high --model M2 \
        --temperatures 40,20,10 --noise_sigmas 0.2,0.1,0.05 --n_episodes 5
"""

from __future__ import annotations

import os
# This driver never renders, but MuJoCo still picks a GL backend at import time;
# match the sweep workers and default to EGL before warp/mujoco are imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import dataclasses
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.evaluation import json_io
from contact_study.evaluation.trajectory import (
    TrajectoryConfig, add_cli_flags as add_record_flags,
)
from contact_study.drivers.episode_pool import (
    EpisodePool, default_worker_count, run_episodes_serially,
)
from contact_study.drivers.run_eval_episode import (
    MODEL_FACTORIES, load_rollout_task, resolve_mppi_schedule,
)
from contact_study.planners import (
    PLANNER_NAMES, make_planner_config, planner_config_cls, resolve_planner_name,
)
from contact_study.planners.mppi import MPPIConfig
from contact_study.tasks.config import EvalSimulatorKind, SceneVariant

RESULTS_DIR = Path(__file__).parents[2] / "results"

# Rough floor for one worker's share of VRAM: a CUDA context plus an MJWarp
# Model/Data at nworld=n_samples plus the (N, H, nu) sample block. Only used to
# warn — the real cost depends on --n_samples/--nconmax/--njmax.
MIN_VRAM_PER_WORKER_GB = 1.5


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def parse_values(raw: str, flag: str) -> list[float]:
    """"40,20,10" or "40 20 10" -> [40.0, 20.0, 10.0].

    Both knobs are strictly positive (temperature divides the cost in MPPI's
    softmax; noise_sigma is a standard deviation), so a zero or negative entry
    is a typo worth failing on rather than an episode that crashes later.
    """
    tokens = [tok for tok in raw.replace(",", " ").split() if tok]
    if not tokens:
        raise ValueError(f"{flag} is empty; give at least one value")
    values = []
    for tok in tokens:
        try:
            v = float(tok)
        except ValueError:
            raise ValueError(f"bad {flag} value {tok!r}; expected a number") from None
        if v <= 0.0:
            raise ValueError(f"{flag} values must be > 0, got {v:g}")
        values.append(v)
    if len(set(values)) != len(values):
        raise ValueError(f"{flag} lists a value twice: {values}")
    return values


def combo_label(model_key: str, obj: str, axes: dict) -> str:
    """Short label encoding model + object + every axis, grid AND cell.

    The cell axes are in here, unlike run_param_cell.py's version, because every
    array task writes into ONE shared directory: two cells that differ only in
    their object, iteration count or sample count would otherwise collide on a
    label and combine_results.py would fold them into a single row.
    """
    parts = []
    for k, v in axes.items():
        short = k[2:] if k.startswith("w_") else k
        parts.append(f"{short}={v:g}" if isinstance(v, (int, float)) else f"{short}={v}")
    return f"{model_key}_{obj}__" + "_".join(parts)


def planner_default(planner: str, field: str):
    """A planner config's own default for `field`.

    The iteration flags have no CLI default so that omitting one lets the config
    default stand (base.PlannerConfig.n_iterations=1, CEMConfig's 3,
    MPPIConfig.max_iterations=10). The cell record and label still want the value
    the run actually used, so read it off the dataclass rather than duplicating
    the number here.
    """
    for f in dataclasses.fields(planner_config_cls(planner)):
        if f.name == field:
            return f.default
    raise KeyError(f"planner {planner!r} has no {field!r} field")


def _device_free_gb() -> float | None:
    """Free memory on the warp CUDA device, in GiB; None if unavailable."""
    try:
        return wp.get_device().free_memory / 1024 ** 3
    except Exception:
        return None


# ---------------------------------------------------------------------------
# One grid point
# ---------------------------------------------------------------------------

class GridSearch:
    """Runs the temperature x noise_sigma grid for one cell.

    A cell is one (model, object, n_iterations, n_samples) — those four are the
    SLURM array's axes and are fixed here; only temperature and noise_sigma vary
    across `points`.

    Holds the per-episode seeds (computed ONCE, so every point is evaluated on
    the same episodes), the resolved schedule and the output directory.
    """

    def __init__(self, args, points: list[tuple[float, float]],
                 default_weights: dict[str, float], overrides: dict[str, float],
                 schedule: tuple[int, int, float], outdir: Path,
                 pool: EpisodePool | None = None):
        self.args    = args
        self.points  = points
        self.outdir  = outdir
        self.pool    = pool
        self.default_weights = default_weights
        self.overrides       = overrides
        self.horizon, self.substeps, self.rollout_dt = schedule

        self.planner     = resolve_planner_name(args.planner)
        self.contact_cfg = MODEL_FACTORIES[args.model]()
        self.obj         = SceneVariant.parse(args.geometry).obj
        self.eval_sim    = (None if args.eval_sim == "none"
                            else EvalSimulatorKind(args.eval_sim))
        self.delta       = ((-args.delta, args.delta) if args.delta is not None
                            else (None, None))

        # The axes the SLURM array owns: fixed for this whole cell, but part of
        # every point's identity because all array tasks share one --outdir.
        # Under --convergence_tol the iteration count is decided per plan() call,
        # so the tolerance stands in for it (and args.n_iterations is ignored).
        if args.convergence_tol is not None:
            self.cell_axes = {"convergence_tol": args.convergence_tol,
                              "n_samples":       args.n_samples}
            self.max_iterations = (args.max_iterations if args.max_iterations is not None
                                   else planner_default(self.planner, "max_iterations"))
        else:
            self.max_iterations = None
            n_iter = (args.n_iterations if args.n_iterations is not None
                      else planner_default(self.planner, "n_iterations"))
            self.cell_axes = {"n_iterations": n_iter,
                              "n_samples":    args.n_samples}

        # THE constant seeds. Spawned once and replayed for every grid point, so
        # two points differ only in their knobs — the whole reason a grid over
        # temperature is readable at all.
        self.episode_seeds = np.random.SeedSequence(args.seed).spawn(args.n_episodes)
        self.record_cfg    = TrajectoryConfig.from_args(args)

        # Ranked at the end; also what grid_summary_*.json carries.
        self.summaries: list[dict] = []

    # -- addressing ---------------------------------------------------------

    def cell_id(self, i: int) -> int:
        """Globally unique id for grid point `i` of this array task.

        Offsetting by the array index (rather than having the SLURM script do
        the arithmetic) keeps the grid size in one place: bash never has to know
        how many points the lists expand to.
        """
        return self.args.array_index * len(self.points) + i

    def cell_path(self, i: int) -> Path:
        return self.outdir / f"cell_{self.cell_id(i):05d}.json"

    # -- one episode's job description --------------------------------------

    def _episode_job(self, i: int, ep: int) -> dict:
        """run_eval_episode's kwargs for episode `ep` of point `i`, as picklable
        plain data.

        `rng` is replaced by `seed_seq`: the worker rebuilds the Generator with
        np.random.default_rng(seed_seq). SeedSequence is stateless, so deriving
        the planner seed here and the Generator there yields exactly what an
        in-process np.random.default_rng(self.episode_seeds[ep]) would have.

        The seed is keyed by EPISODE ONLY, never by grid point: every point must
        see the identical initial states, goals and planner noise, or a score
        difference could not be attributed to the knobs.
        """
        args = self.args
        temperature, noise_sigma = self.points[i]
        planner_kwargs = dict(
            n_samples         = args.n_samples,
            step_horizon      = args.horizon,
            time_horizon      = args.time_horizon,
            step_time         = args.step_time,
            noise_sigma       = noise_sigma,
            step_substeps     = args.substeps,
            warm_start        = args.warm_start,
            use_full_graph    = args.use_full_graph,
            delta_range       = self.delta,
            nconmax           = args.nconmax,
            njmax             = args.njmax,
            seed              = int(self.episode_seeds[ep].generate_state(1)[0]),
            debug             = args.debug,
            resample_interval = args.resample_interval,
            temperature       = temperature,
        )
        # No CLI default on any of these: omitting a key lets the planner
        # config's own default stand (n_iterations 1 for mppi/predictive_sampler
        # and 3 for cem, max_iterations 10). --convergence_tol replaces the fixed
        # count outright, so n_iterations is left out entirely under it.
        if args.convergence_tol is not None:
            planner_kwargs["convergence_tol"] = args.convergence_tol
            if args.max_iterations is not None:
                planner_kwargs["max_iterations"] = args.max_iterations
        elif args.n_iterations is not None:
            planner_kwargs["n_iterations"] = args.n_iterations

        return dict(
            task_name             = args.task,
            geometry              = args.geometry,
            contact_cfg           = self.contact_cfg,
            planner               = self.planner,
            planner_cfg           = make_planner_config(self.planner, **planner_kwargs),
            seed_seq              = self.episode_seeds[ep],
            video_path            = None,
            cost_weight_overrides = self.overrides or None,
            settle_seconds        = args.settle,
            eval_substeps         = args.eval_substeps,
            eval_sim              = self.eval_sim,
            ep_idx                = ep,
            fin_ep_on_success     = True,
            debug                 = args.debug,
            verbose               = args.debug,
            record                = self.record_cfg,
        )

    # -- scoring ------------------------------------------------------------

    def _write_point(self, i: int, episodes: list) -> dict:
        """Score one finished grid point, write its cell JSON, return a summary.

        Ranking is by success rate with mean steps-to-success as the tiebreak,
        the same order combine_results.py prints, so the table this driver ends
        with and the merged table agree.
        """
        args = self.args
        temperature, noise_sigma = self.points[i]

        n_success    = sum(r.success for r in episodes)
        success_rate = n_success / len(episodes) if episodes else 0.0
        succ_steps   = [r.steps_to_success for r in episodes
                        if r.steps_to_success is not None]
        mean_sts     = float(np.mean(succ_steps)) if succ_steps else None
        step_ms      = [r.mean_step_ms for r in episodes]
        step_sd      = [r.std_step_ms  for r in episodes]
        elapsed      = [r.elapsed_seconds for r in episodes]

        # The cell axes (n_iterations/n_samples, or convergence_tol in place of
        # the iteration count) ride in `axes` and in the label alongside the two
        # grid axes: every array task shares one --outdir, so without them two
        # cells differing only in iterations or samples would produce the same
        # label and combine_results.py would merge them into one row.
        axes = {"model":       args.model,
                "object":      self.obj,
                "temperature": temperature,
                "noise_sigma": noise_sigma,
                **self.cell_axes}
        label = combo_label(args.model, self.obj,
                            {"temperature": temperature, "noise_sigma": noise_sigma,
                             **self.cell_axes})

        record = {
            "combo_index":           self.cell_id(i),
            "task":                  args.task,
            "model":                 args.model,
            "label":                 label,
            "overrides":             self.overrides,
            # Every grid axis — what the analysis scripts group rows by.
            # "overrides" stays weights-only for combine_results.py.
            "axes":                  axes,
            "swept_knobs":           ["temperature", "noise_sigma",
                                      *self.cell_axes],
            "full_weights":          {**self.default_weights, **self.overrides},
            "n_episodes":            len(episodes),
            "n_success":             n_success,
            "success_rate":          success_rate,
            "mean_steps_to_success": mean_sts,
            "mean_step_ms":          float(np.mean(step_ms)),
            "std_step_ms":           float(np.mean(step_sd)),
            "mean_elapsed_s":        float(np.mean(elapsed)),
            "mppi": {
                "n_samples":         args.n_samples,
                "time_horizon":      args.time_horizon,
                "temperature":       temperature,
                "noise_sigma":       noise_sigma,
                "step_time":         args.step_time,
                # Resolved, not the raw flags: a flag left off means "the
                # planner config's own default", which is what actually ran.
                "n_iterations":      self.cell_axes.get("n_iterations"),
                "convergence_tol":   args.convergence_tol,
                "max_iterations":    self.max_iterations,
                # Resolved against rollout_dt — what the controller actually ran.
                "step_horizon":      self.horizon,
                "step_substeps":     self.substeps,
                "rollout_dt":        self.rollout_dt,
                "delta":             args.delta,
                "resample_interval": args.resample_interval,
            },
            "seed":       args.seed,
            "eval_sim":   args.eval_sim,
            "geometry":   args.geometry,
            "object":     self.obj,
            "settle":     args.settle,
            "planner":    self.planner,
            "array_index": args.array_index,
            "grid_point": i,
            "episodes":   [r.to_dict() for r in episodes],
        }
        json_io.dump(record, self.cell_path(i), precision=self.record_cfg.precision)

        summary = {k: record[k] for k in
                   ("combo_index", "label", "success_rate", "n_success",
                    "mean_steps_to_success", "mean_step_ms", "mean_elapsed_s")}
        summary.update(temperature=temperature, noise_sigma=noise_sigma,
                       grid_point=i, **self.cell_axes)
        self.summaries.append(summary)

        sstr = f"{mean_sts:.1f}" if mean_sts is not None else "—"
        print(f"  [point {i:02d}/{len(self.points) - 1}]  T={temperature:g} "
              f"sigma={noise_sigma:g}  → success={success_rate*100:.1f}% "
              f"({n_success}/{len(episodes)})  mean_steps={sstr}  "
              f"step_ms={float(np.mean(step_ms)):.3f}")
        return summary

    # -- the grid -----------------------------------------------------------

    def run(self) -> list[dict]:
        """Run every not-yet-finished point; return the summaries, best first."""
        args = self.args

        todo = []
        for i in range(len(self.points)):
            path = self.cell_path(i)
            if path.exists():
                t, s = self.points[i]
                print(f"  [point {i:02d}] T={t:g} sigma={s:g}  already done "
                      f"-> {path.name}, skipping")
                self._load_summary(i, path)
            else:
                todo.append(i)

        if not todo:
            print("  every grid point already has a cell file; nothing to run")
            return self._ranked()

        # Every episode of every remaining point is independent — unlike the BO
        # driver, which must finish a trial before the GP picks the next one. So
        # submit the whole remainder as one flat batch and let the pool stay
        # saturated; a per-point batch of --n_episodes would idle most workers.
        specs = [(i, ep) for i in todo for ep in range(args.n_episodes)]
        jobs  = [self._episode_job(i, ep) for i, ep in specs]
        by_point: dict[int, list] = {i: [] for i in todo}

        print(f"\n  running {len(todo)} point(s) x {args.n_episodes} episodes "
              f"= {len(jobs)} episodes")

        def report(idx: int, result) -> None:
            i, ep = specs[idx]
            t, s  = self.points[i]
            tick  = "✓" if result.success else "✗"
            sstr  = (f"step {result.steps_to_success}"
                     if result.steps_to_success is not None else "—")
            print(f"    T={t:<6g} sigma={s:<6g} ep {ep:02d}  {tick}  "
                  f"success_step={sstr:<9}  "
                  f"step={result.mean_step_ms:.3f}±{result.std_step_ms:.3f} ms")
            # Write each point the moment its last episode lands, rather than at
            # the end: that is what makes a killed cell resumable per point.
            by_point[i].append(result)
            if len(by_point[i]) == args.n_episodes:
                self._write_point(i, by_point[i])
                by_point[i] = []
                gc.collect()

        t0 = time.perf_counter()
        if self.pool is None:
            run_episodes_serially(jobs, on_result=report)
        else:
            self.pool.map_episodes(jobs, on_result=report)
        print(f"\n  grid finished in {(time.perf_counter() - t0)/60:.1f} min")

        return self._ranked()

    def _load_summary(self, i: int, path: Path) -> None:
        """Fold an already-written cell back into the ranking on resume."""
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! could not read {path.name} ({exc}); it will not be ranked")
            return
        t, s = self.points[i]
        self.summaries.append({
            "combo_index":           rec.get("combo_index", self.cell_id(i)),
            "label":                 rec.get("label", ""),
            "success_rate":          rec.get("success_rate"),
            "n_success":             rec.get("n_success"),
            "mean_steps_to_success": rec.get("mean_steps_to_success"),
            "mean_step_ms":          rec.get("mean_step_ms"),
            "mean_elapsed_s":        rec.get("mean_elapsed_s"),
            "temperature":           t,
            "noise_sigma":           s,
            "grid_point":            i,
            **self.cell_axes,
        })

    def _ranked(self) -> list[dict]:
        """Summaries sorted best first: success rate desc, then steps asc.

        A point with no successes has mean_steps_to_success=None; it sorts last
        among its success-rate group rather than first, which a bare None would.
        """
        def key(s):
            steps = s.get("mean_steps_to_success")
            return (-(s.get("success_rate") or 0.0),
                    float("inf") if steps is None else steps)
        return sorted(self.summaries, key=key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- episode settings: names mirror run_eval_episode.py exactly, so a
    #     winning point replays by copy-pasting the flags into that driver ---
    p.add_argument("--task",        type=str, default="grasp_reorient")
    p.add_argument("--geometry",    type=str, default="cube_high_high",
                   help="Scene variant: '<object>' or '<object>_<hand_acc>_<obj_acc>'. "
                        "The object axis rides in here — there is no --object flag.")
    p.add_argument("--model",       type=str, default="M2", choices=list(MODEL_FACTORIES),
                   help="Contact model for this cell. Singular on purpose: the SLURM "
                        "script gives each array task one model and one object, and "
                        "this script grids the two planner knobs inside that cell.")
    p.add_argument("--planner",     type=str, default="mppi", choices=PLANNER_NAMES,
                   help="--temperature is an MPPI-only field, so a temperature grid "
                        "under cem/predictive_sampler would silently collapse to one "
                        "distinct point (make_planner_config drops unknown fields).")
    p.add_argument("--n_samples",   type=int, default=256,
                   help="Rollouts per optimizer iteration. A CELL axis, not a grid "
                        "axis: the SLURM script gives each array task one value, so "
                        "the per-episode cost (and VRAM) is constant within a cell.")
    p.add_argument("--horizon",     type=int, default=None,
                   help="Planning horizon in control steps (ignored with --time_horizon).")
    p.add_argument("--time_horizon", type=float, default=0.352,
                   help="Planning horizon in SECONDS; quantized down to whole control steps.")
    p.add_argument("--step_time",   type=float, default=0.064,
                   help="Control-step duration in SECONDS; quantized down to whole rollout steps.")
    p.add_argument("--n_iterations", type=int, default=None,
                   help="Optimizer iterations per plan() (default: the planner's own). "
                        "A CELL axis, not a grid axis: the SLURM script gives each "
                        "array task one value. Ignored under --convergence_tol.")
    p.add_argument("--convergence_tol", type=float, default=None,
                   help="Switch MPPI to convergence-terminated iteration: keep "
                        "iterating until the returned action settles — "
                        "sum_u (u_i - u_i+1)^2 < tol — instead of running a fixed "
                        "--n_iterations, which is then ignored. Always runs >= 2 "
                        "iterations and at most --max_iterations. Omit for "
                        "fixed-iteration MPPI.")
    p.add_argument("--max_iterations", type=int, default=None,
                   help="Iteration cap for --convergence_tol (default: 10).")
    p.add_argument("--substeps",    type=int, default=None,
                   help="Rollout substeps per control step (control frequency knob).")
    p.add_argument("--eval_substeps", type=int, default=None,
                   help="Eval steps per rollout step (default: task config).")
    p.add_argument("--delta",       type=float, default=None,
                   help="Per-step delta clip magnitude; omit to disable the clamp.")
    p.add_argument("--resample_interval", type=int, default=1,
                   help="Plan steps between noise resamples (1=every step).")
    p.add_argument("--warm_start",  action=argparse.BooleanOptionalAction, default=False,
                   help="Shift the planned sequence forward one control step after each plan().")
    p.add_argument("--use_full_graph", action=argparse.BooleanOptionalAction, default=True,
                   help="Single mega CUDA graph per plan() (default) vs step+reset graphs.")
    p.add_argument("--nconmax",     type=int, default=50)
    p.add_argument("--njmax",       type=int, default=300)
    p.add_argument("--eval_sim",    type=str, default="none",
                   choices=["none", "mujoco", "drake", "pinocchio"],
                   help="Eval simulator: 'none' uses the task default, else override it.")
    p.add_argument("--settle",      type=float, default=1.0)
    p.add_argument("--n_episodes",  type=int, default=5,
                   help="Episodes per grid point. More episodes average out "
                        "episode-specific luck at a proportional cost in wall time.")
    p.add_argument("--seed",        type=int, default=0,
                   help="THE constant episode seed: the same episodes are replayed for "
                        "every grid point, so points differ only in their knobs.")
    p.add_argument("--n_workers",   type=int, default=None,
                   help="Episodes to run concurrently, each in its own process with "
                        "its own planner and eval sim. THIS IS THE VRAM KNOB: every "
                        "worker holds its own CUDA context, MJWarp Data at "
                        "--n_samples worlds and captured graphs, so lower it if a "
                        "worker dies with an OOM. Default: the cores this process may "
                        "use, capped by the episode count. Results are unchanged "
                        "either way — episodes are pure functions of their seed. "
                        "1 runs them in-process. Worth ~n_workers-fold on a CPU-bound "
                        "eval sim (pinocchio) and ~nothing on a GPU-bound one "
                        "(mujoco), where planning is already 89%% of wall time.")

    # --- the grid -----------------------------------------------------------
    p.add_argument("--temperatures", type=str, default="40,20,10",
                   help="MPPI temperatures to grid over, comma- or space-separated. "
                        "Lambda divides the cost in the softmax weighting, so the "
                        "useful value scales with the task's cost magnitude — which "
                        "is why this is worth gridding per (model, object).")
    p.add_argument("--noise_sigmas", type=str, default="0.2,0.1,0.05",
                   help="Action-noise standard deviations to grid over. Pass a single "
                        "value to collapse this to a 1-D temperature sweep.")
    p.add_argument("--weights",     nargs="*", default=[],
                   help="Fixed cost-weight overrides for the WHOLE grid, as name=value "
                        "tokens (e.g. --weights w_quat=20 w_pos_x=7.5). Weights are not "
                        "a grid axis here; that is what param_search.slurm is for.")

    # --- output -------------------------------------------------------------
    p.add_argument("--array_index", type=int, default=0,
                   help="This cell's SLURM_ARRAY_TASK_ID. Only used to offset the cell "
                        "ids (id = array_index * n_points + point), so every array task "
                        "can write into ONE shared --outdir without colliding and a "
                        "single combine_results.py run merges the lot.")
    p.add_argument("--outdir",      type=str, default=None,
                   help="Directory for cell_*.json (auto-named under results/ if omitted). "
                        "Shared by every array task of a submission.")
    add_record_flags(p)
    p.add_argument("--debug",       action="store_true")
    return p


def parse_overrides(tokens: list[str], defaults: dict[str, float]) -> dict[str, float]:
    """`name=value` tokens -> {name: float}, validated against the task."""
    overrides: dict[str, float] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"bad --weights token {tok!r}; expected name=value")
        name, val = tok.split("=", 1)
        name = name.strip()
        if name not in defaults:
            raise ValueError(
                f"--weights {name!r} is not a cost weight of this task; "
                f"expected one of {', '.join(defaults)}"
            )
        overrides[name] = float(val)
    return overrides


def main():
    args = build_parser().parse_args()

    planner      = resolve_planner_name(args.planner)
    temperatures = parse_values(args.temperatures, "--temperatures")
    noise_sigmas = parse_values(args.noise_sigmas, "--noise_sigmas")
    # Temperature outermost so a 1-D sigma collapse leaves the points in
    # temperature order — the order the ranked table and the cell ids follow.
    points = [(t, s) for t in temperatures for s in noise_sigmas]

    if planner != "mppi" and len(temperatures) > 1:
        raise ValueError(
            f"--planner {planner} has no temperature field, so a "
            f"{len(temperatures)}-value --temperatures grid would run the same "
            f"configuration that many times; use --planner mppi"
        )

    # convergence_tol/max_iterations are MPPIConfig-only, and make_planner_config
    # DROPS fields the selected config does not declare — so under another planner
    # the whole cell would silently run plain fixed-iteration instead. Both checks
    # (and MPPIConfig's own max_iterations >= 2 rule) fire here, before wp.init and
    # the worker pool, rather than inside every spawned episode.
    if args.convergence_tol is not None:
        if planner != "mppi":
            raise ValueError(
                f"--convergence_tol is an MPPI-only field, so --planner {planner} "
                f"would silently ignore it and run fixed-iteration; use --planner mppi"
            )
        if args.convergence_tol <= 0.0:
            raise ValueError(
                f"--convergence_tol must be > 0, got {args.convergence_tol:g}")
        if args.max_iterations is not None and args.max_iterations < 2:
            raise ValueError(
                "--convergence_tol needs --max_iterations >= 2: the test compares "
                f"two consecutive updates, got {args.max_iterations}"
            )
        if args.n_iterations is not None:
            print(f"  ! --n_iterations {args.n_iterations} is IGNORED: "
                  f"--convergence_tol {args.convergence_tol:g} decides the "
                  f"iteration count per plan() call")
    elif args.max_iterations is not None:
        print(f"  ! --max_iterations {args.max_iterations} is IGNORED without "
              f"--convergence_tol; it only caps the convergence loop")

    wp.init()

    peek_task       = load_rollout_task(args.task, args.geometry)
    default_weights = dict(peek_task.config.cost_weights)
    overrides       = parse_overrides(args.weights, default_weights)

    schedule = resolve_mppi_schedule(
        MPPIConfig(time_horizon=args.time_horizon, step_time=args.step_time),
        peek_task.config, args.eval_substeps,
    )
    horizon, substeps, rollout_dt = schedule

    obj    = SceneVariant.parse(args.geometry).obj
    outdir = Path(args.outdir) if args.outdir else (
        RESULTS_DIR / f"temp_sigma_grid_{args.task}_"
                      f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  Temperature x noise_sigma grid — task={args.task}  "
          f"model={args.model}  object={obj}  planner={planner}")
    print(f"  eval_sim={args.eval_sim}  geometry={args.geometry}  "
          f"n_episodes={args.n_episodes}  seed={args.seed} (constant)")
    # Resolved once here and recorded in grid_summary: a flag left off means
    # "the planner config's own default", and the summary should say which.
    if args.convergence_tol is not None:
        n_iter_used = None
        cap_used    = (args.max_iterations if args.max_iterations is not None
                       else planner_default(planner, "max_iterations"))
        print(f"  n_samples={args.n_samples}  iteration: converge to "
              f"{args.convergence_tol:g}, cap {cap_used} (fixed for this cell)")
    else:
        cap_used    = None
        n_iter_used = (args.n_iterations if args.n_iterations is not None
                       else planner_default(planner, "n_iterations"))
        print(f"  n_samples={args.n_samples}  n_iterations={n_iter_used} "
              f"(fixed for this cell)")
    print(f"  rollout_dt={rollout_dt*1e3:.3f}ms  step_time={args.step_time:g}s -> "
          f"{substeps} substeps  time_horizon={args.time_horizon:g}s -> {horizon} steps")
    print(f"  temperatures ({len(temperatures)}): "
          f"{', '.join(f'{t:g}' for t in temperatures)}")
    print(f"  noise_sigmas ({len(noise_sigmas)}): "
          f"{', '.join(f'{s:g}' for s in noise_sigmas)}")
    print(f"  grid: {len(points)} points x {args.n_episodes} episodes = "
          f"{len(points) * args.n_episodes} episodes")
    if overrides:
        print(f"  fixed weight overrides: {overrides}")
    print(f"  ranked by success rate, ties broken by mean steps-to-success")
    print(f"  cell ids: {args.array_index} * {len(points)} + point "
          f"-> cell_{args.array_index * len(points):05d}.json ...")
    print(f"  outdir: {outdir}")
    print(f"{'='*70}")

    # --- episode fan-out ---------------------------------------------------
    # Each worker stands up its own planner (own MJWarp Data at nworld=n_samples
    # and its own captured CUDA graphs) plus its own eval sim, so VRAM scales
    # with the worker count. Report the headroom rather than discovering it as
    # an OOM halfway through the grid.
    n_jobs    = len(points) * args.n_episodes
    n_workers = (args.n_workers if args.n_workers is not None
                 else default_worker_count(n_jobs))
    n_workers = max(1, min(n_workers, n_jobs))
    free_gb   = _device_free_gb()
    if free_gb is not None:
        per = free_gb / n_workers
        print(f"  workers: {n_workers} (of {n_jobs} episodes)  "
              f"free VRAM {free_gb:.1f} GiB -> {per:.1f} GiB/worker")
        if per < MIN_VRAM_PER_WORKER_GB:
            print(f"  ! under {MIN_VRAM_PER_WORKER_GB:g} GiB per worker; lower "
                  f"--n_workers or --n_samples if a worker dies with an OOM")
    else:
        print(f"  workers: {n_workers} (of {n_jobs} episodes)")

    pool = EpisodePool(n_workers) if n_workers > 1 else None
    grid = GridSearch(args, points, default_weights, overrides, schedule,
                      outdir, pool=pool)

    t0 = time.perf_counter()
    try:
        ranked = grid.run()
    finally:
        # In a finally so a Ctrl-C or a raised point does not leave workers
        # parked on the GPU.
        if pool is not None:
            pool.shutdown()
    elapsed = time.perf_counter() - t0

    # --- summary -----------------------------------------------------------
    summary_path = outdir / f"grid_summary_{args.array_index:03d}.json"
    json_io.dump({
        "task":         args.task,
        "model":        args.model,
        "object":       obj,
        "geometry":     args.geometry,
        "planner":      planner,
        "eval_sim":     args.eval_sim,
        "temperatures": temperatures,
        "noise_sigmas": noise_sigmas,
        "n_points":     len(points),
        "n_episodes":   args.n_episodes,
        "seed":         args.seed,
        "settle":       args.settle,
        "array_index":  args.array_index,
        "overrides":    overrides,
        "mppi": {
            "n_samples":       args.n_samples,
            "n_iterations":    n_iter_used,
            "convergence_tol": args.convergence_tol,
            "max_iterations":  cap_used,
            "time_horizon": args.time_horizon,
            "step_time":    args.step_time,
            "step_horizon": horizon,
            "step_substeps": substeps,
            "rollout_dt":   rollout_dt,
        },
        "elapsed_seconds": elapsed,
        "ranked":          ranked,
    }, summary_path, precision=TrajectoryConfig.from_args(args).precision)

    print(f"\n{'='*70}")
    print(f"  {args.model} / {obj} — {len(ranked)} points, {elapsed/60:.1f} min")
    print(f"  {'temperature':>12}  {'noise_sigma':>11}  {'succ%':>6}  "
          f"{'mean_steps':>10}  {'step_ms':>8}")
    print(f"  {'-'*12}  {'-'*11}  {'-'*6}  {'-'*10}  {'-'*8}")
    for s in ranked:
        steps = s.get("mean_steps_to_success")
        sstr  = f"{steps:.1f}" if steps is not None else "—"
        rate  = s.get("success_rate")
        rstr  = f"{rate*100:.1f}" if rate is not None else "—"
        ms    = s.get("mean_step_ms")
        mstr  = f"{ms:.3f}" if ms is not None else "—"
        print(f"  {s['temperature']:>12g}  {s['noise_sigma']:>11g}  {rstr:>6}  "
              f"{sstr:>10}  {mstr:>8}")

    if ranked:
        best = ranked[0]
        weight_flags = " ".join(f"{k}={v:g}" for k, v in overrides.items())
        print(f"\n  BEST  T={best['temperature']:g}  "
              f"sigma={best['noise_sigma']:g}  "
              f"success={(best.get('success_rate') or 0.0)*100:.1f}%")
        print(f"\n  Replay it with:")
        print(f"    python -m contact_study.drivers.run_eval_episode \\")
        print(f"        --task {args.task} --geometry {args.geometry} "
              f"--model {args.model} \\")
        print(f"        --planner {planner} --n_samples {args.n_samples} \\")
        print(f"        --time_horizon {args.time_horizon:g} "
              f"--step_time {args.step_time:g} \\")
        print(f"        --temperature {best['temperature']:g} "
              f"--noise_sigma {best['noise_sigma']:g} \\")
        if args.convergence_tol is not None:
            cap = ("" if args.max_iterations is None
                   else f" --max_iterations {args.max_iterations}")
            print(f"        --convergence_tol {args.convergence_tol:g}{cap} \\")
        elif args.n_iterations is not None:
            print(f"        --n_iterations {args.n_iterations} \\")
        print(f"        --seed {args.seed} --settle {args.settle:g}"
              + (f" \\\n        --weights {weight_flags}" if weight_flags else ""))
    print(f"\n  Saved -> {outdir}")
    print(f"{'='*70}")

    return ranked


if __name__ == "__main__":
    # Required, not decorative: the pool spawns workers that re-import this module.
    main()
