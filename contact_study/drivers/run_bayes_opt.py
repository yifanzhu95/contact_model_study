"""Bayesian optimization of a task's cost weights and planner noise knobs.

Every hyperparameter search in this repo has so far been an exhaustive grid —
experiments/run_param_search.py builds an itertools.product over a module-level
WEIGHT_SEARCH_SPACE, and experiments/hpc/param_search.slurm mixed-radix-decodes
$SLURM_ARRAY_TASK_ID over eleven axes. At ~2 minutes per grasp_reorient episode
a grid over even eight cost weights plus noise_sigma and temperature is not
affordable. This driver spends a fixed evaluation budget adaptively instead,
using scikit-optimize's Gaussian-process minimizer (skopt.gp_minimize).

Search space (all log-uniform by default):
  * the cost weights named by --opt_weights, bracketed multiplicatively around
    the task's own defaults, so the space auto-adapts per task/object;
    --no-opt_cost_weights drops them entirely, pinning every weight to the
    task's default and tuning only the planner knobs below;
  * noise_sigma, the sampling planner's action-noise standard deviation;
  * temperature, MPPI's softmax sharpness (dropped for planners that have no
    such field — make_planner_config filters by the config's declared fields).
    With several --models, --per_model_temperature splits this one dimension
    into temperature_<model>, one per contact model over the same range, so a
    model whose costs live on a different scale can get its own sharpness while
    the cost weights stay shared.

Objective (minimized):

    J = -(w_success * success_rate) + w_cost * mean_normalized_goal_error

Both terms live in [0, 1], so --w_success and --w_cost are directly comparable.
The goal-error term comes from BaseTask.goal_errors() on each episode's final
state, normalized by the task's own success_thresholds and clipped — NOT from
EpisodeResult.final_cost, which is ||q_final - q_0||, the displacement from the
START pose, and would reward never moving the object.

Seeding: unlike experiments/hpc/run_param_cell.py, which folds the cell id into
the seed so no two cells share one, this driver derives the per-episode seeds
ONCE from --seed and reuses them for every trial. Combined with a deterministic
initial state that makes the objective a pure function of the hyperparameters,
so two trials differ only by the vector under test. The trade-off is that the
search can overfit that one episode; --n_episodes buys generalization with
compute. --bo_seed is a separate knob for gp_minimize's own random_state.

Every trial writes results/<outdir>/cell_<id>.json in the same schema as
run_param_cell.py, so analysis/param_search_to_csv_dir.py works on the output
directory unchanged. bo_state.json is rewritten after each trial and is what
--resume reads back.

Runs headless on a CUDA machine and never records video (a GL context per
episode is exactly what sweeps avoid):

    python -m contact_study.drivers.run_bayes_opt --task grasp_reorient \
        --models M2 --n_calls 60 --n_episodes 1

Pass several models to search for one weight set that holds across all of them.
Each trial then runs n_models x --n_episodes episodes at identical seeds, up to
--n_workers of them at once:

    python -m contact_study.drivers.run_bayes_opt --task grasp_reorient \
        --models M1 M2 M3 M4 --model_agg worst --n_episodes 4 --n_workers 8

Add --per_model_temperature to let each of those models tune its own MPPI
temperature (same bounds, one dimension each) while sharing the weight vector.
"""

from __future__ import annotations

import os
# This driver never renders, but MuJoCo still picks a GL backend at import time;
# match the sweep workers and default to EGL before warp/mujoco are imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import warp as wp

from skopt import gp_minimize
from skopt.space import Real

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
    PLANNER_NAMES, make_planner_config, resolve_planner_name,
)
from contact_study.planners.mppi import MPPIConfig
from contact_study.tasks.config import DEFAULT_SCENE_VARIANT, EvalSimulatorKind

RESULTS_DIR = Path(__file__).parents[2] / "results"

# The weights param_search.slurm actually sweeps — a sensible default subset of
# grasp_reorient's twelve. Names absent from the task are silently dropped (see
# resolve_opt_weights), so this stays a *default*, not a requirement.
DEFAULT_OPT_WEIGHTS = (
    "w_quat", "w_pos_x", "w_pos_y", "w_pos_z",
    "w_contact", "w_joint", #"w_quat_term", "w_pos_term",
)

# Default weight bounds are [d / BOUND_SCALE, d * BOUND_SCALE] around the task's
# own default d — a multiplicative bracket, which is what a log-uniform prior
# wants and what makes the space transfer between objects with different scales.
BOUND_SCALE = 4.0

# Rough floor for one worker's share of VRAM: a CUDA context plus an MJWarp
# Model/Data at nworld=n_samples plus the (N, H, nu) sample block. Only used to
# warn — the real cost depends on --n_samples/--nconmax/--njmax.
MIN_VRAM_PER_WORKER_GB = 1.5


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def resolve_opt_weights(tokens: list[str] | None, defaults: dict[str, float]) -> list[str]:
    """Which cost weights to optimize, as bare names (bounds parsed separately).

    With no --opt_weights, prefer DEFAULT_OPT_WEIGHTS restricted to the ones this
    task actually declares; if that leaves nothing (a task with entirely
    different weight names), fall back to every weight with a nonzero default,
    since a zero default has no multiplicative bracket to build bounds from.
    """
    if tokens:
        return [tok.split(":", 1)[0].strip() for tok in tokens]
    names = [n for n in DEFAULT_OPT_WEIGHTS if n in defaults]
    if names:
        return names
    return [n for n, v in defaults.items() if float(v) > 0.0]


def parse_weight_specs(tokens: list[str] | None,
                       defaults: dict[str, float]) -> list[tuple[str, float, float]]:
    """Parse `name` or `name:lo:hi` tokens into (name, lo, hi) triples.

    A bare name brackets the task's own default multiplicatively. A zero default
    has no such bracket, so those must carry explicit bounds.
    """
    tokens = list(tokens) if tokens else [n for n in resolve_opt_weights(None, defaults)]
    specs: list[tuple[str, float, float]] = []
    seen: set[str] = set()

    for tok in tokens:
        parts = tok.split(":")
        name  = parts[0].strip()
        if name not in defaults:
            raise ValueError(
                f"--opt_weights {name!r} is not a cost weight of this task; "
                f"expected one of {', '.join(defaults)}"
            )
        if name in seen:
            raise ValueError(f"--opt_weights lists {name!r} twice")
        seen.add(name)

        if len(parts) == 3:
            lo, hi = float(parts[1]), float(parts[2])
        elif len(parts) == 1:
            d = float(defaults[name])
            if d <= 0.0:
                raise ValueError(
                    f"cost weight {name!r} defaults to {d:g}, which has no "
                    f"multiplicative bracket; give explicit bounds as "
                    f"{name}:lo:hi"
                )
            lo, hi = d / BOUND_SCALE, d * BOUND_SCALE
        else:
            raise ValueError(
                f"bad --opt_weights token {tok!r}; expected 'name' or 'name:lo:hi'"
            )
        if lo >= hi:
            raise ValueError(f"--opt_weights {name!r}: lo must be < hi, got {lo}:{hi}")
        specs.append((name, lo, hi))

    if not specs:
        raise ValueError("no cost weights to optimize; pass --opt_weights explicitly")
    return specs


def _real(lo: float, hi: float, name: str) -> Real:
    """A search dimension, log-uniform when the range is strictly positive.

    Weights and sigmas span orders of magnitude, so log-uniform is the right
    prior; a range touching zero cannot be logged and falls back to uniform.
    """
    prior = "log-uniform" if lo > 0.0 else "uniform"
    return Real(lo, hi, prior=prior, name=name)


def build_space(specs: list[tuple[str, float, float]],
                args) -> tuple[list[Real], list[str]]:
    """(dimensions, names) for gp_minimize — weights, then the planner knobs."""
    dims  = [_real(lo, hi, name) for name, lo, hi in specs]
    names = [name for name, _, _ in specs]

    if args.opt_noise_sigma:
        lo, hi = args.noise_sigma_range
        dims.append(_real(lo, hi, "noise_sigma"))
        names.append("noise_sigma")
    if args.opt_temperature:
        lo, hi = args.temperature_range
        if args.per_model_temperature and len(args.models) > 1:
            # One dimension per contact model, all sharing --temperature_range.
            # The weights still have to hold across the models; only MPPI's
            # softmax sharpness is allowed to differ, since a temperature that
            # suits one model's cost scale can be badly wrong for another's.
            # Costs the GP len(models) - 1 extra dimensions, not extra episodes.
            for m in args.models:
                dims.append(_real(lo, hi, f"temperature_{m}"))
                names.append(f"temperature_{m}")
        else:
            dims.append(_real(lo, hi, "temperature"))
            names.append("temperature")

    return dims, names


def model_temperatures(params: dict[str, float], models: list[str],
                       fixed: float) -> dict[str, float]:
    """Each model's temperature: its own dimension, the shared one, or --temperature.

    Single lookup order for every caller, so the per-model and shared layouts
    stay interchangeable everywhere downstream.
    """
    shared = params.get("temperature", fixed)
    return {m: params.get(f"temperature_{m}", shared) for m in models}


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def normalized_goal_error(errs: dict[str, float] | None,
                          thresholds: dict[str, float],
                          clip: float) -> float | None:
    """Scalarize per-criterion goal errors into [0, 1]; None when unavailable.

    Each term is divided by its own success threshold, so it reads ~1 at the
    success boundary and the sum is dimensionless regardless of units. The clip
    keeps one catastrophic episode (object dropped through the floor) from
    dominating the GP's view of the whole space.
    """
    if not errs or not thresholds:
        return None
    keys = [k for k in thresholds if k in errs and thresholds[k] > 0.0]
    if not keys:
        return None
    e = sum(float(errs[k]) / float(thresholds[k]) for k in keys)
    return min(e, clip) / clip


def combo_label(model_key: str, axes: dict[str, float]) -> str:
    """Short label encoding model + every search axis (run_param_cell style)."""
    parts = []
    for k, v in axes.items():
        short = k[2:] if k.startswith("w_") else k
        parts.append(f"{short}={v:g}" if isinstance(v, (int, float)) else f"{short}={v}")
    return f"{model_key}__" + "_".join(parts)


def _device_free_gb() -> float | None:
    """Free memory on the warp CUDA device, in GiB; None if unavailable."""
    try:
        return wp.get_device().free_memory / 1024 ** 3
    except Exception:
        return None


class BOObjective:
    """gp_minimize's objective: one hyperparameter vector -> one scalar.

    Holds the per-episode seeds (computed ONCE, so every trial evaluates the
    same episodes), the trial counter, and the output directory.
    """

    def __init__(self, args, dim_names: list[str], default_weights: dict[str, float],
                 thresholds: dict[str, float], schedule: tuple[int, int, float],
                 outdir: Path, start_trial: int = 0,
                 pool: EpisodePool | None = None):
        self.args            = args
        # None means run the episodes in this process (--n_workers 1). Owned by
        # main(), which shuts it down, so the objective can stay a pure callable.
        self.pool            = pool
        self.dim_names       = dim_names
        self.default_weights = default_weights
        self.thresholds      = thresholds
        self.horizon, self.substeps, self.rollout_dt = schedule
        self.outdir          = outdir
        self.trial           = start_trial

        self.planner     = resolve_planner_name(args.planner)
        # One config per model. Every trial runs its weight vector against all of
        # them; the per-model scores are folded by --model_agg.
        self.contact_cfgs = {k: MODEL_FACTORIES[k]() for k in args.models}
        self.model_tag    = "+".join(args.models)
        self.eval_sim    = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)
        self.delta       = (-args.delta, args.delta) if args.delta is not None else (None, None)

        # THE constant seed. Spawned once and reused by every trial, so the
        # objective is a function of the hyperparameters alone.
        self.episode_seeds = np.random.SeedSequence(args.seed).spawn(args.n_episodes)
        self.record_cfg    = TrajectoryConfig.from_args(args)

        # Kept for the whole run (the winner is looked up in here at the end), so
        # the per-episode trajectories are stripped before appending: at
        # --n_calls 500 with recording on they would be tens of GB of live heap.
        # The on-disk cell_*.json keeps the full record.
        self.records: list[dict] = []
        self._warned_no_goal_err = False

    # -- one episode's job description --------------------------------------

    def _episode_job(self, model_key: str, ep: int, overrides: dict,
                     noise_sigma: float, temperature: float) -> dict:
        """run_eval_episode's kwargs for episode `ep`, as picklable plain data.

        `rng` is replaced by `seed_seq`: the worker rebuilds the Generator with
        np.random.default_rng(seed_seq). SeedSequence is stateless, so deriving
        the planner seed here and the Generator there yields exactly what the
        old in-process `np.random.default_rng(self.episode_seeds[ep])` did.

        The seed is keyed by EPISODE ONLY, never by model: every model must be
        handed the identical initial states, goals and planner noise, or a
        per-model score difference could not be attributed to the contact model.
        """
        args = self.args
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
        # No CLI default: omitting it lets each planner's own default stand
        # (1 for mppi/predictive_sampler, 3 for cem).
        if args.n_iterations is not None:
            planner_kwargs["n_iterations"] = args.n_iterations

        return dict(
            task_name             = args.task,
            geometry              = args.geometry,
            contact_cfg           = self.contact_cfgs[model_key],
            planner               = self.planner,
            planner_cfg           = make_planner_config(self.planner, **planner_kwargs),
            seed_seq              = self.episode_seeds[ep],
            video_path            = None,
            cost_weight_overrides = overrides or None,
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

    def _score(self, results: list) -> dict:
        """One model's episodes -> its success rate, goal error and objective.

        Same scalarization the single-model driver always used; it is applied
        per model now so the models can be compared and folded.
        """
        args = self.args
        n_success    = sum(r.success for r in results)
        success_rate = n_success / len(results) if results else 0.0

        errs = [normalized_goal_error(r.final_goal_errs, self.thresholds, args.err_clip)
                for r in results]
        errs = [e for e in errs if e is not None]
        mean_err = float(np.mean(errs)) if errs else None

        objective = -(args.w_success * success_rate)
        if mean_err is not None:
            objective += args.w_cost * mean_err

        return {
            "n_episodes":         len(results),
            "n_success":          n_success,
            "success_rate":       success_rate,
            "mean_norm_goal_err": mean_err,
            "objective":          objective,
        }

    # -- one trial ----------------------------------------------------------

    def __call__(self, x: list[float]) -> float:
        args     = self.args
        trial    = self.trial
        self.trial += 1

        params    = {name: float(v) for name, v in zip(self.dim_names, x)}
        # Emit overrides in the task's own key order for a stable label; the
        # merge itself is order-insensitive (apply_cost_weight_overrides
        # re-reads config.cost_weights' order), but the label should be stable.
        overrides = {k: params[k] for k in self.default_weights if k in params}
        knobs     = {k: v for k, v in params.items() if k not in self.default_weights}
        axes      = {**overrides, **knobs}
        label     = combo_label(self.model_tag, axes)

        noise_sigma = params.get("noise_sigma", args.noise_sigma)
        temps       = model_temperatures(params, list(args.models), args.temperature)
        # A single value unless --per_model_temperature split the dimension.
        shared_temp = (next(iter(temps.values()))
                       if len(set(temps.values())) == 1 else None)

        print(f"\n[trial {trial:03d}]  {label}")
        tstr = (f"{shared_temp:.5g}" if shared_temp is not None else
                "  ".join(f"{m}={t:.5g}" for m, t in temps.items()))
        print(f"  noise_sigma={noise_sigma:.5g}  temperature={tstr}  "
              f"n_episodes={args.n_episodes}")

        # The trial's full workload: this weight vector against every model, on
        # every episode seed. Flat, because they are all independent and a flat
        # list keeps every worker busy — a per-model batch would idle workers
        # whenever one model finished its episodes early.
        specs = [(m, ep) for m in args.models for ep in range(args.n_episodes)]
        # Plain picklable data (dataclasses, dicts, a SeedSequence), which is
        # what lets an episode cross into a worker — see drivers/episode_pool.py.
        jobs  = [self._episode_job(m, ep, overrides, noise_sigma, temps[m])
                 for m, ep in specs]

        def report(i: int, result) -> None:
            """Print an episode as it lands. Out of order when workers race."""
            model, ep = specs[i]
            tick = "✓" if result.success else "✗"
            sstr = f"step {result.steps_to_success}" if result.steps_to_success is not None else "—"
            gerr = normalized_goal_error(result.final_goal_errs, self.thresholds,
                                         args.err_clip)
            gstr = f"{gerr:.4f}" if gerr is not None else "n/a"
            print(f"    {model:<3} ep {ep:02d}  {tick}  success_step={sstr:<9}  "
                  f"goal_err={gstr}  "
                  f"step={result.mean_step_ms:.3f}±{result.std_step_ms:.3f} ms")

        t0 = time.perf_counter()
        if self.pool is None:
            episodes = run_episodes_serially(jobs, on_result=report)
        else:
            episodes = self.pool.map_episodes(jobs, on_result=report)

        # -- scalarize ------------------------------------------------------
        # Score each model on its own episodes first, then fold. Scoring the
        # pooled episodes instead would let a model that fails every episode be
        # hidden by one that succeeds on all of them — the opposite of looking
        # for weights that hold across models.
        by_model  = {m: [] for m in args.models}
        for (m, _), r in zip(specs, episodes):
            by_model[m].append(r)

        per_model = {m: self._score(rs) for m, rs in by_model.items()}

        if any(v["mean_norm_goal_err"] is None for v in per_model.values()) \
                and not self._warned_no_goal_err:
            self._warned_no_goal_err = True
            print(f"  ! task {args.task!r} has no goal_errors(); optimizing pure "
                  f"success rate (the --w_cost term is inactive)")

        scores = [v["objective"] for v in per_model.values()]
        # 'worst' is a max because the objective is MINIMIZED: the largest
        # per-model objective is the model doing worst under these weights.
        objective = float(np.mean(scores)) if args.model_agg == "mean" else float(max(scores))

        # Reported (and written to the cell record) pooled over every episode of
        # every model, so the CSV columns keep their old meaning. The objective
        # above is what the GP actually sees.
        n_success    = sum(r.success for r in episodes)
        success_rate = n_success / len(episodes)
        errs         = [v["mean_norm_goal_err"] for v in per_model.values()
                        if v["mean_norm_goal_err"] is not None]
        mean_err     = float(np.mean(errs)) if errs else None

        succ_steps = [r.steps_to_success for r in episodes if r.steps_to_success is not None]
        step_ms    = [r.mean_step_ms for r in episodes]

        print(f"  → objective={objective:+.5f} ({args.model_agg} of {len(args.models)} "
              f"model{'s' if len(args.models) > 1 else ''})  "
              f"success={success_rate*100:.1f}% ({n_success}/{len(episodes)})  goal_err="
              f"{mean_err if mean_err is None else round(mean_err, 4)}  "
              f"[{time.perf_counter() - t0:.1f}s]")
        if len(args.models) > 1:
            for m, v in per_model.items():
                gstr = ("n/a" if v["mean_norm_goal_err"] is None
                        else f"{v['mean_norm_goal_err']:.4f}")
                print(f"        {m:<3} objective={v['objective']:+.5f}  "
                      f"success={v['success_rate']*100:5.1f}%  goal_err={gstr}")

        record = {
            # -- run_param_cell.py schema, so param_search_to_csv_dir.py works --
            "combo_index":           trial,
            "task":                  args.task,
            # A joined tag ("M1+M2") so analysis/bayes_opt_to_csv_dir.py keeps a
            # scalar model column; `models` and `per_model` carry the detail.
            "model":                 self.model_tag,
            "models":                list(args.models),
            "model_agg":             args.model_agg,
            "per_model":             per_model,
            "label":                 label,
            "overrides":             overrides,
            "axes":                  axes,
            "swept_knobs":           list(knobs),
            "full_weights":          {**self.default_weights, **overrides},
            "n_episodes":            len(episodes),
            "n_episodes_per_model":  args.n_episodes,
            "n_success":             n_success,
            "success_rate":          success_rate,
            "mean_steps_to_success": float(np.mean(succ_steps)) if succ_steps else None,
            "mean_step_ms":          float(np.mean(step_ms)),
            "std_step_ms":           float(np.mean([r.std_step_ms for r in episodes])),
            "mean_elapsed_s":        float(np.mean([r.elapsed_seconds for r in episodes])),
            "mppi": {
                "n_samples":         args.n_samples,
                "time_horizon":      args.time_horizon,
                # Scalar as it always was, and None when the models were given
                # their own; `temperature_per_model` always carries the detail.
                "temperature":       shared_temp,
                "temperature_per_model": dict(temps),
                "noise_sigma":       noise_sigma,
                "step_time":         args.step_time,
                "step_horizon":      self.horizon,
                "step_substeps":     self.substeps,
                "rollout_dt":        self.rollout_dt,
                "delta":             args.delta,
                "resample_interval": args.resample_interval,
            },
            "seed":     args.seed,
            "eval_sim": args.eval_sim,
            "geometry": args.geometry,
            "settle":   args.settle,
            # -- BO-specific ---------------------------------------------------
            "planner":            self.planner,
            "objective":          objective,
            "mean_norm_goal_err": mean_err,
            "x":                  [float(v) for v in x],
            "episodes": [r.to_dict() for r in episodes],
        }
        self.records.append({**record, "episodes": None})
        # Written inside the objective so a killed job keeps every finished trial.
        json_io.dump(record, self.outdir / f"cell_{trial:05d}.json",
                     precision=self.record_cfg.precision)

        # Each trial builds a fresh rollout task, planner (with CUDA graphs) and
        # eval sim. Drop the references promptly and report free device memory so
        # a leak shows up in the log rather than as an OOM forty trials in.
        del episodes
        gc.collect()
        free_gb = _device_free_gb()
        if free_gb is not None:
            print(f"  gpu_free={free_gb:.2f} GiB")

        return float(objective)


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def save_state(path: Path, dim_names, dims, x_iters, func_vals, args) -> None:
    # `model` is no longer an argparse dest (it is an alias of `models`), but
    # analysis/bayes_opt_to_csv_dir.py reads args["model"] for its CSV column.
    # Emit the joined tag under the old key so the tooling keeps working.
    saved_args = dict(vars(args))
    saved_args.setdefault("model", "+".join(saved_args.get("models", [])))
    with open(path, "w") as f:
        json.dump({
            "dim_names":  list(dim_names),
            "dim_bounds": [[float(d.low), float(d.high), d.prior] for d in dims],
            "x_iters":    [[float(v) for v in x] for x in x_iters],
            "func_vals":  [float(v) for v in func_vals],
            "args":       saved_args,
        }, f, indent=2)


def load_state(path: Path, dim_names: list[str],
               dims: list[Real]) -> tuple[list[list[float]], list[float]]:
    """Prior (x0, y0) from a bo_state.json, refusing a mismatched search space."""
    with open(path) as f:
        state = json.load(f)
    prev = state.get("dim_names", [])
    if prev != list(dim_names):
        raise ValueError(
            f"--resume {path} was written for a different search space:\n"
            f"  checkpoint: {prev}\n  current:    {list(dim_names)}"
        )
    x0 = [[float(v) for v in x] for x in state.get("x_iters", [])]
    y0 = [float(v) for v in state.get("func_vals", [])]
    if len(x0) != len(y0):
        raise ValueError(f"--resume {path} has {len(x0)} x_iters but {len(y0)} func_vals")

    # The dimensions may have been re-bracketed since the checkpoint was
    # written. WIDENING one is fine -- every restored trial is still a legal
    # point and the GP simply refits over the larger box -- but say so, because
    # the restored trials then cover only a corner of the new space and the
    # rebased --n_initial_points may leave no random trials to explore the rest.
    # NARROWING strands the trials that fall outside; skopt would reject them
    # with a message that names neither the dimension nor the value, so catch it
    # here instead.
    for name, d, (lo, hi, prior) in zip(dim_names, dims,
                                        state.get("dim_bounds", [])):
        if (lo, hi, prior) != (float(d.low), float(d.high), d.prior):
            print(f"  ! --resume: {name} was [{lo:.5g}, {hi:.5g}] {prior} in the "
                  f"checkpoint, now [{d.low:.5g}, {d.high:.5g}] {d.prior}")
    for i, x in enumerate(x0):
        for name, d, v in zip(dim_names, dims, x):
            if not (float(d.low) <= v <= float(d.high)):
                raise ValueError(
                    f"--resume {path}: trial {i} has {name}={v:.6g}, outside the "
                    f"current bracket [{d.low:.5g}, {d.high:.5g}]. Narrowing a "
                    f"dimension strands the trials that fall outside it; widen "
                    f"it back, or start a fresh outdir without --resume."
                )
    return x0, y0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- episode settings: names mirror run_eval_episode.py exactly, so a
    #     winning trial replays by copy-pasting the flags into that driver ---
    p.add_argument("--task",        type=str, default="grasp_reorient")
    p.add_argument("--geometry",    type=str, default="cube_high_high",#DEFAULT_SCENE_VARIANT,
                   help="Scene variant: '<object>' or '<object>_<hand_acc>_<obj_acc>'.")
    p.add_argument("--models", "--model", dest="models", nargs="+", default=["M1","M2","M3","M4"],
                   choices=list(MODEL_FACTORIES), metavar="MODEL",
                   help="Contact model(s) to optimize over. With more than one, EVERY "
                        "trial evaluates the SAME weight vector on EVERY model, at the "
                        "same episode seeds, and --model_agg folds the per-model scores "
                        "into the one number the GP sees. The result is a weight set "
                        "that has to work across the models rather than one tuned to a "
                        "single contact model. Cost is proportional: a trial is "
                        "n_models x --n_episodes episodes, which --n_workers runs "
                        "concurrently.")
    p.add_argument("--model_agg",   type=str, default="mean", choices=["mean", "worst"],
                   help="How per-model objectives become the trial's score. 'mean' "
                        "optimizes average performance and lets a good model carry a "
                        "bad one; 'worst' (minimax) optimizes the worst model, which "
                        "is the stricter reading of 'weights that work for all of "
                        "them'. Ignored with a single --models entry.")
    p.add_argument("--planner",     type=str, default="mppi", choices=PLANNER_NAMES)
    p.add_argument("--n_samples",   type=int, default=256)#256)
    p.add_argument("--horizon",     type=int, default=None,
                   help="Planning horizon in control steps (ignored with --time_horizon).")
    p.add_argument("--time_horizon", type=float, default=0.352,
                   help="Planning horizon in SECONDS; quantized down to whole control steps.")
    p.add_argument("--step_time",   type=float, default=0.064,
                   help="Control-step duration in SECONDS; quantized down to whole rollout steps.")
    p.add_argument("--n_iterations", type=int, default=None,
                   help="Optimizer iterations per plan() (default: the planner's own).")
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
    p.add_argument("--n_episodes",  type=int, default=1,
                   help="Episodes per objective evaluation. More episodes average out "
                        "episode-specific luck at a proportional cost in wall time.")
    p.add_argument("--seed",        type=int, default=64,
                   help="THE constant episode seed: the same episodes are replayed for "
                        "every trial, so trials differ only in their hyperparameters.")
    p.add_argument("--n_workers",   type=int, default=None,
                   help="Episodes to run concurrently, each in its own process with "
                        "its own planner and eval sim. THIS IS THE VRAM KNOB: every "
                        "worker holds its own CUDA context, MJWarp Data at "
                        "--n_samples worlds and captured graphs, so lower it if a "
                        "worker dies with an OOM. Default: n_models x --n_episodes, "
                        "capped by the cores this process may use. The objective is "
                        "unchanged either way — episodes are pure functions of their "
                        "seed. 1 runs them in-process. Worth ~n_workers-fold on a "
                        "CPU-bound eval sim (pinocchio) and ~nothing on a GPU-bound "
                        "one (mujoco), where planning is already 89%% of wall time.")

    # --- search space -------------------------------------------------------
    p.add_argument("--opt_weights", nargs="*", default=[
                                                    "w_quat:1.0:50.0",
                                                    "w_pos_x:1.0:50.0",
                                                    "w_pos_y:1.0:50.0",
                                                    "w_pos_z:1.0:50.0",
                                                    "w_contact:0.1:100.0",
                                                    "w_joint:0.1:20.0",
                                                    "w_fallen:100.0:300.0",
                                                    "w_quat_term:100.0:300.0",
                                                    "w_pos_term:100.0:300.0"],
                   help="Cost weights to optimize, as 'name' (bounds bracketed "
                        f"multiplicatively at x{BOUND_SCALE:g} around the task's default) "
                        "or 'name:lo:hi'. The default above is grasp_reorient's "
                        "weight set with explicit brackets, so ANOTHER TASK NEEDS "
                        "THIS FLAG PASSED EXPLICITLY — an unknown name is a hard "
                        "error. Pass a bare 'name' to fall back to the "
                        f"multiplicative bracket, or the {' '.join(DEFAULT_OPT_WEIGHTS)} "
                        "subset for the old behavior.")
    p.add_argument("--opt_cost_weights", action=argparse.BooleanOptionalAction, default=True,
                   help="Search over the cost weights; --no-opt_cost_weights drops "
                        "every weight dimension and pins them to the task's own "
                        "defaults, leaving only the planner knobs (noise_sigma, "
                        "temperature) in the space. Cannot be combined with an "
                        "explicit --opt_weights, and needs at least one planner "
                        "knob left to search.")
    p.add_argument("--opt_noise_sigma", action=argparse.BooleanOptionalAction, default=False,
                   help="Search over noise_sigma; --no-opt_noise_sigma pins it to "
                        "--noise_sigma instead.")
    p.add_argument("--opt_temperature", action=argparse.BooleanOptionalAction, default=True,
                   help="Search over MPPI's temperature; --no-opt_temperature pins it to "
                        "--temperature instead.")
    p.add_argument("--per_model_temperature", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="With several --models, give each its OWN temperature "
                        "dimension (temperature_<model>) instead of one shared "
                        "value; all of them use --temperature_range. The cost "
                        "weights stay shared, so the search still looks for one "
                        "weight set that holds across the models — only MPPI's "
                        "softmax sharpness is allowed to differ. Adds "
                        "n_models - 1 dimensions and no extra episodes. Requires "
                        "--opt_temperature; ignored for a single model.")
    p.add_argument("--noise_sigma_range", type=float, nargs=2, default=(1e-3, 1.0),
                   metavar=("LO", "HI"))
    p.add_argument("--temperature_range", type=float, nargs=2, default=(0.001, 1000.0),
                   metavar=("LO", "HI"))
    p.add_argument("--noise_sigma", type=float, default=0.1,
                   help="Fixed noise_sigma when --no-opt_noise_sigma.")
    p.add_argument("--temperature", type=float, default=30.0,
                   help="Fixed temperature when --no-opt_temperature.")

    # --- objective ----------------------------------------------------------
    p.add_argument("--w_success",  type=float, default=1.0,
                   help="Weight on the success rate (maximized).")
    p.add_argument("--w_cost",     type=float, default=0.1,
                   help="Weight on the normalized final goal error (minimized).")
    p.add_argument("--err_clip",   type=float, default=500.0,
                   help="Goal error is clipped here then divided by it, mapping the "
                        "cost term into [0, 1] so it cannot be dominated by one "
                        "catastrophic episode.")

    # --- optimizer ----------------------------------------------------------
    p.add_argument("--n_calls",    type=int, default=250,
                   help="Total trial budget, INCLUDING any trials restored by --resume.")
    p.add_argument("--n_initial_points", type=int, default=10,
                   help="Random trials before the GP takes over (reduced by the number "
                        "of resumed trials).")
    p.add_argument("--acq_func",   type=str, default="gp_hedge",
                   choices=["EI", "LCB", "PI", "gp_hedge"])
    p.add_argument("--gp_noise",   type=str, default="gaussian",
                   help="gp_minimize's noise model: 'gaussian' (default) or a float. "
                        "At --n_episodes 1 the success term is a two-valued step "
                        "function, which a noiseless GP fits badly.")
    p.add_argument("--bo_seed",    type=int, default=0,
                   help="random_state for gp_minimize itself (distinct from --seed).")

    # --- output -------------------------------------------------------------
    p.add_argument("--outdir",     type=str, default=None,
                   help="Trial directory (auto-named under results/ if omitted).")
    p.add_argument("--resume",     type=str, default=None,
                   help="A bo_state.json to seed the optimizer with prior trials.")
    add_record_flags(p)
    p.add_argument("--debug",      action="store_true",
                   help="Verbose per-step diagnostics (also enables planner debug).")
    return p


def main():
    args    = build_parser().parse_args()
    planner = resolve_planner_name(args.planner)

    wp.init()

    # Peek at the rollout task once for its default weights, success thresholds
    # and dimensions; the episodes build their own tasks.
    peek_task       = load_rollout_task(args.task, args.geometry)
    default_weights = dict(peek_task.config.cost_weights)
    thresholds      = dict(peek_task.config.success_thresholds)

    if args.per_model_temperature and not args.opt_temperature:
        raise ValueError(
            "--per_model_temperature has nothing to split: temperature is "
            "pinned to --temperature by --no-opt_temperature"
        )

    if args.opt_cost_weights:
        specs = parse_weight_specs(args.opt_weights, default_weights)
    else:
        # Weights pinned: no weight dimensions at all, so every trial runs the
        # task's own cost_weights and only the planner knobs move. The space
        # still has to contain something for the GP to search.
        if args.opt_weights:
            raise ValueError(
                "--opt_weights lists weights to search, but --no-opt_cost_weights "
                "pins them all to the task's defaults; drop one of the two"
            )
        if not (args.opt_noise_sigma or args.opt_temperature):
            raise ValueError(
                "--no-opt_cost_weights leaves an empty search space; enable "
                "--opt_noise_sigma and/or --opt_temperature"
            )
        specs = []
    dims, names = build_space(specs, args)

    # Quantize the requested durations into the step counts the controller will
    # resolve internally, for the log line and the trial records.
    schedule = resolve_mppi_schedule(
        MPPIConfig(time_horizon=args.time_horizon, step_time=args.step_time),
        peek_task.config, args.eval_substeps,
    )
    horizon, substeps, rollout_dt = schedule

    model_tag = "+".join(args.models)
    outdir = Path(args.outdir) if args.outdir else (
        RESULTS_DIR / f"bayes_opt_{args.task}_{model_tag}_{planner}_"
                      f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # --- resume ------------------------------------------------------------
    x0 = y0 = None
    n_prior = 0
    if args.resume:
        x0, y0 = load_state(Path(args.resume), names, dims)
        n_prior = len(x0)
        if n_prior == 0:
            x0 = y0 = None

    # skopt counts n_calls as NEW evaluations and, when y0 is supplied, still
    # draws n_initial_points fresh random points on top of the restored ones.
    # Rebase both so --n_calls / --n_initial_points mean the same thing whether
    # or not the run is resumed.
    n_calls_eff  = max(0, args.n_calls - n_prior)
    n_init_eff   = max(0, args.n_initial_points - n_prior)
    if n_calls_eff < max(n_init_eff, 1):
        raise ValueError(
            f"--n_calls {args.n_calls} leaves {n_calls_eff} new trials after "
            f"{n_prior} resumed; need at least {max(n_init_eff, 1)}"
        )

    gp_noise = args.gp_noise
    if gp_noise != "gaussian":
        gp_noise = float(gp_noise)

    print(f"{'='*70}")
    print(f"  Bayesian optimization — task={args.task}  "
          f"model{'s' if len(args.models) > 1 else ''}={model_tag}  planner={planner}")
    if len(args.models) > 1:
        print(f"  one weight vector scored on all {len(args.models)} models at the "
              f"same episode seeds; folded by --model_agg {args.model_agg}")
    print(f"  eval_sim={args.eval_sim}  geometry={args.geometry}  "
          f"n_episodes={args.n_episodes}  seed={args.seed} (constant)")
    print(f"  rollout_dt={rollout_dt*1e3:.3f}ms  step_time={args.step_time:g}s -> "
          f"{substeps} substeps  time_horizon={args.time_horizon:g}s -> {horizon} steps")
    print(f"  objective = -({args.w_success:g} * success_rate) + "
          f"{args.w_cost:g} * norm_goal_err   (err_clip={args.err_clip:g})")
    if not thresholds:
        print(f"  ! task has no success_thresholds; the goal-error term will be inactive")
    print(f"  search space ({len(dims)} dims):")
    for d, name in zip(dims, names):
        print(f"      {name:<16} [{d.low:.5g}, {d.high:.5g}]  {d.prior}")
    if not args.opt_cost_weights:
        print(f"      cost weights     pinned to the task's defaults")
    if not args.opt_noise_sigma:
        print(f"      noise_sigma      pinned to {args.noise_sigma:g}")
    if not args.opt_temperature:
        print(f"      temperature      pinned to {args.temperature:g}")
    print(f"  budget: {args.n_calls} trials total "
          f"({n_prior} resumed, {n_calls_eff} new, {n_init_eff} random)")
    print(f"  outdir: {outdir}")
    print(f"{'='*70}")

    # --- episode fan-out ---------------------------------------------------
    # Each worker stands up its own planner (own MJWarp Data at nworld=n_samples
    # and its own captured CUDA graphs) plus its own eval sim, so VRAM scales
    # with the worker count. Report the headroom rather than discovering it as
    # an OOM forty trials in.
    # A trial is n_models x n_episodes independent episodes; that product is the
    # most concurrency that could ever be used.
    jobs_per_trial = len(args.models) * args.n_episodes
    n_workers = (args.n_workers if args.n_workers is not None
                 else default_worker_count(jobs_per_trial))
    n_workers = max(1, min(n_workers, jobs_per_trial))
    free_gb   = _device_free_gb()
    if free_gb is not None:
        per = free_gb / n_workers
        print(f"  workers: {n_workers} (of {jobs_per_trial} episodes/trial = "
              f"{len(args.models)} models x {args.n_episodes})  "
              f"free VRAM {free_gb:.1f} GiB -> {per:.1f} GiB/worker")
        if per < MIN_VRAM_PER_WORKER_GB:
            print(f"  ! under {MIN_VRAM_PER_WORKER_GB:g} GiB per worker; lower "
                  f"--n_workers or --n_samples if a worker dies with an OOM")
    else:
        print(f"  workers: {n_workers} (of {jobs_per_trial} episodes/trial)")

    pool        = EpisodePool(n_workers) if n_workers > 1 else None
    objective   = BOObjective(args, names, default_weights, thresholds,
                              schedule, outdir, start_trial=n_prior, pool=pool)
    state_path  = outdir / "bo_state.json"

    def on_step(res):
        save_state(state_path, names, dims, res.x_iters, res.func_vals, args)
        best_i = int(np.argmin(res.func_vals))
        print(f"  best so far: objective={float(res.func_vals[best_i]):+.5f} "
              f"at trial {best_i:03d}")

    t0 = time.perf_counter()
    try:
        res = gp_minimize(
            func             = objective,
            dimensions       = dims,
            n_calls          = n_calls_eff,
            n_initial_points = n_init_eff,
            acq_func         = args.acq_func,
            random_state     = args.bo_seed,
            noise            = gp_noise,
            x0               = x0,
            y0               = y0,
            callback         = [on_step],
            verbose          = False,
        )
    finally:
        # Workers hold CUDA contexts; a Ctrl-C or a raised trial must not leave
        # them parked on the GPU.
        if pool is not None:
            pool.shutdown()
    elapsed = time.perf_counter() - t0

    # --- summary -----------------------------------------------------------
    save_state(state_path, names, dims, res.x_iters, res.func_vals, args)

    best_x     = [float(v) for v in res.x]
    best_params = dict(zip(names, best_x))
    best_weights = {**default_weights,
                    **{k: v for k, v in best_params.items() if k in default_weights}}
    # res.x_iters includes the resumed points first, so when the winner came from
    # a resumed run its record is not in objective.records — fall back to the
    # cell files sitting next to the checkpoint we resumed from.
    best_record = next((r for r in objective.records if r["x"] == best_x), None)
    if best_record is None and args.resume:
        for cell in sorted(Path(args.resume).parent.glob("cell_*.json")):
            try:
                with open(cell) as f:
                    rec = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if rec.get("x") == best_x:
                best_record = rec
                break

    best_temps = model_temperatures(best_params, list(args.models), args.temperature)
    best_shared_temp = (next(iter(best_temps.values()))
                        if len(set(best_temps.values())) == 1 else None)

    summary = {
        "task":         args.task,
        "model":        model_tag,
        "models":       list(args.models),
        "model_agg":    args.model_agg,
        "planner":      planner,
        "geometry":     args.geometry,
        "eval_sim":     args.eval_sim,
        "n_episodes":   args.n_episodes,
        "seed":         args.seed,
        "bo_seed":      args.bo_seed,
        "n_calls":      args.n_calls,
        "n_resumed":    n_prior,
        "dim_names":    names,
        "dim_bounds":   [[float(d.low), float(d.high), d.prior] for d in dims],
        "objective_weights": {"w_success": args.w_success, "w_cost": args.w_cost,
                              "err_clip": args.err_clip},
        "best_objective":  float(res.fun),
        "best_x":          best_x,
        "best_params":     best_params,
        "best_weights":    best_weights,
        "best_noise_sigma": best_params.get("noise_sigma", args.noise_sigma),
        # Scalar as it always was, and None once the models carry their own.
        "best_temperature": best_shared_temp,
        "best_temperature_per_model": dict(best_temps),
        "best_success_rate":   best_record["success_rate"] if best_record else None,
        "best_norm_goal_err":  best_record["mean_norm_goal_err"] if best_record else None,
        "trace":        [float(v) for v in res.func_vals],
        "x_iters":      [[float(v) for v in x] for x in res.x_iters],
        "elapsed_seconds": elapsed,
    }
    with open(outdir / "bo_best.json", "w") as f:
        json.dump(summary, f, indent=2)

    weight_flags = " ".join(f"{k}={v:g}" for k, v in best_params.items()
                            if k in default_weights)
    print(f"\n{'='*70}")
    print(f"  BEST  objective={float(res.fun):+.5f}   ({len(res.func_vals)} trials, "
          f"{elapsed/60:.1f} min)")
    if best_record is not None:
        print(f"        success_rate={best_record['success_rate']:.3f}  "
              f"norm_goal_err={best_record['mean_norm_goal_err']}")
    for name in names:
        print(f"        {name:<16} {best_params[name]:.6g}")
    # run_eval_episode takes ONE --temperature, so a per-model winner replays as
    # one command per model rather than a single multi-model one.
    replays = ([(list(args.models), best_shared_temp)] if best_shared_temp is not None
               else [([m], t) for m, t in best_temps.items()])
    print(f"\n  Replay it with:")
    for models, temp in replays:
        print(f"    python -m contact_study.drivers.run_eval_episode \\")
        print(f"        --task {args.task} --geometry {args.geometry} "
              f"--models {' '.join(models)} \\")
        print(f"        --planner {planner} --n_samples {args.n_samples} \\")
        print(f"        --time_horizon {args.time_horizon:g} --step_time {args.step_time:g} \\")
        print(f"        --noise_sigma {best_params.get('noise_sigma', args.noise_sigma):g} \\")
        print(f"        --temperature {temp:g} \\")
        print(f"        --seed {args.seed} --settle {args.settle:g} \\")
        print(f"        --weights {weight_flags}")
    print(f"\n  Saved -> {outdir}")
    print(f"{'='*70}")

    return res


if __name__ == "__main__":
    main()
