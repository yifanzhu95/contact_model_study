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
  * noise_sigma, the sampling planner's action-noise standard deviation;
  * temperature, MPPI's softmax sharpness (dropped for planners that have no
    such field — make_planner_config filters by the config's declared fields).

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
        --model M2 --n_calls 60 --n_episodes 1
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

from skopt import gp_minimize
from skopt.space import Real

import contact_study.tasks  # noqa: F401 — registers all tasks

from contact_study.drivers.run_eval_episode import (
    MODEL_FACTORIES, load_rollout_task, resolve_mppi_schedule, run_eval_episode,
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
        dims.append(_real(lo, hi, "temperature"))
        names.append("temperature")

    return dims, names


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
                 outdir: Path, start_trial: int = 0):
        self.args            = args
        self.dim_names       = dim_names
        self.default_weights = default_weights
        self.thresholds      = thresholds
        self.horizon, self.substeps, self.rollout_dt = schedule
        self.outdir          = outdir
        self.trial           = start_trial

        self.planner     = resolve_planner_name(args.planner)
        self.contact_cfg = MODEL_FACTORIES[args.model]()
        self.eval_sim    = None if args.eval_sim == "none" else EvalSimulatorKind(args.eval_sim)
        self.delta       = (-args.delta, args.delta) if args.delta is not None else (None, None)

        # THE constant seed. Spawned once and reused by every trial, so the
        # objective is a function of the hyperparameters alone.
        self.episode_seeds = np.random.SeedSequence(args.seed).spawn(args.n_episodes)

        self.records: list[dict] = []
        self._warned_no_goal_err = False

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
        label     = combo_label(args.model, axes)

        noise_sigma = params.get("noise_sigma", args.noise_sigma)
        temperature = params.get("temperature", args.temperature)

        print(f"\n[trial {trial:03d}]  {label}")
        print(f"  noise_sigma={noise_sigma:.5g}  temperature={temperature:.5g}  "
              f"n_episodes={args.n_episodes}")

        episodes = []
        t0 = time.perf_counter()
        for ep in range(args.n_episodes):
            ep_seed = int(self.episode_seeds[ep].generate_state(1)[0])
            rng     = np.random.default_rng(self.episode_seeds[ep])

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
                seed              = ep_seed,
                debug             = args.debug,
                resample_interval = args.resample_interval,
                temperature       = temperature,
            )
            # No CLI default: omitting it lets each planner's own default stand
            # (1 for mppi/predictive_sampler, 3 for cem).
            if args.n_iterations is not None:
                planner_kwargs["n_iterations"] = args.n_iterations

            result = run_eval_episode(
                task_name             = args.task,
                geometry              = args.geometry,
                contact_cfg           = self.contact_cfg,
                planner               = self.planner,
                planner_cfg           = make_planner_config(self.planner, **planner_kwargs),
                rng                   = rng,
                video_path            = None,
                cost_weight_overrides = overrides or None,
                settle_seconds        = args.settle,
                eval_substeps         = args.eval_substeps,
                eval_sim              = self.eval_sim,
                ep_idx                = ep,
                fin_ep_on_success     = True,
                debug                 = args.debug,
                verbose               = args.debug,
            )
            episodes.append(result)

            tick = "✓" if result.success else "✗"
            sstr = f"step {result.steps_to_success}" if result.steps_to_success is not None else "—"
            gerr = normalized_goal_error(result.final_goal_errs, self.thresholds,
                                         args.err_clip)
            gstr = f"{gerr:.4f}" if gerr is not None else "n/a"
            print(f"    ep {ep:02d}  {tick}  success_step={sstr:<9}  goal_err={gstr}  "
                  f"step={result.mean_step_ms:.3f}±{result.std_step_ms:.3f} ms")

        # -- scalarize ------------------------------------------------------
        n_success    = sum(r.success for r in episodes)
        success_rate = n_success / len(episodes)

        errs = [normalized_goal_error(r.final_goal_errs, self.thresholds, args.err_clip)
                for r in episodes]
        errs = [e for e in errs if e is not None]
        mean_err = float(np.mean(errs)) if errs else None

        objective = -(args.w_success * success_rate)
        if mean_err is not None:
            objective += args.w_cost * mean_err
        elif not self._warned_no_goal_err:
            self._warned_no_goal_err = True
            print(f"  ! task {args.task!r} has no goal_errors(); optimizing pure "
                  f"success rate (the --w_cost term is inactive)")

        succ_steps = [r.steps_to_success for r in episodes if r.steps_to_success is not None]
        step_ms    = [r.mean_step_ms for r in episodes]

        print(f"  → objective={objective:+.5f}  success={success_rate*100:.1f}% "
              f"({n_success}/{len(episodes)})  goal_err="
              f"{mean_err if mean_err is None else round(mean_err, 4)}  "
              f"[{time.perf_counter() - t0:.1f}s]")

        record = {
            # -- run_param_cell.py schema, so param_search_to_csv_dir.py works --
            "combo_index":           trial,
            "task":                  args.task,
            "model":                 args.model,
            "label":                 label,
            "overrides":             overrides,
            "axes":                  axes,
            "swept_knobs":           list(knobs),
            "full_weights":          {**self.default_weights, **overrides},
            "n_episodes":            len(episodes),
            "n_success":             n_success,
            "success_rate":          success_rate,
            "mean_steps_to_success": float(np.mean(succ_steps)) if succ_steps else None,
            "mean_step_ms":          float(np.mean(step_ms)),
            "std_step_ms":           float(np.mean([r.std_step_ms for r in episodes])),
            "mean_elapsed_s":        float(np.mean([r.elapsed_seconds for r in episodes])),
            "mppi": {
                "n_samples":         args.n_samples,
                "time_horizon":      args.time_horizon,
                "temperature":       temperature,
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
            "episodes": [dataclasses.asdict(r) for r in episodes],
        }
        self.records.append(record)
        # Written inside the objective so a killed job keeps every finished trial.
        with open(self.outdir / f"cell_{trial:05d}.json", "w") as f:
            json.dump(record, f, indent=2)

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
    with open(path, "w") as f:
        json.dump({
            "dim_names":  list(dim_names),
            "dim_bounds": [[float(d.low), float(d.high), d.prior] for d in dims],
            "x_iters":    [[float(v) for v in x] for x in x_iters],
            "func_vals":  [float(v) for v in func_vals],
            "args":       vars(args),
        }, f, indent=2)


def load_state(path: Path, dim_names: list[str]) -> tuple[list[list[float]], list[float]]:
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
    p.add_argument("--geometry",    type=str, default=DEFAULT_SCENE_VARIANT,
                   help="Scene variant: '<object>' or '<object>_<hand_acc>_<obj_acc>'.")
    p.add_argument("--model",       type=str, default="M2", choices=list(MODEL_FACTORIES))
    p.add_argument("--planner",     type=str, default="mppi", choices=PLANNER_NAMES)
    p.add_argument("--n_samples",   type=int, default=256)
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
    p.add_argument("--n_episodes",  type=int, default=2,
                   help="Episodes per objective evaluation. More episodes average out "
                        "episode-specific luck at a proportional cost in wall time.")
    p.add_argument("--seed",        type=int, default=42,
                   help="THE constant episode seed: the same episodes are replayed for "
                        "every trial, so trials differ only in their hyperparameters.")

    # --- search space -------------------------------------------------------
    p.add_argument("--opt_weights", nargs="*", default=None,
                   help="Cost weights to optimize, as 'name' (bounds bracketed "
                        f"multiplicatively at x{BOUND_SCALE:g} around the task's default) "
                        "or 'name:lo:hi'. Default: the subset of "
                        f"{' '.join(DEFAULT_OPT_WEIGHTS)} the task declares.")
    p.add_argument("--opt_noise_sigma", action=argparse.BooleanOptionalAction, default=True,
                   help="Search over noise_sigma; --no-opt_noise_sigma pins it to "
                        "--noise_sigma instead.")
    p.add_argument("--opt_temperature", action=argparse.BooleanOptionalAction, default=True,
                   help="Search over MPPI's temperature; --no-opt_temperature pins it to "
                        "--temperature instead.")
    p.add_argument("--noise_sigma_range", type=float, nargs=2, default=(1e-3, 1.0),
                   metavar=("LO", "HI"))
    p.add_argument("--temperature_range", type=float, nargs=2, default=(0.01, 50.0),
                   metavar=("LO", "HI"))
    p.add_argument("--noise_sigma", type=float, default=0.2,
                   help="Fixed noise_sigma when --no-opt_noise_sigma.")
    p.add_argument("--temperature", type=float, default=30.0,
                   help="Fixed temperature when --no-opt_temperature.")

    # --- objective ----------------------------------------------------------
    p.add_argument("--w_success",  type=float, default=1.0,
                   help="Weight on the success rate (maximized).")
    p.add_argument("--w_cost",     type=float, default=0.1,
                   help="Weight on the normalized final goal error (minimized).")
    p.add_argument("--err_clip",   type=float, default=10.0,
                   help="Goal error is clipped here then divided by it, mapping the "
                        "cost term into [0, 1] so it cannot be dominated by one "
                        "catastrophic episode.")

    # --- optimizer ----------------------------------------------------------
    p.add_argument("--n_calls",    type=int, default=500,
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

    specs      = parse_weight_specs(args.opt_weights, default_weights)
    dims, names = build_space(specs, args)

    # Quantize the requested durations into the step counts the controller will
    # resolve internally, for the log line and the trial records.
    schedule = resolve_mppi_schedule(
        MPPIConfig(time_horizon=args.time_horizon, step_time=args.step_time),
        peek_task.config, args.eval_substeps,
    )
    horizon, substeps, rollout_dt = schedule

    outdir = Path(args.outdir) if args.outdir else (
        RESULTS_DIR / f"bayes_opt_{args.task}_{args.model}_{planner}_"
                      f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # --- resume ------------------------------------------------------------
    x0 = y0 = None
    n_prior = 0
    if args.resume:
        x0, y0 = load_state(Path(args.resume), names)
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
    print(f"  Bayesian optimization — task={args.task}  model={args.model}  "
          f"planner={planner}")
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
    if not args.opt_noise_sigma:
        print(f"      noise_sigma      pinned to {args.noise_sigma:g}")
    if not args.opt_temperature:
        print(f"      temperature      pinned to {args.temperature:g}")
    print(f"  budget: {args.n_calls} trials total "
          f"({n_prior} resumed, {n_calls_eff} new, {n_init_eff} random)")
    print(f"  outdir: {outdir}")
    print(f"{'='*70}")

    objective   = BOObjective(args, names, default_weights, thresholds,
                              schedule, outdir, start_trial=n_prior)
    state_path  = outdir / "bo_state.json"

    def on_step(res):
        save_state(state_path, names, dims, res.x_iters, res.func_vals, args)
        best_i = int(np.argmin(res.func_vals))
        print(f"  best so far: objective={float(res.func_vals[best_i]):+.5f} "
              f"at trial {best_i:03d}")

    t0 = time.perf_counter()
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

    summary = {
        "task":         args.task,
        "model":        args.model,
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
        "best_temperature": best_params.get("temperature", args.temperature),
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
    print(f"\n  Replay it with:")
    print(f"    python -m contact_study.drivers.run_eval_episode \\")
    print(f"        --task {args.task} --geometry {args.geometry} --model {args.model} \\")
    print(f"        --planner {planner} --n_samples {args.n_samples} \\")
    print(f"        --time_horizon {args.time_horizon:g} --step_time {args.step_time:g} \\")
    print(f"        --noise_sigma {best_params.get('noise_sigma', args.noise_sigma):g} \\")
    print(f"        --temperature {best_params.get('temperature', args.temperature):g} \\")
    print(f"        --seed {args.seed} --settle {args.settle:g} \\")
    print(f"        --weights {weight_flags}")
    print(f"\n  Saved -> {outdir}")
    print(f"{'='*70}")

    return res


if __name__ == "__main__":
    main()
