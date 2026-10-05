#!/usr/bin/env python3
"""Bayesian optimization of cost weights, MPPI temperature and noise, per CSV row.

Each data row of the CSV is one *cell*: a fixed set of episode settings (task,
fidelity, horizon, control step, sample count, ...), the contact models to score
on, and the search space. The cell runs a Gaussian-process Bayesian optimization
(scikit-optimize) over that space: each *trial* is one point of it, scored by
running ``n_episodes`` episodes on every listed model.

Every episode runs on one ``EpisodePool`` (``ContactModelStudy/Drivers/EpisodePool.py``)
that lives for the whole cell: planner processes on every GPU, eval-sim workers
on the CPU cores, and several trials in flight at once, so the machine stays
busy. A trial's settings travel with its episodes, so the pool never restarts.

Every trial sees the same episodes: an episode's goals and planner noise come
from ``(seed, episode index)`` alone, and every model gets the same ones. So
two trials differ only in the point under test, and two models only in the
contact model.

::

    python experiments/run_bayes_opt.py bo.csv --check     # validate only
    python experiments/run_bayes_opt.py bo.csv             # every cell, in order
    python experiments/run_bayes_opt.py bo.csv --cell 2    # just row 2 (0-based)
    python experiments/run_bayes_opt.py bo.csv --cell 2 --gpus 0,1 --workers 12
    python experiments/run_bayes_opt.py bo.csv --summarize --outdir results/my_bo

On the HPC, ``hpc/submit_bayes_opt.sh`` submits one array task per row.

Objective (minimized), the old ``run_bayes_opt.py``'s::

    J_model = -w_success * success_rate + w_cost * mean normalized final goal error

The goal error is each final error divided by the task's success threshold,
summed, clipped at ``err_clip`` and divided by it, so both terms are in [0, 1].
An episode that raised counts as a failure with the worst error. With several
models, ``model_agg`` folds their J: ``mean``, or ``worst`` (the largest).

CSV columns
-----------
Everything ``run_episode_batches.py`` accepts (any ``run_episodes.py`` option
with underscores, ``w_*`` weights to pin, ``label``), except ``rollout_model``
(use ``rollout_models``) and ``video``. A blank cell keeps the default.

Search space:

* ``opt_weights``: the weights to search, ``"w_quat:1:50 w_contact:0.1:100"``.
  A bare name searches x/4 to x4 around the object's own value. ``none`` pins
  every weight. Blank: the old study's nine weights and bounds, each widened if
  needed to contain the object's own value.
* ``temperature_range`` / ``noise_sigma_range``: ``"lo hi"``. Blank pins it to
  the ``temperature`` / ``noise_sigma`` column, or the driver default.
* ``per_model_temperature``: ``true`` gives each model its own temperature
  dimension (``temperature_<model>``); the weights stay shared.

Every dimension is log-uniform.

Models and objective: ``rollout_models`` (``"M1 M2 M3 M4"``; blank: the driver
default), ``model_agg`` (``mean``), ``w_success`` (1), ``w_cost`` (0.1),
``err_clip`` (250).

Optimizer: ``n_calls`` (100 trials, including seeds and resumed ones),
``n_initial_points`` (10 random trials, less one per seed), ``acq_func``
(``gp_hedge``; or ``EI``, ``LCB``, ``PI``), ``bo_seed`` (0),
``trials_in_flight`` (default: enough to give every worker an episode).

Seeds, evaluated first:

* ``seed_defaults`` (true): a trial at the object's own weights and the pinned
  temperature and noise.
* ``seed_points``: a CSV of known settings, relative to this CSV; one row per
  point, with columns named after the dimensions. A dimension it leaves out
  takes its default.
* ``seed_from`` + ``seed_top_k`` (5): re-evaluate the best trials of an earlier
  cell folder.

Output
------
``--outdir`` (default ``results/bayes_opt_<csv name>_<job id or time>/``) holds
one folder per cell, ``cell_<row>[_<label>]/``, containing:

* ``trial_<i>/<model>.json`` (+ ``.npy`` if ``save_steps``): that trial's
  episodes on that model, the ``EpisodeRecorder`` output;
* ``trial_<i>/trial.json``: the point, where it came from, its J and per-model scores;
* ``bo_state.json``: the search space and every point and J so far;
* ``bo_summary.json``: the trials ranked, and the best one.

and, at the top, ``summary.csv`` (one row per trial) and ``best.csv`` (each
cell's best). Rerunning into the same folder resumes: finished trials are told
to the optimizer again, and the run continues to ``n_calls``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_episode_batches as batches  # noqa: E402
from ContactModelStudy.Drivers import run_episodes as drv  # noqa: E402
from ContactModelStudy.Tasks.LeapReorient import COST_WEIGHT_KEYS  # noqa: E402

#: The old study's search space: weight -> (low, high).
DEFAULT_OPT_WEIGHTS = {
    "w_quat": (1.0, 50.0), "w_pos_x": (1.0, 50.0), "w_pos_y": (1.0, 50.0),
    "w_pos_z": (1.0, 50.0), "w_contact": (0.1, 100.0), "w_joint": (0.1, 20.0),
    "w_fallen": (100.0, 300.0), "w_quat_term": (100.0, 300.0), "w_pos_term": (100.0, 300.0),
}
#: A bare weight name searches [default / BOUND_SCALE, default * BOUND_SCALE].
BOUND_SCALE = 4.0
ACQ_FUNCS = ("gp_hedge", "EI", "LCB", "PI")

BO_COLUMNS = (
    "rollout_models", "model_agg", "opt_weights", "temperature_range", "noise_sigma_range",
    "per_model_temperature", "n_calls", "n_initial_points", "acq_func", "bo_seed",
    "w_success", "w_cost", "err_clip", "trials_in_flight",
    "seed_defaults", "seed_points", "seed_from", "seed_top_k",
)
OWN_COLUMNS = ("label",) + BO_COLUMNS
#: Set by this script, so not allowed as columns (``video``: sweeps never render).
MANAGED_OPTIONS = batches.MANAGED_OPTIONS + ("rollout_model",)


# -- the cell's search space -------------------------------------------------------
@dataclass
class Dim:
    """One log-uniform search dimension."""

    name: str
    lo: float
    hi: float

    def skopt(self):
        from skopt.space import Real
        return Real(self.lo, self.hi, prior="log-uniform", name=self.name)


@dataclass
class Seed:
    """A point to evaluate before the optimizer's own, and where it came from."""

    x: list[float]
    source: str


@dataclass
class CellSpec:
    """Everything a cell's search needs, parsed and checked from its CSV row."""

    row: dict[str, str]
    args: argparse.Namespace
    argv: list[str]
    models: list[str]
    model_agg: str
    dims: list[Dim]
    base_weights: dict[str, float]
    pinned_temperature: float
    pinned_noise_sigma: float
    thresholds: dict[str, float]
    n_calls: int
    n_initial_points: int
    acq_func: str
    bo_seed: int
    w_success: float
    w_cost: float
    err_clip: float
    trials_in_flight: int | None
    seeds: list[Seed] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.dims]

    @property
    def episodes_per_trial(self) -> int:
        return len(self.models) * self.args.n_episodes

    @property
    def n_random(self) -> int:
        """Random trials: ``n_initial_points``, less one per seed."""
        return max(0, self.n_initial_points - len(self.seeds))

    # -- a point's meaning ------------------------------------------------------
    def params(self, x) -> dict[str, float]:
        return {d.name: float(v) for d, v in zip(self.dims, x)}

    def defaultParams(self) -> dict[str, float]:
        """Every dimension at its default: the object's weights, the pinned knobs."""
        out = {}
        for d in self.dims:
            if d.name in COST_WEIGHT_KEYS:
                out[d.name] = self.base_weights[d.name]
            elif d.name.startswith("temperature"):
                out[d.name] = self.pinned_temperature
            else:
                out[d.name] = self.pinned_noise_sigma
        return out

    def modelSettings(self, x, model: str) -> tuple[dict, dict]:
        """``(planner_params, cost_weights)`` of point ``x`` for one model."""
        p = self.params(x)
        planner = {}
        temp = p.get(f"temperature_{model}", p.get("temperature"))
        if temp is not None:
            planner["temperature"] = temp
        if "noise_sigma" in p:
            planner["noise_sigma"] = p["noise_sigma"]
        weights = {k: v for k, v in p.items() if k in COST_WEIGHT_KEYS}
        return planner, weights

    def describe(self, x) -> str:
        return "  ".join(f"{(n[2:] if n.startswith('w_') else n)}={v:.4g}"
                         for n, v in self.params(x).items())


def _tokens(raw: str) -> list[str]:
    return [t for t in raw.replace(",", " ").split() if t]


def _float(raw: str, column: str) -> float:
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{column}: {raw!r} is not a number") from None


def _int(raw: str, column: str, minimum: int) -> int:
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{column}: {raw!r} is not an integer") from None
    if v < minimum:
        raise ValueError(f"{column} must be >= {minimum}, got {v}")
    return v


def parseRange(raw: str, column: str) -> tuple[float, float]:
    """``"0.01 25"`` -> ``(0.01, 25.0)``; both positive, low first."""
    t = _tokens(raw)
    if len(t) != 2:
        raise ValueError(f"{column} wants two numbers, 'lo hi', got {raw!r}")
    lo, hi = _float(t[0], column), _float(t[1], column)
    if not 0 < lo < hi:
        raise ValueError(f"{column} needs 0 < lo < hi, got {raw!r}")
    return lo, hi


def parseWeightSpecs(raw: str | None, defaults: dict[str, float]) -> list[Dim]:
    """The ``opt_weights`` column as dimensions; see the module docstring."""
    if raw is None:
        # The old space, widened where the object's own value lies outside it,
        # so the defaults seed is always a legal point.
        return [Dim(n, min(lo, defaults[n]), max(hi, defaults[n])) if defaults[n] > 0 else Dim(n, lo, hi)
                for n, (lo, hi) in DEFAULT_OPT_WEIGHTS.items()]
    if raw.strip().lower() == "none":
        return []
    dims, seen = [], set()
    for tok in _tokens(raw):
        parts = tok.split(":")
        name = parts[0]
        if name not in COST_WEIGHT_KEYS:
            raise ValueError(f"opt_weights: {name!r} is not a cost weight; valid: {list(COST_WEIGHT_KEYS)}")
        if name in seen:
            raise ValueError(f"opt_weights lists {name!r} twice")
        seen.add(name)
        if len(parts) == 1:
            d = defaults[name]
            if d <= 0:
                raise ValueError(f"opt_weights: {name} defaults to {d:g}, which has no x{BOUND_SCALE:g} "
                                 f"bracket; give bounds as {name}:lo:hi")
            lo, hi = d / BOUND_SCALE, d * BOUND_SCALE
        elif len(parts) == 3:
            lo, hi = _float(parts[1], "opt_weights"), _float(parts[2], "opt_weights")
            if not 0 < lo < hi:
                raise ValueError(f"opt_weights: {name} needs 0 < lo < hi, got {lo:g}:{hi:g}")
        else:
            raise ValueError(f"opt_weights: bad entry {tok!r}; use 'name' or 'name:lo:hi'")
        dims.append(Dim(name, lo, hi))
    return dims


def _bool(row: dict, column: str, default: bool) -> bool:
    return batches._parseBool(row[column], column) if column in row else default


def _resolvePath(raw: str, csv_dir: Path) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (csv_dir / p).resolve()


def parseCell(row: dict[str, str], csv_dir: Path, cell_dir: Path) -> CellSpec:
    """Parse and check one row. Raises ``ValueError`` naming what is wrong."""
    models = _tokens(row.get("rollout_models", ""))
    if len(set(models)) != len(models):
        raise ValueError(f"rollout_models lists a model twice: {row['rollout_models']!r}")
    base_argv = batches.rowToArgv(row, cell_dir, "trial", own_columns=OWN_COLUMNS)
    if not models:
        models = [drv.parseArgs(base_argv).rollout_model]
    for m in models:
        err = batches.checkRow(base_argv + ["--rollout-model", m])
        if err:
            raise ValueError(f"model {m}: {err}")
    argv = base_argv + ["--rollout-model", models[0]]
    args = drv.parseArgs(argv)
    task, _ = drv.buildTasks(args)               # also checks the scene exists
    base_weights = dict(task.params["cost_weights"])

    model_agg = row.get("model_agg", "mean")
    if model_agg not in ("mean", "worst"):
        raise ValueError(f"model_agg must be 'mean' or 'worst', got {model_agg!r}")

    dims = parseWeightSpecs(row.get("opt_weights"), base_weights)
    pinned_w = [d.name for d in dims if d.name in row]
    if pinned_w:
        raise ValueError(f"{pinned_w} are both pinned (a w_* column) and searched (opt_weights)")
    per_model = _bool(row, "per_model_temperature", False)
    if "temperature_range" in row:
        if "temperature" in row:
            raise ValueError("temperature and temperature_range are both set: pin it or search it")
        lo, hi = parseRange(row["temperature_range"], "temperature_range")
        names = [f"temperature_{m}" for m in models] if per_model and len(models) > 1 else ["temperature"]
        dims += [Dim(n, lo, hi) for n in names]
    elif per_model:
        raise ValueError("per_model_temperature needs a temperature_range to search")
    if "noise_sigma_range" in row:
        if "noise_sigma" in row:
            raise ValueError("noise_sigma and noise_sigma_range are both set: pin it or search it")
        lo, hi = parseRange(row["noise_sigma_range"], "noise_sigma_range")
        dims.append(Dim("noise_sigma", lo, hi))
    if not dims:
        raise ValueError("the search space is empty: give opt_weights, temperature_range or noise_sigma_range")

    acq = row.get("acq_func", "gp_hedge")
    if acq not in ACQ_FUNCS:
        raise ValueError(f"acq_func must be one of {ACQ_FUNCS}, got {acq!r}")
    spec = CellSpec(
        row=row, args=args, argv=argv, models=models, model_agg=model_agg, dims=dims,
        base_weights=base_weights, pinned_temperature=args.temperature,
        pinned_noise_sigma=args.noise_sigma, thresholds=dict(type(task).SUCCESS_THRESHOLDS),
        n_calls=_int(row.get("n_calls", "100"), "n_calls", 1),
        n_initial_points=_int(row.get("n_initial_points", "10"), "n_initial_points", 0),
        acq_func=acq, bo_seed=_int(row.get("bo_seed", "0"), "bo_seed", 0),
        w_success=_float(row.get("w_success", "1.0"), "w_success"),
        w_cost=_float(row.get("w_cost", "0.1"), "w_cost"),
        err_clip=_float(row.get("err_clip", "250"), "err_clip"),
        trials_in_flight=(_int(row["trials_in_flight"], "trials_in_flight", 1)
                          if "trials_in_flight" in row else None),
    )
    if spec.err_clip <= 0:
        raise ValueError(f"err_clip must be > 0, got {spec.err_clip:g}")
    spec.seeds = buildSeeds(spec, row, csv_dir)
    if spec.n_calls < len(spec.seeds):
        raise ValueError(f"n_calls {spec.n_calls} is less than the {len(spec.seeds)} seeds")
    return spec


# -- seeds -------------------------------------------------------------------------
def pointFromParams(spec: CellSpec, params: dict[str, float], source: str, strict: bool = True) -> list[float]:
    """A point from named values; missing dimensions take their defaults.

    ``strict``: a name that is not a dimension must match the value it is
    pinned to (or it is an error); otherwise it is ignored.
    """
    values = {**spec.defaultParams(), **{k: v for k, v in params.items() if k in spec.names}}
    if "temperature" in params and "temperature" not in spec.names:
        # A shared temperature fills per-model dimensions the source leaves out.
        for n in spec.names:
            if n.startswith("temperature_") and n not in params:
                values[n] = params["temperature"]
    if strict:
        for k, v in params.items():
            if k in spec.names or (k == "temperature" and any(n.startswith("temperature_") for n in spec.names)):
                continue
            pinned = (spec.base_weights.get(k) if k in COST_WEIGHT_KEYS else
                      spec.pinned_temperature if k == "temperature" else
                      spec.pinned_noise_sigma if k == "noise_sigma" else None)
            if pinned is None:
                raise ValueError(f"{source}: {k!r} is not a search dimension or a pinned setting")
            if not math.isclose(v, pinned, rel_tol=1e-9):
                raise ValueError(f"{source}: {k}={v:g}, but this row pins {k} to {pinned:g}")
    x = []
    for d in spec.dims:
        v = float(values[d.name])
        if not d.lo <= v <= d.hi:
            raise ValueError(f"{source}: {d.name}={v:g} is outside the search range "
                             f"[{d.lo:g}, {d.hi:g}]" + (" (set seed_defaults=false to skip this seed)"
                                                       if source == "defaults" else ""))
        x.append(v)
    return x


def _readTrials(cell_dir: Path) -> list[dict]:
    """The finished trials of a cell folder, by trial index."""
    out = []
    for p in sorted(cell_dir.glob("trial_*/trial.json")):
        try:
            t = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if t.get("status") == "done":
            out.append(t)
    return sorted(out, key=lambda t: t["trial"])


def buildSeeds(spec: CellSpec, row: dict, csv_dir: Path) -> list[Seed]:
    """Defaults, then the seed-points file, then an earlier run's best; duplicates dropped."""
    seeds: list[Seed] = []
    if _bool(row, "seed_defaults", True):
        seeds.append(Seed(pointFromParams(spec, {}, "defaults"), "defaults"))
    if "seed_points" in row:
        path = _resolvePath(row["seed_points"], csv_dir)
        if not path.is_file():
            raise ValueError(f"seed_points: no such file {path}")
        with open(path, newline="") as f:
            for i, r in enumerate(csv.DictReader(f)):
                vals = {k.strip(): v.strip() for k, v in r.items() if k and v and v.strip()}
                params = {k: _float(v, f"seed_points row {i}") for k, v in vals.items()}
                seeds.append(Seed(pointFromParams(spec, params, f"seed_points:{i}"), f"seed_points:{i}"))
    if "seed_from" in row:
        src = _resolvePath(row["seed_from"], csv_dir)
        trials = _readTrials(src)
        if not trials:
            raise ValueError(f"seed_from: no finished trials in {src}")
        k = _int(row.get("seed_top_k", "5"), "seed_top_k", 1)
        best = sorted(trials, key=lambda t: t["objective"])[:k]
        unknown = sorted({n for t in best for n in t["params"]} - set(spec.names))
        if unknown:
            print(f"  note: seed_from {src.name}: {unknown} are not dimensions here; ignored")
        for t in best:
            # Clipped into the box: an earlier run may have searched a wider one.
            params = {n: min(max(v, d.lo), d.hi) for d in spec.dims
                      for n, v in t["params"].items() if n == d.name}
            seeds.append(Seed(pointFromParams(spec, params, f"seed_from:{t['trial']}", strict=False),
                              f"seed_from:{t['trial']}"))
    unique, seen = [], set()
    for s in seeds:
        key = tuple(round(v, 12) for v in s.x)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


# -- the objective -------------------------------------------------------------------
def normalizedGoalError(errors: dict | None, thresholds: dict[str, float], clip: float) -> float:
    """Final goal errors in [0, 1]: each over its threshold, summed, clipped, scaled.

    1 (the worst) when there are none to read, as for an episode that raised.
    """
    if not errors:
        return 1.0
    keys = [k for k in thresholds if k in errors and thresholds[k] > 0]
    if not keys:
        return 1.0
    e = sum(float(errors[k]) / float(thresholds[k]) for k in keys)
    return min(e, clip) / clip


def scoreModel(summaries: list[dict], spec: CellSpec) -> dict:
    """One model's episodes -> its success rate, goal error and J."""
    n = len(summaries)
    ok = [s for s in summaries if s.get("finish_reason") != "error"]
    n_success = sum(bool(s.get("success")) for s in ok)
    errs = [normalizedGoalError(s.get("goal_errors_end"), spec.thresholds, spec.err_clip)
            if s.get("finish_reason") != "error" else 1.0 for s in summaries]
    steps = [s["steps_to_success"] for s in ok if s.get("steps_to_success") is not None]
    rate = n_success / n if n else 0.0
    err = float(np.mean(errs)) if errs else 1.0
    return {"n_episodes": n, "n_success": n_success, "n_errors": n - len(ok), "success_rate": rate,
            "mean_norm_goal_err": err,
            "mean_steps_to_success": float(np.mean(steps)) if steps else None,
            "mean_goals_reached": float(np.mean([s.get("goals_reached", 0) for s in summaries])) if n else 0.0,
            "objective": -spec.w_success * rate + spec.w_cost * err}


def foldModels(per_model: dict[str, dict], how: str) -> float:
    """The trial's J: the mean of the models' J, or the worst (largest)."""
    js = [v["objective"] for v in per_model.values()]
    return float(np.mean(js)) if how == "mean" else float(max(js))


# -- the optimizer -------------------------------------------------------------------
def makeOptimizer(spec: CellSpec):
    """A GP optimizer over the cell's space; the random and seed phases are run here, not by skopt.

    ``n_initial_points=1`` makes it fit its GP from the first result told, so
    the caller decides when it is consulted (``askNext``).
    """
    from skopt import Optimizer
    from skopt.space import Space
    from skopt.utils import cook_estimator
    dims = [d.skopt() for d in spec.dims]
    rng = np.random.RandomState(spec.bo_seed)
    gp = cook_estimator("GP", space=Space(dims), random_state=rng.randint(0, np.iinfo(np.int32).max),
                        noise="gaussian")
    return Optimizer(dims, base_estimator=gp, n_initial_points=1, acq_func=spec.acq_func,
                     acq_optimizer="auto", random_state=rng)


def askNext(opt, pending: list[list[float]]) -> list[float]:
    """The GP's next point, given points still being evaluated (constant liar).

    A copy of the optimizer is told every pending point with the best J so
    far, so it does not propose them (or their neighbourhood) again.
    """
    tmp = opt.copy(random_state=opt.rng)
    if pending:
        tmp.tell([list(p) for p in pending], [float(min(opt.yi))] * len(pending))
    return [float(v) for v in tmp.ask()]


def randomPoint(opt) -> list[float]:
    return [float(v) for v in opt.space.rvs(n_samples=1, random_state=opt.rng)[0]]


# -- files -----------------------------------------------------------------------------
def trialName(i: int) -> str:
    return f"trial_{i:04d}"


def _stateDims(spec: CellSpec) -> list[list]:
    return [[d.name, d.lo, d.hi, "log-uniform"] for d in spec.dims]


def checkResume(spec: CellSpec, cell_dir: Path, done: list[dict]) -> None:
    """Refuse to resume a cell whose search space changed in a way that strands its trials."""
    state = cell_dir / "bo_state.json"
    if state.exists():
        prev = json.loads(state.read_text())
        names = [d[0] for d in prev.get("dims", [])]
        if names != spec.names:
            raise ValueError(f"{cell_dir} was searched over {names}, this row searches {spec.names}; "
                             f"use another --outdir, or --overwrite")
        for (name, lo, hi, _), d in zip(prev["dims"], spec.dims):
            if (lo, hi) != (d.lo, d.hi):
                print(f"  note: {name} was [{lo:g}, {hi:g}], now [{d.lo:g}, {d.hi:g}]")
    for t in done:
        for d, v in zip(spec.dims, t["x"]):
            if not d.lo <= v <= d.hi:
                raise ValueError(f"trial {t['trial']} has {d.name}={v:g}, outside the new range "
                                 f"[{d.lo:g}, {d.hi:g}]: narrowing strands it. Widen it back, or "
                                 f"use another --outdir")


def writeState(spec: CellSpec, cell_dir: Path, done: list[dict]) -> None:
    (cell_dir / "bo_state.json").write_text(json.dumps({
        "dims": _stateDims(spec), "trials": [t["trial"] for t in done],
        "x_iters": [t["x"] for t in done], "func_vals": [t["objective"] for t in done],
        "sources": [t["source"] for t in done], "row": spec.row,
    }, indent=2))


@dataclass
class _Trial:
    index: int
    x: list[float]
    source: str
    t0: float
    results: dict[str, list] = field(default_factory=dict)
    left: int = 0


def saveTrial(spec: CellSpec, pool_args, cell_dir: Path, trial: _Trial, per_model: dict,
              objective: float) -> dict:
    """Write the trial's recorders (one per model) and ``trial.json``; returns the record."""
    from ContactModelStudy.Drivers.EpisodePool import RemotePlanner, SimStandIn, describeRollout
    from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder
    from ContactModelStudy.Utils.EvalSimulators import evalSimConfig
    tdir = cell_dir / trialName(trial.index)
    tdir.mkdir(parents=True, exist_ok=True)
    settings = {}
    for m in spec.models:
        planner_params, weights = spec.modelSettings(trial.x, m)
        settings[m] = {"planner_params": planner_params}
        task, eval_task = drv.buildTasks(pool_args)
        if weights:
            task.setCostWeights(weights)
        desc = describeRollout(pool_args, task, m, planner_params)
        eval_cls, eval_cfg = evalSimConfig(pool_args.eval_sim, eval_task.timestep)
        rec = EpisodeRecorder(eval_task, SimStandIn(eval_cls, eval_cfg, nq=eval_task.nq, nv=eval_task.nv,
                                                    nu=eval_task.nu),
                              RemotePlanner(desc, task, None, None, -1, None),
                              cli_args=vars(pool_args), eval_sim=pool_args.eval_sim)
        rec.episodes = [r.episode for r in sorted(trial.results[m], key=lambda r: r.job.episode)]
        rec.Save(tdir / f"{m}.json", save_steps=pool_args.save_steps)
    params = spec.params(trial.x)
    n = sum(v["n_episodes"] for v in per_model.values())
    record = {
        "trial": trial.index, "status": "done", "source": trial.source, "x": trial.x,
        "params": params, "cost_weights": {k: v for k, v in params.items() if k in COST_WEIGHT_KEYS},
        "settings": settings, "objective": objective, "model_agg": spec.model_agg,
        "per_model": per_model, "n_episodes": n,
        "success_rate": sum(v["n_success"] for v in per_model.values()) / n if n else 0.0,
        "mean_norm_goal_err": float(np.mean([v["mean_norm_goal_err"] for v in per_model.values()])),
        "wall_s": round(time.time() - trial.t0, 1),
        "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    (tdir / "trial.json").write_text(json.dumps(record, indent=2))
    return record


# -- running -------------------------------------------------------------------------
def runCell(index: int, row: dict[str, str], csv_dir: Path, outdir: Path, overwrite: bool,
            pool_opts: argparse.Namespace) -> int:
    """One cell's whole search, on one pool. Returns 1 if it stopped early."""
    from ContactModelStudy.Drivers.EpisodePool import (EpisodeJob, EpisodePool, availableCpus,
                                                      sigtermAsInterrupt, visibleGpus)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    name = batches.cellName(index, row)
    cell_dir = outdir / name
    if overwrite and cell_dir.exists():
        shutil.rmtree(cell_dir)
    cell_dir.mkdir(parents=True, exist_ok=True)
    spec = parseCell(row, csv_dir, cell_dir)

    done = _readTrials(cell_dir)
    checkResume(spec, cell_dir, done)
    for d in cell_dir.glob("trial_*"):                   # half-finished trials are redone
        if d.is_dir() and d.name not in {trialName(t["trial"]) for t in done}:
            shutil.rmtree(d)
    opt = makeOptimizer(spec)
    if done:
        opt.tell([t["x"] for t in done], [t["objective"] for t in done])
    done_sources = {t["source"] for t in done}
    seeds = [s for s in spec.seeds if s.source not in done_sources]
    n_random_issued = sum(t["source"] == "random" for t in done)
    next_index = max((t["trial"] for t in done), default=-1) + 1

    gpus = visibleGpus() if pool_opts.gpus == "all" else [g.strip() for g in pool_opts.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("no GPU found (set --gpus or CUDA_VISIBLE_DEVICES)")
    n_planners = len(gpus) * pool_opts.planners_per_gpu
    workers = pool_opts.workers or max(1, availableCpus() - n_planners)
    in_flight_max = spec.trials_in_flight or max(1, math.ceil(workers / spec.episodes_per_trial))
    workers = max(1, min(workers, in_flight_max * spec.episodes_per_trial))

    print(f"[{name}] Bayesian optimization over {len(spec.dims)} dimensions, "
          f"models {' '.join(spec.models)} ({spec.model_agg}), {spec.args.n_episodes} episodes each")
    for d in spec.dims:
        print(f"    {d.name:<20} [{d.lo:g}, {d.hi:g}]  log-uniform")
    pinned = []
    if not any(n.startswith("temperature") for n in spec.names):
        pinned.append(f"temperature={spec.pinned_temperature:g}")
    if "noise_sigma" not in spec.names:
        pinned.append(f"noise_sigma={spec.pinned_noise_sigma:g}")
    if pinned:
        print(f"    pinned: {', '.join(pinned)}")
    print(f"  budget {spec.n_calls} trials: {len(spec.seeds)} seeds, {spec.n_random} random, then GP "
          f"({spec.acq_func}); {len(done)} already done")
    print(f"  pool: {len(gpus)} GPU(s) x {pool_opts.planners_per_gpu} planner(s), {workers} workers, "
          f"up to {in_flight_max} trials in flight")
    if len(done) >= spec.n_calls:
        cellSummary(index, row, outdir, spec)
        return 0

    pool = EpisodePool(spec.args, workers, gpus, pool_opts.planners_per_gpu, rollout_models=spec.models)
    in_flight: dict[int, _Trial] = {}
    n_episodes_run = 0

    def nextPoint() -> tuple[list[float], str] | None:
        nonlocal n_random_issued
        if seeds:
            s = seeds.pop(0)
            return s.x, s.source
        if n_random_issued < spec.n_random:
            n_random_issued += 1
            return randomPoint(opt), "random"
        if opt.yi:
            return askNext(opt, [t.x for t in in_flight.values()]), "gp"
        return None                                          # wait for a first result

    def launch() -> None:
        nonlocal next_index
        while len(in_flight) < in_flight_max and len(done) + len(in_flight) < spec.n_calls:
            nxt = nextPoint()
            if nxt is None:
                return
            x, source = nxt
            t = _Trial(next_index, x, source, time.time(), {m: [] for m in spec.models})
            next_index += 1
            for m in spec.models:
                planner_params, weights = spec.modelSettings(x, m)
                for ep in range(spec.args.n_episodes):
                    pool.submit(EpisodeJob(ep, rollout_model=m, planner_params=planner_params,
                                           cost_weights=weights, key=t.index))
                    t.left += 1
            in_flight[t.index] = t
            print(f"[trial {t.index:03d} {source}] {spec.describe(x)}")

    def collect(results, scoring: bool = True) -> None:
        nonlocal n_episodes_run
        n_episodes_run += len(results)
        for r in results:
            t = in_flight.get(r.job.key)
            if t is None:
                continue
            t.results[r.job.rollout_model].append(r)
            t.left -= 1
            if t.left == 0:
                del in_flight[t.index]
                if scoring:
                    finish(t)

    def finish(t: _Trial) -> None:
        summaries = {m: [r.summary for r in rs] for m, rs in t.results.items()}
        reasons = {s.get("finish_reason") for ss in summaries.values() for s in ss}
        if "interrupted" in reasons or (pool.fatal and any(r.error for rs in t.results.values() for r in rs)):
            print(f"[trial {t.index:03d}] incomplete (stopped); it will be redone on resume")
            return
        per_model = {m: scoreModel(ss, spec) for m, ss in summaries.items()}
        J = foldModels(per_model, spec.model_agg)
        record = saveTrial(spec, pool.args, cell_dir, t, per_model, J)
        opt.tell(t.x, J)
        done.append(record)
        done.sort(key=lambda d: d["trial"])
        writeState(spec, cell_dir, done)
        best = min(done, key=lambda d: d["objective"])
        models = "  ".join(f"{m} {v['objective']:+.3f}" for m, v in per_model.items()) if len(per_model) > 1 else ""
        print(f"[trial {t.index:03d}] J={J:+.4f}  success {record['success_rate']:.0%}  "
              f"goal err {record['mean_norm_goal_err']:.3f}  {models}  ({record['wall_s']:.0f} s; "
              f"{len(done)}/{spec.n_calls} done, best J={best['objective']:+.4f} at trial {best['trial']})")

    with sigtermAsInterrupt():
        try:
            pool.start()
            launch()
            while in_flight and not pool.fatal:
                collect(pool.poll(1.0))
                launch()
        except KeyboardInterrupt:
            print("\ninterrupted: closing the episodes in progress; finished trials are kept ...")
            pool.interrupt()
        finally:
            collect(pool.close())
    pool.printUtilization(n_episodes_run)
    cellSummary(index, row, outdir, spec)
    if pool.problems:
        print("\nstopped early: " + "; ".join(pool.problems) + ". Rerun into the same --outdir to resume.")
        return 1
    return 0


def runAll(csv_path: Path, rows: list[dict], outdir: Path, overwrite: bool, stop_on_error: bool,
           pool_opts: argparse.Namespace) -> int:
    """Every cell in order, each in its own process, so one failure does not stop the rest."""
    failed = []
    outdir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        name = batches.cellName(i, row)
        print(f"[{i + 1}/{len(rows)}] {name} ...", flush=True)
        cmd = [sys.executable, __file__, str(csv_path), "--cell", str(i), "--outdir", str(outdir),
               "--gpus", pool_opts.gpus, "--planners-per-gpu", str(pool_opts.planners_per_gpu)]
        if pool_opts.workers:
            cmd += ["--workers", str(pool_opts.workers)]
        if overwrite:
            cmd.append("--overwrite")
        t0 = time.time()
        with open(outdir / f"{name}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"    {'done' if rc == 0 else f'FAILED (exit {rc})'} in {time.time() - t0:.0f} s"
              f"  (log {outdir / f'{name}.log'})", flush=True)
        if rc != 0:
            failed.append(name)
            if stop_on_error:
                break
    summarize(csv_path, rows, outdir)
    if failed:
        print(f"\n{len(failed)} cell(s) stopped early: {', '.join(failed)}")
    return 1 if failed else 0


# -- summaries -------------------------------------------------------------------------
def replayCommands(row: dict[str, str], record: dict, models: list[str]) -> list[str]:
    """``run_episodes.py`` command lines that replay a trial, one per distinct temperature."""
    argv = batches.rowToArgv(row, Path("."), "x", own_columns=OWN_COLUMNS)
    flags = " ".join(argv[:argv.index("--results")])
    weights = " ".join(f"--cost-weight {k}={v:.6g}" for k, v in record.get("cost_weights", {}).items())
    groups: dict[tuple, list[str]] = {}
    for m in models:
        p = record["settings"][m]["planner_params"]
        groups.setdefault(tuple(sorted(p.items())), []).append(m)
    out = []
    for params, ms in groups.items():
        knobs = " ".join(f"--{k.replace('_', '-')} {v:.6g}" for k, v in params)
        for m in ms:
            out.append(f"python ContactModelStudy/Drivers/run_episodes.py {flags} --rollout-model {m} "
                       f"{knobs} {weights}".replace("  ", " ").strip())
    return out


def cellSummary(index: int, row: dict[str, str], outdir: Path, spec: CellSpec | None = None) -> list[dict]:
    """Rank one cell's trials, write ``bo_summary.json`` and print the best."""
    name = batches.cellName(index, row)
    cell_dir = outdir / name
    trials = _readTrials(cell_dir) if cell_dir.exists() else []
    ranked = sorted(trials, key=lambda t: t["objective"])
    for r, t in enumerate(ranked, 1):
        t["rank"] = r
    models = spec.models if spec else _tokens(row.get("rollout_models", "")) or \
        (list(trials[0]["per_model"]) if trials else [])
    best = ranked[0] if ranked else None
    if cell_dir.exists():
        (cell_dir / "bo_summary.json").write_text(json.dumps({
            "cell": index, "name": name, "row": row, "models": models,
            "n_trials": len(trials), "ranked": [t["trial"] for t in ranked],
            "trace": [t["objective"] for t in trials],
            "best_so_far": [float(v) for v in np.minimum.accumulate([t["objective"] for t in trials])]
            if trials else [],
            "best": best, "replay": replayCommands(row, best, models) if best else [],
        }, indent=2))
    print(f"\n{name}: {len(trials)} trials finished")
    if not ranked:
        return trials
    print(f"  {'rank':>4} {'trial':>5} {'source':<14} {'J':>8} {'success':>8} {'goal err':>9}  point")
    for t in ranked[:10]:
        point = "  ".join(f"{(n[2:] if n.startswith('w_') else n)}={v:.3g}" for n, v in t["params"].items())
        print(f"  {t['rank']:>4} {t['trial']:>5} {t['source']:<14} {t['objective']:>+8.4f} "
              f"{t['success_rate']:>7.0%} {t['mean_norm_goal_err']:>9.3f}  {point}")
    print(f"  best: trial {best['trial']} ({best['source']}), J={best['objective']:+.4f}. Replay with:")
    for cmd in replayCommands(row, best, models):
        print(f"    {cmd}")
    return trials


def summarize(csv_path: Path, rows: list[dict], outdir: Path) -> Path | None:
    """Every cell's trials into ``summary.csv``/``summary.json``, and each cell's best into ``best.csv``."""
    table, best = [], []
    for i, row in enumerate(rows):
        name = batches.cellName(i, row)
        trials = cellSummary(i, row, outdir)
        flat = []
        for t in trials:
            entry = {"cell": i, "cell_name": name, **row, "trial": t["trial"], "rank": t.get("rank"),
                     "source": t["source"], "objective": t["objective"], "success_rate": t["success_rate"],
                     "mean_norm_goal_err": t["mean_norm_goal_err"], "n_episodes_run": t["n_episodes"],
                     "wall_s": t.get("wall_s"),
                     **{f"x_{k}": v for k, v in t["params"].items()},
                     **{f"{m}_objective": v["objective"] for m, v in t["per_model"].items()},
                     **{f"{m}_success_rate": v["success_rate"] for m, v in t["per_model"].items()}}
            flat.append(entry)
        table += flat
        ranked = sorted(flat, key=lambda e: e["rank"])
        best.append(ranked[0] if ranked else {"cell": i, "cell_name": name, **row, "status": "no finished trials"})
    if not table:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps({"csv": str(csv_path), "trials": table, "best": best},
                                                    indent=2))
    for fname, rows_out in (("summary.csv", table), ("best.csv", best)):
        columns = list(dict.fromkeys(k for e in rows_out for k in e))
        with open(outdir / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(rows_out)
    print(f"\nsummary: {len(table)} trials across {len(rows)} cells -> "
          f"{outdir / 'summary.csv'}, {outdir / 'best.csv'}")
    return outdir / "summary.csv"


# -- CLI ---------------------------------------------------------------------------------
def defaultOutdir(csv_path: Path) -> Path:
    tag = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") \
        or time.strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / f"bayes_opt_{csv_path.stem}_{tag}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="the CSV, one row (BO run) per cell")
    p.add_argument("--cell", type=int, default=None,
                   help="run only this row (0-based, as SLURM_ARRAY_TASK_ID); omitted, run every row")
    p.add_argument("--outdir", type=Path, default=None,
                   help="where every cell writes; default results/bayes_opt_<csv>_<job id or time>")
    p.add_argument("--overwrite", action="store_true", help="discard a cell's earlier trials and start over")
    p.add_argument("--stop-on-error", action="store_true",
                   help="when running every row, stop at the first cell that stops early")
    p.add_argument("--check", action="store_true", help="validate every row (space, seeds, driver); run nothing")
    p.add_argument("--count", action="store_true", help="print the number of rows and exit")
    p.add_argument("--summarize", action="store_true", help="only (re)build the summaries in --outdir")
    g = p.add_argument_group("pool (as run_episodes_pooled.py)")
    g.add_argument("--gpus", default="all", help="comma-separated GPU ids, or 'all' (default) for every visible GPU")
    g.add_argument("--planners-per-gpu", type=int, default=1, help="planner processes per GPU")
    g.add_argument("--workers", type=int, default=None,
                   help="eval-sim worker processes; default: available cores minus one per planner, "
                        "and no more than the trials in flight can use")
    args = p.parse_args(argv)
    if args.planners_per_gpu < 1:
        p.error("--planners-per-gpu must be >= 1")
    if args.workers is not None and args.workers < 1:
        p.error("--workers must be >= 1")

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        p.error(f"no such CSV: {csv_path}")
    try:
        batches.checkColumns(csv_path, own_columns=OWN_COLUMNS, managed=MANAGED_OPTIONS)
    except ValueError as e:
        p.error(str(e))
    rows = batches.readRows(csv_path)
    if args.count:
        print(len(rows))
        return 0
    if not rows:
        p.error(f"{csv_path} has no data rows")
    outdir = (args.outdir or defaultOutdir(csv_path)).resolve()

    if args.check:
        bad = 0
        for i, row in enumerate(rows):
            name = batches.cellName(i, row)
            try:
                spec = parseCell(row, csv_path.parent, outdir / name)
                err = None
                detail = (f"{len(spec.dims)} dims, {spec.n_calls} trials x {len(spec.models)} model(s) x "
                          f"{spec.args.n_episodes} episodes, {len(spec.seeds)} seed(s)")
            except (ValueError, FileNotFoundError) as e:
                err, detail = str(e), ""
            bad += err is not None
            print(f"{'OK ' if err is None else 'BAD'} {name}: " + (err or detail))
        print(f"\n{len(rows) - bad}/{len(rows)} rows valid")
        return 1 if bad else 0
    if args.summarize:
        return 0 if summarize(csv_path, rows, outdir) else 1
    if args.cell is not None:
        if not 0 <= args.cell < len(rows):
            p.error(f"--cell {args.cell} is out of range: {csv_path.name} has rows 0-{len(rows) - 1}")
        try:
            return runCell(args.cell, rows[args.cell], csv_path.parent, outdir, args.overwrite, args)
        except ValueError as e:
            print(f"cell {args.cell}: {e}", file=sys.stderr)
            return 2
    print(f"{len(rows)} cells from {csv_path} -> {outdir}")
    return runAll(csv_path, rows, outdir, args.overwrite, args.stop_on_error, args)


if __name__ == "__main__":
    raise SystemExit(main())
