#!/usr/bin/env python3
"""How a cell's regret changes as its rollout model's contact parameters move, cell by cell.

For a cell of an episode batch (``run_episode_batches.py``), at each analysed
step of the episodes chosen (``--episodes``, every one by default), it:

1. plans once with the *zero-gap* planner, the cell's planner on the eval
   model, exactly as ``process_batch_results.py`` builds it (``optimalArgs``
   with no overrides, Kamino at ``DEFAULT_KAMINO``): its final mean ``U_A``;
2. plans again with the cell's own planner, its rollout model rebuilt at
   every point θ of a grid over the model's critical contact parameters
   (``SWEEP_PARAMS``): its final mean ``U_B(θ)``;
3. scores each ``U_B(θ)`` and ``U_A`` on the eval model, ``F_A(U_B(θ))`` and
   ``F_A(U_A)``, and their difference, the regret of planning with θ.

Every plan starts from what the episode's planner recorded for that step (its
mean, noise draw, temperature and previous control; ``planStart``), so every
grid point and the zero-gap planner draw the same noise (common random
numbers), and the regret moves with θ rather than with sampling luck. The
scores are those of ``process_batch_results``: ``PairedFreeEnergy`` on one
evaluation noise block per step (``evalNoiseSeed``), here with ``--score-block``
means per rollout, ``U_A`` and ``B - 1`` grid points, so each regret(θ) is a
difference within one rollout. The nominal point, the cell's own parameters,
is always in the grid; its regret is the cell's ``process_batch_results``
regret, up to the eval model's run-to-run noise.

The grid is the full product of each parameter's values (``SWEEP_PARAMS``):

=====  ===================================  ===========================================
Model  Parameters                           Default values
=====  ===================================  ===========================================
M1     solref_timeconst, solref_dampratio   timeconst x {1, 1.5, 2, 4, 8} (nominal 2 dt is
                                            its floor); dampratio x {0.25, 0.5, 1, 2, 4}
M2     solref_timeconst, solref_dampratio   timeconst x {0.4, 0.6, 1, 2, 4} (0.4 x 0.02 s
                                            is 2 dt at 4 ms); dampratio x {0.25, 0.5, 1, 2, 4}
M3     stiffness, damping                   x {0.25, 0.5, 1, 2, 4} each
M4     xpbd_iterations, relaxation          {1, 2, 3, 4, 6}; {1, 1.5, 2, 2.5, 3, 3.5}
M6     alpha, gamma, refinement             x {0.25, 0.5, 1, 2, 4}; x the same; {0, ..., 4}
=====  ===================================  ===========================================

A multiplier is on the parameter's nominal value: the preset, with the task's
tuning (``CONTACT_PARAMS``) and M1's ``2 * dt`` resolved; M2's solref, which the
preset leaves to the scene, is read from the rollout scene. ``--values
NAME=v1,v2`` and ``--multipliers NAME=m1,m2`` replace a parameter's values
(``M3:NAME=...`` for one model only). A point the model's config refuses, or
with ``solref_timeconst`` below ``2 * dt``, is dropped and recorded. A cell
whose rollout model is M5, the ground truth, is skipped.

::

    python experiments/contact_param_sweep.py <batch dir> --count
    python experiments/contact_param_sweep.py <batch dir> --cell 3 --episodes 0-4,9
    python experiments/contact_param_sweep.py <batch dir> --cell 3 --multipliers stiffness=0.5,1,2
    python experiments/contact_param_sweep.py <batch dir>                # every cell, in order
    python experiments/contact_param_sweep.py <batch dir> --summarize    # final table only

On the HPC, ``hpc/submit_contact_param_sweep.sh <batch dir>`` runs one array
task per cell and then the summary.

Output, in the batch folder
---------------------------
* ``sweep/<cell>.partial.npz``: the cell's checkpoint while it runs, rewritten
  after the zero-gap plans, each grid point's plans and each block of scores,
  so a cell cut off by the wall clock resumes from its last finished unit;
  deleted once the cell is done. Kept only while its ``sweep_key`` (the grid,
  the episodes and rows, the Kamino settings and ``--score-block``) matches.
* ``sweep/<cell>.sweep.npz``: per analysed step (T) and grid point (G): ``episode``,
  ``episode_id``, ``row``, ``step``, ``q``, ``q_dot``; ``param_names``, ``params``
  (G x P), ``multipliers``, ``nominal_index``; ``f_used`` / ``f_used_se`` (T x G),
  ``f_zero`` / ``f_zero_se`` (T x blocks: ``F_A(U_A)`` in each block's rollout),
  ``block`` (each grid point's block), ``regret`` (T x G); ``U_used`` (T x G x H x
  nu), ``U_zero``, ``mu_used`` (T x G x nu), ``mu_zero``; the plan times
  ``plan_s_used`` / ``plan_s_zero``, iterations ``n_iter_used`` / ``n_iter_zero``,
  each block's ``score_s`` and each grid point's simulator ``build_s``;
* ``sweep/<cell>.sweep.json``: how it ran (models, nominal parameters, the grid
  and what was dropped, the episodes, the zero-gap planner) and, per grid
  point, the regret's mean with its 95% interval, median and p90 over the
  analysed steps (``process_batch_results.stats``);
* ``sweep/<cell>.sweep_status.json``: done, skipped or failed, with the error;
* ``sweep_results.csv`` / ``.json``: one row per (cell, grid point).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import process_batch_results as pbr  # noqa: E402

SWEEP_DIR = "sweep"

#: The ground-truth model: never swept.
GROUND_TRUTH = "M5"

#: Each model's swept parameters, in grid order: ``(kind, values)``, ``kind`` ``"mult"`` (values multiply
#: the nominal) or ``"abs"`` (values as given). M1's time constant is at its 2 * dt floor already, so it
#: only goes up, and M2's (the scene's 0.02 s) goes down to 0.4x, the floor at a 4 ms rollout step; XPBD
#: goes NaN from a relaxation of 4 (``ContactModelPresets``, M4).
SWEEP_PARAMS: dict[str, dict[str, tuple[str, tuple]]] = {
    "M1": {"solref_timeconst": ("mult", (1.0, 1.5, 2.0, 4.0, 8.0)),
           "solref_dampratio": ("mult", (0.25, 0.5, 1.0, 2.0, 4.0))},
    "M2": {"solref_timeconst": ("mult", (0.4, 0.6, 1.0, 2.0, 4.0)),
           "solref_dampratio": ("mult", (0.25, 0.5, 1.0, 2.0, 4.0))},
    "M3": {"stiffness": ("mult", (0.25, 0.5, 1.0, 2.0, 4.0)),
           "damping": ("mult", (0.25, 0.5, 1.0, 2.0, 4.0))},
    "M4": {"xpbd_iterations": ("abs", (1, 2, 3, 4, 6)),
           "relaxation": ("abs", (1.0, 1.5, 2.0, 2.5, 3.0, 3.5))},
    "M6": {"alpha": ("mult", (0.25, 0.5, 1.0, 2.0, 4.0)),
           "gamma": ("mult", (0.25, 0.5, 1.0, 2.0, 4.0)),
           "refinement": ("abs", (0, 1, 2, 3, 4))},
}

#: The parameters that take whole numbers.
INT_PARAMS = {"xpbd_iterations", "refinement"}

#: The scores' rollout: ``U_A`` and ``DEFAULT_SCORE_BLOCK - 1`` grid points, in that many times the worlds.
DEFAULT_SCORE_BLOCK = 4


def _paths(batch_dir: Path, name: str) -> dict[str, Path]:
    d = batch_dir / SWEEP_DIR
    return {"partial": d / f"{name}.partial.npz", "npz": d / f"{name}.sweep.npz",
            "json": d / f"{name}.sweep.json", "status": d / f"{name}.sweep_status.json"}


def sweepDone(batch_dir: Path, name: str) -> bool:
    return (pbr._readJson(_paths(batch_dir, name)["status"]) or {}).get("status") in ("done", "skipped")


# -- the grid ------------------------------------------------------------------------------
def canonicalModel(model: str) -> str:
    from ContactModelStudy.Utils.ContactModelPresets import getContactModel
    return getContactModel(model).name


def nominalParams(model: str, task, sim_overrides: dict) -> dict:
    """The swept parameters' values as the cell ran them: the preset, the task's tuning, M1's resolved
    ``2 * dt``, and, for M2's solref (``None`` in its preset), the rollout scene's own geom solref."""
    from ContactModelStudy.Utils.ContactModelPresets import _presetConfig
    model = canonicalModel(model)
    cfg = _presetConfig(model, task, **sim_overrides)
    out = {p: getattr(cfg, p) for p in SWEEP_PARAMS[model]}
    if any(v is None for v in out.values()):
        out.update({k: v for k, v in sceneSolref(task.getModelPath()).items() if out.get(k) is None})
    return out


def sceneSolref(xml: str) -> dict:
    """``{solref_timeconst, solref_dampratio}``: the most common geom solref in a scene."""
    import mujoco
    rows, counts = np.unique(mujoco.MjModel.from_xml_path(str(xml)).geom_solref, axis=0, return_counts=True)
    timeconst, dampratio = rows[np.argmax(counts)]
    return {"solref_timeconst": float(timeconst), "solref_dampratio": float(dampratio)}


def parseValueSpecs(specs: list[str] | None, flag: str) -> list[tuple[str | None, str, list[float]]]:
    """``--values`` / ``--multipliers`` entries as ``(model or None, parameter, values)``.

    Each is ``NAME=v1,v2,...`` or ``MODEL:NAME=v1,...``. Raises ``ValueError`` on
    a malformed entry, or a name no model sweeps.
    """
    out = []
    known = {p for params in SWEEP_PARAMS.values() for p in params}
    for spec in specs or []:
        m = re.fullmatch(r"\s*(?:(\w+):)?(\w+)=([^=]+)", spec)
        if not m:
            raise ValueError(f"{flag} {spec!r}: expected NAME=v1,v2,... or MODEL:NAME=v1,v2,...")
        model, name, values = m.groups()
        if model is not None:
            model = canonicalModel(model)
            if name not in SWEEP_PARAMS.get(model, {}):
                raise ValueError(f"{flag} {spec!r}: {model} sweeps {', '.join(SWEEP_PARAMS.get(model, {})) or 'nothing'}")
        elif name not in known:
            raise ValueError(f"{flag} {spec!r}: no model sweeps {name!r}; the swept parameters are "
                             f"{', '.join(sorted(known))}")
        try:
            vals = [float(v) for v in values.split(",") if v.strip()]
        except ValueError:
            raise ValueError(f"{flag} {spec!r}: the values must be numbers") from None
        if not vals:
            raise ValueError(f"{flag} {spec!r}: no values")
        out.append((model, name, vals))
    return out


def _specsFor(model: str, specs) -> dict[str, list[float]]:
    """The entries of ``parseValueSpecs`` that apply to ``model``, by parameter (a later one wins)."""
    return {name: vals for m, name, vals in specs if (m is None or m == model) and name in SWEEP_PARAMS[model]}


def _dedupe(values: list[float]) -> list[float]:
    out = []
    for v in sorted(values):
        if not out or not np.isclose(v, out[-1], rtol=1e-9, atol=0.0):
            out.append(v)
    return out


def paramValues(model: str, nominal: dict, values=None, multipliers=None) -> dict[str, list[float]]:
    """Each swept parameter's values, nominal included, sorted: the defaults, or ``values`` (absolute)
    and ``multipliers`` (on the nominal), each ``parseValueSpecs`` output; whole numbers where needed."""
    model = canonicalModel(model)
    absolute, mults = _specsFor(model, values or []), _specsFor(model, multipliers or [])
    out = {}
    for name, (kind, default) in SWEEP_PARAMS[model].items():
        nom = nominal[name]
        if name in absolute:
            vals = list(absolute[name])
        elif name in mults or kind == "mult":
            vals = [nom * m for m in mults.get(name, default)]
        else:
            vals = list(default)
        vals.append(nom)
        if name in INT_PARAMS:
            vals = [float(round(v)) for v in vals]
        out[name] = _dedupe(vals)
    return out


def buildGrid(model: str, nominal: dict, task, sim_overrides: dict, values=None,
              multipliers=None) -> tuple[list[dict], list[dict]]:
    """``(grid, dropped)``: every combination of ``paramValues``, as config overrides, that the model's
    config accepts and that keeps ``solref_timeconst`` at or above ``2 * dt``; and the ones it drops,
    each with its ``reason``."""
    from ContactModelStudy.Utils.ContactModelPresets import getContactModel
    model = canonicalModel(model)
    per_param = paramValues(model, nominal, values, multipliers)
    names = list(per_param)
    floor = 2.0 * float(task.timestep)
    grid, dropped = [], []
    for combo in itertools.product(*(per_param[n] for n in names)):
        theta = {n: int(v) if n in INT_PARAMS else float(v) for n, v in zip(names, combo)}
        reason = None
        if theta.get("solref_timeconst", floor) < floor * (1 - 1e-9):
            reason = f"solref_timeconst {theta['solref_timeconst']:g} s is below 2 * dt = {floor:g} s"
        else:
            try:
                getContactModel(model).makeConfig(task, **sim_overrides, **theta)
            except (ValueError, TypeError) as e:
                reason = str(e)
        (dropped.append({**theta, "reason": reason}) if reason else grid.append(theta))
    return grid, dropped


def multipliersOf(grid: list[dict], nominal: dict) -> np.ndarray:
    """Each grid point's values over the nominal, ``(G, P)``."""
    names = list(grid[0]) if grid else []
    return np.array([[theta[n] / nominal[n] if nominal[n] else np.nan for n in names] for theta in grid],
                    dtype=float).reshape(len(grid), len(names))


def nominalIndex(grid: list[dict], nominal: dict) -> int:
    for g, theta in enumerate(grid):
        if all(np.isclose(theta[n], nominal[n], rtol=1e-9, atol=0.0) for n in theta):
            return g
    raise ValueError("the nominal parameters are not in the grid (the model's config refuses them?)")


def blocksOf(n_grid: int, score_block: int) -> list[list[int]]:
    """The grid points each scoring rollout takes: ``score_block - 1`` at a time, beside ``U_A``."""
    k = score_block - 1
    return [list(range(i, min(i + k, n_grid))) for i in range(0, n_grid, k)]


# -- the episodes ------------------------------------------------------------------------------
def episodeIndex(ep, position: int) -> int:
    """An episode's index in its batch, as ``process_batch_results`` numbers it."""
    return int(ep.summary.get("episode", position))


def selectEpisodes(episodes: list, spec: str | None, max_episodes: int | None = None) -> list[int]:
    """The positions in ``episodes`` that ``spec`` names, in the cell's order, capped at ``max_episodes``.

    ``spec`` is a comma list of episode indices (``episodeIndex``), inclusive
    ranges ``a-b`` and recorded episode ids, e.g. ``"0,3,7-9"``; ``None`` or
    empty, every episode. A number is an index first and an id only if no
    episode has that index. Raises ``ValueError`` naming what the cell does not have.
    """
    positions = list(range(len(episodes)))
    if spec and spec.strip():
        by_index = {episodeIndex(ep, k): k for k, ep in enumerate(episodes)}
        by_id = {ep.id: k for k, ep in enumerate(episodes)}
        chosen, missing = set(), []
        for token in (t.strip() for t in spec.split(",")):
            if not token:
                continue
            m = re.fullmatch(r"(\d+)-(\d+)", token)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                if lo > hi:
                    raise ValueError(f"--episodes: the range {token} runs backwards")
                for i in range(lo, hi + 1):
                    (chosen.add(by_index[i]) if i in by_index else missing.append(str(i)))
            elif token.isdigit() and int(token) in by_index:
                chosen.add(by_index[int(token)])
            elif token in by_id:
                chosen.add(by_id[token])
            else:
                missing.append(token)
        if missing:
            raise ValueError(f"--episodes: the cell has no episode {', '.join(missing)} (it has indices "
                             f"{', '.join(str(i) for i in sorted(by_index))})")
        positions = sorted(chosen)
    return positions[:max_episodes] if max_episodes else positions


# -- planning ----------------------------------------------------------------------------------
class StepPlanner:
    """One ``(task, planner)`` pair planning analysed steps one by one, each from its recorded start.

    As ``process_batch_results.StepPlanners`` plans each of its planners: load
    the start, plan from the recorded state toward the step's goal, and keep
    the first-action mean and the final mean.
    """

    def __init__(self, pair, reset_mean):
        self.task, self.planner = pair
        self._goal = None
        self.planner.Reset(reset_mean)

    def setGoal(self, goal: np.ndarray) -> None:
        if self._goal is None or not np.array_equal(self._goal, goal):
            self.task.setGoal(goal)
            self._goal = np.array(goal, copy=True)

    def plan(self, goal, start: dict, q, q_dot) -> tuple[np.ndarray, np.ndarray, float, int]:
        """``(mu, U, plan time, iterations)`` from ``start``; ``mu`` and ``U`` NaN when the plan failed."""
        p = self.planner
        self.setGoal(goal)
        p.LoadState({**p.SaveState(), **start})
        mu, _, plan_s = pbr._timedPlan(p, q, q_dot, start.get("u_prev"))
        U = p.last_mean if np.isfinite(mu).all() and p.last_mean is not None else None
        U = np.full((p.horizon, p.nu), np.nan, dtype=np.float32) if U is None else np.asarray(U, np.float32)
        return mu, U, plan_s, int(p.last_n_iterations)


def _planAll(stepper: StepPlanner, steps: dict, starts: list[dict]) -> dict[str, np.ndarray]:
    """``mu`` (T, nu), ``U`` (T, H, nu), ``plan_s`` and ``n_iter`` (T) of every analysed step."""
    mus, Us, times, iters = [], [], [], []
    for t, start in enumerate(starts):
        mu, U, s, n = stepper.plan(steps["goal"][t], start, steps["q"][t], steps["q_dot"][t])
        mus.append(mu)
        Us.append(U)
        times.append(s)
        iters.append(n)
    return {"mu": np.array(mus), "U": np.array(Us), "plan_s": np.array(times), "n_iter": np.array(iters)}


# -- one cell --------------------------------------------------------------------------------
def _sweepKey(model: str, grid: list[dict], episode_ids: list[str], rows: list[np.ndarray],
              eval_kamino: dict, score_block: int) -> str:
    """What a checkpoint was computed under: it is resumed only when this matches."""
    return json.dumps({"model": model, "grid": grid, "episodes": episode_ids,
                       "rows": [np.asarray(r).tolist() for r in rows],
                       "kamino": dict(sorted((eval_kamino or {}).items())), "score_block": score_block})


def _loadCheckpoint(path: Path, key: str) -> dict | None:
    try:
        with np.load(path) as f:
            d = dict(f)
    except Exception:                                   # missing, or cut short
        return None
    return d if str(d.get("sweep_key")) == key else None


def _newCheckpoint(key: str, T: int, G: int, n_blocks: int, H: int, nu: int) -> dict:
    nan = lambda *shape: np.full(shape, np.nan)  # noqa: E731
    return {"sweep_key": np.array(key), "zero_done": np.array(False),
            "U_zero": nan(T, H, nu).astype(np.float32), "mu_zero": nan(T, nu), "plan_s_zero": nan(T),
            "n_iter_zero": nan(T),
            "plans_done": np.zeros(G, dtype=bool), "U_used": nan(T, G, H, nu).astype(np.float32),
            "mu_used": nan(T, G, nu), "plan_s_used": nan(T, G), "n_iter_used": nan(T, G), "build_s": nan(G),
            "blocks_done": np.zeros(n_blocks, dtype=bool), "f_used": nan(T, G), "f_used_se": nan(T, G),
            "f_zero": nan(T, n_blocks), "f_zero_se": nan(T, n_blocks), "score_s": nan(T, n_blocks)}


def checkpointProgress(ck: dict) -> str:
    """Where a checkpoint has got to, in words."""
    return (f"zero-gap plans {'done' if ck['zero_done'] else 'not done'}, "
            f"{int(ck['plans_done'].sum())}/{len(ck['plans_done'])} grid points planned, "
            f"{int(ck['blocks_done'].sum())}/{len(ck['blocks_done'])} score blocks done")


class CellSkipped(Exception):
    """A cell this sweep does not apply to (its rollout model is the ground truth)."""


def processCell(batch_dir: Path, cell: dict, stride: int = 1, n_steps_analysed: int | None = 20,
                episodes_spec: str | None = None, max_episodes: int | None = None, values=None,
                multipliers=None, score_block: int = DEFAULT_SCORE_BLOCK, overwrite: bool = False,
                kamino: dict | None = pbr.DEFAULT_KAMINO) -> dict:
    """One cell swept: every phase not yet in its checkpoint run and saved, then its results written.

    ``values`` and ``multipliers`` are ``parseValueSpecs`` output. Returns the
    cell's record (``sweep/<cell>.sweep.json``). Raises ``CellSkipped`` for an
    M5 cell.
    """
    from ContactModelStudy.Drivers import run_episodes as drv
    from ContactModelStudy.Drivers.EpisodePool import episodeSeeds
    from ContactModelStudy.Utils.ContactModelPresets import _presetConfig
    from ContactModelStudy.Utils.EpisodeRecorder import EpisodeReplayer

    if score_block < 2:
        raise ValueError(f"score_block must be >= 2 (U_A and one grid point), got {score_block}")
    name = cell["name"]
    res = batch_dir / f"{name}.json"
    if not res.exists():
        raise ValueError(f"no results file {res.name} (the batch cell status is "
                         f"{cell['status'].get('status')!r})")
    rep = EpisodeReplayer(res)
    cli_args = rep.configs.get("metadata", {}).get("cli_args")
    if not cli_args:
        raise ValueError(f"{res.name} does not record the command line it was run with")
    args = pbr.argsFromCliArgs(cli_args)
    model = canonicalModel(args.rollout_model)
    if model == GROUND_TRUTH:
        raise CellSkipped(f"the rollout model is {model}, the ground truth: there is nothing to sweep")
    if model not in SWEEP_PARAMS:
        raise ValueError(f"no parameters to sweep for {model}")
    pbr.checkCell(rep, args)
    episodes = list(rep)
    chosen = selectEpisodes(episodes, episodes_spec, max_episodes)
    if not chosen:
        raise ValueError("no episodes selected")

    # The grid, at the cell's own rollout settings.
    task, eval_task = drv.buildTasks(args)
    sim_overrides = drv.rolloutSimOverrides(args, task)
    nominal = nominalParams(model, task, sim_overrides)
    grid, dropped = buildGrid(model, nominal, task, sim_overrides, values, multipliers)
    if not grid:
        raise ValueError(f"every grid point was dropped: {dropped}")
    g_nom = nominalIndex(grid, nominal)
    blocks = blocksOf(len(grid), score_block)

    # The analysed steps, every chosen episode's in turn.
    goals_by_id = dict(zip(rep.ids, pbr.episodeGoals(rep, args)))
    _, _, u0 = eval_task.getInitialState()
    steps = {k: [] for k in ("episode", "episode_id", "row", "step", "goal", "q", "q_dot")}
    starts, all_rows = [], []
    for k in chosen:
        ep = episodes[k]
        rows = pbr.analysedRows(ep.n_saved, stride, n_steps_analysed)
        ep_index = episodeIndex(ep, k)
        noise_seed = episodeSeeds(args.seed, ep_index)[1]
        ep_starts = [pbr.planStart(ep, i, args, u0, noise_seed) for i in rows]
        if pbr.replayMode(ep, ep_starts, len(rows) == ep.n_saved, args.warm_start) != "independent":
            raise ValueError(f"episode {ep.id} did not save the mean each plan started from, so its steps "
                             f"cannot be replanned one by one: rerun the batch with this version")
        goals = goals_by_id[ep.id]
        gidx = pbr.goalIndexAt(goals, ep.step[rows])
        all_rows.append(rows)
        starts += ep_starts
        for j, i in enumerate(rows):
            for key, v in (("episode", ep_index), ("episode_id", ep.id), ("row", i), ("step", ep.step[i]),
                           ("goal", goals[gidx[j]][1]), ("q", ep.q[i]), ("q_dot", ep.q_dot[i])):
                steps[key].append(v)
    steps = {k: np.array(v) for k, v in steps.items()}
    T = len(starts)

    # The zero-gap planner's settings, as process_batch_results builds them.
    used_cfg = _presetConfig(model, task, **sim_overrides)
    zargs = pbr.optimalArgs(args, used_cfg.control_timestep, used_cfg.horizon_duration)
    eval_kamino = pbr.kaminoOverrides(zargs.rollout_model, kamino)
    ztask, _ = drv.buildTasks(zargs)
    zero_cfg = _presetConfig(zargs.rollout_model, ztask, **drv.rolloutSimOverrides(zargs, ztask), **eval_kamino)
    H, nu = used_cfg.resolved_horizon, len(u0)
    reset_mean = u0 if args.control_mode == "absolute" else None

    episode_ids = [episodes[k].id for k in chosen]
    key = _sweepKey(model, grid, episode_ids, all_rows, eval_kamino, score_block)
    paths = _paths(batch_dir, name)
    paths["partial"].parent.mkdir(parents=True, exist_ok=True)
    ck = None if overwrite else _loadCheckpoint(paths["partial"], key)
    if ck is None:
        ck = _newCheckpoint(key, T, len(grid), len(blocks), H, nu)
    else:
        print(f"[{name}] resuming: {checkpointProgress(ck)}", flush=True)
    save = lambda: pbr._saveNpz(paths["partial"], **ck)  # noqa: E731
    print(f"[{name}] {model} rollout vs {args.eval_sim} eval: {len(grid)} grid points over "
          f"{', '.join(grid[0])} (nominal {nominal}; {len(dropped)} dropped), {len(chosen)} episodes, "
          f"{T} analysed steps, scored {score_block - 1} grid points per rollout", flush=True)

    zero = pair = None
    try:
        # 1. The zero-gap plans.
        if not ck["zero_done"]:
            zero = _zeroPlanner(zargs, eval_kamino, H)
            t0 = time.time()
            out = _planAll(StepPlanner(zero, reset_mean), steps, starts)
            ck.update(U_zero=out["U"], mu_zero=out["mu"], plan_s_zero=out["plan_s"], n_iter_zero=out["n_iter"],
                      zero_done=np.array(True))
            save()
            print(f"[{name}] zero-gap plans: {time.time() - t0:.0f} s", flush=True)
        # 2. Every grid point's plans, one rollout simulator at a time.
        for g, theta in enumerate(grid):
            if ck["plans_done"][g]:
                continue
            t0 = time.time()
            used, sim = _usedPlanner(args, task, sim_overrides, theta)
            build_s = time.time() - t0
            try:
                out = _planAll(StepPlanner(used, reset_mean), steps, starts)
            finally:
                sim.Close()
            ck["U_used"][:, g], ck["mu_used"][:, g] = out["U"], out["mu"]
            ck["plan_s_used"][:, g], ck["n_iter_used"][:, g], ck["build_s"][g] = out["plan_s"], out["n_iter"], build_s
            ck["plans_done"][g] = True
            save()
            print(f"[{name}] grid point {g + 1}/{len(grid)} {theta}: {time.time() - t0:.0f} s "
                  f"(plans {np.nanmean(out['plan_s']):.2f} s/step)", flush=True)
        # 3. The scores: U_A and a block of grid points per rollout, on each step's evaluation noise.
        if not ck["blocks_done"].all():
            zero = zero or _zeroPlanner(zargs, eval_kamino, H)
            pair = _pairPlanner(zargs, eval_kamino, score_block)
            regret = pbr.PairedFreeEnergy(zero, pair)
            steppers = [StepPlanner(zero, reset_mean), StepPlanner(pair, reset_mean)]
            for b, idx in enumerate(blocks):
                if ck["blocks_done"][b]:
                    continue
                t0 = time.time()
                for t, start in enumerate(starts):
                    s0 = time.perf_counter()
                    _scoreStep(ck, regret, steppers, steps, t, start, b, idx)
                    ck["score_s"][t, b] = time.perf_counter() - s0
                ck["blocks_done"][b] = True
                save()
                print(f"[{name}] score block {b + 1}/{len(blocks)}: {time.time() - t0:.0f} s", flush=True)
    finally:
        for pp in (zero, pair):
            if pp is not None:
                pp[1].sim.Close()

    record = _finalize(batch_dir, cell, ck, steps, grid, dropped, nominal, g_nom, blocks, model, args,
                       zargs, ztask, zero_cfg, eval_kamino, episode_ids, chosen, episodes, stride,
                       n_steps_analysed, score_block)
    paths["partial"].unlink(missing_ok=True)
    return record


def _zeroPlanner(zargs, eval_kamino: dict, horizon: int):
    """The zero-gap ``(task, planner)``: the cell's planner on the eval model."""
    zero = _evalPlanner(zargs, eval_kamino, zargs.n_samples)
    if zero[1].horizon != horizon:
        zero[1].sim.Close()
        raise ValueError(f"the used planner's mean is {horizon} long, the zero-gap one's {zero[1].horizon}: "
                         f"the regret scores the used plan on the zero-gap planner, so both need the same "
                         f"control step and horizon")
    return zero


def _pairPlanner(zargs, eval_kamino: dict, score_block: int):
    """The zero-gap planner with ``score_block`` times the worlds, for ``PairedFreeEnergy``."""
    return _evalPlanner(zargs, eval_kamino, score_block * zargs.n_samples)


def _evalPlanner(zargs, eval_kamino: dict, n_worlds: int):
    from ContactModelStudy.Drivers import run_episodes as drv
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI
    from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
    ztask, _ = drv.buildTasks(zargs)
    sim = GetContactModelSim(zargs.rollout_model, ztask.getModelPath(), N=n_worlds, task=ztask,
                             **drv.rolloutSimOverrides(zargs, ztask), **(eval_kamino or {}))
    return ztask, MPPI(sim, ztask, drv.buildPlannerConfig(zargs))


def _usedPlanner(args, task, sim_overrides: dict, theta: dict):
    """The cell's own ``(task, planner)``, its rollout model at ``theta``; and its simulator."""
    from ContactModelStudy.Drivers import run_episodes as drv
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI
    from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
    sim = GetContactModelSim(args.rollout_model, task.getModelPath(), N=args.n_samples, task=task,
                             **sim_overrides, **theta)
    return (task, MPPI(sim, task, drv.buildPlannerConfig(args))), sim


def _scoreStep(ck: dict, regret, steppers: list[StepPlanner], steps: dict, t: int, start: dict, b: int,
               idx: list[int]) -> None:
    """Block ``b`` of step ``t``: ``F_A(U_A)`` and ``F_A(U_B(θ))`` for the grid points ``idx``, in one
    rollout, into ``ck``. A failed plan's mean is scored as ``U_A`` and its result left NaN."""
    U_zero = ck["U_zero"][t]
    if not np.isfinite(U_zero).all():
        return
    for s in steppers:
        s.setGoal(steps["goal"][t])
    ok = [bool(np.isfinite(ck["U_used"][t, g]).all()) for g in idx]
    means = [U_zero] + [ck["U_used"][t, g] if good else U_zero for g, good in zip(idx, ok)]
    eval_start = {**start, "noise_seed": pbr.evalNoiseSeed(start), "resample_count": 0}
    scores = regret(eval_start, steps["q"][t], steps["q_dot"][t], means, u=start.get("u_prev"))
    ck["f_zero"][t, b], ck["f_zero_se"][t, b] = scores[0]
    for (f, se), g, good in zip(scores[1:], idx, ok):
        if good:
            ck["f_used"][t, g], ck["f_used_se"][t, g] = f, se


def _finalize(batch_dir, cell, ck, steps, grid, dropped, nominal, g_nom, blocks, model, args, zargs, ztask,
              zero_cfg, eval_kamino, episode_ids, chosen, episodes, stride, n_steps_analysed,
              score_block) -> dict:
    """The cell's ``sweep.npz`` and ``sweep.json`` from its finished checkpoint; returns the record."""
    name = cell["name"]
    names = list(grid[0])
    block_of = np.empty(len(grid), dtype=int)
    for b, idx in enumerate(blocks):
        block_of[idx] = b
    f_zero_own = ck["f_zero"][:, block_of]                    # (T, G): U_A in each grid point's own rollout
    regret = ck["f_used"] - f_zero_own
    mults = multipliersOf(grid, nominal)
    arrays = {
        **steps,
        "param_names": np.array(names), "params": np.array([[theta[n] for n in names] for theta in grid], float),
        "multipliers": mults, "nominal_index": g_nom, "block": block_of, "regret": regret,
        **{k: ck[k] for k in ("f_used", "f_used_se", "f_zero", "f_zero_se", "U_used", "U_zero", "mu_used",
                              "mu_zero", "plan_s_used", "plan_s_zero", "n_iter_used", "n_iter_zero",
                              "score_s", "build_s")},
        "sweep_key": ck["sweep_key"],
    }
    paths = _paths(batch_dir, name)
    pbr._saveNpz(paths["npz"], **arrays)
    with np.errstate(invalid="ignore"):
        # How much F_A(U_A) differs between rollouts of the same noise: the eval model's own run-to-run noise.
        f_zero_spread = np.nanstd(ck["f_zero"], axis=1) if len(blocks) > 1 else np.zeros(len(regret))
    points = []
    for g, theta in enumerate(grid):
        points.append({
            "index": g, **{f"p_{n}": theta[n] for n in names}, **{f"m_{n}": float(mults[g, j]) for j, n in
                                                                  enumerate(names)},
            "is_nominal": g == g_nom, "block": int(block_of[g]),
            **pbr.stats("regret", regret[:, g]), **pbr.stats("f_used", ck["f_used"][:, g]),
            **pbr.stats("f_zero", f_zero_own[:, g]), **pbr.stats("plan_s_used", ck["plan_s_used"][:, g]),
            **pbr.stats("n_iter_used", ck["n_iter_used"][:, g]), "build_s": float(ck["build_s"][g]),
        })
    record = {
        "cell": cell["index"], "name": name, "rollout_model": model, "eval_sim": args.eval_sim,
        "zero_gap_planner": pbr._plannerRecord(zargs, ztask, zero_cfg), "kamino": eval_kamino or None,
        "param_names": names, "nominal": nominal, "nominal_index": g_nom, "n_grid": len(grid),
        "dropped": dropped, "score_block": score_block, "n_blocks": len(blocks),
        "episodes": [episodeIndex(episodes[k], k) for k in chosen], "episode_ids": episode_ids,
        "stride": None if n_steps_analysed is not None else stride, "n_steps_analysed": n_steps_analysed,
        "n_steps_analyzed": int(len(regret)),
        "plan_s_zero_mean": float(np.nanmean(ck["plan_s_zero"])),
        "score_s_mean": float(np.nanmean(ck["score_s"])), "f_zero_spread_mean": float(np.nanmean(f_zero_spread)),
        "points": points,
    }
    pbr._writeJson(paths["json"], record)
    nom = points[g_nom]
    best = min((p for p in points if p["regret_mean"] is not None), key=lambda p: p["regret_mean"], default=None)
    print(f"[{name}] done: nominal regret {pbr._fmtInterval(nom, 'regret_mean')}"
          + (f"; lowest {pbr._fmt(best['regret_mean'])} at {grid[best['index']]}" if best else ""), flush=True)
    return record


def runCell(batch_dir: Path, cell: dict, overwrite: bool, **kw) -> int:
    """Sweep one cell, recording its status; returns 0 when done or skipped."""
    name = cell["name"]
    if not overwrite and sweepDone(batch_dir, name):
        print(f"[{name}] sweep already done; skipping (--overwrite to redo)")
        return 0
    paths = _paths(batch_dir, name)
    paths["status"].parent.mkdir(parents=True, exist_ok=True)
    status = {"cell": cell["index"], "name": name, "host": os.uname().nodename,
              "slurm_job": os.environ.get("SLURM_JOB_ID"), **kw}
    t0 = time.time()
    try:
        processCell(batch_dir, cell, overwrite=overwrite, **kw)
        status["status"] = "done"
        return 0
    except CellSkipped as e:
        status.update(status="skipped", reason=str(e))
        print(f"[{name}] skipped: {e}", flush=True)
        return 0
    except Exception as e:
        status.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        print(f"[{name}] FAILED: {status['error']}", file=sys.stderr, flush=True)
        return 1
    finally:
        status["wall_s"] = round(time.time() - t0, 1)
        pbr._writeJson(paths["status"], status)


# -- the final table ---------------------------------------------------------------------------
def _cellStatus(batch_dir: Path, name: str) -> tuple[str, str | None]:
    """``(status, detail)``: done, skipped, failed, partial (with its progress) or not run."""
    paths = _paths(batch_dir, name)
    st = pbr._readJson(paths["status"]) or {}
    if st.get("status") in ("done", "skipped"):
        return st["status"], st.get("reason")
    if paths["partial"].exists():
        try:
            with np.load(paths["partial"]) as f:
                progress = checkpointProgress(dict(f))
        except Exception:
            progress = None
        return "partial", progress if st.get("status") != "failed" else f"{st.get('error')} ({progress})"
    if st.get("status") == "failed":
        return "failed", st.get("error")
    return "not run", None


def summarize(batch_dir: Path) -> Path | None:
    """``sweep_results.csv`` / ``.json``: one row per grid point of every done cell, one per other cell."""
    cells = pbr.listCells(batch_dir)
    if not cells:
        return None
    rows, lines = [], []
    for c in cells:
        name = c["name"]
        base = {"cell": c["index"], "name": name, **c["status"].get("row", {})}
        status, detail = _cellStatus(batch_dir, name)
        rec = pbr._readJson(_paths(batch_dir, name)["json"]) if status == "done" else None
        if not rec:
            rows.append({**base, "sweep_status": status if status != "done" else "failed",
                         "sweep_detail": detail or ("no sweep.json" if status == "done" else None)})
            lines.append(f"  {name:<32} {status:<8} {detail or ''}")
            continue
        info = {"sweep_status": "done", "sweep_rollout_model": rec["rollout_model"],
                "sweep_eval_sim": rec["eval_sim"], "sweep_episodes": " ".join(str(e) for e in rec["episodes"]),
                "sweep_n_steps_analyzed": rec["n_steps_analyzed"], "sweep_n_grid": rec["n_grid"],
                "sweep_n_dropped": len(rec["dropped"]), "f_zero_spread_mean": rec["f_zero_spread_mean"]}
        for p in rec["points"]:
            rows.append({**base, **info, **{k: v for k, v in p.items() if pbr._scalar(v)}})
        nom = rec["points"][rec["nominal_index"]]
        lines.append(f"  {name:<32} done     {rec['rollout_model']}, {rec['n_grid']} grid points, nominal "
                     f"regret {pbr._fmtInterval(nom, 'regret_mean')}")
    columns = list(dict.fromkeys(k for r in rows for k in r))
    out = batch_dir / "sweep_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    (batch_dir / "sweep_results.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nsweep: {sum(_cellStatus(batch_dir, c['name'])[0] == 'done' for c in cells)}/{len(cells)} cells "
          f"done -> {out}")
    print("\n".join(lines))
    return out


# -- CLI -------------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch_dir", type=Path, help="a results folder written by run_episode_batches.py")
    p.add_argument("--cell", type=int, default=None,
                   help="sweep only this cell (0-based position in the folder's cells, as SLURM_ARRAY_TASK_ID); "
                        "omitted, every cell in order")
    p.add_argument("--count", action="store_true", help="print the number of cells and exit")
    p.add_argument("--summarize", action="store_true", help="only (re)build sweep_results.csv")
    p.add_argument("--overwrite", action="store_true", help="redo cells already done, and start over any checkpoint")
    p.add_argument("--episodes", default=None, metavar="SPEC",
                   help="the episodes to analyse: indices, inclusive ranges and episode ids, comma-separated, "
                        "e.g. 0,3,7-9; omitted, every episode")
    p.add_argument("--max-episodes", type=int, default=None, help="analyse at most this many of them")
    p.add_argument("--stride", type=int, default=1, help="analyse every k-th saved step (without --n-steps-analysed)")
    p.add_argument("--n-steps-analysed", type=int, default=20, metavar="N",
                   help="analyse N saved steps of each episode, spaced evenly (default %(default)s; 0 uses --stride)")
    p.add_argument("--values", action="append", default=None, metavar="[MODEL:]NAME=V1,V2,...",
                   help="a parameter's values, absolute, replacing its defaults; repeatable")
    p.add_argument("--multipliers", action="append", default=None, metavar="[MODEL:]NAME=M1,M2,...",
                   help="a parameter's values as multiples of its nominal, replacing its defaults; repeatable")
    p.add_argument("--score-block", type=int, default=DEFAULT_SCORE_BLOCK, metavar="B",
                   help="means scored per eval-model rollout, U_A included, in B times the planner's worlds "
                        "(default %(default)s)")
    p.add_argument("--kamino-max-contacts", type=int, default=pbr.DEFAULT_KAMINO["max_contacts_per_world"],
                   metavar="N", help="with a Kamino (M5) eval model: the zero-gap planner's contact buffer per "
                                     "world (default %(default)s; Kamino's own is 128)")
    p.add_argument("--kamino-padmm-iterations", type=int, default=pbr.DEFAULT_KAMINO["padmm_max_iterations"],
                   metavar="N", help="with a Kamino (M5) eval model: the zero-gap planner's PADMM iteration cap "
                                     "(default %(default)s; Kamino's own is 200)")
    args = p.parse_args(argv)
    if args.stride < 1:
        p.error("--stride must be >= 1")
    if args.n_steps_analysed < 0:
        p.error("--n-steps-analysed must be >= 0")
    if args.score_block < 2:
        p.error("--score-block must be >= 2")
    if args.kamino_max_contacts < 1 or args.kamino_padmm_iterations < 1:
        p.error("--kamino-max-contacts and --kamino-padmm-iterations must be >= 1")
    try:
        values = parseValueSpecs(args.values, "--values")
        multipliers = parseValueSpecs(args.multipliers, "--multipliers")
    except ValueError as e:
        p.error(str(e))
    batch_dir = args.batch_dir.resolve()
    if not batch_dir.is_dir():
        p.error(f"no such folder: {batch_dir}")
    cells = pbr.listCells(batch_dir)
    if args.count:
        print(len(cells))
        return 0
    if not cells:
        p.error(f"{batch_dir} has no cell_*.status.json files; is it an episode-batch results folder?")
    if args.summarize:
        return 0 if summarize(batch_dir) else 1
    kw = dict(stride=args.stride, n_steps_analysed=args.n_steps_analysed or None, episodes_spec=args.episodes,
              max_episodes=args.max_episodes, values=values, multipliers=multipliers,
              score_block=args.score_block,
              kamino={"max_contacts_per_world": args.kamino_max_contacts,
                      "padmm_max_iterations": args.kamino_padmm_iterations})
    if args.cell is not None:
        if not 0 <= args.cell < len(cells):
            p.error(f"--cell {args.cell} is out of range: {batch_dir.name} has cells 0-{len(cells) - 1}")
        return runCell(batch_dir, cells[args.cell], args.overwrite, **kw)
    # Every cell, each in its own process, so one failure does not stop the rest.
    passthrough = list(argv if argv is not None else sys.argv[1:])
    failed = []
    for i, c in enumerate(cells):
        print(f"[{i + 1}/{len(cells)}] {c['name']} ...", flush=True)
        if subprocess.run([sys.executable, __file__, *passthrough, "--cell", str(i)]).returncode:
            failed.append(c["name"])
    summarize(batch_dir)
    if failed:
        print(f"\n{len(failed)} cell(s) failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
