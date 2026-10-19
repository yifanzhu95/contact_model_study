#!/usr/bin/env python3
"""Analyse the results folder of an episode batch (``run_episode_batches.py``), cell by cell.

For every cell it measures how far the planner that ran was from an *optimal*
planner, one that plans with the eval model itself; what the planner's model
error cost it (its *regret*); and how far the planner's rollout model is from
the eval simulator. It works in two passes:

1. **The analysed steps, saved.** Every k-th saved step of each episode
   (``--stride``, 1 by default) is planned again by three planners and
   stepped on both simulators, and the results are saved per episode
   (``analysis/<cell>/<episode id>.npz``, below).

   The *used* planner is rebuilt from the cell's recorded command line. The
   *zero-gap* planner is the same planner (sample count, temperature, noise,
   cost weights, warm start, control step and horizon as durations) planning
   on the eval scene, at the eval timestep, with the eval simulator's contact
   model (``--plan-on-eval``, rollout model = eval sim): the used planner with
   no sim-to-sim gap. The *optimal* planner is the zero-gap planner with
   settings of its own: ``DEFAULT_OPTIMAL_ARGS`` (iterate to a convergence
   ratio of 2, noise 0.38, temperature 0.001), with ``--optimal-args`` on top,
   flag by flag (``OPTIMAL_FLAGS``: sample count, horizon, noise, temperature,
   convergence, cost weights, ...), the same for every cell.
   Each analysed step is planned on its own (``StepPlanners``): every planner
   loads what the used planner started that step's plan from (its mean, noise
   draw, temperature and previous control, saved with the steps) and plans
   from the recorded state toward that step's goal. They draw the same noise
   around the same mean (common random numbers), so the KL measures the
   models rather than sampling luck or the planners' histories, and a stride
   gives the same values at those steps as analysing them all. Older
   warm-started results, which did not save the mean, are walked in order
   instead, each planner keeping its own warm state; they cannot be strided,
   and their regret is not computed. Each step's first actions are
   moment-matched to Gaussians (``Utils/PlannerKLDiv.py``).

   The **regret** of the used plan is ``F_A(U_B) - F_A(U_A)``: the free energy
   on the ground-truth model A (the zero-gap planner's simulator, cost and
   temperature) of the used planner's final mean ``U_B``, less that of the
   zero-gap planner's ``U_A``. Both are scored on one shared noise block of
   their own (``evalNoiseSeed``), not the noise the plans drew, so ``U_A`` is
   not scored on the noise it was fitted to. Positive when the model gap cost
   the plan; noise in the estimates can make it negative at a single step.

   From each analysed state, the action the episode took and the optimal
   planner's mean action are each stepped one control step on the rollout
   model (rollout scene, rollout timestep) and on the eval simulator (eval
   scene, eval timestep), all of an episode's steps at once as the worlds of
   a vectorized simulator.
2. **The metrics, from the saved steps.** ``KL(used || opt)`` and
   ``KL(opt || used)`` (the covariances shrunk by ``--shrinkage``) and the
   distance between the means; the ``regret``; the state errors (object position (m) and
   orientation (rad), hand joints (RMS rad), full ``qpos`` / ``qvel``)
   between the models under the episode's action (``fsim_*``) and under the
   optimal action (``fsim_opt_*``), between the eval prediction and what the
   episode actually reached next (``eval_vs_recorded_*``: what restarting
   from a recorded state, without the solver's warm start, costs), and
   between the eval simulator's next states under the two actions
   (``eval_used_vs_opt_*``); and ``replay_action_err``, how far the replayed
   used plan is from the action the episode took.

The optimal planner and the eval prediction need the eval simulator to be a
vectorized contact model (``eval_sim`` M1-M4); a cell with a CPU eval
simulator is an error. The cell must have been run with ``save_steps=true``;
with ``save_stride=k`` only every k-th step was saved, and only those are
analysed.

::

    python experiments/process_batch_results.py <batch dir> --count
    python experiments/process_batch_results.py <batch dir> --cell 3
    python experiments/process_batch_results.py <batch dir>               # every cell, in order
    python experiments/process_batch_results.py <batch dir> --stride 4 \\
        --optimal-args="--n-samples 1024 --temperature 0.1"
    python experiments/process_batch_results.py <batch dir> --summarize   # final tables only

On the HPC, ``hpc/submit_process_batch_results.sh <batch dir>`` runs one array
task per cell and then the summary.

Output, in the batch folder
---------------------------
* ``analysis/<cell>/<episode id>.npz``: the episode's analysed steps, one row
  each: ``step`` (control step) and ``row`` (saved row), ``goal``, the state
  ``q`` / ``q_dot``, ``u_prev``, the action taken ``u_used``; each planner's
  first-action Gaussian ``mu_<planner>`` / ``cov_<planner>`` for ``used``,
  ``opt`` and ``zero`` (unshrunk); the regret's free energies
  ``regret_f_used`` (``F_A(U_B)``) and ``regret_f_zero`` (``F_A(U_A)``), each
  with its standard error (``_se``); the next states ``q_next_<model>_<action>`` and
  ``q_dot_next_<model>_<action>`` for model ``rollout`` / ``eval`` and action
  ``used`` / ``opt`` (NaN where a plan failed), and ``q_next_recorded`` /
  ``q_dot_next_recorded``; and ``noise_sigma_used`` / ``noise_sigma_opt``,
  the task's state ``indices`` and the timesteps;
* ``analysis/<cell>.analysis.json``: the cell's metrics (mean, median, p90 of
  each), per episode and overall, and how the planners were built;
* ``analysis/<cell>.analysis.npz``: the per-analysed-step metrics, with each
  episode's step indices as ``step__<id>``;
* ``analysis/<cell>.analysis_status.json``: done or failed, with the error;
* ``final_results.csv`` / ``.json``: one row per cell: its settings, the batch's
  own results (as in ``summary.csv``) and every analysis metric;
* ``final_episodes.csv``: one row per episode, the same way.

Every mean and rate comes with a 95% interval in the columns ``<column>_lo`` and
``<column>_hi``: a Wilson score interval for ``success_rate``, and a t-interval
for every other mean, taking each sample as independent (the analysed steps for
the analysis metrics, over the cell or the one episode; the episodes for the
batch's own means). Consecutive steps are correlated, so the step-level
intervals are narrower than the episodes alone would support. ``--summarize``
recomputes them from the saved ``analysis.npz``, so older analyses gain them
without a rerun.

A cell whose analysis is done is skipped when run again (``--overwrite`` redoes
it, which is needed to rerun with another ``--stride`` or ``--optimal-args``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
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

import run_episode_batches as batches  # noqa: E402

ANALYSIS_DIR = "analysis"


# -- the batch's cells ---------------------------------------------------------------
def listCells(batch_dir: Path) -> list[dict]:
    """The batch's cells, from their status files, in cell order."""
    cells = []
    for p in batch_dir.glob("cell_*.status.json"):
        try:
            status = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        name = p.name[:-len(".status.json")]
        cells.append({"name": name, "index": status.get("cell"), "status": status})
    return sorted(cells, key=lambda c: (c["index"] if c["index"] is not None else 1 << 30, c["name"]))


def _analysisPaths(batch_dir: Path, name: str) -> dict[str, Path]:
    d = batch_dir / ANALYSIS_DIR
    return {"json": d / f"{name}.analysis.json", "npz": d / f"{name}.analysis.npz",
            "status": d / f"{name}.analysis_status.json", "steps": d / name}


def analysisDone(batch_dir: Path, name: str) -> bool:
    try:
        return json.loads(_analysisPaths(batch_dir, name)["status"].read_text()).get("status") == "done"
    except (OSError, ValueError):
        return False


# -- rebuilding what ran ---------------------------------------------------------------
def argsFromCliArgs(cli_args: dict) -> argparse.Namespace:
    """The driver's ``args`` as recorded, with defaults for options added since."""
    from ContactModelStudy.Drivers import run_episodes as drv
    defaults = vars(drv.parseArgs(["--no-results", "--no-video"]))
    args = {**defaults, **cli_args, "results": None, "video": None}
    # Recorded when warm_start was on/off: off kept the last mean unshifted.
    if isinstance(args["warm_start"], bool):
        args["warm_start"] = "shift" if args["warm_start"] else "hold"
    # Recorded before these flags: one uncentered noise draw per plan.
    for name in ("resample_noise", "center_noise"):
        if name not in cli_args:
            args[name] = False
    return argparse.Namespace(**args)


#: The planner settings ``--optimal-args`` may change. The rest define what is
#: compared (the model, the control step and mode, the warm start) and stay the cell's.
OPTIMAL_FLAGS = ("n_samples", "horizon", "time_horizon", "n_iterations", "noise_sigma", "temperature",
                 "adaptive_temp", "delta", "cost_weight", "graph", "nconmax", "njmax", "debug",
                 "convergence_ratio", "max_iterations", "resample_noise", "center_noise")

#: The optimal planner's own settings unless ``--optimal-args`` gives others, flag by flag:
#: iterate to convergence, with wide noise and a near-greedy temperature.
DEFAULT_OPTIMAL_ARGS = "--convergence-ratio 2 --noise-sigma 0.38 --temperature 0.001"


def optimalOverrides(optimal_args: str | None) -> tuple[dict, str]:
    """``DEFAULT_OPTIMAL_ARGS`` with ``optimal_args`` on top: the overrides, and the flags as one string."""
    merged = {**parseOptimalArgs(DEFAULT_OPTIMAL_ARGS), **parseOptimalArgs(optimal_args)}
    return merged, " ".join(s for s in (DEFAULT_OPTIMAL_ARGS, optimal_args) if s)


def parseOptimalArgs(text: str | None) -> dict:
    """The settings ``--optimal-args`` gives the optimal planner, as ``args`` fields.

    ``text`` holds ``run_episodes.py`` flags, e.g. ``"--n-samples 1024
    --temperature 0.1"``. Only the flags given are returned, so every other
    setting stays the cell's; ``--cost-weight`` comes back as ``cost_weights``.
    Raises ``ValueError`` on a bad flag or one not in ``OPTIMAL_FLAGS``.
    """
    from ContactModelStudy.Drivers import run_episodes as drv
    tokens = shlex.split(text or "")
    if not tokens:
        return {}

    def fail(msg):
        raise ValueError(f"--optimal-args: {msg}")
    # Which flags were given: the driver's parser with no defaults.
    p = drv.build_parser()
    for action in p._actions:
        action.default = argparse.SUPPRESS
    p.error = fail
    given, unknown = p.parse_known_args(tokens)
    if unknown:
        fail(f"unrecognised {' '.join(unknown)}")
    given = vars(given)
    bad = sorted(set(given) - set(OPTIMAL_FLAGS))
    if bad:
        fail(f"{', '.join('--' + b.replace('_', '-') for b in bad)} cannot be set for the optimal planner; "
             f"it takes {', '.join('--' + f.replace('_', '-') for f in OPTIMAL_FLAGS)}")
    # Their values, checked as the driver checks them.
    full_parser = drv.build_parser()
    full_parser.error = fail
    full = drv.parseArgs(tokens + ["--no-results", "--no-video"], parser=full_parser)
    out = {k: getattr(full, k) for k in given if k != "cost_weight"}
    if "cost_weight" in given:
        out["cost_weights"] = full.cost_weights
    return out


def optimalArgs(args: argparse.Namespace, control_timestep: float, horizon_duration: float,
                overrides: dict | None = None) -> argparse.Namespace:
    """The used planner's settings, planning on the eval scene with the eval simulator's model.

    Control step and horizon are passed as durations, so the optimal planner
    covers the same span of time at the eval timestep. ``overrides``
    (``parseOptimalArgs``) then replace the settings they give: a horizon in
    either form replaces the used one, and cost weights add to the cell's.
    """
    o = {**vars(args), "plan_on_eval": True, "rollout_model": args.eval_sim, "substeps": None,
         "ctrl_time_step": control_timestep, "horizon": None, "time_horizon": horizon_duration}
    overrides = dict(overrides or {})
    if "horizon" in overrides or "time_horizon" in overrides:
        o["horizon"] = o["time_horizon"] = None
    if "cost_weights" in overrides:
        overrides["cost_weights"] = {**(args.cost_weights or {}), **overrides["cost_weights"]}
    return argparse.Namespace(**{**o, **overrides})


def checkCell(rep, args) -> None:
    """Raise ``ValueError`` if this cell cannot be analysed."""
    from ContactModelStudy.Utils.EvalSimulators import EVAL_PRESETS
    if args.eval_sim not in EVAL_PRESETS:
        raise ValueError(f"the eval simulator is {args.eval_sim!r}, which is not vectorized: the optimal "
                         f"planner and the forward-simulation comparison need a GPU eval model "
                         f"({', '.join(EVAL_PRESETS)}). Rerun the batch with eval_sim set to one of them.")
    if len(rep) == 0:
        raise ValueError("the results hold no episodes")
    if not all(ep.has_steps for ep in rep):
        raise ValueError("the per-step data was not saved; rerun the batch with save_steps=true")


# -- goals ---------------------------------------------------------------------------------
def episodeGoals(rep, args) -> list[list[tuple[int, np.ndarray]]]:
    """Each episode's goals as ``(first step, goal)``, in order.

    Read from the summary when it has them (``goal_values``/``goal_steps``).
    Older results are reconstructed by replaying the eval task's goal stream as
    ``run_episodes.py`` drew it (one stream from ``--seed``, or one per episode
    from its ``goal_seed`` for the parallel drivers), with goal i >= 1 adopted
    at the step the previous one was reached. The replay is checked against the
    saved goal labels.
    """
    episodes = list(rep)
    if all("goal_values" in ep.summary for ep in episodes):
        return [[(int(s), np.asarray(g, dtype=float))
                 for s, g in zip(ep.summary["goal_steps"], ep.summary["goal_values"])] for ep in episodes]
    from ContactModelStudy.Drivers import run_episodes as drv
    _, eval_task = drv.buildTasks(args)
    out = []
    for ep in episodes:
        if ep.summary.get("goal_seed") is not None:
            eval_task.reseed(int(ep.summary["goal_seed"]))
        eval_task._onInitialState()
        labels = ep.summary.get("goals", [])
        starts = [0] + list(ep.summary.get("success_steps", []))
        goals = []
        for i, label in enumerate(labels):
            g = eval_task.sampleNewGoal()
            eval_task.setGoal(g)
            if label is not None and eval_task.goalFace(g) != label:
                raise ValueError(f"episode {ep.id}: goal {i} replays as {eval_task.goalFace(g)!r} but "
                                 f"was recorded as {label!r}; cannot reconstruct the goals")
            goals.append((int(starts[i]) if i < len(starts) else ep.n_steps, np.asarray(g, dtype=float)))
        out.append(goals)
    return out


def goalIndexAt(goals: list[tuple[int, np.ndarray]], steps) -> np.ndarray:
    """Which goal was current at each of ``steps``."""
    starts = np.array([s for s, _ in goals])
    return np.searchsorted(starts, np.asarray(steps), side="right") - 1


def goalIndexPerStep(goals: list[tuple[int, np.ndarray]], n_steps: int) -> np.ndarray:
    """Which goal was current at each step."""
    return goalIndexAt(goals, np.arange(n_steps))


# -- metrics ---------------------------------------------------------------------------------
def quatAngle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Rotation angle (rad) between unit quaternions, row by row; sign-insensitive."""
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    d = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(d)


def stateErrors(qa, va, qb, vb, indices) -> dict[str, np.ndarray]:
    """Per-row errors between two batches of states ``(T, nq)`` / ``(T, nv)``."""
    oq, n_hand = int(indices[0]), int(indices[3])
    return {
        "obj_pos_err": np.linalg.norm(qa[:, oq:oq + 3] - qb[:, oq:oq + 3], axis=1),
        "obj_rot_err": quatAngle(qa[:, oq + 3:oq + 7], qb[:, oq + 3:oq + 7]),
        "hand_rms_err": np.sqrt(np.mean((qa[:, :n_hand] - qb[:, :n_hand]) ** 2, axis=1)),
        "qpos_err": np.linalg.norm(qa - qb, axis=1),
        "qvel_err": np.linalg.norm(va - vb, axis=1),
    }


def stats(prefix: str, values: np.ndarray) -> dict:
    """``<prefix>_mean`` with its 95% interval (``_mean_lo`` / ``_mean_hi``), ``_median`` and
    ``_p90`` over the finite values (and how many were not).

    The interval is a t-interval that takes every value as an independent
    sample (``run_episode_batches.meanInterval``).
    """
    v = np.asarray(values, dtype=float).ravel()
    ok = v[np.isfinite(v)]
    out = batches.withInterval(f"{prefix}_mean", ok)
    out.update({f"{prefix}_median": float(np.median(ok)) if ok.size else None,
                f"{prefix}_p90": float(np.percentile(ok, 90)) if ok.size else None})
    if ok.size != v.size:
        out[f"{prefix}_n_nonfinite"] = int(v.size - ok.size)
    return out


def episodeStats(m: dict[str, np.ndarray]) -> dict:
    """An episode's per-step metrics boiled down: each one's mean, with its 95% interval."""
    out = {}
    for key, values in m.items():
        out.update({k: v for k, v in stats(key, values).items() if k.startswith(f"{key}_mean")})
    return out


def cellStats(episodes: list[dict[str, np.ndarray]]) -> dict:
    """Every metric over all the episodes' analysed steps pooled: ``stats`` of each."""
    out = {}
    for key in dict.fromkeys(k for m in episodes for k in m):
        out.update(stats(key, np.concatenate([np.ravel(m[key]) for m in episodes if key in m])))
    return out


def metricsFromNpz(path: Path) -> tuple[dict, dict[str, dict]]:
    """A cell's metrics, recomputed from its ``analysis.npz``: ``(cell metrics, {episode id: metrics})``."""
    per_episode: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path) as f:
        for name in f.files:
            key, ep_id = name.rsplit("__", 1)
            if key != "step":
                per_episode.setdefault(ep_id, {})[key] = f[name]
    return cellStats(list(per_episode.values())), {i: episodeStats(m) for i, m in per_episode.items()}


# -- planning the analysed steps ---------------------------------------------------------------
KL_KEYS = ("kl_used_opt", "kl_opt_used", "mean_dist")
PLANNERS = ("used", "opt", "zero")
PLAN_KEYS = tuple(f"{k}_{p}" for p in PLANNERS for k in ("mu", "cov"))
#: The two free energies on the ground-truth model the regret is the difference of, with their
#: standard errors: ``F_A(U_B)`` (the deployed plan) and ``F_A(U_A)`` (the zero-gap plan).
REGRET_KEYS = ("regret_f_used", "regret_f_zero", "regret_f_used_se", "regret_f_zero_se")


def _nanPlans(T: int, nu: int) -> dict[str, np.ndarray]:
    out = {}
    for p in PLANNERS:
        out[f"mu_{p}"], out[f"cov_{p}"] = np.full((T, nu), np.nan), np.full((T, nu, nu), np.nan)
    out.update({k: np.full(T, np.nan) for k in REGRET_KEYS})
    return out


def evalNoiseSeed(start: dict) -> int:
    """The seed of the noise a step's free energies are evaluated on.

    Fixed by the step's recorded noise index, so a rerun or a stride gives the
    same values, but a different stream from the one the plans drew: the
    zero-gap plan is not scored on the noise it was fitted to.
    """
    ss = np.random.SeedSequence([int(start["noise_seed"]), int(start["resample_count"]), 0xF1E])
    return int(ss.generate_state(1)[0] % (2**31 - 1))


def planGaussian(planner, q: np.ndarray, q_dot: np.ndarray, u) -> tuple[np.ndarray, np.ndarray]:
    """A planner's first-action ``(mu, cov)`` planned from ``(q, q_dot)``, unshrunk; NaN if the plan failed."""
    from ContactModelStudy.Utils.PlannerKLDiv import PlannerGaussian
    try:
        return PlannerGaussian(planner, q, q_dot, u=u, shrinkage=0.0, restore=False)
    except RuntimeError:                                  # a failed plan
        return np.full(planner.nu, np.nan), np.full((planner.nu, planner.nu), np.nan)


class StepPlanners:
    """The used, optimal and zero-gap planners, each analysed step planned on its own.

    For a step, every planner loads the start the used planner recorded for it
    (``planStart``): the mean it started from, its noise draw, its temperature
    and the control it accumulated from. Each then plans from the recorded
    state toward that step's goal. They draw the same noise (common random
    numbers) around the same mean, so the KL measures the models rather than
    sampling luck or the planners' histories, and any subset of the steps
    gives the same per-step values. An optimal planner given its own
    temperature (``opt_temperature``, from its args) starts every step from it
    rather than the recorded one; given another sample count or noise, its
    draws are no longer the used planner's.

    The zero-gap planner has the used planner's settings on the ground-truth
    model, so it draws exactly the used planner's noise. After both have
    planned, each one's final mean is scored on the zero-gap planner's model
    (``FreeEnergyOf``), on one shared noise block of its own
    (``evalNoiseSeed``): ``F_A(U_B)`` and ``F_A(U_A)``, whose difference is the
    used planner's regret.

    ``used``, ``opt`` and ``zero`` are ``(task, planner)`` pairs.
    """

    def __init__(self, used, opt, zero, reset_mean: np.ndarray | None, opt_temperature: float | None = None):
        pu, po, pz = used[1], opt[1], zero[1]
        self._same_shape = (pu.horizon, pu.nu) == (po.horizon, po.nu)
        if (pz.horizon, pz.nu) != (pu.horizon, pu.nu):
            raise ValueError(f"the used planner's mean is {pu.horizon} x {pu.nu}, the zero-gap one's "
                             f"{pz.horizon} x {pz.nu}: the regret scores the used plan on the zero-gap "
                             f"planner, so both need the same control step and horizon")
        self.used, self.opt, self.zero, self.opt_temperature = used, opt, zero, opt_temperature
        self._goal = None
        for _, planner in (used, opt, zero):
            planner.Reset(reset_mean)       # the seed a warm_start="none" plan starts from

    def forgetGoal(self) -> None:
        """Set the goal again on the next step: the tasks' goals were changed elsewhere."""
        self._goal = None

    def _setGoal(self, goal: np.ndarray) -> None:
        if self._goal is None or not np.array_equal(self._goal, goal):
            for task, _ in (self.used, self.opt, self.zero):
                task.setGoal(goal)
            self._goal = np.array(goal, copy=True)

    def distributionsAt(self, goal: np.ndarray, start: dict, q: np.ndarray,
                        q_dot: np.ndarray) -> dict[str, np.ndarray]:
        """Every planner's first-action Gaussian from ``start`` (``PLAN_KEYS``), unshrunk, and the
        free energies the regret is computed from (``REGRET_KEYS``); NaN for a failed plan."""
        if "U" in start and not self._same_shape:
            pu, po = self.used[1], self.opt[1]
            raise ValueError(f"the used planner's mean is {pu.horizon} x {pu.nu}, the optimal one's "
                             f"{po.horizon} x {po.nu}: a warm-started step starts both from the used "
                             f"planner's mean, so the optimal planner needs the same horizon")
        self._setGoal(goal)
        opt_start = start if self.opt_temperature is None else {**start, "lam": self.opt_temperature}
        out = {}
        means = {}
        for name, (_, planner), s in (("used", self.used, start), ("opt", self.opt, opt_start),
                                      ("zero", self.zero, start)):
            planner.LoadState({**planner.SaveState(), **s})
            out[f"mu_{name}"], out[f"cov_{name}"] = planGaussian(planner, q, q_dot, start.get("u_prev"))
            means[name] = planner.last_mean
        out.update(self._regretTerms(start, q, q_dot, means["used"], means["zero"]))
        return out

    def _regretTerms(self, start: dict, q, q_dot, U_used, U_zero) -> dict[str, float]:
        """``F_A(U_B)``, ``F_A(U_A)`` and their standard errors, on one shared evaluation noise block."""
        out = {k: float("nan") for k in REGRET_KEYS}
        if U_used is None or U_zero is None:                # a failed plan
            return out
        planner = self.zero[1]
        eval_start = {**start, "noise_seed": evalNoiseSeed(start), "resample_count": 0}
        for name, U in (("used", U_used), ("zero", U_zero)):
            planner.LoadState({**planner.SaveState(), **eval_start})
            out[f"regret_f_{name}"], out[f"regret_f_{name}_se"] = planner.FreeEnergyOf(
                q, q_dot, U, u=start.get("u_prev"))
        return out

    def planAlong(self, ep, goals, rows: np.ndarray, starts: list[dict]) -> dict[str, np.ndarray]:
        """``distributionsAt`` the saved rows ``rows`` of an episode, one row each."""
        out = _nanPlans(len(rows), self.used[1].nu)
        gidx = goalIndexAt(goals, ep.step[rows])
        for j, (i, g, start) in enumerate(zip(rows, gidx, starts)):
            for k, v in self.distributionsAt(goals[g][1], start, ep.q[i], ep.q_dot[i]).items():
                out[k][j] = v
        return out


def planStart(ep, i: int, args, u0: np.ndarray, noise_seed: int) -> dict | None:
    """What the used planner started saved row ``i`` from, as ``LoadState`` keys.

    Read off the record when it has one (results saved since plan starts were
    recorded). Older results are rebuilt where that is possible: under
    ``warm_start="none"`` a plan starts from the reset mean, with the episode's
    noise stream at the step's index. ``None`` when it is not: a warm-started
    plan whose mean was not saved.
    """
    step, u_prev = int(ep.step[i]), ep.u_prev[i]
    u_prev = u_prev if np.isfinite(u_prev).all() else (u0 if step == 0 else None)
    if ep.has_plan_starts:
        start = {"noise_seed": int(ep.summary.get("noise_seed") or noise_seed),
                 "resample_count": int(ep.noise_index[i]), "u_prev": u_prev}
        if np.isfinite(ep.temperature[i]):
            start["lam"] = float(ep.temperature[i])
        if args.warm_start != "none":
            if ep.mean_start is None:
                return None
            start["U"] = np.asarray(ep.mean_start[i], dtype=np.float32)
        return start
    if args.warm_start != "none":
        return None
    return {"noise_seed": noise_seed, "resample_count": step, "lam": float(args.temperature), "u_prev": u_prev}


def replayMode(ep, starts: list, stride: int, warm_start: str) -> str:
    """How an episode's plans are replayed: ``"independent"`` or ``"sequential"``.

    Independent when every analysed step's start is known (``starts`` has no
    ``None``). Otherwise the plans can only be walked in order from the first
    step, which needs every step saved and analysed; raises ``ValueError`` when
    they are not.
    """
    if all(s is not None for s in starts):
        return "independent"
    if stride > 1:
        raise ValueError(f"episode {ep.id} was warm-started ({warm_start}) but did not save the mean each "
                         f"plan started from, so its plans can only be replayed in order through every "
                         f"step: rerun without --stride")
    if ep.n_saved != ep.n_steps:
        raise ValueError(f"episode {ep.id} was warm-started ({warm_start}) but saved neither the mean each "
                         f"plan started from nor every step, so its plans cannot be replayed")
    return "sequential"


def _resetPlanner(planner, mean, noise_seed: int | None = None) -> None:
    planner.Reset(mean)
    if noise_seed is not None:
        planner.LoadState({**planner.SaveState(), "u_prev": None, "last_action_seq": None,
                           "noise_seed": int(noise_seed), "resample_count": 0})


def planSequentially(ep, goals, used, opt, zero, args, u0, noise_seed: int) -> dict[str, np.ndarray]:
    """Every planner walked along one recorded episode in order; each step's Gaussians (``PLAN_KEYS``).

    The fallback for older warm-started results, which did not save the mean
    each plan started from: each planner keeps its own warm state from step to
    step, as the episode ran, so every step is replayed (no ``--stride``).
    The regret is left NaN: without a recorded start per step there is no
    noise index to fix its evaluation noise by. ``used``, ``opt`` and
    ``zero`` are ``(task, planner)`` pairs.
    """
    T = ep.n_steps
    reset_mean = u0 if args.control_mode == "absolute" else None
    gidx = goalIndexPerStep(goals, T)
    out = _nanPlans(T, used[1].nu)
    planners = (("used", used), ("opt", opt), ("zero", zero))
    current = None
    for t in range(T):
        if gidx[t] != current:                           # a new goal: as run_episode does
            current = gidx[t]
            for _, (task, planner) in planners:
                task.setGoal(goals[current][1])
                _resetPlanner(planner, reset_mean, noise_seed if t == 0 else None)
        u_prev = u0 if t == 0 else ep.u[t - 1]
        for name, (_, planner) in planners:
            out[f"mu_{name}"][t], out[f"cov_{name}"][t] = planGaussian(planner, ep.q[t], ep.q_dot[t], u_prev)
    return out


# -- stepping the simulators -------------------------------------------------------------------
def stepFrom(sim, steps: int, q: np.ndarray, v: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One control step from each row's state under its action, all rows as worlds of ``sim`` at once."""
    T, N = len(q), sim.N
    pad = lambda a: np.concatenate([a, np.repeat(a[-1:], N - T, axis=0)]) if N > T else a  # noqa: E731
    sim.ClearControlSequence()
    sim.SetState(pad(q), pad(v))
    sim.SetControl(pad(u))
    sim.Step(steps)
    qn, vn = sim.GetState()
    return np.array(qn, dtype=float)[:T], np.array(vn, dtype=float)[:T]


def nextStates(q, v, actions: dict[str, np.ndarray], sims: dict[str, tuple]) -> dict[str, np.ndarray]:
    """``q_next_<model>_<action>`` / ``q_dot_next_<model>_<action>``: every action on every model.

    ``sims`` maps a model name to ``(simulator, physics steps per control
    step)``. A row whose action is NaN (a failed plan) is stepped with the
    first action's row instead, and its result set to NaN.
    """
    out = {}
    fallback = next(iter(actions.values()))
    for a_name, u in actions.items():
        ok = np.isfinite(u).all(axis=1)
        u = np.where(ok[:, None], u, fallback).astype(fallback.dtype)
        for m_name, (sim, steps) in sims.items():
            qn, vn = stepFrom(sim, steps, q, v, u) if len(q) else (np.empty((0, q.shape[1])),
                                                                   np.empty((0, v.shape[1])))
            qn[~ok], vn[~ok] = np.nan, np.nan
            out[f"q_next_{m_name}_{a_name}"], out[f"q_dot_next_{m_name}_{a_name}"] = qn, vn
    return out


# -- the metrics, from the saved steps ---------------------------------------------------------
#: ``(prefix, a, b)``: the state errors between next states ``a`` and ``b``.
STATE_COMPARISONS = (
    ("fsim", "rollout_used", "eval_used"),          # the models, under the action the episode took
    ("fsim_opt", "rollout_opt", "eval_opt"),        # the models, under the optimal planner's action
    ("eval_vs_recorded", "eval_used", "recorded"),  # restarting from a recorded state vs what happened
    ("eval_used_vs_opt", "eval_used", "eval_opt"),  # what the optimal action would have changed
)


def shrunk(cov: np.ndarray, sigma: float, shrinkage: float) -> np.ndarray:
    """``(1 - shrinkage) * cov + shrinkage * sigma^2 I``, row by row; see ``Utils/PlannerKLDiv.py``."""
    return (1.0 - shrinkage) * cov + shrinkage * sigma ** 2 * np.eye(cov.shape[-1])


def episodeMetrics(d, shrinkage: float) -> dict[str, np.ndarray]:
    """Per-analysed-step metrics from an episode's saved steps (an ``analysis/<cell>/<id>.npz``)."""
    from ContactModelStudy.Utils.PlannerKLDiv import GaussianKL
    mu_u, mu_o = d["mu_used"], d["mu_opt"]
    cov_u = shrunk(d["cov_used"], float(d["noise_sigma_used"]), shrinkage)
    cov_o = shrunk(d["cov_opt"], float(d["noise_sigma_opt"]), shrinkage)
    out = {k: np.full(len(mu_u), np.nan) for k in KL_KEYS}
    for i in range(len(mu_u)):
        if np.isfinite(mu_u[i]).all() and np.isfinite(mu_o[i]).all():
            out["kl_used_opt"][i] = GaussianKL(mu_u[i], cov_u[i], mu_o[i], cov_o[i])
            out["kl_opt_used"][i] = GaussianKL(mu_o[i], cov_o[i], mu_u[i], cov_u[i])
            out["mean_dist"][i] = float(np.linalg.norm(mu_u[i] - mu_o[i]))
    # The replayed plan against the action the episode took: zero up to GPU noise when replayed exactly.
    out["replay_action_err"] = np.linalg.norm(mu_u - d["u_used"], axis=1)
    # Regret = F_A(U_B) - F_A(U_A); NaN for steps saved before it was computed.
    nan = np.full(len(mu_u), np.nan)
    out["regret"] = np.asarray(d.get("regret_f_used", nan), dtype=float) - \
        np.asarray(d.get("regret_f_zero", nan), dtype=float)
    with np.errstate(invalid="ignore"):
        for prefix, a, b in STATE_COMPARISONS:
            errs = stateErrors(d[f"q_next_{a}"], d[f"q_dot_next_{a}"], d[f"q_next_{b}"], d[f"q_dot_next_{b}"],
                               d["indices"])
            out.update({f"{prefix}_{k}": e for k, e in errs.items()})
    return out


# -- one cell --------------------------------------------------------------------------------
def _fmt(v) -> str:
    return f"{v:.4g}" if v is not None else "nan"


def _fmtInterval(record: dict, key: str) -> str:
    """``value [lo, hi]``, or the value alone when it has no interval."""
    lo, hi = record.get(f"{key}_lo"), record.get(f"{key}_hi")
    return _fmt(record.get(key)) + (f" [{_fmt(lo)}, {_fmt(hi)}]" if lo is not None and hi is not None else "")


def _plannerRecord(pargs: argparse.Namespace, task, sim) -> dict:
    """How an analysis planner was built, for the cell's record."""
    return {"rollout_model": pargs.rollout_model, "scene": Path(task.getModelPath()).name,
            "timestep": task.timestep, "substeps": sim.config.resolved_substeps,
            "horizon": sim.config.resolved_horizon, "n_samples": pargs.n_samples,
            "n_iterations": pargs.n_iterations, "convergence_ratio": pargs.convergence_ratio,
            "max_iterations": pargs.max_iterations, "noise_sigma": pargs.noise_sigma,
            "temperature": pargs.temperature, "adaptive_temp": pargs.adaptive_temp,
            "resample_noise": pargs.resample_noise, "center_noise": pargs.center_noise,
            "cost_weights": pargs.cost_weights}


def processCell(batch_dir: Path, cell: dict, stride: int = 1, shrinkage: float = 1e-3,
                max_episodes: int | None = None, optimal_args: str | None = None) -> dict:
    """One cell: every episode's analysed steps saved, then the metrics computed from them.

    Writes and returns its analysis record.
    """
    from ContactModelStudy.Drivers import run_episodes as drv
    from ContactModelStudy.Drivers.EpisodePool import episodeSeeds
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI
    from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
    from ContactModelStudy.Utils.EpisodeRecorder import EpisodeReplayer

    name = cell["name"]
    res = batch_dir / f"{name}.json"
    if not res.exists():
        raise ValueError(f"no results file {res.name} (the batch cell status is "
                         f"{cell['status'].get('status')!r})")
    rep = EpisodeReplayer(res)
    cli_args = rep.configs.get("metadata", {}).get("cli_args")
    if not cli_args:
        raise ValueError(f"{res.name} does not record the command line it was run with")
    args = argsFromCliArgs(cli_args)
    checkCell(rep, args)
    overrides, effective_args = optimalOverrides(optimal_args)
    episodes = list(rep)[:max_episodes] if max_episodes else list(rep)
    all_goals = episodeGoals(rep, args)

    # The used planner, as run_episodes.main built it.
    task, eval_task = drv.buildTasks(args)
    used_sim = GetContactModelSim(args.rollout_model, task.getModelPath(), N=args.n_samples,
                                  **drv.rolloutSimOverrides(args, task))
    used = (task, MPPI(used_sim, task, drv.buildPlannerConfig(args)))
    # The optimal planner: the same, with the eval model, and any settings of its own.
    oargs = optimalArgs(args, used_sim.config.control_timestep, used_sim.config.horizon_duration, overrides)
    otask, _ = drv.buildTasks(oargs)
    opt_sim = GetContactModelSim(oargs.rollout_model, otask.getModelPath(), N=oargs.n_samples,
                                 **drv.rolloutSimOverrides(oargs, otask))
    opt = (otask, MPPI(opt_sim, otask, drv.buildPlannerConfig(oargs)))
    # The zero-gap planner: the used planner's own settings, on the eval model.
    zargs = optimalArgs(args, used_sim.config.control_timestep, used_sim.config.horizon_duration)
    ztask, _ = drv.buildTasks(zargs)
    zero_sim = GetContactModelSim(zargs.rollout_model, ztask.getModelPath(), N=zargs.n_samples,
                                  **drv.rolloutSimOverrides(zargs, ztask))
    zero = (ztask, MPPI(zero_sim, ztask, drv.buildPlannerConfig(zargs)))
    _, _, u0 = eval_task.getInitialState()
    reset_mean = u0 if args.control_mode == "absolute" else None
    planners = StepPlanners(used, opt, zero, reset_mean,
                            opt_temperature=oargs.temperature if "temperature" in overrides else None)

    # The forward-simulation pair: one world per analysed step.
    n_worlds = max(1, max(len(range(0, ep.n_saved, stride)) for ep in episodes))
    rollout_steps = used_sim.config.resolved_substeps
    eval_steps = int(round(rollout_steps * task.timestep / eval_task.timestep))
    overrides_sim = drv.rolloutSimOverrides(args, task)
    fs_rollout = GetContactModelSim(args.rollout_model, task.getModelPath(), N=n_worlds, **overrides_sim)
    fs_eval = GetContactModelSim(args.eval_sim, eval_task.getModelPath(), N=n_worlds,
                                 timestep=eval_task.timestep,
                                 **{k: v for k, v in overrides_sim.items() if k in ("nconmax", "njmax")})
    sims = {"rollout": (fs_rollout, rollout_steps), "eval": (fs_eval, eval_steps)}

    paths = _analysisPaths(batch_dir, name)
    steps_dir = paths["steps"]
    steps_dir.mkdir(parents=True, exist_ok=True)
    for old in steps_dir.glob("*.npz"):                # a rerun's episodes replace the last run's
        old.unlink()

    print(f"[{name}] {len(episodes)} episodes; used planner {args.rollout_model} on "
          f"{Path(task.getModelPath()).name} ({rollout_steps} x {task.timestep * 1e3:g} ms), optimal "
          f"{args.eval_sim} on {Path(otask.getModelPath()).name} ({opt_sim.config.resolved_substeps} x "
          f"{otask.timestep * 1e3:g} ms, {effective_args}), zero-gap {zargs.rollout_model}, steps saved every "
          f"{rep.step_saving.get('stride', 1)}, analysed every {stride} saved step(s)", flush=True)
    files, replays = [], set()
    t0 = time.time()
    # 1. Each episode's analysed steps: the state, both planners' action distributions and
    #    the next state under each planner's action on each model, saved per episode.
    for k, (ep, goals) in enumerate(zip(episodes, all_goals)):
        ep_index = int(ep.summary.get("episode", k))
        noise_seed = episodeSeeds(args.seed, ep_index)[1]
        rows = np.arange(0, ep.n_saved, stride)
        starts = [planStart(ep, i, args, u0, noise_seed) for i in rows]
        mode = replayMode(ep, starts, stride, args.warm_start)
        replays.add(mode)
        if mode == "independent":
            plans = planners.planAlong(ep, goals, rows, starts)
        else:
            plans = planSequentially(ep, goals, used, opt, zero, args, u0, noise_seed)
            planners.forgetGoal()
        q, v, u = ep.q[rows], ep.q_dot[rows], ep.u[rows]
        data = {
            "id": np.array(ep.id), "episode": ep_index, "n_steps": ep.n_steps,
            "row": rows, "step": ep.step[rows],
            "goal": np.array([goals[g][1] for g in goalIndexAt(goals, ep.step[rows])]),
            "q": q, "q_dot": v, "u_prev": ep.u_prev[rows], "u_used": u, **plans,
            "q_next_recorded": ep.q_next[rows], "q_dot_next_recorded": ep.q_dot_next[rows],
            **nextStates(q, v, {"used": u, "opt": plans["mu_opt"]}, sims),
            "noise_sigma_used": float(args.noise_sigma), "noise_sigma_opt": float(oargs.noise_sigma),
            "indices": np.asarray(eval_task.indices),
            "control_timestep": used_sim.config.control_timestep,
            "rollout_timestep": task.timestep, "eval_timestep": eval_task.timestep,
        }
        path = steps_dir / f"{ep.id}.npz"
        np.savez(path, **data)
        files.append(path)
        print(f"  episode {ep_index}: {ep.n_steps} steps, {len(rows)} analysed ({mode}) -> "
              f"{path.relative_to(batch_dir)}  ({time.time() - t0:.0f} s)", flush=True)
    for sim in (used_sim, opt_sim, zero_sim, fs_rollout, fs_eval):
        sim.Close()

    # 2. The metrics, from the saved steps.
    ep_arrays: list[tuple[str, np.ndarray, dict]] = []  # every episode's per-analysed-step metrics
    ep_records = []
    for path in files:
        with np.load(path) as f:
            d = dict(f)
        m = episodeMetrics(d, shrinkage)
        rec = {"id": str(d["id"]), "episode": int(d["episode"]), "n_steps": int(d["n_steps"]),
               "n_steps_analyzed": len(d["step"]), **episodeStats(m)}
        ep_records.append(rec)
        ep_arrays.append((rec["id"], d["step"], m))
        print(f"  episode {rec['episode']}: KL(used||opt) {rec.get('kl_used_opt_mean') or float('nan'):.3f}, "
              f"regret {_fmt(rec.get('regret_mean'))}, "
              f"object pos err {1e3 * (rec.get('fsim_obj_pos_err_mean') or float('nan')):.2f} mm", flush=True)

    metrics = cellStats([m for _, _, m in ep_arrays])
    record = {
        "cell": cell["index"], "name": name, "results": res.name,
        "rollout_model": args.rollout_model, "eval_sim": args.eval_sim,
        "plan_on_eval": bool(getattr(args, "plan_on_eval", False)),
        "stride": stride, "save_stride": int(rep.step_saving.get("stride", 1)),
        "replay": "+".join(sorted(replays)) or None, "warm_start": args.warm_start,
        "shrinkage": shrinkage, "n_episodes_analyzed": len(episodes),
        "n_steps_analyzed": sum(r["n_steps_analyzed"] for r in ep_records),
        "optimal_args": effective_args,
        "optimal_planner": _plannerRecord(oargs, otask, opt_sim),
        "zero_gap_planner": _plannerRecord(zargs, ztask, zero_sim),
        "forward_sim": {"rollout_steps": rollout_steps, "eval_steps": eval_steps,
                        "control_timestep": used_sim.config.control_timestep},
        "steps_dir": str(steps_dir.relative_to(batch_dir)),
        "metrics": metrics, "episodes": ep_records, "wall_s": round(time.time() - t0, 1),
    }
    paths["json"].write_text(json.dumps(record, indent=2))
    np.savez(paths["npz"], **{f"{key}__{ep_id}": arr for ep_id, _, m in ep_arrays for key, arr in m.items()},
             **{f"step__{ep_id}": step for ep_id, step, _ in ep_arrays})
    return record


def runCell(batch_dir: Path, cell: dict, overwrite: bool, **kw) -> int:
    """Analyse one cell, recording its status; returns 0 when done."""
    name = cell["name"]
    if not overwrite and analysisDone(batch_dir, name):
        print(f"[{name}] analysis already done; skipping (--overwrite to redo)")
        return 0
    paths = _analysisPaths(batch_dir, name)
    paths["status"].parent.mkdir(parents=True, exist_ok=True)
    status = {"cell": cell["index"], "name": name, "host": os.uname().nodename,
              "slurm_job": os.environ.get("SLURM_JOB_ID"), **kw}
    t0 = time.time()
    try:
        processCell(batch_dir, cell, **kw)
        status["status"] = "done"
        return 0
    except Exception as e:
        status.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        print(f"[{name}] FAILED: {status['error']}", file=sys.stderr, flush=True)
        return 1
    finally:
        status["wall_s"] = round(time.time() - t0, 1)
        paths["status"].write_text(json.dumps(status, indent=2))


# -- the final tables ------------------------------------------------------------------------
def _scalar(v) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def summarize(batch_dir: Path) -> Path | None:
    """``final_results.csv`` (one row per cell) and ``final_episodes.csv`` (one per episode)."""
    cells = listCells(batch_dir)
    if not cells:
        return None
    rows, ep_rows = [], []
    for c in cells:
        st, name = c["status"], c["name"]
        row = st.get("row", {})
        entry = {"cell": c["index"], "name": name, **row, "batch_status": st.get("status"),
                 "batch_wall_s": round(st.get("wall_s", 0) or 0, 1),
                 **batches.cellResultMetrics(batch_dir / f"{name}.json")}
        paths = _analysisPaths(batch_dir, name)
        try:
            ast = json.loads(paths["status"].read_text())
            entry["analysis_status"], entry["analysis_error"] = ast.get("status"), ast.get("error")
        except (OSError, ValueError):
            entry["analysis_status"] = "not run"
        analysis = json.loads(paths["json"].read_text()) if paths["json"].exists() and \
            entry["analysis_status"] == "done" else None
        by_id = {e["id"]: e for e in (analysis or {}).get("episodes", [])}
        if analysis:
            entry.update(analysis["metrics"])
            if paths["npz"].exists():        # recomputed, so older analyses gain what stats adds
                metrics, ep_metrics = metricsFromNpz(paths["npz"])
                entry.update(metrics)
                for i, m in ep_metrics.items():
                    by_id[i] = {**by_id.get(i, {}), **m}
        rows.append(entry)

        res = batch_dir / f"{name}.json"
        if res.exists():
            for e in json.loads(res.read_text()).get("episodes", []):
                summary = {k: v for k, v in e.get("summary", {}).items() if _scalar(v)}
                metrics = {k: v for k, v in by_id.get(e["id"], {}).items() if k not in ("id", "n_steps")}
                ep_rows.append({"cell": c["index"], "name": name, **row, **summary, **metrics})

    for fname, table in (("final_results.csv", rows), ("final_episodes.csv", ep_rows)):
        if not table:
            continue
        columns = list(dict.fromkeys(k for e in table for k in e))
        with open(batch_dir / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(table)
    (batch_dir / "final_results.json").write_text(json.dumps({"cells": rows}, indent=2))
    done = sum(r["analysis_status"] == "done" for r in rows)
    print(f"\nfinal: {done}/{len(rows)} cells analysed -> {batch_dir / 'final_results.csv'}, "
          f"{batch_dir / 'final_episodes.csv'}")
    for r in rows:
        kl = r.get("kl_used_opt_mean")
        pos = r.get("fsim_obj_pos_err_mean")
        print(f"  {r['name']:<32} {r['analysis_status']:<8} "
              + (f"KL(used||opt) {kl:.3f}  regret {_fmtInterval(r, 'regret_mean')}  "
                 f"object pos err {1e3 * pos:.2f} mm"
                 if kl is not None and pos is not None else (r.get("analysis_error") or "")))
    return batch_dir / "final_results.csv"


# -- CLI -------------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch_dir", type=Path, help="a results folder written by run_episode_batches.py")
    p.add_argument("--cell", type=int, default=None,
                   help="analyse only this cell (0-based position in the folder's cells, as "
                        "SLURM_ARRAY_TASK_ID); omitted, every cell in order")
    p.add_argument("--count", action="store_true", help="print the number of cells and exit")
    p.add_argument("--summarize", action="store_true", help="only (re)build the final tables")
    p.add_argument("--overwrite", action="store_true", help="redo cells already analysed")
    p.add_argument("--stride", type=int, default=1,
                   help="analyse every k-th saved step (the plans, the forward simulation and the metrics)")
    p.add_argument("--shrinkage", type=float, default=1e-3,
                   help="covariance shrinkage toward noise_sigma^2 I for the KL")
    p.add_argument("--max-episodes", type=int, default=None, help="analyse at most this many episodes per cell")
    p.add_argument("--optimal-args", default=None, metavar="FLAGS",
                   help="run_episodes.py planner flags for the optimal planner, on top of each cell's own "
                        f"and of the defaults ({DEFAULT_OPTIMAL_ARGS}), flag by flag, "
                        "e.g. --optimal-args=\"--n-samples 1024 --temperature 0.1\" (with '=': the value "
                        "starts with --). It may set " + ", ".join("--" + f.replace("_", "-") for f in OPTIMAL_FLAGS))
    args = p.parse_args(argv)
    if args.stride < 1:
        p.error("--stride must be >= 1")
    if args.optimal_args:
        try:
            parseOptimalArgs(args.optimal_args)
        except ValueError as e:
            p.error(str(e))
    batch_dir = args.batch_dir.resolve()
    if not batch_dir.is_dir():
        p.error(f"no such folder: {batch_dir}")
    cells = listCells(batch_dir)
    if args.count:
        print(len(cells))
        return 0
    if not cells:
        p.error(f"{batch_dir} has no cell_*.status.json files; is it an episode-batch results folder?")
    if args.summarize:
        return 0 if summarize(batch_dir) else 1
    kw = dict(stride=args.stride, shrinkage=args.shrinkage, max_episodes=args.max_episodes,
              optimal_args=args.optimal_args)
    if args.cell is not None:
        if not 0 <= args.cell < len(cells):
            p.error(f"--cell {args.cell} is out of range: {batch_dir.name} has cells 0-{len(cells) - 1}")
        return runCell(batch_dir, cells[args.cell], args.overwrite, **kw)
    # Every cell, each in its own process, so one failure does not stop the rest.
    failed = []
    for i, c in enumerate(cells):
        cmd = [sys.executable, __file__, str(batch_dir), "--cell", str(i), "--stride", str(args.stride),
               "--shrinkage", str(args.shrinkage)]
        if args.max_episodes:
            cmd += ["--max-episodes", str(args.max_episodes)]
        if args.optimal_args:
            cmd.append(f"--optimal-args={args.optimal_args}")
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"[{i + 1}/{len(cells)}] {c['name']} ...", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc:
            failed.append(c["name"])
    summarize(batch_dir)
    if failed:
        print(f"\n{len(failed)} cell(s) failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
