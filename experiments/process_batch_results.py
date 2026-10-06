#!/usr/bin/env python3
"""Analyse the results folder of an episode batch (``run_episode_batches.py``), cell by cell.

For every cell it measures how far the planner that ran was from an *optimal*
planner, one that plans with the eval model itself, and how far the planner's
rollout model is from the eval simulator:

1. **KL divergence between the planners.** The *used* planner is rebuilt from
   the cell's recorded command line. The *optimal* planner is the same planner
   (sample count, temperature, noise, cost weights, control step and horizon
   as durations) planning on the eval scene, at the eval timestep, with the
   eval simulator's contact model (``--plan-on-eval``, rollout model = eval
   sim). Both walk every recorded episode in order, planning from the recorded
   states with the recorded previous controls and the goal of each step; each
   keeps its own warm state, is reset where the episode reset its planner, and
   both draw the same noise (common random numbers), so the KL measures the
   models rather than sampling luck. Each step's first actions are moment-
   matched to Gaussians (``Utils/PlannerKLDiv.py``); both ``KL(used || opt)``
   and ``KL(opt || used)`` are kept.
2. **Forward-simulation error between the simulators.** From every recorded
   state, with that step's recorded action, both the rollout model (rollout
   scene, rollout timestep) and the eval simulator (eval scene, eval timestep)
   are advanced one control step, all steps of an episode at once as the worlds
   of a vectorized simulator. The two predictions are compared: object position
   (m) and orientation (rad), hand joints (RMS rad), and the full ``qpos`` /
   ``qvel``. The eval prediction is also compared with the recorded next state,
   which shows what restarting from a recorded state (without the solver's warm
   start) costs.

The optimal planner and the eval prediction need the eval simulator to be a
vectorized contact model (``eval_sim`` M1-M4); a cell with a CPU eval
simulator is an error. The cell must have been run with ``save_steps=true``.

::

    python experiments/process_batch_results.py <batch dir> --count
    python experiments/process_batch_results.py <batch dir> --cell 3
    python experiments/process_batch_results.py <batch dir>               # every cell, in order
    python experiments/process_batch_results.py <batch dir> --summarize   # final tables only

On the HPC, ``hpc/submit_process_batch_results.sh <batch dir>`` runs one array
task per cell and then the summary.

Output, in the batch folder
---------------------------
* ``analysis/<cell>.analysis.json``: the cell's metrics (mean, median, p90 of
  each), per episode and overall, and how both planners were built;
* ``analysis/<cell>.analysis.npz``: the per-step arrays;
* ``analysis/<cell>.analysis_status.json``: done or failed, with the error;
* ``final_results.csv`` / ``.json``: one row per cell: its settings, the batch's
  own results (as in ``summary.csv``) and every analysis metric;
* ``final_episodes.csv``: one row per episode, the same way.

A cell whose analysis is done is skipped when run again (``--overwrite`` redoes it).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
            "status": d / f"{name}.analysis_status.json"}


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
    return argparse.Namespace(**{**defaults, **cli_args, "results": None, "video": None})


def optimalArgs(args: argparse.Namespace, control_timestep: float, horizon_duration: float) -> argparse.Namespace:
    """The used planner's settings, planning on the eval scene with the eval simulator's model.

    Control step and horizon are passed as durations, so the optimal planner
    covers the same span of time at the eval timestep.
    """
    return argparse.Namespace(**{**vars(args), "plan_on_eval": True, "rollout_model": args.eval_sim,
                                 "substeps": None, "ctrl_time_step": control_timestep,
                                 "horizon": None, "time_horizon": horizon_duration})


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


def goalIndexPerStep(goals: list[tuple[int, np.ndarray]], n_steps: int) -> np.ndarray:
    """Which goal was current at each step."""
    starts = np.array([s for s, _ in goals])
    return np.searchsorted(starts, np.arange(n_steps), side="right") - 1


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
    """``<prefix>_mean``, ``_median`` and ``_p90`` over the finite values (and how many were not)."""
    v = np.asarray(values, dtype=float).ravel()
    ok = v[np.isfinite(v)]
    out = {f"{prefix}_mean": float(ok.mean()) if ok.size else None,
           f"{prefix}_median": float(np.median(ok)) if ok.size else None,
           f"{prefix}_p90": float(np.percentile(ok, 90)) if ok.size else None}
    if ok.size != v.size:
        out[f"{prefix}_n_nonfinite"] = int(v.size - ok.size)
    return out


# -- the two analyses ---------------------------------------------------------------------
def _resetPlanner(planner, mean, noise_seed: int | None = None) -> None:
    planner.Reset(mean)
    if noise_seed is not None:
        planner.LoadState({**planner.SaveState(), "u_prev": None, "last_action_seq": None,
                           "noise_seed": int(noise_seed), "resample_count": 0})


def klReplay(ep, goals, used, opt, args, u0, noise_seed: int, stride: int, shrinkage: float) -> dict:
    """Both planners along one recorded episode; per-step KLs both ways.

    ``used`` and ``opt`` are ``(task, planner)`` pairs.
    """
    from ContactModelStudy.Utils.PlannerKLDiv import GaussianKL, PlannerGaussian
    T = ep.n_steps
    reset_mean = u0 if args.control_mode == "absolute" else None
    gidx = goalIndexPerStep(goals, T)
    out = {k: np.full(T, np.nan) for k in ("kl_used_opt", "kl_opt_used", "mean_dist")}
    current = None
    for t in range(T):
        if gidx[t] != current:                           # a new goal: as run_episode does
            current = gidx[t]
            for task, planner in (used, opt):
                task.setGoal(goals[current][1])
                _resetPlanner(planner, reset_mean, noise_seed if t == 0 else None)
        if t % stride:
            continue
        u_prev = u0 if t == 0 else ep.u[t - 1]
        try:
            mu_u, cov_u = PlannerGaussian(used[1], ep.q[t], ep.q_dot[t], u=u_prev, shrinkage=shrinkage,
                                          restore=False)
            mu_o, cov_o = PlannerGaussian(opt[1], ep.q[t], ep.q_dot[t], u=u_prev, shrinkage=shrinkage,
                                          restore=False)
        except RuntimeError:                              # a failed plan
            continue
        out["kl_used_opt"][t] = GaussianKL(mu_u, cov_u, mu_o, cov_o)
        out["kl_opt_used"][t] = GaussianKL(mu_o, cov_o, mu_u, cov_u)
        out["mean_dist"][t] = float(np.linalg.norm(mu_u - mu_o))
    return out


def forwardSimErrors(ep, rollout_sim, eval_sim, rollout_steps: int, eval_steps: int, indices) -> dict:
    """One control step from every recorded state, on both simulators at once."""
    T, N = ep.n_steps, rollout_sim.N
    pad = lambda a: np.concatenate([a, np.repeat(a[-1:], N - T, axis=0)]) if N > T else a  # noqa: E731
    q, v, u = pad(ep.q), pad(ep.q_dot), pad(ep.u)
    preds = []
    for sim, steps in ((rollout_sim, rollout_steps), (eval_sim, eval_steps)):
        sim.ClearControlSequence()
        sim.SetState(q, v)
        sim.SetControl(u)
        sim.Step(steps)
        qn, vn = sim.GetState()
        preds.append((np.asarray(qn, dtype=float)[:T], np.asarray(vn, dtype=float)[:T]))
    (qr, vr), (qe, ve) = preds
    errs = {f"fsim_{k}": e for k, e in stateErrors(qr, vr, qe, ve, indices).items()}
    # The eval prediction against what the episode actually did next.
    rec = stateErrors(qe[:-1], ve[:-1], ep.q[1:], ep.q_dot[1:], indices)
    for k, e in rec.items():
        errs[f"eval_vs_recorded_{k}"] = np.append(e, np.nan)
    return errs


# -- one cell --------------------------------------------------------------------------------
def processCell(batch_dir: Path, cell: dict, stride: int = 1, shrinkage: float = 1e-3,
                max_episodes: int | None = None) -> dict:
    """Both analyses for one cell; writes and returns its analysis record."""
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
    episodes = list(rep)[:max_episodes] if max_episodes else list(rep)
    all_goals = episodeGoals(rep, args)

    # The used planner, as run_episodes.main built it.
    task, eval_task = drv.buildTasks(args)
    used_sim = GetContactModelSim(args.rollout_model, task.getModelPath(), N=args.n_samples,
                                  **drv.rolloutSimOverrides(args, task))
    used = (task, MPPI(used_sim, task, drv.buildPlannerConfig(args)))
    # The optimal planner: the same, with the eval model.
    oargs = optimalArgs(args, used_sim.config.control_timestep, used_sim.config.horizon_duration)
    otask, _ = drv.buildTasks(oargs)
    opt_sim = GetContactModelSim(oargs.rollout_model, otask.getModelPath(), N=oargs.n_samples,
                                 **drv.rolloutSimOverrides(oargs, otask))
    opt = (otask, MPPI(opt_sim, otask, drv.buildPlannerConfig(oargs)))
    _, _, u0 = eval_task.getInitialState()

    # The forward-simulation pair: one world per recorded step.
    n_worlds = max(ep.n_steps for ep in episodes)
    rollout_steps = used_sim.config.resolved_substeps
    eval_steps = int(round(rollout_steps * task.timestep / eval_task.timestep))
    overrides = drv.rolloutSimOverrides(args, task)
    fs_rollout = GetContactModelSim(args.rollout_model, task.getModelPath(), N=n_worlds, **overrides)
    fs_eval = GetContactModelSim(args.eval_sim, eval_task.getModelPath(), N=n_worlds,
                                 timestep=eval_task.timestep,
                                 **{k: v for k, v in overrides.items() if k in ("nconmax", "njmax")})

    print(f"[{name}] {len(episodes)} episodes; used planner {args.rollout_model} on "
          f"{Path(task.getModelPath()).name} ({rollout_steps} x {task.timestep * 1e3:g} ms), optimal "
          f"{args.eval_sim} on {Path(otask.getModelPath()).name} ({opt_sim.config.resolved_substeps} x "
          f"{otask.timestep * 1e3:g} ms), KL every {stride} step(s)", flush=True)
    per_step: dict[str, list[np.ndarray]] = {}
    ep_records = []
    t0 = time.time()
    for k, (ep, goals) in enumerate(zip(episodes, all_goals)):
        ep_index = int(ep.summary.get("episode", k))
        noise_seed = episodeSeeds(args.seed, ep_index)[1]
        kl = klReplay(ep, goals, used, opt, args, u0, noise_seed, stride, shrinkage)
        fs = forwardSimErrors(ep, fs_rollout, fs_eval, rollout_steps, eval_steps, eval_task.indices)
        rec = {"id": ep.id, "episode": ep_index, "n_steps": ep.n_steps}
        for key, arr in {**kl, **fs}.items():
            per_step.setdefault(key, []).append(arr)
            rec.update({k2: v for k2, v in stats(key, arr).items() if k2.endswith("_mean")})
        ep_records.append(rec)
        print(f"  episode {ep_index}: {ep.n_steps} steps, KL(used||opt) "
              f"{rec.get('kl_used_opt_mean') or float('nan'):.3f}, object pos err "
              f"{1e3 * (rec.get('fsim_obj_pos_err_mean') or float('nan')):.2f} mm  "
              f"({time.time() - t0:.0f} s)", flush=True)
    for sim in (used_sim, opt_sim, fs_rollout, fs_eval):
        sim.Close()

    metrics = {}
    for key, arrs in per_step.items():
        metrics.update(stats(key, np.concatenate(arrs)))
    record = {
        "cell": cell["index"], "name": name, "results": res.name,
        "rollout_model": args.rollout_model, "eval_sim": args.eval_sim,
        "plan_on_eval": bool(getattr(args, "plan_on_eval", False)),
        "stride": stride, "shrinkage": shrinkage, "n_episodes_analyzed": len(episodes),
        "optimal_planner": {"rollout_model": oargs.rollout_model, "scene": Path(otask.getModelPath()).name,
                            "timestep": otask.timestep, "substeps": opt_sim.config.resolved_substeps,
                            "horizon": opt_sim.config.resolved_horizon},
        "forward_sim": {"rollout_steps": rollout_steps, "eval_steps": eval_steps,
                        "control_timestep": used_sim.config.control_timestep},
        "metrics": metrics, "episodes": ep_records, "wall_s": round(time.time() - t0, 1),
    }
    paths = _analysisPaths(batch_dir, name)
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(record, indent=2))
    np.savez(paths["npz"], **{f"{key}__{rec['id']}": arr for key, arrs in per_step.items()
                              for rec, arr in zip(ep_records, arrs)})
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
        if analysis:
            entry.update(analysis["metrics"])
        rows.append(entry)

        res = batch_dir / f"{name}.json"
        if res.exists():
            by_id = {e["id"]: e for e in (analysis or {}).get("episodes", [])}
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
              + (f"KL(used||opt) {kl:.3f}  object pos err {1e3 * pos:.2f} mm" if kl is not None and pos is not None
                 else (r.get("analysis_error") or "")))
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
    p.add_argument("--stride", type=int, default=1, help="compute the KL at every k-th step")
    p.add_argument("--shrinkage", type=float, default=1e-3,
                   help="covariance shrinkage toward noise_sigma^2 I for the KL")
    p.add_argument("--max-episodes", type=int, default=None, help="analyse at most this many episodes per cell")
    args = p.parse_args(argv)
    if args.stride < 1:
        p.error("--stride must be >= 1")
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
    kw = dict(stride=args.stride, shrinkage=args.shrinkage, max_episodes=args.max_episodes)
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
