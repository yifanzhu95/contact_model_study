#!/usr/bin/env python3
"""Solo MPPI Plan() wall time per model, control period and horizon.

Plans repeatedly from the settled initial state of the cube task (warm start as
in run_episodes), timing each Plan() call, and keeps the mean and median after
the warm-up plans. Run it alone on the GPU. The grid is the period sweep's
(``period_sweep_jobs.py``: 16, 24, ..., 128 ms, every horizon up to the window;
``--periods`` narrows it), and the results go to OUT keyed
``"<model>/<period ms>/<horizon>"``; keys already in OUT are skipped, so a
stopped run resumes. ``plot_period_sweep.py`` reads it
as ``plan_time_solo.json`` for the sync plot's real-time cutoff.

Usage:
    python experiments/bench_plan_time.py --models M1 M2 M3 M4 M6 \\
        --out results/horizon_kl_sweep/plan_time_solo.json [--plans 8 --warmup 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ContactModelStudy.Drivers.run_episodes import (  # noqa: E402
    buildPlannerConfig, buildTasks, parseArgs, rolloutSimOverrides, _newGoal)
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI  # noqa: E402
from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim  # noqa: E402
from ContactModelStudy.Utils.EvalSimulators import makeEvalSim  # noqa: E402
from period_sweep_jobs import BASE_DT_MS, PERIODS_MS, horizons  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["M1", "M2", "M3", "M4", "M6"])
ap.add_argument("--plans", type=int, default=8)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--periods", nargs="+", type=int, default=PERIODS_MS, metavar="MS")
ap.add_argument("--out", required=True)
opts, rest = ap.parse_known_args()
# substeps (4 ms each) -> the horizons swept at that control period
GRID = {p // BASE_DT_MS: horizons(p) for p in opts.periods}


class _NullRecorder:
    def recordGoal(self, goal): pass


res = json.loads(Path(opts.out).read_text()) if Path(opts.out).exists() else {}
for m in opts.models:
    for sub, hs in GRID.items():
        base = ["--rollout-model", m, "--substeps", str(sub), "--no-video", "--no-results",
                "--temperature", "0.1"] + rest
        args = parseArgs(base + ["--horizon", "1"])
        task, eval_task = buildTasks(args)
        world = makeEvalSim("mujoco", eval_task.getModelPath(), eval_task.timestep, task=eval_task)
        eval_task.setSimToInitialState(world)
        _newGoal(task, eval_task, None, _NullRecorder())
        world.Step(int(round(args.settle / eval_task.timestep)))
        q, qd = world.GetState()
        u0 = world.GetControl()
        for h in hs:
            key = f"{m}/{sub * 4}/{h}"
            if key in res:
                continue
            a = parseArgs(base + ["--horizon", str(h)])
            if h < 2:
                a.warm_start = "hold"
            sim = GetContactModelSim(a.rollout_model, task.getModelPath(), N=a.n_samples,
                                     task=task, **rolloutSimOverrides(a, task))
            cfg = buildPlannerConfig(a)
            cfg.mean_cost_over_horizon = True
            p = MPPI(sim, task, cfg)
            t = []
            for i in range(opts.warmup + opts.plans):
                t0 = time.perf_counter()
                p.Plan(q, qd, u=u0)
                t.append(time.perf_counter() - t0)
            t = np.array(t[opts.warmup:])
            res[key] = {"mean": float(t.mean()), "median": float(np.median(t)), "all": t.tolist()}
            print(f"{key:12s} mean {1e3 * t.mean():8.1f} ms  median {1e3 * np.median(t):8.1f} ms  "
                  f"budget {sub * 4} ms  {'OK' if t.mean() <= sub * 4e-3 else 'over'}", flush=True)
            Path(opts.out).write_text(json.dumps(res, indent=0))
            del p, sim
