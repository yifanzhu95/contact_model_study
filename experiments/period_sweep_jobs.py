#!/usr/bin/env python3
"""The period sweep's job lists: ``horizon_closed_loop.py`` runs of every model, control period and horizon.

A *run* is one model at one control period P, every 8 ms from 16 to 128 ms
(``--substeps P / 4`` at the 4 ms timestep), with the same settings in seconds,
rounded to whole control steps at P:

* the window W: round(1 s / P) steps (0.96-1.04 s), long enough to separate
  the planners' costs and short enough that none reaches the goal;
* the horizons: log-spaced, the same lengths in seconds at every period: one
  step and round(T / P) steps for T = 1.28 s divided by 2^(1/4) again and
  again, down to 12 ms (10-21 horizons per period: 1-8, 10, 12, 14, 17, 20 at
  64 ms). That is every step up to 8 and then ~19% apart: dense under and
  just past the real-time front, which sits at 4-10 steps for M1-M4 and M6 at
  every period (plan time grows with steps x substeps);
* the trajectories: ``SEEDS`` = 10 (``--seeds`` for fewer), each from the
  task's initial pose toward one goal, ``GOAL_FACE`` = "S" (the cube flipped
  upside down), with seed i's planner noise (``horizon_closed_loop.py --seeds
  --goal-face``). The seeds are the same for every model, period and horizon,
  and each trajectory is scored by its cost, the planner's cost averaged over
  its horizon (``--mean-cost-over-horizon``); the plots show the mean over the
  seeds with its standard error.

Sync runs (``cl_``) cover every model, M1-M6; async runs (``acl_``) M1-M4 and
M6. A run can be too long for one job at short periods (plan time grows with the
horizon), so it can be split into *chunks* of at most ``PER_CHUNK`` horizons each,
interleaved (chunk c of C takes horizons c + 1, c + 1 + C, ...) so every chunk
costs about the same. Every chunk runs the same seeds, so the horizons stay
paired.

Three lists, by ``--stage``:

* ``runs``: ``NAME MODEL SUBSTEPS SEEDS WINDOW GOAL CHUNKS``, one line per run,
  for the merge;
* ``chunks``: ``NAME CHUNK MODEL SUBSTEPS SEEDS WINDOW GOAL HORIZONS...``, one
  line per chunk, CHUNK like ``c03of10``;
* ``models``: the models, one per line: one HPC array task (one GPU) each.

``--job I`` prints line I alone (0-based).

Usage:
    python experiments/period_sweep_jobs.py sync                      # every sync chunk
    python experiments/period_sweep_jobs.py async --stage runs --count
    python experiments/period_sweep_jobs.py async --job 3
    python experiments/period_sweep_jobs.py sync --stage models --job 4     # M5
    python experiments/period_sweep_jobs.py sync --models M1 M2 --periods 16 64 --seeds 3
"""
import argparse
import math
import sys

#: The eval and rollout timestep (ms): a control period is a whole number of these.
BASE_DT_MS = 4
PERIODS_MS = tuple(range(16, 129, 8))
WINDOW_MS = 1000
HORIZON_MS = 1280
#: The horizon grid: HORIZON_MS divided by HORIZON_RATIO this many times (from 0), in whole control steps.
HORIZON_RATIO = 2 ** 0.25
HORIZON_STEPS = 28
SEEDS = 10
GOAL_FACE = "S"
MODELS = {"sync": ("M1", "M2", "M3", "M4", "M5", "M6"), "async": ("M1", "M2", "M3", "M4", "M6")}
PREFIX = {"sync": "cl", "async": "acl"}
#: Horizons per chunk, at most: 40 keeps M1's 16 ms chunks near 4 h on an RTX 4090 (two of them).
#: M5 plans several times slower, so its chunks are smaller.
PER_CHUNK = 40
PER_CHUNK_MODEL = {"M5": 16}


def steps(ms: float, period: int) -> int:
    """``ms`` in whole control steps at ``period`` ms, at least one."""
    return max(1, round(ms / period))


def window(period: int) -> int:
    return steps(WINDOW_MS, period)


def horizons(period: int) -> list[int]:
    """The horizons (control steps) swept at ``period`` ms: one step and ``HORIZON_MS / HORIZON_RATIO^k``
    for k < ``HORIZON_STEPS``, each rounded to whole steps, so they sit evenly on a log axis."""
    return sorted({1} | {steps(HORIZON_MS * HORIZON_RATIO ** -k, period) for k in range(HORIZON_STEPS)})


def nChunks(model: str, period: int) -> int:
    return math.ceil(len(horizons(period)) / PER_CHUNK_MODEL.get(model, PER_CHUNK))


def run(mode: str, model: str, period: int, seeds: int = SEEDS) -> list:
    name = f"{PREFIX[mode]}_{model}_onM5_T0.1_meancost_dt{period}_goal{GOAL_FACE}_seeds{seeds}"
    return [name, model, period // BASE_DT_MS, seeds, window(period), GOAL_FACE, nChunks(model, period)]


def chunks(mode: str, model: str, period: int, seeds: int = SEEDS) -> list[list]:
    name, *settings, C = run(mode, model, period, seeds)
    hs = horizons(period)
    return [[name, f"c{c:02d}of{C:02d}", *settings, *hs[c::C]] for c in range(C)]


def jobs(mode: str, stage: str = "chunks", models=None, periods=None, seeds: int = SEEDS) -> list[list]:
    """Every line of ``stage`` for ``mode``, by period then model (then chunk)."""
    if stage == "models":
        return [[m] for m in models or MODELS[mode]]
    pairs = [(m, p) for p in periods or PERIODS_MS for m in models or MODELS[mode]]
    if stage == "runs":
        return [run(mode, m, p, seeds) for m, p in pairs]
    return [c for m, p in pairs for c in chunks(mode, m, p, seeds)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=sorted(MODELS))
    ap.add_argument("--stage", choices=("runs", "chunks", "models"), default="chunks")
    ap.add_argument("--models", nargs="+", default=None, help="these models only (default: the mode's)")
    ap.add_argument("--periods", nargs="+", type=int, default=None, metavar="MS",
                    help=f"these control periods only, in ms (default: {PERIODS_MS[0]}-{PERIODS_MS[-1]} "
                         f"every {PERIODS_MS[1] - PERIODS_MS[0]})")
    ap.add_argument("--seeds", type=int, default=SEEDS, help=f"trajectories per run (default {SEEDS})")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--count", action="store_true", help="print the number of lines and exit")
    g.add_argument("--job", type=int, default=None, metavar="I", help="print line I (0-based) only")
    opts = ap.parse_args(argv)
    bad = [p for p in opts.periods or () if p <= 0 or p % BASE_DT_MS]
    if bad:
        ap.error(f"--periods {bad}: a control period is a positive multiple of {BASE_DT_MS} ms")
    out = jobs(opts.mode, opts.stage, opts.models, opts.periods, opts.seeds)
    if opts.count:
        print(len(out))
        return 0
    if opts.job is not None:
        if not 0 <= opts.job < len(out):
            print(f"job {opts.job} out of range: {len(out)} {opts.mode} {opts.stage}", file=sys.stderr)
            return 1
        out = [out[opts.job]]
    for line in out:
        print(" ".join(map(str, line)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
