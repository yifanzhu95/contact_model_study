#!/usr/bin/env python3
"""Run batches of episodes described by a CSV, locally or as an HPC job array.

Each data row of the CSV is one *cell*: one call of
``ContactModelStudy/Drivers/run_episodes.py`` (so ``n_episodes`` episodes) with
the settings that row gives. The same script runs a single cell on an HPC node
or every cell in turn on a workstation::

    python experiments/run_episode_batches.py batches.csv --check          # validate only
    python experiments/run_episode_batches.py batches.csv                  # every cell, in order
    python experiments/run_episode_batches.py batches.csv --cell 3         # just row 3 (0-based)
    python experiments/run_episode_batches.py batches.csv --summarize --outdir results/my_run

On the HPC, ``hpc/submit_episode_batches.sh`` submits
``hpc/run_episode_batches.slurm`` as a job array with one task per row; each
task runs ``--cell $SLURM_ARRAY_TASK_ID``.

CSV columns
-----------
* **Driver flags.** Any ``run_episodes.py`` option, named with underscores
  instead of dashes: ``task``, ``n_episodes``, ``steps``, ``rollout_model``,
  ``eval_sim``, ``hand_acc``, ``obj_acc``, ``timestep``, ``ctrl_time_step`` or
  ``substeps``, ``time_horizon`` or ``horizon``, ``n_samples``, ``noise_sigma``,
  ``temperature``, ``control_mode``, ``settle``, ``seed``, ``goal_difficulty``,
  ``nconmax``, ``njmax``, ... (``python ContactModelStudy/Drivers/run_episodes.py
  --help`` lists them all). On/off flags take ``true``/``false``
  (``stop_on_success``, ``warm_start``, ``uncertainty``, ``save_steps``, ...).
* **Cost weights.** ``w_quat``, ``w_pos_x``, ... (any of
  ``LeapReorient.COST_WEIGHT_KEYS``) override the object's tuned weight.
* **``label``.** Optional name for the row; it goes into the output file names.
* **``video``.** ``true`` records each episode's video into the output folder.

A blank cell, or a missing column, leaves that setting at the driver's default.
A column that is none of these is an error, so a misspelled column cannot
silently run a whole sweep at the default. Output paths are set by this script,
so ``results`` and ``video`` paths are not columns.

Output
------
All cells write into ``--outdir``, by default
``results/episode_batches_<csv name>_<job id or time>/``. Each cell writes:

* ``cell_<row>[_<label>].json`` and, unless the row sets ``save_steps=false``,
  one ``.npy`` per episode beside it: the ``EpisodeRecorder`` output.
* ``cell_<row>[_<label>].status.json``: done or failed, the driver command line,
  the wall time and any error.
* ``cell_<row>[_<label>].log``: the driver's output (when run as a subprocess).

A cell whose status is ``done`` is skipped when the batch is run again, so a
sweep that hit the wall clock resumes where it stopped. Run it again with the
same CSV and the same ``--outdir``, or pass ``--overwrite`` to redo cells.
``--summarize`` merges every finished cell into ``summary.csv`` and
``summary.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ContactModelStudy.Drivers import run_episodes as drv  # noqa: E402
from ContactModelStudy.Tasks.LeapReorient import COST_WEIGHT_KEYS  # noqa: E402

#: Columns this script owns rather than forwards to the driver.
OWN_COLUMNS = ("label", "video")
#: Driver options this script sets itself, per cell.
MANAGED_OPTIONS = ("results", "video")
#: Driver options a CSV sets another way (``w_*`` columns for cost weights).
HIDDEN_OPTIONS = ("cost_weight",)
_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


# -- CSV -> driver command line ----------------------------------------------
def _driverOptions() -> dict[str, argparse.Action]:
    """Driver options by destination name, skipping the store_const aliases.

    ``--no-video`` and ``--no-results`` share a destination with ``--video``
    and ``--results``; the option that takes a value is the one a column maps
    to.
    """
    out = {}
    for a in drv.build_parser()._actions:
        if not a.option_strings or a.dest == "help":
            continue
        if isinstance(a, argparse._StoreConstAction) and not isinstance(a, argparse._StoreTrueAction):
            continue
        if a.dest in HIDDEN_OPTIONS:
            continue
        out[a.dest] = a
    return out


def _parseBool(value: str, column: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"column {column!r}: expected true/false, got {value!r}")


def readRows(csv_path: Path) -> list[dict[str, str]]:
    """The CSV's data rows, with blank cells removed and header names trimmed."""
    with open(csv_path, newline="") as f:
        rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()}
                for r in csv.DictReader(f)]
    return [{k: v for k, v in r.items() if k and v != ""} for r in rows]


def checkColumns(csv_path: Path) -> None:
    """Reject any column that is not a driver option, a cost weight or one of ours."""
    with open(csv_path, newline="") as f:
        header = [h.strip() for h in next(csv.reader(f), []) if h.strip()]
    options = _driverOptions()
    bad = [h for h in header if h not in options and h not in COST_WEIGHT_KEYS
           and h not in OWN_COLUMNS]
    managed = [h for h in header if h in MANAGED_OPTIONS and h not in OWN_COLUMNS]
    if bad:
        raise ValueError(f"unknown column(s) {bad}. Valid: driver options "
                         f"{sorted(k for k in options if k not in MANAGED_OPTIONS)}, cost weights "
                         f"{list(COST_WEIGHT_KEYS)}, and {list(OWN_COLUMNS)}.")
    if managed:
        raise ValueError(f"column(s) {managed} are set by this script, per cell")
    if len(set(header)) != len(header):
        raise ValueError(f"duplicate column names in {csv_path}")


def cellName(index: int, row: dict[str, str]) -> str:
    """``cell_<index>`` plus the row's label, made safe for a file name."""
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", row.get("label", "")).strip("-")
    return f"cell_{index:04d}" + (f"_{label}" if label else "")


def rowToArgv(row: dict[str, str], outdir: Path, name: str) -> list[str]:
    """The ``run_episodes.py`` command line for one CSV row."""
    options = _driverOptions()
    argv: list[str] = []
    for column, value in row.items():
        if column in OWN_COLUMNS:
            continue
        if column in COST_WEIGHT_KEYS:
            argv += ["--cost-weight", f"{column}={value}"]
            continue
        action = options[column]
        flag = next(s for s in action.option_strings if s.startswith("--"))
        if isinstance(action, argparse.BooleanOptionalAction):
            argv.append(flag if _parseBool(value, column) else "--no-" + flag[2:])
        elif isinstance(action, argparse._StoreTrueAction):
            if _parseBool(value, column):
                argv.append(flag)
        else:
            argv += [flag, value]
    argv += ["--results", str(outdir / f"{name}.json")]
    if _parseBool(row.get("video", "false"), "video"):
        argv += ["--video", str(outdir / f"{name}.mp4")]
    else:
        argv.append("--no-video")
    return argv


def checkRow(argv: list[str]) -> str | None:
    """``None`` if the driver would accept ``argv``, else the driver's error message."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            drv.parseArgs(argv)
    except SystemExit:
        lines = [ln for ln in err.getvalue().splitlines() if ln.strip()]
        return lines[-1] if lines else "rejected by the driver"
    return None


# -- running -------------------------------------------------------------------
def _statusPath(outdir: Path, name: str) -> Path:
    return outdir / f"{name}.status.json"


def isDone(outdir: Path, name: str) -> bool:
    p = _statusPath(outdir, name)
    try:
        return json.loads(p.read_text()).get("status") == "done"
    except (OSError, ValueError):
        return False


def runCell(index: int, row: dict[str, str], outdir: Path, overwrite: bool) -> bool:
    """Run one cell in this process. Returns whether it finished."""
    name = cellName(index, row)
    if not overwrite and isDone(outdir, name):
        print(f"[{name}] already done; skipping (--overwrite to redo)")
        return True
    argv = rowToArgv(row, outdir, name)
    outdir.mkdir(parents=True, exist_ok=True)
    status = {"cell": index, "name": name, "row": row, "argv": argv,
              "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID")}
    print(f"[{name}] run_episodes.py {' '.join(argv)}", flush=True)
    t0 = time.time()
    try:
        drv.main(argv)
        status.update(status="done")
    except BaseException as e:           # also KeyboardInterrupt / SIGTERM from SLURM
        status.update(status="failed", error=f"{type(e).__name__}: {e}",
                      traceback=traceback.format_exc())
        raise
    finally:
        status["wall_s"] = time.time() - t0
        _statusPath(outdir, name).write_text(json.dumps(status, indent=2))
    return True


def runAll(csv_path: Path, rows: list[dict], outdir: Path, overwrite: bool, stop_on_error: bool) -> int:
    """Every cell in order, each in its own process, so one failure does not stop the rest."""
    failed = []
    for i, row in enumerate(rows):
        name = cellName(i, row)
        if not overwrite and isDone(outdir, name):
            print(f"[{i + 1}/{len(rows)}] {name}: already done")
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[{i + 1}/{len(rows)}] {name} ...", flush=True)
        cmd = [sys.executable, __file__, str(csv_path), "--cell", str(i), "--outdir", str(outdir)]
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
        print(f"\n{len(failed)} cell(s) failed: {', '.join(failed)}")
    return 1 if failed else 0


# -- summary -------------------------------------------------------------------
def summarize(csv_path: Path, rows: list[dict], outdir: Path) -> Path | None:
    """Merge every cell's results into ``summary.csv`` and ``summary.json``."""
    table = []
    for i, row in enumerate(rows):
        name = cellName(i, row)
        entry = {"cell": i, "name": name, **row}
        try:
            status = json.loads(_statusPath(outdir, name).read_text())
            entry["status"], entry["wall_s"] = status.get("status"), round(status.get("wall_s", 0), 1)
        except (OSError, ValueError):
            entry["status"] = "not run"
        res = outdir / f"{name}.json"
        if res.exists():
            doc = json.loads(res.read_text())
            eps = [e["summary"] for e in doc.get("episodes", [])]
            plan = [e["plan_s_mean"] for e in eps if e.get("plan_s_mean") is not None]
            steps = [e["steps_to_success"] for e in eps if e.get("steps_to_success") is not None]
            entry.update(
                n_episodes_run=doc.get("n_episodes"), n_success=doc.get("n_success"),
                n_failed=doc.get("n_failed"), success_rate=doc.get("success_rate"),
                mean_steps_to_success=(sum(steps) / len(steps)) if steps else None,
                mean_goals_reached=(sum(e["goals_reached"] for e in eps) / len(eps)) if eps else None,
                plan_ms_mean=(1e3 * sum(plan) / len(plan)) if plan else None,
                results=res.name,
            )
        table.append(entry)
    if not table:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(
        {"csv": str(csv_path), "cells": table}, indent=2))
    columns = list(dict.fromkeys(k for e in table for k in e))
    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(table)
    done = sum(e["status"] == "done" for e in table)
    print(f"\nsummary: {done}/{len(table)} cells done -> {outdir / 'summary.csv'}")
    for e in table:
        rate = e.get("success_rate")
        print(f"  {e['name']:<32} {e['status']:<8} "
              + (f"success {rate:.0%} ({e['n_success']}/{e['n_episodes_run']})" if rate is not None else ""))
    return outdir / "summary.csv"


# -- CLI -----------------------------------------------------------------------
def defaultOutdir(csv_path: Path) -> Path:
    """``results/episode_batches_<csv>_<job id or time>``; one per job array."""
    tag = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") \
        or time.strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / f"episode_batches_{csv_path.stem}_{tag}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="the batches CSV, one row per cell")
    p.add_argument("--cell", type=int, default=None,
                   help="run only this row (0-based, as SLURM_ARRAY_TASK_ID); omitted, run every row")
    p.add_argument("--outdir", type=Path, default=None,
                   help="where every cell writes; default results/episode_batches_<csv>_<job id or time>")
    p.add_argument("--overwrite", action="store_true", help="rerun cells that are already done")
    p.add_argument("--stop-on-error", action="store_true",
                   help="when running every row, stop at the first failed cell")
    p.add_argument("--check", action="store_true",
                   help="validate every row against the driver and print its command line; run nothing")
    p.add_argument("--count", action="store_true", help="print the number of rows and exit")
    p.add_argument("--summarize", action="store_true", help="only (re)build summary.csv/json in --outdir")
    args = p.parse_args(argv)

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        p.error(f"no such CSV: {csv_path}")
    try:
        checkColumns(csv_path)
    except ValueError as e:
        p.error(str(e))
    rows = readRows(csv_path)
    if args.count:
        print(len(rows))
        return 0
    if not rows:
        p.error(f"{csv_path} has no data rows")
    outdir = (args.outdir or defaultOutdir(csv_path)).resolve()

    if args.check:
        bad = 0
        for i, row in enumerate(rows):
            name = cellName(i, row)
            try:
                argv_i = rowToArgv(row, outdir, name)
                err = checkRow(argv_i)
            except ValueError as e:
                argv_i, err = [], str(e)
            bad += err is not None
            print(f"{'OK ' if err is None else 'BAD'} {name}: " + (err or " ".join(argv_i)))
        print(f"\n{len(rows) - bad}/{len(rows)} rows valid")
        return 1 if bad else 0
    if args.summarize:
        return 0 if summarize(csv_path, rows, outdir) else 1
    if args.cell is not None:
        if not 0 <= args.cell < len(rows):
            p.error(f"--cell {args.cell} is out of range: {csv_path.name} has rows 0-{len(rows) - 1}")
        runCell(args.cell, rows[args.cell], outdir, args.overwrite)
        return 0
    print(f"{len(rows)} cells from {csv_path} -> {outdir}")
    return runAll(csv_path, rows, outdir, args.overwrite, args.stop_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
