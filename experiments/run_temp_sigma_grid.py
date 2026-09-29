#!/usr/bin/env python3
"""Search MPPI's temperature x noise_sigma per cell, with episodes run interwoven.

Each data row of the CSV is one *cell*: a fixed set of episode settings (task,
contact model, fidelity, horizon, control step, sample count, ...) plus the two
lists to search, ``temperatures`` and ``noise_sigmas``. For every point of that
grid the cell runs ``n_episodes`` episodes through
``ContactModelStudy/Drivers/run_episodes_interwoven.py`` (two episodes in
flight: one plans on the GPU while the other steps on the CPU), then ranks the
points.

Why per cell: MPPI's temperature divides the cost inside its softmax, so the
useful value scales with the size of the task cost, which differs per contact
model, object and horizon. The readable picture is one grid per cell, and the
grid worth searching differs per cell, so each row names its own.

Every point of a cell sees the same episodes: an episode's goals and planner
noise come from ``(seed, episode index)`` alone, so two points differ only in
their temperature and noise. Points are ranked by success rate, ties broken by
mean steps to success.

::

    python experiments/run_temp_sigma_grid.py grid.csv --check     # validate only
    python experiments/run_temp_sigma_grid.py grid.csv             # every cell, in order
    python experiments/run_temp_sigma_grid.py grid.csv --cell 2    # just row 2 (0-based)
    python experiments/run_temp_sigma_grid.py grid.csv --summarize --outdir results/my_grid

On the HPC, ``hpc/submit_temp_sigma_grid.sh`` submits one array task per row.

CSV columns
-----------
* ``temperatures`` and ``noise_sigmas`` (required): the values to search,
  separated by spaces or commas (quote a comma list), all positive, no repeats.
  Temperatures are the outer loop. A single noise_sigma makes it a 1-D
  temperature sweep.
* Everything ``run_episode_batches.py`` accepts: any ``run_episodes.py`` option
  with underscores for dashes (``n_episodes``, ``rollout_model``, ``eval_sim``,
  ``time_horizon``, ``ctrl_time_step``, ``n_samples``, ...), ``w_*`` cost
  weights, ``label`` and ``video``. A blank cell keeps the driver's default.
  ``temperature`` and ``noise_sigma`` themselves are not columns: they are
  what the grid sets.

Output
------
``--outdir`` (default ``results/temp_sigma_grid_<csv name>_<job id or time>/``)
holds one folder per cell, ``cell_<row>[_<label>]/``, containing:

* ``point_<i>_T<temperature>_s<noise>.json`` (+ ``.npy`` if ``save_steps``): that
  point's episodes, the ``EpisodeRecorder`` output;
* ``point_<i>_....status.json``: done or failed, the command line, wall time;
* ``grid_summary.json``: the cell's points ranked, and the best one.

and, at the top, ``summary.csv`` (one row per point, with its cell's settings
and its rank within the cell) and ``best.csv`` (each cell's best point).
A point marked done is skipped when the grid is run again into the same folder,
so a cell that hit the wall clock resumes point by point.
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

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_episode_batches as batches  # noqa: E402
from ContactModelStudy.Drivers import run_episodes_interwoven as interwoven  # noqa: E402

GRID_COLUMNS = ("temperatures", "noise_sigmas")
OWN_COLUMNS = batches.OWN_COLUMNS + GRID_COLUMNS
#: Set per point by this script, so not allowed as columns.
MANAGED_OPTIONS = batches.MANAGED_OPTIONS + ("temperature", "noise_sigma")


# -- grid ------------------------------------------------------------------------
def parseValues(raw: str, column: str) -> list[float]:
    """``"40,20,10"`` or ``"40 20 10"`` -> ``[40.0, 20.0, 10.0]``; positive, no repeats."""
    tokens = [t for t in raw.replace(",", " ").split() if t]
    if not tokens:
        raise ValueError(f"{column} is empty")
    values = []
    for t in tokens:
        try:
            v = float(t)
        except ValueError:
            raise ValueError(f"{column}: {t!r} is not a number") from None
        if v <= 0:
            raise ValueError(f"{column}: values must be > 0, got {v:g}")
        values.append(v)
    if len(set(values)) != len(values):
        raise ValueError(f"{column} lists a value twice: {raw!r}")
    return values


def gridPoints(row: dict[str, str]) -> list[tuple[float, float]]:
    """The row's ``(temperature, noise_sigma)`` points, temperature outermost."""
    temps = parseValues(row.get("temperatures", ""), "temperatures")
    sigmas = parseValues(row.get("noise_sigmas", ""), "noise_sigmas")
    return [(t, s) for t in temps for s in sigmas]


def pointName(i: int, t: float, s: float) -> str:
    return f"point_{i:03d}_T{t:g}_s{s:g}"


def pointArgv(row: dict[str, str], cell_dir: Path, i: int, t: float, s: float) -> list[str]:
    """The interwoven driver's command line for one grid point."""
    argv = batches.rowToArgv(row, cell_dir, pointName(i, t, s), own_columns=OWN_COLUMNS)
    return argv + ["--temperature", f"{t:g}", "--noise-sigma", f"{s:g}"]


# -- running ---------------------------------------------------------------------
def runCell(index: int, row: dict[str, str], outdir: Path, overwrite: bool) -> int:
    """Every point of one cell, in this process. Returns 1 if any point failed."""
    name = batches.cellName(index, row)
    cell_dir = outdir / name
    cell_dir.mkdir(parents=True, exist_ok=True)
    points = gridPoints(row)
    print(f"[{name}] {len(points)} points: temperatures {row['temperatures']}, "
          f"noise_sigmas {row['noise_sigmas']}", flush=True)
    failed = 0
    for i, (t, s) in enumerate(points):
        pname = pointName(i, t, s)
        if not overwrite and batches.isDone(cell_dir, pname):
            print(f"  [{i + 1}/{len(points)}] T={t:g} sigma={s:g}: already done", flush=True)
            continue
        argv = pointArgv(row, cell_dir, i, t, s)
        print(f"  [{i + 1}/{len(points)}] T={t:g} sigma={s:g}: run_episodes_interwoven.py "
              + " ".join(argv), flush=True)
        status = {"cell": index, "point": i, "temperature": t, "noise_sigma": s, "row": row,
                  "argv": argv, "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID")}
        t0 = time.time()
        try:
            rc = interwoven.main(argv)
            status["status"] = "done" if rc == 0 else "failed"
            failed += rc != 0
        except KeyboardInterrupt:
            status.update(status="failed", error="interrupted")
            raise
        except Exception as e:
            status.update(status="failed", error=f"{type(e).__name__}: {e}",
                          traceback=traceback.format_exc())
            failed += 1
            print(f"  point failed: {status['error']}", flush=True)
        finally:
            status["wall_s"] = time.time() - t0
            (cell_dir / f"{pname}.status.json").write_text(json.dumps(status, indent=2))
    ranked = cellSummary(index, row, outdir)
    printRanked(name, row, ranked)
    return 1 if failed else 0


def runAll(csv_path: Path, rows: list[dict], outdir: Path, overwrite: bool, stop_on_error: bool) -> int:
    """Every cell in order, each in its own process, so one failure does not stop the rest."""
    failed = []
    outdir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        name = batches.cellName(i, row)
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
        print(f"\n{len(failed)} cell(s) had failures: {', '.join(failed)}")
    return 1 if failed else 0


# -- ranking and summaries ---------------------------------------------------------
def _pointResult(cell_dir: Path, i: int, t: float, s: float) -> dict:
    """One point's outcome, read from its status and results files."""
    pname = pointName(i, t, s)
    entry = {"point": i, "temperature": t, "noise_sigma": s, "name": pname}
    try:
        status = json.loads((cell_dir / f"{pname}.status.json").read_text())
        entry["status"], entry["wall_s"] = status.get("status"), round(status.get("wall_s", 0), 1)
    except (OSError, ValueError):
        entry["status"] = "not run"
    res = cell_dir / f"{pname}.json"
    if res.exists():
        doc = json.loads(res.read_text())
        eps = [e["summary"] for e in doc.get("episodes", [])]
        steps = [e["steps_to_success"] for e in eps if e.get("steps_to_success") is not None]
        plan = [e["plan_s_mean"] for e in eps if e.get("plan_s_mean") is not None]
        entry.update(
            n_episodes_run=doc.get("n_episodes"), n_success=doc.get("n_success"),
            n_failed=doc.get("n_failed"), success_rate=doc.get("success_rate"),
            mean_steps_to_success=(sum(steps) / len(steps)) if steps else None,
            mean_goals_reached=(sum(e["goals_reached"] for e in eps) / len(eps)) if eps else None,
            plan_ms_mean=(1e3 * sum(plan) / len(plan)) if plan else None,
            results=str(res.relative_to(cell_dir.parent)),
        )
    return entry


def _rankKey(e: dict):
    """Success rate (higher first), then mean steps to success (lower first)."""
    steps = e.get("mean_steps_to_success")
    return (-(e.get("success_rate") or 0.0), float("inf") if steps is None else steps)


def cellSummary(index: int, row: dict[str, str], outdir: Path) -> list[dict]:
    """Rank one cell's finished points and write its ``grid_summary.json``."""
    name = batches.cellName(index, row)
    cell_dir = outdir / name
    points = [_pointResult(cell_dir, i, t, s) for i, (t, s) in enumerate(gridPoints(row))]
    ranked = sorted((p for p in points if p.get("success_rate") is not None), key=_rankKey)
    for r, p in enumerate(ranked, 1):
        p["rank"] = r
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "grid_summary.json").write_text(json.dumps({
        "cell": index, "name": name, "row": row,
        "temperatures": parseValues(row["temperatures"], "temperatures"),
        "noise_sigmas": parseValues(row["noise_sigmas"], "noise_sigmas"),
        "points": points, "ranked": [p["point"] for p in ranked],
        "best": ranked[0] if ranked else None,
    }, indent=2))
    return points


def printRanked(name: str, row: dict[str, str], points: list[dict]) -> None:
    ranked = sorted((p for p in points if p.get("success_rate") is not None), key=_rankKey)
    print(f"\n{name}: {len(ranked)}/{len(points)} points finished, ranked")
    print(f"  {'T':>8} {'sigma':>7} {'success':>8} {'mean steps':>11} {'plan ms':>8}")
    for p in ranked:
        steps = p.get("mean_steps_to_success")
        plan = p.get("plan_ms_mean")
        print(f"  {p['temperature']:>8g} {p['noise_sigma']:>7g} {p['success_rate']:>7.0%} "
              f"{'—' if steps is None else f'{steps:.1f}':>11} "
              f"{'' if plan is None else f'{plan:.1f}':>8}")
    if ranked:
        best = ranked[0]
        argv = batches.rowToArgv(row, Path("."), "x", own_columns=OWN_COLUMNS)
        flags = argv[:argv.index("--results")]       # the row's own settings only
        print(f"  best: T={best['temperature']:g} sigma={best['noise_sigma']:g}. Replay with:\n"
              f"    python ContactModelStudy/Drivers/run_episodes.py {' '.join(flags)} "
              f"--temperature {best['temperature']:g} --noise-sigma {best['noise_sigma']:g}")


def summarize(csv_path: Path, rows: list[dict], outdir: Path) -> Path | None:
    """Every cell's points into ``summary.csv``/``summary.json``, and each cell's best into ``best.csv``."""
    table, best = [], []
    for i, row in enumerate(rows):
        name = batches.cellName(i, row)
        points = cellSummary(i, row, outdir)
        settings = {k: v for k, v in row.items()}
        for p in points:
            table.append({"cell": i, "cell_name": name, **settings, **p})
        ranked = sorted((p for p in points if p.get("rank")), key=lambda p: p["rank"])
        best.append({"cell": i, "cell_name": name, **settings,
                     **(ranked[0] if ranked else {"status": "no finished points"})})
    if not table:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps({"csv": str(csv_path), "points": table,
                                                     "best": best}, indent=2))
    for fname, rows_out in (("summary.csv", table), ("best.csv", best)):
        columns = list(dict.fromkeys(k for e in rows_out for k in e))
        with open(outdir / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(rows_out)
    done = sum(p.get("status") == "done" for p in table)
    print(f"\nsummary: {done}/{len(table)} points done across {len(rows)} cells -> "
          f"{outdir / 'summary.csv'}, {outdir / 'best.csv'}")
    for b in best:
        if b.get("success_rate") is not None:
            print(f"  {b['cell_name']:<32} best T={b['temperature']:g} sigma={b['noise_sigma']:g}  "
                  f"success {b['success_rate']:.0%} ({b['n_success']}/{b['n_episodes_run']})")
        else:
            print(f"  {b['cell_name']:<32} no finished points")
    return outdir / "summary.csv"


# -- CLI -------------------------------------------------------------------------
def defaultOutdir(csv_path: Path) -> Path:
    tag = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") \
        or time.strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / f"temp_sigma_grid_{csv_path.stem}_{tag}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="the grid CSV, one row per cell")
    p.add_argument("--cell", type=int, default=None,
                   help="run only this row (0-based, as SLURM_ARRAY_TASK_ID); omitted, run every row")
    p.add_argument("--outdir", type=Path, default=None,
                   help="where every cell writes; default results/temp_sigma_grid_<csv>_<job id or time>")
    p.add_argument("--overwrite", action="store_true", help="rerun points that are already done")
    p.add_argument("--stop-on-error", action="store_true",
                   help="when running every row, stop at the first cell with a failure")
    p.add_argument("--check", action="store_true",
                   help="validate every row and grid point against the driver; run nothing")
    p.add_argument("--count", action="store_true", help="print the number of rows and exit")
    p.add_argument("--summarize", action="store_true", help="only (re)build the summaries in --outdir")
    args = p.parse_args(argv)

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        p.error(f"no such CSV: {csv_path}")
    try:
        batches.checkColumns(csv_path, own_columns=OWN_COLUMNS, managed=MANAGED_OPTIONS,
                             required=GRID_COLUMNS)
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
                points = gridPoints(row)
                err = None
                for j, (t, s) in enumerate(points):   # every point, so a bad value names itself
                    err = batches.checkRow(pointArgv(row, outdir / name, j, t, s))
                    if err:
                        break
                detail = f"{len(points)} points x {row.get('n_episodes', 'default')} episodes"
            except ValueError as e:
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
        return runCell(args.cell, rows[args.cell], outdir, args.overwrite)
    print(f"{len(rows)} cells from {csv_path} -> {outdir}")
    return runAll(csv_path, rows, outdir, args.overwrite, args.stop_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
