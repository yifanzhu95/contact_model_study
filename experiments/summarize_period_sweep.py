#!/usr/bin/env python3
"""Summarise the horizon x control-period sweep (``horizon_closed_loop.py``) as one CSV.

One row per run (model, mode, control period) and horizon, from every merged run .json in DIR
(``cl_*`` sync, ``acl_*`` async; the ``chunks/`` are skipped). ``mode`` is the run's ``driver``
setting: ``sync`` (the sim waits for each Plan()) or ``async`` (planning time passes as sim time).

Costs are the run's ``cost_stats``: the mean over the start states with its standard error and 95%
t-interval, for
  running  - running cost averaged over the window's first W-1 steps (f_run)
  terminal - terminal cost at the end of the window (f_term)
  avg      - (running sum + terminal) / W (f_avg)
``plan_ms`` is the mean Plan() time measured in the run. ``real_time`` compares the control period
with the plan time the plots use: the run's own for async and M5, the solo time in
``plan_time_solo.json`` (``bench_plan_time.py``) for sync, when that file is there.

Usage:
    python experiments/summarize_period_sweep.py [DIR] [--out CSV]
DIR defaults to results/horizon_kl_sweep; --out to DIR/period_sweep_summary.csv.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COSTS = {"running": "f_run", "terminal": "f_term", "avg": "f_avg"}
ASYNC_STATS = ("n_plans", "staleness_ms", "stale_ticks", "held_ticks", "exhausted_ticks")


def mean(xs):
    x = np.asarray(xs, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def rows(f: Path, solo: dict | None):
    d = json.loads(f.read_text())
    mode = d.get("driver") or ("async" if f.name.startswith("acl_") else "sync")
    dt_ms = round(d["control_dt"] * 1e3)
    for h in d["horizons"]:
        k = str(h)
        plan_ms = 1e3 * mean(d["plan_times"][k])
        rt_ms = plan_ms
        if mode == "sync" and d["model"] != "M5":
            s = (solo or {}).get(f"{d['model']}/{dt_ms}/{h}")
            rt_ms = 1e3 * s["mean"] if s else None
        r = {"mode": mode, "model": d["model"], "world": d.get("world"),
             "control_dt_ms": dt_ms, "control_hz": round(1e3 / dt_ms, 3),
             "horizon_steps": h, "horizon_s": round(h * dt_ms / 1e3, 4),
             "window_steps": d["window"], "window_s": round(d["window"] * dt_ms / 1e3, 4),
             "n_starts": len(d["starts"]), "repeats": d["repeats"], "start": d.get("start"),
             "goal_face": d.get("goal_face"), "temperature": d["temperature"],
             "n_samples": d.get("n_samples"), "noise_sigma": d["noise_sigma"],
             "sigma_scaling": d.get("sigma_scaling", "none"), "executor": d.get("executor")}
        for name, key in COSTS.items():
            cs = d["cost_stats"][key][k]
            r.update({f"{name}_mean": cs["mean"], f"{name}_se": cs["se"],
                      f"{name}_lo": cs["lo"], f"{name}_hi": cs["hi"]})
        r["fail_frac"] = d["fail_frac"][k]
        r["success_frac"] = d.get("success_frac", {}).get(k)
        r["plan_ms"] = plan_ms
        r["solo_plan_ms"] = rt_ms if mode == "sync" and d["model"] != "M5" else None
        r["real_time"] = None if rt_ms is None else rt_ms <= dt_ms
        for s in ASYNC_STATS:
            r[f"async_{s}"] = mean(d[f"async_{s}"][k]) if f"async_{s}" in d else None
        r["file"] = f.name
        yield r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", type=Path, nargs="?", default=ROOT / "results" / "horizon_kl_sweep")
    ap.add_argument("--out", type=Path, help="CSV path (default DIR/period_sweep_summary.csv)")
    opts = ap.parse_args(argv)
    files = sorted(opts.dir.glob("cl_*.json")) + sorted(opts.dir.glob("acl_*.json"))
    if not files:
        print(f"no cl_*.json or acl_*.json runs in {opts.dir}", file=sys.stderr)
        return 1
    solo_f = opts.dir / "plan_time_solo.json"
    solo = json.loads(solo_f.read_text()) if solo_f.exists() else None
    out = [r for f in files for r in rows(f, solo)]
    out.sort(key=lambda r: (r["mode"] != "sync", r["model"], r["control_dt_ms"], r["horizon_steps"], r["file"]))
    dest = opts.out or opts.dir / "period_sweep_summary.csv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    n = {m: sum(r["mode"] == m for r in out) for m in ("sync", "async")}
    print(f"{len(files)} runs -> {len(out)} rows ({n['sync']} sync, {n['async']} async) -> {dest}")
    if solo is None and n["sync"]:
        print(f"  no {solo_f.name}: real_time is blank for sync M1-M4/M6", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
