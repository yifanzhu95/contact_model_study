#!/usr/bin/env python3
"""``horizon_closed_loop.py``'s results: writing a run's .npz and .json, and merging a run's horizon chunks.

A run's raw costs (``run`` (S, nh, K, W), ``term`` (S, nh, K), ``fail_at``,
``success_at``, the per-window stats) go to the .npz; the .json holds the run's
settings and, per horizon, the free energies over the K repeats and
``cost_stats``: each cost's mean over the start states (of its mean over the
repeats) with its standard error and 95% t-interval (see ``horizon_closed_loop.py``).

The period sweep splits a run's horizons into chunks (``period_sweep_jobs.py``);
``merge`` joins the chunks' results along the horizon axis into the one
.npz/.json that ``plot_period_sweep.py`` reads, after checking that they share
their start states, repeats and settings.

Usage:
    python experiments/horizon_cl_results.py merge OUT.json CHUNK.json [CHUNK.json ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import process_batch_results as pbr  # noqa: E402
import run_episode_batches as batches  # noqa: E402

#: The per-window stats: the mean Plan() seconds, and under --async the loop's counts.
STAT_KEYS_SYNC = ("plan_s",)
STAT_KEYS_ASYNC = ("plan_s", "n_plans", "stale_ticks", "held_ticks", "exhausted_ticks", "staleness_ms")


def views(run, term):
    """The window costs scored, each (S, nh, K)."""
    W = run.shape[-1]
    return {"f_avg": (run[..., :W - 1].sum(-1) + term) / W,
            "f_run": run[..., :W - 1].sum(-1) / (W - 1),
            "f_runW": run.sum(-1) / W,
            "f_term": term}


def fe(J, lam):
    """Free energy over the K repeats, per start state: (S, nh)."""
    return np.array([[pbr.mppiFreeEnergy(J[s, i], lam)[0] for i in range(J.shape[1])] for s in range(J.shape[0])])


def costStats(x) -> dict:
    """``{mean, se, lo, hi, n}`` over the finite values of ``x`` (one per start state): the mean, its
    standard error and its 95% t-interval (``run_episode_batches.meanInterval``)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    mean, lo, hi = batches.meanInterval(x)
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None
    return {"mean": mean, "se": se, "lo": lo, "hi": hi, "n": int(len(x))}


def save(outp, meta: dict, run, term, fail, st, horizons, starts, stat_keys, success=None):
    """Write OUTP's .npz (raw) and .json (``meta`` and the per-horizon summary); returns the summary, by view."""
    outp = Path(outp)
    hs = [int(h) for h in horizons]
    extra = {"async_stats": st, "async_stat_keys": np.array(stat_keys)} if len(stat_keys) > 1 else {}
    if success is not None:
        extra["success_at"] = success
    np.savez_compressed(outp.with_suffix(".npz"), run=run, term=term, fail_at=fail, plan_time=st[..., 0],
                        horizons=np.array(hs), starts=np.array(starts), **extra)
    v = views(run, term)
    summary = {k: fe(x, meta["temperature"]) for k, x in v.items()}
    out = {**meta, "horizons": hs, "window": int(run.shape[-1]), "repeats": int(run.shape[2]),
           "starts": [[int(e), int(s)] for e, s in starts],
           **{k: {str(h): x[:, i].tolist() for i, h in enumerate(hs)} for k, x in summary.items()},
           **{f"mean_{k}": {str(h): x[:, i].mean(-1).tolist() for i, h in enumerate(hs)} for k, x in v.items()},
           "cost_stats": {k: {str(h): costStats(np.nanmean(x[:, i], -1)) for i, h in enumerate(hs)}
                          for k, x in v.items()},
           "fail_frac": {str(h): float((fail[:, i] >= 0).mean()) for i, h in enumerate(hs)},
           **({"success_frac": {str(h): float((success[:, i] >= 0).mean()) for i, h in enumerate(hs)}}
              if success is not None else {}),
           "plan_times": {str(h): st[:, i, :, 0].ravel().tolist() for i, h in enumerate(hs)},
           **{f"async_{k}": {str(h): st[:, i, :, j].ravel().tolist() for i, h in enumerate(hs)}
              for j, k in enumerate(stat_keys[1:], 1)}}
    outp.write_text(json.dumps(out))
    return summary


#: Settings the chunks of one run may differ in: the horizons, and what is recomputed from them
#: (besides the fields keyed by horizon).
_PER_CHUNK = {"horizons", "chunk", "cost_stats"}


def merge(outp, parts) -> Path:
    """Join the chunk results ``parts`` (.json paths) of one run into OUTP's .npz and .json."""
    parts = [Path(p) for p in parts]
    metas = [json.loads(p.read_text()) for p in parts]
    zs = [np.load(p.with_suffix(".npz")) for p in parts]
    ref = metas[0]
    horizon_keyed = {k for k, v in ref.items() if isinstance(v, dict) and set(v) == {str(h) for h in ref["horizons"]}}
    for p, m, z in zip(parts[1:], metas[1:], zs[1:]):
        for k in set(ref) | set(m):
            if k not in _PER_CHUNK | horizon_keyed and ref.get(k) != m.get(k):
                raise ValueError(f"{p.name}: {k} is {m.get(k)!r}, {parts[0].name} has {ref.get(k)!r}")
        if not np.array_equal(z["starts"], zs[0]["starts"]):
            raise ValueError(f"{p.name}: different start states from {parts[0].name}")
    hs = np.concatenate([z["horizons"] for z in zs])
    if len(set(hs.tolist())) != len(hs):
        raise ValueError(f"a horizon is in more than one chunk: {sorted(hs.tolist())}")
    order = np.argsort(hs)
    keys = ("run", "term", "fail_at") + (("success_at",) if all("success_at" in z for z in zs) else ())
    cat = {k: np.concatenate([z[k] for z in zs], axis=1)[:, order] for k in keys}
    stat_keys = tuple(zs[0]["async_stat_keys"]) if "async_stat_keys" in zs[0] else STAT_KEYS_SYNC
    st = (np.concatenate([z["async_stats"] for z in zs], axis=1) if len(stat_keys) > 1
          else np.concatenate([z["plan_time"][..., None] for z in zs], axis=1))[:, order]
    meta = {k: v for k, v in ref.items() if k not in _PER_CHUNK | horizon_keyed | {"starts"}}
    meta["chunks"] = [p.stem for p in parts]
    save(outp, meta, cat["run"], cat["term"], cat["fail_at"], st, hs[order], zs[0]["starts"], stat_keys,
         success=cat.get("success_at"))
    return Path(outp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="join one run's chunk results")
    m.add_argument("out", type=Path, help="the run's .json (the .npz goes next to it)")
    m.add_argument("parts", type=Path, nargs="+", help="the chunks' .json files")
    opts = ap.parse_args(argv)
    out = merge(opts.out, opts.parts)
    print(f"merged {len(opts.parts)} chunks -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
