#!/usr/bin/env python3
"""Write the final episode-batch CSV from a Bayesian optimization of the weights and temperatures.

Takes one cell folder of ``run_bayes_opt.py`` (a search over the cost weights
and, per model, the temperature, with the noise fixed) and writes a
``run_episode_batches.py`` CSV with one row per rollout model x temperature
scale x hand accuracy: 5 models x 3 scales x 3 accuracies = 45 rows by default.
Every row carries the best trial's cost weights and that model's temperature
times the scale, and everything else the search ran with (task, eval sim,
control step, horizon, sample count, warm start, goal difficulty, ...), with
``save_steps=true`` so ``process_batch_results.py`` can analyse it::

    python experiments/make_final_batch_csv.py results/bayes_opt_bo_123/cell_0000_cube_all \\
        --out experiments/final_batches.csv
    python experiments/make_final_batch_csv.py <bo cell> --out final.csv --temp-scales 0.5 1 2 \\
        --hand-accs high med low --n-episodes 30 --steps 4000 --set uncertainty=true
    python experiments/make_final_batch_csv.py <bo cell> --out final.csv \\
        --optimal-bo results/bayes_opt_opt_456/cell_0000_opt_temp   # also print OPTIMAL_ARGS

The trial is the cell's best (lowest J) unless ``--trial`` picks another. The
seed defaults to the search's seed plus one: the pooled drivers draw episode
k's goals and noise from ``(seed, k)``, so the search's own seed would replay
the very episodes the weights were tuned on.

``--optimal-bo`` names the cell folder of the search over the optimal
(convergence) planner's temperature. The script then prints the
``OPTIMAL_ARGS`` that ``hpc/submit_process_batch_results.sh`` passes on to the
analysis: that search's planner settings and its best temperature. It warns
when that search pinned other cost weights or another eval sim than the
batch rows have, since the optimal planner plans with each row's own.

The CSV is checked against ``run_episodes.py`` before the script exits.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import process_batch_results as pbr  # noqa: E402
import run_bayes_opt as bo  # noqa: E402
import run_episode_batches as batches  # noqa: E402
from ContactModelStudy.Drivers import run_episodes as drv  # noqa: E402
from ContactModelStudy.Tasks.LeapReorient import COST_WEIGHT_KEYS  # noqa: E402

#: Driver options of the search's row that each batch row sets itself.
SET_PER_ROW = ("rollout_model", "hand_acc", "temperature", "noise_sigma", "n_episodes", "steps", "seed",
               "save_steps", "debug", "results", "video")
#: Leading columns, in this order; the rest follow as the search's row had them.
LEADING = ("label", "task", "rollout_model", "eval_sim", "hand_acc", "obj_acc", "n_episodes", "steps", "seed",
           "temperature", "noise_sigma", "save_steps")


def loadSearch(cell_dir: Path, trial: int | None) -> tuple[dict, dict, int]:
    """``(the search's CSV row, the chosen trial's record, finished trials)`` of a BO cell folder."""
    state = pbr._readJson(cell_dir / "bo_state.json")
    if state is None:
        raise SystemExit(f"{cell_dir} has no bo_state.json; is it a run_bayes_opt.py cell folder "
                         f"(results/bayes_opt_<csv>_<id>/cell_<row>_<label>)?")
    trials = bo._readTrials(cell_dir)
    if not trials:
        raise SystemExit(f"{cell_dir} has no finished trials")
    if trial is None:
        return state["row"], min(trials, key=lambda t: t["objective"]), len(trials)
    chosen = [t for t in trials if t["trial"] == trial]
    if not chosen:
        raise SystemExit(f"{cell_dir} has no finished trial {trial}")
    return state["row"], chosen[0], len(trials)


def _default(option: str) -> float:
    return drv.build_parser().get_default(option)


def modelKnob(row: dict, trial: dict, model: str, knob: str) -> float:
    """A model's ``temperature`` / ``noise_sigma`` in the trial: searched, else pinned by the row, else
    the driver default."""
    v = trial["settings"].get(model, {}).get("planner_params", {}).get(knob)
    return float(v if v is not None else row.get(knob, _default(knob)))


def trialWeights(row: dict, trial: dict) -> dict[str, float]:
    """The cost weights the trial ran with that differ from the object's own: pinned, then searched."""
    return {**{k: float(row[k]) for k in COST_WEIGHT_KEYS if k in row},
            **{k: float(v) for k, v in trial.get("cost_weights", {}).items()}}


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def buildRows(row: dict, trial: dict, models: list[str], scales: list[float], hand_accs: list[str],
              n_episodes: int, steps: int, seed: int, extra: dict[str, str]) -> list[dict[str, str]]:
    """One batch row per model x scale x hand accuracy, in that nesting."""
    options = batches._driverOptions()
    carried = {k: v for k, v in row.items() if k in options and k not in SET_PER_ROW}
    weights = {k: _fmt(v) for k, v in trialWeights(row, trial).items()}
    prefix = row.get("task", drv.build_parser().get_default("task")).removesuffix("_reorient")
    out = []
    for m in models:
        temp, sigma = modelKnob(row, trial, m, "temperature"), modelKnob(row, trial, m, "noise_sigma")
        for scale in scales:
            for acc in hand_accs:
                r = {"label": f"{prefix}_{m}_{acc}_Tx{scale:g}", **carried, "rollout_model": m, "hand_acc": acc,
                     "n_episodes": str(n_episodes), "steps": str(steps), "seed": str(seed),
                     "temperature": _fmt(temp * scale), "noise_sigma": _fmt(sigma), "save_steps": "true",
                     **weights, **extra}
                out.append(r)
    return out


def writeCsv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = list(dict.fromkeys([c for c in LEADING if c in rows[0]] + [c for r in rows for c in r]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def checkCsv(path: Path) -> list[str]:
    """Every problem ``run_episode_batches.py --check`` would report, as ``"<cell>: <error>"``."""
    try:
        batches.checkColumns(path)
    except ValueError as e:
        return [str(e)]
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, row in enumerate(batches.readRows(path)):
            name = batches.cellName(i, row)
            err = batches.checkRow(batches.rowToArgv(row, Path(tmp), name))
            if err:
                bad.append(f"{name}: {err}")
    return bad


def optimalArgs(opt_dir: Path, eval_sim: str, batch_weights: dict[str, float], trial: int | None) -> str:
    """The ``OPTIMAL_ARGS`` of the optimal planner's search: its planner flags and best temperature.

    Prints a warning for each way it differs from what the batch rows plan with.
    """
    row, best, n = loadSearch(opt_dir, trial)
    models = bo._tokens(row.get("rollout_models", "")) or [_default("rollout_model")]
    if len(models) != 1:
        raise SystemExit(f"--optimal-bo {opt_dir.name} searched {models}; the optimal planner is one model")
    flags = {k: v for k, v in row.items() if k in pbr.OPTIMAL_FLAGS and k not in ("temperature", "debug")}
    flags["temperature"] = _fmt(modelKnob(row, best, models[0], "temperature"))
    argv = batches.rowToArgv(flags, Path("."), "x")
    text = " ".join(argv[:argv.index("--results")])
    pbr.parseOptimalArgs(text)                            # as the analysis will
    print(f"\noptimal planner: trial {best['trial']} of {n} in {opt_dir}, J={best['objective']:+.4f}")
    model = models[0]
    if model != eval_sim or not batches._parseBool(row.get("plan_on_eval", "false"), "plan_on_eval"):
        print(f"  WARNING: it planned with {model} on the {'eval' if row.get('plan_on_eval') else 'rollout'} "
              f"scene; the analysis's optimal planner plans with the eval sim ({eval_sim}) on the eval scene")
    if row.get("eval_sim", _default("eval_sim")) != eval_sim:
        print(f"  WARNING: it was scored in {row.get('eval_sim')}, the batch in {eval_sim}")
    w_opt = trialWeights(row, best)
    differ = sorted(k for k in set(w_opt) | set(batch_weights)
                    if not math.isclose(w_opt.get(k, math.nan), batch_weights.get(k, math.nan), rel_tol=1e-4))
    if differ:
        print(f"  WARNING: its cost weights differ from the batch's in {differ}; the analysis's optimal "
              f"planner uses each batch row's weights, not these")
    return text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bo_cell", type=Path, help="the run_bayes_opt.py cell folder of the weights/temperature search")
    p.add_argument("--out", type=Path, required=True, help="the batch CSV to write")
    p.add_argument("--trial", type=int, default=None, help="use this trial rather than the best")
    p.add_argument("--models", nargs="+", default=None, help="rollout models (default: the search's)")
    p.add_argument("--temp-scales", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                   help="each model's temperature times these (default 0.5 1 2)")
    p.add_argument("--hand-accs", nargs="+", default=["high", "med", "low"], choices=["high", "med", "low"],
                   help="rollout hand accuracies (default high med low)")
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--steps", type=int, default=4000, help="control steps per episode at most")
    p.add_argument("--seed", type=int, default=None, help="default: the search's seed + 1")
    p.add_argument("--set", action="append", default=[], metavar="COLUMN=VALUE",
                   help="set a column on every row, e.g. --set uncertainty=true; repeat for more")
    p.add_argument("--optimal-bo", type=Path, default=None,
                   help="the cell folder of the optimal planner's temperature search: print its OPTIMAL_ARGS")
    p.add_argument("--optimal-trial", type=int, default=None, help="with --optimal-bo: this trial, not the best")
    args = p.parse_args(argv)
    if any(s <= 0 for s in args.temp_scales):
        p.error("--temp-scales must be positive")
    extra = {}
    for item in args.set:
        k, sep, v = item.partition("=")
        if not sep or not k:
            p.error(f"--set wants COLUMN=VALUE, got {item!r}")
        if k in ("rollout_model", "hand_acc", "temperature", "label"):
            p.error(f"--set {k}: this script sets {k} per row")
        extra[k] = v

    row, trial, n = loadSearch(args.bo_cell.resolve(), args.trial)
    searched = bo._tokens(row.get("rollout_models", "")) or [_default("rollout_model")]
    models = args.models or searched
    missing = [m for m in models if m not in searched]
    if missing:
        p.error(f"{missing} were not in the search ({' '.join(searched)}), so they have no tuned temperature")
    seed = args.seed if args.seed is not None else int(row.get("seed", _default("seed"))) + 1
    rows = buildRows(row, trial, models, args.temp_scales, args.hand_accs, args.n_episodes, args.steps,
                     seed, extra)
    writeCsv(args.out, rows)

    print(f"search: trial {trial['trial']} ({trial['source']}) of {n} in {args.bo_cell}, "
          f"J={trial['objective']:+.4f}, success {trial['success_rate']:.0%}")
    print("  weights  " + "  ".join(f"{k[2:]}={_fmt(v)}" for k, v in trialWeights(row, trial).items()))
    for m in models:
        print(f"  {m}  temperature {_fmt(modelKnob(row, trial, m, 'temperature'))}  "
              f"noise {_fmt(modelKnob(row, trial, m, 'noise_sigma'))}")
    print(f"{len(rows)} rows ({len(models)} models x {len(args.temp_scales)} temperatures x "
          f"{len(args.hand_accs)} hand accuracies), {args.n_episodes} episodes each, seed {seed} -> {args.out}")
    bad = checkCsv(args.out)
    for b in bad:
        print(f"  BAD {b}")
    if bad:
        return 1

    if args.optimal_bo is not None:
        text = optimalArgs(args.optimal_bo.resolve(), row.get("eval_sim", _default("eval_sim")),
                           trialWeights(row, trial), args.optimal_trial)
        print(f"  OPTIMAL_ARGS=\"{text}\"")
        print(f"\nonce the batch has run:\n  OPTIMAL_ARGS=\"{text}\" \\\n"
              f"      experiments/hpc/submit_process_batch_results.sh <batch results dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
