"""rescore_bo_run.py

Re-score a finished Bayesian-optimization run under a DIFFERENT objective, and
write the result as a new run directory that `run_bayes_opt.py --resume` can
seed from.

Why this works
--------------
run_bayes_opt.py's objective is a pure function of what each cell_<id>.json
already stores:

    J = -(w_success * success_rate) + w_cost * mean_normalized_goal_error

    normalized_goal_error(errs) = min(sum_k errs[k]/thr[k], err_clip) / err_clip

`errs` is the episode's `final_goal_errs` block ({"pos", "quat", "vel"} for
grasp_reorient), which every cell carries per episode, and `thr` is the task's
success_thresholds. So NEW coefficients (w_success / w_cost / err_clip) and NEW
goal-error cutoffs can both be applied offline: no episode has to be re-run.
The GP only ever saw the scalar J, so replacing every J with the one the new
objective would have produced gives a legitimate (x, y) seed set.

With several --models, J is scored per model FIRST and the per-model scores are
then folded into the one number the GP sees. The cells keep every episode's
model_label, so that fold is a free choice at rescore time too:

    MODEL_AGG = "worst"  minimax -- J is the WORST model's score (a max, since
                         J is minimized). The strict reading of "weights that
                         work for all of them": a model that fails every
                         episode cannot be hidden by one that succeeds.
    MODEL_AGG = "mean"   average performance, which lets a good model carry a
                         bad one.
    MODEL_AGG = None     keep whatever each cell recorded, i.e. the fold the
                         original run used (--model_agg on the driver).

Switching the fold is the one rescore that changes NOTHING per episode -- it
only re-reads the same per-model scores -- so it is always exact.

What is NOT recoverable
-----------------------
* `success` was decided DURING the episode, at every step, against the run's
  original thresholds, and the episode stopped at the first success
  (fin_ep_on_success=True). The cells keep only the FINAL state, so:
    - SUCCESS_MODE = "stored"      keeps the as-run success flags. Correct when
                                   you are only changing the coefficients.
    - SUCCESS_MODE = "final_state" re-checks `final_goal_errs` against the new
                                   cutoffs. TIGHTENING the cutoffs this way is
                                   sound-ish (a stricter box is a subset of the
                                   old one). LOOSENING them undercounts: an
                                   episode that passed through the wider box
                                   mid-run and drifted out again reads as a
                                   failure, because the steps in between were
                                   never stored (`trajectory` is null unless the
                                   run was launched with recording on).
* `mean_steps_to_success` is likewise an artifact of the original cutoffs; it is
  copied through unchanged and should not be compared across a rescore.
* The ORIGINAL thresholds are not written into the cells at all, so this script
  cannot tell you how far you moved them.

Consistency with the next run (important)
-----------------------------------------
The fold is a driver flag: rescoring with MODEL_AGG = "mean" and then resuming
without --model_agg mean leaves the seeded y0 on a different scale than the new
trials. The final banner prints the flags to carry over.

run_bayes_opt.py reads the cutoffs from the TASK, not from a flag:
contact_study/tasks/grasp_reorient.py -> TaskConfig.success_thresholds. The
coefficients are flags (--w_success/--w_cost/--err_clip). So if GOAL_THRESHOLDS
below differs from the task's current values, edit the task before the next run,
or its fresh trials will be scored on a different scale than the seeded y0
values and the GP will be fitting two different functions at once. Pass
--check_task to have this script import the task and diff them for you (needs
the mujoco env; it loads warp/mujoco, so it is off by default).

Usage
-----
    # a single BO run directory (holds bo_state.json), or a directory of cells
    python tests/rescore_bo_run.py results/bayes_opt_123/grasp_reorient_cube_M2_mppi

    # a whole SLURM job directory: every run inside it is rescored
    python tests/rescore_bo_run.py results/bayes_opt_123 --outdir results/bayes_opt_123_rescored

    # override the module constants from the command line
    python tests/rescore_bo_run.py <run> --w_success 1.0 --w_cost 0.5 \
        --err_clip 10 --thresholds pos=0.03 quat=0.06 vel=0.2 \
        --success_mode final_state

    # re-fold a minimax run as an average one (or the other way round)
    python tests/rescore_bo_run.py <run> --model_agg mean

    # then seed a new search from it
    python -m contact_study.drivers.run_bayes_opt --resume <out>/bo_state.json \
        --outdir results/bo_next --w_success 1.0 --w_cost 0.5 --err_clip 10 \
        <the original --task/--models/--opt_weights/... flags>

The resumed search space must match: load_state() compares dim_names and
refuses a narrowed bracket, so keep the same --opt_weights / --opt_* flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
# analysis/ is a directory of scripts, not a package, and bayes_opt_to_csv_dir
# imports its sibling by bare name -- so put that directory itself on the path.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from bayes_opt_to_csv_dir import is_run_dir, resolve_chain, _read_json  # noqa: E402
from contact_study.evaluation import json_io  # noqa: E402


# ===========================================================================
# The new objective. Edit here, or override any of it from the command line.
# ===========================================================================

# J = -(W_SUCCESS * success_rate) + W_COST * mean_norm_goal_err
W_SUCCESS = 0.0
W_COST    = 1.0
ERR_CLIP  = 100.0

# Goal-error cutoffs: the per-criterion divisor AND (in "final_state" mode) the
# success box. Keys must match the task's goal_errors() -- grasp_reorient emits
# {"pos", "quat", "vel"}. Drop a key to ignore that criterion entirely; set one
# to float("inf") to keep it out of the success test while zeroing its error
# contribution. The task's own defaults are pos=0.02, quat=0.04, vel=0.1.
GOAL_THRESHOLDS = {
    "pos":  0.02,
    "quat": 0.04,
    "vel":  0.1,
}

# "stored"      -> keep each episode's as-run success flag (coefficient-only rescore)
# "final_state" -> re-decide success from final_goal_errs vs GOAL_THRESHOLDS
SUCCESS_MODE = "stored"

# How the per-model scores fold into the one number the GP sees, for a run with
# several --models. "worst" is minimax (a MAX, because J is minimized); "mean"
# averages. None keeps whatever fold each cell recorded, i.e. the original run's.
MODEL_AGG = "worst"


class Config:
    """The rescoring knobs, resolved from the constants above plus the CLI."""

    def __init__(self, args):
        self.w_success    = args.w_success
        self.w_cost       = args.w_cost
        self.err_clip     = args.err_clip
        self.thresholds   = args.thresholds
        self.success_mode = args.success_mode
        self.model_agg    = args.model_agg

    def as_dict(self) -> dict:
        return {
            "w_success":       self.w_success,
            "w_cost":          self.w_cost,
            "err_clip":        self.err_clip,
            "goal_thresholds": dict(self.thresholds),
            "success_mode":    self.success_mode,
            "model_agg":       self.model_agg,
        }


# ---------------------------------------------------------------------------
# Scoring -- kept deliberately identical to run_bayes_opt.py
# ---------------------------------------------------------------------------

def normalized_goal_error(errs: dict | None, thresholds: dict,
                          clip: float) -> float | None:
    """Byte-for-byte the driver's normalized_goal_error(); None when unavailable.

    Duplicated rather than imported because importing run_bayes_opt pulls in
    warp + mujoco, and this script has to run on a laptop with neither.
    """
    if not errs or not thresholds:
        return None
    keys = [k for k in thresholds if k in errs and thresholds[k] > 0.0]
    if not keys:
        return None
    e = sum(float(errs[k]) / float(thresholds[k]) for k in keys)
    return min(e, clip) / clip


def episode_scores(episodes: list[dict], cfg: Config, stats: dict) -> list[dict]:
    """Per-episode (success, normalized error) under the new objective."""
    out = []
    for ep in episodes:
        errs   = ep.get("final_goal_errs")
        stored = bool(ep.get("success"))

        if cfg.success_mode == "stored":
            success = stored
        elif not errs or any(k not in errs for k in cfg.thresholds):
            # An errored episode, or a task whose goal_errors() does not cover
            # every cutoff -- nothing to re-decide on, so keep the as-run flag.
            stats["success_fallbacks"] += 1
            success = stored
        else:
            # Mirrors BaseTask.is_success: every criterion strictly inside.
            success = all(float(errs[k]) < float(cfg.thresholds[k])
                          for k in cfg.thresholds)

        if success != stored:
            stats["success_flips"] += 1

        out.append({
            "success": success,
            "err":     normalized_goal_error(errs, cfg.thresholds, cfg.err_clip),
        })
    return out


def score(scored: list[dict], cfg: Config) -> dict:
    """One model's episodes -> the driver's _score() dict."""
    n_success    = sum(s["success"] for s in scored)
    success_rate = n_success / len(scored) if scored else 0.0

    errs     = [s["err"] for s in scored if s["err"] is not None]
    mean_err = fmean(errs) if errs else None

    objective = -(cfg.w_success * success_rate)
    if mean_err is not None:
        objective += cfg.w_cost * mean_err

    return {
        "n_episodes":         len(scored),
        "n_success":          n_success,
        "success_rate":       success_rate,
        "mean_norm_goal_err": mean_err,
        "objective":          objective,
    }


def group_by_model(rec: dict) -> dict[str, list[dict]]:
    """Split a cell's flat episode list back into its per-model buckets.

    Primary key is the episode's own model_label ("M1_stiff_pyramidal" -> "M1").
    The fallback is the driver's own job ordering, specs = [(m, ep) for m in
    models for ep in range(n_episodes_per_model)], which is the order the
    episodes were written in.
    """
    models   = [m for m in (rec.get("models") or [rec.get("model")]) if m]
    episodes = rec.get("episodes") or []
    groups   = {m: [] for m in models}

    for ep in episodes:
        key = str(ep.get("model_label", "")).split("_")[0]
        if key in groups:
            groups[key].append(ep)

    if sum(len(v) for v in groups.values()) != len(episodes) or not models:
        n_per  = int(rec.get("n_episodes_per_model")
                     or (len(episodes) // max(len(models), 1)) or 1)
        groups = {m: episodes[i * n_per:(i + 1) * n_per]
                  for i, m in enumerate(models)}
    return groups


def rescore_cell(rec: dict, cfg: Config, stats: dict) -> dict:
    """A cell record -> the same record scored under the new objective."""
    groups    = group_by_model(rec)
    scored    = {m: episode_scores(eps, cfg, stats) for m, eps in groups.items()}
    per_model = {m: score(s, cfg) for m, s in scored.items() if s}

    if not per_model:
        raise ValueError(f"cell {rec.get('combo_index')} has no scorable episodes")

    agg    = cfg.model_agg or rec.get("model_agg") or "worst"
    values = [v["objective"] for v in per_model.values()]
    # 'worst' is a MAX because J is minimized: the largest per-model objective
    # is the model doing worst under these weights (same as the driver).
    objective = fmean(values) if agg == "mean" else max(values)

    flat         = [s for m in scored for s in scored[m]]
    n_success    = sum(s["success"] for s in flat)
    success_rate = n_success / len(flat) if flat else 0.0
    errs         = [v["mean_norm_goal_err"] for v in per_model.values()
                    if v["mean_norm_goal_err"] is not None]
    mean_err     = fmean(errs) if errs else None

    out = dict(rec)
    out["per_model"]          = per_model
    out["model_agg"]          = agg
    out["objective"]          = objective
    out["mean_norm_goal_err"] = mean_err
    out["n_success"]          = n_success
    out["success_rate"]       = success_rate
    # mean_steps_to_success is a product of the ORIGINAL cutoffs (success was
    # detected online, step by step) and cannot be recomputed here; it rides
    # along unchanged and is flagged as stale in the block below.
    out["rescored"] = {
        # cfg.model_agg is the OVERRIDE (None = keep the run's own); record the
        # fold that was actually applied.
        **{**cfg.as_dict(), "model_agg": agg},
        "objective_prev":          rec.get("objective"),
        "mean_norm_goal_err_prev": rec.get("mean_norm_goal_err"),
        "success_rate_prev":       rec.get("success_rate"),
        "n_success_prev":          rec.get("n_success"),
        "stale_fields":            ["mean_steps_to_success"],
    }

    if cfg.success_mode != "stored":
        # Non-destructive: the episode keeps its as-run flag, and gains the
        # re-decided one next to it.
        for m, eps in groups.items():
            for ep, s in zip(eps, scored[m]):
                ep["success_rescored"] = s["success"]
    return out


# ---------------------------------------------------------------------------
# Reading a run
# ---------------------------------------------------------------------------

def collect_cells(run_dir: Path) -> tuple[dict[int, dict], list[Path]]:
    """combo_index -> record, following the run's --resume chain when present.

    A long BO run is usually several chained SLURM jobs and only the trials a
    job actually ran leave a cell behind, so the chain is what makes the seed
    set complete. Later jobs win on a repeated index.
    """
    chain = resolve_chain(run_dir) if is_run_dir(run_dir) else [run_dir]
    cells: dict[int, dict] = {}
    for job in chain:
        for path in sorted(job.glob("cell_*.json")):
            rec = _read_json(path)
            if rec is not None and "combo_index" in rec:
                cells[int(rec["combo_index"])] = rec
    return cells, chain


def expand_runs(path: Path) -> list[Path]:
    """The run directories under `path`: itself, or each of its subdirectories."""
    if is_run_dir(path) or list(path.glob("cell_*.json")):
        return [path]
    subs = [p for p in sorted(path.iterdir())
            if p.is_dir() and (is_run_dir(p) or list(p.glob("cell_*.json")))]
    if not subs:
        raise FileNotFoundError(
            f"{path} holds neither bo_state.json, cell_*.json, nor run subdirectories")
    return subs


def _same_point(a: list[float], b: list[float], rtol: float = 1e-6) -> bool:
    """The cells round x to 7 significant digits; bo_state keeps full precision."""
    return len(a) == len(b) and all(
        abs(u - v) <= rtol * max(1.0, abs(u), abs(v)) for u, v in zip(a, b))


def derive_dim_names(rec: dict) -> list[str] | None:
    """dim_names for a run with no bo_state.json, from a cell's `axes` block.

    `axes` is emitted in the task's weight order then the planner knobs, which
    is USUALLY but not provably the search-space order, so the guess is only
    accepted when the axes values reproduce `x` exactly.
    """
    axes = rec.get("axes") or {}
    x    = rec.get("x") or []
    names = list(axes)
    if len(names) != len(x) or not _same_point([float(axes[k]) for k in names], x):
        return None
    return names


# ---------------------------------------------------------------------------
# Writing the rescored run
# ---------------------------------------------------------------------------

def build_state(src_state: dict | None, cells: dict[int, dict],
                new_cells: dict[int, dict], cfg: Config,
                out_run: Path, run_dir: Path) -> tuple[dict | None, list[str]]:
    """The new bo_state.json: the same x_iters, the rescored func_vals."""
    notes: list[str] = []
    order = sorted(new_cells)

    if src_state:
        names  = list(src_state.get("dim_names", []))
        bounds = src_state.get("dim_bounds", [])
        args   = dict(src_state.get("args", {}))
        src_x  = [[float(v) for v in x] for x in src_state.get("x_iters", [])]
        n_trace = len(src_x)
    else:
        first  = new_cells[order[0]]
        names  = derive_dim_names(first) or []
        if not names:
            notes.append(
                "no bo_state.json in the source and the cells' `axes` order does "
                "not reproduce `x`, so dim_names cannot be trusted -- cells were "
                "written but no bo_state.json. Point --src at the run directory "
                "that holds bo_state.json to get a resumable seed.")
            return None, notes
        bounds  = []
        args    = reconstruct_args(first)
        src_x   = []
        n_trace = 0
        notes.append("no bo_state.json in the source: dim_names were derived from "
                     "the cells' `axes` block and dim_bounds are unknown (resume "
                     "only checks bounds it was given).")

    # Prefer the checkpoint's full-precision coordinates over the cell's rounded
    # copy, matching each cell to its trace entry by value.
    used: set[int] = set()
    x_iters, func_vals, trials = [], [], []
    for idx in order:
        cx = [float(v) for v in new_cells[idx].get("x", [])]
        hit = next((i for i, sx in enumerate(src_x)
                    if i not in used and _same_point(sx, cx)), None)
        if hit is not None:
            used.add(hit)
        x_iters.append(src_x[hit] if hit is not None else cx)
        func_vals.append(float(new_cells[idx]["objective"]))
        trials.append(idx)

    if n_trace and len(used) < n_trace:
        notes.append(
            f"{n_trace - len(used)} of {n_trace} trials in the source trace have no "
            f"cell_*.json anywhere in the resume chain (a lost or moved job "
            f"directory); they carry no episode data and were dropped -- the seed "
            f"set is {len(x_iters)} points.")
    if names and x_iters and len(names) != len(x_iters[0]):
        notes.append(f"! dim_names has {len(names)} entries but x has "
                     f"{len(x_iters[0])}; --resume will reject this checkpoint.")

    args.update({
        "w_success": cfg.w_success,
        "w_cost":    cfg.w_cost,
        "err_clip":  cfg.err_clip,
        # The chain is collapsed into this one directory: every seed cell now
        # lives here, so a further --resume must not walk back to the old jobs.
        "resume":    None,
        "outdir":    str(out_run),
    })

    return {
        "dim_names":  names,
        "dim_bounds": bounds,
        "x_iters":    x_iters,
        "func_vals":  func_vals,
        "args":       args,
        # Not read by load_state(); here so the provenance travels with the seed.
        "rescored": {
            **cfg.as_dict(),
            "source_run":    str(run_dir),
            "trial_indices": trials,
            # Aligned one-for-one with trial_indices, so a trial's old and new
            # objective can be read off side by side.
            "func_vals_prev": [None if cells[i].get("objective") is None
                               else float(cells[i]["objective"]) for i in order],
            "notes": notes,
        },
    }, notes


def reconstruct_args(rec: dict) -> dict:
    """A best-effort args block for a source directory with no bo_state.json."""
    mppi = rec.get("mppi") or {}
    return {
        "task":        rec.get("task"),
        "geometry":    rec.get("geometry"),
        "models":      rec.get("models") or [rec.get("model")],
        "model":       rec.get("model"),
        "model_agg":   rec.get("model_agg"),
        "planner":     rec.get("planner"),
        "n_samples":   mppi.get("n_samples"),
        "time_horizon": mppi.get("time_horizon"),
        "step_time":   mppi.get("step_time"),
        "resample_interval": mppi.get("resample_interval"),
        "delta":       mppi.get("delta"),
        "n_episodes":  rec.get("n_episodes_per_model"),
        "seed":        rec.get("seed"),
        "eval_sim":    rec.get("eval_sim"),
        "settle":      rec.get("settle"),
        "reconstructed_from_cells": True,
    }


def recompact(rec: dict) -> dict:
    """Re-mark trajectory arrays so the rewritten cell keeps its one-line layout.

    json_io.compact() is a write-time marker; reading a cell back loses it, and
    a plain indent=2 dump would explode a recorded (2000, 23) qpos block into
    tens of MB of whitespace.
    """
    for ep in rec.get("episodes") or []:
        traj = ep.get("trajectory")
        if isinstance(traj, dict):
            ep["trajectory"] = {k: (json_io.compact(v) if isinstance(v, list) else v)
                                for k, v in traj.items()}
    return rec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_thresholds(items: list[str] | None) -> dict[str, float]:
    if not items:
        return dict(GOAL_THRESHOLDS)
    out = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--thresholds wants key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v)
    return out


def check_task_thresholds(task: str, geometry: str, thresholds: dict) -> None:
    """Diff the cutoffs against the task's own (imports warp/mujoco -- opt-in)."""
    try:
        from contact_study.drivers.run_eval_episode import load_rollout_task
        current = dict(load_rollout_task(task, geometry).config.success_thresholds)
    except Exception as exc:                       # wrong env, missing GPU, ...
        print(f"  ! --check_task could not load {task!r}/{geometry!r}: {exc}")
        return
    if current == thresholds:
        print(f"  task success_thresholds already match: {current}")
    else:
        print(f"  ! task success_thresholds are {current}, this rescore used "
              f"{thresholds}.\n"
              f"    Edit TaskConfig.success_thresholds for {task!r} before the "
              f"next run, or its new trials will be scored on a different scale "
              f"than the seeded y0.")


def rescore_run(run_dir: Path, out_run: Path, cfg: Config, dry_run: bool) -> dict:
    cells, chain = collect_cells(run_dir)
    if not cells:
        raise FileNotFoundError(f"{run_dir} holds no cell_*.json (chain: "
                                f"{[p.name for p in chain]})")

    stats = {"success_flips": 0, "success_fallbacks": 0}
    new_cells = {i: rescore_cell(rec, cfg, stats) for i, rec in cells.items()}

    src_state = _read_json(run_dir / "bo_state.json")
    state, notes = build_state(src_state, cells, new_cells, cfg, out_run, run_dir)

    if not dry_run:
        out_run.mkdir(parents=True, exist_ok=True)
        for idx, rec in new_cells.items():
            json_io.dump(recompact(rec), out_run / f"cell_{idx:05d}.json")
        if state is not None:
            with open(out_run / "bo_state.json", "w") as f:
                json.dump(state, f, indent=2)

    order   = sorted(new_cells)
    best    = min(order, key=lambda i: new_cells[i]["objective"])
    prev_best = min((i for i in order if cells[i].get("objective") is not None),
                    key=lambda i: cells[i]["objective"], default=None)

    print(f"\n{run_dir}  ->  {out_run}")
    print(f"  chain:   {' -> '.join(p.name for p in chain)}")
    print(f"  trials:  {len(new_cells)} cells rescored"
          + (f"  (source trace: {len(src_state.get('x_iters', []))})" if src_state else ""))
    if cfg.success_mode != "stored":
        print(f"  success: {stats['success_flips']} episode flags flipped under the "
              f"new cutoffs; {stats['success_fallbacks']} episodes kept their as-run "
              f"flag (no usable final_goal_errs)")
    if prev_best is not None:
        print(f"  best was trial {prev_best:>4d}  J={cells[prev_best]['objective']:+.5f}"
              f"   (its new J={new_cells[prev_best]['objective']:+.5f})")
    print(f"  best now trial {best:>4d}  J={new_cells[best]['objective']:+.5f}"
          f"   success_rate={new_cells[best]['success_rate']:.3f}"
          f"   norm_goal_err={new_cells[best]['mean_norm_goal_err']}")

    print(f"  top {min(10, len(order))} under the new objective:")
    print(f"    {'trial':>6}  {'J_new':>10}  {'J_old':>10}  {'succ':>6}  {'err':>8}")
    for i in sorted(order, key=lambda i: new_cells[i]["objective"])[:10]:
        err = new_cells[i]["mean_norm_goal_err"]
        old = cells[i].get("objective")
        print(f"    {i:>6d}  {new_cells[i]['objective']:>+10.5f}  "
              f"{('n/a' if old is None else f'{old:+.5f}'):>10}  "
              f"{new_cells[i]['success_rate']:>6.2f}  "
              f"{('n/a' if err is None else f'{err:.5f}'):>8}")
    for note in notes:
        print(f"  ! {note}")
    if dry_run:
        print("  (--dry_run: nothing written)")

    return {"run": str(run_dir), "out": str(out_run), "n_cells": len(new_cells),
            "resumable": state is not None, "notes": notes,
            "best_trial": best, "best_objective": new_cells[best]["objective"],
            **stats}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path,
                   help="A BO run directory (holds bo_state.json and/or cell_*.json), "
                        "or a job directory whose subdirectories are runs.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Destination (default: <src>_rescored). One subdirectory "
                        "per run when src is a job directory.")
    p.add_argument("--w_success", type=float, default=W_SUCCESS)
    p.add_argument("--w_cost",    type=float, default=W_COST)
    p.add_argument("--err_clip",  type=float, default=ERR_CLIP)
    p.add_argument("--thresholds", nargs="+", metavar="KEY=VALUE", default=None,
                   help="Goal-error cutoffs, e.g. pos=0.03 quat=0.06 vel=0.2 "
                        f"(default: {GOAL_THRESHOLDS}).")
    p.add_argument("--success_mode", choices=["stored", "final_state"],
                   default=SUCCESS_MODE,
                   help="'stored' keeps the as-run success flags (right when only "
                        "the coefficients changed); 'final_state' re-decides them "
                        "from final_goal_errs, which undercounts if you LOOSENED "
                        "the cutoffs -- see the module docstring.")
    p.add_argument("--model_agg", choices=["mean", "worst"], default=MODEL_AGG,
                   help="Fold the per-model objectives differently than the run "
                        "did: 'worst' is minimax (the worst model's score, a max "
                        "since J is minimized), 'mean' averages them. Omit to "
                        "keep each cell's own fold. Exact either way -- nothing "
                        "per-episode is re-decided.")
    p.add_argument("--check_task", action="store_true",
                   help="Import the task and diff its success_thresholds against "
                        "the cutoffs used here (needs the mujoco env).")
    p.add_argument("--dry_run", action="store_true",
                   help="Report what would change without writing anything.")
    return p


def main():
    args = build_parser().parse_args()
    args.thresholds = parse_thresholds(args.thresholds)
    cfg = Config(args)

    src  = args.src.resolve()
    runs = expand_runs(src)
    out  = (args.outdir or src.parent / f"{src.name}_rescored").resolve()
    # One run in, one directory out; a job directory keeps its per-run layout.
    single = len(runs) == 1 and runs[0] == src

    print(f"{'='*70}")
    print(f"  rescoring {len(runs)} run{'s' if len(runs) != 1 else ''} under")
    print(f"  J = -({cfg.w_success:g} * success_rate) + {cfg.w_cost:g} * "
          f"norm_goal_err   (err_clip={cfg.err_clip:g})")
    print(f"  goal cutoffs: " + "  ".join(f"{k}={v:g}" for k, v in cfg.thresholds.items())
          + f"   success_mode={cfg.success_mode}")
    print(f"  multi-model fold: "
          + (f"{cfg.model_agg} (overriding each run's own)" if cfg.model_agg
             else "as each run recorded it (--model_agg to re-fold)"))
    print(f"{'='*70}")

    reports = [rescore_run(r, out if single else out / r.name, cfg, args.dry_run)
               for r in runs]

    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "rescore_report.json", "w") as f:
            json.dump({"source": str(src), "objective": cfg.as_dict(),
                       "runs": reports}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  wrote {out}" if not args.dry_run else "  nothing written (--dry_run)")
    print(f"  The next run must use the SAME objective, or the seeded y0 values "
          f"and its fresh trials measure different things:")
    agg_flag = f" --model_agg {cfg.model_agg}" if cfg.model_agg else ""
    print(f"    --w_success {cfg.w_success:g} --w_cost {cfg.w_cost:g} "
          f"--err_clip {cfg.err_clip:g}{agg_flag}")
    print(f"  and the cutoffs come from the TASK, not a flag -- set")
    print(f"    success_thresholds = "
          + "{" + ", ".join(f'"{k}": {v:g}' for k, v in cfg.thresholds.items()) + "}")
    print(f"  in the task's TaskConfig (contact_study/tasks/<task>.py) first.")
    seed = out / "bo_state.json" if single else out / "<run>" / "bo_state.json"
    print(f"\n  Seed a new search with:")
    print(f"    python -m contact_study.drivers.run_bayes_opt \\")
    print(f"        --resume {seed} --outdir results/bo_next \\")
    print(f"        --w_success {cfg.w_success:g} --w_cost {cfg.w_cost:g} "
          f"--err_clip {cfg.err_clip:g}{agg_flag} \\")
    print(f"        <the original --task/--geometry/--models/--opt_* flags>")
    print(f"{'='*70}")

    if args.check_task:
        # Any cell names the task and geometry the run was launched with.
        cells = sorted(runs[0].glob("cell_*.json"))
        rec   = (_read_json(cells[0]) or {}) if cells else {}
        if rec.get("task"):
            check_task_thresholds(rec["task"], rec.get("geometry"), cfg.thresholds)
        else:
            print("  ! --check_task: no cell in the first run names a task")


if __name__ == "__main__":
    main()
