"""plot_kl_divergence_dir.py

Directory plotter for the repository's online closed-loop
planner-approximation-quality sweep — reads a whole
directory of per-cell JSONs written by experiments/hpc/kl_divergence_eval.slurm
(results/kl_divergence_eval_<id>/<model>_n<samples>_i<iters>[_null].json) and
turns them into the figure that sweep exists to produce: success rate vs. the
KL divergence between the higher-compute planner's weighted first-action
distribution and the degraded planner's distribution, each Gaussian-approximated.

Recorded-log replay is a separate workflow documented in
analysis/README_offline_recorded_kl.md; those inputs and their different
state-sampling protocol are intentionally not accepted here.

Unlike the other sweeps in analysis/, this one is NOT a clustered bar chart of
M1-M4: every cell uses the same contact model, and the swept axis is optimizer
compute (n_samples x n_iterations). So the headline panel is a scatter, one
point per cell, and the model appears only in the title.

Panels
------
Top    — scatter of success rate (%) vs. KL, one point per non-null cell.
         Marker colour = n_samples, marker shape = n_iterations; x-error is the
         between-episode SE of the episode-balanced KL mean. The default
         y-error is a Wilson 95% binomial interval, which remains honest at
         0% and 100% success; --success_error se reproduces the old ±1 SE.
Bottom — the independent-run null diagnostic: real KL next to that config's
         matched-budget null KL (same planner settings, different noise seed).
         Independent closed-loop runs can visit different states, so this is
         NOT a matched-state noise floor or a formal significance test. Hollow
         markers indicate that the real value does not exceed the null value
         under the optional descriptive screening rule.

Merge rule: only files with matching recorded scientific settings are pooled.
Objects, geometries, tasks, reference budgets, simulator settings, goals and
other recorded configuration remain separate. Compatible independent seeds are
pooled; duplicate episode seed identities are rejected. Old files without
geometry metadata remain explicitly unknown, never inferred as high_high.
Success rate is episode-weighted and recomputed from the pooled success count
and total episode count. By default, its whisker is a Wilson 95% interval. KL is
also episode-balanced: first summarize the measured steps inside each episode,
then give every episode equal weight. This avoids giving a long timeout episode
more influence than a short successful one, and uses episodes—not correlated
adjacent control steps—as the uncertainty units. Pass --weighting step and
--success_error se to reproduce the legacy pooled-step / binomial-SE display.

Usage:
    python analysis/plot_kl_divergence_dir.py                          # latest kl_divergence_eval_* dir
    python analysis/plot_kl_divergence_dir.py results/kl_divergence_eval_12345
    python analysis/plot_kl_divergence_dir.py --stat median --direction reverse
    python analysis/plot_kl_divergence_dir.py --weighting step  # legacy
    python analysis/plot_kl_divergence_dir.py --reference_filter converged
    python analysis/plot_kl_divergence_dir.py --drop_below_null
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Marker shape carries n_iterations, colour carries n_samples (assigned from a
# sequential map once the sweep's sample counts are known).
ITER_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}
_FALLBACK_MARKER = "P"

NULL_COLOR = "#BBBBBB"

_LABEL_RE = re.compile(r"^(?:(.+)_)?(M\d+)_n(\d+)_i(\d+)(_null)?$")

# Everything else in config participates in the compatibility key. This
# fail-closed policy keeps future scientific knobs separate automatically.
_REPLICATE_OR_SWEEP_KEYS = {
    "n_samples", "n_iterations", "null_control", "seed", "root_seed",
    "n_episodes", "ref_n_samples", "ref_n_iterations",
    "ref_iteration_mode", "ref_convergence_tol", "ref_max_iterations",
    "ref_temperature",
    "comparison_ref_n_samples", "comparison_ref_n_iterations",
    "geometry", "object", "record_kl_moments",
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:12]


def _finite_values(values) -> list[float]:
    """JSON null represents a sanitized invalid measurement, not a zero."""
    result = []
    for value in values or []:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _validate_converged_payload(path: Path, data: dict) -> None:
    """Fail closed before applying the convergence-only selection.

    The filter has a scientific meaning only for convergence-terminated
    reference solves whose recorded flags can be checked against the stopping
    rule. It must not silently treat missing, non-boolean, or internally
    inconsistent metadata as an empty selection.
    """
    cfg = data.get("config", {})
    if cfg.get("ref_iteration_mode") != "convergence":
        raise ValueError(
            f"{path.name}: converged-only plotting requires "
            "config.ref_iteration_mode='convergence'"
        )
    tol = cfg.get("ref_convergence_tol")
    cap = cfg.get("ref_max_iterations")
    if (isinstance(tol, bool) or not isinstance(tol, (int, float))
            or not math.isfinite(float(tol)) or float(tol) <= 0.0):
        raise ValueError(
            f"{path.name}: converged-only plotting requires a finite positive "
            "config.ref_convergence_tol"
        )
    if (isinstance(cap, bool) or not isinstance(cap, int) or cap < 2):
        raise ValueError(
            f"{path.name}: converged-only plotting requires an integer "
            "config.ref_max_iterations >= 2"
        )
    tol = float(tol)

    per_step = data.get("per_step")
    if not isinstance(per_step, list) or not per_step:
        raise ValueError(
            f"{path.name}: converged-only plotting requires per-step "
            "reference_converged metadata"
        )

    selected = {"forward": [], "reverse": []}
    for episode_index, record in enumerate(per_step):
        flags = record.get("reference_converged")
        iterations = record.get("reference_n_iterations")
        residuals = record.get("reference_convergence_residual")
        if flags is None:
            raise ValueError(
                f"{path.name}: converged-only plotting requires per-step "
                "reference_converged metadata"
            )
        if not isinstance(flags, list) or any(type(flag) is not bool for flag in flags):
            raise ValueError(
                f"{path.name}: per_step[{episode_index}].reference_converged "
                "must contain only booleans"
            )
        aligned = {
            "reference_n_iterations": iterations,
            "reference_convergence_residual": residuals,
            "kl_forward": record.get("kl_forward"),
            "kl_reverse": record.get("kl_reverse"),
        }
        for field, values in aligned.items():
            if not isinstance(values, list) or len(values) != len(flags):
                raise ValueError(
                    f"{path.name}: per_step[{episode_index}].{field} and "
                    "reference_converged lengths disagree"
                )

        for index, (flag, count, residual) in enumerate(
            zip(flags, iterations, residuals)
        ):
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    f"{path.name}: per_step[{episode_index}] reference iteration "
                    f"count {index} must be an integer"
                )
            if not 2 <= count <= cap:
                raise ValueError(
                    f"{path.name}: per_step[{episode_index}] reference iteration "
                    f"count {count} is outside [2, {cap}]"
                )
            if (isinstance(residual, bool)
                    or not isinstance(residual, (int, float))
                    or not math.isfinite(float(residual))
                    or float(residual) < 0.0):
                raise ValueError(
                    f"{path.name}: per_step[{episode_index}] convergence residual "
                    f"{index} must be finite and nonnegative"
                )
            residual = float(residual)
            if flag != (residual < tol):
                raise ValueError(
                    f"{path.name}: per_step[{episode_index}] convergence flag "
                    f"{index} disagrees with residual {residual:g} and "
                    f"tolerance {tol:g}"
                )
            if not flag and count != cap:
                raise ValueError(
                    f"{path.name}: per_step[{episode_index}] non-converged solve "
                    f"{index} stopped at {count}, before cap {cap}"
                )

        for direction, field in (("forward", "kl_forward"),
                                 ("reverse", "kl_reverse")):
            selected[direction].extend(_finite_values([
                value for value, flag in zip(record[field], flags) if flag
            ]))

    summaries = data.get("kl_converged_only")
    if not isinstance(summaries, dict):
        raise ValueError(
            f"{path.name}: converged-only plotting requires top-level "
            "kl_converged_only summaries"
        )
    for direction, values in selected.items():
        summary = summaries.get(direction)
        if not isinstance(summary, dict):
            raise ValueError(
                f"{path.name}: kl_converged_only.{direction} is missing"
            )
        expected = {
            "n": len(values),
            "mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p25": float(np.percentile(values, 25)) if values else None,
            "p75": float(np.percentile(values, 75)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
        }
        summary_n = summary.get("n")
        if (isinstance(summary_n, bool) or not isinstance(summary_n, int)
                or summary_n != expected["n"]):
            raise ValueError(
                f"{path.name}: kl_converged_only.{direction}.n disagrees "
                "with per-step convergence flags"
            )
        for field in ("mean", "sd", "median", "p25", "p75", "p95"):
            if field not in summary:
                if field in {"mean", "sd"}:
                    raise ValueError(
                        f"{path.name}: kl_converged_only.{direction}.{field} "
                        "is missing"
                    )
                continue
            actual, target = summary[field], expected[field]
            if target is None:
                consistent = actual is None
            else:
                consistent = (
                    not isinstance(actual, bool)
                    and isinstance(actual, (int, float))
                    and math.isfinite(float(actual))
                    and math.isclose(float(actual), target,
                                     rel_tol=1e-10, abs_tol=1e-12)
                )
            if not consistent:
                raise ValueError(
                    f"{path.name}: kl_converged_only.{direction}.{field} "
                    "disagrees with per-step convergence flags"
                )


def _describe_record(path: Path, data: dict) -> dict | None:
    label = data.get("label")
    if label is None:
        return None
    match = _LABEL_RE.fullmatch(str(label))
    if match is None:
        print(f"  ! skipping {path.name}: unsupported KL cell label {label!r}")
        return None

    cfg = data.get("config", {})
    agg = data.get("aggregate", {})
    prefix, model, ns, ni, null_suffix = match.groups()
    ns, ni, is_null = int(ns), int(ni), null_suffix is not None
    for field, expected in (("n_samples", ns), ("n_iterations", ni),
                            ("null_control", is_null)):
        if field in cfg and cfg[field] != expected:
            raise ValueError(f"{path.name}: label disagrees with config.{field}")
    if data.get("model", model) != model:
        raise ValueError(f"{path.name}: label disagrees with recorded model")
    top_geometry, cfg_geometry = data.get("geometry"), cfg.get("geometry")
    if top_geometry and cfg_geometry and top_geometry != cfg_geometry:
        raise ValueError(f"{path.name}: top-level/config geometry disagree")
    geometry = top_geometry or cfg_geometry or "unknown (legacy)"
    if prefix and geometry != "unknown (legacy)" and prefix != geometry:
        raise ValueError(f"{path.name}: label disagrees with recorded geometry")
    object_name = data.get("object") or cfg.get("object") or "unknown"
    if data.get("object") and cfg.get("object") and data["object"] != cfg["object"]:
        raise ValueError(f"{path.name}: top-level/config object disagree")
    task = data.get("task", agg.get("task_name", "unknown"))
    if data.get("task") and agg.get("task_name") and data["task"] != agg["task_name"]:
        raise ValueError(f"{path.name}: task metadata disagree")
    schema = int(data.get("schema_version", 1))
    base = {
        "schema_version": schema, "task": task, "model": model,
        "geometry": geometry, "object": object_name,
        "config": {k: v for k, v in cfg.items() if k not in _REPLICATE_OR_SWEEP_KEYS},
    }
    # New results retain the requested large-reference budget even in null
    # runs. In older null files that budget is unknown: infer it only when the
    # directory has exactly one otherwise-compatible real reference budget.
    comparison = (cfg.get("comparison_ref_n_samples"),
                  cfg.get("comparison_ref_n_iterations"))
    if schema >= 2 and None in comparison:
        raise ValueError(f"{path.name}: schema {schema} requires comparison_ref_* metadata")
    if comparison == (None, None) and not is_null:
        comparison = (cfg.get("ref_n_samples"), cfg.get("ref_n_iterations"))
    return {
        "path": path, "data": data, "label": label, "model": model,
        "n_samples": ns, "n_iterations": ni, "null": is_null,
        "task_name": task, "geometry": geometry, "object": object_name,
        "schema_version": schema, "base": base, "comparison": comparison,
    }


def _episode_identity_aliases(data: dict) -> list[set[tuple]]:
    """Seed aliases for each episode; repeated seeds are not new evidence.

    Environment seeds alone identify a shared randomized task within an
    otherwise-compatible cell. Deliberately repeated same-seed runs need a
    separate repeated-measures analysis, not ordinary episode pooling.
    """
    cfg = data.get("config", {})
    root_seed = cfg.get("seed", cfg.get("root_seed"))
    episodes = data.get("episodes", []) or []
    n_ep = int(data.get("aggregate", {}).get("n_episodes", len(episodes)) or 0)
    if episodes and len(episodes) != n_ep:
        raise ValueError("episode list length disagrees with aggregate.n_episodes")
    identities = []
    for index in range(n_ep):
        ep = episodes[index] if episodes else {}
        aliases = set()
        if ep.get("environment_seed") is not None:
            aliases.add(("environment_seed", int(ep["environment_seed"])))
        if root_seed is not None:
            aliases.add(("root_seed_episode", int(root_seed),
                         int(ep.get("episode_index", index))))
        identities.append(aliases)
    return identities


def _latest_dir() -> Path:
    dirs = sorted(d for d in RESULTS_DIR.glob("kl_divergence_eval_*") if d.is_dir())
    if not dirs:
        raise FileNotFoundError(
            f"No kl_divergence_eval_* directories found in {RESULTS_DIR}. "
            f"Pass a directory explicitly."
        )
    return dirs[-1]


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Loading + pooling
# ---------------------------------------------------------------------------
def _pool_stats(acc: dict) -> tuple[float | None, float | None, int]:
    """Collapse a {n, sum, sumsq} accumulator into (mean, sd, n).

    sumsq is accumulated as n*(sd^2 + mean^2) per file, so the pooled SD is the
    spread of every per-step sample together rather than a mean of per-file SDs.
    """
    n = acc["n"]
    if n <= 0:
        return None, None, 0
    mean = acc["sum"] / n
    var  = max(acc["sumsq"] / n - mean * mean, 0.0)
    return mean, math.sqrt(var), n


def merge_dir(directory: Path, reference_filter: str = "all") -> tuple[dict[str, dict], list[Path]]:
    """Pool compatible per-cell JSONs without mixing scientific settings.

    Labels remain dictionary keys when unique; conflicting labels are suffixed
    with a stable family ID. ``family_id`` rather than the display label is the
    authoritative grouping and null-pairing key. Missing legacy metadata never
    matches new-schema metadata. Duplicate seed identities raise ValueError.
    """
    if reference_filter not in {"all", "converged"}:
        raise ValueError("reference_filter must be 'all' or 'converged'")
    files = sorted(p for p in directory.glob("*.json") if p.name != "meta.json")
    if not files:
        raise FileNotFoundError(f"No per-cell *.json files found in {directory}")

    records = []
    for path in files:
        record = _describe_record(path, load(path))
        if record is not None:
            records.append(record)

    real_budgets: dict[str, set[tuple]] = {}
    for r in records:
        if not r["null"]:
            real_budgets.setdefault(_canonical(r["base"]), set()).add(r["comparison"])

    acc: dict[tuple, dict] = {}
    for r in records:
        path, d, label = r["path"], r["data"], r["label"]
        if reference_filter == "converged":
            _validate_converged_payload(path, d)
        comparison = r["comparison"]
        if r["null"] and comparison == (None, None):
            candidates = real_budgets.get(_canonical(r["base"]), set())
            if len(candidates) == 1:
                comparison = next(iter(candidates))
                print(f"  ! {path.name}: legacy null requested-reference budget "
                      f"inferred from the single compatible real budget {comparison}")
            elif len(candidates) > 1:
                print(f"  ! {path.name}: ambiguous legacy reference budget; null remains unpaired")

        family_settings = {**r["base"], "comparison_ref": comparison}
        family_id = _fingerprint(family_settings)
        key = (family_id, r["n_samples"], r["n_iterations"], r["null"])

        cfg = d.get("config", {})
        agg = d.get("aggregate", {})
        a = acc.setdefault(key, {
            "label":        label,
            "family_id":    family_id,
            "family_settings": family_settings,
            "geometry":     r["geometry"],
            "object":       r["object"],
            "schema_version": r["schema_version"],
            "model":        r["model"],
            "n_samples":    r["n_samples"],
            "n_iterations": r["n_iterations"],
            "null":         r["null"],
            "task_name":    r["task_name"],
            "ref_n_samples":    cfg.get("ref_n_samples"),
            "ref_n_iterations": cfg.get("ref_n_iterations"),
            "comparison_ref_n_samples": comparison[0],
            "comparison_ref_n_iterations": comparison[1],
            "action_dim":       cfg.get("action_dim"),
            "n_episodes":   0,
            "n_success":    0.0,
            "step_ms":      0.0,
            "series":       {k: {"n": 0, "sum": 0.0, "sumsq": 0.0}
                             for k in ("forward", "reverse",
                                       "ess_ref", "ess_deg", "mu_dist")},
            "episode_series": {k: {"mean": [], "median": []}
                               for k in ("forward", "reverse")},
            "files":        0,
            "source_paths": [],
            "seen_payloads": set(),
            "seen_episode_aliases": set(),
        })
        payload_id = _fingerprint(d)
        if payload_id in a["seen_payloads"]:
            raise ValueError(f"Duplicate result payload in {path.name}; copied files "
                             "must not be counted as independent episodes")
        a["seen_payloads"].add(payload_id)
        aliases_by_episode = _episode_identity_aliases(d)
        for aliases in aliases_by_episode:
            repeated = aliases & a["seen_episode_aliases"]
            if repeated:
                raise ValueError(
                    f"Duplicate episode seed identity in {path.name}: {sorted(repeated)}. "
                    "Same-seed repeats require separate repeated-measures analysis; "
                    "do not pool them as independent episodes."
                )
            a["seen_episode_aliases"].update(aliases)
        if aliases_by_episode and any(not aliases for aliases in aliases_by_episode):
            print(f"  ! {path.name}: legacy episode seeds unavailable; independence "
                  "cannot be verified beyond exact-payload duplicate detection")
        a["files"] += 1
        a["source_paths"].append(str(path))

        n_ep = int(agg.get("n_episodes", 0) or 0)
        a["n_episodes"] += n_ep
        a["n_success"]  += float(agg.get("success_rate", 0.0)) * n_ep
        a["step_ms"]    += float(agg.get("mean_step_ms", 0.0)) * n_ep

        kl_summary = (d.get("kl_converged_only", {})
                      if reference_filter == "converged" else d.get("kl", {}))
        sources = {
            "forward":  kl_summary.get("forward"),
            "reverse":  kl_summary.get("reverse"),
            "ess_ref":  d.get("diagnostics", {}).get("ess_ref"),
            "ess_deg":  d.get("diagnostics", {}).get("ess_deg"),
            "mu_dist":  d.get("diagnostics", {}).get("mu_dist"),
        }
        for key, s in sources.items():
            if not s or not s.get("n") or s.get("mean") is None:
                continue
            n, mean, sd = int(s["n"]), float(s["mean"]), float(s.get("sd") or 0.0)
            t = a["series"][key]
            t["n"]     += n
            t["sum"]   += mean * n
            t["sumsq"] += n * (sd * sd + mean * mean)

        # Recover one KL summary per episode from the raw arrays. Episodes are
        # the uncertainty units for episode-balanced analysis (independence
        # remains an experimental-design assumption). Raw
        # values are also retained for the legacy step-weighted median.
        per_step = d.get("per_step", []) or []
        for r in per_step:
            for key, field in (("forward", "kl_forward"), ("reverse", "kl_reverse")):
                raw_values = r.get(field) or []
                if reference_filter == "converged":
                    flags = r["reference_converged"]
                    vals = _finite_values([
                        value for value, converged in zip(raw_values, flags)
                        if converged
                    ])
                else:
                    vals = _finite_values(raw_values)
                a.setdefault("raw", {}).setdefault(key, []).extend(
                    vals
                )
                if vals:
                    a["episode_series"][key]["mean"].append(float(np.mean(vals)))
                    a["episode_series"][key]["median"].append(float(np.median(vals)))

        # Compatibility fallback for a compact result that kept per-episode KL
        # summaries but omitted raw per-step arrays.
        if not per_step:
            for ep in d.get("episodes", []) or []:
                for key, field in (("forward", "kl_forward"),
                                   ("reverse", "kl_reverse")):
                    if reference_filter == "converged":
                        field = f"kl_converged_only_{key}"
                    s = ep.get(field) or {}
                    for stat in ("mean", "median"):
                        vals = _finite_values([s.get(stat)])
                        a["episode_series"][key][stat].extend(vals)

    cells: dict[str, dict] = {}
    label_counts: dict[str, int] = {}
    for a in acc.values():
        label_counts[a["label"]] = label_counts.get(a["label"], 0) + 1
    for a in acc.values():
        label = a["label"]
        n_ep = a["n_episodes"]
        w    = n_ep if n_ep else 1
        sr   = a["n_success"] / w
        cell = {
            "label":        label,
            "family_id":    a["family_id"],
            "family_settings": a["family_settings"],
            "geometry":     a["geometry"],
            "object":       a["object"],
            "schema_version": a["schema_version"],
            "model":        a["model"],
            "n_samples":    a["n_samples"],
            "n_iterations": a["n_iterations"],
            "null":         a["null"],
            "task_name":    a["task_name"],
            "ref_n_samples":    a["ref_n_samples"],
            "ref_n_iterations": a["ref_n_iterations"],
            "comparison_ref_n_samples": a["comparison_ref_n_samples"],
            "comparison_ref_n_iterations": a["comparison_ref_n_iterations"],
            "action_dim":       a["action_dim"],
            "n_files":      a["files"],
            "source_paths": a["source_paths"],
            "n_episodes":   n_ep,
            "success_rate": sr,
            "success_rate_se":
                math.sqrt(max(sr * (1.0 - sr), 0.0) / n_ep) if n_ep > 1 else 0.0,
            "mean_step_ms": a["step_ms"] / w,
        }
        for key in a["series"]:
            mean, sd, n = _pool_stats(a["series"][key])
            cell[key] = {"mean": mean, "sd": sd, "n": n}
        for key, vals in (a.get("raw") or {}).items():
            cell[key]["median"] = float(np.median(vals)) if vals else None
        cell["episode_series"] = a["episode_series"]
        cell_key = label if label_counts[label] == 1 else f"{label}@{a['family_id']}"
        cells[cell_key] = cell

    return cells, [r["path"] for r in records]


def group_families(cells: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """Partition merged cells for separate, scientifically compatible plots."""
    groups: dict[str, dict[str, dict]] = {}
    for key, cell in cells.items():
        groups.setdefault(cell["family_id"], {})[key] = cell
    return groups


def success_error(cell: dict, method: str) -> tuple[float, float]:
    """Asymmetric (lower, upper) error around the cell's success-rate point."""
    p = float(cell["success_rate"])
    n = int(cell["n_episodes"])
    if method == "none" or n <= 0:
        return 0.0, 0.0
    if method == "se":
        e = float(cell["success_rate_se"])
        return min(e, p), min(e, 1.0 - p)

    # Wilson score interval, 95%. Unlike the Wald p ± SE interval, it does not
    # collapse to zero width when a small pilot observes 0/N or N/N successes.
    z = 1.959963984540054
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / den
    half = z / den * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return p - lo, hi - p


def kl_value(cell: dict, direction: str, stat: str,
             weighting: str = "episode") -> tuple[float | None, float]:
    """(value, ±error) for a cell's KL under the chosen direction and statistic.

    Episode weighting (default) gives every episode one vote and uses the
    between-episode SE. Step weighting reproduces the legacy pooled-step mean.
    The median has no comparable closed-form SE here, so its error is zero.
    """
    if weighting == "episode":
        vals = ((cell.get("episode_series") or {}).get(direction) or {}).get(stat) or []
        if not vals:
            return None, 0.0
        a = np.asarray(vals, dtype=float)
        value = float(np.mean(a)) if stat == "mean" else float(np.median(a))
        error = (float(np.std(a, ddof=1) / math.sqrt(a.size))
                 if stat == "mean" and a.size > 1 else 0.0)
        return value, error

    s = cell.get(direction) or {}
    if stat == "median":
        return s.get("median"), 0.0
    mean, sd, n = s.get("mean"), s.get("sd") or 0.0, s.get("n") or 0
    return mean, (sd / math.sqrt(n) if n > 1 else 0.0)


def pair_with_null(cells: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Split into (real cells, null cells) and attach each real cell's null
    partner, requiring the same scientific family and degraded compute budget.
    """
    reals = [c for c in cells.values() if not c["null"]]
    nulls = [c for c in cells.values() if c["null"]]
    def key(c):
        return (c.get("family_id"), c.get("task_name"), c.get("geometry"),
                c["model"], c["n_samples"], c["n_iterations"])
    by_key = {key(c): c for c in nulls}
    for c in reals:
        c["null_cell"] = by_key.get(key(c))
    reals.sort(key=lambda c: (c.get("family_id", ""), c["n_iterations"], c["n_samples"]))
    nulls.sort(key=lambda c: (c.get("family_id", ""), c["n_iterations"], c["n_samples"]))
    return reals, nulls


def kl_se_available(cell: dict, direction: str, stat: str,
                    weighting: str = "episode") -> bool:
    """Zero returned by kl_value need not mean an estimated zero uncertainty."""
    if stat != "mean":
        return False
    if weighting == "episode":
        vals = ((cell.get("episode_series") or {}).get(direction) or {}).get("mean")
        return len(_finite_values(vals)) >= 2
    return int((cell.get(direction) or {}).get("n") or 0) >= 2


def null_unavailable_reason(cell: dict, direction: str, stat: str,
                            weighting: str = "episode") -> str | None:
    """Explain unavailable diagnostics separately from a negative comparison."""
    null = cell.get("null_cell")
    if null is None:
        return "no compatible null cell"
    if (kl_value(cell, direction, stat, weighting)[0] is None or
            kl_value(null, direction, stat, weighting)[0] is None):
        return "no valid KL comparison"
    if stat == "mean" and weighting == "episode":
        if not all(kl_se_available(c, direction, stat, weighting) for c in (cell, null)):
            return "insufficient replication (<2 valid episodes)"
    return None


def above_null(cell: dict, direction: str, stat: str,
               weighting: str = "episode") -> bool | None:
    """Descriptive independent-run null comparison; not a significance test.

    True means real KL exceeds null KL by more than the two displayed errors
    combined; None means no valid comparison or insufficient replication. Mean
    episode mode needs at least two valid episodes on each side: a placeholder
    zero SE from a single episode must not make the diagnostic pass.
    With --stat median (no error bars)
    this is only a point comparison. Independent runs need not visit the same
    states, so False does not establish that model differences are absent.
    """
    if null_unavailable_reason(cell, direction, stat, weighting) is not None:
        return None
    null = cell["null_cell"]
    v,  e  = kl_value(cell, direction, stat, weighting)
    nv, ne = kl_value(null, direction, stat, weighting)
    if v is None or nv is None:
        return None
    return v - e > nv + ne


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _sample_colors(n_values: list[int]) -> dict[int, tuple]:
    cmap = plt.get_cmap("viridis")
    if len(n_values) == 1:
        return {n_values[0]: cmap(0.35)}
    return {n: cmap(0.12 + 0.76 * i / (len(n_values) - 1))
            for i, n in enumerate(n_values)}


def _scatter_panel(ax, reals, direction, stat, weighting, success_error_method,
                   colors, drop_below_null):
    """Success rate vs. KL, with optional descriptive null screening."""
    plotted = 0
    for c in reals:
        v, e = kl_value(c, direction, stat, weighting)
        if v is None:
            continue
        ok = above_null(c, direction, stat, weighting)
        if ok is False and drop_below_null:
            continue

        color  = colors[c["n_samples"]]
        marker = ITER_MARKERS.get(c["n_iterations"], _FALLBACK_MARKER)
        # Hollow = not above the independent-run diagnostic by this rule.
        face   = color if ok is not False else "none"
        unavailable = null_unavailable_reason(c, direction, stat, weighting)
        if unavailable:
            face = "none" if c.get("null_cell") is None else "#DDDDDD"

        ylo, yhi = success_error(c, success_error_method)
        ax.errorbar(
            v, c["success_rate"] * 100.0,
            xerr=e if e > 0 else None,
            yerr=np.array([[ylo], [yhi]]) * 100.0 if (ylo or yhi) else None,
            fmt="none", ecolor="#666666", elinewidth=1.0,
            capsize=3.0, capthick=1.0, zorder=3,
        )
        ax.plot(v, c["success_rate"] * 100.0, marker=marker, markersize=9,
                markerfacecolor=face, markeredgecolor=color,
                markeredgewidth=1.8, linestyle="none", zorder=4)
        if c.get("null_cell") is None:
            ax.plot(v, c["success_rate"] * 100.0, marker="x", markersize=5,
                    color=color, linestyle="none", zorder=5)
        ax.annotate(
            f"n={c['n_samples']}, i={c['n_iterations']}",
            (v, c["success_rate"] * 100.0),
            textcoords="offset points", xytext=(8, 6),
            fontsize=8, color="#333333",
        )
        plotted += 1

    # Log x: KL spans orders of magnitude across the sweep, but a sweep that
    # lands inside one decade would otherwise show a single labelled tick, so
    # label the 2/3/5 minors too.
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(2.0, 3.0, 5.0)))
    ax.xaxis.set_minor_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:g}")
    )
    ax.tick_params(axis="x", which="minor", labelsize=8)
    weighting_label = "episode-balanced" if weighting == "episode" else "step-weighted"
    ax.set_xlabel(f"KL divergence — {direction} ({stat}, {weighting_label})",
                  fontsize=11)
    ax.set_ylabel("Success rate  (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linewidth=0.4, color="#cccccc", zorder=0)

    # Two legends: colour = n_samples, shape = n_iterations, plus the hollow
    # marker's meaning if any cell was flagged.
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=colors[n], markeredgecolor=colors[n],
                   label=str(n))
        for n in sorted(colors)
    ]
    # Single-episode smoke points often lie at 0%; fixed lower-corner legends
    # would hide the point and its unavailable-diagnostic marker completely.
    legend_edge = "upper" if np.median([c["success_rate"] for c in reals]) <= 0.5 else "lower"
    leg1 = ax.legend(handles=handles, title="n_samples", title_fontsize=9,
                     fontsize=9, loc=f"{legend_edge} left", framealpha=0.9)
    leg1.get_frame().set_linewidth(0.5)
    ax.add_artist(leg1)

    iters = sorted({c["n_iterations"] for c in reals})
    shape_handles = [
        plt.Line2D([], [], marker=ITER_MARKERS.get(i, _FALLBACK_MARKER),
                   linestyle="none", markersize=8, markerfacecolor="#555555",
                   markeredgecolor="#555555", label=str(i))
        for i in iters
    ]
    if any(above_null(c, direction, stat, weighting) is False for c in reals):
        shape_handles.append(
            plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                       markerfacecolor="none", markeredgecolor="#555555",
                       markeredgewidth=1.8, label="not above null diagnostic")
        )
    reasons = {null_unavailable_reason(c, direction, stat, weighting) for c in reals}
    if "insufficient replication (<2 valid episodes)" in reasons:
        shape_handles.append(
            plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                       markerfacecolor="#DDDDDD", markeredgecolor="#555555",
                       label="null diagnostic: insufficient episodes")
        )
    if "no compatible null cell" in reasons:
        shape_handles.append(
            plt.Line2D([], [], marker="x", linestyle="none", markersize=8,
                       color="#555555", label="no compatible null cell")
        )
    if "no valid KL comparison" in reasons:
        shape_handles.append(
            plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                       markerfacecolor="#DDDDDD", markeredgecolor="#555555",
                       label="null diagnostic: no valid KL comparison")
        )
    leg2 = ax.legend(handles=shape_handles, title="n_iterations",
                     title_fontsize=9, fontsize=9, loc=f"{legend_edge} right",
                     framealpha=0.9)
    leg2.get_frame().set_linewidth(0.5)
    return plotted


def _null_panel(ax, reals, direction, stat, weighting, colors):
    """Real KL vs. that config's null-control KL, one cluster per config."""
    centres = np.arange(len(reals))
    bar_w   = 0.38

    for i, c in enumerate(reals):
        v, e   = kl_value(c, direction, stat, weighting)
        null   = c.get("null_cell")
        nv, ne = kl_value(null, direction, stat, weighting) if null else (None, 0.0)
        color  = colors[c["n_samples"]]

        if v is not None:
            ax.bar(centres[i] - bar_w / 2, v, width=bar_w, color=color,
                   label="_nolegend_", zorder=3)
            if e > 0:
                ax.errorbar(centres[i] - bar_w / 2, v, yerr=e, fmt="none",
                            ecolor="black", elinewidth=1.0, capsize=3.0,
                            capthick=1.0, zorder=4)
        if nv is not None:
            ax.bar(centres[i] + bar_w / 2, nv, width=bar_w, color=NULL_COLOR,
                   hatch="//", edgecolor="#888888", label="_nolegend_", zorder=3)
            if ne > 0:
                ax.errorbar(centres[i] + bar_w / 2, nv, yerr=ne, fmt="none",
                            ecolor="black", elinewidth=1.0, capsize=3.0,
                            capthick=1.0, zorder=4)
        elif v is not None:
            ax.annotate("no null", (centres[i] + bar_w / 2, 0),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8, rotation=90, color="#888888")

    ax.set_yscale("log")
    ax.set_ylabel(f"KL — {direction} ({stat}, {weighting})", fontsize=11)
    ax.set_xticks(centres)
    ax.set_xticklabels([f"n={c['n_samples']}\ni={c['n_iterations']}"
                        for c in reals], fontsize=9)
    ax.set_xlabel("Degraded-planner compute", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, color="#cccccc", zorder=0)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   markerfacecolor="#4C72B0", markeredgecolor="#4C72B0",
                   label="vs. reference planner"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   markerfacecolor=NULL_COLOR, markeredgecolor="#888888",
                   label="independent-run null diagnostic"),
    ]
    legend = ax.legend(handles=handles, fontsize=9, loc="upper right",
                       framealpha=0.9)
    legend.get_frame().set_linewidth(0.5)


def plot(reals, direction: str, stat: str, weighting: str,
         success_error_method: str, title: str, out_path: Path,
         drop_below_null: bool):
    if len({c.get("family_id") for c in reals}) > 1:
        raise ValueError("Cannot plot incompatible scientific families together")
    have_null = any(c.get("null_cell") for c in reals)
    n_rows    = 2 if have_null else 1

    # Constrained layout rather than tight_layout: the scatter panel carries two
    # legends and a two-line title, which tight_layout cannot place.
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(max(8, len(reals) * 1.3), 9 if have_null else 6),
        gridspec_kw={"height_ratios": [3, 2]} if have_null else None,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)

    colors = _sample_colors(sorted({c["n_samples"] for c in reals}))

    plotted = _scatter_panel(
        axes[0], reals, direction, stat, weighting, success_error_method,
        colors, drop_below_null,
    )
    axes[0].set_title(title, fontsize=12, fontweight="bold", pad=10)
    if not plotted:
        raise ValueError(
            "Nothing left to plot — every cell was dropped. Re-run without "
            "--drop_below_null to see the cells and their null diagnostics."
        )

    if have_null:
        _null_panel(axes[1], reals, direction, stat, weighting, colors)

    kl_note = ("KL whiskers: +/-1 SE across episodes (when estimable)."
               if weighting == "episode" else "KL whiskers: +/-1 SE across measured steps.")
    if stat == "median":
        kl_note = "Median KL uncertainty is not estimated."
    sr_note = {"wilson": "SR whiskers: Wilson 95% CI.",
               "se": "SR whiskers: +/-1 SE.",
               "none": "SR intervals are not shown."}[success_error_method]
    fig.supxlabel(f"{kl_note}  {sr_note}", fontsize=8, color="#555555")

    # Format follows --out's extension (the default path is .pdf).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
def _print_table(reals, direction, stat, weighting):
    print(f"\n  {'cell':<18} {'eps':>4} {'succ':>7} {'KL':>12} {'±SE':>9} "
          f"{'null KL':>12} {'ESS deg':>8} {'ESS ref':>8}  diagnostic")
    for c in reals:
        v, e   = kl_value(c, direction, stat, weighting)
        null   = c.get("null_cell")
        nv, _  = kl_value(null, direction, stat, weighting) if null else (None, 0.0)
        ok     = above_null(c, direction, stat, weighting)
        verdict = null_unavailable_reason(c, direction, stat, weighting)
        if verdict is None:
            verdict = "above null diagnostic" if ok else "not above null diagnostic"
        se_text = f"{e:.3g}" if kl_se_available(c, direction, stat, weighting) else "n/a"

        ess_d = (c.get("ess_deg") or {}).get("mean")
        ess_r = (c.get("ess_ref") or {}).get("mean")
        # ESS near 1 means concentrated weights; ESS near N means near-uniform
        # weights. These are diagnostic flags, not proof that a covariance is
        # invalid. ESS is not a hard upper bound on covariance rank.
        flags = []
        if ess_d is not None and ess_d < 2.0:
            flags.append("ESS_deg~1")
        if ess_d is not None and ess_d > 0.9 * c["n_samples"]:
            flags.append("ESS_deg~N")
        d = c.get("action_dim")
        if d is not None and ess_d is not None and ess_d <= d:
            flags.append("ESS_deg<=d")
        if d is not None and c["n_samples"] <= d:
            flags.append("raw_cov_rank<=N-1<d")
        if flags:
            verdict += "  [" + ", ".join(flags) + "]"

        print(f"  {c['label']:<18} {c['n_episodes']:>4d} "
              f"{c['success_rate']*100:>6.1f}% "
              f"{v if v is not None else float('nan'):>12.4g} "
              f"{se_text:>9} "
              f"{(nv if nv is not None else float('nan')):>12.4g} "
              f"{(ess_d if ess_d is not None else float('nan')):>8.2f} "
              f"{(ess_r if ess_r is not None else float('nan')):>8.2f}  {verdict}")


def family_output_path(base: Path, family: dict, multiple: bool) -> Path:
    """Keep the old single-family filename; disambiguate multi-family output."""
    if not multiple:
        return base
    geometry = re.sub(r"[^A-Za-z0-9_-]+", "_", family["geometry"]).strip("_")
    suffix = base.suffix or ".pdf"
    stem = base.stem if base.suffix else base.name
    return base.with_name(f"{stem}_{geometry}_{family['family_id']}{suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a directory of online closed-loop KL-divergence "
                    "sweep JSONs: success "
                    "rate vs. KL, with independent-run null diagnostics. "
                    "Incompatible recorded settings produce separate figures."
    )
    parser.add_argument(
        "results_dir", nargs="?", type=Path,
        help="Directory of per-cell JSONs. Defaults to latest "
             "kl_divergence_eval_* dir in results/",
    )
    parser.add_argument(
        "--direction", choices=["forward", "reverse", "auto"], default="auto",
        help="Which KL direction to plot. 'auto' uses the headline_direction "
             "recorded by the sweep (default forward = KL(reference||degraded)).",
    )
    parser.add_argument(
        "--stat", choices=["mean", "median"], default="mean",
        help="Statistic within each episode. Per-step KL is often heavy-tailed "
             "across contact vs. free-flight states, so median is a useful "
             "robustness view.",
    )
    parser.add_argument(
        "--weighting", choices=["episode", "step"], default="episode",
        help="How KL samples contribute to a cell. 'episode' (default) first "
             "summarizes each episode and gives episodes equal weight; 'step' "
             "reproduces the legacy pooled-per-step calculation.",
    )
    parser.add_argument(
        "--reference_filter", choices=["all", "converged"], default="all",
        help="Use every finite planner-valid KL measurement (default), or only "
             "measurements whose convergence-based reference met its tolerance.",
    )
    parser.add_argument(
        "--success_error", choices=["wilson", "se", "none"], default="wilson",
        help="Success-rate uncertainty: Wilson 95%% interval (default), the "
             "legacy symmetric ±1 SE, or no y-error bars.",
    )
    parser.add_argument(
        "--drop_below_null", action="store_true",
        help="Optional descriptive filter: omit cells not above their "
             "independent-run null diagnostic, instead of drawing them hollow. "
             "This is not a significance test.",
    )
    parser.add_argument(
        "--geometry", action="append", default=None,
        help="Only plot this recorded canonical geometry (e.g. duck_high_high). "
             "Repeat to select several. Missing legacy geometry is never inferred.",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default PDF). Multiple compatible families "
                             "append geometry and configuration IDs to this filename.")
    parser.add_argument("--title_note", default="",
                        help="Optional visible figure note, e.g. 'Integration pilot: 3 episodes/cell'.")
    args = parser.parse_args()

    directory = args.results_dir or _latest_dir()
    print(f"Loading directory: {directory}")

    cells, files = merge_dir(directory, args.reference_filter)
    print(f"  Merged {len(files)} file(s) into {len(cells)} cell(s)")
    if args.geometry:
        cells = {key: c for key, c in cells.items() if c["geometry"] in args.geometry}
        if not cells:
            raise ValueError(f"No cells match --geometry {args.geometry}")
    families = group_families(cells)
    real_families = {fid: group for fid, group in families.items()
                     if any(not c["null"] for c in group.values())}
    if not real_families:
        raise ValueError(
            "No non-null cells found. A directory containing only explicit "
            "--null_control diagnostics has no real KL to plot."
        )

    null_only = len(families) - len(real_families)
    if null_only:
        print(f"  ! {null_only} null-only family/families have no compatible real cells")
    print(f"  Producing {len(real_families)} separate scientific-family figure(s)")
    meta = directory / "meta.json"
    legacy_direction = load(meta).get("kl_direction", "forward") if meta.exists() else "forward"
    for family_id, group in sorted(real_families.items()):
        reals, nulls = pair_with_null(group)
        first = reals[0]
        direction = args.direction
        if direction == "auto":
            direction = first["family_settings"]["config"].get("kl_direction", legacy_direction)
        if direction not in ("forward", "reverse"):
            raise ValueError(f"Invalid recorded KL direction {direction!r}")
        print(f"\n  Family {family_id}: {first['task_name']}, "
              f"{first['geometry']}, {first['model']}")
        print(f"  Direction: {direction} ({args.stat}, {args.weighting}-weighted, "
              f"reference_filter={args.reference_filter})")
        print(f"  Real cells: {len(reals)}   null cells: {len(nulls)}")
        if any(c.get("null_cell") is None for c in reals):
            print("  ! Some cells lack a compatible independent-run null diagnostic")
        if any(c["n_episodes"] < 2 for c in reals):
            print("  ! Single-episode diagnostic: not a success-rate conclusion")
        if args.stat == "median":
            print("  ! Median KL uncertainty is not estimated (SE n/a); null screening "
                  "is only a descriptive comparison of point estimates")
        elif any(not kl_se_available(c, direction, args.stat, args.weighting) for c in reals):
            print("  ! KL SE n/a means insufficient valid samples, not zero uncertainty")
        if first["geometry"] == "unknown (legacy)":
            print("  ! Legacy geometry is unrecorded; these results are not verified high_high data")
        _print_table(reals, direction, args.stat, args.weighting)

        ref_ns = first.get("comparison_ref_n_samples")
        ref_ni = first.get("comparison_ref_n_iterations")
        title = (f"Success rate vs. planner KL — {first['task_name']}\n"
                 f"{first['geometry']} | {first['model']} | config {family_id}")
        if ref_ns:
            ref_temperature = first["family_settings"]["config"].get(
                "comparison_ref_temperature"
            )
            temperature_note = (f", T={ref_temperature:g}"
                                if ref_temperature is not None else "")
            ref_mode = first["family_settings"]["config"].get(
                "comparison_ref_iteration_mode", "fixed"
            )
            if ref_mode == "convergence":
                tol = first["family_settings"]["config"].get(
                    "comparison_ref_convergence_tol"
                )
                title += (f"\nreference: {ref_ns} samples, tol {tol:g}, "
                          f"cap {ref_ni} iterations{temperature_note}")
            else:
                title += (f"\nreference: {ref_ns} samples x {ref_ni} iterations"
                          f"{temperature_note}")
        if args.reference_filter == "converged":
            title += " | converged reference solves only"
        if args.title_note:
            title = f"{args.title_note}\n{title}"
        out_base = args.out or directory / f"{directory.name}_plot.pdf"
        out_path = family_output_path(out_base, first, len(real_families) > 1)
        plot(reals, direction, args.stat, args.weighting, args.success_error,
             title, out_path, args.drop_below_null)


if __name__ == "__main__":
    main()
