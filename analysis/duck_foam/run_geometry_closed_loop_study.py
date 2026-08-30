"""Paired, resumable closed-loop Duck geometry study.

This is the task-performance companion to ``benchmark_mppi_static_state.py``.
Every geometry is evaluated against the same reference MuJoCo Duck, with the
same initial state, goal seed, planner settings, and sample count.  Geometry
order rotates between seeds to reduce thermal/order bias.  Each episode runs in
a fresh child process, is checkpointed immediately, and can be resumed safely.

The default is intentionally a first formal local study rather than a paper
claim: 10 paired seeds at the project's 256-sample MPPI setting.  More seeds can
be added later without changing the scientific protocol.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import warp as wp

from contact_study.contact_models.config import ContactModelConfig
from contact_study.drivers.run_eval_episode import run_eval_episode
from contact_study.evaluation import json_io
from contact_study.evaluation.trajectory import TrajectoryConfig
from contact_study.planners.mppi import MPPIConfig
from contact_study.tasks.config import EvalSimulatorKind


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_SELECTORS = (
    "duck_low_high",
    "duck_low_foam4",
    "duck_low_foam16a",
    "duck_low_foam16b",
    "duck_low_foam64",
)
LABELS = {
    "duck_low_high": "8-hull baseline",
    "duck_low_foam4": "FOAM 4",
    "duck_low_foam16a": "FOAM 16A",
    "duck_low_foam16b": "FOAM 16B",
    "duck_low_foam64": "FOAM 64",
}


def planner_config(args: argparse.Namespace) -> MPPIConfig:
    return MPPIConfig(
        n_samples=args.n_samples,
        time_horizon=args.time_horizon,
        step_time=args.step_time,
        noise_sigma=args.noise_sigma,
        temperature=args.temperature,
        n_iterations=1,
        warm_start=False,
        ctrl_relative_to_qpos=True,
        nconmax=args.nconmax,
        njmax=args.njmax,
        debug=False,
        delta_range=(None, None),
        use_full_graph=True,
        seed=args.episode_seed,
        resample_interval=1,
    )


def parse_weight_overrides(tokens: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"bad --weights token {token!r}; expected name=value")
        name, value = token.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"bad --weights token {token!r}; name is empty")
        overrides[name] = float(value)
    return overrides


def run_cell(args: argparse.Namespace) -> dict:
    wp.init()
    result = run_eval_episode(
        task_name="grasp_reorient",
        geometry=args.geometry,
        contact_cfg=ContactModelConfig.M2(),
        planner_cfg=planner_config(args),
        rng=np.random.default_rng(args.episode_seed),
        settle_seconds=args.settle,
        max_steps=args.max_steps,
        plan_warmup=args.plan_warmup,
        eval_sim=EvalSimulatorKind.MUJOCO,
        video_path=None,
        cost_weight_overrides=parse_weight_overrides(args.weights) or None,
        fin_ep_on_success=True,
        debug=False,
        verbose=False,
        record=TrajectoryConfig(
            record_trajectory=False,
            record_planner_dist=False,
        ),
    )
    device = wp.get_device("cuda:0")
    return {
        "ok": True,
        "geometry": args.geometry,
        "label": LABELS.get(args.geometry, args.geometry),
        "episode_index": args.episode_index,
        "episode_seed": args.episode_seed,
        "protocol_hash": args.protocol_hash,
        "result": result.to_dict(),
        "software": {
            "python": sys.version.split()[0],
            "mujoco": mujoco.__version__,
            "warp": wp.__version__,
        },
        "device": {"name": device.name, "arch": int(device.arch)},
    }


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    radius = z * np.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total**2)) / denom
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def _finite(values) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def summarize(records: list[dict], selectors: list[str]) -> list[dict]:
    rows: list[dict] = []
    for geometry in selectors:
        cells = [r for r in records if r.get("geometry") == geometry]
        valid = [r["result"] for r in cells if r.get("ok")]
        successes = sum(bool(r["success"]) for r in valid)
        low, high = wilson_interval(successes, len(valid))
        reason_counts: dict[str, int] = {}
        for result in valid:
            reason = str(result.get("end_reason", "unknown"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        plan_counts = np.asarray(
            [max(0, int(r.get("n_steps_taken", 0))) for r in valid], dtype=float
        )
        means = np.asarray([float(r.get("mean_step_ms", 0.0)) for r in valid])
        weighted_latency = (
            float(np.sum(means * plan_counts) / np.sum(plan_counts))
            if np.sum(plan_counts) > 0
            else 0.0
        )
        success_steps = _finite(
            r["steps_to_success"]
            for r in valid
            if r.get("steps_to_success") is not None
        )

        goal_stats = {}
        for metric in ("pos", "quat", "vel"):
            vals = _finite(
                r["final_goal_errs"][metric]
                for r in valid
                if r.get("final_goal_errs") and metric in r["final_goal_errs"]
            )
            goal_stats[metric] = {
                "median": float(np.median(vals)) if vals.size else None,
                "mean": float(np.mean(vals)) if vals.size else None,
                "p95": float(np.percentile(vals, 95)) if vals.size else None,
            }

        ep_medians = _finite(r.get("median_step_ms", 0.0) for r in valid)
        ep_p95s = _finite(r.get("p95_step_ms", 0.0) for r in valid)
        rows.append({
            "geometry": geometry,
            "label": LABELS.get(geometry, geometry),
            "n_requested": len(cells),
            "n_valid": len(valid),
            "n_errors": len(cells) - len(valid),
            "n_success": successes,
            "success_rate": successes / len(valid) if valid else 0.0,
            "success_ci95_low": low,
            "success_ci95_high": high,
            "end_reason_counts": reason_counts,
            "steps_to_success_median": (
                float(np.median(success_steps)) if success_steps.size else None
            ),
            "steps_to_success_mean": (
                float(np.mean(success_steps)) if success_steps.size else None
            ),
            "plan_latency_ms_weighted_mean": weighted_latency,
            "episode_median_plan_ms_median": (
                float(np.median(ep_medians)) if ep_medians.size else None
            ),
            "episode_p95_plan_ms_median": (
                float(np.median(ep_p95s)) if ep_p95s.size else None
            ),
            "final_goal_errors": goal_stats,
        })
    return rows


def save_episode_csv(records: list[dict], path: Path) -> None:
    fields = [
        "geometry", "label", "episode_index", "episode_seed", "ok", "error",
        "success", "end_reason", "n_steps_taken", "steps_to_success",
        "mean_step_ms", "median_step_ms", "p95_step_ms", "max_step_ms",
        "goal_pos", "goal_quat", "goal_vel",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in records:
            result = cell.get("result", {})
            goal = result.get("final_goal_errs") or {}
            writer.writerow({
                "geometry": cell.get("geometry"),
                "label": cell.get("label"),
                "episode_index": cell.get("episode_index"),
                "episode_seed": cell.get("episode_seed"),
                "ok": cell.get("ok", False),
                "error": cell.get("error"),
                "success": result.get("success"),
                "end_reason": result.get("end_reason"),
                "n_steps_taken": result.get("n_steps_taken"),
                "steps_to_success": result.get("steps_to_success"),
                "mean_step_ms": result.get("mean_step_ms"),
                "median_step_ms": result.get("median_step_ms"),
                "p95_step_ms": result.get("p95_step_ms"),
                "max_step_ms": result.get("max_step_ms"),
                "goal_pos": goal.get("pos"),
                "goal_quat": goal.get("quat"),
                "goal_vel": goal.get("vel"),
            })


def save_plot(summaries: list[dict], path: Path) -> None:
    labels = [row["label"] for row in summaries]
    x = np.arange(len(labels))
    rates = np.asarray([row["success_rate"] for row in summaries])
    low = np.asarray([row["success_ci95_low"] for row in summaries])
    high = np.asarray([row["success_ci95_high"] for row in summaries])
    latency = [row["plan_latency_ms_weighted_mean"] for row in summaries]
    quat = [
        row["final_goal_errors"]["quat"]["median"]
        if row["final_goal_errors"]["quat"]["median"] is not None else np.nan
        for row in summaries
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(x, rates, color="#4c78a8")
    axes[0].errorbar(x, rates, yerr=np.vstack([rates - low, high - rates]),
                     fmt="none", color="black", capsize=4)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Success rate (Wilson 95% CI)")
    axes[0].set_title("Reference-eval task success")

    axes[1].bar(x, latency, color="#f58518")
    axes[1].set_ylabel("Mean plan latency (ms)")
    axes[1].set_title("Closed-loop planning cost")

    axes[2].bar(x, quat, color="#54a24b")
    axes[2].axhline(0.04, color="black", linestyle="--", linewidth=1,
                    label="success threshold")
    axes[2].set_ylabel("Median final quaternion error")
    axes[2].set_title("Orientation accuracy")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Duck rollout geometry: paired MPPI closed-loop study")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def git_provenance() -> dict:
    def command(*parts: str) -> str:
        completed = subprocess.run(
            ["git", *parts], cwd=REPO_ROOT, text=True, capture_output=True
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unknown"

    status = command("status", "--short")
    return {
        "head": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(status),
        "local_changes": status.splitlines(),
        "github_write_performed": False,
    }


def _run_child(
    args: argparse.Namespace,
    geometry: str,
    episode_index: int,
    seed: int,
    protocol_hash: str,
    path: Path,
) -> dict:
    if path.exists():
        with path.open() as stream:
            existing = json.load(stream)
        # Resume only completed cells. A transient CUDA/OOM/host failure should
        # be retried rather than fossilized forever by the checkpoint.
        if existing.get("protocol_hash") == protocol_hash and existing.get("ok"):
            existing["resumed"] = True
            return existing

    command = [
        sys.executable, str(Path(__file__).resolve()), "--cell",
        "--geometry", geometry,
        "--episode_index", str(episode_index),
        "--episode_seed", str(seed),
        "--protocol_hash", protocol_hash,
        "--n_samples", str(args.n_samples),
        "--time_horizon", str(args.time_horizon),
        "--step_time", str(args.step_time),
        "--noise_sigma", str(args.noise_sigma),
        "--temperature", str(args.temperature),
        "--nconmax", str(args.nconmax),
        "--njmax", str(args.njmax),
        "--settle", str(args.settle),
        "--max_steps", str(args.max_steps),
        "--plan_warmup", str(args.plan_warmup),
        "--cell_output", str(path),
    ]
    if args.weights:
        command.extend(["--weights", *args.weights])
    env = os.environ.copy()
    env.setdefault("WARP_CACHE_PATH", "/tmp/contact_study_warp_cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/contact_study_matplotlib")
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    if path.exists():
        with path.open() as stream:
            return json.load(stream)
    return {
        "ok": False,
        "geometry": geometry,
        "label": LABELS.get(geometry, geometry),
        "episode_index": episode_index,
        "episode_seed": seed,
        "protocol_hash": protocol_hash,
        "returncode": completed.returncode,
        "error": (completed.stderr or completed.stdout)[-6000:],
    }


def write_checkpoint(payload: dict, output: Path) -> None:
    payload["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    payload["summaries"] = summarize(
        payload["records"], payload["protocol"]["selectors"]
    )
    json_io.dump(payload, output)
    save_episode_csv(payload["records"], output.with_suffix(".csv"))
    save_plot(payload["summaries"], output.with_suffix(".png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selectors", nargs="+", default=list(DEFAULT_SELECTORS))
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--episode_seeds", nargs="+", type=int, default=None,
        help="Explicit per-episode seeds (duplicates allowed for repeatability "
             "audits); when given, overrides --n_episodes/--seed generation.",
    )
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--time_horizon", type=float, default=0.352)
    parser.add_argument("--step_time", type=float, default=0.064)
    parser.add_argument("--noise_sigma", type=float, default=0.23248210744804804)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--nconmax", type=int, default=200)
    parser.add_argument("--njmax", type=int, default=1000)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--plan_warmup", type=int, default=2)
    parser.add_argument(
        "--weights", nargs="*", default=[],
        help="Rollout cost overrides as name=value tokens; recorded in the protocol.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--geometry", default="duck_low_high", help=argparse.SUPPRESS)
    parser.add_argument("--episode_index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--episode_seed", type=int, default=64, help=argparse.SUPPRESS)
    parser.add_argument("--protocol_hash", default="", help=argparse.SUPPRESS)
    parser.add_argument("--cell_output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.cell:
        try:
            record = run_cell(args)
        except Exception as exc:
            record = {
                "ok": False,
                "geometry": args.geometry,
                "label": LABELS.get(args.geometry, args.geometry),
                "episode_index": args.episode_index,
                "episode_seed": args.episode_seed,
                "protocol_hash": args.protocol_hash,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        if args.cell_output is None:
            print(json.dumps(record, indent=2))
        else:
            args.cell_output.parent.mkdir(parents=True, exist_ok=True)
            json_io.dump(record, args.cell_output)
        if not record["ok"]:
            raise SystemExit(1)
        return

    if args.episode_seeds is not None:
        seeds = list(args.episode_seeds)
        args.n_episodes = len(seeds)
    else:
        seed_sequences = np.random.SeedSequence(args.seed).spawn(args.n_episodes)
        seeds = [int(seq.generate_state(1)[0]) for seq in seed_sequences]
    if args.n_episodes < 1 or args.max_steps < 1:
        raise ValueError("n_episodes and max_steps must both be >= 1")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or HERE / "results" / f"duck_geometry_closed_loop_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    protocol = {
        "study_type": "paired fixed-sample synchronous closed-loop",
        "task": "grasp_reorient",
        "rollout_contact_model": "M2_mjwarp_soft",
        "eval_simulator": "MuJoCo",
        "eval_geometry": "scenes/leap/env_leap_eval_duck.xml (reference 8-hull Duck)",
        "selectors": list(args.selectors),
        "n_episodes_per_geometry": args.n_episodes,
        "paired_episode_seeds": seeds,
        "root_seed": args.seed,
        "explicit_episode_seeds": args.episode_seeds is not None,
        "n_samples": args.n_samples,
        "time_horizon_seconds": args.time_horizon,
        "step_time_seconds": args.step_time,
        "noise_sigma": args.noise_sigma,
        "temperature": args.temperature,
        "n_iterations": 1,
        "warm_start": False,
        "resample_interval": 1,
        "nconmax": args.nconmax,
        "njmax": args.njmax,
        "settle_seconds": args.settle,
        "max_steps": args.max_steps,
        "discarded_warmup_plans": args.plan_warmup,
        "cost_weight_overrides": parse_weight_overrides(args.weights),
        "trajectory_recording": False,
        "geometry_order": "cyclic rotation by paired-seed index",
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()[:16]
    cell_dir = output.parent / f".{output.stem}_cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "purpose": "Duck rollout-geometry accuracy/speed comparison",
        "claim_scope": "local paired study; expand seeds before publication claims",
        "protocol_hash": protocol_hash,
        "protocol": protocol,
        "provenance": git_provenance(),
        "records": [],
        "summaries": [],
    }

    for episode_index, seed in enumerate(seeds):
        offset = episode_index % len(args.selectors)
        order = list(args.selectors[offset:]) + list(args.selectors[:offset])
        for geometry in order:
            print(
                f"pair {episode_index + 1:02d}/{args.n_episodes}  "
                f"{geometry:22s} seed={seed}",
                end=" ... ", flush=True,
            )
            cell_path = cell_dir / f"pair{episode_index:03d}_{geometry}_seed{seed}.json"
            record = _run_child(
                args, geometry, episode_index, seed, protocol_hash, cell_path
            )
            payload["records"].append(record)
            if record.get("ok"):
                result = record["result"]
                print(
                    f"{result['end_reason']:7s} steps={result['n_steps_taken']:4d} "
                    f"plan={result['median_step_ms']:.1f}/{result['p95_step_ms']:.1f} ms",
                    flush=True,
                )
            else:
                print(f"ERROR {record.get('error', '')[:120]}", flush=True)
            write_checkpoint(payload, output)

    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.csv')}")
    print(f"Saved {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
