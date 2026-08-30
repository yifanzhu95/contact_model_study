"""Paired, fixed-state MPPI latency/memory benchmark for Duck geometries.

Unlike a closed-loop episode, every timed ``plan()`` call starts from the same
MuJoCo state.  This keeps an early object drop or a different contact history
from changing the amount of work being compared.  Each (geometry, n_samples)
cell runs in a fresh child process so GPU allocations are released between
cells and an out-of-memory failure cannot poison the rest of the sweep.

The benchmark answers an engineering question only: how much steady planning
time and resident GPU memory each rollout geometry costs on this machine.  It
does not measure task success.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401 - register task implementations
from contact_study.contact_models.config import ContactModelConfig
from contact_study.evaluation import json_io
from contact_study.planners.mppi import MPPIConfig, MPPIController
from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole


HERE = Path(__file__).resolve().parent
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


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def run_cell(args: argparse.Namespace) -> dict:
    wp.init()
    device = wp.get_device("cuda:0")
    free_before = int(device.free_memory)

    task = get_task("grasp_reorient", geometry=args.geometry, role=TaskRole.ROLLOUT)
    mjm, mjd = task.load()
    rollout_dt = task.config.timestep * task.config.eval_substeps_per_rollout
    mjm.opt.timestep = rollout_dt

    rng = np.random.default_rng(args.seed)
    q0, v0, u0 = task.get_inital_state(rng)
    mjd.qpos[:] = q0
    mjd.qvel[:] = v0
    mjd.ctrl[:] = u0
    mujoco.mj_forward(mjm, mjd)
    if hasattr(task, "sample_new_goal"):
        task.sample_new_goal(mjd, rng)

    planner_cfg = MPPIConfig(
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
        seed=args.seed,
        resample_interval=1,
    )
    controller = MPPIController(
        task=task,
        cfg=ContactModelConfig.M2(),
        mppi_cfg=planner_cfg,
        rng=np.random.default_rng(args.seed),
    )
    wp.synchronize()
    free_after_setup = int(device.free_memory)

    for _ in range(args.warmup_plans):
        controller.plan(mjd)
    wp.synchronize()

    times_ms: list[float] = []
    actions: list[np.ndarray] = []
    for _ in range(args.timed_plans):
        start = time.perf_counter()
        actions.append(controller.plan(mjd))
        times_ms.append((time.perf_counter() - start) * 1e3)
    wp.synchronize()
    free_after_plans = int(device.free_memory)

    arr = np.asarray(times_ms, dtype=float)
    action_arr = np.asarray(actions, dtype=float)
    return {
        "ok": True,
        "geometry": args.geometry,
        "label": LABELS.get(args.geometry, args.geometry),
        "n_samples": args.n_samples,
        "horizon": int(controller.horizon),
        "substeps": int(controller.substeps),
        "rollout_dt_seconds": float(controller.rollout_dt),
        "control_dt_seconds": float(controller.control_dt),
        "world_steps_per_plan": int(
            args.n_samples * controller.horizon * controller.substeps
        ),
        "warmup_plans": args.warmup_plans,
        "timed_plans": args.timed_plans,
        "latency_ms": {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": _percentile(arr, 95),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std()),
            "samples": arr.tolist(),
        },
        "throughput_world_steps_per_second": float(
            args.n_samples * controller.horizon * controller.substeps / (arr.mean() * 1e-3)
        ),
        "gpu_memory": {
            "total_bytes": int(device.total_memory),
            "free_before_bytes": free_before,
            "free_after_setup_bytes": free_after_setup,
            "free_after_plans_bytes": free_after_plans,
            "resident_setup_delta_bytes": max(0, free_before - free_after_setup),
            "resident_total_delta_bytes": max(0, free_before - free_after_plans),
        },
        "action_finite": bool(np.isfinite(action_arr).all()),
        "last_plan_ok": bool(controller.last_plan_ok),
        "device": {
            "name": device.name,
            "arch": int(device.arch),
            "warp_version": wp.__version__,
            "mujoco_version": mujoco.__version__,
        },
        "capacities": {"nconmax": args.nconmax, "njmax": args.njmax},
        "seed": args.seed,
    }


def _run_child(args: argparse.Namespace, geometry: str, n_samples: int, path: Path) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cell",
        "--geometry",
        geometry,
        "--n_samples",
        str(n_samples),
        "--time_horizon",
        str(args.time_horizon),
        "--step_time",
        str(args.step_time),
        "--noise_sigma",
        str(args.noise_sigma),
        "--temperature",
        str(args.temperature),
        "--nconmax",
        str(args.nconmax),
        "--njmax",
        str(args.njmax),
        "--warmup_plans",
        str(args.warmup_plans),
        "--timed_plans",
        str(args.timed_plans),
        "--seed",
        str(args.seed),
        "--cell_output",
        str(path),
    ]
    env = os.environ.copy()
    env.setdefault("WARP_CACHE_PATH", "/tmp/contact_study_warp_cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/contact_study_matplotlib")
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    # The child deliberately exits non-zero for a failed cell, but it still
    # writes a structured record first.  Preserve that useful exception instead
    # of replacing it with an often-empty stderr tail.
    if path.exists():
        with path.open() as stream:
            return json.load(stream)
    return {
        "ok": False,
        "geometry": geometry,
        "label": LABELS.get(geometry, geometry),
        "n_samples": n_samples,
        "returncode": completed.returncode,
        "error_tail": (completed.stderr or completed.stdout)[-4000:],
    }


def save_plot(records: list[dict], path: Path) -> None:
    good = [record for record in records if record.get("ok")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for geometry in dict.fromkeys(record["geometry"] for record in good):
        subset = sorted(
            (record for record in good if record["geometry"] == geometry),
            key=lambda record: record["n_samples"],
        )
        xs = [record["n_samples"] for record in subset]
        latency = [record["latency_ms"]["median"] for record in subset]
        memory = [
            record["gpu_memory"]["resident_total_delta_bytes"] / 1024**3
            for record in subset
        ]
        label = subset[0]["label"]
        axes[0].plot(xs, latency, marker="o", label=label)
        axes[1].plot(xs, memory, marker="o", label=label)

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
        axis.set_xlabel("MPPI samples / parallel worlds")
    axes[0].set_ylabel("Median plan latency (ms)")
    axes[0].set_title("Fixed-state steady planning latency")
    axes[1].set_ylabel("Approx. resident GPU memory (GiB)")
    axes[1].set_title("Planner/model resident GPU memory")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selectors", nargs="+", default=list(DEFAULT_SELECTORS))
    parser.add_argument("--samples", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    parser.add_argument("--geometry", default="duck_low_high")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--time_horizon", type=float, default=0.352)
    parser.add_argument("--step_time", type=float, default=0.064)
    parser.add_argument("--noise_sigma", type=float, default=0.23248210744804804)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--nconmax", type=int, default=200)
    parser.add_argument("--njmax", type=int, default=1000)
    parser.add_argument("--warmup_plans", type=int, default=2)
    parser.add_argument("--timed_plans", type=int, default=5)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cell_output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.cell:
        try:
            record = run_cell(args)
        except Exception as exc:  # child isolation reports the cell instead of losing the sweep
            record = {
                "ok": False,
                "geometry": args.geometry,
                "label": LABELS.get(args.geometry, args.geometry),
                "n_samples": args.n_samples,
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

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or HERE / "results" / f"mppi_static_capacity_{stamp}.json"
    cell_dir = out_path.parent / f".{out_path.stem}_cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for n_samples in args.samples:
        for geometry in args.selectors:
            print(f"{geometry:22s} n={n_samples:4d}", end=" ... ", flush=True)
            cell_path = cell_dir / f"{geometry}_n{n_samples}.json"
            record = _run_child(args, geometry, n_samples, cell_path)
            records.append(record)
            if record.get("ok"):
                latency = record["latency_ms"]["median"]
                memory = record["gpu_memory"]["resident_total_delta_bytes"] / 1024**3
                print(f"{latency:8.2f} ms  {memory:5.2f} GiB", flush=True)
            else:
                print("FAILED", flush=True)

    payload = {
        "schema": 1,
        "purpose": "fixed-state MPPI latency and GPU-memory capacity calibration",
        "not_a_success_experiment": True,
        "configuration": {
            "selectors": args.selectors,
            "samples": args.samples,
            "time_horizon_seconds": args.time_horizon,
            "step_time_seconds": args.step_time,
            "noise_sigma": args.noise_sigma,
            "temperature": args.temperature,
            "nconmax": args.nconmax,
            "njmax": args.njmax,
            "warmup_plans": args.warmup_plans,
            "timed_plans": args.timed_plans,
            "paired_seed": args.seed,
        },
        "records": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_io.dump(payload, out_path)
    plot_path = out_path.with_suffix(".png")
    save_plot(records, plot_path)
    print(f"Saved {out_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
