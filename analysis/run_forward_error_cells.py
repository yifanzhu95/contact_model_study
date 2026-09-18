#!/usr/bin/env python3
"""Compute per-cell one-control-step forward error from recorded trajectories.

This is an offline counterpart of ``run_kl_divergence_cell.py``. Each source
cell keeps its own success rate. From that cell's synchronous trajectories,
every ``--stride``-th paired state/action row is replayed through:

  1. the contact model named by the source cell (M1--M4), and
  2. the Pinocchio evaluation simulator used as the reference.

Both start from the same recorded qpos/qvel and receive the same recorded
absolute ctrl for one complete control interval.  Raw aligned states are kept
in JSON; cell summaries are also written to CSV so plotting never needs to
repeat the simulations.  By default, Pinocchio is reconstructed once per
episode and teacher-forced through the recorded states so its iterative contact
solver keeps the same within-episode history; ``--pin-reference-mode fresh`` is
available as a sensitivity check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import contact_study.tasks  # noqa: E402,F401 -- registers task classes
from contact_study.contact_models import api  # noqa: E402
from contact_study.evaluation import json_io  # noqa: E402
from contact_study.tasks.base import TaskRole, get_task  # noqa: E402
from contact_study.tasks.config import EvalSimulatorKind  # noqa: E402

from one_step_forward_error import (  # noqa: E402
    MODEL_FACTORIES,
    _checkpoint_object_errors,
    _contact_count,
    _git_commit,
    _integer_ratio,
    _portable_path,
    _primary_object_error,
    _sha256,
    pinocchio_trace,
    rollout_trace,
    sample_from_record,
    state_errors,
)


def stats(values: list[float]) -> dict[str, float | int | None]:
    """KL-script-compatible descriptive statistics, plus standard error."""
    if not values:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "se": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Cannot aggregate non-finite forward errors")
    sd = float(arr.std())
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": sd,
        "se": sd / math.sqrt(arr.size),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def candidate_rows(record: dict[str, Any], stride: int) -> list[tuple[int, int]]:
    """Every stride-th transition in every episode, never the final state row."""
    rows: list[tuple[int, int]] = []
    episodes = record.get("episodes") or []
    for episode_index, episode in enumerate(episodes):
        trajectory = episode.get("trajectory") or {}
        if trajectory.get("driver") != "sync":
            raise ValueError(
                f"episode {episode_index} is not synchronous: "
                f"driver={trajectory.get('driver')!r}"
            )
        steps = trajectory.get("steps") or {}
        n_states = len(steps.get("qpos") or [])
        if n_states < 2:
            continue
        rows.extend((episode_index, step) for step in range(0, n_states - 1, stride))
    return rows


def evenly_limit_rows(
    rows: list[tuple[int, int]], max_count: int | None
) -> list[tuple[int, int]]:
    """For smoke tests, spread a small subset across the whole cell trajectory."""
    if max_count is None or len(rows) <= max_count:
        return rows
    positions = np.linspace(0, len(rows) - 1, max_count)
    indices = sorted({int(round(value)) for value in positions})
    return [rows[index] for index in indices]


def _action_ctrl_residual(
    sample: dict[str, Any], robot_qpos_adr: int
) -> float | None:
    context = sample["context"]
    if not bool(context.get("ctrl_relative_to_qpos", True)):
        return None
    nu = sample["nu"]
    reconstructed = (
        sample["q0"][robot_qpos_adr : robot_qpos_adr + nu] + sample["action"]
    )
    clip_lo, clip_hi = context.get("clip_lo"), context.get("clip_hi")
    if clip_lo is not None:
        reconstructed = np.clip(reconstructed, clip_lo, clip_hi)
    return float(np.max(np.abs(reconstructed - sample["ctrl"])))


def _ensure_finite(name: str, *arrays: np.ndarray) -> None:
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError(f"{name} produced NaN or infinity")


def _diagnostic_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Selected-interval counters plus episode-cumulative contact extrema."""
    return {
        "n_substeps": int(after.get("n_substeps", 0))
        - int(before.get("n_substeps", 0)),
        "n_contact_substeps": int(after.get("n_contact_substeps", 0))
        - int(before.get("n_contact_substeps", 0)),
        "n_nonconverged": int(after.get("n_nonconverged", 0))
        - int(before.get("n_nonconverged", 0)),
        "max_n_contacts": int(after.get("max_n_contacts", 0)),
        "min_penetration_m": after.get("min_penetration_m"),
        "scope_note": (
            "substep/contact/nonconvergence counts cover this selected control "
            "interval; contact extrema are cumulative from episode start"
        ),
    }


def history_aligned_pinocchio_traces(
    record: dict[str, Any],
    rows: list[tuple[int, int]],
    *,
    task_name: str,
    geometry: str,
    n_rollout_steps: int,
    eval_substeps_per_rollout: int,
    expected_eval_dt: float,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Reconstruct Pinocchio's per-episode solver history with teacher forcing.

    Pinocchio's iterative contact solver carries internal state that is not part
    of recorded qpos/qvel.  A brand-new simulator at every sampled transition
    therefore need not reproduce the evaluator that generated the trajectory.
    Each episode is rebuilt from its recorded pre-settle state, followed by
    every control step in order. Before each step, the visible state is reset to the
    recorded qpos/qvel (preventing JSON-rounding drift), while the simulator and
    solver objects remain alive (preserving contact-solver history).
    """
    selected_by_episode: dict[int, set[int]] = {}
    for episode_index, step_index in rows:
        selected_by_episode.setdefault(episode_index, set()).add(step_index)

    traces: dict[
        tuple[int, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]
    ] = {}
    episodes = record.get("episodes") or []
    n_eval_steps = n_rollout_steps * eval_substeps_per_rollout
    for episode_index, selected_steps in sorted(selected_by_episode.items()):
        trajectory = episodes[episode_index]["trajectory"]
        context = trajectory["context"]
        steps = trajectory["steps"]
        eval_task = get_task(task_name, geometry=geometry, role=TaskRole.EVAL)
        eval_task.load()
        eval_task.config.eval_sim = EvalSimulatorKind.PINOCCHIO
        sim = eval_task.make_eval_simulator(video_path=None, render=False)
        if not math.isclose(
            float(sim.timestep), expected_eval_dt, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(
                f"Pinocchio timestep={sim.timestep}, expected "
                f"eval_dt={expected_eval_dt}"
            )

        sim.reset(
            np.asarray(context["q0"], dtype=np.float64),
            np.asarray(context["v0"], dtype=np.float64),
        )
        initial_ctrl = np.asarray(context["u0"], dtype=np.float64)
        for _ in range(int(context["n_settle_steps"])):
            sim.apply_control(initial_ctrl)
            sim.step(int(context["eval_substeps_per_rollout"]))

        max_selected_step = max(selected_steps)
        print(
            f"  reconstructing Pinocchio history: episode={episode_index} "
            f"through step={max_selected_step} selected={len(selected_steps)}"
        )
        for step_index in range(max_selected_step + 1):
            # Teacher forcing preserves the recorded physical state while the
            # same simulator/ADMM solver carries its internal episode history.
            sim.set_state(
                np.asarray(steps["qpos"][step_index], dtype=np.float64),
                np.asarray(steps["qvel"][step_index], dtype=np.float64),
            )
            sim.apply_control(
                np.asarray(steps["ctrl"][step_index], dtype=np.float64)
            )
            before = sim.diagnostics() if hasattr(sim, "diagnostics") else {}
            if step_index in selected_steps:
                qpos_trace, qvel_trace = [], []
                for _ in range(n_rollout_steps):
                    sim.step(eval_substeps_per_rollout)
                    state = sim.get_state()
                    qpos_trace.append(
                        np.asarray(state.qpos, dtype=np.float64).copy()
                    )
                    qvel_trace.append(
                        np.asarray(state.qvel, dtype=np.float64).copy()
                    )
                after = sim.diagnostics() if hasattr(sim, "diagnostics") else {}
                traces[(episode_index, step_index)] = (
                    np.asarray(qpos_trace),
                    np.asarray(qvel_trace),
                    _diagnostic_delta(before, after),
                )
            else:
                sim.step(n_eval_steps)
    if len(traces) != len(rows):
        raise RuntimeError(
            f"History-aligned Pinocchio produced {len(traces)} traces for "
            f"{len(rows)} selected transitions"
        )
    return traces


def run_cell(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    with path.open() as stream:
        record = json.load(stream)

    source_model = str(record.get("model", ""))
    if source_model not in MODEL_FACTORIES:
        raise ValueError(
            f"{path}: model={source_model!r}; expected one of {tuple(MODEL_FACTORIES)}"
        )
    rows_all = candidate_rows(record, args.stride)
    rows = evenly_limit_rows(rows_all, args.max_samples_per_cell)
    if not rows:
        raise ValueError(f"{path}: no replayable state/action transitions")

    first = sample_from_record(record, *rows[0])
    context = first["context"]
    task_name = str(context.get("task", record.get("task", "")))
    geometry = str(context.get("geometry", record.get("geometry", "")))
    source_eval_sim = str(context.get("eval_sim", "")).lower()
    if args.pin_reference_mode == "history_aligned" and source_eval_sim != "pinocchio":
        raise ValueError(
            "history_aligned Pinocchio replay requires a source trajectory "
            f"recorded with eval_sim='pinocchio'; got {source_eval_sim!r}. "
            "Use --pin-reference-mode fresh for a new independent reference."
        )
    control_dt = float(context["control_dt"])
    rollout_dt = float(context["rollout_dt"])
    eval_dt = float(context["eval_dt"])
    eval_substeps = int(context["eval_substeps_per_rollout"])
    n_rollout_steps = _integer_ratio(
        control_dt, rollout_dt, name="control_dt / rollout_dt"
    )
    n_eval_steps = _integer_ratio(control_dt, eval_dt, name="control_dt / eval_dt")
    if n_rollout_steps != int(context["substeps"]):
        raise ValueError("Recorded rollout substeps disagree with the timing ratio")
    if n_eval_steps != int(context["eval_steps_per_control"]):
        raise ValueError("Recorded eval substeps disagree with the timing ratio")
    if n_eval_steps != n_rollout_steps * eval_substeps:
        raise ValueError("Recorded timing hierarchy is internally inconsistent")

    rollout_task = get_task(task_name, geometry=geometry, role=TaskRole.ROLLOUT)
    mjm, _ = rollout_task.load()
    mjm.opt.timestep = rollout_dt
    if (mjm.nq, mjm.nv, mjm.nu) != (first["nq"], first["nv"], first["nu"]):
        raise ValueError("Recorded and rollout-model state dimensions disagree")
    index = np.asarray(rollout_task.index_vector, dtype=int)
    object_qpos_adr, object_qvel_adr = int(index[0]), int(index[1])
    robot_qpos_adr = int(context["robot_qpos_adr"])
    robot_qvel_adr = 0
    if object_qpos_adr != robot_qpos_adr + mjm.nu:
        raise ValueError("Unexpected qpos layout")
    if object_qvel_adr != robot_qvel_adr + mjm.nu:
        raise ValueError("Unexpected qvel layout")

    contact_cfg = MODEL_FACTORIES[source_model]()
    sample_records: list[dict[str, Any]] = []
    start = time.perf_counter()

    print(
        f"[{path.name}] model={source_model} candidates={len(rows_all)} "
        f"selected={len(rows)} stride={args.stride}"
    )
    pin_history = None
    if args.pin_reference_mode == "history_aligned":
        pin_history = history_aligned_pinocchio_traces(
            record,
            rows,
            task_name=task_name,
            geometry=geometry,
            n_rollout_steps=n_rollout_steps,
            eval_substeps_per_rollout=eval_substeps,
            expected_eval_dt=eval_dt,
        )
    for ordinal, (episode_index, step_index) in enumerate(rows):
        sample = sample_from_record(record, episode_index, step_index)
        sample_context = sample["context"]
        timing_tuple = (
            float(sample_context["control_dt"]),
            float(sample_context["rollout_dt"]),
            float(sample_context["eval_dt"]),
            int(sample_context["eval_substeps_per_rollout"]),
        )
        if timing_tuple != (control_dt, rollout_dt, eval_dt, eval_substeps):
            raise ValueError(
                f"episode {episode_index} step {step_index} uses different timing"
            )
        if not math.isclose(
            sample["recorded_transition_dt"],
            control_dt,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"episode {episode_index} step {step_index}: recorded "
                f"t[k+1]-t[k]={sample['recorded_transition_dt']} but "
                f"control_dt={control_dt}"
            )
        residual = _action_ctrl_residual(sample, robot_qpos_adr)
        if residual is not None and residual > args.ctrl_reconstruction_tolerance:
            raise ValueError(
                f"episode {episode_index} step {step_index}: action/ctrl identity "
                f"residual {residual:.3e} exceeds "
                f"{args.ctrl_reconstruction_tolerance:.3e}"
            )

        # Build a fresh device model as well as fresh data for every sample.
        # Some backends keep mutable device-side model state, so reusing only
        # the model can make a later sample depend on an earlier replay.
        model = api.put_model(mjm, contact_cfg)
        rollout_qpos, rollout_qvel = rollout_trace(
            mjm,
            model,
            sample["q0"],
            sample["v0"],
            sample["ctrl"],
            n_rollout_steps,
            nconmax=args.nconmax,
            njmax=args.njmax,
        )
        if pin_history is not None:
            pin_qpos, pin_qvel, pin_diag = pin_history[
                (episode_index, step_index)
            ]
        else:
            pin_qpos, pin_qvel, pin_diag = pinocchio_trace(
                task_name,
                geometry,
                sample["q0"],
                sample["v0"],
                sample["ctrl"],
                n_rollout_steps,
                eval_substeps,
                eval_dt,
            )
        _ensure_finite(
            f"episode {episode_index} step {step_index}",
            rollout_qpos,
            rollout_qvel,
            pin_qpos,
            pin_qvel,
        )

        endpoint = state_errors(
            rollout_qpos[-1],
            rollout_qvel[-1],
            pin_qpos[-1],
            pin_qvel[-1],
            object_qpos_adr=object_qpos_adr,
            object_qvel_adr=object_qvel_adr,
            robot_qpos_adr=robot_qpos_adr,
            robot_qvel_adr=robot_qvel_adr,
            nu=mjm.nu,
        )
        primary = _primary_object_error(endpoint)
        pin_vs_recorded = state_errors(
            pin_qpos[-1],
            pin_qvel[-1],
            sample["q_recorded_next"],
            sample["v_recorded_next"],
            object_qpos_adr=object_qpos_adr,
            object_qvel_adr=object_qvel_adr,
            robot_qpos_adr=robot_qpos_adr,
            robot_qvel_adr=robot_qvel_adr,
            nu=mjm.nu,
        )
        checkpoint_pos, checkpoint_rot, checkpoint_combined = (
            _checkpoint_object_errors(rollout_qpos, pin_qpos, object_qpos_adr)
        )

        sample_record: dict[str, Any] = {
            "sample_ordinal": ordinal,
            "episode_index": episode_index,
            "step_index": step_index,
            "recorded_time_k_s": sample["recorded_time_k"],
            "recorded_transition_dt_s": sample["recorded_transition_dt"],
            "ctrl_reconstruction_max_abs": residual,
            "rollout_geometry_contacts_at_k": _contact_count(
                mjm, sample["q0"], sample["v0"]
            ),
            "pinocchio_diagnostics": pin_diag,
            "primary_error": primary,
            "errors": endpoint,
            "pinocchio_reference_vs_original_recorded_next": pin_vs_recorded,
            "state": {
                "qpos_k": json_io.compact(sample["q0"], precision=0),
                "qvel_k": json_io.compact(sample["v0"], precision=0),
                "selected_action_k": json_io.compact(sample["action"], precision=0),
                "ctrl_k": json_io.compact(sample["ctrl"], precision=0),
                "original_recorded_qpos_k_plus_1": json_io.compact(
                    sample["q_recorded_next"], precision=0
                ),
                "original_recorded_qvel_k_plus_1": json_io.compact(
                    sample["v_recorded_next"], precision=0
                ),
            },
        }
        if args.save_checkpoints:
            sample_record["aligned_checkpoints"] = {
                "time_from_k_s": json_io.compact(
                    np.arange(1, n_rollout_steps + 1, dtype=np.float64)
                    * rollout_dt,
                    precision=0,
                ),
                "rollout_qpos": json_io.compact(rollout_qpos, precision=0),
                "rollout_qvel": json_io.compact(rollout_qvel, precision=0),
                "pinocchio_qpos": json_io.compact(pin_qpos, precision=0),
                "pinocchio_qvel": json_io.compact(pin_qvel, precision=0),
                "object_position_error_m": json_io.compact(
                    checkpoint_pos, precision=0
                ),
                "object_orientation_so3_error_rad": json_io.compact(
                    checkpoint_rot, precision=0
                ),
                "unweighted_object_error_m_plus_rad": json_io.compact(
                    checkpoint_combined, precision=0
                ),
            }
        sample_records.append(sample_record)
        print(
            f"  sample {ordinal + 1:>3}/{len(rows)}  ep={episode_index} "
            f"step={step_index:>4}  error={primary['unweighted_sum_m_plus_rad']:.6g}"
        )

    elapsed = time.perf_counter() - start
    metric_paths = {
        "primary_unweighted_m_plus_rad": (
            "primary_error",
            "unweighted_sum_m_plus_rad",
        ),
        "object_position_m": ("errors", "object_position_m", "l2"),
        "object_orientation_so3_rad": (
            "errors",
            "object_orientation",
            "geodesic_rad",
        ),
        "object_linear_velocity_m_per_s": (
            "errors",
            "object_linear_velocity_m_per_s",
            "l2",
        ),
        "object_angular_velocity_rad_per_s": (
            "errors",
            "object_angular_velocity_rad_per_s",
            "l2",
        ),
        "hand_joint_position_rad": ("errors", "hand_joint_position_rad", "l2"),
        "hand_joint_velocity_rad_per_s": (
            "errors",
            "hand_joint_velocity_rad_per_s",
            "l2",
        ),
        "pin_replay_vs_recorded_object_position_m": (
            "pinocchio_reference_vs_original_recorded_next",
            "object_position_m",
            "l2",
        ),
        "pin_replay_vs_recorded_orientation_rad": (
            "pinocchio_reference_vs_original_recorded_next",
            "object_orientation",
            "geodesic_rad",
        ),
    }

    def metric_value(sample_record: dict[str, Any], keys: tuple[str, ...]) -> float:
        value: Any = sample_record
        for key in keys:
            value = value[key]
        return float(value)

    # The transition-weighted summary gives every sampled control transition
    # equal weight. That is useful for describing the distribution of local
    # one-step errors, but a long timeout episode then contributes more samples
    # than a short successful episode.  Keep that summary for compatibility and
    # add an episode-balanced view: first average within each episode, then give
    # those episode means equal weight.
    summaries: dict[str, dict[str, float | int | None]] = {}
    episode_balanced_summaries: dict[str, dict[str, float | int | None]] = {}
    sampled_episode_indices = sorted(
        {int(sample["episode_index"]) for sample in sample_records}
    )
    for name, keys in metric_paths.items():
        values = [metric_value(sample, keys) for sample in sample_records]
        summaries[name] = stats(values)
        episode_means = [
            float(
                np.mean(
                    [
                        metric_value(sample, keys)
                        for sample in sample_records
                        if int(sample["episode_index"]) == episode_index
                    ]
                )
            )
            for episode_index in sampled_episode_indices
        ]
        episode_balanced_summaries[name] = stats(episode_means)

    source_episodes = record.get("episodes") or []
    per_episode: list[dict[str, Any]] = []
    for episode_index in sampled_episode_indices:
        episode_samples = [
            sample
            for sample in sample_records
            if int(sample["episode_index"]) == episode_index
        ]
        episode_source = source_episodes[episode_index]
        per_episode.append(
            {
                "episode_index": episode_index,
                "success": bool(episode_source.get("success", False)),
                "end_reason": episode_source.get("end_reason"),
                "n_steps_taken": episode_source.get("n_steps_taken"),
                "steps_to_success": episode_source.get("steps_to_success"),
                "n_forward_error_samples": len(episode_samples),
                "forward_error": {
                    name: stats(
                        [metric_value(sample, keys) for sample in episode_samples]
                    )
                    for name, keys in metric_paths.items()
                },
            }
        )

    diagnostics = [sample["pinocchio_diagnostics"] for sample in sample_records]
    pinocchio_diagnostics_summary = {
        "n_samples": len(diagnostics),
        "total_substeps": sum(int(item.get("n_substeps", 0)) for item in diagnostics),
        "total_contact_substeps": sum(
            int(item.get("n_contact_substeps", 0)) for item in diagnostics
        ),
        "total_nonconverged_substeps": sum(
            int(item.get("n_nonconverged", 0)) for item in diagnostics
        ),
        "n_samples_with_nonconvergence": sum(
            int(item.get("n_nonconverged", 0)) > 0 for item in diagnostics
        ),
        "max_n_contacts": max(
            (int(item.get("max_n_contacts", 0)) for item in diagnostics),
            default=0,
        ),
        "deepest_penetration_m": min(
            (float(item["min_penetration_m"]) for item in diagnostics
             if item.get("min_penetration_m") is not None),
            default=None,
        ),
    }

    n_episodes = int(record.get("n_episodes", len(record.get("episodes") or [])))
    success_rate = float(record.get("success_rate", 0.0))
    success_rate_se = (
        math.sqrt(max(success_rate * (1.0 - success_rate), 0.0) / n_episodes)
        if n_episodes > 1
        else 0.0
    )
    output = {
        "schema": "contact-study.forward-error.cell.v2",
        "definition": {
            "state_source": "recorded synchronous state/action pairs",
            "rollout_model": "the contact model named by the source cell",
            "reference": (
                "Pinocchio replay from the same state and ctrl, with "
                f"reference mode {args.pin_reference_mode}"
            ),
            "pin_reference_mode": args.pin_reference_mode,
            "duration": "one complete recorded control interval",
            "headline_metric": (
                "object-position L2 metres + SO(3) geodesic radians, unweighted"
            ),
            "planner_distribution_used": False,
            "aggregations": {
                "step_weighted": (
                    "pool all sampled transitions; long episodes contribute "
                    "more observations"
                ),
                "episode_balanced": (
                    "average sampled transitions within each episode, then "
                    "give each episode mean equal weight"
                ),
            },
        },
        "provenance": {
            "repository_commit": _git_commit(),
            "input_path": _portable_path(path),
            "input_sha256": _sha256(path),
        },
        "source_cell": {
            "label": record.get("label"),
            "task": task_name,
            "model": source_model,
            "model_label": contact_cfg.label,
            "geometry": geometry,
            "driver": record.get("driver"),
            "eval_sim": context.get("eval_sim"),
            "n_episodes": n_episodes,
            "n_success": int(record.get("n_success", round(success_rate * n_episodes))),
            "success_rate": success_rate,
            "success_rate_se": success_rate_se,
        },
        "sampling": {
            "stride_control_steps": args.stride,
            "candidate_transitions_after_stride": len(rows_all),
            "max_samples_per_cell": args.max_samples_per_cell,
            "selected_samples": len(rows),
            "limited_subset_selection": (
                "evenly spaced across stride-qualified rows"
                if args.max_samples_per_cell is not None and len(rows_all) > len(rows)
                else "all stride-qualified rows"
            ),
        },
        "time_alignment": {
            "control_dt_s": control_dt,
            "rollout_dt_s": rollout_dt,
            "rollout_steps_per_control": n_rollout_steps,
            "eval_dt_s": eval_dt,
            "eval_substeps_per_rollout": eval_substeps,
            "eval_steps_per_control": n_eval_steps,
        },
        "state_layout": {
            "nq": int(mjm.nq),
            "nv": int(mjm.nv),
            "nu": int(mjm.nu),
            "robot_qpos_adr": robot_qpos_adr,
            "robot_qvel_adr": robot_qvel_adr,
            "object_qpos_adr": object_qpos_adr,
            "object_qvel_adr": object_qvel_adr,
        },
        "runtime": {
            "elapsed_s": elapsed,
            "mean_s_per_sample": elapsed / len(sample_records),
        },
        "forward_error": summaries,
        "episode_balanced_forward_error": episode_balanced_summaries,
        "per_episode": per_episode,
        "pinocchio_diagnostics_summary": pinocchio_diagnostics_summary,
        "samples": sample_records,
    }
    output_path = args.outdir / f"{path.stem}_forward_error.json"
    json_io.dump(output, output_path, precision=12)
    print(
        f"  -> {output_path}  mean headline error="
        f"{summaries['primary_unweighted_m_plus_rad']['mean']:.6g}"
    )
    return output, output_path


def summary_row(output: dict[str, Any], output_path: Path) -> dict[str, Any]:
    source = output["source_cell"]
    sampling = output["sampling"]
    primary = output["forward_error"]["primary_unweighted_m_plus_rad"]
    pos = output["forward_error"]["object_position_m"]
    rot = output["forward_error"]["object_orientation_so3_rad"]
    episode_primary = output["episode_balanced_forward_error"][
        "primary_unweighted_m_plus_rad"
    ]
    return {
        "result_json": _portable_path(output_path),
        "source_label": source.get("label"),
        "task": source["task"],
        "model": source["model"],
        "model_label": source["model_label"],
        "pin_reference_mode": output["definition"]["pin_reference_mode"],
        "geometry": source["geometry"],
        "n_episodes": source["n_episodes"],
        "n_success": source["n_success"],
        "success_rate": source["success_rate"],
        "success_rate_se": source["success_rate_se"],
        "stride_control_steps": sampling["stride_control_steps"],
        "n_forward_error_samples": primary["n"],
        "forward_error_mean_m_plus_rad": primary["mean"],
        "forward_error_sd_m_plus_rad": primary["sd"],
        "forward_error_se_m_plus_rad": primary["se"],
        "forward_error_median_m_plus_rad": primary["median"],
        "forward_error_p95_m_plus_rad": primary["p95"],
        "episode_balanced_n": episode_primary["n"],
        "episode_balanced_forward_error_mean_m_plus_rad": episode_primary["mean"],
        "episode_balanced_forward_error_sd_m_plus_rad": episode_primary["sd"],
        "episode_balanced_forward_error_se_m_plus_rad": episode_primary["se"],
        "episode_balanced_forward_error_median_m_plus_rad": episode_primary["median"],
        "episode_balanced_forward_error_p95_m_plus_rad": episode_primary["p95"],
        "position_error_mean_m": pos["mean"],
        "position_error_sd_m": pos["sd"],
        "orientation_error_mean_rad": rot["mean"],
        "orientation_error_sd_rad": rot["sd"],
        "pinocchio_total_nonconverged_substeps": output[
            "pinocchio_diagnostics_summary"
        ]["total_nonconverged_substeps"],
        "pinocchio_samples_with_nonconvergence": output[
            "pinocchio_diagnostics_summary"
        ]["n_samples_with_nonconvergence"],
        "elapsed_s": output["runtime"]["elapsed_s"],
    }


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("No summary rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="source cell JSON files")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--max-samples-per-cell",
        type=int,
        help="testing limit; chosen evenly across stride-qualified rows",
    )
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--nconmax", type=int, default=200)
    parser.add_argument("--njmax", type=int, default=500)
    parser.add_argument(
        "--pin-reference-mode",
        choices=("history_aligned", "fresh"),
        default="history_aligned",
        help=(
            "preserve reconstructed per-episode Pinocchio solver history "
            "(default), or build a fresh reference simulator per sample"
        ),
    )
    parser.add_argument("--ctrl-reconstruction-tolerance", type=float, default=2e-6)
    parser.add_argument("--summary-csv", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    if args.max_samples_per_cell is not None and args.max_samples_per_cell <= 0:
        raise ValueError("max-samples-per-cell must be positive")
    if args.nconmax <= 0 or args.njmax <= 0:
        raise ValueError("nconmax and njmax must be positive")
    if args.ctrl_reconstruction_tolerance < 0.0:
        raise ValueError("ctrl-reconstruction-tolerance must be non-negative")

    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in args.inputs:
        output, output_path = run_cell(path.resolve(), args)
        rows.append(summary_row(output, output_path))
    summary_path = (
        args.summary_csv.resolve()
        if args.summary_csv is not None
        else args.outdir / "forward_error_summary.csv"
    )
    write_summary_csv(rows, summary_path)
    print(f"Summary CSV -> {summary_path}")


if __name__ == "__main__":
    main()
