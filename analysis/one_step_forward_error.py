#!/usr/bin/env python3
"""Replay one recorded control interval through one rollout contact model.

Scientific definition
---------------------
For a recorded Pinocchio evaluation trajectory, take the state, selected action,
and absolute control command at control step k::

    (q_k, v_k, action_k, u_k) = trajectory.steps[k]

Initialize both a fresh rollout model and a fresh Pinocchio simulator at exactly
(q_k, v_k), hold u_k constant, and advance both for one complete control
interval.  The forward error compares those two fresh replays.  The originally
recorded (q_{k+1}, v_{k+1}) is retained only as a replay audit.

For the current grasp-reorientation experiment this means:

    rollout model:  16 * 0.0040 s = 0.064 s
    Pinocchio eval: 128 * 0.0005 s = 0.064 s

States from both simulations are saved at every aligned rollout checkpoint
(each 0.004 s).  This preserves the full state for future metric changes while
the headline metric follows the current analysis definition: object-position
L2 plus SO(3) orientation error, unweighted. The script deliberately starts with one sample;
it is the validation layer before expanding to many states and M1--M4.

Example
-------
python analysis/one_step_forward_error.py \
  --input results/local_cube_pilot_20260903/cell_00000.json \
  --episode 0 --step 0 --tested-model M1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


# A direct ``python analysis/...py`` invocation otherwise puts analysis/ rather
# than the repository root first on sys.path.  This is especially important on
# machines that also have another editable checkout of contact_study installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402
import warp as wp  # noqa: E402

import contact_study.tasks  # noqa: E402,F401 -- registers task classes
from contact_study.contact_models import api  # noqa: E402
from contact_study.contact_models.config import ContactModelConfig  # noqa: E402
from contact_study.evaluation import json_io  # noqa: E402
from contact_study.tasks.base import TaskRole, get_task  # noqa: E402
from contact_study.tasks.config import EvalSimulatorKind  # noqa: E402


MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
}


def _as_finite_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (size,):
        raise ValueError(f"{name} has shape {arr.shape}; expected ({size},)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return arr


def _integer_ratio(numerator: float, denominator: float, *, name: str) -> int:
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError(
            f"{name} requires positive durations; got {numerator} and {denominator}"
        )
    ratio = numerator / denominator
    nearest = round(ratio)
    if not math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(
            f"{name}: {numerator} / {denominator} = {ratio}, not an integer"
        )
    return int(nearest)


def quaternion_geodesic_error(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Shortest rotation angle between wxyz quaternions, in radians.

    abs(dot) makes q and -q equivalent: they encode the same 3-D rotation.
    """
    q_a = np.asarray(q_a, dtype=np.float64)
    q_b = np.asarray(q_b, dtype=np.float64)
    norm_a = float(np.linalg.norm(q_a))
    norm_b = float(np.linalg.norm(q_b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise ValueError("Cannot compare a zero-norm quaternion")
    dot = float(np.dot(q_a / norm_a, q_b / norm_b))
    return 2.0 * math.acos(float(np.clip(abs(dot), 0.0, 1.0)))


def _l2_and_max_abs(delta: np.ndarray) -> dict[str, float]:
    delta = np.asarray(delta, dtype=np.float64)
    return {
        "l2": float(np.linalg.norm(delta)),
        "max_abs": float(np.max(np.abs(delta))),
    }


def state_errors(
    predicted_qpos: np.ndarray,
    predicted_qvel: np.ndarray,
    reference_qpos: np.ndarray,
    reference_qvel: np.ndarray,
    *,
    object_qpos_adr: int,
    object_qvel_adr: int,
    robot_qpos_adr: int,
    robot_qvel_adr: int,
    nu: int,
) -> dict[str, Any]:
    """Return physically interpretable errors instead of one mixed-state norm."""
    oq = object_qpos_adr
    ov = object_qvel_adr
    rq = robot_qpos_adr
    rv = robot_qvel_adr

    angle_rad = quaternion_geodesic_error(
        predicted_qpos[oq + 3 : oq + 7],
        reference_qpos[oq + 3 : oq + 7],
    )
    return {
        "object_position_m": _l2_and_max_abs(
            predicted_qpos[oq : oq + 3] - reference_qpos[oq : oq + 3]
        ),
        "object_orientation": {
            "geodesic_rad": angle_rad,
            "geodesic_deg": math.degrees(angle_rad),
            "predicted_quaternion_norm": float(
                np.linalg.norm(predicted_qpos[oq + 3 : oq + 7])
            ),
            "reference_quaternion_norm": float(
                np.linalg.norm(reference_qpos[oq + 3 : oq + 7])
            ),
        },
        "object_linear_velocity_m_per_s": _l2_and_max_abs(
            predicted_qvel[ov : ov + 3] - reference_qvel[ov : ov + 3]
        ),
        "object_angular_velocity_rad_per_s": _l2_and_max_abs(
            predicted_qvel[ov + 3 : ov + 6] - reference_qvel[ov + 3 : ov + 6]
        ),
        "hand_joint_position_rad": _l2_and_max_abs(
            predicted_qpos[rq : rq + nu] - reference_qpos[rq : rq + nu]
        ),
        "hand_joint_velocity_rad_per_s": _l2_and_max_abs(
            predicted_qvel[rv : rv + nu] - reference_qvel[rv : rv + nu]
        ),
    }


def _metric_self_checks() -> None:
    """Small independent checks for the two easy-to-miss metric properties."""
    q = np.array([0.9238795325, 0.0, 0.3826834324, 0.0])
    if quaternion_geodesic_error(q, -q) > 1e-12:
        raise AssertionError("Quaternion metric is not invariant to q -> -q")
    q_90 = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    if not math.isclose(
        quaternion_geodesic_error(q_identity, q_90),
        math.pi / 2.0,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise AssertionError("Quaternion metric failed the known 90-degree case")


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Use a repository-relative path when possible, avoiding host-specific paths."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def sample_from_record(
    record: dict[str, Any], episode_index: int, step_index: int
) -> dict[str, Any]:
    """Extract one paired state/action row from an already loaded cell record."""
    episodes = record.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Record contains no episodes")
    if not 0 <= episode_index < len(episodes):
        raise IndexError(
            f"episode {episode_index} is out of range for {len(episodes)} episode(s)"
        )

    episode = episodes[episode_index]
    trajectory = episode.get("trajectory")
    if not isinstance(trajectory, dict):
        raise ValueError("Selected episode has no recorded trajectory")
    if trajectory.get("driver") != "sync":
        raise ValueError(
            "This first validation tool requires a synchronous trajectory; "
            f"got driver={trajectory.get('driver')!r}"
        )

    context = trajectory.get("context", {})
    steps = trajectory.get("steps", {})
    qpos = np.asarray(steps.get("qpos", []), dtype=np.float64)
    qvel = np.asarray(steps.get("qvel", []), dtype=np.float64)
    action = np.asarray(steps.get("action", []), dtype=np.float64)
    ctrl = np.asarray(steps.get("ctrl", []), dtype=np.float64)
    timestamps = np.asarray(steps.get("t", []), dtype=np.float64)
    if any(arr.ndim != 2 for arr in (qpos, qvel, action, ctrl)):
        raise ValueError(
            "qpos, qvel, action, and ctrl must all be two-dimensional arrays"
        )
    if not (len(qpos) == len(qvel) == len(action) == len(ctrl)):
        raise ValueError(
            f"trajectory column lengths differ: qpos={len(qpos)}, "
            f"qvel={len(qvel)}, action={len(action)}, ctrl={len(ctrl)}"
        )
    if timestamps.ndim != 1 or len(timestamps) != len(qpos):
        raise ValueError(
            f"trajectory timestamps must be one-dimensional and match the "
            f"state rows: t={timestamps.shape}, qpos rows={len(qpos)}"
        )
    if not np.isfinite(timestamps).all():
        raise ValueError("trajectory timestamps contain NaN or infinity")
    if not 0 <= step_index < len(qpos) - 1:
        raise IndexError(
            f"step {step_index} has no recorded next state; choose 0..{len(qpos)-2}"
        )

    nq = int(context.get("nq", qpos.shape[1]))
    nv = int(context.get("nv", qvel.shape[1]))
    nu = int(context.get("nu", ctrl.shape[1]))
    sample = {
        "record": record,
        "episode": episode,
        "context": context,
        "nq": nq,
        "nv": nv,
        "nu": nu,
        "q0": _as_finite_vector(qpos[step_index], name="qpos[k]", size=nq),
        "v0": _as_finite_vector(qvel[step_index], name="qvel[k]", size=nv),
        "action": _as_finite_vector(
            action[step_index], name="action[k]", size=nu
        ),
        "ctrl": _as_finite_vector(ctrl[step_index], name="ctrl[k]", size=nu),
        "q_recorded_next": _as_finite_vector(
            qpos[step_index + 1], name="qpos[k+1]", size=nq
        ),
        "v_recorded_next": _as_finite_vector(
            qvel[step_index + 1], name="qvel[k+1]", size=nv
        ),
        "recorded_time_k": float(timestamps[step_index]),
        "recorded_transition_dt": float(
            timestamps[step_index + 1] - timestamps[step_index]
        ),
    }
    return sample


def load_sample(path: Path, episode_index: int, step_index: int) -> dict[str, Any]:
    """Load a cell JSON and extract one paired state/action row."""
    with path.open() as stream:
        record = json.load(stream)
    return sample_from_record(record, episode_index, step_index)


def rollout_trace(
    mjm: mujoco.MjModel,
    model: Any,
    q0: np.ndarray,
    v0: np.ndarray,
    ctrl: np.ndarray,
    n_steps: int,
    *,
    nconmax: int,
    njmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Upload a fresh state and save one state after every rollout step."""
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = q0
    mjd.qvel[:] = v0
    mjd.ctrl[:] = ctrl
    mujoco.mj_forward(mjm, mjd)

    data = api.put_data(
        mjm, mjd, model, nworld=1, nconmax=nconmax, njmax=njmax
    )
    qpos_trace, qvel_trace = [], []
    for _ in range(n_steps):
        api.step(model, data)
        wp.synchronize()
        api.get_data_into(mjm, model, data, mjd)
        qpos_trace.append(mjd.qpos.copy())
        qvel_trace.append(mjd.qvel.copy())
    return np.asarray(qpos_trace), np.asarray(qvel_trace)


def pinocchio_trace(
    task_name: str,
    geometry: str,
    q0: np.ndarray,
    v0: np.ndarray,
    ctrl: np.ndarray,
    n_rollout_steps: int,
    eval_substeps_per_rollout: int,
    expected_eval_dt: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fresh Pinocchio replay sampled at rollout-aligned checkpoints."""
    eval_task = get_task(task_name, geometry=geometry, role=TaskRole.EVAL)
    eval_task.load()
    eval_task.config.eval_sim = EvalSimulatorKind.PINOCCHIO
    sim = eval_task.make_eval_simulator(video_path=None, render=False)
    if not math.isclose(
        float(sim.timestep), expected_eval_dt, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError(
            f"Pinocchio timestep={sim.timestep}, expected eval_dt={expected_eval_dt}"
        )

    sim.reset(q0.copy(), v0.copy())
    sim.apply_control(ctrl.copy())
    qpos_trace, qvel_trace = [], []
    for _ in range(n_rollout_steps):
        sim.step(eval_substeps_per_rollout)
        state = sim.get_state()
        qpos_trace.append(np.asarray(state.qpos, dtype=np.float64).copy())
        qvel_trace.append(np.asarray(state.qvel, dtype=np.float64).copy())
    diagnostics = sim.diagnostics() if hasattr(sim, "diagnostics") else {}
    return np.asarray(qpos_trace), np.asarray(qvel_trace), diagnostics


def _contact_count(mjm: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray) -> int:
    """MuJoCo collision-detection count used only as sample metadata."""
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = qpos
    mjd.qvel[:] = qvel
    mujoco.mj_forward(mjm, mjd)
    return int(mjd.ncon)


def _primary_object_error(errors: dict[str, Any]) -> dict[str, float | str]:
    """Primary descriptive metric: position L2 + SO(3) angle, weight 1 each."""
    pos = float(errors["object_position_m"]["l2"])
    rot = float(errors["object_orientation"]["geodesic_rad"])
    return {
        "object_position_l2_m": pos,
        "object_orientation_so3_rad": rot,
        "unweighted_sum_m_plus_rad": pos + rot,
        "units_note": (
            "This is an unweighted descriptive sum of metres and radians; "
            "its components are retained separately."
        ),
    }


def _checkpoint_object_errors(
    rollout_qpos: np.ndarray,
    pinocchio_qpos: np.ndarray,
    object_qpos_adr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oq = object_qpos_adr
    pos = np.linalg.norm(
        rollout_qpos[:, oq : oq + 3] - pinocchio_qpos[:, oq : oq + 3], axis=1
    )
    rot = np.asarray(
        [
            quaternion_geodesic_error(a[oq + 3 : oq + 7], b[oq + 3 : oq + 7])
            for a, b in zip(rollout_qpos, pinocchio_qpos)
        ],
        dtype=np.float64,
    )
    return pos, rot, pos + rot


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    _metric_self_checks()

    input_path = args.input.resolve()
    sample = load_sample(input_path, args.episode, args.step)
    context = sample["context"]

    task_name = str(context.get("task", sample["record"].get("task", "")))
    geometry = str(context.get("geometry", sample["record"].get("geometry", "")))
    if not task_name or not geometry:
        raise ValueError("Recorded task and geometry are required")

    control_dt = float(context["control_dt"])
    rollout_dt = float(context["rollout_dt"])
    eval_dt = float(context["eval_dt"])
    eval_substeps_per_rollout = int(context["eval_substeps_per_rollout"])
    if not math.isclose(
        sample["recorded_transition_dt"],
        control_dt,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "The selected recorded rows are not one control interval apart: "
            f"t[k+1]-t[k]={sample['recorded_transition_dt']}, "
            f"control_dt={control_dt}"
        )
    n_rollout_steps = _integer_ratio(
        control_dt, rollout_dt, name="control_dt / rollout_dt"
    )
    n_eval_steps = _integer_ratio(control_dt, eval_dt, name="control_dt / eval_dt")
    if n_rollout_steps != int(context["substeps"]):
        raise ValueError(
            f"computed {n_rollout_steps} rollout steps but context says "
            f"substeps={context['substeps']}"
        )
    if n_eval_steps != int(context["eval_steps_per_control"]):
        raise ValueError(
            f"computed {n_eval_steps} eval steps but context says "
            f"eval_steps_per_control={context['eval_steps_per_control']}"
        )
    if n_eval_steps != n_rollout_steps * eval_substeps_per_rollout:
        raise ValueError(
            "Timing hierarchy is inconsistent: eval steps per control must equal "
            "rollout steps per control times eval substeps per rollout"
        )

    task = get_task(task_name, geometry=geometry, role=TaskRole.ROLLOUT)
    mjm, _ = task.load()
    mjm.opt.timestep = rollout_dt
    if (mjm.nq, mjm.nv, mjm.nu) != (sample["nq"], sample["nv"], sample["nu"]):
        raise ValueError(
            "Recorded/model dimensions disagree: "
            f"recorded={(sample['nq'], sample['nv'], sample['nu'])}, "
            f"model={(mjm.nq, mjm.nv, mjm.nu)}"
        )

    index_vector = np.asarray(task.index_vector, dtype=int)
    object_qpos_adr = int(index_vector[0])
    object_qvel_adr = int(index_vector[1])
    robot_qpos_adr = int(context["robot_qpos_adr"])
    # The Leap-hand model has its 16 joint velocities first; the object
    # free-joint starts at object_qvel_adr.  Make this assumption explicit and
    # fail rather than silently slicing the wrong state on a future model.
    robot_qvel_adr = 0
    if object_qpos_adr != robot_qpos_adr + sample["nu"]:
        raise ValueError("Unexpected qpos layout: hand joints do not precede object")
    if object_qvel_adr != robot_qvel_adr + sample["nu"]:
        raise ValueError("Unexpected qvel layout: hand joints do not precede object")

    # The planner records both its selected delta action and the absolute command
    # actually sent to the simulator.  Dynamics must receive the absolute ctrl;
    # the reconstruction check proves that the paired action/state was read from
    # the same row and interpreted with the original controller semantics.
    ctrl_reconstruction_max_abs = None
    if bool(context.get("ctrl_relative_to_qpos", True)):
        reconstructed_ctrl = (
            sample["q0"][robot_qpos_adr : robot_qpos_adr + sample["nu"]]
            + sample["action"]
        )
        clip_lo, clip_hi = context.get("clip_lo"), context.get("clip_hi")
        if clip_lo is not None:
            reconstructed_ctrl = np.clip(reconstructed_ctrl, clip_lo, clip_hi)
        ctrl_reconstruction_max_abs = float(
            np.max(np.abs(reconstructed_ctrl - sample["ctrl"]))
        )
        # The source JSON was rounded to seven significant digits, so exact
        # equality is neither expected nor desirable as a validity condition.
        if ctrl_reconstruction_max_abs > args.ctrl_reconstruction_tolerance:
            raise ValueError(
                "Recorded state/action/ctrl identity failed: "
                f"max|q_hand+action-ctrl|={ctrl_reconstruction_max_abs:.3e}, "
                f"tol={args.ctrl_reconstruction_tolerance:.3e}"
            )

    cfg = MODEL_FACTORIES[args.tested_model]()
    model_a = api.put_model(mjm, cfg)

    rollout_qpos_a, rollout_qvel_a = rollout_trace(
        mjm,
        model_a,
        sample["q0"],
        sample["v0"],
        sample["ctrl"],
        n_rollout_steps,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    # Rebuild both device model and data.  If the second trace differs, the
    # discrepancy is backend/runtime nondeterminism rather than mutable data
    # accidentally carried over from the first trace.
    model_b = api.put_model(mjm, cfg)
    rollout_qpos_b, rollout_qvel_b = rollout_trace(
        mjm,
        model_b,
        sample["q0"],
        sample["v0"],
        sample["ctrl"],
        n_rollout_steps,
        nconmax=args.nconmax,
        njmax=args.njmax,
    )
    repeat_qpos_max_abs = float(np.max(np.abs(rollout_qpos_a - rollout_qpos_b)))
    repeat_qvel_max_abs = float(np.max(np.abs(rollout_qvel_a - rollout_qvel_b)))
    repeat_qpos_tol = float(args.repeat_qpos_tolerance)
    repeat_qvel_tol = float(args.repeat_qvel_tolerance)
    if (
        repeat_qpos_max_abs > repeat_qpos_tol
        or repeat_qvel_max_abs > repeat_qvel_tol
    ):
        raise RuntimeError(
            "Fresh-state repeatability check failed: "
            f"qpos max|delta|={repeat_qpos_max_abs:.3e}, "
            f"qvel max|delta|={repeat_qvel_max_abs:.3e}, "
            f"tolerances=({repeat_qpos_tol:.3e}, {repeat_qvel_tol:.3e})"
        )

    pin_qpos_a, pin_qvel_a, pin_diag_a = pinocchio_trace(
        task_name,
        geometry,
        sample["q0"],
        sample["v0"],
        sample["ctrl"],
        n_rollout_steps,
        eval_substeps_per_rollout,
        eval_dt,
    )
    pin_qpos_b, pin_qvel_b, pin_diag_b = pinocchio_trace(
        task_name,
        geometry,
        sample["q0"],
        sample["v0"],
        sample["ctrl"],
        n_rollout_steps,
        eval_substeps_per_rollout,
        eval_dt,
    )
    pin_repeat_qpos_max_abs = float(np.max(np.abs(pin_qpos_a - pin_qpos_b)))
    pin_repeat_qvel_max_abs = float(np.max(np.abs(pin_qvel_a - pin_qvel_b)))
    if (
        pin_repeat_qpos_max_abs > args.pin_repeat_qpos_tolerance
        or pin_repeat_qvel_max_abs > args.pin_repeat_qvel_tolerance
    ):
        raise RuntimeError(
            "Fresh Pinocchio repeatability check failed: "
            f"qpos max|delta|={pin_repeat_qpos_max_abs:.3e}, "
            f"qvel max|delta|={pin_repeat_qvel_max_abs:.3e}, "
            f"tolerances=({args.pin_repeat_qpos_tolerance:.3e}, "
            f"{args.pin_repeat_qvel_tolerance:.3e})"
        )

    errors = state_errors(
        rollout_qpos_a[-1],
        rollout_qvel_a[-1],
        pin_qpos_a[-1],
        pin_qvel_a[-1],
        object_qpos_adr=object_qpos_adr,
        object_qvel_adr=object_qvel_adr,
        robot_qpos_adr=robot_qpos_adr,
        robot_qvel_adr=robot_qvel_adr,
        nu=sample["nu"],
    )
    primary_error = _primary_object_error(errors)
    pin_rerun_vs_recorded = state_errors(
        pin_qpos_a[-1],
        pin_qvel_a[-1],
        sample["q_recorded_next"],
        sample["v_recorded_next"],
        object_qpos_adr=object_qpos_adr,
        object_qvel_adr=object_qvel_adr,
        robot_qpos_adr=robot_qpos_adr,
        robot_qvel_adr=robot_qvel_adr,
        nu=sample["nu"],
    )
    rollout_vs_recorded = state_errors(
        rollout_qpos_a[-1],
        rollout_qvel_a[-1],
        sample["q_recorded_next"],
        sample["v_recorded_next"],
        object_qpos_adr=object_qpos_adr,
        object_qvel_adr=object_qvel_adr,
        robot_qpos_adr=robot_qpos_adr,
        robot_qvel_adr=robot_qvel_adr,
        nu=sample["nu"],
    )
    checkpoint_pos, checkpoint_rot, checkpoint_combined = _checkpoint_object_errors(
        rollout_qpos_a, pin_qpos_a, object_qpos_adr
    )

    source_model = str(
        sample["record"].get("model", context.get("model_label", "unknown"))
    )
    result = {
        "schema": "contact-study.forward-error.single-sample.v2",
        "definition": {
            "reference": (
                "fresh Pinocchio replay initialized from recorded state k and "
                "advanced under recorded absolute ctrl[k]"
            ),
            "prediction": (
                "fresh rollout-model state initialized at k, with recorded "
                "absolute ctrl[k] held for one complete control interval"
            ),
            "recorded_next_state_role": (
                "audit only: checks whether a fresh Pinocchio replay reproduces "
                "the next state seen during the original closed-loop episode"
            ),
            "headline_metric": (
                "object position L2 in metres plus object SO(3) geodesic angle "
                "in radians, with coefficient 1 on each term"
            ),
            "shared_state_convention": (
                "MuJoCo index layout; object position in world frame, quaternion "
                "wxyz, linear velocity in world frame, angular velocity in the "
                "object/body-local frame"
            ),
        },
        "provenance": {
            "repository_commit": _git_commit(),
            "input_path": _portable_path(input_path),
            "input_sha256": _sha256(input_path),
            "source_task": task_name,
            "source_geometry": geometry,
            "source_model": source_model,
            "source_model_label": context.get("model_label"),
            "source_driver": sample["record"].get("driver", context.get("driver")),
            "source_eval_sim": context.get("eval_sim"),
            "source_episode_index": int(args.episode),
            "source_episode_end_reason": sample["episode"].get("end_reason"),
            "source_episode_success": bool(sample["episode"].get("success", False)),
            "source_step_index": int(args.step),
            "source_state_precision_significant_digits": context.get("precision"),
        },
        "tested_rollout_model": {
            "short_name": args.tested_model,
            "label": cfg.label,
            "backend": cfg.backend.value,
            "nconmax": int(args.nconmax),
            "njmax": int(args.njmax),
        },
        "time_alignment": {
            "control_dt_s": control_dt,
            "recorded_t_k_s": sample["recorded_time_k"],
            "recorded_t_k_plus_1_minus_t_k_s": sample["recorded_transition_dt"],
            "rollout_dt_s": rollout_dt,
            "rollout_steps": n_rollout_steps,
            "rollout_advanced_time_s": n_rollout_steps * rollout_dt,
            "eval_dt_s": eval_dt,
            "eval_substeps_per_rollout_step": eval_substeps_per_rollout,
            "eval_steps_in_recorded_transition": n_eval_steps,
            "eval_advanced_time_s": n_eval_steps * eval_dt,
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
        "sample_metadata": {
            "recorded_time_k_s": sample["recorded_time_k"],
            "action_semantics": (
                "selected joint-position delta relative to measured hand qpos"
                if context.get("ctrl_relative_to_qpos", True)
                else "selected delta accumulated onto the previous command"
            ),
            "simulator_input": "recorded absolute ctrl[k]",
            "ctrl_reconstruction_max_abs": ctrl_reconstruction_max_abs,
            "mujoco_rollout_geometry_contacts_at_k": _contact_count(
                mjm, sample["q0"], sample["v0"]
            ),
            "mujoco_rollout_geometry_contacts_at_recorded_k_plus_1": _contact_count(
                mjm, sample["q_recorded_next"], sample["v_recorded_next"]
            ),
            "mujoco_rollout_geometry_contacts_at_pinocchio_rerun_end": _contact_count(
                mjm, pin_qpos_a[-1], pin_qvel_a[-1]
            ),
        },
        "validation": {
            "metric_self_checks_passed": True,
            "dimensions_match": True,
            "time_alignment_exact": True,
            "recorded_action_ctrl_identity_checked": (
                ctrl_reconstruction_max_abs is not None
            ),
            "recorded_action_ctrl_tolerance": args.ctrl_reconstruction_tolerance,
            "fresh_rollout_qpos_tolerance": repeat_qpos_tol,
            "fresh_rollout_qvel_tolerance": repeat_qvel_tol,
            "fresh_rollout_qpos_max_abs_difference": repeat_qpos_max_abs,
            "fresh_rollout_qvel_max_abs_difference": repeat_qvel_max_abs,
            "fresh_rollout_repeatability_passed": True,
            "fresh_pinocchio_qpos_tolerance": args.pin_repeat_qpos_tolerance,
            "fresh_pinocchio_qvel_tolerance": args.pin_repeat_qvel_tolerance,
            "fresh_pinocchio_qpos_max_abs_difference": pin_repeat_qpos_max_abs,
            "fresh_pinocchio_qvel_max_abs_difference": pin_repeat_qvel_max_abs,
            "fresh_pinocchio_repeatability_passed": True,
        },
        "primary_error": primary_error,
        "errors": errors,
        "audit_errors": {
            "fresh_pinocchio_vs_original_recorded_next": pin_rerun_vs_recorded,
            "rollout_vs_original_recorded_next": rollout_vs_recorded,
        },
        "pinocchio_diagnostics": {
            "first_fresh_replay": pin_diag_a,
            "second_fresh_replay": pin_diag_b,
        },
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
            "pinocchio_rerun_qpos_end": json_io.compact(pin_qpos_a[-1], precision=0),
            "pinocchio_rerun_qvel_end": json_io.compact(pin_qvel_a[-1], precision=0),
            "rollout_qpos_end": json_io.compact(rollout_qpos_a[-1], precision=0),
            "rollout_qvel_end": json_io.compact(rollout_qvel_a[-1], precision=0),
        },
        "aligned_checkpoints": {
            "definition": (
                "checkpoint i is after (i+1) rollout steps and after the same "
                "physical time in Pinocchio"
            ),
            "time_from_k_s": json_io.compact(
                np.arange(1, n_rollout_steps + 1, dtype=np.float64) * rollout_dt,
                precision=0,
            ),
            "rollout_qpos": json_io.compact(rollout_qpos_a, precision=0),
            "rollout_qvel": json_io.compact(rollout_qvel_a, precision=0),
            "pinocchio_qpos": json_io.compact(pin_qpos_a, precision=0),
            "pinocchio_qvel": json_io.compact(pin_qvel_a, precision=0),
            "object_position_error_m": json_io.compact(checkpoint_pos, precision=0),
            "object_orientation_so3_error_rad": json_io.compact(
                checkpoint_rot, precision=0
            ),
            "unweighted_object_error_m_plus_rad": json_io.compact(
                checkpoint_combined, precision=0
            ),
        },
    }

    if args.output is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_episode{args.episode:02d}_step{args.step:05d}_"
            f"test{args.tested_model}_forward_error.json"
        )
    else:
        output_path = args.output.resolve()
    json_io.dump(result, output_path, precision=12)
    return result, output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="recorded cell JSON")
    parser.add_argument("--episode", type=int, default=0, help="episode index")
    parser.add_argument("--step", type=int, default=0, help="control-step index k")
    parser.add_argument(
        "--tested-model", choices=tuple(MODEL_FACTORIES), default="M1"
    )
    parser.add_argument("--nconmax", type=int, default=200)
    parser.add_argument("--njmax", type=int, default=500)
    parser.add_argument(
        "--ctrl-reconstruction-tolerance",
        type=float,
        default=2e-6,
        help="allowed JSON-rounding residual in q_hand + action = ctrl",
    )
    parser.add_argument(
        "--repeat-qpos-tolerance",
        type=float,
        default=1e-6,
        help="maximum qpos difference between two fresh identical rollouts",
    )
    parser.add_argument(
        "--repeat-qvel-tolerance",
        type=float,
        default=1e-4,
        help="maximum qvel difference between two fresh identical rollouts",
    )
    parser.add_argument(
        "--pin-repeat-qpos-tolerance",
        type=float,
        default=1e-10,
        help="maximum qpos difference between two fresh identical Pinocchio replays",
    )
    parser.add_argument(
        "--pin-repeat-qvel-tolerance",
        type=float,
        default=1e-8,
        help="maximum qvel difference between two fresh identical Pinocchio replays",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.nconmax <= 0 or args.njmax <= 0:
        raise ValueError("nconmax and njmax must be positive")
    tolerances = (
        args.ctrl_reconstruction_tolerance,
        args.repeat_qpos_tolerance,
        args.repeat_qvel_tolerance,
        args.pin_repeat_qpos_tolerance,
        args.pin_repeat_qvel_tolerance,
    )
    if any(value < 0.0 for value in tolerances):
        raise ValueError("validation tolerances must be non-negative")
    result, output_path = run(args)
    errors = result["errors"]
    validation = result["validation"]
    timing = result["time_alignment"]
    print("Single-sample forward replay completed")
    print(f"  output: {output_path}")
    print(
        "  matched time: "
        f"{timing['rollout_steps']} x {timing['rollout_dt_s']:.6f} s = "
        f"{timing['rollout_advanced_time_s']:.6f} s"
    )
    print(
        "  object error: "
        f"position={errors['object_position_m']['l2']:.6e} m, "
        f"orientation={errors['object_orientation']['geodesic_rad']:.6e} rad "
        f"({errors['object_orientation']['geodesic_deg']:.6e} deg)"
    )
    print(
        "  headline unweighted position+orientation error: "
        f"{result['primary_error']['unweighted_sum_m_plus_rad']:.6e} m+rad"
    )
    print(
        "  object velocity error: "
        f"linear={errors['object_linear_velocity_m_per_s']['l2']:.6e} m/s, "
        f"angular={errors['object_angular_velocity_rad_per_s']['l2']:.6e} rad/s"
    )
    print(
        "  hand error: "
        f"position={errors['hand_joint_position_rad']['l2']:.6e} rad, "
        f"velocity={errors['hand_joint_velocity_rad_per_s']['l2']:.6e} rad/s"
    )
    print(
        "  rollout repeatability max|difference|: "
        f"qpos={validation['fresh_rollout_qpos_max_abs_difference']:.3e}, "
        f"qvel={validation['fresh_rollout_qvel_max_abs_difference']:.3e}"
    )
    print(
        "  Pinocchio repeatability max|difference|: "
        f"qpos={validation['fresh_pinocchio_qpos_max_abs_difference']:.3e}, "
        f"qvel={validation['fresh_pinocchio_qvel_max_abs_difference']:.3e}"
    )
    audit = result["audit_errors"]["fresh_pinocchio_vs_original_recorded_next"]
    print(
        "  fresh Pinocchio vs original recorded next state: "
        f"position={audit['object_position_m']['l2']:.6e} m, "
        f"orientation={audit['object_orientation']['geodesic_rad']:.6e} rad"
    )


if __name__ == "__main__":
    main()
