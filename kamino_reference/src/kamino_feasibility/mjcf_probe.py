"""Reproducible MJCF-to-Kamino compatibility probe.

This module deliberately stays outside ``contact_model_study``.  It answers a
narrow question before integration work starts: can a target MJCF be imported,
stepped by SolverKamino, observed, and controlled through Newton's actual
control interface?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import newton
import warp as wp


@dataclass(frozen=True)
class PathRemap:
    """One asset path correction made while importing an MJCF scene."""

    requested: str
    resolved: str


@dataclass(frozen=True)
class ProbeReport:
    """Host-side evidence produced by :func:`probe_scene`."""

    scene: str
    device: str
    world_count: int
    body_count: int
    joint_count: int
    joint_dof_count: int
    joint_coord_count: int
    shape_count: int
    mujoco_actuator_count: int
    position_target_indices: tuple[int, ...]
    collision_pipeline: str
    path_remaps: tuple[PathRemap, ...]
    one_step_finite: bool
    default_q0_delta: float
    mujoco_ctrl_q0_delta: float
    joint_target_q0_delta: float


class ScenePathResolver:
    """Resolve MJCF resources and record a narrow scene-local objects fallback.

    Newton 1.6 expands included ``compiler meshdir`` declarations differently
    from MuJoCo for the target Leap-Hand scene.  Normal paths are untouched.
    If the expanded path is a missing ``objects/<name>`` resource, this resolver
    checks ``<scene-dir>/objects/<name>`` and records the correction.
    """

    def __init__(self, scene: str | Path):
        self.scene = Path(scene).expanduser().resolve()
        self.remaps: list[PathRemap] = []

    def __call__(self, base_dir: str | None, file_path: str) -> str:
        base = Path(base_dir) if base_dir is not None else self.scene.parent
        candidate = (base / file_path).resolve()
        if candidate.exists():
            return str(candidate)

        alternate = (self.scene.parent / "objects" / candidate.name).resolve()
        if candidate.parent.name == "objects" and alternate.exists():
            remap = PathRemap(str(candidate), str(alternate))
            if remap not in self.remaps:
                self.remaps.append(remap)
            return str(alternate)
        return str(candidate)


def infer_position_target_indices(model: Any) -> tuple[int, ...]:
    """Return scalar joint-coordinate indices controlled as position targets.

    The conversion is intentionally conservative.  It accepts only scalar
    joints whose imported target mode contains the POSITION bit.  Free or ball
    joints need an explicit actuator mapping and are therefore rejected here.
    """

    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    modes = model.joint_target_mode.numpy()
    position_bit = int(newton.JointTargetMode.POSITION)
    indices: list[int] = []

    for joint_index in range(model.joint_count):
        q_begin = int(q_start[joint_index])
        q_end = (
            int(q_start[joint_index + 1])
            if joint_index + 1 < len(q_start)
            else int(model.joint_coord_count)
        )
        qd_begin = int(qd_start[joint_index])
        qd_end = (
            int(qd_start[joint_index + 1])
            if joint_index + 1 < len(qd_start)
            else int(model.joint_dof_count)
        )
        if q_end - q_begin != 1 or qd_end - qd_begin != 1:
            continue
        if int(modes[qd_begin]) & position_bit:
            indices.append(q_begin)

    return tuple(indices)


def _run_control_case(
    *,
    model: Any,
    solver: Any,
    target_index: int,
    case: str,
    dt: float,
    steps: int,
    target_value: float,
) -> tuple[float, bool]:
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    control.clear(model)

    if case == "mujoco_ctrl":
        values = np.zeros(control.mujoco.ctrl.shape[0], dtype=np.float32)
        values[0] = target_value
        control.mujoco.ctrl.assign(values)
    elif case == "joint_target":
        values = control.joint_target_q.numpy()
        values[target_index] = target_value
        control.joint_target_q.assign(values)
    elif case != "default":
        raise ValueError(f"Unknown control probe case: {case}")

    solver.reset(state_0)
    initial_q = float(state_0.joint_q.numpy()[target_index])
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    arrays = (state_0.body_q, state_0.body_qd, state_0.joint_q, state_0.joint_qd)
    finite = all(np.isfinite(array.numpy()).all() for array in arrays if array is not None)
    final_q = float(state_0.joint_q.numpy()[target_index])
    return final_q - initial_q, bool(finite)


def probe_scene(
    scene: str | Path,
    *,
    device: str = "cuda:0",
    dt: float = 0.002,
    control_steps: int = 10,
    target_value: float = 0.5,
) -> ProbeReport:
    """Import a target scene and exercise SolverKamino's real control path."""

    scene_path = Path(scene).expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"MJCF scene does not exist: {scene_path}")
    if control_steps < 1:
        raise ValueError("control_steps must be positive")

    wp.init()
    resolver = ScenePathResolver(scene_path)
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.add_mjcf(str(scene_path), path_resolver=resolver)
        model = builder.finalize()

        config = newton.solvers.SolverKamino.Config.from_model(model)
        collision_pipeline = str(config.collision_detector.pipeline)
        solver = newton.solvers.SolverKamino(model, config=config)

        target_indices = infer_position_target_indices(model)
        actuator_count = len(model.mujoco.actuator_label)
        if actuator_count == 0:
            raise RuntimeError("Target MJCF did not import any MuJoCo actuators")
        if len(target_indices) != actuator_count:
            raise RuntimeError(
                "Cannot infer a one-to-one scalar position-target mapping: "
                f"{actuator_count} actuators versus {len(target_indices)} targets"
            )

        target_index = target_indices[0]
        deltas: dict[str, float] = {}
        finite_by_case: list[bool] = []
        for case in ("default", "mujoco_ctrl", "joint_target"):
            delta, finite = _run_control_case(
                model=model,
                solver=solver,
                target_index=target_index,
                case=case,
                dt=dt,
                steps=control_steps,
                target_value=target_value,
            )
            deltas[case] = delta
            finite_by_case.append(finite)

    return ProbeReport(
        scene=str(scene_path),
        device=device,
        world_count=int(model.world_count),
        body_count=int(model.body_count),
        joint_count=int(model.joint_count),
        joint_dof_count=int(model.joint_dof_count),
        joint_coord_count=int(model.joint_coord_count),
        shape_count=int(model.shape_count),
        mujoco_actuator_count=actuator_count,
        position_target_indices=target_indices,
        collision_pipeline=collision_pipeline,
        path_remaps=tuple(resolver.remaps),
        one_step_finite=all(finite_by_case),
        default_q0_delta=deltas["default"],
        mujoco_ctrl_q0_delta=deltas["mujoco_ctrl"],
        joint_target_q0_delta=deltas["joint_target"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path, help="MJCF scene to probe")
    parser.add_argument("--device", default="cuda:0", help="Warp device")
    parser.add_argument("--dt", default=0.002, type=float, help="Physics time step in seconds")
    parser.add_argument("--control-steps", default=10, type=int)
    parser.add_argument("--target-value", default=0.5, type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = probe_scene(
        args.scene,
        device=args.device,
        dt=args.dt,
        control_steps=args.control_steps,
        target_value=args.target_value,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
