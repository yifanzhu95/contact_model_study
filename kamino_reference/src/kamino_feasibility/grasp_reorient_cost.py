"""Engine-neutral reference for the current grasp-reorient cost function."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraspReorientCostComponents:
    quaternion: float
    position_x: float
    position_y: float
    position_z: float
    position_l2: float
    object_velocity: float
    fingertip_distance: float
    joint_home: float
    joint_velocity: float
    fallen: float


def grasp_reorient_cost_numpy(
    qpos: np.ndarray,
    qvel: np.ndarray,
    fingertip_positions: np.ndarray,
    *,
    terminal: bool,
    goal: np.ndarray,
    weights: np.ndarray,
    object_qpos_address: int = 16,
    object_qvel_address: int = 16,
    robot_qpos_address: int = 0,
    manipulated_joint_count: int = 16,
) -> tuple[float, GraspReorientCostComponents]:
    """Evaluate the exact arithmetic currently present in the main source.

    The source computes object velocity but currently multiplies ``weights[4]``
    by position L2 rather than by that velocity term.  This function preserves
    that behavior deliberately: a reference comparison must match the live cost
    before deciding whether the apparent source bug should be fixed globally.
    """

    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    tips = np.asarray(fingertip_positions, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if tips.shape != (4, 3):
        raise ValueError(f"fingertip_positions must have shape (4,3), got {tips.shape}")
    if goal.shape != (7 + manipulated_joint_count + 1,):
        raise ValueError("goal has the wrong grasp-reorient layout")
    if weights.shape != (12,):
        raise ValueError("weights must have shape (12,)")
    if object_qpos_address < 0 or object_qpos_address + 7 > qpos.size:
        raise ValueError("object qpos slice is out of bounds")
    if object_qvel_address < 0 or object_qvel_address + 6 > qvel.size:
        raise ValueError("object qvel slice is out of bounds")
    if robot_qpos_address < 0 or robot_qpos_address + manipulated_joint_count > qpos.size:
        raise ValueError("robot qpos slice is out of bounds")
    if manipulated_joint_count > qvel.size:
        raise ValueError("robot qvel slice is out of bounds")
    for name, array in (
        ("qpos", qpos),
        ("qvel", qvel),
        ("fingertip_positions", tips),
        ("goal", goal),
        ("weights", weights),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")

    object_position = qpos[object_qpos_address : object_qpos_address + 3]
    object_quaternion = qpos[object_qpos_address + 3 : object_qpos_address + 7]
    target_position = goal[:3]
    target_quaternion = goal[3:7]
    dot = float(target_quaternion @ object_quaternion)
    quaternion = 1.0 - dot * dot
    position_delta = object_position - target_position
    position_abs = np.abs(position_delta)
    position_l2 = float(np.linalg.norm(position_delta))
    object_velocity = float(
        qvel[object_qvel_address : object_qvel_address + 6]
        @ qvel[object_qvel_address : object_qvel_address + 6]
    )
    fingertip_distance = float(
        np.sum(np.linalg.norm(tips - object_position[np.newaxis, :], axis=1))
    )
    joint_delta = (
        qpos[robot_qpos_address : robot_qpos_address + manipulated_joint_count]
        - goal[7 : 7 + manipulated_joint_count]
    )
    joint_home = float(joint_delta @ joint_delta)
    joint_velocity = float(
        qvel[:manipulated_joint_count] @ qvel[:manipulated_joint_count]
    )
    fallen = float(object_position[2] < goal[7 + manipulated_joint_count])
    components = GraspReorientCostComponents(
        quaternion=quaternion,
        position_x=float(position_abs[0]),
        position_y=float(position_abs[1]),
        position_z=float(position_abs[2]),
        position_l2=position_l2,
        object_velocity=object_velocity,
        fingertip_distance=fingertip_distance,
        joint_home=joint_home,
        joint_velocity=joint_velocity,
        fallen=fallen,
    )
    if terminal:
        cost = (
            weights[9] * quaternion
            + weights[10] * position_l2
            + weights[11] * fallen
        )
    else:
        cost = (
            weights[0] * quaternion
            + weights[1] * position_abs[0]
            + weights[2] * position_abs[1]
            + weights[3] * position_abs[2]
            + weights[4] * position_l2
            + weights[5] * fingertip_distance
            + weights[6] * joint_home
            + weights[7] * joint_velocity
            + weights[8] * fallen
        )
    return float(cost), components
