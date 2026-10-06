"""Resolve initial free-body penetrations with a minimum translation.

This module is deliberately MuJoCo-side preprocessing.  It constructs a clean
initial state before that same state is passed to MuJoCo and Kamino; it does not
change either engine's contact solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import mujoco
import numpy as np


@dataclass(frozen=True)
class PenetrationResolution:
    """Summary of a translational free-body penetration correction."""

    joint_name: str
    clearance_m: float
    translation_m: tuple[float, float, float]
    passes: int
    initial_penetrating_records: int
    initial_deepest_distance_m: float | None
    final_penetrating_records: int
    final_deepest_distance_m: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def minimum_norm_halfspace_translation(
    normals: np.ndarray,
    lower_bounds: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Solve ``min ||x||`` subject to ``normals @ x >= lower_bounds``.

    The unknown is three-dimensional.  At a feasible closest point, no more
    than three linearly independent inequalities need be active, so enumerating
    active sets of size one through three is exact for this small problem.
    """

    a = np.asarray(normals, dtype=np.float64)
    b = np.asarray(lower_bounds, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"normals must have shape (n, 3), got {a.shape}")
    if b.shape != (a.shape[0],):
        raise ValueError(f"lower_bounds must have shape {(a.shape[0],)}, got {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("halfspace inputs contain NaN or Inf")
    if a.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)

    lengths = np.linalg.norm(a, axis=1)
    if np.any(lengths <= tolerance):
        raise ValueError("constraint normals must be nonzero")
    a = a / lengths[:, None]
    b = b / lengths

    # Collision manifolds often repeat one normal at several witness points.
    # Keep only the strictest lower bound for each numerically identical normal.
    reduced: dict[tuple[float, float, float], tuple[np.ndarray, float]] = {}
    for row, bound in zip(a, b, strict=True):
        key = tuple(np.round(row, decimals=10))
        previous = reduced.get(key)
        if previous is None or bound > previous[1]:
            reduced[key] = (row, float(bound))
    a = np.stack([item[0] for item in reduced.values()])
    b = np.asarray([item[1] for item in reduced.values()], dtype=np.float64)

    candidates: list[np.ndarray] = []
    zero = np.zeros(3, dtype=np.float64)
    if np.all(a @ zero >= b - tolerance):
        candidates.append(zero)

    for active_count in range(1, min(3, len(b)) + 1):
        for active in combinations(range(len(b)), active_count):
            active_a = a[np.asarray(active)]
            active_b = b[np.asarray(active)]
            gram = active_a @ active_a.T
            multipliers, _, rank, _ = np.linalg.lstsq(gram, active_b, rcond=None)
            if rank < active_count or np.any(multipliers < -tolerance):
                continue
            candidate = active_a.T @ multipliers
            if np.all(a @ candidate >= b - tolerance):
                candidates.append(candidate)

    if not candidates:
        raise RuntimeError("penetration halfspaces are infeasible for translation-only correction")
    return min(candidates, key=lambda item: float(item @ item))


def _free_joint_details(model: mujoco.MjModel, joint_name: str) -> tuple[int, int, set[int]]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"unknown MuJoCo joint: {joint_name}")
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"joint {joint_name!r} is not a free joint")

    qpos_address = int(model.jnt_qposadr[joint_id])
    root_body = int(model.jnt_bodyid[joint_id])
    moving_bodies = {root_body}
    for body_id in range(1, model.nbody):
        ancestor = body_id
        while ancestor > 0:
            ancestor = int(model.body_parentid[ancestor])
            if ancestor == root_body:
                moving_bodies.add(body_id)
                break
    return qpos_address, root_body, moving_bodies


def _penetration_constraints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    moving_bodies: set[int],
    clearance_m: float,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    normals: list[np.ndarray] = []
    bounds: list[float] = []
    distances: list[float] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1_moves = int(model.geom_bodyid[geom1]) in moving_bodies
        body2_moves = int(model.geom_bodyid[geom2]) in moving_bodies
        if body1_moves == body2_moves or float(contact.dist) >= clearance_m:
            continue

        # MuJoCo's contact normal points from geom1 toward geom2.  Reverse it
        # when geom1 belongs to the body that we translate.
        normal = np.asarray(contact.frame[:3], dtype=np.float64)
        normals.append(-normal if body1_moves else normal)
        bounds.append(clearance_m - float(contact.dist))
        distances.append(float(contact.dist))

    if not normals:
        return np.empty((0, 3)), np.empty(0), distances
    return np.stack(normals), np.asarray(bounds), distances


def resolve_free_joint_penetration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    *,
    clearance_m: float = 5.0e-4,
    max_passes: int = 8,
) -> PenetrationResolution:
    """Translate one free-joint subtree until its contacts have clearance.

    Orientation, joint velocities, and every other generalized coordinate are
    preserved.  Repeated passes handle newly exposed collision features after
    the local linearized correction.
    """

    if not np.isfinite(clearance_m) or clearance_m < 0.0:
        raise ValueError("clearance_m must be finite and non-negative")
    if max_passes < 1:
        raise ValueError("max_passes must be positive")

    qpos_address, _, moving_bodies = _free_joint_details(model, joint_name)
    mujoco.mj_forward(model, data)
    initial_normals, _, initial_distances = _penetration_constraints(
        model, data, moving_bodies, 0.0
    )
    del initial_normals
    total_translation = np.zeros(3, dtype=np.float64)
    passes = 0

    for passes in range(1, max_passes + 1):
        normals, bounds, _ = _penetration_constraints(
            model, data, moving_bodies, clearance_m
        )
        if len(bounds) == 0:
            passes -= 1
            break
        correction = minimum_norm_halfspace_translation(normals, bounds)
        data.qpos[qpos_address : qpos_address + 3] += correction
        total_translation += correction
        mujoco.mj_forward(model, data)
    else:
        normals, _, _ = _penetration_constraints(model, data, moving_bodies, clearance_m)
        if len(normals):
            raise RuntimeError(
                f"failed to resolve {joint_name!r} penetration in {max_passes} passes"
            )

    _, _, final_distances = _penetration_constraints(model, data, moving_bodies, 0.0)
    return PenetrationResolution(
        joint_name=joint_name,
        clearance_m=float(clearance_m),
        translation_m=tuple(float(value) for value in total_translation),
        passes=passes,
        initial_penetrating_records=len(initial_distances),
        initial_deepest_distance_m=min(initial_distances) if initial_distances else None,
        final_penetrating_records=len(final_distances),
        final_deepest_distance_m=min(final_distances) if final_distances else None,
    )
