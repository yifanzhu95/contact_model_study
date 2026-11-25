"""Quaternion helpers, all in MuJoCo's ``(w, x, y, z)`` order."""

from __future__ import annotations

import numpy as np


def matToQuat(R: np.ndarray) -> np.ndarray:
    """wxyz quaternion of a 3x3 rotation matrix (Shepperd's method)."""
    t = np.trace(R)
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def axisAngleQuat(axis, angle: float) -> np.ndarray:
    """wxyz quaternion for a rotation of ``angle`` (rad) about unit ``axis``."""
    c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
    return np.array([c, s * axis[0], s * axis[1], s * axis[2]], dtype=float)


def quatMul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of wxyz quaternions — MuJoCo's ``mju_mulQuat``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])
