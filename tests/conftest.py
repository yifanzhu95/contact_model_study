"""Shared setup for the ContactModelStudy tests.

Run from the repo root::

    python -m pytest tests                 # everything
    python -m pytest tests -m "not slow"   # skip the end-to-end driver runs

Markers:
    gpu:    needs CUDA (MJWarp, ComFree, XPBD, the planner).
    legacy: compares against the old ``contact_study`` package.
    slow:   runs the driver end to end; a minute or more each.

Tests that need something missing (a GPU, Pinocchio, Drake, the old package)
are skipped, not failed.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# Before any mujoco import: offscreen rendering needs a GL backend.
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENES = REPO_ROOT / "scenes" / "leap"
EVAL_CUBE = str(SCENES / "env_leap_eval_cube.xml")
ROLLOUT_CUBE = str(SCENES / "env_leap_rollout_cube_high_high.xml")


def pytest_configure(config):
    for name, doc in (("gpu", "needs CUDA"), ("legacy", "compares against contact_study"),
                      ("slow", "end-to-end driver runs")):
        config.addinivalue_line("markers", f"{name}: {doc}")
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def _has_cuda() -> bool:
    try:
        import warp as wp
        wp.config.quiet = True
        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def _has_legacy() -> bool:
    try:
        import contact_study.contact_models.api  # noqa: F401
        return True
    except Exception:
        return False


HAS_CUDA = _has_cuda()
HAS_LEGACY = _has_legacy()


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="no CUDA device")
    skip_legacy = pytest.mark.skip(reason="old contact_study package not importable")
    for item in items:
        if "gpu" in item.keywords and not HAS_CUDA:
            item.add_marker(skip_gpu)
        if "legacy" in item.keywords and not HAS_LEGACY:
            item.add_marker(skip_legacy)


# -- shared objects ----------------------------------------------------------
@pytest.fixture(scope="session")
def cube_tasks():
    """``(rollout, eval)`` cube tasks at 2 ms, rollout step equal to eval step."""
    from ContactModelStudy.Tasks.CubeReorient import CubeReorient
    from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig
    from ContactModelStudy.Tasks.TaskBase import TaskRole
    ro = CubeReorient(LeapReorientConfig(role=TaskRole.ROLLOUT))
    ev = CubeReorient(LeapReorientConfig(role=TaskRole.EVAL))
    return ro, ev


@pytest.fixture(scope="session")
def cube_initial(cube_tasks):
    """The cube task's ``(q0, v0, u0)``."""
    return cube_tasks[0].getInitialState()


def flexion_actuators(mjm) -> list[int]:
    """Actuators that curl a digit (mcp, pip, dip, ipl), by name."""
    return [a for a in range(mjm.nu) if mjm.actuator(a).name.split("_")[1] in ("mcp", "pip", "dip", "ipl")]


def curl_controls(mjm, u0: np.ndarray, fraction: float, n: int) -> list[np.ndarray]:
    """``n`` commands that curl the fingers ``fraction`` of their range and back."""
    flex = flexion_actuators(mjm)
    tgt = u0.copy()
    tgt[flex] += fraction * (mjm.actuator_ctrlrange[flex, 1] - u0[flex])
    return [u0 + (0.5 - 0.5 * np.cos(2 * np.pi * k / n)) * (tgt - u0) for k in range(n)]


def matches_within_noise(new: np.ndarray, olds: list[np.ndarray]) -> bool:
    """Whether ``new`` (N, nq) agrees with the old backend as well as it agrees with itself.

    Compared per world, by the median over worlds: under heavy random control
    an occasional world branches differently from run to run (the old backend
    does this against itself too), and one such world must not decide the
    test. ``olds`` are repeated runs of the old backend.
    """
    ref = olds[0]
    diff = np.abs(new - ref).max(axis=1)
    noise = np.max([np.abs(o - ref).max(axis=1) for o in olds[1:]], axis=0)
    return bool(np.isfinite(new).all() and np.median(diff) <= max(3 * np.median(noise), 1e-4))
