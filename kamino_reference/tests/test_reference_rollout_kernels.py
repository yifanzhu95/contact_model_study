"""CPU checks for the semantic bridge used by the future CUDA rollout."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from kamino_feasibility.batched_target_rollout import (
    _accumulate_grasp_reorient_cost_kernel,
    _write_relative_position_targets_kernel,
)
from kamino_feasibility.grasp_reorient_cost import grasp_reorient_cost_numpy
from kamino_feasibility.target_adapter import newton_qpos_to_mujoco


def test_relative_action_reads_each_world_current_joint_position() -> None:
    actions = np.array(
        [
            [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
            [[0.4, 0.5, 0.6], [-0.4, -0.5, -0.6]],
        ],
        dtype=np.float32,
    )
    joint_q = np.array(
        [10.0, 1.0, 2.0, 3.0, 0.0, 20.0, 4.0, 5.0, 6.0, 0.0],
        dtype=np.float32,
    )
    world_starts = np.array([0, 5], dtype=np.int32)
    target_indices = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.int32)
    targets = wp.full(6, -99.0, dtype=wp.float32, device="cpu")

    wp.launch(
        _write_relative_position_targets_kernel,
        dim=(2, 3),
        inputs=[
            1,
            wp.array(actions, dtype=wp.float32, device="cpu"),
            wp.array(world_starts, dtype=wp.int32, device="cpu"),
            1,
            wp.array(joint_q, dtype=wp.float32, device="cpu"),
            wp.array(target_indices, dtype=wp.int32, device="cpu"),
            targets,
        ],
        device="cpu",
    )
    np.testing.assert_allclose(
        targets.numpy(),
        np.array([1.4, 3.6, 2.5, 4.5, 3.6, 5.4], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-7,
    )


@pytest.mark.parametrize("terminal", [False, True])
def test_newton_exact_cost_kernel_matches_engine_neutral_reference(
    terminal: bool,
) -> None:
    rng = np.random.default_rng(91)
    newton_qpos = rng.normal(0.0, 0.2, 23).astype(np.float32)
    qvel = rng.normal(0.0, 0.3, 22).astype(np.float32)
    newton_qpos[16:19] = np.array([0.03, 0.01, 0.12], dtype=np.float32)
    # Newton free-quaternion order is xyzw.
    newton_qpos[19:23] = np.array(
        [0.0, 0.3826834, 0.0, 0.9238795], dtype=np.float32
    )
    tips = np.array(
        [
            [0.02, 0.00, 0.11],
            [0.04, 0.00, 0.13],
            [0.03, 0.03, 0.12],
            [0.03, -0.01, 0.10],
        ],
        dtype=np.float32,
    )
    site_transforms = np.zeros((5, 7), dtype=np.float32)
    site_transforms[:, 6] = 1.0
    site_transforms[1:, :3] = tips
    home = np.linspace(-0.25, 0.25, 16, dtype=np.float32)
    goal = np.concatenate(
        (
            np.array([0.02, 0.035, 0.10], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            home,
            np.array([0.08], dtype=np.float32),
        )
    )
    weights = np.array(
        [5, 2, 3, 4, 0.7, 0.2, 0.1, 0.05, 50, 30, 20, 200],
        dtype=np.float32,
    )
    costs = wp.zeros(1, dtype=wp.float32, device="cpu")

    wp.launch(
        _accumulate_grasp_reorient_cost_kernel,
        dim=1,
        inputs=[
            terminal,
            wp.array([0], dtype=wp.int32, device="cpu"),
            wp.array([0], dtype=wp.int32, device="cpu"),
            wp.array(newton_qpos, dtype=wp.float32, device="cpu"),
            wp.array(qvel, dtype=wp.float32, device="cpu"),
            wp.array(site_transforms, dtype=wp.transformf, device="cpu"),
            wp.array([1, 2, 3, 4], dtype=wp.int32, device="cpu"),
            5,
            wp.array(goal, dtype=wp.float32, device="cpu"),
            wp.array(weights, dtype=wp.float32, device="cpu"),
            costs,
        ],
        device="cpu",
    )
    mujoco_qpos = newton_qpos_to_mujoco(newton_qpos, (16,))
    expected, _ = grasp_reorient_cost_numpy(
        mujoco_qpos,
        qvel,
        tips,
        terminal=terminal,
        goal=goal,
        weights=weights,
    )
    assert float(costs.numpy()[0]) == pytest.approx(expected, abs=2.0e-6)
