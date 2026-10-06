"""Opt-in regression checks for the real Leap multi-world rollout."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp

from kamino_feasibility.batched_target_rollout import KaminoBatchedLeapRollout
from kamino_feasibility.penetration_resolution import resolve_free_joint_penetration


def test_two_world_target_reset_contact_and_replay() -> None:
    scene_value = os.environ.get("KAMINO_TARGET_SCENE")
    if not scene_value:
        pytest.skip("Set KAMINO_TARGET_SCENE to run the target multi-world test")
    if wp.get_cuda_device_count() == 0:
        pytest.skip("The target multi-world test requires CUDA")

    scene = Path(scene_value)
    mj_model = mujoco.MjModel.from_xml_path(str(scene))
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    resolve_free_joint_penetration(
        mj_model,
        mj_data,
        "obj_joint",
        clearance_m=0.0005,
    )

    rollout = KaminoBatchedLeapRollout(
        scene,
        world_count=2,
        horizon_steps=9,
        padmm_tolerance=5.0e-4,
        dynamics_storage="sparse",
        sparse_linear_solver="CRF",
    )
    rollout.set_initial_state(mj_data.qpos.copy(), mj_data.qvel.copy())
    rollout.set_target_object_position(mj_data.qpos[16:19] + [0.0, 0.0, 0.01])
    action = np.linspace(-0.1, 0.1, rollout.nu, dtype=np.float32)
    actions = np.broadcast_to(action, (9, 2, rollout.nu)).copy()
    rollout.upload_action_sequences(actions)

    rollout.reset_device()
    rollout.rollout_device()
    first = rollout.download_result()
    rollout.reset_device()
    rollout.rollout_device()
    second = rollout.download_result()

    assert np.all(first.reset_success)
    assert np.all(first.final_status["converged"] == 1)
    assert np.all(first.converged_steps == 9)
    assert np.all(first.max_active_contacts == 2)
    assert np.isfinite(first.flattened()).all()
    assert np.isfinite(first.site_world).all()
    np.testing.assert_allclose(second.costs, first.costs, rtol=0.0, atol=1.0e-5)
    np.testing.assert_allclose(
        second.final_object_q, first.final_object_q, rtol=0.0, atol=1.0e-5
    )
    np.testing.assert_allclose(
        second.final_object_qd, first.final_object_qd, rtol=0.0, atol=1.0e-5
    )
    np.testing.assert_allclose(
        first.final_object_q[1], first.final_object_q[0], rtol=0.0, atol=1.0e-5
    )
    np.testing.assert_allclose(
        first.final_object_qd[1], first.final_object_qd[0], rtol=0.0, atol=2.0e-4
    )
