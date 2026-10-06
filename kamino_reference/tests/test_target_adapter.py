"""Opt-in target-scene checks for the one-world Kamino adapter."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp

from kamino_feasibility.target_adapter import KaminoOneWorldAdapter
from kamino_feasibility.penetration_resolution import resolve_free_joint_penetration


def _external_device() -> str:
    device = os.environ.get("KAMINO_TARGET_DEVICE", "cuda:0")
    if device.startswith("cuda") and wp.get_cuda_device_count() == 0:
        pytest.skip("Requested target-scene CUDA test, but CUDA is unavailable")
    return device


def test_state_site_and_control_adapter() -> None:
    scene_value = os.environ.get("KAMINO_TARGET_SCENE")
    if not scene_value:
        pytest.skip("Set KAMINO_TARGET_SCENE to run the external-scene adapter test")
    device = _external_device()

    scene = Path(scene_value)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    data.qpos[:16] = np.linspace(-0.08, 0.08, 16)
    half_angle = 0.25
    data.qpos[16:23] = [
        0.2,
        -0.1,
        0.5,
        np.cos(half_angle),
        0.0,
        0.0,
        np.sin(half_angle),
    ]
    data.qvel[:] = np.linspace(-0.2, 0.2, model.nv)
    mujoco.mj_forward(model, data)

    adapter = KaminoOneWorldAdapter(
        scene,
        device=device,
        dt=float(model.opt.timestep),
    )
    np.testing.assert_allclose(adapter.model.shape_gap.numpy(), 0.0, atol=0.0)
    assert adapter.contacts.force is not None
    adapter.set_state(data.qpos.copy(), data.qvel.copy())
    observation = adapter.observation()

    np.testing.assert_allclose(observation.qpos, data.qpos, rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(observation.qvel, data.qvel, rtol=0.0, atol=2.0e-5)
    assert observation.site_names == ("tag", "if_tip", "mf_tip", "rf_tip", "th_tip")
    np.testing.assert_allclose(observation.site_xpos, data.site_xpos, rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(
        observation.site_xmat,
        data.site_xmat.reshape(model.nsite, 3, 3),
        rtol=0.0,
        atol=3.0e-5,
    )

    action = np.linspace(-0.1, 0.1, model.nu)
    adapter.step(action)
    stepped = adapter.observation()
    assert np.isfinite(stepped.qpos).all()
    assert np.isfinite(stepped.qvel).all()
    np.testing.assert_allclose(stepped.control, action, rtol=0.0, atol=1.0e-7)


def test_eval_scene_sparse_contact_construction_and_step() -> None:
    scene_value = os.environ.get("KAMINO_EVAL_SCENE")
    if not scene_value:
        pytest.skip("Set KAMINO_EVAL_SCENE to run the high-fidelity eval-scene gate")
    device = _external_device()

    scene = Path(scene_value)
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 0.0005
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    resolution = resolve_free_joint_penetration(
        model,
        data,
        "obj_joint",
        clearance_m=0.0005,
    )
    assert resolution.final_penetrating_records == 0
    sparse_solver = "CRF" if device.startswith("cuda") else "CR"

    adapter = KaminoOneWorldAdapter(
        scene,
        device=device,
        dt=0.0005,
        enable_contacts=True,
        contact_gap_m=0.0,
        dynamics_storage="sparse",
        sparse_linear_solver=sparse_solver,
        padmm_tolerance=5.0e-4,
        padmm_max_iterations=200,
    )
    adapter.set_state(data.qpos.copy(), data.qvel.copy())
    action = np.linspace(-0.1, 0.1, model.nu)
    adapter.step(action)
    observation = adapter.observation()
    status = adapter.solver_status()

    assert adapter.dynamics_storage == "sparse"
    assert adapter.sparse_linear_solver == sparse_solver
    assert np.isfinite(observation.qpos).all()
    assert np.isfinite(observation.qvel).all()
    assert np.isfinite(observation.site_xpos).all()
    assert status["iterations"] <= 200
    assert np.isfinite([status["r_p"], status["r_d"], status["r_c"]]).all()
