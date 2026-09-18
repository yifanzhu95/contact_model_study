"""CPU-only contract checks for each existing high/high scene pair.

These do not invoke task.load(), which creates GPU cost arrays, and do not
claim that parsing/forward kinematics validate a full control episode.
"""

import mujoco
import numpy as np
import pytest

import contact_study.tasks  # noqa: F401
from contact_study.tasks.base import get_task
from contact_study.tasks.config import TaskRole


@pytest.mark.parametrize("obj", ["cube", "duck", "ball", "spam", "tomato"])
def test_high_high_assets_and_state_control_layout(obj):
    models = []
    for role in (TaskRole.ROLLOUT, TaskRole.EVAL):
        task = get_task("grasp_reorient", geometry=f"{obj}_high_high", role=role)
        model = mujoco.MjModel.from_xml_path(str(task.resolve_scene_path()))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert (model.nq, model.nv, model.nu) == (23, 22, 16)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert np.isfinite(data.xpos).all()
        assert model.opt.ccd_iterations == 35
        joint = model.joint("obj_joint")
        assert int(joint.qposadr[0]) == 16
        assert int(joint.dofadr[0]) == 16
        for tip in ("if_tip", "mf_tip", "rf_tip", "th_tip"):
            assert model.site(tip).id >= 0
        models.append(model)
    rollout, evaluation = models
    assert [rollout.actuator(i).name for i in range(16)] == [
        evaluation.actuator(i).name for i in range(16)]
    assert [rollout.joint(int(rollout.actuator(i).trnid[0])).name for i in range(16)] == [
        evaluation.joint(int(evaluation.actuator(i).trnid[0])).name for i in range(16)]
