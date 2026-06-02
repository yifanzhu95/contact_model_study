from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from .base import BaseTask, ContactComplexity, TaskSpec, register


@wp.func
def push_cost_wp(qpos: wp.array(dtype=float), qvel: wp.array(dtype=float), ctrl: wp.array(dtype=float),
                 terminal: bool, goal: wp.array(dtype=float), indices: wp.array(dtype=int),
                 xpos: wp.array(dtype=wp.vec3),
                 xquat: wp.array(dtype=wp.quat),
                 site_xpos: wp.array(dtype=wp.vec3),
                 weights: wp.array(dtype=float)) -> float:
    adr = indices[0]
    dx = qpos[adr] - goal[0]
    dy = qpos[adr + 1] - goal[1]
    dist = wp.sqrt(dx*dx + dy*dy)
    if terminal:
        return dist * weights[1]
    return dist * weights[0]


@register("push")
class PushTask(BaseTask):
    """Push a box to a target position on a table.

    Contact complexity: LOW (single contact point between pusher and box).
    The task is largely quasi-static; soft contact models should work well.
    """

    @property
    def spec(self) -> TaskSpec:
        return TaskSpec(
            name              = "push",
            complexity        = ContactComplexity.LOW,
            xml_path_template = "tasks/push_{geometry}.xml",
            max_steps         = 200,
            success_threshold = 0.02,
            cost_weights      = {"running": 1.0, "terminal": 10.0}
        )

    def initialize_task(self):
        mjm = self.mjm
        obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
        obj_body = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")

        tips = ["ff_tip", "mf_tip", "rf_tip", "th_tip"]
        tip_ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, t) for t in tips]

        self.index_vector = np.array([
            mjm.jnt_qposadr[obj_jnt],
            mjm.jnt_dofadr[obj_jnt],
            0,
            16,
            obj_body,
            *tip_ids
        ], dtype=np.int32)

        if mjm.nkey > 0:
            robot_start = self.index_vector[2]
            n_manip = self.index_vector[3]
            home_state = mjm.key_qpos[0, robot_start : robot_start + n_manip].copy()
        else:
            raise ValueError("No keyframe defined in the XML model. A keyframe is required to define the manipulator's home state.")

        target_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "obj_target")
        self.goal_vector = np.concatenate([
            mjm.site_pos[target_id],
            mjm.site_quat[target_id],
            home_state
        ]).astype(np.float32)

        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32, device="cuda")
        self.goal_vector_wp = wp.array(self.goal_vector, dtype=wp.float32, device="cuda")

        w = self.spec.cost_weights
        self.weights_wp = wp.array([w["running"], w["terminal"]], dtype=wp.float32, device="cuda")

    def get_inital_state(self, rng: np.random.Generator):
        mjm = self.mjm
        if mjm.nkey > 0:
            q0 = mjm.key_qpos[0].copy()
            v0 = mjm.key_qvel[0].copy()
            ctrl0 = mjm.key_ctrl[0].copy()
        else:
            raise ValueError("No keyframe defined in the XML model. A keyframe is required for the initial state.")

        box_qpos_adr = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
        if box_qpos_adr >= 0:
            adr = mjm.jnt_qposadr[box_qpos_adr]
            q0[adr:adr+2] += rng.uniform(-0.1, 0.1, 2)
        return q0, v0, ctrl0

    def cost_fn(self, qpos, qvel, ctrl, terminal: bool, goal, indices) -> np.ndarray:
        qpos_np = np.asarray(qpos.numpy() if hasattr(qpos, "numpy") else qpos)
        adr = indices[0]
        box_pos = qpos_np[:, adr:adr+2] if qpos_np.ndim == 2 else qpos_np[adr:adr+2]
        cost = np.linalg.norm(box_pos - goal, axis=-1).astype(np.float32)
        if terminal:
            cost *= 10.0
        return cost

    @property
    def cost_fn_wp(self) -> wp.func:
        return push_cost_wp

    def is_success(self, mjd: mujoco.MjData) -> bool:
        box_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
        if box_id < 0:
            return False
        target_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, "obj_target")
        if target_id < 0:
            return False
        box_pos    = mjd.xpos[box_id, :2]
        target_pos = mjd.site_xpos[target_id, :2]
        return bool(np.linalg.norm(box_pos - target_pos) < self.spec.success_threshold)
