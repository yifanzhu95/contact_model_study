from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from .base import BaseTask, ContactComplexity, TaskSpec, register


@wp.func
def peg_in_hole_cost_wp(qpos: wp.array(dtype=float), qvel: wp.array(dtype=float), ctrl: wp.array(dtype=float),
                        terminal: bool, goal: wp.array(dtype=float), indices: wp.array(dtype=int),
                        xpos: wp.array(dtype=wp.vec3),
                        xquat: wp.array(dtype=wp.quat),
                        site_xpos: wp.array(dtype=wp.vec3),
                        weights: wp.array(dtype=float)) -> float:
    adr = indices[0]
    z_err = wp.abs(qpos[adr + 2] - goal[0])
    dx = qpos[adr] - goal[1]
    dy = qpos[adr + 1] - goal[2]
    xy_err = wp.sqrt(dx*dx + dy*dy)
    cost = weights[0] * z_err + weights[1] * xy_err
    if terminal:
        return cost * weights[2]
    return cost


@register("peg_in_hole")
class PegInHoleTask(BaseTask):
    """Insert a peg into a tight-tolerance hole.

    Contact complexity: HIGH (multi-point contact during insertion,
    tight clearance, requires precise force control).
    """

    @property
    def spec(self) -> TaskSpec:
        return TaskSpec(
            name              = "peg_in_hole",
            complexity        = ContactComplexity.HIGH,
            xml_path_template = "tasks/peg_in_hole_{geometry}.xml",
            max_steps         = 400,
            success_threshold = 0.005,
            cost_weights      = {"w_z": 1.0, "w_xy": 5.0, "w_term": 30.0}
        )

    def initialize_task(self):
        mjm = self.mjm
        peg_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
        self.index_vector = np.array([mjm.jnt_qposadr[peg_jnt]], dtype=np.int32)

        self.goal_vector = np.array([-0.05, 0.0, 0.0], dtype=np.float32)

        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32, device="cuda")
        self.goal_vector_wp = wp.array(self.goal_vector, dtype=wp.float32, device="cuda")

        w = self.spec.cost_weights
        self.weights_wp = wp.array([w["w_z"], w["w_xy"], w["w_term"]], dtype=wp.float32, device="cuda")

    def get_inital_state(self, rng: np.random.Generator):
        mjm = self.mjm
        q0 = mjm.qpos0.copy()
        peg_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
        if peg_jnt >= 0:
            adr = mjm.jnt_qposadr[peg_jnt]
            q0[adr:adr+2] += rng.uniform(-0.003, 0.003, 2)
        return q0, np.zeros(mjm.nv), None

    def cost_fn(self, qpos, qvel, ctrl, terminal: bool, goal, indices) -> np.ndarray:
        qpos_np = np.asarray(qpos.numpy() if hasattr(qpos, "numpy") else qpos)
        adr = indices[0]
        target_z, target_xy = goal[0], goal[1:]
        z_err  = np.abs(qpos_np[:, adr+2] - target_z)
        xy_err = np.linalg.norm(qpos_np[:, adr:adr+2] - target_xy, axis=-1)
        cost   = (z_err + 5.0 * xy_err).astype(np.float32)
        if terminal:
            cost *= 30.0
        return cost

    @property
    def cost_fn_wp(self) -> tuple[wp.func, wp.array, wp.array, wp.array]:
        return peg_in_hole_cost_wp, self.goal_vector_wp, self.index_vector_wp, self.weights_wp

    def is_success(self, mjd: mujoco.MjData) -> bool:
        peg_id  = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "peg")
        hole_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, "hole_bottom")
        if peg_id < 0 or hole_id < 0:
            return False
        peg_tip  = mjd.xpos[peg_id].copy()
        hole_pos = mjd.site_xpos[hole_id].copy()
        depth    = hole_pos[2] - peg_tip[2]
        lateral  = np.linalg.norm(peg_tip[:2] - hole_pos[:2])
        return bool(depth > self.spec.success_threshold and lateral < 0.003)
