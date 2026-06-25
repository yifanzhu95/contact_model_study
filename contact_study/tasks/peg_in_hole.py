from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from .base import BaseTask, ContactComplexity, SCENES_DIR, TaskSpec, register


@wp.func
def peg_in_hole_cost_wp(qpos:      wp.array(dtype=float),
                        qvel:      wp.array(dtype=float),
                        ctrl:      wp.array(dtype=float),
                        site_xpos: wp.array(dtype=wp.vec3),
                        site_xmat: wp.array(dtype=wp.mat33),
                        terminal:  bool,
                        goal:      wp.array(dtype=float),
                        indices:   wp.array(dtype=int),
                        weights:   wp.array(dtype=float)) -> float:
    peg_tip_id     = indices[0]
    bottom_hole_id = indices[1]
    n_joints       = indices[2]
    joint_vel_adr  = indices[3]

    # Squared distance between peg tip and hole bottom
    p_tip  = site_xpos[peg_tip_id]
    p_goal = wp.vec3(goal[0], goal[1], goal[2])
    diff   = p_tip - p_goal
    c_dist = wp.dot(diff, diff)

    # Orientation alignment: trace-based distance between site rotation matrices.
    # (3 - trace(R_tip^T @ R_hole)) / 4  → 0 when aligned, 1 when 90° off.
    R_tip  = site_xmat[peg_tip_id]
    R_hole = site_xmat[bottom_hole_id]
    R_rel  = wp.transpose(R_tip) * R_hole
    trace  = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
    c_orient = (3.0 - trace) / 4.0

    # Joint velocity penalty
    c_vel = float(0.0)
    for i in range(n_joints):
        v = qvel[joint_vel_adr + i]
        c_vel = c_vel + v * v

    cost = weights[0] * c_dist + weights[2] * c_orient + weights[3] * c_vel
    if terminal:
        cost = weights[1] * c_dist + weights[4] * c_orient
    return cost


@register("peg_in_hole")
class PegInHoleTask(BaseTask):
    """Insert a peg into a tight-tolerance hole.

    Contact complexity: HIGH (multi-point contact during insertion,
    tight clearance, requires precise force control).

    Goal: drive the peg_tip site to the bottom_of_hole site.
    """

    @property
    def spec(self) -> TaskSpec:
        return TaskSpec(
            name               = "peg_in_hole",
            complexity         = ContactComplexity.HIGH,
            xml_path_template  = "peg_in_hole/peg_in_hole_scene.xml",
            max_steps          = 750,
            success_thresholds = {"dist": 0.01},
            cost_weights       = {
                "w_dist":        3.0,
                "w_dist_term":   10.0,
                "w_orient":      1.0,
                "w_vel":         0.0,
                "w_orient_term": 5.0,
            },
        )

    # BaseTask.load() now reads self.config (TaskConfig), which this legacy
    # TaskSpec-only task doesn't set; load straight from `spec` instead.
    def load(self, full_path: str | None = None):
        if full_path is None:
            xml_path = self.spec.xml_path_template.format(geometry=self.geometry.value)
            full_path = SCENES_DIR / xml_path
        self._mjm = mujoco.MjModel.from_xml_path(str(full_path))
        self._mjd = mujoco.MjData(self._mjm)
        self.initialize_task()
        return self._mjm, self._mjd

    def initialize_task(self):
        mjm = self.mjm
        mjd = self._mjd

        peg_tip_id     = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "peg_tip")
        bottom_hole_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "bottom_of_hole")

        if peg_tip_id < 0:
            raise ValueError("Site 'peg_tip' not found in model.")
        if bottom_hole_id < 0:
            raise ValueError("Site 'bottom_of_hole' not found in model.")

        self.peg_tip_id     = peg_tip_id
        self.bottom_hole_id = bottom_hole_id

        # Goal = world position of bottom_of_hole at load time (static mocap body).
        mujoco.mj_forward(mjm, mjd)
        self.goal_pos = mjd.site_xpos[bottom_hole_id].copy()

        # Joint velocity info: use all DOFs (fixed-base arm has no freejoint)
        n_joints = mjm.nv
        joint_vel_adr = 0  # first DOF address for a fixed-base robot

        self.index_vector = np.array(
            [peg_tip_id, bottom_hole_id, n_joints, joint_vel_adr],
            dtype=np.int32,
        )
        self.goal_vector  = self.goal_pos.astype(np.float32)

        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32,   device="cuda")
        self.goal_vector_wp  = wp.array(self.goal_vector,  dtype=wp.float32, device="cuda")

        w = self.spec.cost_weights
        self.weights_wp = wp.array(
            [w["w_dist"], w["w_dist_term"], w["w_orient"], w["w_vel"], w["w_orient_term"]],
            dtype=wp.float32, device="cuda",
        )

    def get_inital_state(self, rng: np.random.Generator):
        mjm = self.mjm
        if mjm.nkey < 1:
            raise ValueError("No keyframe defined in the XML model.")
        q0    = mjm.key_qpos[0].copy()
        v0    = mjm.key_qvel[0].copy()
        ctrl0 = mjm.key_ctrl[0].copy()
        return q0, v0, ctrl0

    @property
    def cost_fn_wp(self) -> wp.func:
        return peg_in_hole_cost_wp

    def is_success(self, mjd: mujoco.MjData) -> bool:
        p_tip  = mjd.site_xpos[self.peg_tip_id]
        p_goal = self.goal_pos
        dist   = float(np.linalg.norm(p_tip - p_goal))
        return dist < self.spec.success_thresholds["dist"]

    def has_failed(self, mjd: mujoco.MjData) -> bool:
        p_tip  = mjd.site_xpos[self.peg_tip_id]
        p_goal = self.goal_pos
        dist   = float(np.linalg.norm(p_tip - p_goal))
        return dist > 0.5
