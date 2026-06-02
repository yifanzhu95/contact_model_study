from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from .base import BaseTask, ContactComplexity, TaskSpec, register

@wp.func
def grasp_reorient_cost_wp(qpos: wp.array(dtype=float),
                           qvel: wp.array(dtype=float),
                           ctrl: wp.array(dtype=float),
                           site_xpos: wp.array(dtype=wp.vec3),
                           terminal: bool,
                           goal: wp.array(dtype=float),
                           indices: wp.array(dtype=int),
                           weights: wp.array(dtype=float)) -> float:
    obj_qpos_adr   = indices[0]
    obj_qvel_adr   = indices[1]
    robot_qpos_adr = indices[2]
    n_manip        = indices[3]

    p_obj = wp.vec3(qpos[obj_qpos_adr], qpos[obj_qpos_adr + 1], qpos[obj_qpos_adr + 2])
    q_obj_v4 = wp.vec4(qpos[obj_qpos_adr + 3], qpos[obj_qpos_adr + 4], qpos[obj_qpos_adr + 5], qpos[obj_qpos_adr + 6])
    v_obj = wp.vec3(qvel[obj_qvel_adr], qvel[obj_qvel_adr + 1], qvel[obj_qvel_adr + 2])
    w_obj = wp.vec3(qvel[obj_qvel_adr + 3], qvel[obj_qvel_adr + 4], qvel[obj_qvel_adr + 5])

    p_target = wp.vec3(goal[0], goal[1], goal[2])
    q_target = wp.vec4(goal[3], goal[4], goal[5], goal[6])

    dot_prod = wp.dot(q_target, q_obj_v4)
    c_quat = 1.0 - dot_prod * dot_prod

    pos_diff = p_obj - p_target
    c_pos = wp.dot(pos_diff, pos_diff)

    c_joint = float(0.0)
    for i in range(n_manip):
        dq = qpos[robot_qpos_adr + i] - goal[7 + i]
        c_joint = c_joint + dq * dq

    c_joint_velo = float(0.0)
    for i in range(n_manip):
        dq = qvel[robot_qpos_adr + i]
        c_joint_velo = c_joint_velo + dq * dq

    c_contact = float(0.0)
    for i in range(5, 9):
        p_tip = site_xpos[indices[i]]
        dp = wp.length(p_obj - p_tip) - float(0.035)
        if dp > 0.0:
            c_contact = c_contact + dp*dp

    fallen = float(0.0)
    if qpos[obj_qpos_adr + 2] < 0.05:
        fallen = 1.0

    c_velo = wp.dot(v_obj, v_obj) + wp.dot(w_obj, w_obj)

    cost = (
        weights[0] * c_quat +
        weights[1] * c_pos +
        weights[2] * c_velo +
        weights[3] * c_contact +
        weights[4] * c_joint +
        weights[5] * c_joint_velo +
        weights[6] * fallen
    )
    if terminal:
        cost = (weights[5] * c_quat) + (weights[6] * c_pos) + weights[7]*fallen
    return cost


@register("grasp_reorient")
class GraspReorientTask(BaseTask):
    """Grasp a cylindrical object and reorient it to a target pose.

    Contact complexity: MEDIUM (4+ contacts between gripper fingers and object,
    dynamic lifting and rotation).
    """

    @property
    def spec(self) -> TaskSpec:
        return TaskSpec(
            name              = "grasp_reorient",
            complexity        = ContactComplexity.MEDIUM,
            xml_path_template = "leap_hand/scene_leap_cube.xml",
            max_steps         = 1000,
            success_threshold = 0.05,
            velocity_threshold = 0.1,
            cost_weights      = {
                "w_quat": 5.0,
                "w_pos": 10.0,
                "w_velo": 0.002,
                "w_contact": 3.5,
                "w_joint": 0.1,
                "w_joint_velo": 0.0,
                "w_fallen": 20.0,
                "w_quat_term": 10.0,
                "w_pos_term": 10.0,
                "w_fallen_term": 0.0,
            }
        )

    def initialize_task(self):
        mjm = self.mjm
        obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")

        obj_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
        tip_ids = [
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "if_tip"),
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "mf_tip"),
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "rf_tip"),
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "th_tip")
        ]

        self.index_vector = np.array([
            mjm.jnt_qposadr[obj_jnt],
            mjm.jnt_dofadr[obj_jnt],
            0,
            16,
            obj_id,
            *tip_ids
        ], dtype=np.int32)

        target_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj_target")
        mocap_id = mjm.body_mocapid[target_id]
        self.target_pos = self._mjd.mocap_pos[mocap_id]
        self.target_quat = self._mjd.mocap_quat[mocap_id]

        if mjm.nkey > 0:
            key_idx = 1 if mjm.nkey > 1 else 0
            robot_start = self.index_vector[2]
            n_manip = self.index_vector[3]
            self.home_state = mjm.key_ctrl[key_idx, robot_start : robot_start + n_manip].copy()
        else:
            raise ValueError("No keyframe defined in the XML model. A keyframe is required to define the manipulator's home state.")

        self.goal_vector = np.concatenate([
            self.target_pos, self.target_quat, self.home_state
        ]).astype(np.float32)

        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32, device="cuda")
        self.goal_vector_wp = wp.array(self.goal_vector, dtype=wp.float32, device="cuda")

        w = self.spec.cost_weights
        weights_list = [
            w["w_quat"], w["w_pos"], w["w_velo"],
            w["w_contact"], w["w_joint"], w["w_joint_velo"],
            w["w_fallen"],
            w["w_quat_term"], w["w_pos_term"], w["w_fallen_term"]
        ]
        self.weights_wp = wp.array(weights_list, dtype=wp.float32, device="cuda")

    def get_inital_state(self, rng: np.random.Generator):
        mjm = self.mjm
        if mjm.nkey > 0:
            q0 = mjm.key_qpos[0].copy()
            v0 = mjm.key_qvel[0].copy()
            ctrl0 = mjm.key_ctrl[0].copy()
        else:
            raise ValueError("No keyframe defined in the XML model. A keyframe is required for the initial state.")
        return q0, v0, ctrl0

    @property
    def cost_fn_wp(self) -> tuple[wp.func, wp.array, wp.array, wp.array]:
        return grasp_reorient_cost_wp, self.goal_vector_wp, self.index_vector_wp, self.weights_wp

    def is_success(self, mjd: mujoco.MjData) -> bool:
        mjm = self.mjm
        obj_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
        if obj_id < 0:
            return False

        pos_err = np.linalg.norm(mjd.xpos[obj_id] - self.target_pos)
        obj_quat = mjd.xquat[obj_id]
        quat_err = 1.0 - np.dot(obj_quat, self.target_quat)**2

        pose_ok = pos_err < self.spec.success_threshold and quat_err < self.spec.success_threshold
        if not pose_ok:
            return False

        vel_threshold = getattr(self.spec, "velocity_threshold", 1.0)
        return bool(np.linalg.norm(mjd.cvel[obj_id]) < vel_threshold)

    def has_fallen(self, mjd: mujoco.MjData) -> bool:
        mjm = self.mjm
        obj_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
        if obj_id < 0:
            return False
        return bool(mjd.xpos[obj_id][2] < 0.0)

    def sample_new_goal(self, mjd: mujoco.MjData, rng: np.random.Generator):
        """Sample a new goal orientation by rotating +/- 90 degrees around an object-local cardinal axis."""
        axis_idx = rng.integers(0, 3)
        axis = np.zeros(3)
        axis[axis_idx] = 1.0

        angle = rng.choice([np.pi / 2.0, -np.pi / 2.0])

        c = np.cos(angle / 2.0)
        s = np.sin(angle / 2.0)
        q_rot = np.array([c, s * axis[0], s * axis[1], s * axis[2]])

        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, self.target_quat, q_rot)

        target_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "obj_target")
        if target_id >= 0:
            mocap_id = self.mjm.body_mocapid[target_id]
            if mocap_id >= 0:
                mjd.mocap_quat[mocap_id] = new_quat

        self.target_quat = new_quat.copy()
        self.goal_vector = np.concatenate([
            self.target_pos, self.target_quat, self.home_state
        ]).astype(np.float32)

        if self.goal_vector_wp is not None:
            self.goal_vector_wp.assign(self.goal_vector)
