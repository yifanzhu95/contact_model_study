"""Concrete task implementations.

Tasks are ordered by contact complexity:
  PushTask          (LOW)    - planar pushing, 1-2 contacts
  GraspReorientTask (MEDIUM) - grasp and reorient, ~4 contacts
  PegInHoleTask     (HIGH)   - peg insertion, tight multi-contact
"""

from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from contact_study.contact_models.config import GeometryVariant
from .base import BaseTask, ContactComplexity, TaskSpec, register

# ---------------------------------------------------------------------------
# Warp Cost Functions (GPU-side)
# ---------------------------------------------------------------------------

@wp.func
def push_cost_wp(qpos: wp.array(dtype=float), qvel: wp.array(dtype=float), ctrl: wp.array(dtype=float), 
                 terminal: bool, goal: wp.array(dtype=float), indices: wp.array(dtype=int),
                 xpos: wp.array(dtype=wp.vec3),
                 xquat: wp.array(dtype=wp.quat),
                 site_xpos: wp.array(dtype=wp.vec3),
                 weights: wp.array(dtype=float)) -> float:
    # indices[0]: Box qpos address
    adr = indices[0]
    dx = qpos[adr] - goal[0]
    dy = qpos[adr + 1] - goal[1]
    dist = wp.sqrt(dx*dx + dy*dy)
    if terminal:
        return dist * weights[1]
    return dist * weights[0]

@wp.func
def peg_in_hole_cost_wp(qpos: wp.array(dtype=float), qvel: wp.array(dtype=float), ctrl: wp.array(dtype=float), 
                        terminal: bool, goal: wp.array(dtype=float), indices: wp.array(dtype=int),
                        xpos: wp.array(dtype=wp.vec3),
                        xquat: wp.array(dtype=wp.quat),
                        site_xpos: wp.array(dtype=wp.vec3),
                        weights: wp.array(dtype=float)) -> float:
    # indices[0]: Peg qpos address. goal: [target_z, target_x, target_y]
    adr = indices[0]
    z_err = wp.abs(qpos[adr + 2] - goal[0])
    dx = qpos[adr] - goal[1]
    dy = qpos[adr + 1] - goal[2]
    xy_err = wp.sqrt(dx*dx + dy*dy)
    cost = weights[0] * z_err + weights[1] * xy_err
    if terminal:
        return cost * weights[2]
    return cost

# ---------------------------------------------------------------------------
# Task 1: Planar Pushing (LOW complexity)
# ---------------------------------------------------------------------------

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
            success_threshold = 0.02,  # 2 cm
            cost_weights      = {"running": 1.0, "terminal": 10.0}
        )

    def initialize_task(self):
        mjm = self.mjm
        # Get joint and body indices
        obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
        obj_body = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
        
        # Fingertip site IDs for contact cost
        tips = ["ff_tip", "mf_tip", "rf_tip", "th_tip"]
        tip_ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, t) for t in tips]

        # 1. Index Vector Mapping
        self.index_vector = np.array([
            mjm.jnt_qposadr[obj_jnt], # 0: obj qpos
            mjm.jnt_dofadr[obj_jnt],  # 1: obj qvel
            0,                        # 2: robot qpos starts at 0
            16,                       # 3: n_manip (Allegro has 16 joints)
            obj_body,                 # 4: obj body id
            *tip_ids                  # 5-8: fingertip ids
        ], dtype=np.int32)
        
        # Determine the reference posture (manipulator home state)
        if mjm.nkey > 0:
            # Use the robot part of the first keyframe for the posture reference
            robot_start = self.index_vector[2]
            n_manip = self.index_vector[3]
            home_state = mjm.key_qpos[0, robot_start : robot_start + n_manip].copy()
        else:
            raise ValueError("No keyframe defined in the XML model. A keyframe is required to define the manipulator's home state.")

        # 2. Goal Vector Mapping
        target_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "obj_target")
        # goal stores: [pos(3), quat(4), q_ref(16)]
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
            # Use the first keyframe defined in the XML (e.g., key0)
            q0 = mjm.key_qpos[0].copy()
            v0 = mjm.key_qvel[0].copy()
            ctrl0 = mjm.key_ctrl[0].copy()
        else:
            raise ValueError("No keyframe defined in the XML model. A keyframe is required for the initial state.")

        # Randomize box x,y position within ±0.1 m of nominal
        box_qpos_adr = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
        if box_qpos_adr >= 0:
            adr = mjm.jnt_qposadr[box_qpos_adr]
            q0[adr:adr+2] += rng.uniform(-0.1, 0.1, 2)
        return q0, v0, ctrl0

    def cost_fn(self, qpos, qvel, ctrl, terminal: bool, goal, indices) -> np.ndarray:
        """L2 distance of box position to target."""
        qpos_np = np.asarray(qpos.numpy() if hasattr(qpos, "numpy") else qpos)
        adr = indices[0]
        box_pos = qpos_np[:, adr:adr+2] if qpos_np.ndim == 2 else qpos_np[adr:adr+2]
        
        cost = np.linalg.norm(box_pos - goal, axis=-1).astype(np.float32)
        if terminal:
            cost *= 10.0
        return cost

    @property
    def cost_fn_wp(self) -> tuple[wp.func, wp.array, wp.array, wp.array]:
        return push_cost_wp, self.goal_vector_wp, self.index_vector_wp, self.weights_wp

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


# ---------------------------------------------------------------------------
# Task 2: Grasp and Reorient (MEDIUM complexity)
# ---------------------------------------------------------------------------

@wp.func
def grasp_reorient_cost_wp(qpos: wp.array(dtype=float), 
                           qvel: wp.array(dtype=float), 
                           ctrl: wp.array(dtype=float), 
                           site_xpos: wp.array(dtype=wp.vec3),
                           terminal: bool, 
                           goal: wp.array(dtype=float), 
                           indices: wp.array(dtype=int),
                           weights: wp.array(dtype=float)) -> float:
    # Index Mapping MUST match initialize_task
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

    # 1. Orientation error
    dot_prod = wp.dot(q_target, q_obj_v4)
    c_quat = 1.0 - dot_prod * dot_prod

    # 2. Position error
    pos_diff = p_obj - p_target
    c_pos = wp.dot(pos_diff, pos_diff)

    # 3. Joint deviation 
    c_joint = float(0.0) 
    for i in range(n_manip):
        dq = qpos[robot_qpos_adr + i] - goal[7 + i]
        c_joint = c_joint + dq * dq

    c_joint_velo = float(0.0) 
    for i in range(n_manip):
        dq = qvel[robot_qpos_adr + i]
        c_joint_velo = c_joint_velo + dq * dq

    # 4. Contact cost
    c_contact = float(0.0)
    for i in range(5, 9):
        p_tip = site_xpos[indices[i]]
        dp = wp.length(p_obj - p_tip) - float(0.035)
        if dp > 0.0:
            c_contact = c_contact + dp*dp

    #5 
    fallen = float(0.0)
    if qpos[obj_qpos_adr + 2] < 0.06:
        fallen = 1.0

    #6 object velocity
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
            xml_path_template = "leap_hand/scene_leap_cube.xml",#"scenes/test_data/allegro/allegro_right_hand_armature.xml",
            max_steps         = 100,
            success_threshold = 0.025,  # combined pose error
            cost_weights      = {
                "w_quat": 5.0, #0.5,
                "w_pos": 5.0,#0.1,
                "w_velo": 0.01,
                "w_contact": 2.0,#0.5,
                "w_joint": 0.5,
                "w_joint_velo": 0.0,
                "w_fallen": 20.0,#50.0,
                "w_quat_term": 10.0,#10.0,
                "w_pos_term": 10.0,#5.0,
                "w_fallen_term": 0.0,#100.0
            }
        )

    def initialize_task(self):
        mjm = self.mjm
        obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
        
        # Required Site IDs for site_xpos/site_xquat
        obj_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
        tip_ids = [
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "if_tip"),
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "mf_tip"),
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "rf_tip"),
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "th_tip")
        ]

        # Construct index vector (Length 9)
        self.index_vector = np.array([
            mjm.jnt_qposadr[obj_jnt], # 0
            mjm.jnt_dofadr[obj_jnt],  # 1
            0,                        # 2: robot_qpos_adr (Hand is first in XML)
            16,                       # 3: n_manip (Allegro has 16 joints)
            obj_id,                   # 4
            *tip_ids                  # 5, 6, 7, 8
        ], dtype=np.int32)
        
        target_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj_target")
        mocap_id = mjm.body_mocapid[target_id]
        self.target_pos = self._mjd.mocap_pos[mocap_id]
        self.target_quat = self._mjd.mocap_quat[mocap_id]
        
        # Use keyframe for reference posture (home state) if available
        if mjm.nkey > 0:
            # Keyframe index 1 is used as goal if it exists, otherwise key0 for posture reference
            key_idx = 1 if mjm.nkey > 1 else 0
            robot_start = self.index_vector[2]
            n_manip = self.index_vector[3]
            self.home_state = mjm.key_ctrl[key_idx, robot_start : robot_start + n_manip].copy()
        else:
            raise ValueError("No keyframe defined in the XML model. A keyframe is required to define the manipulator's home state.")

        # Concatenate pos (3), quat (4), and manipulator home pose (16)
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
            # Use the first keyframe defined in the XML (e.g., key0)
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

        pos_err = np.linalg.norm(mjd.xpos[obj_id] - self.target_pos)
        obj_quat = mjd.xquat[obj_id]
        quat_err = 1.0 - np.dot(obj_quat, self.target_quat)**2
        
        return bool(pos_err < self.spec.success_threshold and quat_err < self.spec.success_threshold)

    def sample_new_goal(self, mjd: mujoco.MjData, rng: np.random.Generator):
        """Sample a new goal orientation by rotating +/- 90 degrees around an object-local cardinal axis."""
        # Select a cardinal axis (0:X, 1:Y, 2:Z)
        axis_idx = rng.integers(0, 3)
        axis = np.zeros(3)
        axis[axis_idx] = 1.0

        # Select rotation angle (+90 or -90 degrees)
        angle = rng.choice([np.pi / 2.0, -np.pi / 2.0])

        # Construct rotation quaternion [cos(theta/2), sin(theta/2)*axis]
        c = np.cos(angle / 2.0)
        s = np.sin(angle / 2.0)
        q_rot = np.array([c, s * axis[0], s * axis[1], s * axis[2]])

        # Apply rotation around local axes: new = current * q_rot
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, self.target_quat, q_rot)

        # Update mocap body orientation in the simulation data
        target_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "obj_target")
        if target_id >= 0:
            mocap_id = self.mjm.body_mocapid[target_id]
            if mocap_id >= 0:
                mjd.mocap_quat[mocap_id] = new_quat

        # Update internal cached target and rebuild the goal vector
        self.target_quat = new_quat.copy()
        self.goal_vector = np.concatenate([
            self.target_pos, self.target_quat, self.home_state
        ]).astype(np.float32)

        # Assign the new goal vector to the Warp array (planner will see this on next step)
        if self.goal_vector_wp is not None:
            self.goal_vector_wp.assign(self.goal_vector)
# ---------------------------------------------------------------------------
# Task 3: Peg-in-Hole Assembly (HIGH complexity)
# ---------------------------------------------------------------------------

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
            success_threshold = 0.005,  # 5 mm insertion depth
            cost_weights      = {"w_z": 1.0, "w_xy": 5.0, "w_term": 30.0}
        )

    def initialize_task(self):
        mjm = self.mjm
        peg_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
        self.index_vector = np.array([mjm.jnt_qposadr[peg_jnt]], dtype=np.int32)
        
        # Target: z = -0.05, xy = 0.0
        self.goal_vector = np.array([-0.05, 0.0, 0.0], dtype=np.float32)
        
        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32, device="cuda")
        self.goal_vector_wp = wp.array(self.goal_vector, dtype=wp.float32, device="cuda")

        w = self.spec.cost_weights
        self.weights_wp = wp.array([w["w_z"], w["w_xy"], w["w_term"]], dtype=wp.float32, device="cuda")

    def get_inital_state(self, rng: np.random.Generator):
        mjm = self.mjm
        q0  = mjm.qpos0.copy()
        # Small random offset in x,y above the hole
        peg_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
        if peg_jnt >= 0:
            adr = mjm.jnt_qposadr[peg_jnt]
            q0[adr:adr+2] += rng.uniform(-0.003, 0.003, 2)
        return q0, np.zeros(mjm.nv), None

    def cost_fn(self, qpos, qvel, ctrl, terminal: bool, goal, indices) -> np.ndarray:
        """Penalize peg height (reward insertion depth) + lateral misalignment."""
        qpos_np = np.asarray(qpos.numpy() if hasattr(qpos, "numpy") else qpos)
        adr = indices[0]
        target_z, target_xy = goal[0], goal[1:]

        # z: reward insertion (minimize height above hole bottom)
        z_err   = np.abs(qpos_np[:, adr+2] - target_z)
        # x,y: penalize lateral offset
        xy_err  = np.linalg.norm(qpos_np[:, adr:adr+2] - target_xy, axis=-1)
        cost    = (z_err + 5.0 * xy_err).astype(np.float32)
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