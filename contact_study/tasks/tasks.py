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
# Global Task Constants
# ---------------------------------------------------------------------------

# Predefined home position for the manipulator joints (e.g., 16 joints for Allegro Hand)
MANIPULATOR_HOME_STATE = np.array([
    0.127, 0.5, 1.5, 1.0,  # Index
    0.0, 0.3, 1.42, 1.0,  # Middle
    -0.127, 0.5, 1.5, 1.0,  # Ring
    0.25, 1.5, 1.7, 1.0   # Thumb
], dtype=np.float32)

# ---------------------------------------------------------------------------
# GraspReorient cost weights (paper Eq. 15)
# Tune ω_k and (ε1, ε2) based on object mass and geometry.
# ---------------------------------------------------------------------------

# Running-cost weights
GRASP_W_QUAT    = 1.5    # ω1 — orientation error  c_quat = 1 − (q_target·q_obj)²
GRASP_W_PX      = 1.0    # ω2 — absolute x-position error
GRASP_W_PY      = 1.0    # ω3 — absolute y-position error
GRASP_W_PZ      = 1.0    # ω4 — absolute z-position error
GRASP_W_CONTACT = 0.1    # ω5 — fingertip-to-object-center distance (single-env only)
GRASP_W_JOINT   = 0.01   # ω6 — deviation of robot joints from home pose
GRASP_W_FALLEN  = 100.0  # Ω  — large penalty when object is dropped

# Terminal-cost weights
GRASP_EPS1 = 10.0   # ε1 — ||p_obj − p_target||²
GRASP_EPS2 = 10.0   # ε2 — (1 − (q_target·q_obj)²)

# Drop-detection: object is considered fallen if its z-coordinate drops
# below this threshold (the "box above the hand" lower boundary, in metres).
GRASP_FALLEN_Z = 0.05

# Fingertip site names for the Allegro hand (order: index, middle, ring, thumb)
ALLEGRO_FINGERTIP_SITES = [
    "index_fingertip",
    "middle_fingertip",
    "ring_fingertip",
    "thumb_fingertip",
]

# ---------------------------------------------------------------------------
# Warp Cost Functions (GPU-side)
# ---------------------------------------------------------------------------

@wp.func
def push_cost_wp(qpos: wp.array(dtype=float), qvel: wp.array(dtype=float), ctrl: wp.array(dtype=float), 
                 terminal: bool, goal: wp.array(dtype=float), indices: wp.array(dtype=int)) -> float:
    # indices[0]: Box qpos address
    adr = indices[0]
    dx = qpos[adr] - goal[0]
    dy = qpos[adr + 1] - goal[1]
    dist = wp.sqrt(dx*dx + dy*dy)
    if terminal:
        return dist * 10.0
    return dist


@wp.func
def grasp_reorient_cost_wp(qpos: wp.array(dtype=float), 
                           qvel: wp.array(dtype=float), 
                           ctrl: wp.array(dtype=float), 
                           terminal: bool, 
                           goal: wp.array(dtype=float), 
                           indices: wp.array(dtype=int),
                           xpos: wp.array(dtype=wp.vec3),   
                           xquat: wp.array(dtype=wp.quat)) -> float:
    # Index Mapping:
    # 0: obj_qpos_adr, 1: obj_qvel_adr, 2: robot_qpos_adr, 3: n_manip
    # 4: obj_body_id, 5-8: fingertip body IDs
    obj_qpos_adr   = indices[0]
    robot_qpos_adr = indices[2]
    n_manip        = indices[3]
    obj_id         = indices[4]

    # Current object state from MuJoCo xpos/xquat
    p_obj = xpos[obj_id]
    q_obj = xquat[obj_id] 

    # Target state from goal vector
    p_target = wp.vec3(goal[0], goal[1], goal[2])
    # MuJoCo quat is (w, x, y, z). goal[3:7] follows this.
    q_target = wp.vec4(goal[3], goal[4], goal[5], goal[6])
    q_obj_v4 = wp.vec4(q_obj.w, q_obj.x, q_obj.y, q_obj.z)

    # ---- 1. Orientation error: c_quat = 1 - (q_target^T * q_obj)^2 ----
    dot_prod = wp.dot(q_target, q_obj_v4)
    c_quat = 1.0 - dot_prod * dot_prod

    # ---- 2. Per-axis absolute position errors (px, py, pz) ----
    px = wp.abs(p_obj[0] - p_target[0])
    py = wp.abs(p_obj[1] - p_target[1])
    pz = wp.abs(p_obj[2] - p_target[2])

    # ---- 3. Joint deviation: c_joint = ||q_robot - q_ref||^2 ----
    c_joint = float(0.0) 
    for i in range(n_manip):
        # Renamed to 'dq' to avoid type clash with 'dp' later
        dq = qpos[robot_qpos_adr + i] - goal[7 + i]
        c_joint = c_joint + dq * dq

    # ---- 4. Fallen indicator: I_fallen = 1 if p_z < 0.05 ----
    fallen = float(0.0)
    if p_obj[2] < 0.05:
        fallen = 1.0

    # ---- 5. Contact cost: c_contact = sum(||p_obj - p_tip_i||^2) ----
    c_contact = float(0.0)
    for i in range(5, 9):
        p_tip = xpos[indices[i]]
        # Renamed to 'dp' (vec3 type) to keep types distinct
        dp = p_obj - p_tip
        c_contact = c_contact + wp.dot(dp, dp)

    # ---- 6. Running cost (Equations 14 & 15 from paper) ----
    # Weights: w1=1.5, w2-w4=1.0, w5=5.0, w6=0.01, Omega=100.0
    cost = (0.5 * c_quat) + (1.0 * px) + (1.0 * py) + (1.0 * pz) + \
           (0.1 * c_contact) + (0.01 * c_joint) + (10.0 * fallen)

    # ---- 7. Terminal cost: V = phi1*||dp||^2 + phi2*c_quat ----
    if terminal:
        d_pos_terminal = p_obj - p_target
        pos_err_sq = wp.dot(d_pos_terminal, d_pos_terminal)
        # Weights: phi1=10.0, phi2=10.0
        return 10.0 * pos_err_sq + 10.0 * c_quat

    return cost


@wp.func
def peg_in_hole_cost_wp(qpos: wp.array(dtype=float), qvel: wp.array(dtype=float), ctrl: wp.array(dtype=float), 
                        terminal: bool, goal: wp.array(dtype=float), indices: wp.array(dtype=int)) -> float:
    # indices[0]: Peg qpos address. goal: [target_z, target_x, target_y]
    adr = indices[0]
    z_err = wp.abs(qpos[adr + 2] - goal[0])
    dx = qpos[adr] - goal[1]
    dy = qpos[adr + 1] - goal[2]
    xy_err = wp.sqrt(dx*dx + dy*dy)
    cost = z_err + 5.0 * xy_err
    if terminal:
        return cost * 30.0
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
        )

    def initialize_task(self):
        mjm = self.mjm
        # Get joint and body indices
        obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
        obj_body = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
        
        # Fingertip body IDs for contact cost
        tips = ["ff_tip", "mf_tip", "rf_tip", "th_tip"]
        tip_ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, t) for t in tips]

        # 1. Index Vector Mapping
        self.index_vector = np.array([
            mjm.jnt_qposadr[obj_jnt], # 0: obj qpos
            mjm.jnt_dofadr[obj_jnt],  # 1: obj qvel
            0,                        # 2: robot qpos starts at 0
            16,                       # 3: n_manip (Allegro has 16 joints)
            obj_body,                 # 4: obj body id
            *tip_ids                  # 5-8: fingertip ids
        ], dtype=np.int32)
        
        # 2. Goal Vector Mapping
        target_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "obj_target")
        # goal stores: [pos(3), quat(4), q_ref(16)]
        self.goal_vector = np.concatenate([
            mjm.site_pos[target_id], 
            mjm.site_quat[target_id], 
            MANIPULATOR_HOME_STATE
        ]).astype(np.float32)
        
        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32, device="cuda")
        self.goal_vector_wp = wp.array(self.goal_vector, dtype=wp.float32, device="cuda")

    def sample_initial_state(self, rng: np.random.Generator):
        mjm = self.mjm
        q0  = mjm.qpos0.copy()
        # Randomize box x,y position within ±0.1 m of nominal
        box_qpos_adr = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
        if box_qpos_adr >= 0:
            adr = mjm.jnt_qposadr[box_qpos_adr]
            q0[adr:adr+2] += rng.uniform(-0.1, 0.1, 2)
        return q0, np.zeros(mjm.nv), None

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
    def cost_fn_wp(self) -> tuple[wp.func, wp.array, wp.array]:
        return push_cost_wp, self.goal_vector_wp, self.index_vector_wp

    def is_success(self, mjd: mujoco.MjData) -> bool:
        box_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "box")
        if box_id < 0:
            return False
        target_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, "target")
        if target_id < 0:
            return False
        box_pos    = mjd.xpos[box_id, :2]
        target_pos = mjd.site_xpos[target_id, :2]
        return bool(np.linalg.norm(box_pos - target_pos) < self.spec.success_threshold)


# ---------------------------------------------------------------------------
# Task 2: Grasp and Reorient (MEDIUM complexity)
# ---------------------------------------------------------------------------

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
            xml_path_template = "scenes/test_data/allegro/allegro_right_hand_armature.xml",
            max_steps         = 1000,
            success_threshold = 0.05,  # combined pose error
        )

    def initialize_task(self):
        mjm = self.mjm
        obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
        
        # Look up body IDs to calculate Cartesian spatial distances
        obj_body = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
        ff_tip   = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "ff_tip")
        mf_tip   = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "mf_tip")
        rf_tip   = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "rf_tip")
        th_tip   = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "th_tip")

        # Map addresses and IDs to the index vector for Warp
        self.index_vector = np.array([
            mjm.jnt_qposadr[obj_jnt], 
            mjm.jnt_dofadr[obj_jnt],
            obj_body,
            ff_tip,
            mf_tip,
            rf_tip,
            th_tip
        ], dtype=np.int32)
        
        target_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "obj_target")
        target_pos = mjm.site_pos[target_id]
        target_quat = mjm.site_quat[target_id]
        
        # Concatenate pos (3) and quat (4)
        self.goal_vector = np.concatenate([target_pos, target_quat]).astype(np.float32)
        
        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32, device="cuda")
        self.goal_vector_wp = wp.array(self.goal_vector, dtype=wp.float32, device="cuda")

    def sample_initial_state(self, rng: np.random.Generator):
        mjm = self.mjm
        q0  = mjm.qpos0.copy()
        v0  = np.zeros(mjm.nv, dtype=np.float32)
        ctrl0 = np.zeros(mjm.nu, dtype=np.float32)

        n_manip = len(MANIPULATOR_HOME_STATE)
        if mjm.nq >= n_manip:
            q0[:n_manip] = MANIPULATOR_HOME_STATE
        if mjm.nu >= n_manip:
            ctrl0[:n_manip] = MANIPULATOR_HOME_STATE

        return q0, v0, ctrl0

    def cost_fn(self, qpos, qvel, ctrl, terminal: bool, xpos, xquat) -> np.ndarray:
        """Weighted sum of position error + orientation error + velocity penalty + contact cost."""
        qpos_np = np.asarray(qpos.numpy() if hasattr(qpos, "numpy") else qpos)
        qvel_np = np.asarray(qvel.numpy() if hasattr(qvel, "numpy") else qvel)
        xpos_np = np.asarray(xpos.numpy() if hasattr(xpos, "numpy") else xpos)
        xquat_np = np.asarray(xquat.numpy() if hasattr(xquat, "numpy") else xquat)
        
        nworld = qpos_np.shape[0] if qpos_np.ndim == 2 else 1
        mjm = self.mjm
        
        obj_jnt   = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
        target_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "obj_target")

        # IDs unpacked from index_vector mapping
        obj_id  = self.index_vector[2]
        tip_ids = self.index_vector[3:7]

        if obj_jnt < 0 or target_id < 0:
            return np.zeros(nworld, dtype=np.float32)

        target_pos = mjm.site_pos[target_id]
        target_quat = mjm.site_quat[target_id]

        # Ensure correct batch indexing for xpos and xquat shapes
        obj_pos = xpos_np[:, obj_id] if xpos_np.ndim == 3 else xpos_np[obj_id]
        obj_quat_val = xquat_np[:, obj_id] if xquat_np.ndim == 3 else xquat_np[obj_id]

        # 1. Position error
        pos_err = np.linalg.norm(obj_pos - target_pos, axis=-1)

        # 2. Orientation error
        if xquat_np.ndim == 3:
            dot_prod = np.sum(obj_quat_val * target_quat, axis=-1)
        else:
            dot_prod = np.dot(obj_quat_val, target_quat)
        quat_err = 1.0 - dot_prod**2

        # 3. Velocity error
        v_adr = mjm.jnt_dofadr[obj_jnt]
        obj_vel = qvel_np[:, v_adr:v_adr+3] if qvel_np.ndim == 2 else qvel_np[v_adr:v_adr+3]
        vel_err = np.linalg.norm(obj_vel, axis=-1)

        # 4. Contact cost
        contact_cost = np.zeros(nworld, dtype=np.float32)
        for tip_id in tip_ids:
            tip_pos = xpos_np[:, tip_id] if xpos_np.ndim == 3 else xpos_np[tip_id]
            contact_cost += np.sum((obj_pos - tip_pos)**2, axis=-1)

        cost = (pos_err + 1.5 * quat_err + 0.1 * vel_err + 5.0 * contact_cost).astype(np.float32)
        
        if terminal:
            cost *= 20.0
        return cost

    @property
    def cost_fn_wp(self) -> wp.func:
        return grasp_reorient_cost_wp

    def is_success(self, mjd: mujoco.MjData) -> bool:
        mjm = self.mjm
        obj_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
        target_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "obj_target")

        if obj_id < 0 or target_id < 0:
            return False

        pos_err = np.linalg.norm(mjd.xpos[obj_id] - mjd.site_xpos[target_id])
        obj_quat = mjd.xquat[obj_id]
        target_quat = mjm.site_quat[target_id]
        quat_err = 1.0 - np.dot(obj_quat, target_quat)**2
        
        return bool(pos_err < self.spec.success_threshold and quat_err < self.spec.success_threshold)
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
        )

    def initialize_task(self):
        mjm = self.mjm
        peg_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
        self.index_vector = np.array([mjm.jnt_qposadr[peg_jnt]], dtype=np.int32)
        
        # Target: z = -0.05, xy = 0.0
        self.goal_vector = np.array([-0.05, 0.0, 0.0], dtype=np.float32)
        
        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32, device="cuda")
        self.goal_vector_wp = wp.array(self.goal_vector, dtype=wp.float32, device="cuda")

    def sample_initial_state(self, rng: np.random.Generator):
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
    def cost_fn_wp(self) -> tuple[wp.func, wp.array, wp.array]:
        return peg_in_hole_cost_wp, self.goal_vector_wp, self.index_vector_wp

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