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
def grasp_reorient_cost_wp(
    qpos:    wp.array(dtype=float),
    qvel:    wp.array(dtype=float),
    ctrl:    wp.array(dtype=float),
    terminal: bool,
    goal:    wp.array(dtype=float),
    indices: wp.array(dtype=int),
) -> float:
    """GPU cost function matching paper Eq. 15.

    goal layout  (23 values):
        [0:3]   target position   (p_target)
        [3:7]   target quaternion (q_target)
        [7:23]  reference joint positions (q_ref, 16 values)

    indices layout (4 values):
        [0]  obj qpos address   (obj_qpos_adr)
        [1]  obj qvel address   (obj_qvel_adr)
        [2]  robot qpos start   (robot_qpos_adr, 0 for Allegro)
        [3]  n_manip            (number of manipulator joints, e.g. 16)

    NOTE: The fingertip-to-object contact term (c_contact) requires
    full forward kinematics and cannot be computed from qpos alone on
    the GPU without a kinematic chain kernel.  It is therefore omitted
    here; the numpy cost_fn computes it for single-environment evaluation.
    """
    obj_qpos_adr  = indices[0]
    obj_qvel_adr  = indices[1]
    robot_qpos_adr = indices[2]
    n_manip        = indices[3]

    # ---- 1. Orientation error  c_quat = 1 − (q_target · q_obj)² ---------
    dot_prod = (
        qpos[obj_qpos_adr + 3] * goal[3] +
        qpos[obj_qpos_adr + 4] * goal[4] +
        qpos[obj_qpos_adr + 5] * goal[5] +
        qpos[obj_qpos_adr + 6] * goal[6]
    )
    c_quat = 1.0 - dot_prod * dot_prod

    # ---- 2. Per-axis absolute position errors ----------------------------
    px = wp.abs(qpos[obj_qpos_adr + 0] - goal[0])
    py = wp.abs(qpos[obj_qpos_adr + 1] - goal[1])
    pz = wp.abs(qpos[obj_qpos_adr + 2] - goal[2])

    # ---- 3. Joint deviation from home  c_joint = ||q_robot − q_ref||² ---
    c_joint = float(0.0)
    for i in range(n_manip):
        diff = qpos[robot_qpos_adr + i] - goal[7 + i]
        c_joint = c_joint + diff * diff

    # ---- 4. Fallen indicator  I_fallen = 1 if p_z < GRASP_FALLEN_Z ------
    fallen = float(0.0)
    if qpos[obj_qpos_adr + 2] < 0.05:
        fallen = 1.0

    # ---- 5. Running cost ---------------------------------------------------
    cost = (
        1.5  * c_quat   +   # ω1
        1.0  * px       +   # ω2
        1.0  * py       +   # ω3
        1.0  * pz       +   # ω4
        # ω5 · c_contact omitted (needs FK)
        0.01 * c_joint  +   # ω6
        100.0 * fallen      # Ω
    )

    # ---- 6. Terminal cost  V = ε1·||Δp||² + ε2·c_quat -------------------
    if terminal:
        dx = qpos[obj_qpos_adr + 0] - goal[0]
        dy = qpos[obj_qpos_adr + 1] - goal[1]
        dz = qpos[obj_qpos_adr + 2] - goal[2]
        pos_err_sq = dx*dx + dy*dy + dz*dz
        cost = 10.0 * pos_err_sq + 10.0 * c_quat

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
        box_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
        self.index_vector = np.array([mjm.jnt_qposadr[box_jnt]], dtype=np.int32)
        
        target_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "target")
        # Extract target position from site
        self.goal_vector = mjm.site_pos[target_id][:2].astype(np.float32)
        
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

    Cost function from paper Eq. 15:

    Running:
        c = ω1·c_quat + ω2·|Δpx| + ω3·|Δpy| + ω4·|Δpz|
          + ω5·c_contact + ω6·c_joint + Ω·I_fallen

    Terminal:
        V = ε1·||p_obj − p_target||² + ε2·(1 − (q_target·q_obj)²)

    where
        c_quat    = 1 − (q_target · q_obj)²          (quaternion dot-product error)
        c_contact = Σ_i ||p_obj − p_fingertip_i||²   (4 fingertips, single-env only)
        c_joint   = ||q_robot − q_ref||²              (deviation from home pose)
        I_fallen  = 1  if  p_z < GRASP_FALLEN_Z      (object dropped below workspace box)
    """

    @property
    def spec(self) -> TaskSpec:
        return TaskSpec(
            name              = "grasp_reorient",
            complexity        = ContactComplexity.MEDIUM,
            xml_path_template = "scenes/test_data/allegro/allegro_right_hand_armature.xml",
            max_steps         = 500,
            success_threshold = 0.05,  # combined pose error
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize_task(self):
        mjm = self.mjm

        # ---- object joint addresses -----------------------------------
        obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
        obj_qpos_adr = mjm.jnt_qposadr[obj_jnt]
        obj_qvel_adr = mjm.jnt_dofadr[obj_jnt]

        # ---- target pose from model site ------------------------------
        target_id    = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "obj_target")
        target_pos   = mjm.site_pos[target_id].copy()    # (3,)
        target_quat  = mjm.site_quat[target_id].copy()   # (4,)

        # ---- fingertip site IDs (graceful fallback) ------------------
        self._fingertip_site_ids: list[int] = []
        for name in ALLEGRO_FINGERTIP_SITES:
            sid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                self._fingertip_site_ids.append(sid)

        # ---- numpy helpers -------------------------------------------
        self._obj_qpos_adr = obj_qpos_adr
        self._obj_qvel_adr = obj_qvel_adr
        self._target_pos   = target_pos
        self._target_quat  = target_quat
        n_manip = len(MANIPULATOR_HOME_STATE)
        self._n_manip      = n_manip

        # ---- goal vector for warp  (23 values) -----------------------
        #  [0:3]  target_pos, [3:7] target_quat, [7:23] q_ref (16 joints)
        self.goal_vector = np.concatenate(
            [target_pos, target_quat, MANIPULATOR_HOME_STATE]
        ).astype(np.float32)

        # ---- index vector for warp  (4 values) -----------------------
        #  [0] obj_qpos_adr, [1] obj_qvel_adr, [2] robot_qpos_adr (0), [3] n_manip
        self.index_vector = np.array(
            [obj_qpos_adr, obj_qvel_adr, 0, n_manip], dtype=np.int32
        )

        self.goal_vector_wp  = wp.array(self.goal_vector,  dtype=wp.float32, device="cuda")
        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32,   device="cuda")

    # ------------------------------------------------------------------
    # Initial state sampling
    # ------------------------------------------------------------------

    def sample_initial_state(self, rng: np.random.Generator):
        mjm   = self.mjm
        q0    = mjm.qpos0.copy()
        v0    = np.zeros(mjm.nv, dtype=np.float32)
        ctrl0 = np.zeros(mjm.nu, dtype=np.float32)

        # 1. Set manipulator to the predefined home state
        n_manip = len(MANIPULATOR_HOME_STATE)
        if mjm.nq >= n_manip:
            q0[:n_manip] = MANIPULATOR_HOME_STATE
        if mjm.nu >= n_manip:
            ctrl0[:n_manip] = MANIPULATOR_HOME_STATE

        # 2. Randomize object position and orientation
        obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
        if obj_jnt >= 0:
            adr = mjm.jnt_qposadr[obj_jnt]
            # Uncomment to add randomisation:
            # q0[adr:adr+2] += rng.uniform(-0.03, 0.03, 2)
            # q0[adr+3:adr+7] += rng.uniform(-0.2, 0.2, 4)

        return q0, v0, ctrl0

    # ------------------------------------------------------------------
    # Cost function (numpy, CPU) — paper Eq. 15
    # ------------------------------------------------------------------

    def cost_fn(self, qpos, qvel, ctrl, terminal: bool) -> np.ndarray:
        """Implements the paper's running and terminal costs (Eq. 15).

        Running cost per step:
            c = ω1·c_quat + ω2·|Δpx| + ω3·|Δpy| + ω4·|Δpz|
              + ω5·c_contact + ω6·c_joint + Ω·I_fallen

        Terminal cost (replaces running cost at last step):
            V = ε1·||Δp||² + ε2·c_quat

        The fingertip contact term (ω5·c_contact) is computed only when
        qpos is a single state vector (non-batched), because it requires a
        MuJoCo forward-kinematics pass.  For batched rollouts it is skipped.
        """
        qpos_np = np.asarray(qpos.numpy() if hasattr(qpos, "numpy") else qpos)
        qvel_np = np.asarray(qvel.numpy() if hasattr(qvel, "numpy") else qvel)
        batched = qpos_np.ndim == 2
        nworld  = qpos_np.shape[0] if batched else 1

        adr          = self._obj_qpos_adr
        v_adr        = self._obj_qvel_adr
        target_pos   = self._target_pos    # (3,)
        target_quat  = self._target_quat   # (4,)
        n_manip      = self._n_manip

        # ---- Object position and orientation -------------------------
        if batched:
            obj_pos  = qpos_np[:, adr:adr+3]     # (N, 3)
            obj_quat = qpos_np[:, adr+3:adr+7]   # (N, 4)
        else:
            obj_pos  = qpos_np[adr:adr+3]         # (3,)
            obj_quat = qpos_np[adr+3:adr+7]       # (4,)

        # ---- Terminal cost  V = ε1·||Δp||² + ε2·c_quat --------------
        if terminal:
            pos_err_sq = np.sum((obj_pos - target_pos) ** 2, axis=-1)
            dot_prod   = (
                np.sum(obj_quat * target_quat, axis=-1) if batched
                else np.dot(obj_quat, target_quat)
            )
            c_quat = 1.0 - dot_prod ** 2
            cost   = (GRASP_EPS1 * pos_err_sq + GRASP_EPS2 * c_quat).astype(np.float32)
            return cost.reshape(nworld) if not batched else cost

        # ---- 1. Orientation error  c_quat ----------------------------
        dot_prod = (
            np.sum(obj_quat * target_quat, axis=-1) if batched
            else np.dot(obj_quat, target_quat)
        )
        c_quat = (1.0 - dot_prod ** 2)                      # scalar or (N,)

        # ---- 2. Per-axis absolute position errors --------------------
        delta_pos = obj_pos - target_pos                     # (3,) or (N,3)
        if batched:
            px = np.abs(delta_pos[:, 0])
            py = np.abs(delta_pos[:, 1])
            pz = np.abs(delta_pos[:, 2])
        else:
            px = abs(float(delta_pos[0]))
            py = abs(float(delta_pos[1]))
            pz = abs(float(delta_pos[2]))

        # ---- 3. Joint deviation from home  c_joint = ||q − q_ref||² -
        if batched:
            robot_q = qpos_np[:, :n_manip]                  # (N, n_manip)
            c_joint = np.sum((robot_q - MANIPULATOR_HOME_STATE) ** 2, axis=-1)
        else:
            c_joint = float(np.sum((qpos_np[:n_manip] - MANIPULATOR_HOME_STATE) ** 2))

        # ---- 4. Fallen indicator  I_fallen = 1 if p_z < threshold ---
        #   "Object is in some box above the hand": fallen when it exits
        #   that box by dropping below GRASP_FALLEN_Z (default 0.05 m).
        if batched:
            obj_z  = obj_pos[:, 2]
        else:
            obj_z  = float(obj_pos[2])
        fallen = (np.asarray(obj_z) < GRASP_FALLEN_Z).astype(np.float32)

        # ---- 5. Fingertip contact term  c_contact (single-env only) --
        #   c_contact = Σ_i ||p_obj − p_fingertip_i||²
        #   Requires a MuJoCo FK pass; only feasible for single states.
        c_contact = np.zeros(nworld, dtype=np.float32)
        if not batched and self._fingertip_site_ids:
            # Build a temporary MjData, set qpos, run forward kinematics
            mjd_tmp = mujoco.MjData(self.mjm)
            mjd_tmp.qpos[:] = qpos_np
            mujoco.mj_kinematics(self.mjm, mjd_tmp)
            obj_pos_fk = mjd_tmp.xpos[
                mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
            ]
            contact_sum = sum(
                np.sum((obj_pos_fk - mjd_tmp.site_xpos[sid]) ** 2)
                for sid in self._fingertip_site_ids
            )
            c_contact[0] = float(contact_sum)

        # ---- 6. Assemble running cost --------------------------------
        cost = (
            GRASP_W_QUAT    * c_quat    +
            GRASP_W_PX      * px        +
            GRASP_W_PY      * py        +
            GRASP_W_PZ      * pz        +
            GRASP_W_CONTACT * c_contact.squeeze() +
            GRASP_W_JOINT   * c_joint   +
            GRASP_W_FALLEN  * fallen
        )
        return np.asarray(cost, dtype=np.float32).reshape(nworld) if not batched else cost.astype(np.float32)

    # ------------------------------------------------------------------
    # Warp cost function (GPU) — paper Eq. 15, c_contact omitted
    # ------------------------------------------------------------------

    @property
    def cost_fn_wp(self) -> tuple[wp.func, wp.array, wp.array]:
        return grasp_reorient_cost_wp, self.goal_vector_wp, self.index_vector_wp

    # ------------------------------------------------------------------
    # Success criterion
    # ------------------------------------------------------------------

    def is_success(self, mjd: mujoco.MjData) -> bool:
        mjm = self.mjm
        obj_id    = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
        target_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "obj_target")

        if obj_id < 0 or target_id < 0:
            return False

        # Object must not have been dropped
        if mjd.xpos[obj_id, 2] < GRASP_FALLEN_Z:
            return False

        pos_err  = np.linalg.norm(mjd.xpos[obj_id] - mjd.site_xpos[target_id])
        obj_quat = mjd.xquat[obj_id]
        target_quat = mjm.site_quat[target_id]
        quat_err = 1.0 - np.dot(obj_quat, target_quat) ** 2
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