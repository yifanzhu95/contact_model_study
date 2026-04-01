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

    def sample_initial_state(self, rng: np.random.Generator):
        mjm = self.mjm
        q0  = mjm.qpos0.copy()
        # Randomize box x,y position within ±0.1 m of nominal
        box_qpos_adr = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "box_freejoint")
        if box_qpos_adr >= 0:
            adr = mjm.jnt_qposadr[box_qpos_adr]
            q0[adr:adr+2] += rng.uniform(-0.1, 0.1, 2)
        return q0, np.zeros(mjm.nv)

    def cost_fn(self, qpos, qvel, ctrl, terminal: bool) -> np.ndarray:
        """L2 distance of box position to target."""
        # Target position is stored in a site named "target"
        # Here we use a placeholder: minimize qpos[7:9] distance to origin
        qpos_np = np.asarray(qpos.numpy() if hasattr(qpos, "numpy") else qpos)
        # box freejoint qpos starts at index 7 (after robot joints)
        # This is task-specific; adjust index for your XML
        target = np.array([0.5, 0.0])
        box_pos = qpos_np[:, 7:9] if qpos_np.ndim == 2 else qpos_np[7:9]
        cost = np.linalg.norm(box_pos - target, axis=-1).astype(np.float32)
        if terminal:
            cost *= 10.0
        return cost

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
            xml_path_template = "tasks/grasp_reorient_{geometry}.xml",
            max_steps         = 300,
            success_threshold = 0.05,  # combined pose error
        )

    def sample_initial_state(self, rng: np.random.Generator):
        mjm = self.mjm
        q0  = mjm.qpos0.copy()
        # Randomize object yaw orientation
        obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        if obj_jnt >= 0:
            adr = mjm.jnt_qposadr[obj_jnt]
            # perturb quaternion slightly (small yaw)
            dtheta = rng.uniform(-0.3, 0.3)
            q0[adr+6] += dtheta  # w component (unnormalized; mj_forward normalizes)
        return q0, np.zeros(mjm.nv)

    def cost_fn(self, qpos, qvel, ctrl, terminal: bool) -> np.ndarray:
        """Weighted sum of position error + orientation error."""
        qpos_np = np.asarray(qpos.numpy() if hasattr(qpos, "numpy") else qpos)
        nworld  = qpos_np.shape[0] if qpos_np.ndim == 2 else 1

        # Placeholder: penalize distance of object freejoint from target
        target_pos = np.array([0.0, 0.0, 0.4])
        obj_start  = 7  # adjust to match XML
        pos_err  = np.linalg.norm(qpos_np[:, obj_start:obj_start+3] - target_pos, axis=-1)
        cost     = pos_err.astype(np.float32)
        if terminal:
            cost *= 20.0
        return cost

    def is_success(self, mjd: mujoco.MjData) -> bool:
        obj_id    = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "object")
        target_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, "object_target")
        if obj_id < 0 or target_id < 0:
            return False
        pos_err = np.linalg.norm(mjd.xpos[obj_id] - mjd.site_xpos[target_id])
        # Orientation error via quaternion distance
        obj_quat    = mjd.xquat[obj_id]
        target_site = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, "object_target")
        # Use position error only for now; add quat distance for full eval
        return bool(pos_err < self.spec.success_threshold)


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

    def sample_initial_state(self, rng: np.random.Generator):
        mjm = self.mjm
        q0  = mjm.qpos0.copy()
        # Small random offset in x,y above the hole
        peg_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
        if peg_jnt >= 0:
            adr = mjm.jnt_qposadr[peg_jnt]
            q0[adr:adr+2] += rng.uniform(-0.003, 0.003, 2)
        return q0, np.zeros(mjm.nv)

    def cost_fn(self, qpos, qvel, ctrl, terminal: bool) -> np.ndarray:
        """Penalize peg height (reward insertion depth) + lateral misalignment."""
        qpos_np = np.asarray(qpos.numpy() if hasattr(qpos, "numpy") else qpos)
        peg_start = 7  # adjust to match XML

        # z: reward insertion (minimize height above hole bottom)
        z_err   = np.abs(qpos_np[:, peg_start+2] - (-0.05))   # target z = -5cm
        # x,y: penalize lateral offset
        xy_err  = np.linalg.norm(qpos_np[:, peg_start:peg_start+2], axis=-1)
        cost    = (z_err + 5.0 * xy_err).astype(np.float32)
        if terminal:
            cost *= 30.0
        return cost

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
