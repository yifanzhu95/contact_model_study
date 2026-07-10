"""Balance-stick task — a UR5e arm holds a plate and must keep a free-standing
stick balanced upright on top of it (an inverted-pendulum-on-a-tray problem).

Rollout and eval both use the same precompiled MJCF (scenes/balance_stick/
ur5e_balance_stick.xml, GeometryVariant-free — no {geometry} template). There
is no Drake/URDF counterpart, so eval defaults to MuJoCo.

Cost is built from three terms, evaluated against the "center_of_plate" site
and the stick's "end1"/"end2" tip sites (both already present in the scene):
  - tilt:  how far the stick has fallen from vertical (0 = upright).
  - home:  how far the 6 arm joints are from the "home" keyframe pose.
  - plate: distance from the stick's closest end to the plate center — i.e.
           how far the stick has drifted/toppled off the plate.
"""

from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from .base import BaseTask, ContactComplexity, register, SCENES_DIR
from .config import TaskConfig, EvalSimulatorKind

BALANCE_SCENE_XML = "balance_stick/ur5e_balance_stick.xml"

# Arm joints, in qpos/ctrl order (matches the "home" keyframe in the XML).
_ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# "home" keyframe values, duplicated here so the cost's arm target doesn't
# depend on re-reading the keyframe at runtime.
_HOME_ARM_QPOS = np.array(
    [-1.5708, -1.5708, 1.5708, 0.0, 1.5708, 0.0], dtype=np.float64
)
_STICK_INIT_POS  = np.array([-0.13399598, 0.59876665, 0.5825799], dtype=np.float64)
_STICK_INIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@wp.func
def balance_stick_cost_wp(qpos: wp.array(dtype=float),
                           qvel: wp.array(dtype=float),
                           ctrl: wp.array(dtype=float),
                           site_xpos: wp.array(dtype=wp.vec3),
                           site_xmat: wp.array(dtype=wp.mat33),
                           terminal: bool,
                           goal: wp.array(dtype=float),
                           indices: wp.array(dtype=int),
                           weights: wp.array(dtype=float)) -> float:
    arm_qpos_adr = indices[0]
    n_arm        = indices[1]
    plate_site   = indices[2]
    end1_site    = indices[3]
    end2_site    = indices[4]

    p_plate = site_xpos[plate_site]
    p_end1  = site_xpos[end1_site]
    p_end2  = site_xpos[end2_site]

    # How far the stick has fallen: 1 - cos(angle from vertical), using the
    # end1->end2 vector as the stick's long axis. 0 when upright, up to 2
    # when fully inverted.
    stick_vec = p_end2 - p_end1
    stick_len = wp.length(stick_vec)
    c_tilt = 1.0 - stick_vec[2] / stick_len

    # Distance from the plate center to whichever end of the stick is closer
    # (the stick topples/slides off the plate in either direction).
    d1_sq = wp.length_sq(p_end1 - p_plate)
    d2_sq = wp.length_sq(p_end2 - p_plate)
    c_plate = wp.min(d1_sq, d2_sq)

    # Distance of the 6 arm joints from the "home" pose.
    c_home = float(0.0)
    for i in range(n_arm):
        dq = qpos[arm_qpos_adr + i] - goal[i]
        c_home = c_home + dq * dq

    if terminal:
        return (
            weights[3] * c_tilt +
            weights[4] * c_home +
            weights[5] * c_plate
        )
    return (
        weights[0] * c_tilt +
        weights[1] * c_home +
        weights[2] * c_plate
    )


@register("balance_stick")
class BalanceStickTask(BaseTask):
    """Keep a free stick balanced upright on a plate held by a UR5e arm.

    Contact complexity: LOW (the stick rests on the plate; no grasping).
    """

    # Half-angle (rad) of random tilt noise added to the stick at reset.
    init_tilt_noise: float = 0.05

    def __init__(self, geometry=None, role=None):
        kwargs = {}
        if geometry is not None:
            kwargs["geometry"] = geometry
        if role is not None:
            kwargs["role"] = role
        super().__init__(**kwargs)

        self.config = TaskConfig(
            name               = "balance_stick",
            complexity         = ContactComplexity.LOW,
            max_steps          = 500,
            success_thresholds = {"tilt": 0.1, "plate": 0.05, "vel": 0.5},
            cost_weights       = {
                "w_tilt":      200.0,
                "w_home":      2.0,
                "w_plate":     300.0,
                "w_tilt_term": 400.0,
                "w_home_term": 2.0,
                "w_plate_term":600.0,
            },
            # No {geometry} variant for this scene.
            xml_path_template  = BALANCE_SCENE_XML,
            rollout_model_path = str(SCENES_DIR / BALANCE_SCENE_XML),
            rollout_is_urdf    = False,
            eval_sim           = EvalSimulatorKind.MUJOCO,
            eval_model_paths   = {EvalSimulatorKind.MUJOCO: str(SCENES_DIR / BALANCE_SCENE_XML)},
            cam_pos            = (1.2, -1.2, 0.8),
            cam_fps            = 30.0,
            timestep           = 0.002,
            eval_substeps_per_rollout = 10,
        )

    def initialize_task(self):
        mjm = self.mjm
        arm_qpos_adr = mjm.jnt_qposadr[
            mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, _ARM_JOINTS[0])
        ]
        plate_site = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "center_of_plate")
        end1_site  = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "end1")
        end2_site  = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "end2")

        self.index_vector = np.array([
            arm_qpos_adr, len(_ARM_JOINTS), plate_site, end1_site, end2_site,
        ], dtype=np.int32)

        self.goal_vector = _HOME_ARM_QPOS.astype(np.float32).copy()

        w = self.config.cost_weights
        weights_list = [
            w["w_tilt"], w["w_home"], w["w_plate"],
            w["w_tilt_term"], w["w_home_term"], w["w_plate_term"],
        ]

        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32, device="cuda")
        self.goal_vector_wp  = wp.array(self.goal_vector, dtype=wp.float32, device="cuda")
        self.weights_wp      = wp.array(weights_list, dtype=wp.float32, device="cuda")

    def get_inital_state(self, rng: np.random.Generator):
        mjm = self.mjm
        q0 = np.zeros(mjm.nq, dtype=np.float64)
        q0[0:6] = _HOME_ARM_QPOS
        q0[6:9] = _STICK_INIT_POS

        # Small random tilt about a random horizontal axis, so the balance
        # controller has to react rather than starting perfectly upright.
        axis = rng.normal(size=3)
        axis[2] = 0.0
        norm = np.linalg.norm(axis)
        if norm > 1e-8:
            axis /= norm
        angle = rng.uniform(-self.init_tilt_noise, self.init_tilt_noise)
        dquat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dquat, axis, float(angle))
        q0[9:13] = np.zeros(4)
        mujoco.mju_mulQuat(q0[9:13], dquat, _STICK_INIT_QUAT)

        v0 = np.zeros(mjm.nv, dtype=np.float64)
        ctrl0 = _HOME_ARM_QPOS.copy()
        return q0, v0, ctrl0

    @property
    def cost_fn_wp(self) -> wp.func:
        return balance_stick_cost_wp

    def _tilt_and_plate_err(self, mjd: mujoco.MjData) -> tuple[float, float]:
        plate_site, end1_site, end2_site = self.index_vector[2:5]
        p_plate = mjd.site_xpos[plate_site]
        p_end1  = mjd.site_xpos[end1_site]
        p_end2  = mjd.site_xpos[end2_site]

        stick_vec = p_end2 - p_end1
        tilt = 1.0 - stick_vec[2] / np.linalg.norm(stick_vec)
        plate_err = min(
            np.linalg.norm(p_end1 - p_plate), np.linalg.norm(p_end2 - p_plate)
        )
        return float(tilt), float(plate_err)

    def is_success(self, mjd: mujoco.MjData) -> bool:
        tilt, plate_err = self._tilt_and_plate_err(mjd)
        thr = self.config.success_thresholds
        return bool(
            tilt < thr["tilt"]
            and plate_err < thr["plate"]
            and np.linalg.norm(mjd.qvel) < thr["vel"]
        )

    def has_failed(self, mjd: mujoco.MjData) -> bool:
        # Fallen (>~60 deg from vertical) or slid/toppled off the plate.
        tilt, plate_err = self._tilt_and_plate_err(mjd)
        return bool(tilt > 1.0 or plate_err > 0.3)
