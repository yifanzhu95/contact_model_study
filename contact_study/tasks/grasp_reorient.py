from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from .base import BaseTask, ContactComplexity, TaskSpec, register, SCENES_DIR
from .config import TaskConfig, EvalSimulatorKind

# --- LEAP hand: MuJoCo(rollout) <-> URDF(Drake eval) joint correspondence -----
# MuJoCo control/qpos order is [if_mcp,if_rot,if_pip,if_dip, mf_*, rf_*,
# th_cmc,th_axl,th_mcp,th_ipl]. The URDF (scenes/leap_hand/leap_hand_right.urdf)
# names joints "0".."15" in a different order; this list maps MuJoCo control
# index -> URDF joint name by *kinematic chain position* (validated against the
# per-joint limits — the 12 finger joints match exactly; see KNOWN GAP below).
#
# KNOWN GAP (eval fidelity): the two thumb middle joints (th_axl, th_mcp) have
# DIFFERENT limit ranges between the URDF and the MuJoCo hand, i.e. the thumb
# kinematics/zero-references differ between the two models. Finger behavior is
# faithful; thumb eval will be approximate until the URDF thumb is recalibrated.
_MJ_CTRL_TO_URDF_JOINT = [
    "1", "0", "2", "3",     # index : if_mcp, if_rot, if_pip, if_dip
    "5", "4", "6", "7",     # middle: mf_mcp, mf_rot, mf_pip, mf_dip
    "9", "8", "10", "11",   # ring  : rf_mcp, rf_rot, rf_pip, rf_dip
    "12", "13", "14", "15", # thumb : th_cmc, th_axl, th_mcp, th_ipl
]

# MuJoCo palm world pose (scenes/leap_hand_old/leap_right_hand.xml). The Drake
# hand's palm_lower link is welded here so the Drake world frame coincides with
# the MuJoCo world frame (so mirrored cube poses and the ground plane line up).
_PALM_POS  = (0.0, 0.0, 0.01)
_PALM_QUAT = (0.0, 0.97236992, 0.0, -0.23344536)  # wxyz
# URDF base_joint origin (base -> palm_lower), needed to weld the *base* link
# (the URDF root) such that palm_lower lands at the MuJoCo palm pose. (Welding
# palm_lower directly is rejected by Drake: it already has the base_joint inboard.)
_BASE_TO_PALM_XYZ = (0.0, 0.038, 0.098)
_BASE_TO_PALM_RPY = (0.0, -1.57079, 0.0)

# Free cube (matches scenes/leap_hand_old/scene_leap_cube.xml obj geom).
_CUBE_HALF = 0.035
_CUBE_MASS = 0.14
_CUBE_FRIC = 0.5
_FLOOR_Z   = -0.25       # MuJoCo floor height in world

# MuJoCo position-servo gains (leap_right_hand.xml default: kp=3.0 kv=0.01).
# MuJoCo's servo has no force limit here, so 0.95 (the URDF effort) is too weak
# to hold the hand; use a higher clamp. All three are starting points to tune
# against Drake's solver on the user's machine.
_PD_KP, _PD_KD, _PD_EFFORT = 3.0, 0.01, 5.0

@wp.func
def grasp_reorient_cost_wp(qpos: wp.array(dtype=float),
                           qvel: wp.array(dtype=float),
                           ctrl: wp.array(dtype=float),
                           site_xpos: wp.array(dtype=wp.vec3),
                           site_xmat: wp.array(dtype=wp.mat33),
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

    Goal difficulty levels (set via task.goal_difficulty):
        1 — Easiest: ±90° spin around the normal of the currently-shown face.
            The face stays the same; only the in-plane orientation changes.
        2 — Medium:  ±90° rotation around a randomly-chosen object-frame axis
            (X, Y, or Z).  May change which face is shown.
        3 — Hard:    Jump to a completely different cube face (5 candidates),
            with a random 90° twist on top.
    """

    # Controls which sampling method sample_new_goal dispatches to.
    # Override on an instance before the first episode to change difficulty.
    goal_difficulty: int = 1

    # Object-frame axis that is the surface normal for each face index.
    # Matches the face_twist_axes table in sample_new_goal_by_face.
    _FACE_NORMALS = [
        [0., 0., 1.],  # face 0: +Z up  (initial)
        [0., 1., 0.],  # face 1: +Y up
        [0., 1., 0.],  # face 2: -Y up
        [0., 0., 1.],  # face 3: -Z up
        [1., 0., 0.],  # face 4: -X up
        [1., 0., 0.],  # face 5: +X up
    ]

    def __init__(self, geometry=None, role=None):
        kwargs = {}
        if geometry is not None:
            kwargs["geometry"] = geometry
        if role is not None:
            kwargs["role"] = role
        super().__init__(**kwargs)

        s = self.spec  # legacy TaskSpec (rollout path); mirror into the config.
        self.config = TaskConfig(
            name               = s.name,
            complexity         = s.complexity,
            max_steps          = s.max_steps,
            cost_weights       = s.cost_weights,
            success_thresholds = s.success_thresholds,
            xml_path_template  = s.xml_path_template,
            rollout_model_path = str(SCENES_DIR / s.xml_path_template),
            rollout_is_urdf    = False,
            eval_sim           = EvalSimulatorKind.DRAKE,
            eval_model_path    = str(SCENES_DIR / "leap_hand/leap_hand_right.urdf"),
            cam_pos            = (0.0, -0.35, 0.12),
            cam_fps            = 30.0,
            timestep           = 0.005,
            difficulty         = self.goal_difficulty,
        )

    @property
    def spec(self) -> TaskSpec:
        return TaskSpec(
            name              = "grasp_reorient",
            complexity        = ContactComplexity.MEDIUM,
            xml_path_template = "leap_hand_old/scene_leap_cube.xml",
            max_steps          = 2500,
            success_thresholds = {"pos": 0.05, "quat": 0.05, "vel": 0.1},
            cost_weights      = {
                "w_quat": 5.0,
                "w_pos": 40.0,
                "w_velo": 0.0,
                "w_contact": 2.5,
                "w_joint": 0.1,
                "w_joint_velo": 0.0,
                "w_fallen": 30.0,
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
        self.target_pos          = self._mjd.mocap_pos[mocap_id]
        self.target_quat         = self._mjd.mocap_quat[mocap_id].copy()
        self._canonical_quat     = self.target_quat.copy()
        self._face_index         = 0

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
    def cost_fn_wp(self) -> wp.func:
        return grasp_reorient_cost_wp

    def is_success(self, mjd: mujoco.MjData) -> bool:
        obj_qpos_adr = int(self.index_vector[0])
        obj_qvel_adr = int(self.index_vector[1])

        pos  = mjd.qpos[obj_qpos_adr     : obj_qpos_adr + 3]
        quat = mjd.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7]
        vel  = mjd.qvel[obj_qvel_adr     : obj_qvel_adr + 6]

        pos_err  = np.linalg.norm(pos - self.target_pos)
        quat_err = 1.0 - np.dot(quat, self.target_quat) ** 2

        #print(pos,self.target_pos)
        #print(quat, self.target_quat)
        #print(pos_err,quat_err)

        thr     = self.spec.success_thresholds
        pose_ok = pos_err < thr["pos"] and quat_err < thr["quat"]
        if not pose_ok:
            return False

        return bool(np.linalg.norm(vel) < self.spec.success_thresholds["vel"])

    def has_failed(self, mjd: mujoco.MjData) -> bool:
        mjm = self.mjm
        obj_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "obj")
        if obj_id < 0:
            return False
        return bool(mjd.xpos[obj_id][2] < 0.0)

    def sample_new_goal_by_face(self, mjd: mujoco.MjData, rng: np.random.Generator):
        """Sample a new goal orientation corresponding to a different cube face.

        All 6 face orientations are derived from the canonical goal quaternion
        (set at load time) by right-multiplying with one of 6 object-frame
        rotations — one per cube face.  The current face index is tracked so
        the selected face is always different from the previous goal.
        """
        # Six object-frame rotations, one per cube face (right-multiply onto
        # canonical quat).  Using Rx/Ry only ensures all 6 faces are distinct:
        #   face 0 — identity          (initial face)
        #   face 1 — Rx +90°
        #   face 2 — Rx -90°
        #   face 3 — Rx 180°           (opposite face)
        #   face 4 — Ry +90°
        #   face 5 — Ry -90°
        def _aa(axis, angle):
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            return np.array([c, s * axis[0], s * axis[1], s * axis[2]])

        face_rots = [
            _aa([1, 0, 0],  0.0),
            _aa([1, 0, 0],  np.pi / 2),
            _aa([1, 0, 0], -np.pi / 2),
            _aa([1, 0, 0],  np.pi),
            _aa([0, 1, 0],  np.pi / 2),
            _aa([0, 1, 0], -np.pi / 2),
        ]

        # Pick uniformly from the 5 faces that are not the current one
        candidates = [i for i in range(6) if i != self._face_index]
        self._face_index = int(rng.choice(candidates))

        # Random 90° twist (0°/90°/180°/270°) around the selected face's normal
        twist_angle = int(rng.integers(0, 4)) * (np.pi / 2)
        q_twist = _aa(self._FACE_NORMALS[self._face_index], twist_angle)

        # Combine: face rotation then twist (both in object frame)
        q_face_twist = np.zeros(4)
        mujoco.mju_mulQuat(q_face_twist, face_rots[self._face_index], q_twist)

        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, self._canonical_quat, q_face_twist)

        self._update_goal(mjd, new_quat)

    def sample_new_goal_by_rot(self, mjd: mujoco.MjData, rng: np.random.Generator):
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

        self._update_goal(mjd, new_quat)

    def sample_new_goal_by_z_rot(self, mjd: mujoco.MjData, rng: np.random.Generator):
        """Difficulty 1: ±90° spin around the normal of the currently-shown face.

        The face stays the same — only the in-plane orientation changes.
        This is the easiest goal because the controller never needs to tip
        the cube onto a new face.
        """
        axis = np.array(self._FACE_NORMALS[self._face_index], dtype=float)
        angle = rng.choice([np.pi / 2.0, -np.pi / 2.0])

        c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
        q_rot = np.array([c, s * axis[0], s * axis[1], s * axis[2]])

        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, self.target_quat, q_rot)

        self._update_goal(mjd, new_quat)

    def sample_new_goal(self, mjd: mujoco.MjData, rng: np.random.Generator):
        """Dispatch to the appropriate sampler based on self.goal_difficulty."""
        if self.goal_difficulty == 1:
            self.sample_new_goal_by_z_rot(mjd, rng)
        elif self.goal_difficulty == 2:
            self.sample_new_goal_by_rot(mjd, rng)
        else:
            self.sample_new_goal_by_face(mjd, rng)

    def _update_goal(self, mjd: mujoco.MjData, new_quat: np.ndarray) -> None:
        """Write a new target quaternion to the mocap body, goal vector, and GPU array."""
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

    # --- Drake eval simulator ----------------------------------------------
    def make_eval_simulator(self, video_path: str | None = None, render: bool = True):
        if self.config.eval_sim != EvalSimulatorKind.DRAKE:
            return super().make_eval_simulator(video_path=video_path, render=render)

        import numpy as _np
        from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
        from pydrake.common.eigen_geometry import Quaternion
        from pydrake.multibody.tree import SpatialInertia, UnitInertia
        from pydrake.geometry import Box, HalfSpace
        from pydrake.multibody.plant import CoulombFriction

        from contact_study.contact_models.drake_sim import (
            DrakeSimulator, DrakeJointChannel, DrakeFreeBodyChannel, DrakePdActuation,
        )

        side = 2.0 * _CUBE_HALF

        def _build_scene(plant):
            # Weld the hand's *base* link (the URDF root) so palm_lower lands at
            # the MuJoCo palm world pose -> the Drake and MuJoCo world frames
            # coincide (cube poses + ground line up). Welding palm_lower directly
            # is rejected by Drake (it already has the base_joint inboard).
            qw, qx, qy, qz = _PALM_QUAT
            X_world_palm = RigidTransform(
                RotationMatrix(Quaternion(w=qw, x=qx, y=qy, z=qz)), list(_PALM_POS)
            )
            X_base_palm = RigidTransform(
                RotationMatrix(RollPitchYaw(*_BASE_TO_PALM_RPY)), list(_BASE_TO_PALM_XYZ)
            )
            X_world_base = X_world_palm @ X_base_palm.inverse()
            base = plant.GetBodyByName("base")
            plant.WeldFrames(plant.world_frame(), base.body_frame(), X_world_base)

            # Free cube (matches the MuJoCo obj geom).
            inst = plant.AddModelInstance("objects")
            G = UnitInertia.SolidBox(side, side, side)
            M = SpatialInertia(_CUBE_MASS, _np.zeros(3), G)
            cube = plant.AddRigidBody("obj", inst, M)
            fric = CoulombFriction(_CUBE_FRIC, _CUBE_FRIC)
            box = Box(side, side, side)
            plant.RegisterCollisionGeometry(cube, RigidTransform(), box, "obj_col", fric)
            plant.RegisterVisualGeometry(
                cube, RigidTransform(), box, "obj_vis", _np.array([1.0, 1.0, 0.0, 1.0])
            )

            # Ground plane (matches the MuJoCo floor height).
            plant.RegisterCollisionGeometry(
                plant.world_body(), RigidTransform([0.0, 0.0, _FLOOR_Z]),
                HalfSpace(), "ground", CoulombFriction(1.0, 1.0),
            )

        joint_channels = [
            DrakeJointChannel(_MJ_CTRL_TO_URDF_JOINT[i], "revolute", q_adr=i, v_adr=i)
            for i in range(16)
        ]
        float_channels = [
            DrakeFreeBodyChannel(
                "obj", q_adr=int(self.index_vector[0]), v_adr=int(self.index_vector[1])
            )
        ]
        pd = DrakePdActuation(
            kp=_PD_KP, kd=_PD_KD,
            ctrl_joint_names=list(_MJ_CTRL_TO_URDF_JOINT), effort=_PD_EFFORT,
        )

        return DrakeSimulator(
            model_path     = self.config.eval_model_path,
            config         = self.config,
            nq             = self.mjm.nq,
            nv             = self.mjm.nv,
            joint_channels = joint_channels,
            float_channels = float_channels,
            n_ctrl         = 0,
            weld_base      = False,            # the scene builder welds the palm
            video_path     = video_path,
            # Position control: prefer Drake's native PD-controlled actuators on
            # a discrete plant at this dt (implicit -> stable, fast). DrakeSimulator
            # auto-falls-back to manual PD on a continuous plant when the installed
            # Drake lacks native PD actuators (e.g. 1.12). `time_step` is ignored
            # when pd is set (the mode picks it).
            time_step      = 0.0,
            pd_native_dt   = 0.005,
            extra_models_fn = _build_scene,
            pd             = pd,
        )
