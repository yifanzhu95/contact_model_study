from __future__ import annotations

import numpy as np
import mujoco
import warp as wp

from .base import BaseTask, ContactComplexity, register, SCENES_DIR
from .config import TaskConfig, EvalSimulatorKind

# --- LEAP hand: MuJoCo(rollout) <-> URDF(Drake eval) joint correspondence -----
# The MuJoCo hand and the Drake eval URDF (scenes/leap_hand/leap_hand_right.urdf)
# both name their joints "0".."15", but assign those names DIFFERENTLY: e.g. for
# each finger MuJoCo "0"=mcp, "1"=rot while the URDF has "0"=rot, "1"=mcp. So the
# map is by *kinematic role* (per-finger order [mcp, rot, pip, dip]), keyed by
# MuJoCo qpos slot / actuator index -> URDF joint name. Validated against the
# per-joint limit ranges (the role fingerprint); see KNOWN GAP below. This is a
# naming map, NOT a reordering, so it is unaffected by a MuJoCo joint *rename*
# that preserves the per-finger kinematic order.
#
# KNOWN GAP (eval fidelity): the two thumb middle joints have DIFFERENT limit
# ranges between the URDF and the MuJoCo hand, i.e. the thumb kinematics/zero-
# references differ between the two models. Finger behavior is faithful; thumb
# eval will be approximate until the URDF thumb is recalibrated.
_MJ_CTRL_TO_URDF_JOINT = [
    "1", "0", "2", "3",     # index : mcp, rot, pip, dip
    "5", "4", "6", "7",     # middle
    "9", "8", "10", "11",   # ring
    "12", "13", "14", "15", # thumb
]

# The rollout (MuJoCo) model: the bare hand asset carrying the floor, the free
# "obj" cube, and the fingertip sites (if_tip/mf_tip/rf_tip/th_tip) the cost
# kernel reads. The initial hand+cube state, the joint cost target, and the goal
# pose are all defined in this file (see _INIT_QPOS / _INIT_CTRL / _TARGET_*), so
# the scene needs no keyframe or obj_target mocap body.
#
# Both rollout and eval load from the same URDF-derived geometry, so they share
# one world frame already (the URDF's fixed base->palm_lower transform): welding
# the Drake "base" link to the world at identity (DrakeSimulator's
# `weld_base=True`) lines palm_lower up with the MuJoCo placement with no extra
# calibration, and "obj"/"floor" need no Drake-side scene-building at all.
# DIAGNOSTIC: the _nocontact variant disables all collisions (contype=0
# conaffinity=0 on every geom) to isolate the actuator/PD model from the contact
# model in the MuJoCo-vs-Pinocchio eval comparison. Swap back to the plain
# scene for normal grasping runs (the no-contact scene can't grasp the cube).
GRASP_SCENE_XML = "leap_hand/leap_hand_right_w_sites_yoke_removed.xml"
#GRASP_SCENE_XML = "leap_hand/leap_hand_right_w_sites_yoke_removed_capsule.xml"

# Drake PidController gains for the eval hand (position control, mirroring the
# MuJoCo position servos kp=3.0 kv=0.01). Starting points to tune against Drake's
# solver; _PID_EFFORT is the per-joint actuator force clamp DrakeSimulator adds.
_PID_KP = 3.0
_PID_KI = 0.0
_PID_KD = 0.01
_JOINT_DAMPING = 0.1 #this is maybe too low it was 0.1 befor
_ARMATURE = 0.001

# Pinocchio eval: mirror tests/replay_pinocchio_controls.py's simulation scheme —
_PIN_USE_DIRECT_KD  = True  # False (default): derive kd from zeta + mass matrix. True: use _PIN_KD directly.
_PIN_KD             = _JOINT_DAMPING + _PID_KD   # direct damping gain, used when _PIN_USE_DIRECT_KD is True
_PIN_GRAVITY_COMP   = False  # gravity-compensation torque on the hand PD
# Joint-level constraints mirroring the scene's <joint range> / <joint frictionloss>.
# Pinocchio's aba enforces neither on its own (model.lower/upperPositionLimit and
# model.friction are inert metadata, like model.damping); both are added as
# constraint models in the same ADMM solve as the contacts.
_PIN_ENFORCE_JOINT_LIMITS = True
_PIN_LIMIT_MARGIN         = 0.0    # rad; >0 engages a bound before it is crossed
_PIN_JOINT_FRICTION       = False   # honor the scene's <joint frictionloss>
# Pinocchio render only: overlay a translucent copy of the goal cube (same colors
# and shape as the manipulated "obj" cube, cloned from its <geom> shell) at the
# target reorientation pose (_TARGET_POS/_TARGET_QUAT), so the video shows where
# and how the cube should end up. Visual-only — it never enters the contact solve.
# _PIN_GOAL_OPACITY scales every goal-geom alpha (0 = invisible, 1 = opaque).
_PIN_SHOW_GOAL            = True
_PIN_GOAL_OPACITY         = 0.3

_DRAKE_PID_EFFORT = 100.0

# Fixed initial state + control for the hand and cube, used to initialize BOTH
# the rollout and eval simulators (overrides the scene keyframe). Layout:
#   qpos = [16 hand joints, obj pos(3), obj quat(wxyz)(4)]  (nq = 23)
#   ctrl = [16 hand joint position targets]                  (nu = 16)
# Initial velocity is zero (hand + cube start at rest).
_INIT_QPOS = np.array([
    0.74346777,  -0.56903687,  0.91440081,   0.5741493,
    -0.010605284, -0.08351411, 0.70321997,   1.0184264,
    0.80782262,   0.61122899,  0.92718954,   0.61047876,
    0.69887738,   1.438706,    1.3375555,    0.19482527,

    0.018495468,  0.033628956, 0.083264539,  
    0.93823638, 0.12995374, 0.31377877,  0.066086313,
], dtype=np.float64)

_INIT_CTRL = np.array([
    0.765751,   -0.568012,  0.916951,  0.573897,
    -0.0191225, -0.0837503, 0.709056,  1.01884,
    0.830768,    0.610365,  0.929305,  0.610097,
    0.69912,     1.44581,   1.33179,   0.192794,
], dtype=np.float64)


def _euler_to_quat(euler) -> np.ndarray:
    """Intrinsic-xyz Euler angles (rad) -> wxyz quaternion (MuJoCo convention,
    matching a <body ... euler="..."> with the default eulerseq="xyz")."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for axis, angle in zip(np.eye(3), euler):
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, axis, float(angle))
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, q, dq)
        q = out
    return q


# Goal/target pose for the cube reorientation, defined here rather than read
# from a mocap body in the scene. pos + intrinsic-xyz Euler (rad).
_TARGET_POS   = np.array([0.01 , 0.03, 0.08], dtype=np.float64)#np.array([0.02, 0.03, 0.08], dtype=np.float64)#np.array([0.012, 0.04, 0.085], dtype=np.float64)
_TARGET_EULER = np.array([0.0, 0.5235, 0.0], dtype=np.float64)
_TARGET_QUAT  = _euler_to_quat(_TARGET_EULER)   # wxyz

# Camera: matches the "top" camera in scenes/leap_hand_old/scene_leap_cube.xml:
#   <camera name="top" pos="0.2 0.02 0.4" xyaxes="0 1 0  -1 0 0.5"/>
# MuJoCo xyaxes gives the camera's right (+x) and up (+y) axes in the world
# frame, and the camera looks down its own -z. We convert that to the
# (right, down, forward) world_from_camera columns this config uses (the Drake
# optical-axis convention: +X right, +Y down, +Z forward/viewing direction).
_CAM_POS     = (0.2, 0.02, 0.4)
_cam_right   = np.array([0.0, 1.0, 0.0]);  _cam_right /= np.linalg.norm(_cam_right)
_cam_up      = np.array([-1.0, 0.0, 0.5]); _cam_up    /= np.linalg.norm(_cam_up)
_cam_fwd     = -np.cross(_cam_right, _cam_up)   # camera -z = viewing direction
_cam_down    = -_cam_up
_CAM_ROTMAT  = tuple(   # columns = camera (right, down, forward) axes in world frame
    tuple(float(v) for v in row)
    for row in np.column_stack([_cam_right, _cam_down, _cam_fwd])
)

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
    # Per-axis squared position error, weighted independently in X/Y/Z (see the
    # cost sum below). c_pos is their isotropic total, used by the terminal cost.
    c_pos_x = pos_diff[0] * pos_diff[0]
    c_pos_y = pos_diff[1] * pos_diff[1]
    c_pos_z = pos_diff[2] * pos_diff[2]
    c_pos = c_pos_x + c_pos_y + c_pos_z

    c_joint = float(0.0)
    for i in range(n_manip):
        dq = qpos[robot_qpos_adr + i] - goal[7 + i]
        c_joint = c_joint + dq * dq

    c_joint_velo = float(0.0)
    for i in range(n_manip):
        dq = qvel[robot_qpos_adr + i]
        c_joint_velo = c_joint_velo + dq * dq

    # Contact: squared distance from each fingertip to the cube *surface* (a box
    # of half-extent 0.035), evaluated in the cube's local frame. This is an
    # exterior box SDF: it is 0 once a tip touches the cube and stays 0 as the
    # finger presses in, so closing onto the cube monotonically lowers the cost
    # (the old distance-to-center penalty bottomed out above the surface and then
    # *rose* as fingers curled past, fighting a firm grasp).
    # half = float(0.035)
    # # MuJoCo quat is (w,x,y,z); Warp quat is (x,y,z,w).
    # q_cube = wp.quat(q_obj_v4[1], q_obj_v4[2], q_obj_v4[3], q_obj_v4[0])
    # c_contact = float(0.0)
    # for i in range(5, 9):
    #     p_tip = site_xpos[indices[i]]
    #     local = wp.quat_rotate_inv(q_cube, p_tip - p_obj)
    #     dx = wp.max(wp.abs(local[0]) - half, 0.0)
    #     dy = wp.max(wp.abs(local[1]) - half, 0.0)
    #     dz = wp.max(wp.abs(local[2]) - half, 0.0)
    #     c_contact = c_contact + (dx * dx + dy * dy + dz * dz)
    c_contact = float(0.0)
    for i in range(5, 9):
        p_tip = site_xpos[indices[i]]
        dp = wp.length(p_obj - p_tip) - float(0.035)
        #dp = wp.length(p_tip) - float(0.035)
        if dp > 0.0:
            c_contact = c_contact + dp*dp

    fallen = float(0.0)
    if qpos[obj_qpos_adr + 2] < 0.07:
        fallen = 1.0

    c_velo = wp.dot(v_obj, v_obj) + wp.dot(w_obj, w_obj)

    cost = (
        weights[0] * c_quat +
        weights[1] * c_pos_x +      # w_pos_x
        weights[2] * c_pos_y +      # w_pos_y
        weights[3] * c_pos_z +      # w_pos_z
        weights[4] * c_velo +
        weights[5] * c_contact +
        weights[6] * c_joint +
        weights[7] * c_joint_velo +
        weights[8] * fallen
    )
    if terminal:
        cost = (
            weights[9] * c_quat +    # w_quat_term
            weights[10] * c_pos +    # w_pos_term (isotropic X/Y/Z)
            weights[11] * fallen     # w_fallen_term
        )
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

        self.config = TaskConfig(
            name               = "grasp_reorient",
            complexity         = ContactComplexity.MEDIUM,
            max_steps          = 250,
            success_thresholds = {"pos": 0.02, "quat": 0.04, "vel": 0.1},
            # NOTE: insertion order must match the weights[...] indexing in
            # grasp_reorient_cost_wp AND the weights_list below — the --weights CLI
            # override rebuilds the array from this dict's key order.
            cost_weights       = {
                "w_quat": 5.0,
                "w_pos_x": 10.0,   #I think X is down the fingers # separate X/Y/Z position-error weights
                "w_pos_y": 20.0,    #Y is across the fingers
                "w_pos_z": 10.0,
                "w_velo": 0.0,
                "w_contact": 2.0,
                "w_joint": 0.1,
                "w_joint_velo": 0.0,
                "w_fallen": 200.0,
                "w_quat_term": 50.0,
                "w_pos_term": 2500.0,
                "w_fallen_term": 0.0,
            },
            # BaseTask.load() loads this static file directly — no MJCF is
            # built at task-load time.
            xml_path_template  = GRASP_SCENE_XML,
            rollout_model_path = str(SCENES_DIR / GRASP_SCENE_XML),
            rollout_is_urdf    = False,
            eval_sim           = EvalSimulatorKind.PINOCCHIO,
            # Drake evals the URDF-derived hand; MuJoCo and Pinocchio both eval
            # the same MJCF scene the rollouts plan with (GRASP_SCENE_XML).
            eval_model_paths   = {
                EvalSimulatorKind.DRAKE:     str(SCENES_DIR / "leap_hand/leap_hand_right.urdf"),
                EvalSimulatorKind.MUJOCO:    str(SCENES_DIR / GRASP_SCENE_XML),
                EvalSimulatorKind.PINOCCHIO: str(SCENES_DIR / GRASP_SCENE_XML),
            },
            cam_pos            = _CAM_POS,
            cam_rotmat         = _CAM_ROTMAT,
            cam_fps            = 30.0,
            # Eval ("real") sim timestep; rollout_dt = 10x this = 0.001 (the
            # MuJoCo planning step the GPU rollouts use).
            timestep           = 0.0001,
            eval_substeps_per_rollout = 20,
            difficulty         = self.goal_difficulty,
        )

    # load() is inherited from BaseTask: it loads GRASP_SCENE_XML directly via
    # mujoco.MjModel.from_xml_path, no on-the-fly construction.

    def initialize_task(self):
        mjm = self.mjm
        obj_jnt = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint")

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

        # Goal pose and canonical (initial-face) orientation come from the
        # header constants, not a mocap body in the scene.
        self.target_pos          = _TARGET_POS.copy()
        self.target_quat         = _TARGET_QUAT.copy()
        self._canonical_quat     = self.target_quat.copy()
        self._face_index         = 0

        # The cost's joint target is the initial grasp command (_INIT_CTRL), so
        # the joint term penalizes drift away from the commanded grasp pose.
        self.home_state = _INIT_CTRL.copy()

        self.goal_vector = np.concatenate([
            self.target_pos, self.target_quat, self.home_state
        ]).astype(np.float32)

        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32, device="cuda")
        self.goal_vector_wp = wp.array(self.goal_vector, dtype=wp.float32, device="cuda")

        w = self.config.cost_weights
        weights_list = [
            w["w_quat"], w["w_pos_x"], w["w_pos_y"], w["w_pos_z"], w["w_velo"],
            w["w_contact"], w["w_joint"], w["w_joint_velo"],
            w["w_fallen"],
            w["w_quat_term"], w["w_pos_term"], w["w_fallen_term"]
        ]
        self.weights_wp = wp.array(weights_list, dtype=wp.float32, device="cuda")

    def get_inital_state(self, rng: np.random.Generator):
        # Fixed hand+cube initial state and control, applied to BOTH simulators:
        # the driver resets the eval sim to (q0, v0) and mirrors it into the
        # planning MjData, and uses ctrl0 as the initial command for both.
        mjm = self.mjm
        if _INIT_QPOS.shape[0] != mjm.nq or _INIT_CTRL.shape[0] != mjm.nu:
            raise ValueError(
                f"_INIT_QPOS ({_INIT_QPOS.shape[0]}) / _INIT_CTRL "
                f"({_INIT_CTRL.shape[0]}) do not match model nq={mjm.nq} / nu={mjm.nu}."
            )
        q0 = _INIT_QPOS.copy()
        v0 = np.zeros(mjm.nv, dtype=np.float64)
        ctrl0 = _INIT_CTRL.copy()
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

        thr     = self.config.success_thresholds
        pose_ok = pos_err < thr["pos"] and quat_err < thr["quat"]
        if not pose_ok:
            return False

        return bool(np.linalg.norm(vel) < thr["vel"])

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
        if self.config.eval_sim == EvalSimulatorKind.PINOCCHIO:
            return self._make_pinocchio_simulator(video_path=video_path, render=render)
        if self.config.eval_sim != EvalSimulatorKind.DRAKE:
            return super().make_eval_simulator(video_path=video_path, render=render)

        from contact_study.contact_models.drake_sim import (
            DrakeSimulator, DrakeJointChannel, DrakeFreeBodyChannel, DrakePidActuation,
        )

        # No extra_models_fn needed: "floor" and "obj" are parsed straight out
        # of the URDF (eval_model_paths[DRAKE]) along with the hand, and "base" welds
        # to the world at identity (weld_base=True) just like the MuJoCo
        # rollout's own base->palm_lower transform, so both world frames and
        # both floor/cube placements coincide automatically.
        #
        # _MJ_CTRL_TO_URDF_JOINT maps MuJoCo qpos slot / actuator index -> URDF
        # joint name by kinematic role; both share qpos==ctrl order, so the same
        # list keys joint_channels (q_adr=i) and the PID's ctrl_joint_names.
        joint_channels = [
            DrakeJointChannel(_MJ_CTRL_TO_URDF_JOINT[i], "revolute", q_adr=i, v_adr=i)
            for i in range(16)
        ]
        float_channels = [
            DrakeFreeBodyChannel(
                "obj", q_adr=int(self.index_vector[0]), v_adr=int(self.index_vector[1])
            )
        ]
        pid = DrakePidActuation(
            kp=_PID_KP, ki=_PID_KI, kd=_PID_KD,
            ctrl_joint_names=list(_MJ_CTRL_TO_URDF_JOINT), effort=_DRAKE_PID_EFFORT,
        )

        return DrakeSimulator(
            model_path     = self.config.eval_model_paths[self.config.eval_sim],
            config         = self.config,
            nq             = self.mjm.nq,
            nv             = self.mjm.nv,
            joint_channels = joint_channels,
            float_channels = float_channels,
            weld_base      = True,             # welds "base" to world at identity
            video_path     = video_path,
            # Position control via Drake's PidController. pid_plant_dt is the eval
            # plant's discrete step; it must be far smaller than the control dt
            # because the discrete SAP solver treats PID actuation explicitly and
            # diverges at large steps (mirrors tests/view_model_drake.py's 1e-4).
            # Half the nominal eval timestep keeps the SAP solver well inside its
            # stable range regardless of how config.timestep is tuned.
            pid_plant_dt   = self.config.timestep / 2.0,
            pid            = pid,
        )

    def _make_pinocchio_simulator(self, video_path: str | None = None, render: bool = True):
        """Pinocchio + ADMM eval simulator. Unlike the Drake path it parses the
        MJCF at eval_model_paths[PINOCCHIO] (the same scene the rollout model
        uses), so the 16 hand joints, the obj freejoint, and the control order
        all align 1:1 with the
        MuJoCo qpos/qvel/ctrl indices — the channels are an identity map (no
        URDF name translation needed). The hand joints are named "0".."15" and
        the cube is the "obj_joint" freejoint, matching index_vector."""
        from contact_study.contact_models.pinocchio_sim import (
            PinocchioSimulator, PinocchioJointChannel, PinocchioFreeBodyChannel,
            PinocchioPdActuation, PinocchioContactConfig,
            PinocchioJointConstraintConfig,
        )

        # Pinocchio joint names come from the MJCF, so read them off the loaded
        # MuJoCo model instead of assuming "0".."15" (that only held for the older
        # index-named scene; this one names joints "if_mcp", "if_rot", ...). Every
        # 1-DOF (non-free) joint is a hand joint; its MuJoCo qpos/qvel address maps
        # to the Pinocchio joint of the same name.
        mjm = self.mjm
        hand_jids = [
            j for j in range(mjm.njnt)
            if mjm.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        ]
        joint_channels = [
            PinocchioJointChannel(
                pin_name=mjm.joint(j).name,
                q_adr=int(mjm.jnt_qposadr[j]),
                v_adr=int(mjm.jnt_dofadr[j]),
            )
            for j in hand_jids
        ]
        free_channels = [
            PinocchioFreeBodyChannel(
                pin_name="obj_joint",
                q_adr=int(self.index_vector[0]), v_adr=int(self.index_vector[1]),
            )
        ]
        # PD desired positions arrive in MuJoCo control order, so name the
        # controlled joints by each actuator's target joint (trnid).
        ctrl_joint_names = [
            mjm.joint(int(mjm.actuator(a).trnid[0])).name for a in range(mjm.nu)
        ]
        # Per-actuator torque limit from each target joint's actuatorfrcrange
        # (MuJoCo jnt_actfrcrange), in control order — so the Pinocchio PD
        # saturates its torque exactly like MuJoCo's actuator does. Joints with
        # no limit get (-inf, inf) (no clamp). The LEAP joints all cap at
        # +/-0.95 N*m, which the +/-45deg step commands hit hard; without this
        # the unclamped PD tracked far faster than MuJoCo (see _substep).
        force_limit = np.array([
            mjm.jnt_actfrcrange[int(mjm.actuator(a).trnid[0])]
            if mjm.jnt_actfrclimited[int(mjm.actuator(a).trnid[0])]
            else (-np.inf, np.inf)
            for a in range(mjm.nu)
        ], dtype=np.float64)
        # kp is _PID_KP directly (the closed-loop time constant); kd is either
        # derived per-joint from the mass matrix (critically damped at that
        # stiffness) or applied directly, depending on _PIN_USE_DIRECT_KD (see
        # PinocchioSimulator._substep).
        pid = PinocchioPdActuation(
            ctrl_joint_names=ctrl_joint_names,
            kp=_PID_KP,
            gravity_comp=_PIN_GRAVITY_COMP,
            armature=_ARMATURE,
            kd=_PIN_KD,
            force_limit=force_limit,
        )
        # cube<->hand / hand<->hand frictional point contacts, each with a native
        # Baumgarte corrector on the contact position error.
        contact_cfg = PinocchioContactConfig()

        # Joint position limits + dry friction, enforced as ADMM constraints.
        # frictionloss is read off the MuJoCo model (dof_frictionloss) rather than
        # Pinocchio's model.friction: this scene sets it via a <default> block, and
        # Pinocchio's MJCF parser does not apply <default> inheritance (the same gap
        # it has for contype/conaffinity), so its model.friction would read 0.
        frictionloss = None
        if _PIN_JOINT_FRICTION:
            frictionloss = {
                mjm.joint(j).name: float(mjm.dof_frictionloss[int(mjm.jnt_dofadr[j])])
                for j in hand_jids
            }
        joint_cfg = PinocchioJointConstraintConfig(
            enforce_limits = _PIN_ENFORCE_JOINT_LIMITS,
            limit_margin   = _PIN_LIMIT_MARGIN,
            frictionloss   = frictionloss,
        )

        return PinocchioSimulator(
            model_path     = self.config.eval_model_paths[self.config.eval_sim],  # the MJCF (not URDF)
            config         = self.config,
            nq             = self.mjm.nq,
            nv             = self.mjm.nv,
            pid            = pid,
            joint_channels = joint_channels,
            free_channels  = free_channels,
            contact_cfg    = contact_cfg,
            joint_cfg      = joint_cfg,
            video_path     = video_path,
            render         = render,
            # Translucent goal-cube overlay at the reorientation target (render only).
            goal_pose      = (_TARGET_POS, _TARGET_QUAT) if _PIN_SHOW_GOAL else None,
            goal_opacity   = _PIN_GOAL_OPACITY,
        )
