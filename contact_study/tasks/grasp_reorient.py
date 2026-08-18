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
# Scenes are selected by scene variant (see contact_study.tasks.config.
# SceneVariant): the CLI's --geometry string names the object and the hand /
# object collision fidelity, and the two templates in TaskConfig below resolve
# it to a rollout scene and an eval scene.

# Drake PidController gains for the eval hand (position control, mirroring the
# MuJoCo position servos kp=3.0 kv=0.01). Starting points to tune against Drake's
# solver; _PID_EFFORT is the per-joint actuator force clamp DrakeSimulator adds.
_PID_KP = 3.0
_PID_KI = 0.0
_PID_KD = 0.01
# Passive joint damping. MUST track <default><joint damping> in the leap hand MJCFs
# (scenes/leap/leap_right_hand{,_eval}.xml): MuJoCo brakes with damping + the position
# actuator's kv (0.3 + 0.01), and Pinocchio — whose aba ignores model.damping — reaches
# the same 0.31 through _PIN_KD below. Changing one without the other desyncs the sims.
_JOINT_DAMPING = 0.3#0.1 #this is maybe too low it was 0.1 befor
# Rotor inertia on the 16 hand joints. MUST equal <default><joint armature> in the
# leap hand MJCFs, which is where MuJoCo reads its dof_armature from.
#
# It has to be injected here as well, rather than inherited from that MJCF, because
# Pinocchio LOSES it on the way in: buildModelsFromMJCF does parse it correctly (a
# freshly-parsed hand subtree reads model.armature == 0.001), but pin.appendModel —
# which merge_models uses to stitch the split single-root subtrees back together —
# drops model.armature while preserving model.damping. Verified on this scene: the
# parsed part has armature 0.001, the merged model has 0.0.
#
# So: XML feeds MuJoCo, this constant feeds Pinocchio, and the two must be kept equal
# by hand. If merge_models is ever fixed to carry armature across, set this to 0.0 —
# PinocchioSimulator.__init__ *adds* (model.armature[ctrl] += pid.armature) and would
# otherwise double-count. tasks/balance_stick.py and tasks/actuator_test.py carry the
# same workaround (_PIN_ARMATURE) against the same appendModel behavior.
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
# OFF by default: _TARGET_POS sits where the cube is held, so this overlay drew a
# translucent "shadow" cube right on top of the real one in the palm. The scene's
# <body name="goal"> marker (upper-right of frame, textured like the cube, spun by
# _update_goal) shows the target orientation instead — see _GOAL_MARKER_BODY.
_PIN_SHOW_GOAL            = False
_PIN_GOAL_OPACITY         = 0.3

# Scene body (a MuJoCo mocap body) used as the on-screen reorientation target.
# _update_goal rewrites its orientation for both renderers: mjd.mocap_quat for
# MuJoCo, PinocchioSimulator.set_goal_quat for the Pinocchio/Panda3d video.
_GOAL_MARKER_BODY = "goal"

_DRAKE_PID_EFFORT = 100.0

# Fixed initial state + control for the hand and cube, used to initialize BOTH
# the rollout and eval simulators (overrides the scene keyframe). Layout:
#   qpos = [16 hand joints, obj pos(3), obj quat(wxyz)(4)]  (nq = 23)
#   ctrl = [16 hand joint position targets]                  (nu = 16)
# Initial velocity is zero (hand + cube start at rest).
# _INIT_QPOS = np.array([
#     0.74346777,  -0.56903687,  0.91440081,   0.5741493,
#     -0.010605284, -0.08351411, 0.70321997,   1.0184264,
#     0.80782262,   0.61122899,  0.92718954,   0.61047876,
#     0.69887738,   1.438706,    1.3375555,    0.19482527,

#     0.018495468,  0.033628956, 0.083264539,
#     0.93823638, 0.12995374, 0.31377877,  0.066086313,
# ], dtype=np.float64)

# Hand at its old tuned pose + cube at the scene XML's default "obj" body
# pos/quat, so the cube starts resting in the center of the palm.
# _INIT_QPOS = np.array([
#     0.74346777,  -0.56903687,  0.91440081,   0.5741493,
#     -0.010605284, -0.08351411, 0.70321997,   1.0184264,
#     0.80782262,   0.61122899,  0.92718954,   0.61047876,
#     0.69887738,   1.438706,    1.3375555,    0.19482527,

#     0.01, 0.0258, 0.08,
#     0.965926, 0.0, 0.258819, 0.0,
# ], dtype=np.float64)
_INIT_QPOS = np.array([
7.41953443e-01, -5.14095650e-01,  6.97705793e-01,  5.73857360e-01,
  3.11686592e-01, -2.08901684e-05,  7.04119781e-01,  1.01887562e+00,
  7.14271347e-01,  2.63610945e-01,  6.97700993e-01,  6.10100133e-01,
  7.00288255e-01,  1.52604395e+00,  1.33871871e+00,  8.68983906e-01,
  1.70374863e-02,  3.65435775e-02,  8.36225067e-02 + 0.01,  
  1, 0, 0, 0
], dtype=np.float64)


_INIT_CTRL = np.array([
0.7672  , -0.51303 ,  0.701455,  0.573897,  0.33472 ,  0.      ,
  0.709056,  1.01884 ,  0.74176 ,  0.26175 ,  0.701455,  0.610097,
  0.69912 ,  1.53211 ,  1.33179 ,  0.8657
], dtype=np.float64)

def _axis_angle_quat(axis, angle: float) -> np.ndarray:
    """wxyz quaternion for a rotation of `angle` (rad) about `axis` (unit vector)."""
    c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
    return np.array([c, s * axis[0], s * axis[1], s * axis[2]])


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
_TARGET_POS   = np.array([0.02, 0.03, 0.09], dtype=np.float64)#np.array([0.02, 0.03, 0.08], dtype=np.float64)#np.array([0.012, 0.04, 0.085], dtype=np.float64)
_TARGET_EULER = np.array([0.0, 0.0, 0.0], dtype=np.float64)
_TARGET_QUAT  = _euler_to_quat(_TARGET_EULER)   # wxyz

# Per-object overrides, keyed by the scene variant's object name. _INIT_QPOS's
# layout is [16 hand joints, obj pos(3), obj quat wxyz(4)] (nq = 23); the hand
# block and _INIT_CTRL are shared by every object, so only the trailing
# 7-element free-body block, the reorientation target, and the "fallen" drop
# height vary. Missing keys fall back to _OBJ_DEFAULTS (the cube's values), so a
# new object only declares what actually differs.
_OBJ_DEFAULTS = {
    "init_obj_qpos": _INIT_QPOS[16:].copy(),   # settled in-palm pose of the cube
    "target_pos":    _TARGET_POS,
    "fallen_z":      0.08,                     # cost-kernel drop threshold
}
_OBJ_OVERRIDES: dict[str, dict] = {
    "cube": {},
    # TODO(retune): settle the hand in the viewer against
    # env_leap_rollout_duck_low_high.xml and paste qpos[16:23] here. The duck's
    # bounding half-extents are (0.0325, 0.0466, 0.0512) about a centre offset
    # of (0, 0.0032, 0.0086) — i.e. ~1.5 cm taller than the 3.5 cm cube, so it
    # needs to start higher and its target sits higher too.
    "duck": {
        "init_obj_qpos": np.array([
            1.37863074e-02,  2.68674345e-02 + 0.02,  9.00480655e-02 + 0.01,
            0.7071068, 0, 0, 0.7071068
        ]),
        "target_pos":    np.array([0.015, 0.045, 0.10]),
        "fallen_z":      0.08,
    },
}

# Camera: matches the "top" camera in scenes/leap_hand_old/scene_leap_cube.xml:
#   <camera name="top" pos="0.2 0.02 0.4" xyaxes="0 1 0  -1 0 0.5"/>
# MuJoCo xyaxes gives the camera's right (+x) and up (+y) axes in the world
# frame, and the camera looks down its own -z. We convert that to the
# (right, down, forward) world_from_camera columns this config uses (the Drake
# optical-axis convention: +X right, +Y down, +Z forward/viewing direction).
_CAM_POS     = (0.19, 0.01, 0.4)
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
    # Per-axis L1 (absolute) position error, weighted independently in X/Y/Z (see
    # the cost sum below), matching irisim_warp's LeapDexReal. c_pos is their L1
    # total, used by the terminal cost.
    c_pos_x = wp.abs(pos_diff[0])
    c_pos_y = wp.abs(pos_diff[1])
    c_pos_z = wp.abs(pos_diff[2])
    
    #c_pos = c_pos_x + c_pos_y + c_pos_z
    c_pos = wp.norm_l2(pos_diff) 
    # if c_pos < 0.0:
    #     c_pos = 0.0

    # c_pos_x = c_pos_x - 0.02
    # if c_pos_x < 0.0:
    #     c_pos_x = 0.0
    # c_pos_y = c_pos_y - 0.01
    # if c_pos_y < 0.0:
    #     c_pos_y = 0.0
    # c_pos_z = c_pos_z - 0.02
    # if c_pos_z < 0.0:
    #     c_pos_z = 0.0

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
        dp = wp.length(p_obj - p_tip)
        c_contact = c_contact + dp

    # Drop threshold lives in the goal array (slot 7 + n_manip, right after the
    # per-joint home pose) so it can vary per object — a squatter object would
    # otherwise read as "fallen" while still held. See _OBJ_DEFAULTS.
    fallen = float(0.0)
    if qpos[obj_qpos_adr + 2] < goal[7 + n_manip]:
        fallen = 1.0

    c_velo = wp.dot(v_obj, v_obj) + wp.dot(w_obj, w_obj)

    cost = (
        weights[0] * c_quat +
        weights[1] * c_pos_x +      # w_pos_x
        weights[2] * c_pos_y +      # w_pos_y
        weights[3] * c_pos_z +      # w_pos_z
        weights[4] * c_pos + #<- BIG CHANGE!!!!!!
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
        0 — Trivial: fixed 90° clockwise spin around the normal of the
            currently-shown face. Like level 1 but the turn direction is not
            sampled — it never spins counter-clockwise.
        1 — Easiest: ±90° spin around the normal of the currently-shown face.
            The face stays the same; only the in-plane orientation changes.
        2 — Medium:  ±90° rotation around a randomly-chosen object-frame axis
            (X, Y, or Z).  May change which face is shown.
        3 — Adjacent face: jump to one of the 4 faces adjacent to the
            currently-shown face (excludes the current face and its
            opposite), with a random 90° twist (0/90/180/270) on the new
            face on top.
        4 — Hard:    Jump to a completely different cube face (5 candidates),
            with a random 90° twist on top.
        5 — Flip:    Fixed 180° roll about an axis lying in the currently-shown
            face, so that face ends up on the bottom and the opposite face is
            shown. No twist, no sampling.
        6 — Adjacent face, no twist: same face selection as level 3 (one of
            the 4 faces adjacent to the currently-shown face), but with no
            random twist — the new face's canonical (untwisted) orientation
            is always used.
    """

    # Controls which sampling method sample_new_goal dispatches to.
    # Override on an instance before the first episode to change difficulty.
    goal_difficulty: int = 6

    # Most recently built eval simulator, so _update_goal can spin its goal
    # marker. Class-level on purpose: the drivers build TWO task instances per
    # episode (a ROLLOUT one that samples goals and an EVAL one that owns the
    # simulator and the video), and it is the rollout instance that gets
    # sample_new_goal called on it. A per-instance handle would therefore always
    # be None on the side that needs it. One eval sim exists at a time per
    # process, so the latest one is unambiguously the one being recorded.
    _active_eval_sim = None


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

    # Face that ends up shown after a 180° flip of each face (see _FACE_ROTS
    # below). Pairs share a normal axis, so a flip never changes
    # _FACE_NORMALS[self._face_index].
    _OPPOSITE_FACES = [3, 2, 1, 0, 5, 4]

    # Six object-frame rotations, one per cube face (right-multiplied onto the
    # canonical goal quat to select that face). Using Rx/Ry only ensures all 6
    # faces are distinct:
    #   face 0 — identity          (initial face)
    #   face 1 — Rx +90°
    #   face 2 — Rx -90°
    #   face 3 — Rx 180°           (opposite face)
    #   face 4 — Ry +90°
    #   face 5 — Ry -90°
    # Shared by sample_new_goal_by_face and sample_new_goal_by_adjacent_face.
    _FACE_ROTS = [
        _axis_angle_quat([1, 0, 0], 0.0),
        _axis_angle_quat([1, 0, 0], np.pi / 2),
        _axis_angle_quat([1, 0, 0], -np.pi / 2),
        _axis_angle_quat([1, 0, 0], np.pi),
        _axis_angle_quat([0, 1, 0], np.pi / 2),
        _axis_angle_quat([0, 1, 0], -np.pi / 2),
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
            max_steps          = 4000,
            success_thresholds = {"pos": 0.02, "quat": 0.04, "vel": 0.1},
            # NOTE: insertion order must match the weights[...] indexing in
            # grasp_reorient_cost_wp AND the weights_list below — the --weights CLI
            # override rebuilds the array from this dict's key order.
            cost_weights       = {
                "w_quat": 20.0,#100.0,
                "w_pos_x": 7.50,#60.0,   #I think X is down the fingers # separate X/Y/Z position-error weights
                "w_pos_y": 15.0,#80.0,    #Y is across the fingers
                "w_pos_z": 5.0,#15.0,
                "w_velo": 0.0,
                "w_contact": 15.0,#12.50,
                "w_joint": 4.0,
                "w_joint_velo": 0.0,
                "w_fallen": 200.0,
                "w_quat_term": 200.0,#500.0,
                "w_pos_term": 200.0,
                "w_fallen_term": 0.0,
            },
            # Scene variant -> scene files, by convention. BaseTask.load()
            # picks the template matching this instance's role and fills it
            # from the parsed --geometry string; the rollout scene degrades
            # hand/object collision geometry, the eval scene never does.
            rollout_xml_template = "leap/env_leap_rollout_{obj}_{hand_acc}_{obj_acc}.xml",
            eval_xml_template    = "leap/env_leap_eval_{obj}.xml",
            xml_path_template  = None,   # role templates cover both roles
            rollout_model_path = None,   # published by _publish_eval_model_paths
            rollout_is_urdf    = False,
            eval_sim           = EvalSimulatorKind.PINOCCHIO,
            # MuJoCo and Pinocchio eval the resolved eval scene, filled in by
            # BaseTask._publish_eval_model_paths at load time. Drake's hand is a
            # URDF that predates the scene-variant convention and has no
            # per-object form, so it stays an explicit path.
            eval_model_paths   = {
                EvalSimulatorKind.DRAKE: str(SCENES_DIR / "leap_hand/leap_hand_right.urdf"),
            },
            cam_pos            = _CAM_POS,
            cam_rotmat         = _CAM_ROTMAT,
            cam_fps            = 30.0,
            # Eval ("real") sim timestep; rollout_dt = 10x this = 0.001 (the
            # MuJoCo planning step the GPU rollouts use).
            timestep           = 0.0005,
            eval_substeps_per_rollout = 8,
            difficulty         = self.goal_difficulty,
        )

    @property
    def obj_params(self) -> dict:
        """Per-object init pose / target / drop threshold, resolved from the
        scene variant's object name against _OBJ_DEFAULTS."""
        return {**_OBJ_DEFAULTS,
                **_OBJ_OVERRIDES.get(self.scene_variant.obj, {})}

    def _build_goal_vector(self) -> np.ndarray:
        """[target_pos(3), target_quat(4), home_state(16), fallen_z(1)].

        The trailing drop threshold is read by grasp_reorient_cost_wp as
        goal[7 + n_manip]; keep the layout in sync with that kernel.
        """
        return np.concatenate([
            self.target_pos, self.target_quat, self.home_state,
            [float(self.obj_params["fallen_z"])],
        ]).astype(np.float32)

    # load() is inherited from BaseTask: it resolves the role's scene template
    # against the scene variant and loads that file via
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
        # per-object table, not a mocap body in the scene.
        self.target_pos          = np.asarray(
            self.obj_params["target_pos"], dtype=np.float64).copy()
        self.target_quat         = _TARGET_QUAT.copy()
        self._canonical_quat     = self.target_quat.copy()
        self._face_index         = 0

        # The cost's joint target is the initial grasp command (_INIT_CTRL), so
        # the joint term penalizes drift away from the commanded grasp pose.
        self.home_state = _INIT_CTRL.copy()

        self.goal_vector = self._build_goal_vector()

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
        # Fixed hand+object initial state and control, applied to BOTH
        # simulators: the driver resets the eval sim to (q0, v0) and mirrors it
        # into the planning MjData, and uses ctrl0 as the initial command for
        # both. The hand block is shared across objects; the trailing free-body
        # block comes from the per-object table.
        mjm = self.mjm
        n_hand = _INIT_CTRL.shape[0]
        if mjm.nq != n_hand + 7 or mjm.nu != n_hand:
            raise ValueError(
                f"Scene {self.scene_variant.raw!r} has nq={mjm.nq} nu={mjm.nu}; "
                f"this task requires nq={n_hand + 7} ({n_hand} hand joints + one "
                f"free body) and nu={n_hand}."
            )
        obj_qpos = np.asarray(self.obj_params["init_obj_qpos"], dtype=np.float64)
        if obj_qpos.shape[0] != 7:
            raise ValueError(
                f"init_obj_qpos for object {self.scene_variant.obj!r} has "
                f"{obj_qpos.shape[0]} elements; expected 7 (pos + wxyz quat)."
            )
        q0 = np.concatenate([_INIT_QPOS[:n_hand], obj_qpos])
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
        (set at load time) by right-multiplying with one of the 6 object-frame
        rotations in _FACE_ROTS (one per cube face).  The current face index is
        tracked so the selected face is always different from the previous goal.
        """
        # Pick uniformly from the 5 faces that are not the current one
        candidates = [i for i in range(6) if i != self._face_index]
        self._face_index = int(rng.choice(candidates))

        # Random 90° twist (0°/90°/180°/270°) around the selected face's normal
        twist_angle = int(rng.integers(0, 4)) * (np.pi / 2)
        q_twist = _axis_angle_quat(self._FACE_NORMALS[self._face_index], twist_angle)

        # Combine: face rotation then twist (both in object frame)
        q_face_twist = np.zeros(4)
        mujoco.mju_mulQuat(q_face_twist, self._FACE_ROTS[self._face_index], q_twist)

        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, self._canonical_quat, q_face_twist)

        self._update_goal(mjd, new_quat)

    def sample_new_goal_by_adjacent_face(self, mjd: mujoco.MjData, rng: np.random.Generator):
        """Difficulty 3: jump to one of the 4 faces adjacent to the currently-
        shown face (i.e. any face except the current one and its opposite),
        then apply a random 90° twist (0°/90°/180°/270°) around the new face's
        normal.  Easier than difficulty 4 (which can also jump to the opposite
        face) but still requires tipping the cube onto a new face.
        """
        candidates = [
            i for i in range(6)
            if i != self._face_index and i != self._OPPOSITE_FACES[self._face_index]
        ]
        self._face_index = int(rng.choice(candidates))

        # Random 90° twist (0°/90°/180°/270°) around the selected face's normal
        twist_angle = int(rng.integers(0, 4)) * (np.pi / 2)
        q_twist = _axis_angle_quat(self._FACE_NORMALS[self._face_index], twist_angle)

        # Combine: face rotation then twist (both in object frame)
        q_face_twist = np.zeros(4)
        mujoco.mju_mulQuat(q_face_twist, self._FACE_ROTS[self._face_index], q_twist)

        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, self._canonical_quat, q_face_twist)

        self._update_goal(mjd, new_quat)

    def sample_new_goal_by_adjacent_face_no_twist(self, mjd: mujoco.MjData, rng: np.random.Generator):
        """Difficulty 6: jump to one of the 4 faces adjacent to the currently-
        shown face (same face selection as difficulty 3), but with no random
        twist — the new face is always shown in its canonical orientation.
        Easier than difficulty 3 since the controller only needs to tip the
        cube onto the new face, never rotate it in-plane afterward.
        """
        candidates = [
            i for i in range(6)
            if i != self._face_index and i != self._OPPOSITE_FACES[self._face_index]
        ]
        self._face_index = int(rng.choice(candidates))

        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, self._canonical_quat, self._FACE_ROTS[self._face_index])

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

    def sample_new_goal_by_z_rot_cw(self, mjd: mujoco.MjData, rng: np.random.Generator):
        """Difficulty 0: fixed 90° clockwise spin around the normal of the
        currently-shown face.

        Identical to difficulty 1 except the rotation direction is not sampled —
        the goal always spins clockwise (negative angle by the right-hand rule
        about the face normal), never counter-clockwise. This is the easiest
        goal since both the face and the turn direction are fixed.
        """
        axis = np.array(self._FACE_NORMALS[self._face_index], dtype=float)
        angle = -np.pi / 2.0  # clockwise about the face normal

        c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
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

    def sample_new_goal_by_flip(self, mjd: mujoco.MjData, rng: np.random.Generator):
        """Difficulty 5: flip the cube over — the shown face becomes the bottom.

        A 180° rotation about any axis lying in the shown face sends that face's
        normal to its negative, so the opposite face is shown. The axis is the
        first cardinal object axis perpendicular to the current face normal
        (X, else Y), making the goal fully deterministic: no face choice, no
        twist, no direction to sample. Harder than levels 0-2 because the cube
        must actually be tipped over, but with a single fixed goal per face.
        """
        normal = np.array(self._FACE_NORMALS[self._face_index], dtype=float)

        # First cardinal axis orthogonal to the face normal.
        axis = np.zeros(3)
        for i in range(3):
            if abs(normal[i]) < 0.5:
                axis[i] = 1.0
                break

        # 180° about that axis: cos(90°) = 0, sin(90°) = 1.
        q_rot = np.array([0.0, axis[0], axis[1], axis[2]])

        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, self.target_quat, q_rot)

        self._face_index = self._OPPOSITE_FACES[self._face_index]
        self._update_goal(mjd, new_quat)

    def sample_new_goal(self, mjd: mujoco.MjData, rng: np.random.Generator):
        """Dispatch to the appropriate sampler based on self.goal_difficulty."""
        if self.goal_difficulty == 0:
            self.sample_new_goal_by_z_rot_cw(mjd, rng)
        elif self.goal_difficulty == 1:
            self.sample_new_goal_by_z_rot(mjd, rng)
        elif self.goal_difficulty == 2:
            self.sample_new_goal_by_rot(mjd, rng)
        elif self.goal_difficulty == 3:
            self.sample_new_goal_by_adjacent_face(mjd, rng)
        elif self.goal_difficulty == 5:
            self.sample_new_goal_by_flip(mjd, rng)
        elif self.goal_difficulty == 6:
            self.sample_new_goal_by_adjacent_face_no_twist(mjd, rng)
        else:
            self.sample_new_goal_by_face(mjd, rng)

    def _update_goal(self, mjd: mujoco.MjData, new_quat: np.ndarray) -> None:
        """Write a new target quaternion to the mocap body, goal vector, and GPU array."""
        # On-screen target marker. _GOAL_MARKER_BODY is this scene's mocap body;
        # "obj_target" is the name older scenes (scenes/scripts/build_grasp_
        # reorient_scene.py, env_allegro_cube.xml) use for the same thing.
        # Break only on a body we could actually drive: scenes that carry a
        # non-mocap <body name="goal"> (env_leap_duck/spam/tomato) must still
        # fall through to an "obj_target" mocap body if they have one.
        for body_name in (_GOAL_MARKER_BODY, "obj_target"):
            target_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if target_id >= 0:
                mocap_id = self.mjm.body_mocapid[target_id]
                if mocap_id >= 0:
                    mjd.mocap_quat[mocap_id] = new_quat
                    break

        # Mirror onto the eval simulator's own renderer. The drivers sample goals
        # on the ROLLOUT task but record video from the EVAL sim, so without this
        # the marker would only ever turn in the MuJoCo view. Guarded by hasattr:
        # only PinocchioSimulator implements it.
        sim = GraspReorientTask._active_eval_sim
        if sim is not None and hasattr(sim, "set_goal_quat"):
            sim.set_goal_quat(new_quat)

        self.target_quat = new_quat.copy()
        self.goal_vector = self._build_goal_vector()

        if self.goal_vector_wp is not None:
            self.goal_vector_wp.assign(self.goal_vector)

    # --- Drake eval simulator ----------------------------------------------
    def make_eval_simulator(self, video_path: str | None = None, render: bool = True,
                            use_mp4: bool = True):
        if self.config.eval_sim == EvalSimulatorKind.PINOCCHIO:
            sim = self._make_pinocchio_simulator(video_path=video_path, render=render,
                                                 use_mp4=use_mp4)
            GraspReorientTask._active_eval_sim = sim   # see _update_goal
            return sim
        if self.config.eval_sim != EvalSimulatorKind.DRAKE:
            sim = super().make_eval_simulator(video_path=video_path, render=render,
                                              use_mp4=use_mp4)
            GraspReorientTask._active_eval_sim = sim   # see _update_goal
            return sim

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
            use_mp4        = use_mp4,
        )

    def _make_pinocchio_simulator(self, video_path: str | None = None, render: bool = True,
                                  use_mp4: bool = True):
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
            use_mp4        = use_mp4,
            # Translucent goal-cube overlay at the reorientation target (render only).
            goal_pose      = (_TARGET_POS, _TARGET_QUAT) if _PIN_SHOW_GOAL else None,
            goal_opacity   = _PIN_GOAL_OPACITY,
        )
