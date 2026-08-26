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
# pose are all defined in this file (see _OBJ_PARAMS / _TARGET_QUAT), so the
# scene needs no keyframe or obj_target mocap body.
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
# target reorientation pose (this object's "target_pos" + _TARGET_QUAT), so the
# video shows where and how the cube should end up. Visual-only — it never enters the contact solve.
# _PIN_GOAL_OPACITY scales every goal-geom alpha (0 = invisible, 1 = opaque).
# OFF by default: the target position sits where the cube is held, so this drew a
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

# Fixed initial state + control for the hand and object, used to initialize
# BOTH the rollout and eval simulators (overrides the scene keyframe). Layout:
#   qpos = [16 hand joints, obj pos(3), obj quat(wxyz)(4)]  (nq = 23)
#   ctrl = [16 hand joint position targets]                  (nu = 16)
# Initial velocity is zero (hand + object start at rest).
#
# Both are PER OBJECT and live in the _OBJ_PARAMS table below, as "init_qpos"
# and "init_ctrl": a taller or rounder object generally wants its own pre-grasp
# hand pose and grasp command, not just its own free-body pose. init_qpos is one
# 23-element array — the whole qpos, exactly as the viewer prints it — so a pose
# settled in the viewer pastes in as a single block.
#
# Historical cube poses, kept for reference:
#   hand = [ 0.74346777,  -0.56903687, 0.91440081, 0.5741493,
#           -0.010605284, -0.08351411, 0.70321997, 1.0184264,
#            0.80782262,   0.61122899, 0.92718954, 0.61047876,
#            0.69887738,   1.438706,   1.3375555,  0.19482527]
#   cube = [0.018495468, 0.033628956, 0.083264539,
#           0.93823638,  0.12995374,  0.31377877,  0.066086313]
#   cube, at the scene XML's default "obj" body pos/quat (resting in the centre
#   of the palm) = [0.01, 0.0258, 0.08,  0.965926, 0.0, 0.258819, 0.0]

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


# Goal orientation for the reorientation, defined here rather than read from a
# mocap body in the scene: intrinsic-xyz Euler (rad) -> wxyz quaternion. Shared
# by every object; the goal POSITION is per-object ("target_pos" below), since
# it depends on how tall the object sits in the palm.
_TARGET_EULER = np.array([0.0, 0.0, 0.0], dtype=np.float64)
_TARGET_QUAT  = _euler_to_quat(_TARGET_EULER)   # wxyz

# Canonical cost-weight names, in the order the cost kernel indexes them. THE
# ORDER IS LOAD-BEARING: it must match the weights[...] indexing in
# grasp_reorient_cost_wp and the weights_list built in initialize_task, because
# --weights and the param sweeps rebuild the array by iterating the resolved
# cost_weights dict's keys. Every object spells out every one of these keys in
# _OBJ_PARAMS and obj_params re-emits them in THIS order, so however they happen
# to be typed in the table below they can never desync from the kernel.
_COST_WEIGHT_KEYS = (
    "w_quat", "w_pos_x", "w_pos_y", "w_pos_z", "w_velo", "w_contact",
    "w_joint", "w_joint_velo", "w_fallen",
    "w_quat_term", "w_pos_term", "w_fallen_term",
)

# Everything that varies per manipulated object, keyed by the scene variant's
# object name (the "duck" in `--geometry duck_low_high`). Each entry is
# SELF-CONTAINED: nothing is inherited from another object, so every parameter
# for an object reads in one place and retuning one object can never perturb
# another. A new object copies a full entry and edits it.
#
# Required keys — obj_params rejects a missing or an unknown one:
#   init_qpos      (23) full qpos at reset, exactly as the viewer prints it:
#                       [16 hand joints, obj pos(3), obj quat wxyz(4)]
#   init_ctrl      (16) hand joint position targets at reset -> ctrl[:16]. Also
#                       the cost's joint "home" target, so the joint term
#                       penalizes drift away from this commanded grasp pose.
#   target_pos     (3)  goal position (goal orientation is _TARGET_QUAT above)
#   fallen_z            cost-kernel drop threshold; a squatter object would
#                       otherwise read as "fallen" while still held
#   cost_weights        exactly the keys of _COST_WEIGHT_KEYS
_OBJ_PARAMS: dict[str, dict] = {
    "cube": {
        # 16 hand joints, then the cube's settled in-palm pos(3) + quat(4).
        "init_qpos": np.array([
            7.41953443e-01, -5.14095650e-01,  6.97705793e-01,  5.73857360e-01,
            3.11686592e-01, -2.08901684e-05,  7.04119781e-01,  1.01887562e+00,
            7.14271347e-01,  2.63610945e-01,  6.97700993e-01,  6.10100133e-01,
            7.00288255e-01,  1.52604395e+00 + 1.0,  1.33871871e+00,  8.68983906e-01,

            1.70374863e-02,  3.65435775e-02,  8.36225067e-02 + 0.02,
            1, 0, 0, 0
        ]),
        "init_ctrl": np.array([
            0.7672  , -0.51303 ,  0.701455,  0.573897,  0.33472 ,  0.      ,
            0.709056,  1.01884 ,  0.74176 ,  0.26175 ,  0.701455,  0.610097,
            0.69912 ,  1.53211 ,  1.33179 ,  0.8657
        ]),
        "target_pos":    np.array([0.02, 0.035, 0.09]),#np.array([0.02, 0.03, 0.08])#np.array([0.012, 0.04, 0.085])
        "fallen_z":      0.08,
        "cost_weights": {
            "w_quat": 11.188,#100.0,
            "w_pos_x": 1.875,#60.0,   #I think X is down the fingers # separate X/Y/Z position-error weights
            "w_pos_y": 3.750,#80.0,    #Y is across the fingers
            "w_pos_z": 20.0,#15.0,
            "w_velo": 0.0,
            "w_contact": 50.0,#15.0,#12.50,
            "w_joint": 5.18,#4.0,
            "w_joint_velo": 0.0,
            "w_fallen": 200.0,
            "w_quat_term": 200.0,#500.0,
            "w_pos_term": 200.0,
            "w_fallen_term": 0.0,
        },
    },
    # TODO(retune): the hand block of init_qpos, and init_ctrl, are still the
    # cube's — only the trailing free-body block was settled against the duck.
    # Settle the hand in the viewer against env_leap_rollout_duck_low_high.xml
    # and paste the whole qpos here, plus the held ctrl. The duck's bounding
    # half-extents are (0.0325, 0.0466, 0.0512) about a centre
    # offset of (0, 0.0032, 0.0086) — i.e. ~1.5 cm taller than the 3.5 cm cube,
    # so it needs to start higher and its target sits higher too.
    "duck": {
        "init_qpos": np.array([
            5.80303253e-01, -4.60995160e-01,  1.09418735e+00,  9.71266541e-01,
            3.86906512e-01, -1.10955454e-05,  7.04409919e-01,  1.01898232e+00,
            5.78769646e-01,  2.09500294e-01,  9.62015366e-01,  1.01933705e+00,
            7.38906405e-01,  1.34131893e+00 + 1.0,  1.28450986e+00,  6.59284046e-01,
            2.56294614e-02 - 0.02,  4.96063104e-02,  8.82105693e-02 + 0.05,  
            6.97889476e-01,  1.36622853e-02, -3.23292221e-02,  7.15344982e-01
        ]),
        "init_ctrl": np.array([
            0.60184 , -0.46068 ,  1.09597 ,  0.97044 ,  0.41104 ,  0.      ,
            0.709056,  1.01884 ,  0.60184 ,  0.2094  ,  0.964465,  1.0186  ,
            0.738135,  1.251165,  1.2778  ,  0.6564
        ]),
        "target_pos":    np.array([0.03, 0.03, 0.095]),
        "fallen_z":      0.08,
        # Retuned weights for the duck. It is lighter (0.05 kg vs 0.14) and
        # rounder than the cube, so the terms most likely to want retuning are
        # w_contact and w_quat.
        "cost_weights": {
            "w_quat": 11.106,#100.0,
            "w_pos_x": 50.0,#60.0,   #I think X is down the fingers # separate X/Y/Z position-error weights
            "w_pos_y": 31.0,#80.0,    #Y is across the fingers
            "w_pos_z": 6.6,#15.0,
            "w_velo": 0.0,
            "w_contact": 50.0,#12.50,
            "w_joint": 1.70,
            "w_joint_velo": 0.0,
            "w_fallen": 100.0,
            "w_quat_term": 100.0,#500.0,
            "w_pos_term": 500.0,
            "w_fallen_term": 0.0,
        },
    },
    # TODO(tune): UNSETTLED. Every number below is the duck's, written out here
    # so spam can be retuned in place without touching the duck. Nothing was
    # measured against spamTin: the hand block of init_qpos and init_ctrl trace
    # back to the cube, the free-body tail and target_pos to the duck, and the
    # weights were tuned for a 0.05 kg duck while this tin is 0.14 kg.
    # spamTin's bounding half-extents are (0.0513, 0.0448, 0.0298) about a
    # centre offset of (0.0006, 0, 0.0003) -- a flat slab, ~2 cm shorter than
    # the duck and much wider, so it likely wants to start LOWER (and with a
    # lower target_pos / fallen_z) than these duck values.
    # The scene rests it at pos "0.01 0.04 0.08" quat "1 0 0 0"
    # (scenes/leap/env_leap_rollout_spam_low_high.xml); settle the hand around
    # it in the viewer and paste the whole qpos plus the held ctrl here.
    "spam": {
        "init_qpos": np.array([
            4.10655251e-01, -4.61003231e-01,  1.08723708e+00,  9.67698757e-01,
            2.82767934e-01, -2.46220295e-05,  7.04015722e-01,  1.01883455e+00,
            3.97543778e-01,  2.09665967e-01,  9.57814722e-01,  1.01723690e+00,
            7.38972940e-01,  1.37821740e+00 + 1.0,  1.28447502e+00,  6.59311373e-01,
            2.92811082e-02,  4.15312645e-02,  8.12087409e-02 + 0.025,  
            5.13057883e-01, -5.07358909e-01, -5.03345343e-01,  4.75396689e-01
        ]),
        "init_ctrl": np.array([
            0.43648 , -0.46068 ,  1.09597 ,  0.97044 ,  0.30928 ,  0.      ,
            0.709056,  1.01884 ,  0.42376 ,  0.2094  ,  0.964465,  1.0186  ,
            0.738135,  1.38553 ,  1.2778  ,  0.6564
        ]),
        "target_pos":    np.array([0.03, 0.04, 0.09]),
        "fallen_z":      0.08,
        "cost_weights": {
            "w_quat": 20.0,
            "w_pos_x": 15.0,   # X is down the fingers
            "w_pos_y": 15.0,   # Y is across the fingers
            "w_pos_z": 15.0,
            "w_velo": 0.0,
            "w_contact": 10.0,
            "w_joint": 4.0,
            "w_joint_velo": 0.0,
            "w_fallen": 200.0,
            "w_quat_term": 400.0,
            "w_pos_term": 400.0,
            "w_fallen_term": 0.0,
        },
    },
    # TODO(tune): UNSETTLED, same story as "spam" above -- these are the duck's
    # numbers written out so the tomato tin can be retuned in place. Nothing was
    # measured against tomatoSoupTin, whose mass is 0.14 kg (vs the 0.05 kg duck
    # the weights were tuned for).
    # tomatoSoupTin's bounding half-extents are (0.054, 0.037, 0.0367) about a
    # centre offset of (-0.0005, -0.0001, -0.0006) -- a cylinder lying on its
    # side, ~1.5 cm shorter than the duck and longer along X.
    # The scene rests it at pos "-0.00 0.05 0.07" quat "1 0 0 0"
    # (scenes/leap/env_leap_rollout_tomato_low_high.xml) -- 1 cm lower than the
    # spam tin, so this one especially wants its own settled qpos.
    "tomato": {
        "init_qpos": np.array([
            4.06353480e-01, -5.75808861e-01,  8.76286342e-01,  9.70795703e-01,
            2.86334323e-01, -2.41655761e-05,  7.04045028e-01,  1.03065633e+00,
            3.95587251e-01,  6.38444737e-01,  8.15628397e-01,  1.01806994e+00,
            6.63837652e-01,  1.08306430e+00 + 1.0,  1.28407566e+00,  6.58929725e-01,
            3.64721709e-02 - 0.01,  3.67488698e-02,  7.07468503e-02 + 0.03,  
            6.85716576e-01,  2.10744215e-01,  2.10036538e-01,  6.64277280e-01
        ]),
        "init_ctrl": np.array([
            0.43648 , -0.57585 ,  0.88078 ,  0.97044 ,  0.30928 ,  0.      ,
            0.709056,  1.03064 ,  0.42376 ,  0.63867 ,  0.821005,  1.0186  ,
            0.664845,  1.09237 ,  1.2778  ,  0.6564  
        ]),
        "target_pos":    np.array([0.035, 0.035, 0.1]),
        "fallen_z":      0.07,
        "cost_weights": {
            "w_quat": 20.0,
            "w_pos_x": 15.0,   # X is down the fingers
            "w_pos_y": 15.0,   # Y is across the fingers
            "w_pos_z": 15.0,
            "w_velo": 0.0,
            "w_contact": 10.0,
            "w_joint": 4.0,
            "w_joint_velo": 0.0,
            "w_fallen": 200.0,
            "w_quat_term": 400.0,
            "w_pos_term": 400.0,
            "w_fallen_term": 0.0,
        },
    },
}


# Array-valued _OBJ_PARAMS entries and their required lengths. obj_params
# checks these once per resolve, so a mis-sized paste from the viewer fails at
# task construction with the key named, not as a downstream shape error.
_OBJ_ARRAY_KEYS = {
    "init_qpos":  23,   # 16 hand joints + obj pos(3) + obj quat wxyz(4)
    "init_ctrl":  16,
    "target_pos": 3,
}
# Hand joints occupy the leading qpos slots; the object's free joint is the
# 7-element tail. get_inital_state cross-checks this split against the scene.
_N_HAND_JOINTS = _OBJ_ARRAY_KEYS["init_ctrl"]
_OBJ_PARAM_KEYS = frozenset(_OBJ_ARRAY_KEYS) | {"fallen_z", "cost_weights"}

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
    # otherwise read as "fallen" while still held. See _OBJ_PARAMS.
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
            max_steps          = 1000,#4000,
            success_thresholds = {"pos": 0.02, "quat": 0.04, "vel": 0.1},
            # This object's weights from _OBJ_PARAMS, emitted in
            # _COST_WEIGHT_KEYS order — which must match the weights[...]
            # indexing in grasp_reorient_cost_wp AND the weights_list below,
            # because the --weights CLI override and the param sweeps rebuild
            # the array from this dict's key order.
            cost_weights       = self.obj_params["cost_weights"],
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
        """This object's entry in _OBJ_PARAMS, resolved from the scene
        variant's object name and validated.

        Returns a fresh dict of fresh arrays every access, so callers can never
        mutate the table. cost_weights is re-emitted in _COST_WEIGHT_KEYS order
        regardless of how it is typed in the table, because the kernel indexes
        it positionally; a missing or unknown weight is an error rather than a
        silent shift of every weight after it.
        """
        obj = self.scene_variant.obj
        try:
            entry = _OBJ_PARAMS[obj]
        except KeyError:
            raise KeyError(
                f"No parameters for object {obj!r} (from scene variant "
                f"{self.scene_variant.raw!r}); known objects are "
                f"{sorted(_OBJ_PARAMS)}. Add a full entry to _OBJ_PARAMS in "
                f"{__name__}."
            ) from None

        missing = _OBJ_PARAM_KEYS - set(entry)
        unknown = set(entry) - _OBJ_PARAM_KEYS
        if missing or unknown:
            raise KeyError(
                f"_OBJ_PARAMS[{obj!r}] is malformed: "
                f"missing {sorted(missing)}, unknown {sorted(unknown)}. Each "
                f"entry is self-contained and must declare exactly "
                f"{sorted(_OBJ_PARAM_KEYS)}."
            )

        params = {"fallen_z": float(entry["fallen_z"])}
        for key, n in _OBJ_ARRAY_KEYS.items():
            arr = np.asarray(entry[key], dtype=np.float64).ravel().copy()
            if arr.shape[0] != n:
                raise ValueError(
                    f"_OBJ_PARAMS[{obj!r}][{key!r}] has {arr.shape[0]} "
                    f"elements; expected {n}."
                )
            params[key] = arr

        w = entry["cost_weights"]
        missing_w = set(_COST_WEIGHT_KEYS) - set(w)
        unknown_w = set(w) - set(_COST_WEIGHT_KEYS)
        if missing_w or unknown_w:
            raise KeyError(
                f"cost_weights for object {obj!r}: missing "
                f"{sorted(missing_w)}, unknown {sorted(unknown_w)}. Each object "
                f"declares exactly the weights {list(_COST_WEIGHT_KEYS)}, since "
                f"the cost kernel indexes them positionally."
            )
        params["cost_weights"] = {k: float(w[k]) for k in _COST_WEIGHT_KEYS}
        return params

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

        params = self.obj_params

        # Goal pose and canonical (initial-face) orientation come from the
        # per-object table, not a mocap body in the scene.
        self.target_pos          = params["target_pos"]
        self.target_quat         = _TARGET_QUAT.copy()
        self._canonical_quat     = self.target_quat.copy()
        self._face_index         = 0

        # The cost's joint target is this object's initial grasp command, so the
        # joint term penalizes drift away from the commanded grasp pose. It is
        # per-object because the home pose is (see _OBJ_PARAMS "init_ctrl").
        self.home_state = params["init_ctrl"]

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
        # both. All three blocks come from this object's _OBJ_PARAMS entry, so
        # the hand's home pose and grasp command can differ per object.
        mjm    = self.mjm
        params = self.obj_params
        q0     = params["init_qpos"]
        ctrl0  = params["init_ctrl"]

        n_hand = _N_HAND_JOINTS
        if mjm.nq != q0.shape[0] or mjm.nu != n_hand:
            raise ValueError(
                f"Scene {self.scene_variant.raw!r} has nq={mjm.nq} nu={mjm.nu}; "
                f"this task requires nq={q0.shape[0]} ({n_hand} hand joints + one "
                f"free body) and nu={n_hand}."
            )
        v0 = np.zeros(mjm.nv, dtype=np.float64)
        return q0, v0, ctrl0

    @property
    def cost_fn_wp(self) -> wp.func:
        return grasp_reorient_cost_wp

    def goal_errors(self, mjd: mujoco.MjData) -> dict[str, float]:
        """Distance to the reorientation goal, keyed like success_thresholds.

        The three terms are exactly the ones is_success thresholds, so the two
        can never drift apart, and a hyperparameter search can scalarize them by
        dividing each by its own threshold (see run_bayes_opt.py)."""
        obj_qpos_adr = int(self.index_vector[0])
        obj_qvel_adr = int(self.index_vector[1])

        pos  = mjd.qpos[obj_qpos_adr     : obj_qpos_adr + 3]
        quat = mjd.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7]
        vel  = mjd.qvel[obj_qvel_adr     : obj_qvel_adr + 6]

        return {
            "pos":  float(np.linalg.norm(pos - self.target_pos)),
            "quat": float(1.0 - np.dot(quat, self.target_quat) ** 2),
            "vel":  float(np.linalg.norm(vel)),
        }

    def is_success(self, mjd: mujoco.MjData) -> bool:
        err = self.goal_errors(mjd)
        thr = self.config.success_thresholds
        return all(err[k] < thr[k] for k in thr)

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
            goal_pose      = ((self.obj_params["target_pos"], _TARGET_QUAT)
                              if _PIN_SHOW_GOAL else None),
            goal_opacity   = _PIN_GOAL_OPACITY,
        )
