"""Standalone Pinocchio replay of a pre-recorded grasp_reorient control log.

Builds the same LEAP-hand + free-cube Pinocchio/ADMM simulation as
tests/test_pinochio.py (same MJCF, same model-splitting/merging, same ADMM
contact solve) — no dependency on contact_study.contact_models or the task
abstraction (get_task/TaskConfig/rollout task/MPPI) at all.

Instead of test_pinochio.py's sinusoidal joint sweep, this script starts the
hand+cube at the grasp_reorient task's fixed initial state, settles it for a
few seconds while holding the initial grasp command (mirroring the settle
phase in contact_study/drivers/run_eval_episode*.py), then plays back a
control log — one absolute joint-target command per control step, as saved by
contact_study/drivers/run_eval_episode_record_controls.py — and renders a
video of the replay.

The hand joints in this MJCF are named "0".."15" in body/document order, which
is also MuJoCo's qpos/ctrl order and Pinocchio's post-merge joint order (each
single-root sub-model is appended in the same worldbody order), so recorded
control rows map onto Pinocchio joints with no name translation.

Run headless under xvfb (Panda3D's EGL pipe still needs a display context):
    python tests/replay_pinocchio_controls.py --controls results/controls/grasp_reorient_M2_controls.npy
"""

import argparse
import datetime
import os
import uuid
import xml.etree.ElementTree as ET

from panda3d.core import loadPrcFileData

# Force Panda3D's EGL-based headless GL pipe instead of the default GLX pipe,
# which would otherwise require a real X server / DISPLAY.
loadPrcFileData("", "load-display p3headlessgl")

import mediapy
import numpy as np
import pinocchio as pin
from pinocchio.visualize import Panda3dVisualizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(REPO_ROOT, "scenes/leap_hand/leap_hand_right_w_sites.xml")
SCENE_DIR = os.path.dirname(MJCF_PATH)
OUT_PATH = os.path.join(REPO_ROOT, "videos/grasp_reorient_pinocchio_replay.mp4")

FPS = 30

# --- scene composition ----------------------------------------------------------
INCLUDE_CUBE = True   # False -> build the hand alone, with no free "obj" cube at all

# --- contact / dynamics tuning (mirrors grasp_reorient.py's Pinocchio eval tuning,
# itself derived from this file's own scheme) ----------------------------------
MU = 0.5              # Coulomb friction coefficient for point contacts
USE_MESH_TIPS = True  # False -> box-vs-box collision only (cube is a box: no mesh BVH)
KP = 3.0              # hand position-servo stiffness (fixed, not mass-scaled)
ZETA = 1.0            # hand PD damping ratio (1.0 = critically damped)
GRAVITY_COMP = False   # add gravity-compensation torque so the hand holds its target

# --- grasp_reorient's fixed initial state (contact_study/tasks/grasp_reorient.py) --
# qpos = [16 hand joint angles, obj pos(3), obj quat(wxyz)(4)]  (nq = 23)
# ctrl = [16 hand joint position targets]                        (nu = 16)
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

_OBJ_POS0  = _INIT_QPOS[16:19]
_OBJ_QUAT0 = _INIT_QPOS[19:23]  # wxyz

# Camera pose, copied from contact_study/tasks/grasp_reorient.py's TaskConfig
# (cam_pos/cam_rotmat), which is what run_eval_episode.py's eval sim renders
# with: the "top" camera, pos="0.2 0.02 0.4" xyaxes="0 1 0  -1 0 0.5".
_CAM_POS     = np.array([0.2, 0.02, 0.4])
_cam_right   = np.array([0.0, 1.0, 0.0]);  _cam_right /= np.linalg.norm(_cam_right)
_cam_up      = np.array([-1.0, 0.0, 0.5]); _cam_up    /= np.linalg.norm(_cam_up)
_cam_fwd     = -np.cross(_cam_right, _cam_up)   # camera -z = viewing direction
_cam_down    = -_cam_up
_CAM_ROTMAT  = np.column_stack([_cam_right, _cam_down, _cam_fwd])

# --- fine-step timing (grasp_reorient.py's TaskConfig.timestep / eval_substeps) --
TIMESTEP                  = 0.0001  # fine Pinocchio substep dt
EVAL_SUBSTEPS_PER_ROLLOUT = 10      # eval steps per rollout step


def set_object_world_pose(model, q, obj_jid, oq, pos, quat_wxyz):
    """Write the cube's free-joint q so its WORLD pose equals (pos, quat). The free
    joint's jointPlacement is not identity, so q is jointPlacement^-1 * desired."""
    w, x, y, z = quat_wxyz
    M_des = pin.SE3(pin.Quaternion(w, x, y, z).matrix(), np.asarray(pos, dtype=float))
    M_q = model.jointPlacements[obj_jid].inverse() * M_des
    quat = pin.Quaternion(M_q.rotation)
    q[oq:oq + 3] = M_q.translation
    q[oq + 3:oq + 7] = np.array([quat.x, quat.y, quat.z, quat.w])


def split_into_single_root_mjcfs(mjcf_path, scene_dir, include_cube=True):
    """Pinocchio's MJCF parser only follows the first <body> under <worldbody>,
    but this model has 5 independent root bodies (3 fingers, thumb, free object).
    Write one temp MJCF per root body so each can be parsed into its own model.
    If include_cube is False, the free "obj" body is dropped entirely, so the
    merged model ends up hand-only (no free joint, no cube geometry)."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    asset = root.find("asset")
    worldbody = root.find("worldbody")

    loose_geoms = [el for el in worldbody if el.tag == "geom"]
    bodies = [el for el in worldbody if el.tag == "body"]
    if not include_cube:
        bodies = [b for b in bodies if b.attrib.get("name") != "obj"]

    token = (f"{datetime.datetime.now():%Y%m%d_%H%M%S}_"
             f"{os.getpid()}_{uuid.uuid4().hex[:8]}")

    tmp_paths = []
    for i, body in enumerate(bodies):
        new_root = ET.Element("mujoco", root.attrib)
        if compiler is not None:
            new_root.append(compiler)
        if asset is not None:
            new_root.append(asset)
        new_worldbody = ET.SubElement(new_root, "worldbody")
        if i == 0:
            for geom in loose_geoms:
                new_worldbody.append(geom)
        new_worldbody.append(body)

        tmp_path = os.path.join(scene_dir, f"_tmp_pin_split_{token}_{i}.xml")
        ET.ElementTree(new_root).write(tmp_path)
        tmp_paths.append(tmp_path)

    return tmp_paths


def merge_models(parts):
    """Append the 5 single-root (model, collision, visual) tuples into one combined
    model at the universe frame (frame 0). appendModel merges one geometry model per
    call, so run it twice per subtree against the same pre-merge snapshot to keep the
    collision and visual geometry index-consistent. Returns (model, coll, vis)."""
    model, coll, vis = parts[0]
    aMb = pin.SE3.Identity()
    for (mB, collB, visB) in parts[1:]:
        prev = model
        model, coll = pin.appendModel(prev, mB, coll, collB, 0, aMb)
        _, vis = pin.appendModel(prev, mB, vis, visB, 0, aMb)
    return model, coll, vis


def _box_half_extents(go):
    """Return the box half-side as a length-3 array, or None for non-box geoms."""
    try:
        return np.asarray(go.geometry.halfSide, dtype=float)
    except Exception:
        return None


def _is_adjacent(model, j1, j2):
    """True if joints j1/j2 are the same or in a direct parent-child relationship
    (their geoms are always near each other at the shared joint, so a contact there
    would be a spurious, permanently-active constraint)."""
    return j1 == j2 or model.parents[j1] == j2 or model.parents[j2] == j1


def build_collision_pairs(model, geom_model, obj_jid):
    """Wipe the parser's default pairs and add (a) cube<->hand and (b) hand<->hand
    self-collision pairs, both excluding the floor and adjacent (parent-child) body
    pairs. Geoms are classified by parentJoint (robust; MJCF auto-names geoms).
    obj_jid may be None (no cube in the scene), in which case every non-floor geom
    is just a hand geom and only hand<->hand pairs are added.
    Returns (obj_ids, hand_ids)."""
    geom_model.removeAllCollisionPairs()
    obj_ids, hand_ids = [], []
    for gid, go in enumerate(geom_model.geometryObjects):
        if obj_jid is not None and go.parentJoint == obj_jid:
            obj_ids.append(gid)
            continue
        half = _box_half_extents(go)
        is_floor = half is not None and half[0] > 0.5 and half[1] > 0.5
        if is_floor:
            continue
        is_mesh = half is None
        if is_mesh and not USE_MESH_TIPS:
            continue
        hand_ids.append(gid)

    # cube <-> hand
    for oi in obj_ids:
        for hi in hand_ids:
            geom_model.addCollisionPair(pin.CollisionPair(oi, hi))
    # hand <-> hand (skip adjacent bodies to avoid spurious at-the-joint contacts)
    for a in range(len(hand_ids)):
        for b in range(a + 1, len(hand_ids)):
            ga, gb = hand_ids[a], hand_ids[b]
            ja = geom_model.geometryObjects[ga].parentJoint
            jb = geom_model.geometryObjects[gb].parentJoint
            if _is_adjacent(model, ja, jb):
                continue
            geom_model.addCollisionPair(pin.CollisionPair(ga, gb))
    return obj_ids, hand_ids


def _enable_contact_manifolds(geom_data):
    """Best-effort: ask coal to populate contact points (position/normal) per pair.
    The attribute path varies across pinocchio/coal versions, so guard everything."""
    try:
        for req in geom_data.collisionRequests:
            req.enable_contact = True
            req.num_max_contacts = 4
    except Exception:
        pass


def _rotation_from_normal(n):
    """3x3 rotation whose third column is the unit normal n (the contact frame's
    z-axis = contact normal; x,y span the friction-tangent plane). Falls back to
    identity for a degenerate normal."""
    n = np.asarray(n, dtype=float)
    nn = np.linalg.norm(n)
    if nn < 1e-9 or not np.all(np.isfinite(n)):
        return np.eye(3)
    z = n / nn
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = a - z * (a @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def detect_contacts_to_constraints(model, data, geom_model, geom_data, q, obj_jid):
    """Run collision detection at q and convert every colliding pair (cube<->hand
    and hand<->hand) into a PointContactConstraintModel. The contact frame is
    oriented so its z-axis is the contact normal (approximated by the direction
    between the two geom centers), which is essential for sideways pushing —
    an identity rotation would assume a world-up normal and ignore lateral
    contact forces."""
    pin.forwardKinematics(model, data, q)
    pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
    pin.computeCollisions(model, data, geom_model, geom_data, q, False)

    constraint_models = []
    for k in range(len(geom_model.collisionPairs)):
        cr = geom_data.collisionResults[k]
        if not cr.isCollision():
            continue
        cp = geom_model.collisionPairs[k]
        j1 = geom_model.geometryObjects[cp.first].parentJoint
        j2 = geom_model.geometryObjects[cp.second].parentJoint
        c1 = np.asarray(geom_data.oMg[cp.first].translation, dtype=float)
        c2 = np.asarray(geom_data.oMg[cp.second].translation, dtype=float)

        # Contact normal: prefer coal's finite normal, else the center-to-center
        # direction (body1 -> body2).
        normal = None
        try:
            nrm = np.asarray(cr.getContact(0).normal, dtype=float)
            if np.all(np.isfinite(nrm)) and np.linalg.norm(nrm) > 1e-9:
                normal = nrm
        except Exception:
            pass
        if normal is None:
            normal = c2 - c1
        R_n = _rotation_from_normal(normal)

        # World contact points: finite coal witness points, else geom-center midpoint.
        world_points = []
        try:
            n_contacts = cr.numContacts()
        except Exception:
            try:
                n_contacts = len(cr.getContacts())
            except Exception:
                n_contacts = 0
        for i in range(n_contacts):
            try:
                p = np.asarray(cr.getContact(i).pos, dtype=float)
                if np.all(np.isfinite(p)):  # box-box manifolds can return NaN
                    world_points.append(p)
            except Exception:
                pass
        if not world_points:
            world_points.append(0.5 * (c1 + c2))

        for p_world in world_points:
            # Same world contact frame (point + normal) expressed in each joint's
            # local coordinates.
            M_world = pin.SE3(R_n, p_world)
            plc1 = data.oMi[j1].inverse() * M_world
            plc2 = data.oMi[j2].inverse() * M_world
            cm = pin.PointContactConstraintModel(model, j1, plc1, j2, plc2)
            cm.setFriction(MU)
            constraint_models.append(pin.ConstraintModel(cm))

    constraint_datas = [cm.createData() for cm in constraint_models]
    return constraint_models, constraint_datas


def step_dynamics(model, data, q, v, q_tgt, hand_q_idx, hand_v_idx,
                  constraint_models, constraint_datas,
                  solver, settings, result, dt, fext):
    """One physics substep: fixed-stiffness PD torque on the hand toward q_tgt
    (kp=KP, kd derived per-joint from the mass matrix so the loop is critically
    damped at that stiffness: kd = 2*zeta*sqrt(kp*M_ii)), ADMM-resolved contacts
    on the cube, full-system explicit integration. Returns (q_new, v_new, ok)."""
    pin.crba(model, data, q, pin.Convention.WORLD)
    m_diag = np.diag(data.M)[hand_v_idx]
    kd = 2.0 * ZETA * np.sqrt(KP * m_diag)
    tau = np.zeros(model.nv)
    tau[hand_v_idx] = (KP * (q_tgt[hand_q_idx] - q[hand_q_idx])
                       - kd * v[hand_v_idx])
    if GRAVITY_COMP:
        g_tau = pin.computeGeneralizedGravity(model, data, q)
        tau[hand_v_idx] += g_tau[hand_v_idx]

    v_free = v + dt * pin.aba(model, data, q, v, tau, fext)

    if len(constraint_models) == 0:
        q_new = pin.integrate(model, q, v_free * dt)
        return q_new, v_free, True

    for cm, cd in zip(constraint_models, constraint_datas):
        cm.calc(model, data, cd)

    chol = pin.ConstraintCholeskyDecomposition(
        model, data, constraint_models, constraint_datas)
    chol.compute(model, data, constraint_models, constraint_datas, 1e-10)
    delassus = chol.getDelassusOperatorCholeskyExpression()

    Jc = pin.getConstraintsJacobian(model, data, constraint_models, constraint_datas)
    g = Jc @ v_free

    converged = solver.solve(
        delassus, g, constraint_models, constraint_datas, settings, result)

    impulses = result.retrieveConstraintImpulses()
    forces = (1.0 / dt) * impulses
    tau_c = Jc.T @ forces
    v_new = v + dt * pin.aba(model, data, q, v, tau + tau_c, fext)
    q_new = pin.integrate(model, q, v_new * dt)
    return q_new, v_new, converged


def build_model():
    """Load + merge the LEAP-hand MJCF into one Pinocchio model, wire up the
    cube<->hand / hand<->hand ADMM collision pairs, and return everything the
    replay loop needs."""
    tmp_paths = split_into_single_root_mjcfs(MJCF_PATH, SCENE_DIR, include_cube=INCLUDE_CUBE)
    try:
        parts = [pin.buildModelsFromMJCF(p, contacts=False) for p in tmp_paths]
    finally:
        for p in tmp_paths:
            os.remove(p)

    model, collision_model, visual_model = merge_models(parts)
    model.gravity = pin.Motion(np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0]))
    data = model.createData()
    print(f"merged model: nq={model.nq} nv={model.nv} njoints={model.njoints}")

    obj_jid = None
    if INCLUDE_CUBE:
        obj_jid = model.getJointId("obj_joint")
        if obj_jid >= model.njoints:
            # Fallback: the only free joint (nq == 7).
            obj_jid = next(j for j in range(1, model.njoints) if model.joints[j].nq == 7)

    hand_q_idx = np.array([model.joints[j].idx_q for j in range(1, model.njoints)
                           if model.joints[j].nq == 1])
    hand_v_idx = np.array([model.joints[j].idx_v for j in range(1, model.njoints)
                           if model.joints[j].nv == 1])
    if len(hand_q_idx) != len(_INIT_CTRL):
        raise ValueError(
            f"found {len(hand_q_idx)} hinge joints, expected {len(_INIT_CTRL)} "
            f"to match _INIT_CTRL/the recorded control log's column count."
        )

    obj_ids, hand_ids = build_collision_pairs(model, collision_model, obj_jid)
    print(f"collision pairs: {len(collision_model.collisionPairs)}  "
          f"(obj geoms={len(obj_ids)}, hand geoms={len(hand_ids)})")
    geom_data = pin.GeometryData(collision_model)
    _enable_contact_manifolds(geom_data)

    return model, data, collision_model, visual_model, geom_data, obj_jid, hand_q_idx, hand_v_idx


def make_solver():
    solver = pin.ADMMConstraintSolver()
    settings = pin.ADMMSolverSettings()
    settings.max_iterations = 5000
    settings.absolute_feasibility_tol = 1e-10
    settings.relative_feasibility_tol = 1e-12
    settings.absolute_complementarity_tol = 1e-10
    settings.relative_complementarity_tol = 1e-12
    settings.admm_update_rule = pin.ADMMUpdateRule.SPECTRAL
    settings.anderson_capacity = 10
    settings.mu_prox = 1e-4
    settings.stat_record = False
    settings.solve_ncp = True
    result = pin.ADMMSolverResult()
    return solver, settings, result


def replay(controls: np.ndarray, out_path: str = OUT_PATH, settle_seconds: float = 1.0,
           eval_substeps: int = EVAL_SUBSTEPS_PER_ROLLOUT, substeps_per_control: int = 16):
    (model, data, collision_model, visual_model, geom_data,
     obj_jid, hand_q_idx, hand_v_idx) = build_model()
    solver, settings, result = make_solver()
    fext = [pin.Force.Zero() for _ in range(model.njoints)]

    # --- initial state: grasp_reorient's fixed hand (+ cube, if included) pose --
    q = pin.neutral(model)
    q[hand_q_idx] = _INIT_QPOS[:len(hand_q_idx)]
    if obj_jid is not None:
        oq = model.joints[obj_jid].idx_q
        set_object_world_pose(model, q, obj_jid, oq, _OBJ_POS0, _OBJ_QUAT0)
    v = np.zeros(model.nv)
    pin.forwardKinematics(model, data, q)
    if obj_jid is not None:
        print(f"cube spawn world pos: {data.oMi[obj_jid].translation}")

    for go in visual_model.geometryObjects:
        go.overrideMaterial = True

    viz = Panda3dVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=False)
    viz.loadViewerModel(group_name="leap")
    viz.displayVisuals(True)
    viz.viewer._app.camLens.set_near(0.01)
    # Same camera pose PinocchioSimulator._setup_viewer derives from a TaskConfig:
    # forward = R[:, 2], look_at = pos + forward * |pos|.
    _forward = _CAM_ROTMAT[:, 2]
    _lookat = _CAM_POS + _forward * (np.linalg.norm(_CAM_POS) or 1.0)
    viz.viewer.reset_camera(pos=tuple(_CAM_POS), look_at=tuple(_lookat))

    q_tgt_full = np.zeros(model.nq)

    # --- settle: hold the initial grasp command so the hand doesn't go limp --
    n_settle = int(settle_seconds / (TIMESTEP * eval_substeps))
    print(f"settling {settle_seconds}s ({n_settle} steps of {eval_substeps} substeps)...")
    q_tgt_full[hand_q_idx] = _INIT_CTRL
    for _ in range(n_settle):
        for _ in range(eval_substeps):
            cmodels, cdatas = detect_contacts_to_constraints(
                model, data, collision_model, geom_data, q, obj_jid)
            q, v, ok = step_dynamics(
                model, data, q, v, q_tgt_full, hand_q_idx, hand_v_idx,
                cmodels, cdatas, solver, settings, result, TIMESTEP, fext)

    # --- replay the recorded control log ------------------------------------
    substeps_per_row = substeps_per_control * eval_substeps
    frames = []
    print(f"replaying {len(controls)} control steps ({substeps_per_row} fine steps each)...")
    for t, u in enumerate(controls):
        q_tgt_full[hand_q_idx] = u
        for s in range(substeps_per_row):
            cmodels, cdatas = detect_contacts_to_constraints(
                model, data, collision_model, geom_data, q, obj_jid)
            q, v, ok = step_dynamics(
                model, data, q, v, q_tgt_full, hand_q_idx, hand_v_idx,
                cmodels, cdatas, solver, settings, result, TIMESTEP, fext)
            if cmodels and not ok:
                print(f"[warn] control step {t} substep {s}: ADMM did not converge "
                      f"({len(cmodels)} contacts); continuing.")

        viz.display(q)
        frames.append(viz.captureImage())
        if t % 25 == 0:
            print(f"  step {t:4d}/{len(controls)}  u[0]={float(u[0]):+8.3f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mediapy.write_video(out_path, frames, fps=FPS)
    print(f"Saved video to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controls", type=str, required=True,
                   help="Path to a .npy control log (shape (n_steps, 16)), e.g. as "
                        "saved by contact_study/drivers/run_eval_episode_record_controls.py.")
    p.add_argument("--out", type=str, default=OUT_PATH)
    p.add_argument("--settle", type=float, default=1.0)
    p.add_argument("--eval_substeps", type=int, default=EVAL_SUBSTEPS_PER_ROLLOUT,
                   help="Fine Pinocchio steps per rollout step (task default: 10).")
    p.add_argument("--substeps", type=int, default=16,
                   help="MPPI rollout substeps per control step used when the control "
                        "log was recorded (must match, to know how many fine steps "
                        "correspond to each recorded row).")
    args = p.parse_args()

    controls = np.load(args.controls)
    if controls.ndim != 2:
        raise ValueError(f"expected a 2D (n_steps, nu) control array, got shape {controls.shape}")

    replay(controls, out_path=args.out, settle_seconds=args.settle,
           eval_substeps=args.eval_substeps, substeps_per_control=args.substeps)


if __name__ == "__main__":
    main()
