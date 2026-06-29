"""Load the LEAP hand MJCF into Pinocchio as ONE unified dynamic model, drive the
hand toward a joint sweep with PD torques, let a free cube fall under gravity, and
resolve finger<->cube contacts each step with Pinocchio's ADMM constraint solver.

The MJCF parser only follows the first <body> under <worldbody>, so the scene is
split into 5 single-root MJCFs (3 fingers, thumb, free cube), parsed separately,
then merged into one model via pin.appendModel. Contacts are computed only between
the cube and the hand geoms; the full system is integrated dynamically and rendered
to a video via EGL.

See results/admm-constraint-solver.py (ADMM solve) and results/collisions.py
(collision detection) for the patterns this builds on.
"""

import os
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
MJCF_PATH = os.path.join(
    REPO_ROOT, "scenes/leap_hand/leap_hand_right_w_sites.xml"
)
SCENE_DIR = os.path.dirname(MJCF_PATH)
OUT_PATH = os.path.join(REPO_ROOT, "videos/leap_hand_pinocchio.mp4")

N_FRAMES = 150
FPS = 30

# --- contact / dynamics tuning -------------------------------------------------
MU = 0.5             # Coulomb friction coefficient for point contacts
USE_MESH_TIPS = True  # False -> box-vs-box collision only (cube is a box: no mesh BVH)
SUBSTEPS = 20        # physics substeps per rendered frame (stability of PD + contact)
# Hand PD gains are derived per-joint from the mass-matrix diagonal so the closed
# loop is critically damped at a fixed natural frequency OMEGA regardless of the
# (tiny, ~1e-5) finger inertias — fixed scalar gains otherwise explode under
# explicit integration. kp = M_ii*OMEGA^2, kd = 2*ZETA*M_ii*OMEGA.
OMEGA = 30.0         # hand PD natural frequency [rad/s]
ZETA = 1.0           # hand PD damping ratio (1.0 = critically damped)
GRAVITY_COMP = True  # add gravity-compensation torque so the hand tracks the sweep

# Cube spawn pose from the MJCF (in WORLD frame): pos="0.01 0.0258 0.08"
# quat="0.965926 0 0.258819 0" (MJCF quat order is w,x,y,z).
OBJ_W0_POS = np.array([0.01, 0.0258, 0.08])
OBJ_W0_QUAT_WXYZ = np.array([0.965926, 0.0, 0.258819, 0.0])  # (w, x, y, z)


def set_object_world_pose(model, q, obj_jid, oq, pos, quat_wxyz):
    """Write the cube's free-joint q so its WORLD pose equals (pos, quat). The free
    joint's jointPlacement is not identity, so q is jointPlacement^-1 * desired."""
    w, x, y, z = quat_wxyz
    M_des = pin.SE3(pin.Quaternion(w, x, y, z).matrix(), np.asarray(pos, dtype=float))
    M_q = model.jointPlacements[obj_jid].inverse() * M_des
    quat = pin.Quaternion(M_q.rotation)
    q[oq:oq + 3] = M_q.translation
    q[oq + 3:oq + 7] = np.array([quat.x, quat.y, quat.z, quat.w])


def split_into_single_root_mjcfs(mjcf_path, scene_dir):
    """Pinocchio's MJCF parser only follows the first <body> under <worldbody>,
    but this model has 5 independent root bodies (3 fingers, thumb, free object).
    Write one temp MJCF per root body so each can be parsed into its own model."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    asset = root.find("asset")
    worldbody = root.find("worldbody")

    loose_geoms = [el for el in worldbody if el.tag == "geom"]
    bodies = [el for el in worldbody if el.tag == "body"]

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

        tmp_path = os.path.join(scene_dir, f"_tmp_pin_split_{i}.xml")
        ET.ElementTree(new_root).write(tmp_path)
        tmp_paths.append(tmp_path)

    return tmp_paths


def build_joint_sweep(model, n_frames=N_FRAMES):
    q0 = pin.neutral(model)
    q_traj = []
    for t in range(n_frames):
        q = q0.copy()
        phase = 2 * np.pi * t / n_frames
        for jid in range(1, model.njoints):
            joint = model.joints[jid]
            if joint.nq != 1:
                continue
            idx_q = joint.idx_q
            lo = model.lowerPositionLimit[idx_q]
            hi = model.upperPositionLimit[idx_q]
            q[idx_q] = lo + (hi - lo) * (0.5 + 0.5 * np.sin(phase))
        q_traj.append(q)
    return q_traj


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
    Returns (obj_ids, hand_ids)."""
    geom_model.removeAllCollisionPairs()
    obj_ids, hand_ids = [], []
    for gid, go in enumerate(geom_model.geometryObjects):
        if go.parentJoint == obj_jid:
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
    """One physics substep: PD torque on the hand toward q_tgt, ADMM-resolved
    contacts on the cube, full-system integration. Returns (q_new, v_new, ok)."""
    # crba populates data.M; use its diagonal to set inertia-scaled, critically
    # damped per-joint gains (kp = M_ii*OMEGA^2, kd = 2*ZETA*M_ii*OMEGA).
    pin.crba(model, data, q, pin.Convention.WORLD)
    m_diag = np.diag(data.M)[hand_v_idx]
    kp = m_diag * OMEGA ** 2
    kd = 2.0 * ZETA * m_diag * OMEGA
    tau = np.zeros(model.nv)
    tau[hand_v_idx] = (kp * (q_tgt[hand_q_idx] - q[hand_q_idx])
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


def main():
    tmp_paths = split_into_single_root_mjcfs(MJCF_PATH, SCENE_DIR)
    try:
        parts = [pin.buildModelsFromMJCF(p, contacts=False) for p in tmp_paths]
    finally:
        for p in tmp_paths:
            os.remove(p)

    n_q = sum(m.nq for m, _, _ in parts)
    n_j = sum(m.njoints - 1 for m, _, _ in parts) + 1  # shared universe joint
    model, collision_model, visual_model = merge_models(parts)
    model.gravity = pin.Motion(np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0]))
    data = model.createData()
    print(f"merged model: nq={model.nq} nv={model.nv} njoints={model.njoints} "
          f"(expected nq={n_q}, njoints={n_j})")

    obj_jid = model.getJointId("obj_joint")
    if obj_jid >= model.njoints:
        # Fallback: the only free joint (nq == 7).
        obj_jid = next(j for j in range(1, model.njoints) if model.joints[j].nq == 7)
    oq = model.joints[obj_jid].idx_q
    ov = model.joints[obj_jid].idx_v
    print(f"object joint id={obj_jid}  idx_q={oq}  idx_v={ov}")

    hand_q_idx = np.array([model.joints[j].idx_q for j in range(1, model.njoints)
                           if model.joints[j].nq == 1])
    hand_v_idx = np.array([model.joints[j].idx_v for j in range(1, model.njoints)
                           if model.joints[j].nv == 1])

    obj_ids, hand_ids = build_collision_pairs(model, collision_model, obj_jid)
    print(f"collision pairs: {len(collision_model.collisionPairs)}  "
          f"(obj geoms={len(obj_ids)}, hand geoms={len(hand_ids)})")
    geom_data = pin.GeometryData(collision_model)
    _enable_contact_manifolds(geom_data)

    # ADMM solver setup (mirrors results/admm-constraint-solver.py).
    solver = pin.ADMMConstraintSolver()
    settings = pin.ADMMSolverSettings()
    settings.max_iterations = 1000
    settings.absolute_feasibility_tol = 1e-10
    settings.relative_feasibility_tol = 1e-12
    settings.absolute_complementarity_tol = 1e-10
    settings.relative_complementarity_tol = 1e-12
    settings.admm_update_rule = pin.ADMMUpdateRule.SPECTRAL
    settings.mu_prox = 1e-6
    settings.stat_record = False
    settings.solve_ncp = True
    result = pin.ADMMSolverResult()

    fext = [pin.Force.Zero() for _ in range(model.njoints)]

    # Initial state: cube at its MJCF spawn pose in WORLD frame (compensating for
    # the free joint's non-identity jointPlacement).
    q = pin.neutral(model)
    set_object_world_pose(model, q, obj_jid, oq, OBJ_W0_POS, OBJ_W0_QUAT_WXYZ)
    pin.forwardKinematics(model, data, q)
    print(f"cube spawn world pos: {data.oMi[obj_jid].translation}")
    v = np.zeros(model.nv)

    q_sweep = build_joint_sweep(model)

    # Each geom's MJCF rgba is stored in meshColor, but the Panda3d visualizer only
    # applies it when overrideMaterial is set (otherwise it uses a flat default
    # material). Enable it so the render reflects the MJCF visuals: yellow cube,
    # slate floor, white hand.
    for go in visual_model.geometryObjects:
        go.overrideMaterial = True

    viz = Panda3dVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=False)
    viz.loadViewerModel(group_name="leap")
    viz.displayVisuals(True)
    # Default near clip plane (1.0m) clips this hand-scale (~0.2m) scene; pull it in.
    viz.viewer._app.camLens.set_near(0.01)
    viz.viewer.reset_camera(pos=(0.45, -0.4, 0.2), look_at=(0.0, 0.02, 0.02))

    dt = 1.0 / (FPS * SUBSTEPS)
    frames = []
    for t in range(N_FRAMES):
        q_cur = q_sweep[t]
        q_next = q_sweep[t + 1] if t + 1 < N_FRAMES else q_sweep[t]
        for s in range(SUBSTEPS):
            alpha = (s + 1) / SUBSTEPS
            q_tgt = (1.0 - alpha) * q_cur + alpha * q_next
            cmodels, cdatas = detect_contacts_to_constraints(
                model, data, collision_model, geom_data, q, obj_jid)
            q, v, ok = step_dynamics(
                model, data, q, v, q_tgt, hand_q_idx, hand_v_idx,
                cmodels, cdatas, solver, settings, result, dt, fext)
            if cmodels and not ok:
                print(f"[warn] frame {t} substep {s}: ADMM did not converge "
                      f"({len(cmodels)} contacts); continuing.")

        viz.display(q)
        frames.append(viz.captureImage())

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    mediapy.write_video(OUT_PATH, frames, fps=FPS)
    print(f"Saved video to {OUT_PATH}")


if __name__ == "__main__":
    main()
