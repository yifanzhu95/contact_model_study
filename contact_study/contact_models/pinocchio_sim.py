"""Pinocchio-backed eval simulator.

Wraps a Pinocchio model + ADMM contact solver as the high-fidelity "real"
environment, exposing the EvalSimulator interface so it is interchangeable with
MujocoSimulator and DrakeSimulator. This generalizes tests/test_pinochio.py
(LEAP-hand grasp with finger<->cube and finger<->finger contacts) into a
task-agnostic simulator configured through channels, mirroring drake_sim.py.

Pinocchio's MJCF parser only follows the first <body> under <worldbody>, so a
multi-root scene (hand fingers + free object) is split into single-root MJCFs,
parsed separately, and merged via pin.appendModel into one model. Contacts are
detected each fine step (coal collision) and resolved with the ADMM constraint
solver; the hand joints are position-controlled by a compliant, torque-limited
joint constraint solved *jointly* with the contacts (see _substep), so actuation
and contact forces stay mutually consistent and the fingers do not drive through
the object.

Like DrakeSimulator, Pinocchio stores state in its own joint layout, so the
caller supplies channels mapping it onto the MuJoCo-ordered qpos/qvel the
EvalSimulator interface promises:
  * `joint_channels` — 1-DOF revolute/prismatic joints -> MuJoCo qpos/qvel addr.
  * `free_channels`  — floating bodies -> MuJoCo freejoint (7 qpos [pos, quat
                       wxyz] + 6 qvel [lin(world), ang(body-local)]).

Two convention conversions are handled internally (never via raw q slices):
  * Pinocchio free-joint quaternion is (x,y,z,w); MuJoCo's is (w,x,y,z).
  * A free joint's body world pose is oMi[jid]; its q is jointPlacement^-1 * pose
    (the parser bakes the body offset into jointPlacement), and its velocity is
    expressed in the joint/body frame (MuJoCo's linear part is world-frame).

pinocchio (and panda3d for rendering) are imported lazily so importing this
module does not require them on the GPU-rollout machines.
"""

from __future__ import annotations

import datetime
import os
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from contact_study.sim.base import EvalSimulator, EvalState, camera_pose_from_config


# ---------------------------------------------------------------------------
# Channel / actuation configuration (mirrors drake_sim.py)
# ---------------------------------------------------------------------------
@dataclass
class PinocchioJointChannel:
    """Maps one Pinocchio 1-DOF joint (by name) to a MuJoCo qpos/qvel address."""
    pin_name: str
    q_adr: int
    v_adr: int


@dataclass
class PinocchioFreeBodyChannel:
    """Maps a Pinocchio free-flyer joint (by name) to a MuJoCo freejoint
    (7 qpos [px,py,pz, qw,qx,qy,qz] / 6 qvel [vx,vy,vz, wx,wy,wz] ang body-local)."""
    pin_name: str
    q_adr: int
    v_adr: int


@dataclass
class PinocchioPdActuation:
    """Position control expressed as a compliant, torque-limited joint constraint
    solved *inside* the ADMM contact solve (see PinocchioSimulator._substep).

    Each controlled joint contributes one row of a pin.JointFrictionConstraintModel
    (a joint-space, box-bounded force whose Jacobian is the joint-selection matrix):
      * compliance   = 1/kp  -> the row behaves like a spring of stiffness kp,
      * box bound    = ±tau_max (a real torque limit; the finger pushes either way
                        but never with infinite force -> it cannot crush the cube),
      * target       = drive the joint toward q_des via a Baumgarte-style velocity
                        bias `pd_track_gain*(q_des - q)/dt` injected into the solve's
                        free constraint velocity g (mirroring the contact Baumgarte).
    Because the actuator rows and the contact rows are solved together, the finger's
    tracking force and the contact reaction are mutually consistent in one solve, so
    the finger no longer drives through the cube (the old two-stage backward-Euler PD
    reconciled them only after the fact, leaving persistent penetration).

    ctrl_joint_names is in MuJoCo *control* order: apply_control(ctrl)[k] is the
    desired position for Pinocchio joint ctrl_joint_names[k].
    """
    ctrl_joint_names: list[str]
    kp: float = 3.0
    kd: float = 0.01
    joint_damping: float = 0.1
    gravity_comp: bool = False
    # Per-joint torque limit (N*m) for the actuator constraint's box bound. Large
    # enough not to bind normal finger motion, finite so grasp force stays bounded.
    tau_max: float = 5.0
    # Position-tracking gain on the Baumgarte-style velocity bias: each substep the
    # actuator aims to close this fraction of the joint position error. Like the
    # contact baumgarte_gain it must stay well below 1.0 -- 1.0 is deadbeat (remove
    # all error in one dt) and, coupled to the contacts, goes unstable; ~0.2 tracks
    # to ~0 rest error while keeping finger<->object penetration sub-mm (validated).
    pd_track_gain: float = 0.05
    # --- accepted-but-ignored (kept so existing call sites construct unchanged; the
    # backward-Euler / inertia-scaled / armature PD paths were removed) -------------
    omega: float = 50.0
    zeta: float = 1.0
    use_direct_gains: bool = True
    armature_pd: bool = False


@dataclass
class PinocchioContactConfig:
    """Contact-detection knobs. Pairs are all non-adjacent, non-floor geom pairs
    (cube<->hand and hand<->hand); floor geoms are dropped by size."""
    friction: float = 0.5
    use_mesh_geoms: bool = True
    floor_halfextent_thresh: float = 0.5
    admm_max_iterations: int = 1000
    mu_prox: float = 1e-4
    # Baumgarte position stabilization: each step, target a separating contact
    # velocity of (baumgarte_gain * penetration_depth / dt) so the solver actively
    # removes existing penetration instead of only preventing further approach
    # (the bare velocity-level solve leaves any overlap uncorrected). 0 disables it;
    # ~0.2 removes ~20% of the penetration per substep. Values >1 over-correct.
    baumgarte_gain: float = 0.2
    # Cap (m/s) on that separating velocity. A deep or erratic mesh-contact
    # penetration otherwise asks for baumgarte_gain*pen/dt = many m/s (dt=1e-4),
    # and because the normal contact impulse is one-sided the object can never
    # reabsorb it -> it is flung to infinity. Capping the correction speed removes
    # overlap gently over several steps instead of in one impulsive kick; this is
    # the actual fix for the fling (validated in tests/test_pinocchio_baumgarte_ab.py).
    # <= 0 disables the cap (the old, fling-prone behavior).
    baumgarte_max_vel: float = 0.05
    # Penetration deadband (m): ignore the first baumgarte_slop metres of overlap
    # so sub-mm contact jitter isn't fought (reduces buzzing at rest).
    baumgarte_slop: float = 0.0005


# ---------------------------------------------------------------------------
# Model build / contact helpers (self-contained; mirrors tests/test_pinochio.py)
# ---------------------------------------------------------------------------
def split_into_single_root_mjcfs(mjcf_path, scene_dir):
    """Write one temp MJCF per <worldbody> root <body> (with the loose worldbody
    geoms folded into the first), so Pinocchio's first-body-only MJCF parser can
    read each independent root into its own model. Returns the temp paths."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    asset = root.find("asset")
    default = root.find("default")
    worldbody = root.find("worldbody")

    loose_geoms = [el for el in worldbody if el.tag == "geom"]
    bodies = [el for el in worldbody if el.tag == "body"]

    # Instance-specific suffix so concurrent runs (e.g. one Pinocchio eval per
    # HPC node/process) don't write, read, and delete the same shared temp files
    # in scene_dir and clobber each other. Timestamp for readability + PID and a
    # short uuid to stay unique across nodes and repeated calls in one process.
    token = (f"{datetime.datetime.now():%Y%m%d_%H%M%S}_"
             f"{os.getpid()}_{uuid.uuid4().hex[:8]}")

    tmp_paths = []
    for i, body in enumerate(bodies):
        new_root = ET.Element("mujoco", root.attrib)
        if compiler is not None:
            new_root.append(compiler)
        # The <default> block must travel with the body: geoms/joints reference
        # its classes (e.g. `class="tip"` supplies type+mesh), and Pinocchio's
        # MJCF parser raises IndexError('unordered_map::at') on an unresolved
        # class if it isn't carried into each single-root split.
        if default is not None:
            new_root.append(default)
        if asset is not None:
            new_root.append(asset)
        new_worldbody = ET.SubElement(new_root, "worldbody")
        # Pinocchio's MJCF parser welds the single root <body> to the universe at
        # identity and discards that body's own pos/quat. Nest the real body inside
        # a dummy identity wrapper so its transform survives as an honored *child*
        # placement — without this, a rotated/translated base (e.g. the hand palm's
        # pos/quat) lands at the origin and every descendant geom is mis-placed,
        # opening a gap so the fingers never contact the object.
        wrapper = ET.SubElement(new_worldbody, "body", {"name": f"_root_wrap_{i}"})
        if i == 0:
            for geom in loose_geoms:
                wrapper.append(geom)
        wrapper.append(body)
        tmp_path = os.path.join(scene_dir, f"_tmp_pin_sim_split_{i}_{token}.xml")
        ET.ElementTree(new_root).write(tmp_path)
        tmp_paths.append(tmp_path)
    return tmp_paths


def merge_models(pin, parts):
    """Append single-root (model, collision, visual) tuples into one combined
    model at the universe frame. appendModel merges one geometry model per call,
    so run it twice per subtree against the same pre-merge snapshot to keep the
    collision and visual geometry index-consistent. Returns (model, coll, vis)."""
    model, coll, vis = parts[0]
    aMb = pin.SE3.Identity()
    for (mB, collB, visB) in parts[1:]:
        prev = model
        model, coll = pin.appendModel(prev, mB, coll, collB, 0, aMb)
        _, vis = pin.appendModel(prev, mB, vis, visB, 0, aMb)
    return model, coll, vis


def _box_half_extents(go):
    try:
        return np.asarray(go.geometry.halfSide, dtype=float)
    except Exception:
        return None


def _is_adjacent(model, j1, j2):
    """Same joint or a direct parent-child relationship between two non-universe
    joints (a contact there would be a spurious, permanently-active constraint).

    Universe (j=0) is the computational root, not a physical body. Loose
    worldbody geoms (e.g. the palm) land at parentJoint=0, and free/revolute
    joints that are direct children of universe also have parents[jid]=0.
    Treating universe as a body in the adjacency check would filter out every
    palm<->object and palm<->finger contact, so we skip the parent-child test
    whenever either joint is universe.
    """
    if j1 == j2:
        return True
    if j1 > 0 and j2 > 0:
        return model.parents[j1] == j2 or model.parents[j2] == j1
    return False


def build_collision_pairs(pin, model, geom_model, cfg: PinocchioContactConfig):
    """Wipe the parser's default pairs and add every non-adjacent, non-floor geom
    pair (cube<->hand and hand<->hand). Returns the list of collidable geom ids."""
    geom_model.removeAllCollisionPairs()
    ids = []
    for gid, go in enumerate(geom_model.geometryObjects):
        half = _box_half_extents(go)
        is_floor = half is not None and (
            half[0] > cfg.floor_halfextent_thresh and half[1] > cfg.floor_halfextent_thresh
        )
        if is_floor:
            continue
        if half is None and not cfg.use_mesh_geoms:  # mesh geom
            continue
        ids.append(gid)

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            ga, gb = ids[a], ids[b]
            ja = geom_model.geometryObjects[ga].parentJoint
            jb = geom_model.geometryObjects[gb].parentJoint
            if _is_adjacent(model, ja, jb):
                continue
            geom_model.addCollisionPair(pin.CollisionPair(ga, gb))
    return ids


def _rotation_from_normal(n):
    """3x3 rotation whose third column is the unit normal n (contact-frame z-axis
    = contact normal; x,y span the friction-tangent plane)."""
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


# Panda3D's ShowBase is a per-process singleton — constructing a second one
# (e.g. one PinocchioSimulator per episode during a sweep) raises "Attempt to
# spawn multiple ShowBase instances!". We build the offscreen viewer once and
# reattach every simulator's visualizer to it (see _setup_viewer).
_PANDA_VIEWER = None


class PinocchioSimulator(EvalSimulator):
    def __init__(
        self,
        model_path: str,
        config,
        nq: int,
        nv: int,
        pid: PinocchioPdActuation,
        joint_channels: list[PinocchioJointChannel] | None = None,
        free_channels: list[PinocchioFreeBodyChannel] | None = None,
        contact_cfg: PinocchioContactConfig | None = None,
        video_path: str | None = None,
        render: bool = True,
        explicit_pd: bool = False,
    ):
        import pinocchio as pin

        self._pin = pin
        self._config = config
        self.nq = nq
        self.nv = nv
        self._joint_channels = joint_channels or []
        self._free_channels = free_channels or []
        self._pid = pid
        # `explicit_pd` is accepted for call-site compatibility but no longer used:
        # PD is now a constraint inside the ADMM solve, not a torque integrated by a
        # separate Euler step (see _substep and PinocchioPdActuation).
        self._contact_cfg = contact_cfg or PinocchioContactConfig()
        self._timestep = float(config.timestep)
        self._video_path = video_path
        self._want_render = render or (video_path is not None)

        # --- build the combined Pinocchio model from the (multi-root) MJCF ----
        scene_dir = os.path.dirname(model_path)
        tmp_paths = split_into_single_root_mjcfs(model_path, scene_dir)
        try:
            parts = [pin.buildModelsFromMJCF(p, contacts=False) for p in tmp_paths]
        finally:
            for p in tmp_paths:
                os.remove(p)
        model, coll, vis = merge_models(pin, parts)
        model.gravity = pin.Motion(np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0]))

        self._model = model
        self._collision_model = coll
        self._visual_model = vis
        self._data = model.createData()
        build_collision_pairs(pin, model, coll, self._contact_cfg)
        self._geom_data = pin.GeometryData(coll)
        # Ask coal to compute the contact manifold (normal + penetration depth) per
        # pair. Without this the collision results carry NaN normals (the code then
        # falls back to a crude center-to-center normal) and no penetration depth,
        # so Baumgarte stabilization has nothing to act on.
        for req in self._geom_data.collisionRequests:
            req.enable_contact = True
            req.num_max_contacts = 4

        # Resolve channel joint names -> ids / state addresses.
        self._joint_jid = {ch.pin_name: model.getJointId(ch.pin_name)
                           for ch in self._joint_channels}
        self._free_jid = {ch.pin_name: model.getJointId(ch.pin_name)
                          for ch in self._free_channels}

        # PD: pin q/v indices of the controlled joints, in ctrl order.
        self._ctrl_qadr = np.array(
            [model.joints[model.getJointId(n)].idx_q for n in pid.ctrl_joint_names],
            dtype=int,
        )
        self._ctrl_vadr = np.array(
            [model.joints[model.getJointId(n)].idx_v for n in pid.ctrl_joint_names],
            dtype=int,
        )

        # Actuator-as-constraint: one pin.JointFrictionConstraintModel over the
        # controlled joints (its Jacobian is the joint-selection matrix, its cone a
        # box on the joint force). Configured as a PD actuator: compliance = 1/kp
        # (spring of stiffness kp) and box bound = ±tau_max (torque limit). Built once
        # here (structure is state-independent); each substep it is appended to the
        # contact constraint list and its target velocity bias is set in `g`.
        ctrl_jids = pin.StdVec_Index()
        for n in pid.ctrl_joint_names:
            ctrl_jids.append(model.getJointId(n))
        act = pin.JointFrictionConstraintModel(model, ctrl_jids)
        n_act = act.residualSize()
        kp = max(float(pid.kp), 1e-9)
        act.setCompliance(np.full(n_act, 1.0 / kp))
        # Solver works in impulses (forces = impulses/dt); a torque limit tau_max is
        # an impulse bound tau_max*dt over the substep.
        lim = float(pid.tau_max) * self._timestep
        act.setFrictionLowerLimit(np.full(n_act, -lim))
        act.setFrictionUpperLimit(np.full(n_act, lim))
        self._act_cm = pin.ConstraintModel(act)
        self._act_cd = self._act_cm.createData()
        self._n_act = n_act

        # ADMM solver setup (mirrors results/admm-constraint-solver.py).
        self._solver = pin.ADMMConstraintSolver()
        s = pin.ADMMSolverSettings()
        s.max_iterations = self._contact_cfg.admm_max_iterations
        s.absolute_feasibility_tol = 1e-10
        s.relative_feasibility_tol = 1e-12
        s.absolute_complementarity_tol = 1e-10
        s.relative_complementarity_tol = 1e-12
        s.admm_update_rule = pin.ADMMUpdateRule.SPECTRAL
        s.anderson_capacity = 10
        s.mu_prox = self._contact_cfg.mu_prox
        s.stat_record = False
        s.solve_ncp = True
        self._settings = s
        self._result = pin.ADMMSolverResult()
        self._fext = [pin.Force.Zero() for _ in range(model.njoints)]

        # Running state (Pinocchio layout) + PD desired positions.
        self._q = pin.neutral(model)
        self._v = np.zeros(model.nv)
        self._q_des = self._q.copy()

        # Rendering (throttled to cam_fps, mirroring MujocoSimulator).
        self._frames: list[np.ndarray] = []
        self._t = 0.0
        self._frame_dt = 1.0 / config.cam_fps if config.cam_fps > 0 else 0.0
        self._next_frame_t = 0.0
        self._viz = None
        if self._want_render:
            self._setup_viewer()

    # -- viewer --------------------------------------------------------------
    def _setup_viewer(self):
        global _PANDA_VIEWER
        # Panda3D's headless EGL pipe must be selected before the panda3d import.
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "load-display p3headlessgl")
        from pinocchio.visualize import Panda3dVisualizer

        # Apply MJCF rgba (stored in meshColor) instead of the default material.
        for go in self._visual_model.geometryObjects:
            go.overrideMaterial = True

        viz = Panda3dVisualizer(self._model, self._collision_model, self._visual_model)
        # Reuse the one process-wide ShowBase (created on the first episode);
        # later episodes reattach to it. append_group(remove_if_exists=True)
        # inside loadViewerModel swaps in this episode's model, so reusing the
        # fixed "pin_eval" group name drops the previous episode's geometry.
        if _PANDA_VIEWER is None:
            viz.initViewer(open=False)
            _PANDA_VIEWER = viz.viewer
        else:
            viz.initViewer(viewer=_PANDA_VIEWER, open=False)
        viz.loadViewerModel(group_name="pin_eval")
        viz.displayVisuals(True)

        # Camera from the shared TaskConfig pose (same framing as MuJoCo/Drake).
        R, eye = camera_pose_from_config(self._config)
        forward = np.asarray(R)[:, 2]
        dist = float(np.linalg.norm(eye)) or 1.0
        lookat = np.asarray(eye) + forward * dist
        viz.viewer._app.camLens.set_near(0.01)
        viz.viewer.reset_camera(pos=tuple(np.asarray(eye, dtype=float)),
                                look_at=tuple(lookat.astype(float)))
        self._viz = viz

    # -- per-channel read/write ---------------------------------------------
    def _write_joint(self, ch, qpos, qvel):
        j = self._model.joints[self._joint_jid[ch.pin_name]]
        self._q[j.idx_q] = qpos[ch.q_adr]
        self._v[j.idx_v] = qvel[ch.v_adr]

    def _read_joint(self, ch, qpos, qvel):
        j = self._model.joints[self._joint_jid[ch.pin_name]]
        qpos[ch.q_adr] = self._q[j.idx_q]
        qvel[ch.v_adr] = self._v[j.idx_v]

    def _write_free(self, ch, qpos, qvel):
        pin = self._pin
        jid = self._free_jid[ch.pin_name]
        j = self._model.joints[jid]
        a, va = ch.q_adr, ch.v_adr
        # Desired world body pose (MuJoCo quat is wxyz).
        quat = np.asarray(qpos[a + 3:a + 7], dtype=float)
        n = np.linalg.norm(quat)
        quat = quat / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])
        w, x, y, z = quat
        R = pin.Quaternion(w, x, y, z).matrix()
        M_des = pin.SE3(R, np.asarray(qpos[a:a + 3], dtype=float))
        # q is jointPlacement^-1 * world pose (parser bakes offset into placement).
        M_q = self._model.jointPlacements[jid].inverse() * M_des
        qq = pin.Quaternion(M_q.rotation)
        self._q[j.idx_q:j.idx_q + 3] = M_q.translation
        self._q[j.idx_q + 3:j.idx_q + 7] = np.array([qq.x, qq.y, qq.z, qq.w])
        # Velocity: MuJoCo [lin(world), ang(body-local)] -> joint-frame [lin, ang].
        self._v[j.idx_v:j.idx_v + 3] = R.T @ np.asarray(qvel[va:va + 3], dtype=float)
        self._v[j.idx_v + 3:j.idx_v + 6] = np.asarray(qvel[va + 3:va + 6], dtype=float)

    def _read_free(self, ch, qpos, qvel):
        pin = self._pin
        jid = self._free_jid[ch.pin_name]
        j = self._model.joints[jid]
        a, va = ch.q_adr, ch.v_adr
        X = self._data.oMi[jid]               # body world pose (free joint == body)
        R = X.rotation
        qpos[a:a + 3] = X.translation
        quat = pin.Quaternion(R)
        qpos[a + 3] = quat.w
        qpos[a + 4] = quat.x
        qpos[a + 5] = quat.y
        qpos[a + 6] = quat.z
        # joint-frame velocity -> MuJoCo [lin(world), ang(body-local)].
        v_lin = self._v[j.idx_v:j.idx_v + 3]
        qvel[va:va + 3] = R @ v_lin
        qvel[va + 3:va + 6] = self._v[j.idx_v + 3:j.idx_v + 6]

    # -- contact detection + dynamics ---------------------------------------
    def _detect_contacts(self):
        pin = self._pin
        model, data = self._model, self._data
        gm, gd = self._collision_model, self._geom_data
        q = self._q
        pin.forwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, gm, gd, q)
        pin.computeCollisions(model, data, gm, gd, q, False)

        cms = []
        pens = []   # penetration depth (m, >0 == overlap) per constraint, in cms order
        for k in range(len(gm.collisionPairs)):
            cr = gd.collisionResults[k]
            if not cr.isCollision():
                continue
            cp = gm.collisionPairs[k]
            j1 = gm.geometryObjects[cp.first].parentJoint
            j2 = gm.geometryObjects[cp.second].parentJoint
            c1 = np.asarray(gd.oMg[cp.first].translation, dtype=float)
            c2 = np.asarray(gd.oMg[cp.second].translation, dtype=float)

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

            # (world contact point, penetration depth) per manifold point. coal
            # reports penetration_depth < 0 when the geoms overlap; flip the sign so
            # `pen > 0` means "penetrating by pen metres".
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
                    ct = cr.getContact(i)
                    p = np.asarray(ct.pos, dtype=float)
                    if np.all(np.isfinite(p)):  # box-box manifolds can return NaN
                        depth = float(getattr(ct, "penetration_depth", 0.0))
                        pen = -depth if np.isfinite(depth) else 0.0
                        world_points.append((p, pen))
                except Exception:
                    pass
            if not world_points:
                world_points.append((0.5 * (c1 + c2), 0.0))

            for p_world, pen in world_points:
                M_world = pin.SE3(R_n, p_world)
                plc1 = data.oMi[j1].inverse() * M_world
                plc2 = data.oMi[j2].inverse() * M_world
                cm = pin.PointContactConstraintModel(model, j1, plc1, j2, plc2)
                cm.setFriction(self._contact_cfg.friction)
                cms.append(pin.ConstraintModel(cm))
                pens.append(pen)
        cds = [cm.createData() for cm in cms]
        return cms, cds, np.asarray(pens, dtype=float)

    # Bisection depth cap for _substep's non-convergence retry: 3 halvings takes the
    # worst-case internal dt from 1e-4s to 1.25e-5s, far finer than any legitimate
    # contact event needs. Worst case (every half-step also fails to converge) costs
    # up to 2^3 + 2^2 + 2^1 = 14 ADMM solves per original substep instead of 1 --
    # kept modest because a genuinely unrecoverable configuration (e.g. commanding
    # every finger into its hard joint limit simultaneously) can otherwise make every
    # bisected half-step fail too, and deeper recursion buys ~4x more solves for no
    # improvement. Beyond this depth the state is committed as-is (protected by the
    # finite-state safety net) rather than recursing further.
    _MAX_BISECT_DEPTH = 3

    def _substep(self, dt: float | None = None, _depth: int = 0):
        pin = self._pin
        model, data = self._model, self._data
        q, v = self._q, self._v
        if dt is None:
            dt = self._timestep
        cms, cds, pens = self._detect_contacts()

        # Full constraint system for this substep: the detected point contacts
        # followed by the single joint-actuator constraint (appended LAST so the
        # contact rows stay first and the contact-Baumgarte slicing below is valid).
        n_c = len(cms)
        cms = cms + [self._act_cm]
        cds = cds + [self._act_cd]

        # data.M must be populated before ConstraintCholeskyDecomposition.compute.
        pin.crba(model, data, q, pin.Convention.WORLD)

        # Free velocity carries NO PD torque any more (the PD force is produced by the
        # actuator constraint in the solve). Optional gravity comp + passive joint
        # damping are the only explicit torques.
        tau = np.zeros(model.nv)
        if self._pid.gravity_comp:
            tau[self._ctrl_vadr] += pin.computeGeneralizedGravity(model, data, q)[self._ctrl_vadr]
        if self._pid.joint_damping:
            tau[self._ctrl_vadr] -= self._pid.joint_damping * v[self._ctrl_vadr]
        v_free = v + dt * pin.aba(model, data, q, v, tau, self._fext)

        for cm, cd in zip(cms, cds):
            cm.calc(model, data, cd)
        chol = pin.ConstraintCholeskyDecomposition(model, data, cms, cds)
        chol.compute(model, data, cms, cds, 1e-10)
        delassus = chol.getDelassusOperatorCholeskyExpression()
        Jc = pin.getConstraintsJacobian(model, data, cms, cds)
        g = Jc @ v_free

        # --- Baumgarte biases on the free constraint velocity g -------------------
        # Contact block = first 3*n_c rows (each 3D point contact is [tx, ty, n]);
        # actuator block = the trailing self._n_act rows (one per controlled joint).
        n_crows = 3 * n_c
        # Contacts: bias the normal component (every 3rd entry, contact-frame z) so
        # the solver targets a separating velocity proportional to penetration, past
        # a slop deadband and CAPPED at baumgarte_max_vel so a deep/erratic mesh
        # contact can't inject a huge one-shot velocity (the fling).
        beta = self._contact_cfg.baumgarte_gain
        if beta != 0.0 and pens.size:
            corr = beta * np.maximum(pens - self._contact_cfg.baumgarte_slop, 0.0) / dt
            vmax = self._contact_cfg.baumgarte_max_vel
            if vmax and vmax > 0.0:
                corr = np.minimum(corr, vmax)
            g[2:n_crows:3] -= corr
        # Actuator: drive each controlled joint toward q_des. The row's free velocity
        # is the current joint velocity; subtracting the desired closing velocity
        # pd_track_gain*(q_des - q)/dt makes the (compliant, torque-limited) row act
        # as a position servo of stiffness kp, solved jointly with the contacts.
        q_err = self._q_des[self._ctrl_qadr] - q[self._ctrl_qadr]
        g[n_crows:] -= self._pid.pd_track_gain * q_err / dt

        converged = self._solver.solve(delassus, g, cms, cds, self._settings, self._result)
        # On non-convergence, don't commit this solve: the returned forces are an
        # incomplete, unreliable partial solution, and blindly integrating them for a
        # "full-size" dt is exactly how a fast-appearing contact tunnels through a mesh
        # in one step (observed: penetration jumping from ~0.1mm to >20mm in a single
        # substep once contacts stopped converging, then staying pegged there because
        # the solver kept being fed an already-deep, still-unconverged state). Retry
        # this physical interval as two half-size substeps instead -- finer temporal
        # resolution gives the collision solve a chance to catch the contact while it's
        # still shallow, rather than integrating through it. Bounded by
        # _MAX_BISECT_DEPTH so a genuinely unrecoverable configuration still commits
        # (protected by the finite-state safety net below) instead of recursing forever.
        if not converged and _depth < self._MAX_BISECT_DEPTH:
            self._substep(dt / 2, _depth + 1)
            self._substep(dt / 2, _depth + 1)
            return
        forces = (1.0 / dt) * self._result.retrieveConstraintImpulses()
        v_new = v + dt * pin.aba(model, data, q, v, tau + Jc.T @ forces, self._fext)
        q_new = pin.integrate(model, q, v_new * dt)
        # Safety net: never commit a non-finite state. Without this, one bad substep
        # (e.g. an unconverged, ill-conditioned solve) turns q/v to NaN, and every
        # later call to _detect_contacts runs coal collision on degenerate NaN
        # geometry -- which floods stderr with Eigen Jacobi-eigensolver warnings on
        # every subsequent substep instead of failing once. Freeze at the last good
        # state instead (zero velocity, so it doesn't keep "moving" through NaN).
        if np.all(np.isfinite(q_new)) and np.all(np.isfinite(v_new)):
            self._q = q_new
            self._v = v_new
        else:
            self._v = np.zeros_like(v)

    # -- EvalSimulator interface --------------------------------------------
    def reset(self, qpos, qvel) -> None:
        self._t = 0.0
        self._next_frame_t = 0.0
        self._frames = []
        self.set_state(qpos, qvel)

    def set_state(self, qpos, qvel) -> None:
        qpos = np.asarray(qpos, dtype=float)
        qvel = np.asarray(qvel, dtype=float)
        for ch in self._joint_channels:
            self._write_joint(ch, qpos, qvel)
        for ch in self._free_channels:
            self._write_free(ch, qpos, qvel)
        # Keep PD desired aligned to the new pose (no command yet -> hold).
        self._q_des = self._q.copy()
        self._pin.forwardKinematics(self._model, self._data, self._q)

    def get_state(self) -> EvalState:
        self._pin.forwardKinematics(self._model, self._data, self._q)
        qpos = np.zeros(self.nq)
        qvel = np.zeros(self.nv)
        for ch in self._joint_channels:
            self._read_joint(ch, qpos, qvel)
        for ch in self._free_channels:
            self._read_free(ch, qpos, qvel)
        return EvalState(qpos, qvel)

    def apply_control(self, ctrl) -> None:
        # Clamp to the joint's own physical position limits. The actuator is now a
        # constraint *inside* the ADMM solve (see _substep), so an unclamped target
        # beyond the joint's mechanical range becomes an infeasible velocity bias for
        # the box-bounded actuator row: the solver can't satisfy it, ADMM iterations
        # saturate every substep, and (especially if the caller keeps integrating an
        # unbounded command, e.g. MPPI's u += action with no ctrlrange clip) the
        # tracking error can run away over many substeps until the state blows up to
        # NaN -- which then floods collision detection with degenerate geometry and
        # spams Eigen's Jacobi eigensolver on every subsequent substep. Clamping here
        # mirrors what a real position actuator's ctrlrange enforces upstream (and
        # exactly matches it: verified model.lowerPositionLimit/upperPositionLimit at
        # these q-indices equal the MuJoCo actuator_ctrlrange for this scene).
        ctrl = np.asarray(ctrl, dtype=float)
        lo = self._model.lowerPositionLimit[self._ctrl_qadr]
        hi = self._model.upperPositionLimit[self._ctrl_qadr]
        self._q_des[self._ctrl_qadr] = np.clip(ctrl, lo, hi)

    def step(self, n_substeps: int = 1) -> None:
        for _ in range(n_substeps):
            self._substep()
            self._t += self._timestep

    def render(self) -> None:
        if self._viz is None:
            return
        if self._frame_dt > 0 and self._t + 1e-9 < self._next_frame_t:
            return
        self._next_frame_t += self._frame_dt
        self._viz.display(self._q)
        self._frames.append(self._viz.captureImage())

    def save_video(self, path: str | None = None) -> None:
        if not self._frames:
            return
        import mediapy as media
        out = path or self._video_path
        if out is None:
            return
        kwargs = {"codec": "gif"} if str(out).lower().endswith(".gif") else {}
        media.write_video(out, self._frames, fps=int(self._config.cam_fps), **kwargs)

    @property
    def timestep(self) -> float:
        return self._timestep
