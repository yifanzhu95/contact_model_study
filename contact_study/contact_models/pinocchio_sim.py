"""Pinocchio-backed eval simulator.

Wraps a Pinocchio model + ADMM contact solver as the high-fidelity "real"
environment, exposing the EvalSimulator interface so it is interchangeable with
MujocoSimulator and DrakeSimulator. The simulation scheme mirrors
tests/replay_pinocchio_controls.py exactly:

  * multi-root MJCF (hand fingers + free object) split into single-root MJCFs,
    parsed separately, and merged via pin.appendModel into one model;
  * each fine step: detect coal contacts, build a PointContactConstraintModel per
    contact with a native Baumgarte corrector, and resolve them with the ADMM
    solver;
  * the hand joints are position-controlled with an explicit forward-Euler PD:
    fixed stiffness kp, per-joint critically-damped kd = 2*zeta*sqrt(kp*M_ii).

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
from dataclasses import dataclass

import numpy as np

from contact_study.sim.base import EvalSimulator, EvalState, camera_pose_from_config

# A geom whose in-plane box half-extents both exceed this is treated as the floor
# and excluded from all collision pairs.
_FLOOR_HALFEXTENT_THRESH = 0.5


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
    """Position control via an explicit forward-Euler PD on the hand joints.

    ctrl_joint_names is in MuJoCo *control* order: apply_control(ctrl)[k] is the
    desired position for Pinocchio joint ctrl_joint_names[k]. The stiffness `kp`
    is used directly (it sets the closed-loop time constant).

    By default (use_direct_kd=False) the damping is derived per-joint from the
    mass-matrix diagonal so the loop stays critically damped at that stiffness:
    kd = 2*zeta*sqrt(kp*M_ii). Set use_direct_kd=True to instead apply a fixed
    `kd` directly to every controlled joint, bypassing the mass-matrix/zeta
    derivation entirely (zeta is then unused).

    `armature` is a per-DOF rotor inertia added to each controlled joint's mass
    (model.armature). The LEAP finger inertias are tiny (M_ii ~ 3e-6), so the
    contact solve treats the PD-held fingers as near-massless next to the cube; a
    large armature raises their effective inertia (better-conditioned Delassus,
    steadier grasp). It also enters the critically-damped kd via M_ii above
    (when use_direct_kd=False)."""
    ctrl_joint_names: list[str]
    kp: float = 3.0
    zeta: float = 1.0
    gravity_comp: bool = False
    armature: float = 0.0
    use_direct_kd: bool = False
    kd: float = 0.0


@dataclass
class PinocchioContactConfig:
    """Frictional point-contact knobs. Collision pairs are all non-adjacent,
    non-floor geom pairs (cube<->hand and hand<->hand). Each contact carries a
    native Baumgarte corrector: its drift gets a push-back Kp*position_error/dt
    (+ Kd*velocity_error/dt), read straight off the constraint in the solve loop
    (mirroring results/g1-constraint-simulation.py). Kp=0 disables it."""
    friction: float = 0.5
    use_mesh_geoms: bool = True
    baumgarte_kp: float = 0.0
    baumgarte_kd: float = 0.0
    admm_max_iterations: int = 5000


# ---------------------------------------------------------------------------
# Model build / contact helpers (self-contained; mirrors replay_pinocchio_controls.py)
# ---------------------------------------------------------------------------
def split_into_single_root_mjcfs(mjcf_path, scene_dir):
    """Write one temp MJCF per <worldbody> root <body> (with the loose worldbody
    geoms folded into the first), so Pinocchio's first-body-only MJCF parser can
    read each independent root into its own model. Returns the temp paths."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    asset = root.find("asset")
    worldbody = root.find("worldbody")

    loose_geoms = [el for el in worldbody if el.tag == "geom"]
    bodies = [el for el in worldbody if el.tag == "body"]

    # Instance-specific suffix so concurrent runs (e.g. one Pinocchio eval per
    # HPC node/process) don't write, read, and delete the same shared temp files
    # in scene_dir and clobber each other.
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

        tmp_path = os.path.join(scene_dir, f"_tmp_pin_sim_split_{token}_{i}.xml")
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
    """Return the box half-side as a length-3 array, or None for non-box geoms."""
    try:
        return np.asarray(go.geometry.halfSide, dtype=float)
    except Exception:
        return None


def _is_adjacent(model, j1, j2):
    """Same joint or a direct parent-child relationship between two non-universe
    joints (a contact there would be a spurious, permanently-active constraint).
    Universe (j=0) is the computational root, not a physical body, so the
    parent-child test is skipped whenever either joint is universe (otherwise
    every palm<->finger / palm<->object contact would be filtered out)."""
    if j1 == j2:
        return True
    if j1 > 0 and j2 > 0:
        return model.parents[j1] == j2 or model.parents[j2] == j1
    return False


def build_collision_pairs(pin, model, geom_model, use_mesh_geoms):
    """Wipe the parser's default pairs and add every non-adjacent, non-floor geom
    pair (cube<->hand and hand<->hand). Returns the list of collidable geom ids."""
    geom_model.removeAllCollisionPairs()
    ids = []
    for gid, go in enumerate(geom_model.geometryObjects):
        half = _box_half_extents(go)
        is_floor = half is not None and (
            half[0] > _FLOOR_HALFEXTENT_THRESH and half[1] > _FLOOR_HALFEXTENT_THRESH
        )
        if is_floor:
            continue
        if half is None and not use_mesh_geoms:  # mesh geom
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
    ):
        import pinocchio as pin

        self._pin = pin
        self._config = config
        self.nq = nq
        self.nv = nv
        self._joint_channels = joint_channels or []
        self._free_channels = free_channels or []
        self._pid = pid
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
        build_collision_pairs(pin, model, coll, self._contact_cfg.use_mesh_geoms)
        self._geom_data = pin.GeometryData(coll)
        # Ask coal to populate the contact manifold (normal + witness points) per
        # pair; without this the collision results carry NaN normals.
        for req in self._geom_data.collisionRequests:
            req.enable_contact = True
            req.num_max_contacts = 4

        # Resolve channel joint names -> ids.
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

        # Rotor inertia on the controlled joints: crba (=> the Delassus the ADMM
        # solve builds) and the mass-scaled kd both see A = M + armature, so the
        # contact solve doesn't treat the PD-held fingers as near-massless.
        if pid.armature:
            model.armature[self._ctrl_vadr] += pid.armature

        # ADMM solver setup (mirrors replay_pinocchio_controls.py's make_solver).
        self._solver = pin.ADMMConstraintSolver()
        s = pin.ADMMSolverSettings()
        s.max_iterations = self._contact_cfg.admm_max_iterations
        s.absolute_feasibility_tol = 1e-10
        s.relative_feasibility_tol = 1e-12
        s.absolute_complementarity_tol = 1e-10
        s.relative_complementarity_tol = 1e-12
        s.admm_update_rule = pin.ADMMUpdateRule.SPECTRAL
        s.anderson_capacity = 10
        s.admm_proximal_rule = pin.ADMMProximalRule.AUTOMATIC
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
        # later episodes reattach to it and swap in this episode's geometry via
        # the fixed "pin_eval" group name.
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
        """Detect coal contacts at the current q and turn each into a frictional
        PointContactConstraintModel with a native Baumgarte corrector. The contact
        frame's z-axis is the contact normal, so lateral (pushing) forces resolve
        correctly. Returns (constraint_models, constraint_datas)."""
        pin = self._pin
        model, data = self._model, self._data
        gm, gd = self._collision_model, self._geom_data
        cfg = self._contact_cfg
        q = self._q
        pin.forwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, gm, gd, q)
        pin.computeCollisions(model, data, gm, gd, q, False)

        baumgarte = pin.BaumgarteCorrectorParameters(cfg.baumgarte_kp, cfg.baumgarte_kd)
        cms = []
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

            # World contact points: finite coal witness points, else geom midpoint.
            world_points = []
            try:
                n_contacts = cr.numContacts()
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
                M_world = pin.SE3(R_n, p_world)
                plc1 = data.oMi[j1].inverse() * M_world
                plc2 = data.oMi[j2].inverse() * M_world
                cm = pin.PointContactConstraintModel(model, j1, plc1, j2, plc2)
                cm.setFriction(cfg.friction)
                cm.setBaumgarteCorrectorParameters(baumgarte)
                cms.append(pin.ConstraintModel(cm))
        cds = [cm.createData() for cm in cms]
        return cms, cds

    def _substep(self):
        pin = self._pin
        model, data = self._model, self._data
        q, v = self._q, self._v
        dt = self._timestep
        cms, cds = self._detect_contacts()

        # Explicit forward-Euler PD on the hand: fixed stiffness kp. Damping is
        # either a fixed kd applied directly (use_direct_kd=True) or derived
        # per-joint from the mass-matrix diagonal so the loop stays critically
        # damped at that stiffness: kd = 2*zeta*sqrt(kp*M_ii) (default).
        pin.crba(model, data, q, pin.Convention.WORLD)
        m_diag = np.diag(data.M)[self._ctrl_vadr]
        kp = self._pid.kp
        if self._pid.use_direct_kd:
            kd = np.full_like(m_diag, self._pid.kd)
        else:
            kd = 2.0 * self._pid.zeta * np.sqrt(kp * m_diag)
        tau = np.zeros(model.nv)
        tau[self._ctrl_vadr] = (kp * (self._q_des[self._ctrl_qadr] - q[self._ctrl_qadr])
                                - kd * v[self._ctrl_vadr])
        if self._pid.gravity_comp:
            tau[self._ctrl_vadr] += pin.computeGeneralizedGravity(model, data, q)[self._ctrl_vadr]

        v_free = v + dt * pin.aba(model, data, q, v, tau, self._fext)

        if len(cms) == 0:
            self._q = pin.integrate(model, q, v_free * dt)
            self._v = v_free
            return

        for cm, cd in zip(cms, cds):
            cm.calc(model, data, cd)
        chol = pin.ConstraintCholeskyDecomposition(model, data, cms, cds)
        chol.compute(model, data, cms, cds, 1e-10)
        delassus = chol.getDelassusOperatorCholeskyExpression()
        Jc = pin.getConstraintsJacobian(model, data, cms, cds)
        g = Jc @ v_free

        # Baumgarte stabilization: bias each contact's drift by its position error
        # scaled by the Kp set on the constraint (g1-style), g += Kp * perr / dt.
        idx = 0
        for cm, cd in zip(cms, cds):
            size = cm.residualSize()
            kp_b = cm.baumgarte_corrector_parameters.Kp
            if kp_b != 0.0:
                g[idx:idx + size] += kp_b * cd.extract().constraint_position_error / dt
            idx += size

        self._solver.solve(delassus, g, cms, cds, self._settings, self._result)
        forces = (1.0 / dt) * self._result.retrieveConstraintImpulses()
        v_new = v + dt * pin.aba(model, data, q, v, tau + Jc.T @ forces, self._fext)
        self._q = pin.integrate(model, q, v_new * dt)
        self._v = v_new

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
        ctrl = np.asarray(ctrl, dtype=float)
        self._q_des[self._ctrl_qadr] = ctrl

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
