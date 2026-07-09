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
solver; the hand joints are position-controlled with inertia-scaled, critically
damped PD torques.

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
from scipy.linalg import cho_factor, cho_solve

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
    """Position control via inertia-scaled, critically damped PD torques.

    Per-joint gains are derived from the mass-matrix diagonal so the closed loop
    has a fixed natural frequency `omega` regardless of the (tiny) finger
    inertias: kp = M_ii*omega^2, kd = 2*zeta*M_ii*omega. ctrl_joint_names is in
    MuJoCo *control* order: apply_control(ctrl)[k] is the desired position for
    Pinocchio joint ctrl_joint_names[k].

    If `use_direct_gains` is True, `kp` and `kd` are used directly instead of
    computing them from `omega` and `zeta`. Both scalars are broadcast across all
    controlled joints (not inertia-scaled).
    """
    ctrl_joint_names: list[str]
    omega: float = 50.0
    zeta: float = 1.0
    gravity_comp: bool = False
    use_direct_gains: bool = True
    kp: float = 3.0
    kd: float = 0.01
    joint_damping: float = 0.1
    # Fold the joint-PD impedance dt*(kd + joint_damping) into model.armature so the
    # ADMM contact solve's effective-inertia operator (the Delassus) is built from
    # A = M + that diagonal instead of the bare mass matrix M, so the contact solve
    # doesn't treat the PD-held fingers as near-massless next to the cube (much better
    # conditioned). Integration stays the implicit-PD path (kp forward-Euler, kd +
    # joint_damping via the implicit operator); armature_pd only sets the inertia.
    # Requires use_direct_gains=True (the armature is derived from the fixed gains).
    armature_pd: bool = True


@dataclass
class PinocchioContactConfig:
    """Contact detection + constraint knobs. Each substep the ADMM solver resolves a
    mixed constraint set — frictional point contacts, joint limits, and joint friction
    — mirroring results/g1-constraint-simulation.py. Collision pairs are all
    non-adjacent, non-floor geom pairs (cube<->hand and hand<->hand)."""
    friction: float = 0.5
    use_mesh_geoms: bool = True
    floor_halfextent_thresh: float = 0.5
    admm_max_iterations: int = 5000
    mu_prox: float = 1e-4
    # --- joint-space constraints (limits from the MJCF ranges, friction bounded) ---
    add_joint_limits: bool = True
    add_joint_friction: bool = True
    # Per-joint dry-friction torque bound applied to every controlled joint (the LEAP
    # finger joints all carry actuatorfrcrange +/-0.95 in the MJCF). The eval wiring
    # may override with per-joint bounds.
    joint_friction_bound: float = 0.95
    # --- native (g1-style) per-constraint Baumgarte correctors ---------------------
    # Position-level constraints get an optional drift term Kp*residual/dt read from
    # the constraint's own residual (see _substep._apply_baumgarte_drift), CAPPED so a
    # deep residual can't inject a huge one-shot velocity (the fling / limit blow-up).
    # The unilateral ADMM constraints already enforce non-penetration and the joint
    # limits on their own; these correctors only *reduce residual* and default OFF.
    #   * contacts: normal position error (kp=0 -> velocity-level, no fling).
    #   * joint limits: violated-only push-back (g1's raw resid/dt term is unusable at
    #     dt=1e-4, where residual/dt reaches ~2e4 -> blow-up; the bare constraint
    #     already clamps the joint at its range, so this is just optional cleanup).
    contact_baumgarte_kp: float = 0.2
    joint_limit_baumgarte_kp: float = 0.2
    baumgarte_max_vel: float = 0.05        # cap (m/s) on the contact correction speed
    joint_limit_max_vel: float = 10.0       # cap (rad/s) on the joint-limit push-back


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
        # When True, apply the PD velocity term (-kv*v) as an explicit joint torque
        # evaluated at the current (q, v) and integrate with a plain forward-Euler
        # aba step, instead of the linearly-implicit backward-Euler solve. Simpler
        # but only conditionally stable (see _substep). Defaults to the implicit path.
        self._explicit_pd = explicit_pd
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

        # Wrap the merged model in a RobotWrapper for convenience (shared data, frame
        # kinematics helpers). Note RobotWrapper.BuildFromMJCF can't build this scene
        # (Pinocchio's MJCF parser follows only the first <body>), so the model is
        # still assembled via split_into_single_root_mjcfs + merge_models above.
        self._robot = pin.RobotWrapper(model, coll, vis)
        self._model = self._robot.model
        self._collision_model = coll
        self._visual_model = vis
        self._data = self._robot.data
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

        # Mirror the passive joint damping onto model.damping for visibility.
        # NOTE: Pinocchio's aba/crba do NOT auto-apply model.damping (verified) — it
        # is a no-op in the dynamics here. joint_damping is actually applied by the
        # implicit operator in _substep; this stores the value on the model only so
        # it is inspectable alongside model.armature.
        model.damping[self._ctrl_vadr] = pid.joint_damping

        # armature-PD (optional): fold the joint-PD impedance dt*(kd + joint_damping)
        # into the joint inertia so crba AND the constraint Cholesky (=> the Delassus
        # the ADMM solve uses) both see A = M + that diagonal, so the contact solve
        # doesn't treat the PD-held fingers as near-massless next to the cube.
        # Integration stays the implicit path (see _substep); this only sets inertia.
        # Constant (direct) gains only.
        if pid.armature_pd:
            if not pid.use_direct_gains:
                raise ValueError(
                    "PinocchioPdActuation.armature_pd requires use_direct_gains=True "
                    "(the armature term is derived from the fixed kd/joint_damping)."
                )
            armature = np.zeros(model.nv)
            armature[self._ctrl_vadr] = self._timestep * (pid.kd + pid.joint_damping)
            model.armature = armature

        # --- joint-space constraints (built once; re-calc'd each substep) ---------
        # Controlled (finger) joint ids; the free obj_joint is excluded so the cube
        # is never spuriously limited. Limits come straight from the MJCF ranges,
        # already on model.lower/upperPositionLimit; friction bounds from the config.
        finger_jids = [int(model.getJointId(n)) for n in pid.ctrl_joint_names]
        self._finger_idx = pin.StdVec_Index()
        for j in finger_jids:
            self._finger_idx.append(j)

        self._joint_limit_cm = self._joint_limit_cd = None
        if self._contact_cfg.add_joint_limits:
            jl = pin.JointLimitConstraintModel(
                model, self._finger_idx,
                model.lowerPositionLimit, model.upperPositionLimit,
            )
            self._joint_limit_cm = pin.ConstraintModel(jl)
            self._joint_limit_cd = self._joint_limit_cm.createData()

        self._joint_friction_cm = self._joint_friction_cd = None
        if self._contact_cfg.add_joint_friction:
            b = self._contact_cfg.joint_friction_bound
            lb = np.zeros(model.nv)
            ub = np.zeros(model.nv)
            for j in finger_jids:
                jj = model.joints[j]
                lb[jj.idx_v:jj.idx_v + jj.nv] = -b
                ub[jj.idx_v:jj.idx_v + jj.nv] = b
            jf = pin.JointFrictionConstraintModel(model, self._finger_idx, lb, ub)
            jf.setTimeStep(self._timestep)   # friction is velocity-level -> needs dt
            self._joint_friction_cm = pin.ConstraintModel(jf)
            self._joint_friction_cd = self._joint_friction_cm.createData()

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

    def _substep(self):
        pin = self._pin
        model, data = self._model, self._data
        q, v = self._q, self._v
        dt = self._timestep
        cms, cds, pens = self._detect_contacts()

        # data.M must be populated before ConstraintCholeskyDecomposition.compute,
        # regardless of the gain mode.
        pin.crba(model, data, q, pin.Convention.WORLD)

        m_diag = np.diag(data.M)[self._ctrl_vadr]

        # PD gains on the controlled joints. kp (position/stiffness) is applied
        # forward-Euler as an explicit torque below; kd (PD derivative) and the passive
        # joint_damping are velocity-proportional and go through the implicit operator.
        # kv = kd + joint_damping is their sum (the implicit velocity coefficient).
        if self._pid.use_direct_gains:
            kp = self._pid.kp
            kd = self._pid.kd
        else:
            kp = m_diag * self._pid.omega ** 2
            kd = 2.0 * self._pid.zeta * m_diag * self._pid.omega
        kv = kd + self._pid.joint_damping

        # Stiffness (+ optional gravity comp) torque, common to both paths.
        tau = np.zeros(model.nv)
        tau[self._ctrl_vadr] = kp * (self._q_des[self._ctrl_qadr] - q[self._ctrl_qadr])
        if self._pid.gravity_comp:
            tau[self._ctrl_vadr] += pin.computeGeneralizedGravity(model, data, q)[self._ctrl_vadr]

        if self._explicit_pd:
            # Simple path: add the velocity term -kv*v as an explicit torque at the
            # current v and integrate with a plain forward-Euler aba step (no implicit
            # solve). Only conditionally stable: -kv*v blows up when kv/M_ii*dt > 2,
            # which the tiny finger inertias (M_ii ~ 3e-6) hit for any nontrivial
            # damping, so use this only with small kv/large M_ii.
            tau[self._ctrl_vadr] -= kv * v[self._ctrl_vadr]

            def _implicit(v_explicit):
                return v_explicit
        else:
            # kp is applied forward-Euler (the explicit stiffness torque above); only
            # the velocity term kv = kd + joint_damping is handled implicitly.
            # Linearly-implicit (backward-Euler) damping is unconditionally stable,
            # whereas an explicit -kv*v blows up when kv/M_ii*dt > 2 (the tiny finger
            # inertias, M_ii ~ 3e-6, hit this for any nontrivial damping). Because it is
            # linear the implicit step is a single SPD solve, no Newton iteration:
            #   (A + dt*diag(kv)) v_{n+1} = A v_explicit
            # where A = data.M (= M + armature when armature_pd is on) and v_explicit is
            # the explicit free velocity from the forward-Euler stiffness torque. diag
            # entries are nonzero only on the controlled joints, so the cube's free
            # dynamics are unchanged.
            A = np.triu(data.M)
            A = A + np.triu(A, 1).T
            d_diag = np.zeros(model.nv)
            d_diag[self._ctrl_vadr] = kv        # kd + joint_damping; kp is forward-Euler
            A_op = A + np.diag(dt * d_diag)
            # A_op is SPD and constant for this substep, so factor it once and reuse the
            # factorization for both the free and post-contact solves below.
            A_factor = cho_factor(A_op)

            def _implicit(v_explicit):
                return cho_solve(A_factor, A @ v_explicit)

        v_free = _implicit(v + dt * pin.aba(model, data, q, v, tau, self._fext))

        # Assemble the full constraint set: freshly-detected contacts + the persistent
        # joint-limit and joint-friction models (built once in __init__, re-calc'd here).
        if self._joint_limit_cm is not None:
            cms.append(self._joint_limit_cm)
            cds.append(self._joint_limit_cd)
        if self._joint_friction_cm is not None:
            cms.append(self._joint_friction_cm)
            cds.append(self._joint_friction_cd)

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
        self._apply_baumgarte_drift(g, cms, cds, dt)
        has_converged = self._solver.solve(delassus, g, cms, cds, self._settings, self._result)
        #print("Conv.",has_converged)
        forces = (1.0 / dt) * self._result.retrieveConstraintImpulses()
        v_new = _implicit(v + dt * pin.aba(model, data, q, v, tau + Jc.T @ forces, self._fext))
        self._q = pin.integrate(model, q, v_new * dt)
        self._v = v_new

    def _apply_baumgarte_drift(self, g, cms, cds, dt):
        """Optional g1-style Baumgarte push-back on the position-level constraints,
        reading each constraint's own residual from its data. Both terms default OFF
        (kp=0) — the unilateral ADMM constraints already enforce non-penetration and
        the joint limits; these only *reduce residual* and are capped so a deep
        residual can't inject a huge one-shot velocity at dt=1e-4.
          * point contacts — bias the normal component (contact-frame z) apart.
          * joint limits — push back only when a limit is violated (resid < 0).
        Joint friction is velocity-level and gets no drift term (as in g1)."""
        cfg = self._contact_cfg
        idx = 0
        for cm, cd in zip(cms, cds):
            size = cm.residualSize()
            name = cm.shortname()
            if name == "PointContactConstraintModel" and cfg.contact_baumgarte_kp != 0.0:
                perr = np.asarray(cd.extract().constraint_position_error, dtype=float)
                corr = cfg.contact_baumgarte_kp * perr[2] / dt   # contact-frame z = normal
                vmax = cfg.baumgarte_max_vel
                if vmax and vmax > 0.0:
                    corr = float(np.clip(corr, -vmax, vmax))
                g[idx + 2] += corr
            elif name == "JointLimitConstraintModel" and cfg.joint_limit_baumgarte_kp != 0.0:
                resid = np.asarray(cd.extract().constraint_residual, dtype=float)
                vmax = cfg.joint_limit_max_vel
                viol = resid < 0.0                       # only correct actual violations
                corr = cfg.joint_limit_baumgarte_kp * resid / dt
                if vmax and vmax > 0.0:
                    corr = np.maximum(corr, -vmax)       # cap the push-back speed
                g[idx:idx + size][viol] += corr[viol]
            idx += size

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
