"""Pinocchio with an ADMM contact solver, wrapped as a ``Simulator``.

A rigid-body model built from the task's MJCF, with contacts detected by coal
and resolved as frictional point contacts by Pinocchio's ADMM constraint
solver. Ported from ``contact_study/contact_models/pinocchio_sim.py``.

**Where parameters come from.** Everything the MJCF defines is read from the
MJCF, through ``MjcfModelInfo`` (MuJoCo's compiler), not from Pinocchio's own
parser, which ignores ``<default>`` classes and loses armature when models are
merged:

* per-DOF armature, damping and frictionloss;
* joint limits (only where ``limited``) and their margins;
* which geom pairs can collide (``contype``/``conaffinity``, weld and
  parent-child filtering, ``<exclude>``), and each pair's friction, combined
  with MuJoCo's max rule;
* the position servos: ``kp``, ``kv``, ``ctrlrange`` and force limits.

``PinocchioConfig`` holds only what the MJCF cannot say: solver settings.

**Dynamics, per step.** Detect contacts; build one point-contact constraint per
contact point (with a Baumgarte position corrector), plus joint-limit and
dry-friction constraints; solve for the constraint impulses with ADMM; then
integrate semi-implicitly. The servos are explicit:

    tau_act = clip(kp * (clip(u, ctrlrange) - q) - kv * qdot, forcerange)
    tau     = tau_act - damping * qdot

The passive damping is outside the clip, as in MuJoCo, where it is
``qfrc_passive`` and not part of the actuator force.

**State layout.** ``GetState``/``SetState`` use MuJoCo's ``qpos``/``qvel``
layout, so this simulator is interchangeable with ``Mujoco`` and a MuJoCo
renderer can draw its state. Two conventions are converted internally: a
Pinocchio free joint stores its quaternion ``(x, y, z, w)`` (MuJoCo:
``(w, x, y, z)``), and its linear velocity in the body frame (MuJoCo: world).

Pinocchio is imported lazily, so importing this module does not need it.
"""

from __future__ import annotations

import datetime
import math
import os
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ContactModelStudy.Simulators.Simulator import Simulator, SimulatorConfig
from ContactModelStudy.Utils.MjcfModelInfo import MjcfModelInfo

_UPDATE_RULES = ("spectral", "linear")

# Below this many collision pairs, the AABB-tree broadphase costs more to
# maintain than the narrowphase it saves, so every pair is narrowphased.
_BROADPHASE_MIN_PAIRS = 64


@dataclass
class PinocchioConfig(SimulatorConfig):
    """Physics parameters for the Pinocchio simulator.

    ``timestep``, ``substeps`` and ``gravity`` from ``SimulatorConfig``. The
    rest are solver settings, which the MJCF has no way to express. Everything
    physical (friction, damping, armature, limits, servo gains, which geoms
    collide) comes from the MJCF instead; see the module docstring.

    Attributes:
        admm_max_iterations: ADMM iteration cap per step.
        absolute_tolerance: Absolute feasibility and complementarity tolerance.
        relative_tolerance: Relative feasibility and complementarity tolerance.
        solve_ncp: ``True`` solves the nonlinear complementarity problem (true
            Coulomb friction). ``False`` solves its convex relaxation (CCP),
            which is easier to solve but lets sliding contacts drift apart.
        admm_update_rule: How ADMM adapts its penalty, ``"spectral"`` or
            ``"linear"``.
        anderson_capacity: History length for Anderson acceleration; 0 turns
            it off.
        baumgarte_kp: Baumgarte position gain on each contact. A penetration
            ``e`` adds ``kp * e / dt`` of separating velocity. 0 disables it.
        baumgarte_kd: Baumgarte velocity gain on each contact.
        delassus_regularization: Diagonal regularization added when factoring
            the Delassus operator.
        use_convex_hulls: Replace every triangle-mesh collision geom by its
            convex hull. coal then uses its analytic convex-convex path, which
            gives one stable normal and an accurate depth, instead of
            per-triangle queries whose normals jump between facets and made
            objects stick to mesh fingertips.
        max_contacts_per_pair: Contact points coal may return per geom pair.
    """

    admm_max_iterations: int = 1000
    absolute_tolerance: float = 1e-10
    relative_tolerance: float = 1e-12
    solve_ncp: bool = True
    admm_update_rule: str = "spectral"
    anderson_capacity: int = 20
    baumgarte_kp: float = 10.0
    baumgarte_kd: float = 0.0
    delassus_regularization: float = 1e-10
    use_convex_hulls: bool = True
    max_contacts_per_pair: int = 32

    def __post_init__(self) -> None:
        super().__post_init__()

        def bad(name, why):
            raise ValueError(f"{name} {why}, got {getattr(self, name)!r}")

        if self.admm_max_iterations < 1:
            bad("admm_max_iterations", "must be >= 1")
        if self.absolute_tolerance <= 0:
            bad("absolute_tolerance", "must be positive")
        if self.relative_tolerance <= 0:
            bad("relative_tolerance", "must be positive")
        if self.admm_update_rule not in _UPDATE_RULES:
            bad("admm_update_rule", f"must be one of {_UPDATE_RULES}")
        if self.anderson_capacity < 0:
            bad("anderson_capacity", "must be >= 0")
        if self.baumgarte_kp < 0:
            bad("baumgarte_kp", "must be >= 0")
        if self.baumgarte_kd < 0:
            bad("baumgarte_kd", "must be >= 0")
        if self.delassus_regularization < 0:
            bad("delassus_regularization", "must be >= 0")
        if self.max_contacts_per_pair < 1:
            bad("max_contacts_per_pair", "must be >= 1")


class Pinocchio(Simulator):
    """One Pinocchio world, stepped on the CPU.

    Attributes:
        info: The ``MjcfModelInfo`` every physical parameter was taken from.
        model, collision_model, data, geometry_data: The Pinocchio objects.
    """

    def __init__(self, xml: str | Path, sim_config: PinocchioConfig | None = None):
        """Build the model from an MJCF file.

        Args:
            xml: Path to the MJCF. Must be a path: the model is split into
                temporary per-root files next to it, and meshes resolve
                relative to it.
            sim_config: Solver parameters. Defaults to ``PinocchioConfig()``.

        Raises:
            ValueError: If ``xml`` is inline XML, or the MJCF uses a joint or
                actuator type this simulator cannot reproduce.
        """
        super().__init__(xml, sim_config if sim_config is not None else PinocchioConfig())
        if self.model_path is None:
            raise ValueError("Pinocchio needs the MJCF as a file path, not inline XML.")
        import pinocchio as pin

        self._pin = pin
        cfg = self.config
        self.info = info = MjcfModelInfo.fromXml(self.model_path)
        for j in info.joints:
            if j.type == "ball":
                raise ValueError(f"ball joint {j.name!r} is not supported")

        # -- model -----------------------------------------------------------
        # Pinocchio's MJCF parser reads only the first root body, so each root
        # body is parsed from its own temporary file and the parts are merged.
        path = str(self.model_path)
        tmp_paths, root_poses = _splitIntoSingleRootMjcfs(path, os.path.dirname(path))
        try:
            parts = [pin.buildModelsFromMJCF(p, contacts=False) for p in tmp_paths]
        finally:
            for p in tmp_paths:
                os.remove(p)
        model, coll, _vis = _mergeModels(pin, parts, root_poses)
        model.gravity = pin.Motion(np.array([*cfg.gravity, 0.0, 0.0, 0.0]))
        self.model = model
        self.collision_model = coll
        self.data = model.createData()

        # -- joint map: MuJoCo layout <-> Pinocchio layout -------------------
        self._joints = []   # (JointInfo, pin joint id)
        for j in info.joints:
            if not model.existJointName(j.name):
                raise ValueError(f"joint {j.name!r} is in the MJCF but not the Pinocchio model")
            self._joints.append((j, model.getJointId(j.name)))

        # Per-DOF parameters, in Pinocchio's velocity layout. Armature goes into
        # the model (crba and aba read it); damping is applied by hand, because
        # aba ignores model.damping.
        self._damping = np.zeros(model.nv)
        for j, jid in self._joints:
            iv = model.joints[jid].idx_v
            model.armature[iv:iv + j.nv] = j.armature
            self._damping[iv:iv + j.nv] = j.damping

        # -- servos ----------------------------------------------------------
        acts = info.actuators
        self._ctrl_q = np.array([model.joints[model.getJointId(a.joint)].idx_q for a in acts], dtype=int)
        self._ctrl_v = np.array([model.joints[model.getJointId(a.joint)].idx_v for a in acts], dtype=int)
        self._kp = np.array([a.kp for a in acts])
        self._kv = np.array([a.kv for a in acts])
        self._ctrl_lo = np.array([a.ctrl_range[0] for a in acts])
        self._ctrl_hi = np.array([a.ctrl_range[1] for a in acts])
        self._frc_lo = np.array([a.force_range[0] for a in acts])
        self._frc_hi = np.array([a.force_range[1] for a in acts])
        self._force_limited = bool(np.isfinite(self._frc_lo).any() or np.isfinite(self._frc_hi).any())

        # -- collision -------------------------------------------------------
        if cfg.use_convex_hulls:
            _convexifyMeshGeoms(coll)
        self._pin_to_mj_geom = self._matchGeoms()
        mj_to_pin = {g: k for k, g in enumerate(self._pin_to_mj_geom) if g is not None}
        coll.removeAllCollisionPairs()
        mus = []
        for g1, g2 in sorted(info.allowed_pairs):
            a, b = mj_to_pin[g1], mj_to_pin[g2]
            coll.addCollisionPair(pin.CollisionPair(min(a, b), max(a, b)))
        for cp in coll.collisionPairs:
            mus.append(info.pairFriction(self._pin_to_mj_geom[cp.first],
                                         self._pin_to_mj_geom[cp.second]))
        self._pair_friction = np.array(mus)

        self.geometry_data = pin.GeometryData(coll)
        # Without enable_contact coal returns no normals or witness points.
        for req in self.geometry_data.collisionRequests:
            req.enable_contact = True
            req.num_max_contacts = cfg.max_contacts_per_pair

        n_pair = len(coll.collisionPairs)
        self._all_pairs = list(range(n_pair))
        self._broadphase = None
        if n_pair >= _BROADPHASE_MIN_PAIRS:
            # coal's dynamic AABB tree returns only the pairs whose boxes
            # overlap. The collect-only callback does no narrowphase, so we know
            # exactly which collision results are fresh this step.
            self._broadphase = pin.BroadPhaseManager_DynamicAABBTreeCollisionManager(
                model, coll, self.geometry_data)
            self._broadphase_cb = pin.CollisionCallBackCollect(coll, self.geometry_data)

        # -- joint limits and dry friction -----------------------------------
        # Both are inert in aba, so they are constraints in the same ADMM solve
        # as the contacts.
        limited = [(j, jid) for j, jid in self._joints if j.limited and j.type != "free"]
        self._limit_raw = None
        self._limit_margin = np.zeros(model.nq)
        if limited:
            # The MJCF's ranges go into the model first: the constraint reads
            # its bounds from the model when it is built.
            for j, jid in limited:
                iq = model.joints[jid].idx_q
                model.lowerPositionLimit[iq], model.upperPositionLimit[iq] = j.range
                self._limit_margin[iq] = j.margin
            jv = pin.StdVec_Index()
            for _j, jid in limited:
                jv.append(jid)
            self._limit_raw = pin.JointLimitConstraintModel(model, jv)
            self._limit_lo = np.array(model.lowerPositionLimit, dtype=float)
            self._limit_hi = np.array(model.upperPositionLimit, dtype=float)
            if self._limit_margin.any():
                self._limit_raw.setPositionLimitAndMargin(
                    self._limit_lo, self._limit_hi, self._limit_margin)
        # (name, idx_q, idx_v, impulse bound). The solver works in impulses, so
        # the frictionloss torque bound is scaled by dt.
        self._friction_spec = []
        for j, jid in self._joints:
            if j.type == "free":
                if j.frictionloss.any():
                    raise ValueError(f"frictionloss on free joint {j.name!r} is not supported")
                continue
            if j.frictionloss[0] > 0:
                jt = model.joints[jid]
                self._friction_spec.append(
                    (j.name, jid, jt.idx_q, jt.idx_v, float(j.frictionloss[0]) * cfg.timestep))
        self._friction_cache = {}

        # -- solver ----------------------------------------------------------
        self._baumgarte = pin.BaumgarteCorrectorParameters(cfg.baumgarte_kp, cfg.baumgarte_kd)
        self._solver = pin.ADMMConstraintSolver()
        s = pin.ADMMSolverSettings()
        s.max_iterations = cfg.admm_max_iterations
        s.absolute_feasibility_tol = cfg.absolute_tolerance
        s.absolute_complementarity_tol = cfg.absolute_tolerance
        s.relative_feasibility_tol = cfg.relative_tolerance
        s.relative_complementarity_tol = cfg.relative_tolerance
        s.admm_update_rule = {"spectral": pin.ADMMUpdateRule.SPECTRAL,
                              "linear": pin.ADMMUpdateRule.LINEAR}[cfg.admm_update_rule]
        s.anderson_capacity = cfg.anderson_capacity
        s.admm_proximal_rule = pin.ADMMProximalRule.AUTOMATIC
        s.stat_record = False
        s.solve_ncp = cfg.solve_ncp
        self._settings = s
        self._result = pin.ADMMSolverResult()
        self._fext = [pin.Force.Zero() for _ in range(model.njoints)]

        # -- state -----------------------------------------------------------
        self._q = pin.neutral(model)
        self._v = np.zeros(model.nv)
        self._u = np.zeros(info.nu)
        self._time = 0.0
        self._diag = {"n_steps": 0, "n_contact_steps": 0, "max_n_contacts": 0,
                      "min_penetration_m": 0.0, "n_nonconverged": 0}
        pin.forwardKinematics(model, self.data, self._q)

    # -- setup helpers -------------------------------------------------------
    def _matchGeoms(self) -> list[int | None]:
        """Map each Pinocchio collision geom to its MuJoCo geom id.

        By name when the MJCF names the geom, which is the normal case.
        Otherwise by position among the colliding geoms of the same body.
        Every MuJoCo geom that can collide must be found, or its contacts would
        silently go missing.
        """
        info, model = self.info, self.model
        by_name = {g.name: g.id for g in info.geoms if g.name}
        out: list[int | None] = []
        seen_per_body: dict[str, int] = {}
        for go in self.collision_model.geometryObjects:
            if go.name in by_name:
                out.append(by_name[go.name])
                continue
            body = _canonicalBodyName(model.frames[go.parentFrame].name)
            k = seen_per_body.get(body, 0)
            seen_per_body[body] = k + 1
            cands = info.geomsOfBody(body)
            out.append(cands[k].id if k < len(cands) else None)
        missing = [g.name or f"#{g.id}" for g in info.geoms
                   if g.collides and g.id not in out]
        if missing:
            raise ValueError(f"colliding MJCF geoms missing from the Pinocchio model: {missing}")
        return out

    # -- state ---------------------------------------------------------------
    def SetState(self, q: np.ndarray, q_dot: np.ndarray | None = None) -> None:
        """Overwrite the state, given in MuJoCo's ``qpos``/``qvel`` layout."""
        pin, model = self._pin, self.model
        qpos = self._check(q, self.nq, "q")
        qvel = np.zeros(self.nv) if q_dot is None else self._check(q_dot, self.nv, "q_dot")
        for j, jid in self._joints:
            jt = model.joints[jid]
            a, va = j.qposadr, j.dofadr
            if j.type != "free":
                self._q[jt.idx_q] = qpos[a]
                self._v[jt.idx_v] = qvel[va]
                continue
            quat = qpos[a + 3:a + 7]
            n = np.linalg.norm(quat)
            w, x, y, z = quat / n if n > 0 else (1.0, 0.0, 0.0, 0.0)
            R = pin.Quaternion(w, x, y, z).matrix()
            # The parser bakes the body offset into the joint placement, so the
            # joint's q is jointPlacement^-1 * (world pose).
            M_q = model.jointPlacements[jid].inverse() * pin.SE3(R, qpos[a:a + 3].copy())
            qq = pin.Quaternion(M_q.rotation)
            self._q[jt.idx_q:jt.idx_q + 3] = M_q.translation
            self._q[jt.idx_q + 3:jt.idx_q + 7] = [qq.x, qq.y, qq.z, qq.w]
            # MuJoCo: linear in world, angular in body. Pinocchio: both in body.
            self._v[jt.idx_v:jt.idx_v + 3] = R.T @ qvel[va:va + 3]
            self._v[jt.idx_v + 3:jt.idx_v + 6] = qvel[va + 3:va + 6]
        pin.forwardKinematics(model, self.data, self._q)

    def GetState(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(qpos, qvel)`` in MuJoCo's layout."""
        pin, model = self._pin, self.model
        pin.forwardKinematics(model, self.data, self._q)
        qpos = np.zeros(self.nq)
        qvel = np.zeros(self.nv)
        for j, jid in self._joints:
            jt = model.joints[jid]
            a, va = j.qposadr, j.dofadr
            if j.type != "free":
                qpos[a] = self._q[jt.idx_q]
                qvel[va] = self._v[jt.idx_v]
                continue
            X = self.data.oMi[jid]
            R = X.rotation
            quat = pin.Quaternion(R)
            qpos[a:a + 3] = X.translation
            qpos[a + 3:a + 7] = [quat.w, quat.x, quat.y, quat.z]
            qvel[va:va + 3] = R @ self._v[jt.idx_v:jt.idx_v + 3]
            qvel[va + 3:va + 6] = self._v[jt.idx_v + 3:jt.idx_v + 6]
        return qpos, qvel

    # -- control -------------------------------------------------------------
    def SetControl(self, u: np.ndarray) -> None:
        """Set the servo targets, in MuJoCo actuator order.

        Clamped to each actuator's ``ctrlrange`` when stepping, as MuJoCo
        does; ``GetControl`` returns the value as given.
        """
        self._u = self._check(u, self.nu, "u").copy()

    def GetControl(self) -> np.ndarray:
        return self._u.copy()

    # -- stepping ------------------------------------------------------------
    def Step(self, steps: int = 1) -> None:
        """Advance by ``steps`` timesteps, holding the control fixed."""
        for _ in range(steps):
            self._step()
            self._time += self.config.timestep

    def _step(self) -> None:
        pin = self._pin
        model, data = self.model, self.data
        q, v = self._q, self._v
        dt = self.config.timestep
        self._diag["n_steps"] += 1

        cms, cds = self._detectContacts()
        # Contacts are the leading rows; the Baumgarte pass below relies on it.
        n_contact = len(cms)
        jcms, jcds = self._jointConstraints(q)
        cms, cds = cms + jcms, cds + jcds

        # Servos, then passive damping outside the force clamp (see module doc).
        u = np.clip(self._u, self._ctrl_lo, self._ctrl_hi)
        act = self._kp * (u - q[self._ctrl_q]) - self._kv * v[self._ctrl_v]
        if self._force_limited:
            act = np.clip(act, self._frc_lo, self._frc_hi)
        tau = -self._damping * v
        tau[self._ctrl_v] += act

        # The constraint Cholesky below factors the joint-space inertia crba
        # leaves in data; without this call the solve returns NaN.
        pin.crba(model, data, q, pin.Convention.WORLD)
        v_free = v + dt * pin.aba(model, data, q, v, tau, self._fext)
        if not cms:
            self._q = pin.integrate(model, q, v_free * dt)
            self._v = v_free
            return

        for cm, cd in zip(cms, cds):
            cm.calc(model, data, cd)
        chol = pin.ConstraintCholeskyDecomposition(model, data, cms, cds)
        chol.compute(model, data, cms, cds, self.config.delassus_regularization)
        delassus = chol.getDelassusOperatorCholeskyExpression()
        # reshape: with a single constraint row the Jacobian comes back 1-D.
        Jc = np.asarray(pin.getConstraintsJacobian(model, data, cms, cds)).reshape(-1, model.nv)
        g = Jc @ v_free

        # Baumgarte: push each contact's drift by kp * position_error / dt. Only
        # the contact rows carry this state; the joint rows appended after them
        # do not, and need none.
        idx = 0
        for cm, cd in zip(cms[:n_contact], cds[:n_contact]):
            size = cm.residualSize()
            kp_b = cm.baumgarte_corrector_parameters.Kp
            if kp_b != 0.0:
                g[idx:idx + size] += kp_b * cd.extract().constraint_position_error / dt
            idx += size

        if not self._solver.solve(delassus, g, cms, cds, self._settings, self._result):
            self._diag["n_nonconverged"] += 1
        forces = np.asarray(self._result.retrieveConstraintImpulses()).ravel() / dt
        v_new = v + dt * pin.aba(model, data, q, v, tau + Jc.T @ forces, self._fext)
        self._q = pin.integrate(model, q, v_new * dt)
        self._v = v_new

    def _detectContacts(self):
        """One point-contact constraint per coal contact point at the current q.

        The contact frame's z axis is the contact normal. Returns
        ``(constraint_models, constraint_datas)``.
        """
        pin = self._pin
        model, data = self.model, self.data
        gm, gd = self.collision_model, self.geometry_data
        q = self._q
        if self._broadphase is not None:
            # Runs kinematics and geometry placement too. Sorted so rows come
            # out in pair order; the ADMM solve is sensitive to row order.
            pin.computeCollisions(model, data, self._broadphase, self._broadphase_cb, q)
            hits = [k for k in sorted(self._broadphase_cb.pair_indexes)
                    if pin.computeCollision(gm, gd, k)]
        else:
            pin.forwardKinematics(model, data, q)
            pin.updateGeometryPlacements(model, data, gm, gd, q)
            pin.computeCollisions(gm, gd, False)
            hits = [k for k in self._all_pairs if gd.collisionResults[k].isCollision()]

        # Per contact, every step: kept scalar, since numpy on 3-vectors is
        # mostly dispatch overhead at this call rate.
        oMi = data.oMi
        geoms, pairs, results = gm.geometryObjects, gm.collisionPairs, gd.collisionResults
        inf = math.inf
        cms, penetrations = [], []
        for k in hits:
            cr, cp = results[k], pairs[k]
            j1 = geoms[cp.first].parentJoint
            j2 = geoms[cp.second].parentJoint
            # Finite witness points, else the geoms' midpoint (box-box manifolds
            # can return NaN). The normal is contact 0's, else centre to centre.
            world_points, normal = [], None
            for i in range(cr.numContacts()):
                c = cr.getContact(i)
                if i == 0:
                    nx, ny, nz = c.normal.tolist()
                    d = nx * nx + ny * ny + nz * nz
                    if d == d and d < inf and d > 1e-18:
                        normal = c.normal
                px, py, pz = c.pos.tolist()
                if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
                    continue
                world_points.append(c.pos)
                if math.isfinite(c.penetration_depth):
                    penetrations.append(c.penetration_depth)
            if normal is None or not world_points:
                c1 = gd.oMg[cp.first].translation
                c2 = gd.oMg[cp.second].translation
                if normal is None:
                    normal = c2 - c1
                if not world_points:
                    world_points.append(0.5 * (c1 + c2))
            R_n = _rotationFromNormal(normal)
            mu = float(self._pair_friction[k])
            # inverse() * M, not actInv(M): they round differently, and this
            # form matches the old simulator bit for bit.
            inv1, inv2 = oMi[j1].inverse(), oMi[j2].inverse()
            for p_world in world_points:
                M_world = pin.SE3(R_n, p_world)
                cm = pin.PointContactConstraintModel(model, j1, inv1 * M_world, j2, inv2 * M_world)
                cm.setFriction(mu)
                cm.setBaumgarteCorrectorParameters(self._baumgarte)
                cms.append(pin.ConstraintModel(cm))
        cds = [cm.createData() for cm in cms]

        # coal's depth is signed: negative is overlap, so the worst is the min.
        if penetrations:
            dg = self._diag
            dg["n_contact_steps"] += 1
            dg["max_n_contacts"] = max(dg["max_n_contacts"], len(penetrations))
            dg["min_penetration_m"] = min(dg["min_penetration_m"], min(penetrations))
        return cms, cds

    def _jointConstraints(self, q):
        """Joint-limit and dry-friction constraints at q, to go after the contacts.

        The limit constraint is re-selected every step, because only joints
        near a bound engage, and re-wrapped, because ``pin.ConstraintModel``
        copies the model and would freeze a stale active set.
        """
        pin = self._pin
        cms, cds = [], []
        at_limit = set()
        if self._limit_raw is not None:
            self._limit_raw.makeSelectionFilteredByLimitProximity(q)
            if self._limit_raw.residualSize() > 0:
                cm = pin.ConstraintModel(self._limit_raw)
                cms.append(cm)
                cds.append(cm.createData())
                at_limit = self._jointsAtLimit(q)
        if self._friction_spec:
            fcm = self._frictionModel(at_limit)
            if fcm is not None:
                cms.append(fcm)
                cds.append(fcm.createData())
        return cms, cds

    def _jointsAtLimit(self, q) -> set[str]:
        """Friction joints at or past a limit, by the limit constraint's own rule."""
        at = set()
        for name, _jid, iq, _iv, _b in self._friction_spec:
            m = self._limit_margin[iq]
            if q[iq] >= self._limit_hi[iq] - m or q[iq] <= self._limit_lo[iq] + m:
                at.add(name)
        return at

    def _frictionModel(self, exclude):
        """Dry-friction constraint over every friction joint not in ``exclude``.

        A joint at an active limit must be left out: its limit row and friction
        row are linearly dependent, which makes the Delassus operator singular
        and the solve return NaN. Cached per exclusion set.
        """
        key = frozenset(exclude)
        if key in self._friction_cache:
            return self._friction_cache[key]
        pin, model = self._pin, self.model
        jv = pin.StdVec_Index()
        lb, ub = np.zeros(model.nv), np.zeros(model.nv)
        for name, jid, _iq, iv, bound in self._friction_spec:
            if name in key:
                continue
            jv.append(jid)
            lb[iv], ub[iv] = -bound, bound
        cm = (pin.ConstraintModel(pin.JointFrictionConstraintModel(model, jv, lb, ub))
              if len(jv) else None)
        self._friction_cache[key] = cm
        return cm

    # -- dimensions ----------------------------------------------------------
    @property
    def nq(self) -> int:
        return self.info.nq

    @property
    def nv(self) -> int:
        return self.info.nv

    @property
    def nu(self) -> int:
        return self.info.nu

    @property
    def time(self) -> float:
        """Simulated time in seconds, advanced by ``Step``; ``SetState`` keeps it."""
        return self._time

    def Diagnostics(self) -> dict:
        """Running contact and solver counters since construction.

        ``min_penetration_m`` is the deepest signed penetration seen (negative
        is overlap). ``n_nonconverged`` counts steps where ADMM hit its
        iteration cap.
        """
        return dict(self._diag)

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _check(x, dim: int, name: str) -> np.ndarray:
        a = np.asarray(x, dtype=float).ravel()
        if a.shape != (dim,):
            raise ValueError(f"{name} must have shape ({dim},), got {np.shape(x)}")
        return a


# ---------------------------------------------------------------------------
# Model building. Pinocchio's MJCF parser handles one root body and does not
# follow <include>, so the scene is flattened, split per root body, parsed
# piecewise and merged back.
# ---------------------------------------------------------------------------

# Wrapper body that carries the loose <worldbody> geoms. It stands for MuJoCo's
# "world" body; see _canonicalBodyName.
_WORLD_GEOMS_BODY = "world_geoms"

# Sections MuJoCo merges, rather than repeats, across included files.
_MJCF_MERGE_SECTIONS = frozenset({
    "asset", "worldbody", "contact", "actuator", "sensor", "equality", "tendon", "keyframe",
})


def _canonicalBodyName(name: str) -> str:
    """Pinocchio frame name -> MuJoCo body name. MuJoCo's root is ``world``."""
    return "world" if name in ("universe", _WORLD_GEOMS_BODY) else name


def _quatMul(a, b):
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _eulerToQuatWxyz(angles, seq="xyz"):
    """MuJoCo's default intrinsic, radian Euler angles -> wxyz quaternion."""
    axes = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for ch, ang in zip(seq, angles):
        ax, ay, az = axes[ch]
        s = np.sin(0.5 * ang)
        q = _quatMul(q, np.array([np.cos(0.5 * ang), ax * s, ay * s, az * s]))
    return q


def _rootBodyPose(body):
    """The world pose to reapply to a root body when merging, or identity.

    Pinocchio's parser drops the placement of a root body that has no joint
    (welded to the world); that pose is read back here. A root body with a
    joint keeps its placement in the joint, so reapplying it would count it
    twice.
    """
    if body.find("freejoint") is not None or body.findall("joint"):
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ", dtype=float)
    if pos.shape[0] != 3:
        pos = np.zeros(3)
    if body.get("quat") is not None:
        quat = np.fromstring(body.get("quat"), sep=" ", dtype=float)
        quat = quat / (np.linalg.norm(quat) or 1.0)
    elif body.get("euler") is not None:
        quat = _eulerToQuatWxyz(np.fromstring(body.get("euler"), sep=" ", dtype=float))
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    return pos, quat


def _loadMjcfRootWithIncludes(mjcf_path):
    """Parse an MJCF and splice every ``<include>`` inline, recursively.

    Container sections (asset, worldbody, contact, ...) are merged the way
    MuJoCo merges them; other sections are appended as siblings. Include paths
    resolve relative to the file that contains them.
    """
    root = ET.parse(mjcf_path).getroot()
    merged = ET.Element(root.tag, dict(root.attrib))
    sections = {}

    def _append(child):
        if child.tag in _MJCF_MERGE_SECTIONS:
            existing = sections.get(child.tag)
            if existing is None:
                sections[child.tag] = child
                merged.append(child)
            else:
                for sub in list(child):
                    existing.append(sub)
                for k, v in child.attrib.items():
                    existing.attrib.setdefault(k, v)
        else:
            merged.append(child)

    def _splice(node, node_dir):
        for child in list(node):
            if child.tag == "include":
                inc = os.path.join(node_dir, child.get("file"))
                _splice(_loadMjcfRootWithIncludes(inc), os.path.dirname(os.path.abspath(inc)))
            else:
                _append(child)

    _splice(root, os.path.dirname(os.path.abspath(mjcf_path)))
    return merged


def _splitIntoSingleRootMjcfs(mjcf_path, scene_dir):
    """Write one temporary MJCF per root body, plus one for loose world geoms.

    Returns ``(temp_paths, root_poses)``; ``root_poses[i]`` is the world pose
    ``_mergeModels`` reapplies to part ``i``. The files go next to the scene so
    relative mesh paths still resolve, with a per-process token so concurrent
    runs do not collide.
    """
    root = _loadMjcfRootWithIncludes(mjcf_path)
    compiler = root.find("compiler")
    asset = root.find("asset")
    # Defaults travel with every part: a class reference to an undefined class
    # makes Pinocchio's parser throw.
    defaults = root.findall("default")
    # Pinocchio's parser also throws on a <material> with a class attribute.
    # Materials are visual only, so the attribute is dropped.
    if asset is not None:
        for mat in asset.findall("material"):
            mat.attrib.pop("class", None)
    worldbody = root.find("worldbody")
    loose_geoms = [el for el in worldbody if el.tag == "geom"]
    bodies = [el for el in worldbody if el.tag == "body"]
    token = f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}_{uuid.uuid4().hex[:8]}"

    def _write(children, label):
        new_root = ET.Element("mujoco", root.attrib)
        if compiler is not None:
            new_root.append(compiler)
        if asset is not None:
            new_root.append(asset)
        for d in defaults:
            new_root.append(d)
        wb = ET.SubElement(new_root, "worldbody")
        for c in children:
            wb.append(c)
        path = os.path.join(scene_dir, f"_tmp_pin_split_{token}_{label}.xml")
        ET.ElementTree(new_root).write(path)
        return path

    paths, poses = [], []
    for i, body in enumerate(bodies):
        paths.append(_write([body], str(i)))
        poses.append(_rootBodyPose(body))
    # Loose world geoms get their own identity-placed wrapper body, so no
    # root body's pose is applied to them.
    if loose_geoms:
        wrapper = ET.Element("body", {"name": _WORLD_GEOMS_BODY, "pos": "0 0 0"})
        for g in loose_geoms:
            wrapper.append(g)
        paths.append(_write([wrapper], "worldgeoms"))
        poses.append((np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
    return paths, poses


def _mergeModels(pin, parts, root_poses):
    """Append the per-root ``(model, collision, visual)`` parts into one model.

    Starts from an empty model so the first part's root pose is applied too.
    ``appendModel`` merges one geometry model per call, so it runs twice per
    part against the same pre-merge model, keeping collision and visual
    indices consistent.
    """
    model, coll, vis = pin.Model(), pin.GeometryModel(), pin.GeometryModel()
    for (mB, collB, visB), (pos, quat) in zip(parts, root_poses):
        w, x, y, z = quat
        aMb = pin.SE3(pin.Quaternion(w, x, y, z).matrix(), np.asarray(pos, dtype=float))
        prev = model
        model, coll = pin.appendModel(prev, mB, coll, collB, 0, aMb)
        _, vis = pin.appendModel(prev, mB, vis, visB, 0, aMb)
    return model, coll, vis


def _convexifyMeshGeoms(geom_model) -> int:
    """Replace every triangle-mesh collision geom by its convex hull, in place."""
    try:
        import coal
    except ImportError:
        import hppfcl as coal
    n = 0
    for go in geom_model.geometryObjects:
        if not type(go.geometry).__name__.startswith("BVHModel"):
            continue
        verts = np.asarray(go.geometry.vertices(), dtype=float)
        if verts.ndim != 2 or verts.shape[0] < 4:
            continue
        pts = coal.StdVec_Vec3s()
        for row in verts:
            pts.append(row)
        # keepTriangles=True needs qhull's "Qt" (triangulated output).
        go.geometry = coal.Convex.convexHull(pts, True, "Qt")
        n += 1
    return n


def _rotationFromNormal(n):
    """Rotation whose z column is the unit normal ``n``.

    Scalar ``math`` rather than numpy: it runs once per contact per step, and
    numpy's per-call overhead dominates on 3-vectors.
    """
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
    if not (nn >= 1e-9) or math.isinf(nn):
        return np.eye(3)
    zx, zy, zz = nx / nn, ny / nn, nz / nn
    # Tangent x: the world axis least aligned with z, projected onto the plane.
    if abs(zx) < 0.9:
        xx, xy, xz = 1.0 - zx * zx, -zy * zx, -zz * zx
    else:
        xx, xy, xz = -zx * zy, 1.0 - zy * zy, -zz * zy
    xn = math.sqrt(xx * xx + xy * xy + xz * xz)
    xx, xy, xz = xx / xn, xy / xn, xz / xn
    return np.array([
        [xx, zy * xz - zz * xy, zx],
        [xy, zz * xx - zx * xz, zy],
        [xz, zx * xy - zy * xx, zz],
    ])
