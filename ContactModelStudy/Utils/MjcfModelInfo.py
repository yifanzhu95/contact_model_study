"""Scene parameters read from an MJCF, resolved by MuJoCo's own compiler.

The Pinocchio and Drake simulators load the same MJCF as MuJoCo, but neither
parser resolves it the way MuJoCo does:

* Pinocchio ignores ``<default>`` classes (so ``contype``, ``frictionloss``
  and friction coming from a class read as zero or unset), and its
  ``appendModel`` drops armature.
* Drake parses most of the scene, but loses a position actuator's ``kv`` and
  takes actuator effort limits from ``ctrlrange``.

So both simulators compile the file with MuJoCo too, and take every parameter
the MJCF defines from here. The MJCF stays the single place those numbers are
set, and the three simulators cannot disagree about the scene.

Everything is keyed by names (joints, bodies, geoms, actuators), because that
is what survives the trip through another simulator's parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco

_JOINT_TYPES = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}


@dataclass
class JointInfo:
    """One MuJoCo joint and the parameters of its DOFs.

    Attributes:
        name: Joint name.
        type: ``"free"``, ``"ball"``, ``"slide"`` or ``"hinge"``.
        body: Name of the body the joint moves.
        qposadr: Start of this joint in MuJoCo's ``qpos``.
        dofadr: Start of this joint in MuJoCo's ``qvel``.
        limited: Whether ``range`` is enforced.
        range: ``(lo, hi)`` position limits; meaningful only if ``limited``.
        margin: Distance from a limit at which the limit becomes active.
        damping: Passive viscous damping per DOF, ``(ndof,)``.
        armature: Rotor inertia added to the mass matrix per DOF, ``(ndof,)``.
        frictionloss: Dry-friction torque/force per DOF, ``(ndof,)``.
    """

    name: str
    type: str
    body: str
    qposadr: int
    dofadr: int
    limited: bool
    range: tuple[float, float]
    margin: float
    damping: np.ndarray
    armature: np.ndarray
    frictionloss: np.ndarray

    @property
    def nq(self) -> int:
        return {"free": 7, "ball": 4}.get(self.type, 1)

    @property
    def nv(self) -> int:
        return {"free": 6, "ball": 3}.get(self.type, 1)


@dataclass
class ActuatorInfo:
    """One MuJoCo position servo, in MuJoCo control order.

    MuJoCo computes its force as ``kp * (ctrl - q) - kv * qdot``, clamped to
    ``force_range``. ``ctrl`` itself is first clamped to ``ctrl_range``.

    Attributes:
        name: Actuator name.
        joint: Name of the joint it drives.
        kp: Position gain (``gainprm[0]``).
        kv: Velocity gain (``-biasprm[2]``).
        ctrl_range: ``(lo, hi)`` the command is clamped to; ``(-inf, inf)``
            when not ``ctrllimited``.
        force_range: ``(lo, hi)`` the force is clamped to. This is the
            actuator's ``forcerange`` intersected with its joint's
            ``actuatorfrcrange``; ``(-inf, inf)`` when neither is limited.
    """

    name: str
    joint: str
    kp: float
    kv: float
    ctrl_range: tuple[float, float]
    force_range: tuple[float, float]


@dataclass
class GeomInfo:
    """One MuJoCo geom's contact parameters.

    Attributes:
        id: MuJoCo geom id.
        name: Geom name, or ``""`` if unnamed.
        body: Name of the body it belongs to.
        contype, conaffinity: Collision bitmasks.
        friction: Sliding friction coefficient (``friction[0]``).
    """

    id: int
    name: str
    body: str
    contype: int
    conaffinity: int
    friction: float

    @property
    def collides(self) -> bool:
        """Whether this geom can take part in any contact at all."""
        return bool(self.contype or self.conaffinity)


@dataclass
class MjcfModelInfo:
    """Everything a non-MuJoCo simulator needs to inherit from an MJCF.

    Build it with ``MjcfModelInfo.fromXml(path)``.

    Attributes:
        mjm: The compiled ``MjModel``, for anything not extracted below.
        nq, nv, nu: MuJoCo's dimensions, which the other simulators report.
        gravity: The MJCF's ``<option gravity>``.
        joints: Every joint, in MuJoCo order.
        actuators: Every actuator, in MuJoCo control order.
        geoms: Every geom, in MuJoCo order.
        allowed_pairs: Geom-id pairs ``(i, j)``, ``i < j``, that MuJoCo is
            allowed to put in contact. See ``_allowedPairs`` for the rule.
    """

    mjm: mujoco.MjModel
    nq: int
    nv: int
    nu: int
    gravity: np.ndarray
    joints: list[JointInfo] = field(default_factory=list)
    actuators: list[ActuatorInfo] = field(default_factory=list)
    geoms: list[GeomInfo] = field(default_factory=list)
    allowed_pairs: set[tuple[int, int]] = field(default_factory=set)

    # -- construction --------------------------------------------------------
    @classmethod
    def fromXml(cls, xml: str | Path) -> "MjcfModelInfo":
        """Compile an MJCF file with MuJoCo and extract its parameters.

        Raises:
            ValueError: If an actuator is not a joint position servo with unit
                gear. Pinocchio and Drake only reproduce that actuator type.
        """
        mjm = mujoco.MjModel.from_xml_path(str(xml))
        info = cls(
            mjm=mjm, nq=mjm.nq, nv=mjm.nv, nu=mjm.nu,
            gravity=np.array(mjm.opt.gravity, dtype=float),
        )
        info.joints = [cls._joint(mjm, j) for j in range(mjm.njnt)]
        info.actuators = [cls._actuator(mjm, a) for a in range(mjm.nu)]
        info.geoms = [cls._geom(mjm, g) for g in range(mjm.ngeom)]
        info.allowed_pairs = cls._allowedPairs(mjm)
        return info

    @staticmethod
    def _joint(mjm: mujoco.MjModel, j: int) -> JointInfo:
        jtype = _JOINT_TYPES[int(mjm.jnt_type[j])]
        dof = int(mjm.jnt_dofadr[j])
        ndof = {"free": 6, "ball": 3}.get(jtype, 1)
        return JointInfo(
            name=mjm.joint(j).name,
            type=jtype,
            body=mjm.body(int(mjm.jnt_bodyid[j])).name,
            qposadr=int(mjm.jnt_qposadr[j]),
            dofadr=dof,
            limited=bool(mjm.jnt_limited[j]),
            range=(float(mjm.jnt_range[j, 0]), float(mjm.jnt_range[j, 1])),
            margin=float(mjm.jnt_margin[j]),
            damping=np.array(mjm.dof_damping[dof:dof + ndof], dtype=float),
            armature=np.array(mjm.dof_armature[dof:dof + ndof], dtype=float),
            frictionloss=np.array(mjm.dof_frictionloss[dof:dof + ndof], dtype=float),
        )

    @staticmethod
    def _actuator(mjm: mujoco.MjModel, a: int) -> ActuatorInfo:
        name = mjm.actuator(a).name
        is_servo = (
            mjm.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
            and mjm.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
            and mjm.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
            and mjm.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE
            and np.isclose(mjm.actuator_biasprm[a, 1], -mjm.actuator_gainprm[a, 0])
        )
        if not is_servo:
            raise ValueError(
                f"actuator {name!r} is not a joint <position> servo; only those "
                f"can be reproduced outside MuJoCo."
            )
        if not np.allclose(mjm.actuator_gear[a], [1, 0, 0, 0, 0, 0]):
            raise ValueError(f"actuator {name!r} has a non-unit gear; not supported.")

        jid = int(mjm.actuator_trnid[a, 0])
        inf = float("inf")
        ctrl = (
            (float(mjm.actuator_ctrlrange[a, 0]), float(mjm.actuator_ctrlrange[a, 1]))
            if mjm.actuator_ctrllimited[a] else (-inf, inf)
        )
        lo, hi = -inf, inf
        if mjm.actuator_forcelimited[a]:
            lo, hi = float(mjm.actuator_forcerange[a, 0]), float(mjm.actuator_forcerange[a, 1])
        if mjm.jnt_actfrclimited[jid]:
            lo = max(lo, float(mjm.jnt_actfrcrange[jid, 0]))
            hi = min(hi, float(mjm.jnt_actfrcrange[jid, 1]))
        return ActuatorInfo(
            name=name,
            joint=mjm.joint(jid).name,
            kp=float(mjm.actuator_gainprm[a, 0]),
            kv=float(-mjm.actuator_biasprm[a, 2]),
            ctrl_range=ctrl,
            force_range=(lo, hi),
        )

    @staticmethod
    def _geom(mjm: mujoco.MjModel, g: int) -> GeomInfo:
        return GeomInfo(
            id=g,
            name=mjm.geom(g).name,
            body=mjm.body(int(mjm.geom_bodyid[g])).name,
            contype=int(mjm.geom_contype[g]),
            conaffinity=int(mjm.geom_conaffinity[g]),
            friction=float(mjm.geom_friction[g, 0]),
        )

    @staticmethod
    def _allowedPairs(mjm: mujoco.MjModel) -> set[tuple[int, int]]:
        """The geom pairs MuJoCo's collision filter lets through.

        MuJoCo's rule (``engine_collision_driver.c``), for geoms on bodies
        ``b1``, ``b2`` with weld roots ``w1``, ``w2``:

        1. The bitmasks must match: ``(contype1 & conaffinity2) or
           (contype2 & conaffinity1)``.
        2. Same weld group (``w1 == w2``) never collides. This covers two geoms
           on one body, and two bodies welded to each other or to the world.
        3. Parent-child weld groups never collide, unless the ``filterparent``
           disable flag is set or either group is the world.
        4. ``<exclude>`` body pairs never collide.

        Explicit ``<pair>`` entries bypass all of the above and are added.
        """
        weld = mjm.body_weldid
        parent = mjm.body_parentid
        filterparent = not (mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
        excluded = {
            frozenset((int(s >> 16), int(s & 0xFFFF))) for s in mjm.exclude_signature
        }

        cand = [g for g in range(mjm.ngeom) if mjm.geom_contype[g] or mjm.geom_conaffinity[g]]
        pairs = set()
        for i, g1 in enumerate(cand):
            b1 = int(mjm.geom_bodyid[g1])
            w1 = int(weld[b1])
            for g2 in cand[i + 1:]:
                b2 = int(mjm.geom_bodyid[g2])
                w2 = int(weld[b2])
                if not ((mjm.geom_contype[g1] & mjm.geom_conaffinity[g2])
                        or (mjm.geom_contype[g2] & mjm.geom_conaffinity[g1])):
                    continue
                if w1 == w2:
                    continue
                if filterparent and w1 != 0 and w2 != 0 and (
                    w1 == weld[parent[w2]] or w2 == weld[parent[w1]]
                ):
                    continue
                if frozenset((b1, b2)) in excluded:
                    continue
                pairs.add((g1, g2))
        for p in range(mjm.npair):
            g1, g2 = int(mjm.pair_geom1[p]), int(mjm.pair_geom2[p])
            pairs.add((min(g1, g2), max(g1, g2)))
        return pairs

    # -- lookups -------------------------------------------------------------
    def joint(self, name: str) -> JointInfo:
        """The joint with this name."""
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(f"no joint named {name!r}")

    def geomsOfBody(self, body: str, colliding_only: bool = True) -> list[GeomInfo]:
        """A body's geoms in MuJoCo order, by default only those that can collide.

        Used to match unnamed geoms across simulators by their order within a
        body, when names are not available.
        """
        return [g for g in self.geoms
                if g.body == body and (g.collides or not colliding_only)]

    def pairFriction(self, g1: int, g2: int) -> float:
        """Sliding friction of a contact between two geoms, as MuJoCo combines it.

        MuJoCo takes the larger of the two coefficients. An explicit ``<pair>``
        overrides that, with its own friction.
        """
        mjm = self.mjm
        for p in range(mjm.npair):
            if {int(mjm.pair_geom1[p]), int(mjm.pair_geom2[p])} == {g1, g2}:
                return float(mjm.pair_friction[p, 0])
        return max(self.geoms[g1].friction, self.geoms[g2].friction)
