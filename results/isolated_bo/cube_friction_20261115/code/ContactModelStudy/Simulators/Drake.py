"""Drake's MultibodyPlant, wrapped as a ``Simulator``.

A discrete ``MultibodyPlant`` parsed from the task's MJCF (the same file MuJoCo
loads), stepped by a Drake ``Simulator``. Ported from
``contact_study/contact_models/drake_sim.py``, which loaded a separate URDF and
needed a hand-kept joint-name map and PID gains. This version needs neither.

**Where parameters come from.** Drake's own MJCF parser builds the bodies,
joints and geometry. Everything physical is then checked against, and
overwritten from, ``MjcfModelInfo`` (MuJoCo's compiler), because Drake's parser
drops a position servo's ``kv`` and takes actuator effort limits from
``ctrlrange``:

* joint damping, and armature (as actuator rotor inertia);
* joint limits (only where ``limited``);
* the position servos, as Drake's built-in PD actuators: ``kp``, ``kd = kv``,
  effort limit from the MJCF force range, and ``ctrlrange`` on the command;
* which geom pairs can collide, set with collision filters so Drake's
  candidate pairs are exactly MuJoCo's.

``DrakeConfig`` holds only what the MJCF cannot say: the contact model and
discrete solver settings.

**Known differences from MuJoCo**, all Drake's own physics rather than
porting choices:

* Friction between two geoms combines as ``2 μa μb / (μa + μb)``, not MuJoCo's
  ``max(μa, μb)``.
* Drake has no dry joint friction or joint-limit margin. A nonzero
  ``frictionloss`` or ``margin`` in the MJCF raises.
* The PD servo is implicit in the SAP solver, so it cannot be combined with
  the TAMSI approximation.

**State layout.** ``GetState``/``SetState`` use MuJoCo's ``qpos``/``qvel``
layout. Free bodies go through Drake's pose and spatial-velocity APIs, so the
quaternion order (Drake stores ``[quat, pos]``) is never touched directly, and
angular velocity is converted to MuJoCo's body frame.

pydrake is imported lazily, so importing this module does not need it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ContactModelStudy.Simulators.Simulator import Simulator, SimulatorConfig
from ContactModelStudy.Utils.MjcfModelInfo import MjcfModelInfo

_CONTACT_MODELS = ("point", "hydroelastic", "hydroelastic_with_fallback")
_APPROXIMATIONS = ("sap", "similar", "lagged")


@dataclass
class DrakeConfig(SimulatorConfig):
    """Physics parameters for the Drake simulator.

    ``timestep``, ``substeps`` and ``gravity`` from ``SimulatorConfig``; the
    plant's discrete step is ``timestep``. The rest are contact-solver
    settings, which the MJCF has no way to express. Every one defaults to
    ``None``, meaning "Drake's default".

    Attributes:
        contact_model: ``"point"``, ``"hydroelastic"`` or
            ``"hydroelastic_with_fallback"``. The MJCF gives no geometry
            hydroelastic properties, so both hydroelastic options end up using
            point contact for this study's scenes.
        discrete_approximation: SAP-family contact approximation: ``"sap"``,
            ``"similar"`` or ``"lagged"``. TAMSI is not offered, because it
            cannot run the implicit PD servos.
        penetration_allowance: Penetration (m) the point-contact stiffness is
            tuned to allow under the model's weight. Smaller is stiffer.
        stiction_tolerance: Slip speed (m/s) below which friction is treated
            as stiction.
        sap_near_rigid_threshold: SAP's threshold for treating a contact as
            near-rigid, which bounds its effective stiffness for stability.
    """

    contact_model: Optional[str] = None
    discrete_approximation: Optional[str] = None
    penetration_allowance: Optional[float] = None
    stiction_tolerance: Optional[float] = None
    sap_near_rigid_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()

        def bad(name, why):
            raise ValueError(f"{name} {why}, got {getattr(self, name)!r}")

        if self.contact_model is not None and self.contact_model not in _CONTACT_MODELS:
            bad("contact_model", f"must be one of {_CONTACT_MODELS}")
        if (self.discrete_approximation is not None
                and self.discrete_approximation not in _APPROXIMATIONS):
            bad("discrete_approximation",
                f"must be one of {_APPROXIMATIONS} (TAMSI cannot run PD servos)")
        if self.penetration_allowance is not None and self.penetration_allowance <= 0:
            bad("penetration_allowance", "must be positive")
        if self.stiction_tolerance is not None and self.stiction_tolerance <= 0:
            bad("stiction_tolerance", "must be positive")
        if self.sap_near_rigid_threshold is not None and self.sap_near_rigid_threshold < 0:
            bad("sap_near_rigid_threshold", "must be >= 0")


class Drake(Simulator):
    """One Drake world, stepped on the CPU.

    Attributes:
        info: The ``MjcfModelInfo`` every physical parameter was taken from.
        plant, scene_graph, diagram: The Drake systems.
    """

    def __init__(self, xml: str | Path, sim_config: DrakeConfig | None = None):
        """Parse the MJCF into a discrete plant and reconcile it with MuJoCo.

        Args:
            xml: Path to the MJCF. Must be a path, for Drake's parser.
            sim_config: Solver parameters. Defaults to ``DrakeConfig()``.

        Raises:
            ValueError: If ``xml`` is inline XML, or the MJCF uses something
                Drake cannot reproduce (dry joint friction, limit margins,
                asymmetric force limits, armature on an unactuated joint).
        """
        super().__init__(xml, sim_config if sim_config is not None else DrakeConfig())
        if self.model_path is None:
            raise ValueError("Drake needs the MJCF as a file path, not inline XML.")
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import (
            AddMultibodyPlantSceneGraph, ContactModel, DiscreteContactApproximation,
        )
        from pydrake.systems.analysis import Simulator as DrakeSimulator
        from pydrake.systems.framework import DiagramBuilder
        from pydrake.common.eigen_geometry import Quaternion
        from pydrake.math import RigidTransform, RotationMatrix
        from pydrake.multibody.math import SpatialVelocity

        self._Quaternion, self._RigidTransform = Quaternion, RigidTransform
        self._RotationMatrix, self._SpatialVelocity = RotationMatrix, SpatialVelocity
        cfg = self.config
        self.info = info = MjcfModelInfo.fromXml(self.model_path)
        for j in info.joints:
            if j.type == "ball":
                raise ValueError(f"ball joint {j.name!r} is not supported")
            if j.frictionloss.any():
                raise ValueError(f"joint {j.name!r} has frictionloss; Drake has no dry joint friction")
            if j.margin:
                raise ValueError(f"joint {j.name!r} has a limit margin; Drake has none")

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, cfg.timestep)
        if cfg.contact_model is not None:
            plant.set_contact_model({
                "point": ContactModel.kPoint,
                "hydroelastic": ContactModel.kHydroelastic,
                "hydroelastic_with_fallback": ContactModel.kHydroelasticWithFallback,
            }[cfg.contact_model])
        if cfg.discrete_approximation is not None:
            plant.set_discrete_contact_approximation({
                "sap": DiscreteContactApproximation.kSap,
                "similar": DiscreteContactApproximation.kSimilar,
                "lagged": DiscreteContactApproximation.kLagged,
            }[cfg.discrete_approximation])
        if cfg.penetration_allowance is not None:
            plant.set_penetration_allowance(cfg.penetration_allowance)
        if cfg.stiction_tolerance is not None:
            plant.set_stiction_tolerance(cfg.stiction_tolerance)
        if cfg.sap_near_rigid_threshold is not None:
            plant.set_sap_near_rigid_threshold(cfg.sap_near_rigid_threshold)

        # Drake warns about every MJCF element it skips (sites, lights, cameras,
        # textures, sensors) and about the ctrlrange-derived effort limits that
        # are overwritten below. None of it matters here, so the parse is
        # quietened.
        drake_log = logging.getLogger("drake")
        level = drake_log.level
        drake_log.setLevel(logging.ERROR)
        try:
            Parser(plant).AddModels(str(self.model_path))
        finally:
            drake_log.setLevel(level)

        self._reconcileJoints(plant)
        self._reconcileActuators(plant)
        plant.mutable_gravity_field().set_gravity_vector(np.array(cfg.gravity))
        plant.Finalize()
        self._reconcileCollisionFilters(plant, scene_graph)

        self.plant, self.scene_graph = plant, scene_graph
        self.diagram = builder.Build()
        self._simulator = DrakeSimulator(self.diagram)
        self._simulator.Initialize()
        self._context = self._simulator.get_mutable_context()
        self._plant_context = plant.GetMyMutableContextFromRoot(self._context)

        # Desired-state ports, one per model instance with servos. Each takes
        # [q_d; v_d] in that instance's actuator order.
        self._desired = []   # (port, MuJoCo actuator indices in Drake order)
        by_instance: dict = {}
        for k, a in enumerate(info.actuators):
            act = plant.GetJointActuatorByName(a.name)
            by_instance.setdefault(act.model_instance(), []).append((int(act.index()), k))
        for inst, lst in by_instance.items():
            order = [k for _i, k in sorted(lst)]
            self._desired.append((plant.get_desired_state_input_port(inst), np.array(order)))
        self._ctrl_lo = np.array([a.ctrl_range[0] for a in info.actuators])
        self._ctrl_hi = np.array([a.ctrl_range[1] for a in info.actuators])

        self._joints = []   # (JointInfo, drake joint or free body)
        for j in info.joints:
            if j.type == "free":
                self._joints.append((j, plant.GetBodyByName(j.body)))
            else:
                self._joints.append((j, plant.GetJointByName(j.name)))

        self._u = np.zeros(info.nu)
        self._n_steps = 0
        self._t0 = 0.0
        self.SetControl(self._u)

    # -- reconciliation with the MJCF ---------------------------------------
    def _reconcileJoints(self, plant) -> None:
        """Damping and limits from the MJCF, on every non-free joint."""
        for j in self.info.joints:
            if j.type == "free":
                if j.damping.any() or j.armature.any():
                    raise ValueError(f"damping/armature on free joint {j.name!r} is not supported")
                continue
            joint = plant.GetJointByName(j.name)
            joint.set_default_damping_vector(j.damping)
            lo, hi = j.range if j.limited else (-np.inf, np.inf)
            joint.set_position_limits([lo], [hi])

    def _reconcileActuators(self, plant) -> None:
        """PD gains, effort limits and rotor inertia from the MJCF servos."""
        from pydrake.multibody.tree import PdControllerGains

        actuated = set()
        for a in self.info.actuators:
            act = plant.GetJointActuatorByName(a.name)
            if act.joint().name() != a.joint:
                raise ValueError(f"actuator {a.name!r} drives {act.joint().name()!r} in Drake, "
                                 f"{a.joint!r} in the MJCF")
            act.set_controller_gains(PdControllerGains(p=a.kp, d=a.kv))
            lo, hi = a.force_range
            if np.isfinite(lo) or np.isfinite(hi):
                if not np.isclose(-lo, hi):
                    raise ValueError(f"actuator {a.name!r} has an asymmetric force range "
                                     f"{a.force_range}; Drake's effort limit is symmetric")
                act.set_effort_limit(float(hi))
            else:
                act.set_effort_limit(float("inf"))
            act.set_default_gear_ratio(1.0)
            act.set_default_rotor_inertia(float(self.info.joint(a.joint).armature[0]))
            actuated.add(a.joint)
        for j in self.info.joints:
            if j.type != "free" and j.name not in actuated and j.armature.any():
                raise ValueError(f"armature on unactuated joint {j.name!r}; Drake models "
                                 f"armature only as actuator rotor inertia")

    def _reconcileCollisionFilters(self, plant, scene_graph) -> None:
        """Make Drake's collision candidates exactly MuJoCo's allowed pairs."""
        from pydrake.geometry import CollisionFilterDeclaration, GeometrySet, Role

        inspector = scene_graph.model_inspector()
        ids = inspector.GetAllGeometryIds(Role.kProximity)
        self._geom_to_mj = self._matchGeoms(plant, inspector, ids)
        mj_to_geom = {g: gid for gid, g in self._geom_to_mj.items() if g is not None}

        want = {(mj_to_geom[a], mj_to_geom[b]) for a, b in self.info.allowed_pairs}
        want = {frozenset(p) for p in want}
        have = {frozenset(p) for p in inspector.GetCollisionCandidates()}
        manager = scene_graph.collision_filter_manager()
        for pair in have - want:
            a, b = tuple(pair)
            manager.Apply(CollisionFilterDeclaration().ExcludeBetween(GeometrySet(a), GeometrySet(b)))
        for pair in want - have:
            a, b = tuple(pair)
            manager.Apply(CollisionFilterDeclaration().AllowBetween(GeometrySet(a), GeometrySet(b)))
        final = {frozenset(p) for p in inspector.GetCollisionCandidates()}
        if final != want:
            raise RuntimeError(f"could not match MuJoCo's collision filter: "
                               f"{len(final - want)} extra, {len(want - final)} missing pairs")

    def _matchGeoms(self, plant, inspector, ids) -> dict:
        """Map each Drake proximity geometry to its MuJoCo geom id.

        Drake names a parsed geometry ``"<model>::<geom name>"``. Unnamed geoms
        are matched by their order among the colliding geoms of their body.
        """
        info = self.info
        by_name = {g.name: g.id for g in info.geoms if g.name}
        out, seen = {}, {}
        for gid in ids:
            name = inspector.GetName(gid).split("::")[-1]
            if name in by_name:
                out[gid] = by_name[name]
                continue
            body = plant.GetBodyFromFrameId(inspector.GetFrameId(gid)).name()
            k = seen.get(body, 0)
            seen[body] = k + 1
            cands = info.geomsOfBody(body)
            out[gid] = cands[k].id if k < len(cands) else None
        missing = [g.name or f"#{g.id}" for g in info.geoms
                   if g.collides and g.id not in out.values()]
        if missing:
            raise ValueError(f"colliding MJCF geoms missing from the Drake model: {missing}")
        return out

    # -- state ---------------------------------------------------------------
    def SetState(self, q: np.ndarray, q_dot: np.ndarray | None = None) -> None:
        """Overwrite the state, given in MuJoCo's ``qpos``/``qvel`` layout."""
        qpos = self._check(q, self.nq, "q")
        qvel = np.zeros(self.nv) if q_dot is None else self._check(q_dot, self.nv, "q_dot")
        plant, ctx = self.plant, self._plant_context
        for j, obj in self._joints:
            a, va = j.qposadr, j.dofadr
            if j.type == "hinge":
                obj.set_angle(ctx, float(qpos[a]))
                obj.set_angular_rate(ctx, float(qvel[va]))
            elif j.type == "slide":
                obj.set_translation(ctx, float(qpos[a]))
                obj.set_translation_rate(ctx, float(qvel[va]))
            else:
                quat = qpos[a + 3:a + 7]
                n = np.linalg.norm(quat)
                w, x, y, z = quat / n if n > 0 else (1.0, 0.0, 0.0, 0.0)
                R = self._RotationMatrix(self._Quaternion(w=w, x=x, y=y, z=z))
                plant.SetFreeBodyPose(ctx, obj, self._RigidTransform(R, qpos[a:a + 3].copy()))
                # MuJoCo's angular velocity is in the body frame; Drake wants world.
                w_world = R.matrix() @ qvel[va + 3:va + 6]
                plant.SetFreeBodySpatialVelocity(
                    ctx, obj, self._SpatialVelocity(w_world, qvel[va:va + 3].copy()))

    def GetState(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(qpos, qvel)`` in MuJoCo's layout."""
        plant, ctx = self.plant, self._plant_context
        qpos = np.zeros(self.nq)
        qvel = np.zeros(self.nv)
        for j, obj in self._joints:
            a, va = j.qposadr, j.dofadr
            if j.type == "hinge":
                qpos[a] = obj.get_angle(ctx)
                qvel[va] = obj.get_angular_rate(ctx)
            elif j.type == "slide":
                qpos[a] = obj.get_translation(ctx)
                qvel[va] = obj.get_translation_rate(ctx)
            else:
                X = obj.EvalPoseInWorld(ctx)
                quat = X.rotation().ToQuaternion()
                qpos[a:a + 3] = X.translation()
                qpos[a + 3:a + 7] = [quat.w(), quat.x(), quat.y(), quat.z()]
                V = obj.EvalSpatialVelocityInWorld(ctx)
                qvel[va:va + 3] = V.translational()
                qvel[va + 3:va + 6] = X.rotation().matrix().T @ V.rotational()
        return qpos, qvel

    # -- control -------------------------------------------------------------
    def SetControl(self, u: np.ndarray) -> None:
        """Set the servo targets, in MuJoCo actuator order.

        Clamped to each actuator's ``ctrlrange`` before it reaches the plant,
        as MuJoCo does. ``GetControl`` returns the value as given.
        """
        self._u = self._check(u, self.nu, "u").copy()
        q_d = np.clip(self._u, self._ctrl_lo, self._ctrl_hi)
        for port, order in self._desired:
            port.FixValue(self._plant_context, np.concatenate([q_d[order], np.zeros(len(order))]))

    def GetControl(self) -> np.ndarray:
        return self._u.copy()

    # -- stepping ------------------------------------------------------------
    def Step(self, steps: int = 1) -> None:
        """Advance by ``steps`` plant timesteps, holding the control fixed.

        The target time is computed from a step count, not accumulated, so
        rounding never makes ``AdvanceTo`` stop one discrete update short.
        """
        self._n_steps += steps
        self._simulator.AdvanceTo(self._t0 + self._n_steps * self.config.timestep)

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
        return float(self._context.get_time())

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _check(x, dim: int, name: str) -> np.ndarray:
        a = np.asarray(x, dtype=float).ravel()
        if a.shape != (dim,):
            raise ValueError(f"{name} must have shape ({dim},), got {np.shape(x)}")
        return a
