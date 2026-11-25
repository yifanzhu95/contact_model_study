"""Use the simulation driver's task, planner and contact-model builders on hardware."""
from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ContactModelStudy.Drivers import run_episodes as reference
from ContactModelStudy.Hardware.Scene import JOINTS, joint_parameter
from ContactModelStudy.Hardware.Hand import delta_limits
from ContactModelStudy.Hardware.JointReferences import load_reference
from ContactModelStudy.Hardware.JointLimits import load_joint_limits
from ContactModelStudy.Hardware.Rollouts import CalibratedRollouts, CpuRollouts
from ContactModelStudy.Tasks.LeapReorient import FINGERTIP_SITES, LeapReorient
from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim, getContactModel
from ContactModelStudy.Utils.EvalSimulators import isGpuEvalSim, makeEvalSim


def calibrate_scene(source, calibration, output, hardware_config=None, object_friction=None,
                    fingertip_friction=None, palm_friction=None, finger_friction=None):
    """Keep the selected scene's collision geometry and transplant hand dynamics."""
    source = Path(source).expanduser().resolve()
    calibrated = mujoco.MjModel.from_xml_path(str(calibration))
    measured_limits = load_joint_limits(hardware_config or {})
    if hardware_config is not None:
        delta_limits(hardware_config)  # validate without changing absolute MJCF actuator ranges
    spec = mujoco.MjSpec.from_file(str(source))
    spec.meshdir = str((source.parent / spec.meshdir).resolve())
    spec.texturedir = str((source.parent / spec.texturedir).resolve())
    hand_bodies = {calibrated.body(i).name: i for i in range(1, calibrated.nbody)}
    for body in spec.bodies:
        if body.name in hand_bodies:
            i = hand_bodies[body.name]
            body.pos, body.quat = calibrated.body_pos[i], calibrated.body_quat[i]
            body.mass, body.inertia = calibrated.body_mass[i], calibrated.body_inertia[i]
            body.ipos, body.iquat = calibrated.body_ipos[i], calibrated.body_iquat[i]
            body.explicitinertial = True
    tag_id = mujoco.mj_name2id(calibrated, mujoco.mjtObj.mjOBJ_SITE, 'tag')
    if tag_id >= 0 and spec.site('tag') is not None:
        # The tag mount defines camera->base alignment, independently of hand
        # collision fidelity. Keep it identical across all selected scenes.
        spec.site('tag').pos = calibrated.site_pos[tag_id]
        spec.site('tag').quat = calibrated.site_quat[tag_id]
    for name in JOINTS:
        i = calibrated.joint(name).id
        joint = spec.joint(name)
        joint.range = calibrated.jnt_range[i]
        if measured_limits is not None:
            k = JOINTS.index(name)
            joint.range = [measured_limits.lower[k], measured_limits.upper[k]]
            joint.limited = 1
        joint.pos, joint.axis = calibrated.jnt_pos[i], calibrated.jnt_axis[i]
        dof = calibrated.jnt_dofadr[i]
        for field in ("damping", "armature", "frictionloss"):
            setattr(joint, field, joint_parameter(calibrated, dof, field))
        actuator = spec.actuator(name + "_act")
        i = calibrated.actuator(name + "_act").id
        for field in ("gainprm", "biasprm", "dynprm", "gear", "ctrlrange", "forcerange", "actrange"):
            setattr(actuator, field, getattr(calibrated, "actuator_" + field)[i])
        for field in ("ctrllimited", "forcelimited", "actlimited", "actearly"):
            setattr(actuator, field, int(getattr(calibrated, "actuator_" + field)[i]))
        actuator.dyntype = mujoco.mjtDyn(int(calibrated.actuator_dyntype[i]))
        actuator.gaintype = mujoco.mjtGain(int(calibrated.actuator_gaintype[i]))
        actuator.biastype = mujoco.mjtBias(int(calibrated.actuator_biastype[i]))
        actuator.actdim = 1
    if object_friction is not None:
        friction = np.asarray(object_friction, dtype=float)
        if friction.shape != (3,) or not np.isfinite(friction).all() or np.any(friction < 0):
            raise ValueError('Object friction requires three finite nonnegative values')
        obj = spec.body('obj')
        if obj is None:
            raise ValueError('Scene has no obj body for the cube friction override')
        pending, changed = [obj], 0
        while pending:
            body = pending.pop()
            pending.extend(body.bodies)
            for geom in body.geoms:
                if geom.contype or geom.conaffinity:
                    geom.friction = friction
                    changed += 1
        if not changed:
            raise ValueError('Cube friction override found no colliding object geometry')
    overrides = dict(fingertip=fingertip_friction, palm=palm_friction, finger=finger_friction)
    for group, value in overrides.items():
        if value is not None:
            friction = np.asarray(value, dtype=float)
            if friction.shape != (3,) or not np.isfinite(friction).all() or np.any(friction < 0):
                raise ValueError(f'{group} friction requires three finite nonnegative values')
    if any(value is not None for value in overrides.values()):
        changed = dict.fromkeys(overrides, 0)
        distal_bodies = {'if_ds', 'mf_ds', 'rf_ds', 'th_ds'}
        for body in spec.bodies:
            if body.name not in hand_bodies:
                continue
            for geom in body.geoms:
                if not (geom.contype or geom.conaffinity):
                    continue
                if body.name == 'palm':
                    group = 'palm'
                elif body.name.startswith(('if_', 'mf_', 'rf_', 'th_')):
                    # High/eval pads are named meshes. Low/med replace each
                    # pad with an unnamed capsule on the same distal link.
                    is_tip = (geom.name in FINGERTIP_SITES
                              or getattr(geom.classname, 'name', '') in ('tip', 'thumb_tip')
                              or (body.name in distal_bodies and not geom.name
                                  and geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE))
                    group = 'fingertip' if is_tip else 'finger'
                else:
                    continue
                if overrides[group] is not None:
                    geom.friction = overrides[group]
                    changed[group] += 1
        for group, value in overrides.items():
            if value is not None and not changed[group]:
                raise ValueError(f'{group} friction override found no matching colliding hand geometry')
    tree = ET.fromstring(spec.to_xml())
    def numbers(values):
        return " ".join(format(float(v), ".17g") for v in np.atleast_1d(values))
    # MjSpec's default writer rounds calibrated values; retain full precision.
    for body in tree.findall(".//worldbody//body"):
        if body.get("name") in hand_bodies:
            i = hand_bodies[body.get("name")]
            for attr in ("pos", "quat"):
                body.set(attr, numbers(getattr(calibrated, "body_" + attr)[i]))
            inertial = body.find("inertial")
            for attr, field in (("pos", "ipos"), ("quat", "iquat"), ("mass", "mass"), ("diaginertia", "inertia")):
                inertial.set(attr, numbers(getattr(calibrated, "body_" + field)[i]))
    for joint in tree.findall(".//worldbody//joint"):
        if joint.get("name") in JOINTS:
            index = calibrated.joint(joint.get("name")).id
            dof = calibrated.jnt_dofadr[index]
            for field in ("pos", "axis", "range"):
                joint.set(field, numbers(getattr(calibrated, "jnt_" + field)[index]))
            if measured_limits is not None:
                k = JOINTS.index(joint.get('name'))
                joint.set('range', numbers([measured_limits.lower[k], measured_limits.upper[k]]))
                joint.set('limited', 'true')
            for field in ("damping", "armature", "frictionloss"):
                joint.set(field, numbers(joint_parameter(calibrated, dof, field, for_xml=True)))
    for site in tree.findall('.//worldbody//site'):
        if site.get('name') == 'tag' and tag_id >= 0:
            site.set('pos', numbers(calibrated.site_pos[tag_id]))
            site.set('quat', numbers(calibrated.site_quat[tag_id]))
    for actuator in tree.findall("./actuator/*"):
        i = calibrated.actuator(actuator.get("name")).id
        for field in ("gainprm", "biasprm", "dynprm", "gear", "ctrlrange", "forcerange"):
            actuator.set(field, numbers(getattr(calibrated, "actuator_" + field)[i]))
    ET.indent(tree)
    Path(output).write_text(ET.tostring(tree, encoding="unicode") + "\n")
    return str(output)


class SceneTaskMixin:
    def __init__(self, scene, config):
        self.scene = str(scene)
        self.target_position_override = None
        super().__init__(config)

    def getModelPath(self):
        return self.scene

    def setTargetPosition(self, position):
        self.target_position_override = np.asarray(position, dtype=float).copy()

    def setGoal(self, goal):
        super().setGoal(goal)
        if self.target_position_override is not None:
            # DuckReorient normally moves the body origin to hold its COM fixed.
            # An explicit position must survive that calculation and every resample.
            self.params["target_pos"] = self.target_position_override.copy()
            LeapReorient.setGoal(self, goal)


def build_tasks(args, directory, hardware_config=None):
    rollout, evaluation = reference.buildTasks(args)
    joint_reference = (load_reference(rollout.OBJECT, getattr(args, 'joint_reference_file', None))
                       if getattr(args, 'object_joint_reference', True) else None)
    result, metadata = [], {}
    for name, task in (("rollout", rollout), ("eval", evaluation)):
        try:
            source = str(args.model) if args.model is not None else task.getModelPath()
            if not Path(source).is_file():
                raise FileNotFoundError(source)
        except FileNotFoundError as exc:
            raise ValueError(f"Scene unavailable for {args.task}, hand={args.hand_acc}, "
                             f"object={args.obj_acc}: {exc}") from exc
        path = calibrate_scene(source, args.calibrated_hand, Path(directory) / f"{name}.xml",
                               hardware_config=hardware_config,
                               object_friction=getattr(args, 'cube_friction', None),
                               **{f'{group}_friction': getattr(args, f'{group}_friction', None)
                                  for group in ('fingertip', 'palm', 'finger')})
        # Inherit the actual task class: cost, goal sampling and CONTACT_PARAMS
        # remain the same objects used by the simulation driver.
        cls = type(f"Calibrated{type(task).__name__}", (SceneTaskMixin, type(task)), {})
        calibrated_task = cls(path, task.config)
        if joint_reference is not None:
            calibrated_task.setJointReference(joint_reference['joint_reference'])
        result.append(calibrated_task)
        metadata[name] = dict(source_scene=source, scene_xml=Path(path).read_text(),
                              task_config=asdict(task.config),joint_reference=joint_reference)
    return *result, metadata


def build_rollout(args, task, lower, upper):
    overrides = {**reference.rolloutSimOverrides(args, task), **args.contact_params,
                 "device": args.device, "gravity": tuple(task.mjm.opt.gravity)}
    preset = getContactModel(args.rollout_model)
    if args.device == "cpu":
        if preset.simulator != "mujoco":
            raise ValueError("Serial CPU rollouts support M1/M2 only; select a CUDA device for M3–M6")
        cfg = preset.makeConfig(task, **overrides)
        return CpuRollouts(task.getModelPath(), cfg, args.n_samples, lower, upper)
    backend = GetContactModelSim(args.rollout_model, task.getModelPath(), N=args.n_samples,
                                 task=task, **overrides)
    return CalibratedRollouts(backend)


def make_evaluator(args, task):
    if isGpuEvalSim(args.eval_sim):
        from ContactModelStudy.Simulators.SingleWorld import SingleWorld
        backend = GetContactModelSim(args.eval_sim, task.getModelPath(), N=1, task=task,
                                    timestep=task.timestep, gravity=tuple(task.mjm.opt.gravity),
                                    device=args.device)
        return SingleWorld(CalibratedRollouts(backend), use_graph=args.graph)
    return makeEvalSim(args.eval_sim, task.getModelPath(), task.timestep, task=task)


def seed_evaluator(sim, q, v, u):
    if hasattr(sim, "inner") and hasattr(sim.inner, "activation"):
        sim.inner.activation.assign(np.asarray(u, dtype=np.float32))
    sim.SetState(q, v)
    sim.SetControl(u)
    if hasattr(sim, "mjd"):
        sim.mjd.act[:] = u


def evaluator_activation(sim, fallback):
    if hasattr(sim, "mjd"):
        return sim.mjd.act.copy()
    if hasattr(sim, "inner") and isinstance(sim.inner, CalibratedRollouts):
        inner = sim.inner
        return (inner.backend.d.act if inner.native else inner.act).numpy()[0].copy()
    return fallback
