"""Drake-backed eval simulator.

Wraps a pydrake MultibodyPlant as the high-fidelity "real" environment, exposing
the EvalSimulator interface so it is interchangeable with MujocoSimulator. This
generalizes the Drake half of tests/test_drake.py's `run_cartpole_drake`.

Drake stores state per-joint / per-floating-body, not in a single MuJoCo-ordered
qpos/qvel vector, so the caller supplies:
  * `joint_channels`     — revolute/prismatic joints mapped onto MuJoCo qpos/qvel
                           addresses (cart_pole slider/hinge; the LEAP hand joints).
  * `float_channels`     — floating bodies (the grasp cube) mapped onto a 7-dim
                           MuJoCo freejoint qpos [pos, quat(wxyz)] + 6-dim qvel
                           [lin(world), ang(body-local)].

Actuation is always position control via Drake's PidController system. A
PidController reads the plant state, compares the actuated joints' [q, v]
against the desired [q*, 0], and writes the actuation input port:
tau = kp*(q*-q) + ki*int(q*-q) - kd*qd  (see tests/view_model_drake.py).
Runs on a plant whose time_step is pid_plant_dt; keep it small/continuous
since the discrete SAP solver treats this external actuation explicitly and
can go unstable under stiff gains.

Conventions verified against this repo's models (Drake 1.12, mujoco 3.8):
  * Drake floating-body qpos is [quat(wxyz), pos]; MuJoCo's is [pos, quat]. We map
    via the pose/velocity APIs, never raw qpos slices, so the order is handled.
  * MuJoCo freejoint angular velocity is body-local: omega_world = R_WB @ omega_local.

pydrake is imported lazily so importing this module does not require Drake on the
GPU-rollout machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from contact_study.sim.base import EvalSimulator, EvalState, camera_pose_from_config


@dataclass
class DrakeJointChannel:
    """Maps one Drake revolute/prismatic joint to a MuJoCo qpos/qvel address."""
    drake_name: str
    kind: str          # "revolute" | "prismatic"
    q_adr: int
    v_adr: int


@dataclass
class DrakeFreeBodyChannel:
    """Maps a Drake floating body to a MuJoCo freejoint (7 qpos / 6 qvel)."""
    body_name: str
    q_adr: int         # MuJoCo qpos start: [px,py,pz, qw,qx,qy,qz]
    v_adr: int         # MuJoCo qvel start: [vx,vy,vz, wx,wy,wz] (ang body-local)


@dataclass
class DrakePidActuation:
    """Position control via Drake's PidController system.

    ctrl_joint_names is in MuJoCo *control* order: apply_control(ctrl) uses
    ctrl[k] as the desired position for joint ctrl_joint_names[k]. Each named
    joint must end up with exactly one actuator (reused from the parsed model or
    added by DrakeSimulator with the `effort` clamp).
    """
    kp: float
    ki: float
    kd: float
    ctrl_joint_names: list[str]
    effort: float = 100.0   # per-joint clamp for actuators DrakeSimulator adds


def _actuated_state_projection(plant) -> np.ndarray:
    """(2*nu, nq+nv) selection matrix mapping the plant's full state [q; v] onto
    the actuated joints' [q; v] in actuator-index order — the PidController
    state_projection (mirrors tests/view_model_drake.py)."""
    from pydrake.multibody.tree import JointActuatorIndex
    nu = plant.num_actuators()
    nq = plant.num_positions()
    nv = plant.num_velocities()
    P = np.zeros((2 * nu, nq + nv))
    for i in range(nu):
        joint = plant.get_joint_actuator(JointActuatorIndex(i)).joint()
        P[i, joint.position_start()] = 1.0
        P[nu + i, nq + joint.velocity_start()] = 1.0
    return P


class DrakeSimulator(EvalSimulator):
    def __init__(
        self,
        model_path: str,
        config,
        nq: int,
        nv: int,
        pid: DrakePidActuation,
        joint_channels: list[DrakeJointChannel] | None = None,
        float_channels: list[DrakeFreeBodyChannel] | None = None,
        weld_base: bool = False,
        video_path: str | None = None,
        pid_plant_dt: float = 0.0,
        extra_models_fn: Callable | None = None,
    ):
        from pydrake.math import RigidTransform, RotationMatrix
        from pydrake.common.eigen_geometry import Quaternion
        from pydrake.multibody.math import SpatialVelocity
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        from pydrake.multibody.tree import JointActuatorIndex
        from pydrake.systems.analysis import Simulator
        from pydrake.systems.controllers import PidController
        from pydrake.systems.framework import DiagramBuilder
        from pydrake.visualization import VideoWriter

        # Keep Drake type handles for hot-path use in get_state/set_state.
        self._RigidTransform = RigidTransform
        self._RotationMatrix = RotationMatrix
        self._Quaternion = Quaternion
        self._SpatialVelocity = SpatialVelocity

        self._config = config
        self.nq = nq
        self.nv = nv
        self._joint_channels = joint_channels or []
        self._float_channels = float_channels or []
        self._timestep = float(config.timestep)
        self._video_path = video_path
        self._pid = pid

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, pid_plant_dt)
        models = Parser(plant).AddModels(model_path)

        if weld_base:
            root_idx = plant.GetBodyIndices(models[0])[0]
            root_body = plant.get_body(root_idx)
            plant.WeldFrames(plant.world_frame(), root_body.body_frame())

        # Task-specific extra geometry (e.g. the grasp cube + ground), added
        # before Finalize so it is part of the multibody system.
        if extra_models_fn is not None:
            extra_models_fn(plant)

        # Actuators must exist before Finalize. The PidController only needs a
        # plain actuator on each controlled joint to write to: reuse any the
        # parsed model declares (the cart-pole URDF's <transmission> gives Drake
        # a CartSlider actuator) and add one otherwise (the LEAP URDF declares
        # none).
        self._hand_instance = models[0]
        existing = {
            plant.get_joint_actuator(a).joint().name()
            for a in plant.GetJointActuatorIndices(self._hand_instance)
        }
        for jname in pid.ctrl_joint_names:
            if jname not in existing:
                j = plant.GetJointByName(jname)
                plant.AddJointActuator(jname + "_act", j, pid.effort)

        plant.Finalize()

        # PidController is added after Finalize (it needs the actuator count and
        # the plant's state port) and before Build. Wire plant state -> PID ->
        # actuation; the desired-state port is left unconnected and Fixed each
        # apply_control.
        nu = plant.num_actuators()
        P = _actuated_state_projection(plant)
        self._pid_sys = builder.AddSystem(PidController(
            P, np.full(nu, pid.kp), np.full(nu, pid.ki), np.full(nu, pid.kd),
        ))
        builder.Connect(plant.get_state_output_port(),
                        self._pid_sys.get_input_port_estimated_state())
        builder.Connect(self._pid_sys.get_output_port_control(),
                        plant.get_actuation_input_port())
        # apply_control feeds ctrl in pid.ctrl_joint_names order; the PID's
        # desired state is in actuator-index order (matching P). Precompute
        # the actuator-index -> ctrl-index permutation.
        act_joints = [
            plant.get_joint_actuator(JointActuatorIndex(i)).joint().name()
            for i in range(nu)
        ]
        self._pid_act_to_ctrl = np.array(
            [pid.ctrl_joint_names.index(n) for n in act_joints], dtype=int
        )
        self._pid_nu = nu

        self._video = None
        if video_path is not None:
            R, p = camera_pose_from_config(config)
            camera_pose = RigidTransform(RotationMatrix(R), p)
            self._video = VideoWriter.AddToBuilder(
                filename=video_path, builder=builder,
                sensor_pose=camera_pose, fps=config.cam_fps,
            )

        self._diagram = builder.Build()
        self._simulator = Simulator(self._diagram)
        self._simulator.Initialize()
        context = self._simulator.get_mutable_context()
        self._context = context
        self._plant = plant
        self._plant_context = plant.GetMyMutableContextFromRoot(context)

        self._joints = {
            ch.drake_name: plant.GetJointByName(ch.drake_name) for ch in self._joint_channels
        }
        self._bodies = {
            ch.body_name: plant.GetBodyByName(ch.body_name) for ch in self._float_channels
        }

        # Fix the controller's desired-state input [q*, v*]; set each
        # apply_control. Initialize to zero (overwritten before stepping).
        self._pid_context = self._pid_sys.GetMyMutableContextFromRoot(context)
        self._pid_desired_port = self._pid_sys.get_input_port_desired_state()
        self._pid_desired_port.FixValue(
            self._pid_context, np.zeros(2 * self._pid_nu)
        )

        self._t = 0.0

    # -- per-channel read/write ---------------------------------------------
    def _read_joint(self, ch, qpos, qvel):
        j = self._joints[ch.drake_name]; ctx = self._plant_context
        if ch.kind == "revolute":
            qpos[ch.q_adr] = j.get_angle(ctx)
            qvel[ch.v_adr] = j.get_angular_rate(ctx)
        elif ch.kind == "prismatic":
            qpos[ch.q_adr] = j.get_translation(ctx)
            qvel[ch.v_adr] = j.get_translation_rate(ctx)
        else:
            raise ValueError(f"Unsupported Drake joint kind: {ch.kind}")

    def _write_joint(self, ch, qpos, qvel):
        j = self._joints[ch.drake_name]; ctx = self._plant_context
        if ch.kind == "revolute":
            j.set_angle(ctx, float(qpos[ch.q_adr]))
            j.set_angular_rate(ctx, float(qvel[ch.v_adr]))
        elif ch.kind == "prismatic":
            j.set_translation(ctx, float(qpos[ch.q_adr]))
            j.set_translation_rate(ctx, float(qvel[ch.v_adr]))
        else:
            raise ValueError(f"Unsupported Drake joint kind: {ch.kind}")

    def _read_free(self, ch, qpos, qvel):
        body = self._bodies[ch.body_name]; ctx = self._plant_context
        X = self._plant.GetFreeBodyPose(ctx, body)
        a = ch.q_adr
        qpos[a:a + 3] = X.translation()
        quat = X.rotation().ToQuaternion()
        qpos[a + 3] = quat.w(); qpos[a + 4] = quat.x()
        qpos[a + 5] = quat.y(); qpos[a + 6] = quat.z()
        V = self._plant.EvalBodySpatialVelocityInWorld(ctx, body)
        v = ch.v_adr
        qvel[v:v + 3] = V.translational()
        # MuJoCo freejoint angular velocity is body-local.
        qvel[v + 3:v + 6] = X.rotation().matrix().T @ V.rotational()

    def _write_free(self, ch, qpos, qvel):
        body = self._bodies[ch.body_name]; ctx = self._plant_context
        a = ch.q_adr
        q = np.asarray(qpos[a + 3:a + 7], dtype=float)
        n = np.linalg.norm(q)
        q = q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])
        quat = self._Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
        R = self._RotationMatrix(quat)
        X = self._RigidTransform(R, np.asarray(qpos[a:a + 3], dtype=float))
        self._plant.SetFreeBodyPose(ctx, body, X)
        v = ch.v_adr
        w_world = R.matrix() @ np.asarray(qvel[v + 3:v + 6], dtype=float)
        V = self._SpatialVelocity(w_world, np.asarray(qvel[v:v + 3], dtype=float))
        self._plant.SetFreeBodySpatialVelocity(body, V, ctx)

    # -- EvalSimulator interface --------------------------------------------
    def reset(self, qpos, qvel) -> None:
        self._context.SetTime(0.0)
        self._t = 0.0
        self.set_state(qpos, qvel)

    def set_state(self, qpos, qvel) -> None:
        for ch in self._joint_channels:
            self._write_joint(ch, qpos, qvel)
        for ch in self._float_channels:
            self._write_free(ch, qpos, qvel)

    def get_state(self) -> EvalState:
        qpos = np.zeros(self.nq)
        qvel = np.zeros(self.nv)
        for ch in self._joint_channels:
            self._read_joint(ch, qpos, qvel)
        for ch in self._float_channels:
            self._read_free(ch, qpos, qvel)
        return EvalState(qpos, qvel)

    def apply_control(self, ctrl) -> None:
        # Desired position per actuator (desired velocity 0); the PidController
        # turns the tracking error into actuation. ctrl is MuJoCo-ordered, so
        # remap to actuator-index order for the desired-state port.
        ctrl = np.asarray(ctrl, dtype=float)
        q_d = ctrl[self._pid_act_to_ctrl]
        self._pid_desired_port.FixValue(
            self._pid_context, np.concatenate([q_d, np.zeros(self._pid_nu)])
        )

    def step(self, n_substeps: int = 1) -> None:
        self._t += n_substeps * self._timestep
        self._simulator.AdvanceTo(self._t)

    def render(self) -> None:
        # VideoWriter captures automatically during AdvanceTo at its fps clock.
        pass

    def save_video(self, path: str | None = None) -> None:
        if self._video is None:
            return
        self._video.Save()
        if path is not None and path != self._video_path:
            os.replace(self._video_path, path)

    @property
    def timestep(self) -> float:
        return self._timestep
