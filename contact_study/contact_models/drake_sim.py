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

Two actuation modes:
  * torque   (pd=None)   — write the MuJoCo control straight to the plant's
                           actuation input port (cart_pole).
  * manual PD (pd set)   — the URDF hand has no transmissions, and MuJoCo drives
                           it with `position kp=3 kv=0.01` servos. We reproduce
                           that by computing tau = kp*(target-q) - kd*qd per joint
                           each control step and applying it through the plant's
                           applied-generalized-force input port. Held constant
                           over the AdvanceTo, so it matches the MuJoCo servo
                           cadence exactly at substeps=1.

Conventions verified against this repo's models (Drake 1.12, mujoco 3.8):
  * Drake floating-body qpos is [quat(wxyz), pos]; MuJoCo's is [pos, quat]. We map
    via the pose/velocity APIs, never raw qpos slices, so the order is handled.
  * MuJoCo freejoint angular velocity is body-local: omega_world = R_WB @ omega_local.

pydrake is imported lazily so importing this module does not require Drake on the
GPU-rollout machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
class DrakePdActuation:
    """Manual PD position control matching MuJoCo `position` servos.

    ctrl_joint_names is in MuJoCo *control* order, so apply_control(ctrl) uses
    ctrl[k] as the target for joint ctrl_joint_names[k].
    """
    kp: float
    kd: float
    ctrl_joint_names: list[str]
    effort: float = 0.95   # per-joint torque clamp (URDF effort limit)


@dataclass
class DrakePidActuation:
    """Position control via Drake's PidController system.

    A PidController reads the plant state, compares the actuated joints' [q, v]
    against the desired [q*, 0], and writes the actuation input port:
    tau = kp*(q*-q) + ki*int(q*-q) - kd*qd  (see tests/view_model_drake.py).

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
        joint_channels: list[DrakeJointChannel] | None = None,
        float_channels: list[DrakeFreeBodyChannel] | None = None,
        n_ctrl: int = 0,
        ctrl_mj_to_drake: np.ndarray | None = None,
        weld_base: bool = False,
        video_path: str | None = None,
        time_step: float = 0.0,
        extra_models_fn: Callable | None = None,
        pd: DrakePdActuation | None = None,
        pd_native_dt: float = 0.0,
        pid: DrakePidActuation | None = None,
        pid_plant_dt: float = 0.0,
    ):
        from pydrake.math import RigidTransform, RotationMatrix
        from pydrake.common.eigen_geometry import Quaternion
        from pydrake.multibody.math import SpatialVelocity
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        from pydrake.multibody.tree import JointActuator, JointActuatorIndex
        from pydrake.systems.analysis import Simulator
        from pydrake.systems.framework import DiagramBuilder
        from pydrake.visualization import VideoWriter

        # Position control comes in two flavors, selected by which arg is set:
        #   * pid — Drake's PidController system reads the plant state, compares
        #     the actuated joints against a desired [q*, 0] and writes the
        #     actuation port (see tests/view_model_drake.py). The shipped tasks
        #     use this. Runs on a plant whose time_step is pid_plant_dt; keep it
        #     small/continuous since the discrete SAP solver treats this external
        #     actuation explicitly and can go unstable under stiff gains.
        #   * pd — native PD-controlled actuators (implicit, discrete) with a
        #     manual-PD continuous fallback on old Drake. Legacy path.
        self._use_native_pd = (
            pd is not None
            and pid is None
            and pd_native_dt > 0.0
            and hasattr(JointActuator, "set_controller_gains")
        )
        if pid is not None:
            time_step = pid_plant_dt
        elif pd is not None:
            time_step = pd_native_dt if self._use_native_pd else 0.0

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
        self._pd = pd
        self._pid = pid
        self._ctrl_map = (
            np.arange(n_ctrl) if ctrl_mj_to_drake is None
            else np.asarray(ctrl_mj_to_drake, dtype=int)
        )

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
        models = Parser(plant).AddModels(model_path)

        if weld_base:
            root_idx = plant.GetBodyIndices(models[0])[0]
            root_body = plant.get_body(root_idx)
            plant.WeldFrames(plant.world_frame(), root_body.body_frame())

        # Task-specific extra geometry (e.g. the grasp cube + ground), added
        # before Finalize so it is part of the multibody system.
        if extra_models_fn is not None:
            extra_models_fn(plant)

        # Actuators must exist before Finalize. The PidController (pid path) only
        # needs a plain actuator on each controlled joint to write to: reuse any
        # the parsed model declares (the cart-pole SDF exposes its CartSlider
        # actuator) and add one otherwise (the LEAP URDF declares none).
        self._hand_instance = models[0]
        if pid is not None:
            existing = {
                plant.get_joint_actuator(a).joint().name()
                for a in plant.GetJointActuatorIndices(self._hand_instance)
            }
            for jname in pid.ctrl_joint_names:
                if jname not in existing:
                    j = plant.GetJointByName(jname)
                    plant.AddJointActuator(jname + "_act", j, pid.effort)
        elif self._use_native_pd:
            # Native PD-controlled actuators: one per joint, in pd.ctrl_joint_names
            # order, so apply_control can feed targets as ctrl[:n] directly. The
            # parsed model must contribute none (extras would widen the actuation
            # port and, indexed in the model's own joint order, scramble the
            # target<->joint mapping).
            from pydrake.multibody.tree import PdControllerGains
            n_existing = len(plant.GetJointActuatorIndices(self._hand_instance))
            if n_existing:
                raise RuntimeError(
                    f"{model_path} already declares {n_existing} actuator(s) on "
                    "the hand instance; DrakeSimulator adds its own PD actuators "
                    "and needs the model to declare none (drop URDF "
                    "<transmission> blocks / MJCF actuators)."
                )
            gains = PdControllerGains(p=pd.kp, d=pd.kd)
            for jname in pd.ctrl_joint_names:
                j = plant.GetJointByName(jname)
                act = plant.AddJointActuator(jname + "_act", j, pd.effort)
                act.set_controller_gains(gains)

        plant.Finalize()

        # PidController is added after Finalize (it needs the actuator count and
        # the plant's state port) and before Build. Wire plant state -> PID ->
        # actuation; the desired-state port is left unconnected and Fixed each
        # apply_control.
        self._pid_sys = None
        if pid is not None:
            from pydrake.systems.controllers import PidController
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

        self._act_port = self._force_port = self._desired_state_port = None
        self._pid_desired_port = None
        if pid is not None:
            # PID: Fix the controller's desired-state input [q*, v*]; set each
            # apply_control. Initialize to zero (overwritten before stepping).
            self._pid_context = self._pid_sys.GetMyMutableContextFromRoot(context)
            self._pid_desired_port = self._pid_sys.get_input_port_desired_state()
            self._pid_desired_port.FixValue(
                self._pid_context, np.zeros(2 * self._pid_nu)
            )
        elif pd is None:
            # Torque actuation: write straight to the actuation port.
            self._act_port = plant.get_actuation_input_port()
            if n_ctrl > 0:
                self._act_port.FixValue(self._plant_context, np.zeros(n_ctrl))
        elif self._use_native_pd:
            # Native PD: feed-forward actuation = 0; desired state = [q*, 0].
            n = len(pd.ctrl_joint_names)
            plant.get_actuation_input_port(self._hand_instance).FixValue(
                self._plant_context, np.zeros(n)
            )
            self._desired_state_port = plant.get_desired_state_input_port(
                self._hand_instance
            )
            self._desired_state_port.FixValue(self._plant_context, np.zeros(2 * n))
        else:
            # Manual PD on a continuous plant via the applied-force port.
            self._force_port = plant.get_applied_generalized_force_input_port()
            self._force_port.FixValue(self._plant_context, np.zeros(nv))
            self._pd_joints = []
            for jname in pd.ctrl_joint_names:
                j = plant.GetJointByName(jname)
                lo = float(np.ravel(j.position_lower_limit())[0])
                hi = float(np.ravel(j.position_upper_limit())[0])
                self._pd_joints.append((j, j.velocity_start(), lo, hi))

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
        ctrl = np.asarray(ctrl, dtype=float)
        ctx = self._plant_context
        if self._pid is not None:
            # Desired position per actuator (desired velocity 0); the PidController
            # turns the tracking error into actuation. ctrl is MuJoCo-ordered, so
            # remap to actuator-index order for the desired-state port.
            q_d = ctrl[self._pid_act_to_ctrl]
            self._pid_desired_port.FixValue(
                self._pid_context, np.concatenate([q_d, np.zeros(self._pid_nu)])
            )
            return
        if self._pd is None:
            # Torque actuation (cart_pole).
            self._act_port.FixValue(ctx, ctrl[self._ctrl_map])
            return
        if self._use_native_pd:
            # Native PD-controlled actuators: desired state = [targets, zeros].
            n = len(self._pd.ctrl_joint_names)
            desired = np.concatenate([ctrl[:n], np.zeros(n)])
            self._desired_state_port.FixValue(ctx, desired)
            return
        # Manual PD: tau = kp*(target - q) - kd*qd, clamped, into the force port.
        tau = np.zeros(self.nv)
        for k, (j, v_start, lo, hi) in enumerate(self._pd_joints):
            target = float(np.clip(ctrl[k], lo, hi))
            q = j.get_angle(ctx)
            qd = j.get_angular_rate(ctx)
            t = self._pd.kp * (target - q) - self._pd.kd * qd
            tau[v_start] = float(np.clip(t, -self._pd.effort, self._pd.effort))
        self._force_port.FixValue(ctx, tau)

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
