"""Drake-backed eval simulator.

Wraps a pydrake MultibodyPlant as the high-fidelity "real" environment, exposing
the EvalSimulator interface so it is interchangeable with MujocoSimulator. This
generalizes the Drake half of tests/test_drake.py's `run_cartpole_drake`.

Drake stores state per-joint, not in a single MuJoCo-ordered qpos/qvel vector, so
the caller supplies a list of `DrakeJointChannel`s describing how each Drake joint
maps onto MuJoCo qpos/qvel addresses. get_state()/apply_control() use that map to
translate, keeping the driver simulator-agnostic.

pydrake is imported lazily (inside __init__) so that importing this module — and
hence `contact_study.contact_models` — does not require Drake on machines that
only run the GPU rollouts.

Phase A supports revolute/prismatic joints (cart_pole). Floating-base bodies
(the grasp cube) are a Phase-B extension point, marked below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from contact_study.sim.base import EvalSimulator, EvalState, camera_pose_from_config


@dataclass
class DrakeJointChannel:
    """Maps one Drake joint to a slice of the MuJoCo qpos/qvel layout."""
    drake_name: str
    kind: str          # "revolute" | "prismatic"
    q_adr: int         # MuJoCo qpos address
    v_adr: int         # MuJoCo qvel address (dof address)


class DrakeSimulator(EvalSimulator):
    def __init__(
        self,
        model_path: str,
        config,
        nq: int,
        nv: int,
        joint_channels: list[DrakeJointChannel],
        n_ctrl: int,
        ctrl_mj_to_drake: np.ndarray | None = None,
        weld_base: bool = False,
        video_path: str | None = None,
    ):
        # Lazy Drake imports.
        from pydrake.math import RigidTransform, RotationMatrix
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        from pydrake.systems.analysis import Simulator
        from pydrake.systems.framework import DiagramBuilder
        from pydrake.visualization import VideoWriter

        self._config = config
        self.nq = nq
        self.nv = nv
        self._channels = joint_channels
        self._timestep = float(config.timestep)
        self._video_path = video_path
        # Identity remap if none given (MuJoCo ctrl order == Drake actuation order).
        self._ctrl_map = (
            np.arange(n_ctrl) if ctrl_mj_to_drake is None
            else np.asarray(ctrl_mj_to_drake, dtype=int)
        )

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)  # continuous
        models = Parser(plant).AddModels(model_path)

        if weld_base:
            # URDFs whose root link isn't attached to the world (e.g. the LEAP
            # hand) would free-fall; weld the first model's root body in place.
            root_idx = plant.GetBodyIndices(models[0])[0]
            root_body = plant.get_body(root_idx)
            plant.WeldFrames(plant.world_frame(), root_body.body_frame())

        plant.Finalize()

        # Offscreen camera from the shared TaskConfig pose.
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

        # Resolve joint handles once.
        self._joints = {
            ch.drake_name: plant.GetJointByName(ch.drake_name) for ch in joint_channels
        }
        self._act_port = plant.get_actuation_input_port()
        self._act_port.FixValue(self._plant_context, np.zeros(n_ctrl))

        self._t = 0.0

    # -- per-joint read/write -----------------------------------------------
    def _read_channel(self, ch: DrakeJointChannel, qpos, qvel) -> None:
        j = self._joints[ch.drake_name]
        ctx = self._plant_context
        if ch.kind == "revolute":
            qpos[ch.q_adr] = j.get_angle(ctx)
            qvel[ch.v_adr] = j.get_angular_rate(ctx)
        elif ch.kind == "prismatic":
            qpos[ch.q_adr] = j.get_translation(ctx)
            qvel[ch.v_adr] = j.get_translation_rate(ctx)
        else:
            raise ValueError(f"Unsupported Drake joint kind: {ch.kind}")

    def _write_channel(self, ch: DrakeJointChannel, qpos, qvel) -> None:
        j = self._joints[ch.drake_name]
        ctx = self._plant_context
        if ch.kind == "revolute":
            j.set_angle(ctx, float(qpos[ch.q_adr]))
            j.set_angular_rate(ctx, float(qvel[ch.v_adr]))
        elif ch.kind == "prismatic":
            j.set_translation(ctx, float(qpos[ch.q_adr]))
            j.set_translation_rate(ctx, float(qvel[ch.v_adr]))
        else:
            raise ValueError(f"Unsupported Drake joint kind: {ch.kind}")

    # -- EvalSimulator interface --------------------------------------------
    def reset(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self._context.SetTime(0.0)
        self._t = 0.0
        self.set_state(qpos, qvel)

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        for ch in self._channels:
            self._write_channel(ch, qpos, qvel)

    def get_state(self) -> EvalState:
        qpos = np.zeros(self.nq)
        qvel = np.zeros(self.nv)
        for ch in self._channels:
            self._read_channel(ch, qpos, qvel)
        return EvalState(qpos, qvel)

    def apply_control(self, ctrl: np.ndarray) -> None:
        u = np.asarray(ctrl, dtype=float)[self._ctrl_map]
        self._act_port.FixValue(self._plant_context, u)

    def step(self, n_substeps: int = 1) -> None:
        self._t += n_substeps * self._timestep
        self._simulator.AdvanceTo(self._t)

    def render(self) -> None:
        # Drake's VideoWriter captures frames automatically during AdvanceTo at
        # its own fps clock, so there is nothing to do per control step.
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
