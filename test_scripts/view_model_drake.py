"""Load any URDF/SDF/MJCF into a Drake scene and render a PID-controlled,
random-walk-of-desired-position control.

Generic smoke-test: builds a MultibodyPlant from a single model file, wraps
every actuated joint with a PID controller, and drives the *desired position*
of each actuated joint with an independent random walk

    q_d_{k+1} = q_d_k + N(0, sigma)

(desired velocity is held at 0), then renders the simulation to a video.
CPU-only (no CUDA/GPU, no MPPI) — just Drake + its VTK software renderer — so
it runs on any machine.

Works with any number of actuators, including underactuated models (e.g. the
cart-pole's unactuated pole joint): a state-projection matrix selects the
actuated joints' positions/velocities out of the plant's full state for PID
feedback. The walk dimension is read from plant.num_actuators() at build time.

Demo (cart-pole, 1 actuator):
    python tests/view_model_drake.py
    python tests/view_model_drake.py --model scenes/cart_pole.urdf --sigma 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import JointActuatorIndex
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import PidController
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import VideoWriter

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = THIS_DIR.parent / "scenes" / "cart_pole.urdf"
VIDEOS_DIR = THIS_DIR.parent / "videos"


class RandomWalkDesiredState(LeafSystem):
    """Outputs a PidController-style desired state [q_d, v_d] of size 2*nu.

    q_d is an nu-dimensional random walk: q_d_{k+1} = q_d_k + N(0, sigma).
    v_d is always 0 (we only walk the desired position).
    """

    def __init__(self, nu: int, sigma: float, period_sec: float, seed: int | None = None):
        super().__init__()
        self._sigma = sigma
        self._rng = np.random.default_rng(seed)

        self.DeclareDiscreteState(nu)
        self.DeclareVectorOutputPort("desired_state", 2 * nu, self._calc_output)
        self.DeclarePeriodicDiscreteUpdateEvent(period_sec, 0.0, self._step_walk)

    def _step_walk(self, context, discrete_state):
        q_d = context.get_discrete_state(0).get_value()
        noise = self._rng.normal(0.0, self._sigma, q_d.shape)
        discrete_state.get_mutable_vector(0).SetFromVector(q_d + noise)

    def _calc_output(self, context, output):
        q_d = context.get_discrete_state(0).get_value()
        output.SetFromVector(np.concatenate([q_d, np.zeros_like(q_d)]))


def _build_actuated_state_projection(plant) -> np.ndarray:
    """Builds the (2*nu, nq+nv) matrix selecting [q_d; v_d] of actuated joints
    out of the plant's full state [q; v], for PidController's state_projection.
    """
    nu = plant.num_actuators()
    nq = plant.num_positions()
    nv = plant.num_velocities()
    state_projection = np.zeros((2 * nu, nq + nv))
    for i in range(nu):
        joint = plant.get_joint_actuator(JointActuatorIndex(i)).joint()
        state_projection[i, joint.position_start()] = 1.0
        state_projection[nu + i, nq + joint.velocity_start()] = 1.0
    return state_projection


def run(
    model_path: str,
    sigma: float = 0.05,
    kp: float = 1.0,
    ki: float = 0.0,
    kd: float = 0.05,
    sim_time: float = 10.0,
    fps: float = 30.0,
    seed: int | None = None,
    video_path: str | None = None,
):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0001)  # continuous
    model_instances = Parser(plant).AddModels(model_path)

    root_body_index = plant.GetBodyIndices(model_instances[0])[0]
    root_body = plant.get_body(root_body_index)
    plant.WeldFrames(plant.world_frame(), root_body.body_frame())

    plant.Finalize()

    nu = plant.get_actuation_input_port().size()
    state_projection = _build_actuated_state_projection(plant)
    pid = builder.AddSystem(PidController(
        state_projection, np.full(nu, kp), np.full(nu, ki), np.full(nu, kd),
    ))
    walk = builder.AddSystem(RandomWalkDesiredState(nu, sigma, period_sec=1.0 / fps, seed=seed))
    builder.Connect(plant.get_state_output_port(), pid.get_input_port_estimated_state())
    builder.Connect(walk.get_output_port(0), pid.get_input_port_desired_state())
    builder.Connect(pid.get_output_port_control(), plant.get_actuation_input_port())

    # Offscreen camera in front of the scene looking toward -x, world +z up.
    R_world_camera = RotationMatrix(np.array([
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]))
    camera_pose = RigidTransform(R_world_camera, [0.50, 0.0, 0.05])

    if video_path is None:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        video_path = str(VIDEOS_DIR / f"{Path(model_path).stem}_random_walk.gif")
    video = VideoWriter.AddToBuilder(
        filename=video_path, builder=builder, sensor_pose=camera_pose, fps=fps,
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()

    print(f"  model={model_path}  nu={nu}  sigma={sigma}  kp={kp} ki={ki} kd={kd}  sim_time={sim_time}s")
    simulator.AdvanceTo(sim_time)

    video.Save()
    print(f"  Saved video -> {video_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                   help="Path to a URDF/SDF/MJCF model file.")
    p.add_argument("--sigma", type=float, default=0.10,
                   help="Std dev of the per-step random-walk increment in desired position"
                        " (rad or m, in joint units).")
    p.add_argument("--kp", type=float, default=1.0, help="PID proportional gain (per actuator).")
    p.add_argument("--ki", type=float, default=0.0, help="PID integral gain (per actuator).")
    p.add_argument("--kd", type=float, default=0.05, help="PID derivative gain (per actuator).")
    p.add_argument("--sim_time", type=float, default=1.00)
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--video", type=str, default=None)
    args = p.parse_args()

    run(
        model_path=args.model,
        sigma=args.sigma,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        sim_time=args.sim_time,
        fps=args.fps,
        seed=args.seed,
        video_path=args.video,
    )


if __name__ == "__main__":
    main()
