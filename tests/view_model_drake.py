"""Load any URDF/SDF/MJCF into a Drake scene and render a random-walk control.

Generic smoke-test: builds a MultibodyPlant from a single model file, drives
every actuator with an independent random walk in action space

    u_{k+1} = u_k + N(0, sigma)

and renders the simulation to a video. CPU-only (no CUDA/GPU, no MPPI) — just
Drake + its VTK software renderer — so it runs on any machine.

Works with any number of actuators: the walk dimension is read from
plant.get_actuation_input_port().size() at build time.

Demo (cart-pole, 1 actuator):
    python tests/view_model_drake.py
    python tests/view_model_drake.py --model scenes/cart_pole.sdf --sigma 5.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import VideoWriter

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = THIS_DIR.parent / "scenes" / "cart_pole.sdf"
VIDEOS_DIR = THIS_DIR.parent / "videos"


class RandomWalkActuation(LeafSystem):
    """Outputs an nu-dimensional random walk: u_{k+1} = u_k + N(0, sigma)."""

    def __init__(self, nu: int, sigma: float, period_sec: float, seed: int | None = None):
        super().__init__()
        self._sigma = sigma
        self._rng = np.random.default_rng(seed)

        self.DeclareDiscreteState(nu)
        self.DeclareVectorOutputPort("u", nu, self._copy_state)
        self.DeclarePeriodicDiscreteUpdateEvent(period_sec, 0.0, self._step_walk)

    def _step_walk(self, context, discrete_state):
        u = context.get_discrete_state(0).get_value()
        noise = self._rng.normal(0.0, self._sigma, u.shape)
        discrete_state.get_mutable_vector(0).SetFromVector(u + noise)

    def _copy_state(self, context, output):
        output.SetFromVector(context.get_discrete_state(0).get_value())


def run(
    model_path: str,
    sigma: float = 5.0,
    sim_time: float = 10.0,
    fps: float = 30.0,
    seed: int | None = None,
    video_path: str | None = None,
):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)  # continuous
    model_instances = Parser(plant).AddModels(model_path)

    root_body_index = plant.GetBodyIndices(model_instances[0])[0]
    root_body = plant.get_body(root_body_index)
    plant.WeldFrames(plant.world_frame(), root_body.body_frame())

    plant.Finalize()

    nu = plant.get_actuation_input_port().size()
    walk = builder.AddSystem(RandomWalkActuation(nu, sigma, period_sec=1.0 / fps, seed=seed))
    builder.Connect(walk.get_output_port(0), plant.get_actuation_input_port())

    # Offscreen camera in front of the scene looking toward +y, world +z up.
    R_world_camera = RotationMatrix(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]))
    camera_pose = RigidTransform(R_world_camera, [0.0, -1, 0.25])

    if video_path is None:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        video_path = str(VIDEOS_DIR / f"{Path(model_path).stem}_random_walk.gif")
    video = VideoWriter.AddToBuilder(
        filename=video_path, builder=builder, sensor_pose=camera_pose, fps=fps,
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()

    print(f"  model={model_path}  nu={nu}  sigma={sigma}  sim_time={sim_time}s")
    simulator.AdvanceTo(sim_time)

    video.Save()
    print(f"  Saved video -> {video_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                   help="Path to a URDF/SDF/MJCF model file.")
    p.add_argument("--sigma", type=float, default=0.10,
                   help="Std dev of the per-step random-walk increment, in actuator units.")
    p.add_argument("--sim_time", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--video", type=str, default=None)
    args = p.parse_args()

    run(
        model_path=args.model,
        sigma=args.sigma,
        sim_time=args.sim_time,
        fps=args.fps,
        seed=args.seed,
        video_path=args.video,
    )


if __name__ == "__main__":
    main()
