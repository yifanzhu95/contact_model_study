# Build a cart-pole in Drake, drive the cart with a sinusoidal force, and
# render the simulation to an offscreen video file.
#
# Run with the `mujoco` conda env, which is where pydrake is installed:
#   /Users/kyleg/miniforge3/envs/mujoco/bin/python tests/test_drake.py
import os

import numpy as np

from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import Sine
from pydrake.visualization import VideoWriter

# Locate the scene relative to this file so the script is runnable from anywhere.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(THIS_DIR, os.pardir, "scenes", "cart_pole.sdf")
# PIL (the default VideoWriter backend) can write .gif out of the box. For an
# .mp4 instead, `pip install imageio[ffmpeg]` and change this extension to .mp4.
OUTPUT = os.path.join(THIS_DIR, "cart_pole.gif")

# Simulation / control / rendering settings.
SIM_TIME = 10.0      # seconds
TIME_STEP = 0.0      # 0.0 -> continuous-time plant
FPS = 30.0           # video frame rate
FORCE_AMPLITUDE = 12.0   # N, peak force applied to the cart
FORCE_FREQUENCY = 1.0    # Hz of the sinusoidal drive

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, TIME_STEP)
Parser(plant).AddModels(SCENE)
plant.Finalize()

# Sinusoidal control signal -> cart actuator.
# The CartPole model has a single actuator on the prismatic "CartSlider" joint.
# Sine(amplitudes, frequencies [rad/s], phases, size) outputs amp * sin(2*pi*f*t).
n_act = plant.get_actuation_input_port().size()
sine = builder.AddSystem(
    Sine(
        amplitudes=np.full(n_act, FORCE_AMPLITUDE),
        frequencies=np.full(n_act, 2.0 * np.pi * FORCE_FREQUENCY),
        phases=np.zeros(n_act),
    )
)
builder.Connect(sine.get_output_port(0), plant.get_actuation_input_port())

# Offscreen camera that records the scene to a video file.
# Drake's camera frame looks down its +z axis, with +y pointing down in the
# image. Place it in front of the scene (at -y) looking toward +y, with world
# +z mapped to image-up, so the cart-pole appears upright.
# Columns of this matrix are the camera axes expressed in the world frame:
#   camera +x -> world +x (image right)
#   camera +y -> world -z (image down)
#   camera +z -> world +y (view direction, pointing into the scene)
R_world_camera = RotationMatrix(
    np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ])
)
camera_pose = RigidTransform(R_world_camera, [0.0, -2.5, -0.25])
video = VideoWriter.AddToBuilder(
    filename=OUTPUT,
    builder=builder,
    sensor_pose=camera_pose,
    fps=FPS,
)

diagram = builder.Build()

simulator = Simulator(diagram)
simulator.Initialize()

# Give the pole a small initial tilt so the motion is interesting.
context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(context)
plant.GetJointByName("PolePin").set_angle(plant_context, 0.1)

simulator.AdvanceTo(SIM_TIME)

video.Save()
print(f"Wrote video to {OUTPUT}")
