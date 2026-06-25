"""test_robot_mujoco_model.py

Quick sanity check for a MuJoCo model: loads the XML, drives every
actuator with a small sine wave, and prints qpos as the sim steps.

Usage:
    python tests/test_robot_mujoco_model.py --xml scenes/leap_hand/leap_hand_right.xml
"""

import argparse

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=str, default="scenes/leap_hand/leap_hand_right.xml",
                        help="Path to the MJCF model to load")
    parser.add_argument("--amplitude", type=float, default=0.2, help="Sine wave amplitude (rad)")
    parser.add_argument("--frequency", type=float, default=0.5, help="Sine wave frequency (Hz)")
    parser.add_argument("--steps", type=int, default=500, help="Number of sim steps to run")
    parser.add_argument("--print_every", type=int, default=50, help="Print qpos every N steps")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    print(f"Loaded {args.xml}  (nq={model.nq}, nv={model.nv}, nu={model.nu})")

    for step in range(args.steps):
        t = step * model.opt.timestep
        data.ctrl[:] = args.amplitude * np.sin(2 * np.pi * args.frequency * t)

        mujoco.mj_step(model, data)

        if step % args.print_every == 0:
            print(f"[step {step:5d}] t={t:6.3f}s  ctrl={data.ctrl}  qpos={data.qpos}")


if __name__ == "__main__":
    main()
