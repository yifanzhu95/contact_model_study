"""Load a URDF in MuJoCo and save it as an MJCF (.xml) file alongside it."""

import argparse
import os

import mujoco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf_path", help="Path to the input URDF file")
    args = parser.parse_args()

    urdf_path = args.urdf_path
    mjcf_path = os.path.splitext(urdf_path)[0] + ".xml"

    model = mujoco.MjModel.from_xml_path(urdf_path)
    mujoco.mj_saveLastXML(mjcf_path, model)

    print(f"Saved MJCF to {mjcf_path}")


if __name__ == "__main__":
    main()
