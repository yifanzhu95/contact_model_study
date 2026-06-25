"""Load a URDF in MuJoCo and save it as an MJCF (.xml) file alongside it.

Joints referenced by a <transmission> block in the URDF get a matching
position-controlled <actuator> added to the saved MJCF, since MuJoCo's
URDF importer otherwise drops actuation info.
"""

import argparse
import os
import xml.etree.ElementTree as ET
import numpy as np

import mujoco


def find_actuated_joints(urdf_path):
    """Return the names of joints referenced by a <transmission> in the URDF."""
    root = ET.parse(urdf_path).getroot()
    joint_names = []
    for transmission in root.findall("transmission"):
        for joint in transmission.findall("joint"):
            name = joint.get("name")
            if name is not None:
                joint_names.append(name)
    return joint_names


def add_position_actuators(mjcf_path, joint_ranges, kp, kd):
    """Add a <default> block and one <position> actuator per joint to a saved MJCF file."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    default = ET.Element("default")
    ET.SubElement(default, "position", {"kp": str(kp), "kv": str(kd)})
    ET.SubElement(default, "joint", {"damping": "0.03", "frictionloss": "0.001"})
    root.insert(0, default)

    actuator = ET.SubElement(root, "actuator")
    for joint_name, joint_range in joint_ranges.items():
        ET.SubElement(
            actuator,
            "position",
            {
                "name": joint_name,
                "joint": joint_name,
                "ctrllimited": "true",
                "ctrlrange": f"{joint_range[0]} {joint_range[1]}",
            },
        )

    ET.indent(tree, space="  ")
    tree.write(mjcf_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf_path", help="Path to the input URDF file")
    parser.add_argument("--kp", type=float, default=3.0, help="Position gain for added actuators")
    parser.add_argument("--kd", type=float, default=0.0, help="Velocity (damping) gain for added actuators")
    args = parser.parse_args()

    urdf_path = args.urdf_path
    mjcf_path = os.path.splitext(urdf_path)[0] + ".xml"

    actuated_joints = find_actuated_joints(urdf_path)

    spec = mujoco.MjSpec.from_file(urdf_path)
    joint_names = {joint.name for joint in spec.joints}

    joint_ranges = {}
    for joint_name in actuated_joints:
        if joint_name not in joint_names:
            print(f"Warning: transmission joint '{joint_name}' not found in MJCF model, skipping")
            continue

        joint = spec.joint(joint_name)

        # act = spec.add_actuator()
        # act.name = joint_name
        # act.target = joint_name
        # act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        # act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        # act.gainprm = [args.kp, 0.0, 0.0] + list(act.gainprm[3:])
        # act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        # act.biasprm = [0.0, -args.kp, -args.kd] + list(act.biasprm[3:])
        # act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        # act.ctrlrange = list(joint.range)

        joint_ranges[joint_name] = list(joint.range)

    spec.compile()
    spec.to_file(mjcf_path)
    add_position_actuators(mjcf_path, joint_ranges, args.kp, args.kd)

    print(f"Saved MJCF to {mjcf_path}")
    if actuated_joints:
        print(f"Added actuators for joints: {', '.join(actuated_joints)}")


if __name__ == "__main__":
    main()
