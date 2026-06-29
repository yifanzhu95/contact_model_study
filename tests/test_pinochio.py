"""Load the LEAP hand MJCF into Pinocchio, sweep its joints, and render a video via EGL."""

import os
import xml.etree.ElementTree as ET

from panda3d.core import loadPrcFileData

# Force Panda3D's EGL-based headless GL pipe instead of the default GLX pipe,
# which would otherwise require a real X server / DISPLAY.
loadPrcFileData("", "load-display p3headlessgl")

import mediapy
import numpy as np
import pinocchio as pin
from pinocchio.visualize import Panda3dVisualizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(
    REPO_ROOT, "scenes/leap_hand/leap_hand_right_w_sites_simple.xml"
)
SCENE_DIR = os.path.dirname(MJCF_PATH)
OUT_PATH = os.path.join(REPO_ROOT, "videos/leap_hand_pinocchio.mp4")

N_FRAMES = 150
FPS = 30


def split_into_single_root_mjcfs(mjcf_path, scene_dir):
    """Pinocchio's MJCF parser only follows the first <body> under <worldbody>,
    but this model has 5 independent root bodies (3 fingers, thumb, free object).
    Write one temp MJCF per root body so each can be parsed into its own model."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    asset = root.find("asset")
    worldbody = root.find("worldbody")

    loose_geoms = [el for el in worldbody if el.tag == "geom"]
    bodies = [el for el in worldbody if el.tag == "body"]

    tmp_paths = []
    for i, body in enumerate(bodies):
        new_root = ET.Element("mujoco", root.attrib)
        if compiler is not None:
            new_root.append(compiler)
        if asset is not None:
            new_root.append(asset)
        new_worldbody = ET.SubElement(new_root, "worldbody")
        if i == 0:
            for geom in loose_geoms:
                new_worldbody.append(geom)
        new_worldbody.append(body)

        tmp_path = os.path.join(scene_dir, f"_tmp_pin_split_{i}.xml")
        ET.ElementTree(new_root).write(tmp_path)
        tmp_paths.append(tmp_path)

    return tmp_paths


def build_joint_sweep(model, n_frames=N_FRAMES):
    q0 = pin.neutral(model)
    q_traj = []
    for t in range(n_frames):
        q = q0.copy()
        phase = 2 * np.pi * t / n_frames
        for jid in range(1, model.njoints):
            joint = model.joints[jid]
            if joint.nq != 1:
                continue
            idx_q = joint.idx_q
            lo = model.lowerPositionLimit[idx_q]
            hi = model.upperPositionLimit[idx_q]
            q[idx_q] = lo + (hi - lo) * (0.5 + 0.5 * np.sin(phase))
        q_traj.append(q)
    return q_traj


def main():
    tmp_paths = split_into_single_root_mjcfs(MJCF_PATH, SCENE_DIR)
    try:
        models = [
            pin.buildModelsFromMJCF(p, contacts=False) for p in tmp_paths
        ]
    finally:
        for p in tmp_paths:
            os.remove(p)

    visualizers = []
    shared_viewer = None
    for i, (model, collision_model, visual_model) in enumerate(models):
        viz = Panda3dVisualizer(model, collision_model, visual_model)
        if shared_viewer is None:
            viz.initViewer(open=False)
            shared_viewer = viz.viewer
        else:
            viz.initViewer(viewer=shared_viewer)
        viz.loadViewerModel(group_name=f"part_{i}")
        viz.displayVisuals(True)
        visualizers.append(viz)

    # Default near clip plane (1.0m) clips this hand-scale (~0.2m) scene; pull it in.
    shared_viewer._app.camLens.set_near(0.01)
    shared_viewer.reset_camera(pos=(0.45, -0.4, 0.2), look_at=(0.0, 0.02, 0.02))

    q_trajs = [build_joint_sweep(model) for model, _, _ in models]

    frames = []
    for t in range(N_FRAMES):
        for viz, q_traj in zip(visualizers, q_trajs):
            viz.display(q_traj[t])
        frames.append(visualizers[0].captureImage())

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    mediapy.write_video(OUT_PATH, frames, fps=FPS)
    print(f"Saved video to {OUT_PATH}")


if __name__ == "__main__":
    main()
