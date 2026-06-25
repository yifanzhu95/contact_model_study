"""One-off generator for scenes/leap_hand/scene_grasp_reorient.xml.

Run this manually whenever leap_hand_right.xml's hand/floor/obj geometry
changes; the output is committed and loaded as a static file by
contact_study/tasks/grasp_reorient.py (no MJCF building happens at task-load
time). Adds the pieces that have no URDF equivalent and so can't live in
leap_hand_right.urdf/.xml: fingertip sites, the obj_target mocap goal, and the
"home" keyframe (fully-open finger pose + cube rest state).

Usage:
    python scenes/scripts/build_grasp_reorient_scene.py
"""

from pathlib import Path

import mujoco
import numpy as np

SCENES_DIR = Path(__file__).parents[1]
HAND_XML   = SCENES_DIR / "leap_hand" / "leap_hand_right.xml"
OUT_XML    = SCENES_DIR / "leap_hand" / "scene_grasp_reorient.xml"

# Hand starts fully open (every joint at 0, its fully-extended end of range —
# all 16 joint ranges include 0) so the controller has to reach in and close
# around the cube, rather than starting pre-curled around it.
HOME_FINGER_QPOS = [0.0] * 16
OBJ_TARGET_EULER = (0.0, 0.5235, 0.0)   # ~30 deg about y; goal sampling is
                                         # relative to this, exact value is arbitrary


def euler_to_wxyz(rx: float, ry: float, rz: float) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([rx, ry, rz]), "xyz")
    return quat


def main():
    spec = mujoco.MjSpec.from_file(str(HAND_XML))

    obj_body = spec.body("obj")
    obj_home_pos = np.array(obj_body.pos, dtype=float)

    # Fingertip sites at each tip body's local origin (good enough proxy for
    # "tip location"; this asset's per-body local frames aren't calibrated to
    # expose a precise surface-contact offset).
    for site_name, body_name in [
        ("if_tip", "fingertip"), ("mf_tip", "fingertip_2"),
        ("rf_tip", "fingertip_3"), ("th_tip", "thumb_fingertip"),
    ]:
        site = spec.body(body_name).add_site()
        site.name = site_name

    target = spec.worldbody.add_body()
    target.name  = "obj_target"
    target.mocap = True
    target.pos   = list(obj_home_pos)
    target.quat  = list(euler_to_wxyz(*OBJ_TARGET_EULER))
    ts = target.add_site()
    ts.name = "obj_target"
    ts.type = mujoco.mjtGeom.mjGEOM_BOX
    ts.size = [0.035, 0.035, 0.035]
    ts.rgba = [1.0, 1.0, 0.0, 0.1]

    key = spec.add_key()
    key.name = "home"
    key.qpos = np.concatenate([HOME_FINGER_QPOS, obj_home_pos, [1.0, 0.0, 0.0, 0.0]])
    key.qvel = np.zeros(16 + 6)
    key.ctrl = np.array(HOME_FINGER_QPOS, dtype=float)

    model = spec.compile()  # validate before writing
    print(f"compiled OK: nq={model.nq} nv={model.nv} nu={model.nu} nkey={model.nkey}")

    OUT_XML.write_text(spec.to_xml())
    print(f"wrote {OUT_XML}")


if __name__ == "__main__":
    main()
