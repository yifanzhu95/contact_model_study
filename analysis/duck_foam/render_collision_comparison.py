"""Render the Duck collision candidates from one fixed MuJoCo viewpoint.

The original visual mesh (geom group 0) is kept visible and the collision
geometry (geom group 3) is overlaid.  No controller or FOAM checkout is needed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
import mujoco


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "results" / "duck_collision_models_comparison.png"

SCENES = (
    ("8 convex hulls\n(reference baseline)", "env_leap_rollout_duck_low_high.xml"),
    ("FOAM4\n4 spheres", "env_leap_rollout_duck_low_foam4.xml"),
    ("FOAM16A\n16 spheres", "env_leap_rollout_duck_low_foam16a.xml"),
    ("FOAM16B\n16 spheres", "env_leap_rollout_duck_low_foam16b.xml"),
    ("FOAM64\n64 spheres", "env_leap_rollout_duck_low_foam64.xml"),
)


def render_scene(xml_path: Path, width: int, height: int):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1  # hand and original Duck visual geometry
    option.geomgroup[3] = 1  # collision hulls or FOAM spheres

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera="demo-cam", scene_option=option)
        return renderer.render().copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel-width", type=int, default=520)
    parser.add_argument("--panel-height", type=int, default=520)
    args = parser.parse_args()

    scene_dir = PROJECT_ROOT / "scenes" / "leap"
    images = [
        render_scene(scene_dir / filename, args.panel_width, args.panel_height)
        for _, filename in SCENES
    ]

    fig, axes = plt.subplots(1, len(SCENES), figsize=(18, 4.2), constrained_layout=True)
    for axis, image, (title, _) in zip(axes, images, SCENES, strict=True):
        axis.imshow(image)
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    fig.suptitle(
        "Duck rollout collision geometry (fixed low-detail hand and camera)",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
