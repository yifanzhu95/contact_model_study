#!/usr/bin/env python3
"""Smoke test for the refactored Simulators + Renderers, on the leap cube scene.

Starts the hand at the grasp_reorient task's fixed initial state and control,
curls every finger linearly inward, uncurls back to the start, and writes the
result to a video.

This is the first end-to-end exercise of the new `ContactModelStudy` package:
`Mujoco` steps the physics, `MujocoVideoRenderer` draws it, and nothing in the
simulator knows a renderer exists. Nothing from the old `contact_study.sim` or
`contact_study.contact_models` is involved — only the task's initial-state
table, so the hand starts exactly where the study's own task starts it.

Run:

    python tests/render_finger_curl.py
    python tests/render_finger_curl.py --curl 0.8 --seconds 1.5 --out /tmp/curl.mp4

Offscreen rendering needs a GL context. With no DISPLAY this defaults
MUJOCO_GL to "egl"; override it in the environment if that is wrong for your
machine.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Must happen before mujoco is imported anywhere: MUJOCO_GL is read once, when
# the GL backend is first resolved. Only fills in a default, never overrides.
if not os.environ.get("MUJOCO_GL") and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ContactModelStudy.Renderers.MujocoVideoRenderer import (  # noqa: E402
    MujocoVideoRenderer,
    MujocoVideoRendererConfig,
)
from ContactModelStudy.Simulators.Mujoco import Mujoco  # noqa: E402
from ContactModelStudy.Simulators.Simulator import SimulatorConfig  # noqa: E402

SCENE = REPO_ROOT / "scenes" / "leap" / "env_leap_eval_cube.xml"

# Joint-name suffixes that flex a digit. The leap hand names every actuated
# joint <digit>_<suffix>: if/mf/rf (index, middle, ring) have mcp/rot/pip/dip,
# the thumb has cmc/axl/mcp/ipl. Curling means driving the flexion joints only
# — `rot` spreads the fingers sideways, and the thumb's cmc/axl position the
# whole thumb rather than bending it, so all three are held at their initial
# command.
_FLEXION_SUFFIXES = ("mcp", "pip", "dip", "ipl")


def load_task_initial_state() -> tuple[np.ndarray, np.ndarray]:
    """Return the cube task's fixed (init_qpos, init_ctrl) from the old codebase.

    Read live from `contact_study.tasks.grasp_reorient` rather than copied here,
    so this script follows the task if those numbers are retuned. It is the same
    table `GraspReorientTask.get_inital_state` reads; the task class itself is
    not instantiated because that builds Warp arrays on a GPU, which this test
    has no use for.
    """
    try:
        from contact_study.tasks.grasp_reorient import _OBJ_PARAMS
    except ImportError as exc:
        raise SystemExit(
            f"Could not import the old task to read its initial state: {exc}\n"
            "Run this with the interpreter that has the study's deps installed, "
            "e.g. ~/anaconda3/envs/contact_modeling/bin/python"
        ) from exc

    cube = _OBJ_PARAMS["cube"]
    return np.asarray(cube["init_qpos"], float), np.asarray(cube["init_ctrl"], float)


def flexion_actuator_ids(mjm) -> list[int]:
    """Actuator indices that curl a digit, found by joint-name suffix.

    Resolved from the model rather than hardcoded, so a scene that renames or
    reorders actuators does not silently curl the wrong ones.
    """
    import mujoco

    ids = []
    for i in range(mjm.nu):
        joint = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, mjm.actuator_trnid[i, 0])
        if joint and joint.rsplit("_", 1)[-1] in _FLEXION_SUFFIXES:
            ids.append(i)
    return ids


def curl_control(ctrl0: np.ndarray, targets: np.ndarray, s: float) -> np.ndarray:
    """Interpolate from the initial command toward the fully-curled one.

    Args:
        ctrl0: The task's initial command, shape (nu,).
        targets: The fully-curled command, shape (nu,).
        s: Curl fraction in [0, 1]; 0 is the initial pose, 1 fully curled.
    """
    return ctrl0 + s * (targets - ctrl0)


def curl_fraction(step: int, steps_in: int, steps_out: int) -> float:
    """Triangular profile: 0 -> 1 over `steps_in`, then 1 -> 0 over `steps_out`.

    Linear in time in both directions, which is what makes the video read as a
    steady squeeze and release rather than a step input the position servos
    would simply chase.
    """
    if step < steps_in:
        return step / steps_in
    return max(0.0, 1.0 - (step - steps_in) / steps_out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "videos" / "leap_cube_finger_curl.mp4"),
                    help="output video (.mp4 or .gif)")
    ap.add_argument("--seconds", type=float, default=2.0,
                    help="seconds for the curl-in phase; the curl-out phase matches")
    ap.add_argument("--curl", type=float, default=0.6,
                    help="how far to curl, as a fraction of each flexion joint's "
                         "remaining range toward its upper control limit")
    ap.add_argument("--timestep", type=float, default=0.002, help="simulator timestep (s)")
    ap.add_argument("--fps", type=float, default=30.0, help="video frames per second")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--camera", default="demo-cam",
                    help="camera defined in the scene; 'none' for the free camera")
    args = ap.parse_args()

    if not 0.0 <= args.curl <= 1.0:
        ap.error(f"--curl must be in [0, 1], got {args.curl}")

    q0, ctrl0 = load_task_initial_state()

    sim = Mujoco(SCENE, SimulatorConfig(timestep=args.timestep))
    if sim.nq != q0.size or sim.nu != ctrl0.size:
        raise SystemExit(
            f"{SCENE.name} has nq={sim.nq} nu={sim.nu}, but the task's initial state "
            f"is nq={q0.size} nu={ctrl0.size}"
        )

    # Zero velocity: the task starts the hand and cube at rest.
    sim.SetState(q0)
    sim.SetControl(ctrl0)

    # Fully-curled command: each flexion joint driven `--curl` of the way from
    # its initial command to its upper control limit. Taking a fraction of the
    # REMAINING range (rather than a fixed angle) keeps every joint inside its
    # ctrlrange by construction, whatever pose the task starts from.
    flexion = flexion_actuator_ids(sim.mjm)
    upper = sim.mjm.actuator_ctrlrange[:, 1]
    targets = ctrl0.copy()
    targets[flexion] = ctrl0[flexion] + args.curl * (upper[flexion] - ctrl0[flexion])

    cfg = MujocoVideoRendererConfig(
        width=args.width, height=args.height, fps=args.fps,
        camera=None if args.camera.lower() == "none" else args.camera,
    )
    renderer = MujocoVideoRenderer(str(SCENE), cfg)

    steps_in = max(1, round(args.seconds / sim.timestep))
    steps_out = steps_in
    every = cfg.steps_per_frame(sim.timestep)

    print(f"scene      {SCENE.name}  (nq={sim.nq}, nv={sim.nv}, nu={sim.nu})")
    print(f"curling    {len(flexion)} of {sim.nu} actuators, {args.curl:.0%} of remaining range")
    print(f"schedule   {steps_in} steps in + {steps_out} steps out "
          f"@ dt={sim.timestep} ({2 * args.seconds:.1f} s of sim)")
    print(f"capture    every {every} steps -> {args.fps} fps")

    try:
        for step in range(steps_in + steps_out):
            sim.SetControl(curl_control(ctrl0, targets, curl_fraction(step, steps_in, steps_out)))
            sim.Step()
            if step % every == 0:
                renderer.RenderState(sim.GetState()[0])
        # Always end on the final state, so the video shows the hand back at the
        # start pose rather than stopping wherever the capture cadence landed.
        renderer.RenderState(sim.GetState()[0])

        out = renderer.Save(args.out)
    finally:
        renderer.Close()

    q_end, v_end = sim.GetState()
    # Measured against the COMMANDED pose, not against q0: the task's init_qpos
    # does not match its own init_ctrl (th_axl starts ~0.99 rad away, from a
    # literal "+ 1.0" in the task's table), so the servos pull the hand off q0
    # within the first few steps no matter what this script does. Comparing to
    # q0 would report that as drift.
    settled = np.abs(q_end[:sim.nu] - ctrl0).max()
    start_offset = np.abs(q0[:sim.nu] - ctrl0).max()
    print(f"\nhand back to within {settled:.4f} rad of the commanded pose "
          f"(the task's own init_qpos starts {start_offset:.4f} rad off it)")
    print(f"cube moved {np.linalg.norm(q_end[sim.nu:sim.nu + 3] - q0[sim.nu:sim.nu + 3]):.4f} m, "
          f"final speed {np.linalg.norm(v_end[sim.nu:sim.nu + 3]):.4f} m/s")
    print(f"wrote {renderer.frame_count} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
