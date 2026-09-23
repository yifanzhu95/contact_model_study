#!/usr/bin/env python3
"""Run closed-loop MPC episodes and report what happened.

Wires the package together: a task supplies the scene, the initial state and
the cost; a ``VectorizedMujoco`` rolls samples out on the GPU; ``MPPI`` plans
against it; a CPU ``Mujoco`` stands in for reality; and a
``MujocoVideoRenderer`` records it.

The planner and the evaluator load *different scenes from the same task* — the
rollout scene the planner predicts with, and the accurate eval scene it is
judged on. That gap is what the study measures, so it is the default here
rather than something to switch on.

Examples::

    python -m ContactModelStudy.Drivers.run_episodes
    python -m ContactModelStudy.Drivers.run_episodes --n-episodes 5 --steps 200
    python -m ContactModelStudy.Drivers.run_episodes --hand-acc low --no-video
    python -m ContactModelStudy.Drivers.run_episodes --n-samples 512 --temperature 5 \\
        --results results/run.json

Offscreen rendering needs a GL context; with no DISPLAY this defaults
MUJOCO_GL to "egl".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Before any mujoco import: MUJOCO_GL is read once, when the GL backend is
# first resolved. Only fills in a default, never overrides.
if not os.environ.get("MUJOCO_GL") and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ContactModelStudy.Renderers.MujocoVideoRenderer import (  # noqa: E402
    MujocoVideoRenderer,
    MujocoVideoRendererConfig,
)
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config  # noqa: E402
from ContactModelStudy.Simulators.Mujoco import Mujoco  # noqa: E402
from ContactModelStudy.Simulators.Simulator import SimulatorConfig  # noqa: E402
from ContactModelStudy.Simulators.VectorizedMujoco import (  # noqa: E402
    VectorizedMujoco,
    VectorizedMujocoConfig,
)
from ContactModelStudy.Tasks.CubeReorient import CubeReorient  # noqa: E402
from ContactModelStudy.Tasks.TaskBase import TaskRole  # noqa: E402

#: Tasks this driver can run, by the name passed to --task.
TASKS = {"cube_reorient": CubeReorient}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    g = p.add_argument_group("task")
    g.add_argument("--task", default="cube_reorient", choices=sorted(TASKS))
    g.add_argument("--hand-acc", default="high", choices=["low", "med", "high"],
                   help="hand mesh fidelity of the planner's rollout scene")
    g.add_argument("--obj-acc", default="high",
                   help="object mesh fidelity of the planner's rollout scene")

    g = p.add_argument_group("episode")
    g.add_argument("--n-episodes", type=int, default=1)
    g.add_argument("--steps", type=int, default=100,
                   help="control steps per episode")
    g.add_argument("--timestep", type=float, default=0.002,
                   help="physics timestep (s), used by both simulators")
    g.add_argument("--substeps", type=int, default=4,
                   help="physics steps per control step; control rate is "
                        "1/(timestep*substeps)")
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("planner")
    g.add_argument("--n-samples", type=int, default=256,
                   help="N: rollout worlds, one per sampled control sequence")
    g.add_argument("--horizon", type=int, default=8,
                   help="H: planning horizon in control steps")
    g.add_argument("--n-iterations", type=int, default=1)
    g.add_argument("--noise-sigma", type=float, default=0.05)
    g.add_argument("--temperature", type=float, default=1.0,
                   help="MPPI lambda; smaller is greedier")
    g.add_argument("--control-mode", default="relative", choices=["relative", "absolute"])
    g.add_argument("--delta", type=float, default=None,
                   help="symmetric per-step clip on the control delta; omitted "
                        "leaves it unclipped (the study's default)")
    g.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--adaptive-temp", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True,
                   help="capture the rollout into a CUDA graph (much faster)")

    g = p.add_argument_group("gpu buffers")
    g.add_argument("--nconmax", type=int, default=None,
                   help="contact buffer capacity; None lets MJWarp size it")
    g.add_argument("--njmax", type=int, default=None,
                   help="constraint buffer capacity; None lets MJWarp size it")

    g = p.add_argument_group("output")
    g.add_argument("--video", default="videos/run_episodes.mp4",
                   help="video path; episode index is appended when >1 episode")
    g.add_argument("--no-video", dest="video", action="store_const", const=None)
    g.add_argument("--fps", type=float, default=30.0)
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--camera", default="demo-cam", help="'none' for the free camera")
    g.add_argument("--results", default=None, help="write a JSON summary here")
    g.add_argument("--debug", action="store_true", help="per-plan MPPI diagnostics")
    return p


def run_episode(
    episode: int, args, task, eval_task, sim, planner, renderer
) -> dict:
    """Run one closed-loop episode and return its summary.

    The planner sees the eval simulator's state and returns an absolute command;
    the eval simulator holds that command for ``substeps`` physics steps. Costs
    are reported with the *eval* task, so the number being tracked is
    performance on the accurate scene, not on the planner's own model of it.
    """
    q0, v0, u0 = eval_task.getInitialState()
    sim.SetState(q0, v0)
    sim.SetControl(u0)
    # Under absolute control the mean IS the command, so a zero mean would open
    # the hand and drop the object on the first plan; seed it with the task's
    # initial grasp. Under relative control the mean is a delta and zero
    # correctly means "hold the measured pose".
    planner.Reset(u0 if args.control_mode == "absolute" else None)
    if renderer is not None:
        renderer.Reset()

    every = max(1, round(1.0 / (args.fps * sim.timestep * args.substeps)))
    costs, plan_ms = [], []

    for step in range(args.steps):
        q, q_dot = sim.GetState()
        costs.append(float(eval_task.calcCosts(q, q_dot)))

        t0 = time.perf_counter()
        u = planner.Plan(q, q_dot)
        plan_ms.append((time.perf_counter() - t0) * 1e3)

        sim.SetControl(u)
        sim.Step(args.substeps)
        if renderer is not None and step % every == 0:
            renderer.RenderState(sim.GetState()[0])

    q, q_dot = sim.GetState()
    costs.append(float(eval_task.calcCosts(q, q_dot)))
    if renderer is not None:
        renderer.RenderState(q)

    obj_adr = int(task.indices[0])
    target = np.asarray(task.params["target_pos"], float)
    fallen_z = float(task.params["fallen_z"])
    held = bool(q[obj_adr + 2] > fallen_z)

    return {
        "episode": episode,
        "steps": args.steps,
        "cost_start": costs[0],
        "cost_end": costs[-1],
        "cost_min": min(costs),
        "cost_mean": float(np.mean(costs)),
        "object_dist_start": float(np.linalg.norm(q0[obj_adr:obj_adr + 3] - target)),
        "object_dist_end": float(np.linalg.norm(q[obj_adr:obj_adr + 3] - target)),
        "object_height_end": float(q[obj_adr + 2]),
        "held": held,
        # The first plan of a run pays kernel compilation and CUDA-graph
        # capture — often 100x the steady-state cost — so it is reported
        # separately rather than being averaged into a figure that then
        # describes neither.
        "plan_ms_first": float(plan_ms[0]),
        "plan_ms_mean": float(np.mean(plan_ms[1:])) if len(plan_ms) > 1 else float(plan_ms[0]),
        "plan_ms_max": float(np.max(plan_ms[1:])) if len(plan_ms) > 1 else float(plan_ms[0]),
        "costs": costs,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    task_cls = TASKS[args.task]
    # Two views of the same task: the scene the planner predicts with, and the
    # accurate one it is scored on.
    task = task_cls(role=TaskRole.ROLLOUT, hand_acc=args.hand_acc, obj_acc=args.obj_acc)
    eval_task = task_cls(role=TaskRole.EVAL)

    sim = Mujoco(eval_task.getModelPath(), SimulatorConfig(timestep=args.timestep))
    rollout_sim = VectorizedMujoco(
        task.getModelPath(),
        VectorizedMujocoConfig(
            timestep=args.timestep, substeps=args.substeps, horizon=args.horizon,
            nconmax=args.nconmax, njmax=args.njmax,
        ),
        N=args.n_samples,
    )
    delta_range = (None, None) if args.delta is None else (-args.delta, args.delta)
    planner = MPPI(rollout_sim, task, MPPI_Config(
        noise_sigma=args.noise_sigma, n_iterations=args.n_iterations,
        temperature=args.temperature, adaptive_temp=args.adaptive_temp,
        control_mode=args.control_mode, delta_range=delta_range,
        warm_start=args.warm_start, seed=args.seed, use_graph=args.graph,
        debug=args.debug,
    ))

    control_hz = 1.0 / (args.timestep * args.substeps)
    print(f"task       {args.task}  (rollout {args.hand_acc}_{args.obj_acc})")
    print(f"eval scene {Path(eval_task.getModelPath()).name}")
    print(f"rollout    {Path(task.getModelPath()).name}")
    print(f"planner    {planner!r}")
    print(f"control    {control_hz:.1f} Hz  ({args.steps} steps = "
          f"{args.steps / control_hz:.2f} s per episode)")

    summaries = []
    for episode in range(args.n_episodes):
        renderer = None
        video_path = None
        if args.video:
            video_path = args.video
            if args.n_episodes > 1:
                p = Path(args.video)
                video_path = str(p.with_name(f"{p.stem}_ep{episode}{p.suffix}"))
            renderer = MujocoVideoRenderer(eval_task, MujocoVideoRendererConfig(
                width=args.width, height=args.height, fps=args.fps,
                camera=None if args.camera.lower() == "none" else args.camera,
            ))
        try:
            s = run_episode(episode, args, task, eval_task, sim, planner, renderer)
            if renderer is not None:
                s["video"] = renderer.Save(video_path)
        finally:
            if renderer is not None:
                renderer.Close()

        summaries.append(s)
        print(f"\nepisode {episode}: cost {s['cost_start']:.3f} -> {s['cost_end']:.3f} "
              f"(min {s['cost_min']:.3f})")
        print(f"  object {s['object_dist_start']:.4f} -> {s['object_dist_end']:.4f} m "
              f"from target, {'held' if s['held'] else 'DROPPED'} "
              f"(z={s['object_height_end']:.4f})")
        print(f"  plan {s['plan_ms_mean']:.1f} ms mean, {s['plan_ms_max']:.1f} ms max "
              f"({1e3 / s['plan_ms_mean']:.0f} Hz), first {s['plan_ms_first']:.0f} ms "
              f"(compile + graph capture)")
        if s.get("video"):
            print(f"  video {s['video']}")

    if args.n_episodes > 1:
        held = sum(s["held"] for s in summaries)
        print(f"\n{args.n_episodes} episodes: {held} held, "
              f"mean final cost {np.mean([s['cost_end'] for s in summaries]):.3f}, "
              f"mean final distance "
              f"{np.mean([s['object_dist_end'] for s in summaries]):.4f} m")

    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"config": vars(args), "episodes": summaries}, indent=2, default=str
        ))
        print(f"\nresults {out}")

    rollout_sim.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
