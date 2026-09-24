#!/usr/bin/env python3
"""Run closed-loop MPC episodes and report what happened.

Wires the package together: a task supplies the scene, the initial state and
the cost; a ``VectorizedMujoco`` rolls samples out on the GPU; ``MPPI`` plans
against it; a CPU ``Mujoco`` stands in for reality; and a
``MujocoVideoRenderer`` films it; an ``EpisodeRecorder`` keeps every step.

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

``--results`` writes the ``EpisodeRecorder`` output: ``run.json`` (configs and
one summary per episode) and one ``run_<id>.npy`` per episode holding every
control step's state, action, action uncertainty and planning time. Read it
back with ``ContactModelStudy.Utils.EpisodeRecorder.EpisodeReplayer``.

Offscreen rendering needs a GL context; with no DISPLAY this defaults
MUJOCO_GL to "egl".
"""

from __future__ import annotations

import argparse
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

from ContactModelStudy.Renderers.MujocoVideoRenderer import MujocoVideoRenderer  # noqa: E402
from ContactModelStudy.Renderers.RendererBase import VideoRendererBaseConfig  # noqa: E402
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config  # noqa: E402
from ContactModelStudy.Simulators.VectorizedMujoco import (  # noqa: E402
    VectorizedMujoco,
    VectorizedMujocoConfig,
)
from ContactModelStudy.Tasks.CubeReorient import CubeReorient  # noqa: E402
from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig  # noqa: E402
from ContactModelStudy.Tasks.TaskBase import TaskRole  # noqa: E402
from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder  # noqa: E402
from ContactModelStudy.Utils.EvalSimulators import EVAL_SIMS, makeEvalSim  # noqa: E402

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
    g.add_argument("--steps", type=int, default=4000,
                   help="control steps per episode")
    g.add_argument("--timestep", type=float, default=0.0005,
                   help="eval (fine) physics timestep (s); the rollout model runs "
                        "at timestep * --eval-steps-per-rollout-step")
    g.add_argument("--eval-steps-per-rollout-step", type=int, default=8,
                   help="eval steps per rollout step (the old task used 8, with "
                        "a 0.5 ms eval timestep)")
    g.add_argument("--substeps", type=int, default=4,
                   help="rollout steps per control step; control rate is "
                        "1/(rollout timestep * substeps)")
    g.add_argument("--eval-sim", default="mujoco", choices=sorted(EVAL_SIMS),
                   help="simulator the episode is scored in (the rollouts always run "
                        "on MJWarp); every one runs the same eval MJCF")
    g.add_argument("--settle", type=float, default=1.0,
                   help="seconds to let the hand and object come to rest, holding "
                        "the initial grasp, before planning starts (filmed, not "
                        "counted in --steps)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--goal-difficulty", type=int, default=8, choices=range(10),
                   metavar="{0..9}",
                   help="goal sampler; 8 (the old default) rolls to show 'O' or "
                        "'B'. See LeapReorient.GOAL_DIFFICULTIES")
    g.add_argument("--stop-on-success", action=argparse.BooleanOptionalAction, default=True,
                   help="end the episode at the first success (the old driver's "
                        "default); with --no-stop-on-success each success samples "
                        "a new goal and the episode carries on to --steps")

    g = p.add_argument_group("planner")
    g.add_argument("--n-samples", type=int, default=256,
                   help="N: rollout worlds, one per sampled control sequence")
    g.add_argument("--horizon", type=int, default=8,
                   help="H: planning horizon in control steps")
    g.add_argument("--n-iterations", type=int, default=1)
    g.add_argument("--noise-sigma", type=float, default=0.1)
    g.add_argument("--temperature", type=float, default=25.0,
                   help="MPPI lambda; smaller is greedier")
    g.add_argument("--control-mode", default="pos_relative",
                   choices=["pos_relative", "ctrl_relative", "absolute"],
                   help="pos_relative: ctrl = q + U; ctrl_relative: ctrl = "
                        "ctrl_prev + U; absolute: ctrl = U")
    g.add_argument("--delta", type=float, default=None,
                   help="symmetric per-step clip on the control delta; omitted "
                        "leaves it unclipped (the study's default)")
    g.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--adaptive-temp", action=argparse.BooleanOptionalAction, default=False)
    g.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True,
                   help="capture the rollout into a CUDA graph (much faster)")

    g = p.add_argument_group("gpu buffers")
    g.add_argument("--nconmax", type=int, default=None,
                   help="max contacts per rollout world; None lets MJWarp size it")
    g.add_argument("--njmax", type=int, default=None,
                   help="max constraint rows per rollout world; None lets MJWarp "
                        "size it (64 for the leap scenes, which overflows in a grasp)")

    g = p.add_argument_group("output")
    g.add_argument("--video", default="videos/run_episodes.mp4",
                   help="video path; episode index is appended when >1 episode")
    g.add_argument("--no-video", dest="video", action="store_const", const=None)
    g.add_argument("--fps", type=float, default=30.0)
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--camera", default=None,
                   help="camera defined in the scene; 'none' for the free "
                        "camera. Omitted, the task's own camera is used")
    g.add_argument("--results", default=None,
                   help="save the recorded episodes: a JSON summary here, plus one "
                        ".npy of per-step data per episode beside it")
    g.add_argument("--uncertainty", action=argparse.BooleanOptionalAction, default=True,
                   help="have the planner report, and the recorder keep, the "
                        "uncertainty of each action")
    g.add_argument("--debug", action="store_true", help="per-plan MPPI diagnostics")
    return p


def _newGoal(task, eval_task, renderer) -> str:
    """Sample one goal and give it to both tasks and the renderer.

    The two task instances load different scenes but must chase one goal, so it
    is sampled once — on the eval task, which owns the episode and its seed —
    and set on both. Each new goal is a rotation of the previous one. The
    renderer keeps its own copy of the scene, so it is told too, or the video's
    goal marker would never move. Returns the letter of the face the goal shows.
    """
    goal = eval_task.sampleNewGoal()
    eval_task.setGoal(goal)
    task.setGoal(goal)
    if renderer is not None:
        eval_task.setRendererToGoal(renderer)
    return eval_task.goalFace() if hasattr(eval_task, "goalFace") else "?"


def run_episode(
    episode: int, args, task, eval_task, sim, planner, renderer, recorder
) -> dict:
    """Run one closed-loop episode, recording every control step, and summarize it.

    Each step's state, action, action uncertainty (when the planner reports
    one) and planning time go to ``recorder``. The episode is left open there;
    the caller finishes it with the returned summary, once the video is saved,
    so the video's path can go in too.

    The planner sees the eval simulator's state and returns an absolute command;
    the eval simulator holds that command for ``substeps`` physics steps.

    The episode is judged with the *eval* task's ``isSuccess`` / ``isFailure``,
    evaluated on the start-of-step state as the old driver did. A failure always
    ends it. A success ends it too unless ``--no-stop-on-success``, in which
    case a fresh goal is sampled and the episode carries on — the old driver's
    multi-goal mode.
    """
    eval_task.setSimToInitialState(sim)
    _, _, u0 = eval_task.getInitialState()
    # Under absolute control the mean IS the command, so a zero mean would open
    # the hand and drop the object on the first plan; seed it with the task's
    # initial grasp. Under both relative modes the mean is a delta, and zero
    # correctly means "hold" — the pose, or the command.
    reset_mean = u0 if args.control_mode == "absolute" else None
    goals = [_newGoal(task, eval_task, renderer)]
    planner.Reset(reset_mean)
    if renderer is not None:
        renderer.Reset()

    # One control step advances the eval simulator this many of its own (fine)
    # steps: `substeps` rollout steps, each eval_steps_per_rollout_step long.
    eval_steps = args.substeps * eval_task.config.eval_steps_per_rollout_step
    # Frames are due every `steps_per_frame` eval steps — on the simulation
    # clock, not the control clock. A control step can be longer than a frame
    # (64 ms vs 33 ms at 30 fps), so capturing once per control step would drop
    # frames and play back fast; _advance steps the eval sim in pieces that end
    # exactly on each frame deadline instead.
    steps_per_frame = renderer.getStepsPerFrame() if renderer is not None else 0
    eval_clock = [0]            # eval steps taken this episode

    def _advance(n: int) -> None:
        """Advance the eval sim n steps, capturing every frame that falls due."""
        if renderer is None:
            sim.Step(n)
            return
        while n > 0:
            chunk = min(n, steps_per_frame - eval_clock[0] % steps_per_frame)
            sim.Step(chunk)
            eval_clock[0] += chunk
            n -= chunk
            if eval_clock[0] % steps_per_frame == 0:
                renderer.RenderState(sim.GetState()[0])

    if renderer is not None:
        renderer.RenderState(sim.GetState()[0])
    # Settle: the object drops into the palm and the hand closes on it under
    # the initial grasp command, which setSimToInitialState applied and which
    # is held here. Filmed like the rest, so the video opens with it; nothing is
    # planned or recorded until it is over, and it does not count as a step.
    settle_steps = int(round(args.settle / eval_task.timestep))
    _advance(settle_steps)

    goal_errors = getattr(eval_task, "goalErrors", None)
    # The episode starts from the settled state: start errors and the first
    # success/failure checks are taken on it.
    q_start, v_start = sim.GetState()
    plan_ms, planner_cost, successes = [], [], []
    end_reason, steps_taken = "timeout", args.steps

    for step in range(args.steps):
        if eval_task.isFailure(sim):
            end_reason, steps_taken = "failure", step
            break
        if eval_task.isSuccess(sim):
            successes.append(step)
            if args.stop_on_success:
                end_reason, steps_taken = "success", step
                break
            goals.append(_newGoal(task, eval_task, renderer))
            planner.Reset(reset_mean)

        q, q_dot = sim.GetState()
        t0 = time.perf_counter()
        # The applied control is passed so ctrl_relative always accumulates from
        # what the hand is actually commanded — at step 0 the task's initial
        # grasp, not whatever the planner returned last episode.
        out = planner.Plan(q, q_dot, u=sim.GetControl())
        dt = time.perf_counter() - t0
        u, sigma = out if isinstance(out, tuple) else (out, None)
        plan_ms.append(dt * 1e3)
        recorder.recordStateAndAction(q, q_dot, u, sigma, planning_time=dt)
        # The planner's best rollout cost, on its own (rollout) model. A
        # progress signal, not a score: the eval task has no host-side cost.
        planner_cost.append(float(getattr(planner, "last_min_cost", float("nan"))))

        sim.SetControl(u)
        _advance(eval_steps)

    q, q_dot = sim.GetState()
    # End on the final state, unless that state was itself a frame deadline and
    # is already the last frame — a duplicate would stretch playback by a frame.
    if renderer is not None and eval_clock[0] % steps_per_frame != 0:
        renderer.RenderState(q)
    # A multi-goal run that timed out still succeeded if it reached any goal.
    if end_reason == "timeout" and successes:
        end_reason = "success"

    summary = {
        "episode": episode,
        "end_reason": end_reason,
        "success": bool(successes),
        "steps_to_success": successes[0] if successes else None,
        "goals_reached": len(successes),
        "success_steps": successes,
        "goals": goals,
        "failed": end_reason == "failure",
        "steps_taken": steps_taken,
        "planner_min_cost": planner_cost,
        "settle_s": settle_steps * eval_task.timestep,
        # The recorded steps hold the states actions were planned from; the
        # state the last action led to is only here.
        "q_end": q.tolist(),
        "q_dot_end": q_dot.tolist(),
    }
    if goal_errors is not None:
        # Start errors are against the first goal; end errors against whichever
        # goal was active when the episode stopped.
        summary["goal_errors_start"] = goal_errors(q_start, v_start)
        summary["goal_errors_end"] = goal_errors(q, q_dot)
    if plan_ms:
        # The first plan of a run pays kernel compilation and CUDA-graph
        # capture — often 100x the steady-state cost — so it is reported
        # separately rather than being averaged into a figure that then
        # describes neither.
        rest = plan_ms[1:] or plan_ms
        summary.update(
            plan_ms_first=float(plan_ms[0]),
            plan_ms_mean=float(np.mean(rest)),
            plan_ms_max=float(np.max(rest)),
        )
    return summary


def _printEpisode(episode: int, s: dict, args) -> None:
    """One episode's result, as printed after it finishes."""
    outcome = {"success": "SUCCESS", "failure": "FAILED", "timeout": "timeout"}[s["end_reason"]]
    where = (f" at step {s['steps_to_success']}" if s["success"]
             else f" at step {s['steps_taken']}" if s["failed"] else "")
    print(f"\nepisode {episode}: {outcome}{where}  ({s['steps_taken']}/{args.steps} steps)")
    print(f"  goals {' -> '.join(s['goals'])}  ({s['goals_reached']} reached)")
    if "goal_errors_end" in s:
        e0, e1 = s["goal_errors_start"], s["goal_errors_end"]
        print(f"  pos {e0['pos']:.4f} -> {e1['pos']:.4f} m   "
              f"quat {e0['quat']:.4f} -> {e1['quat']:.4f}   "
              f"vel {e1['vel']:.4f}")
    if "plan_ms_mean" in s:
        # Only the run's very first plan compiles kernels and captures the
        # graph; later episodes reuse both.
        note = " (compile + graph capture)" if episode == 0 else ""
        print(f"  plan {s['plan_ms_mean']:.1f} ms mean, {s['plan_ms_max']:.1f} ms max "
              f"({1e3 / s['plan_ms_mean']:.0f} Hz), first {s['plan_ms_first']:.0f} ms{note}")
    if s.get("video"):
        print(f"  video {s['video']}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.settle < 0:
        parser.error(f"--settle must be >= 0, got {args.settle}")

    task_cls = TASKS[args.task]
    # Two views of the same task: the scene the planner predicts with, and the
    # accurate one it is scored on.
    # Goals are sampled on the eval task (seeded, so a run is reproducible) and
    # copied to the rollout task; see _newGoal.
    k = args.eval_steps_per_rollout_step
    task = task_cls(LeapReorientConfig(
        role=TaskRole.ROLLOUT, timestep=args.timestep, hand_acc=args.hand_acc,
        obj_acc=args.obj_acc, eval_steps_per_rollout_step=k,
        goal_difficulty=args.goal_difficulty,
    ))
    eval_task = task_cls(LeapReorientConfig(
        role=TaskRole.EVAL, timestep=args.timestep, eval_steps_per_rollout_step=k,
        goal_difficulty=args.goal_difficulty, seed=args.seed,
    ))

    # Each simulator at its own task's timestep: the eval sim fine, the rollout
    # model coarse (they are equal when k == 1).
    sim = makeEvalSim(args.eval_sim, eval_task.getModelPath(), eval_task.timestep)
    rollout_sim = VectorizedMujoco(
        task.getModelPath(),
        VectorizedMujocoConfig(
            timestep=task.timestep, substeps=args.substeps, horizon=args.horizon,
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
        debug=args.debug, return_uncertainty=args.uncertainty,
    ))
    recorder = EpisodeRecorder(eval_task, sim, planner, cli_args=vars(args),
                               eval_sim=args.eval_sim)

    control_hz = 1.0 / (task.timestep * args.substeps)
    print(f"task       {args.task}  (rollout {args.hand_acc}_{args.obj_acc})")
    print(f"eval scene {Path(eval_task.getModelPath()).name}  (on {args.eval_sim})")
    print(f"rollout    {Path(task.getModelPath()).name}")
    print(f"planner    {planner!r}")
    print(f"timesteps  eval {eval_task.timestep * 1e3:g} ms, rollout "
          f"{task.timestep * 1e3:g} ms ({k} eval steps per rollout step)")
    print(f"control    {control_hz:.1f} Hz  ({args.steps} steps = "
          f"{args.steps / control_hz:.2f} s per episode)")

    summaries = []
    try:
        for episode in range(args.n_episodes):
            renderer = None
            video_path = None
            if args.video:
                video_path = args.video
                if args.n_episodes > 1:
                    p = Path(args.video)
                    video_path = str(p.with_name(f"{p.stem}_ep{episode}{p.suffix}"))
                # The CLI's own settings, then whatever the task needs (its
                # camera), then any CLI override of the task's choice — in that
                # order, so an explicit --camera wins over the task.
                cfg = VideoRendererBaseConfig(width=args.width, height=args.height, fps=args.fps)
                eval_task.alignRendererConfigWithTask(cfg)
                if args.camera is not None:
                    cfg.cam_name = None if args.camera.lower() == "none" else args.camera
                renderer = MujocoVideoRenderer(eval_task, cfg)
            try:
                s = run_episode(episode, args, task, eval_task, sim, planner, renderer, recorder)
                if renderer is not None:
                    s["video"] = str(renderer.Save(video_path))
            finally:
                if renderer is not None:
                    renderer.Close()
            recorder.episodeFinished(s["end_reason"], **s)

            summaries.append(s)
            _printEpisode(episode, s, args)

        if args.n_episodes > 1:
            n_ok = sum(s["success"] for s in summaries)
            n_fail = sum(s["failed"] for s in summaries)
            print(f"\n{args.n_episodes} episodes: {n_ok} succeeded, {n_fail} failed, "
                  f"{args.n_episodes - n_ok - n_fail} timed out  "
                  f"(success rate {n_ok / args.n_episodes:.0%})")
    finally:
        # Also on an error or Ctrl-C, so the episodes already finished are kept.
        # An episode cut off part-way is closed as "interrupted" rather than lost.
        if recorder.recording:
            recorder.episodeFinished("interrupted")
        if args.results and len(recorder):
            print(f"\nresults {recorder.Save(args.results)}  ({len(recorder)} episodes)")

    rollout_sim.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
