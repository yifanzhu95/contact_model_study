#!/usr/bin/env python3
"""Run closed-loop MPC episodes and report what happened.

Wires the package together: a task supplies the scene, the initial state and
the cost; one of the study's rollout contact models (M1-M4, see
``Utils.ContactModelPresets``) rolls samples out on the GPU; ``MPPI`` plans
against it; a CPU simulator stands in for reality; a ``MujocoVideoRenderer``
films it; and an ``EpisodeRecorder`` keeps every step and owns the summaries.

The planner and the evaluator load *different scenes from the same task* — the
rollout scene the planner predicts with, and the accurate eval scene it is
judged on. That gap is what the study measures, so it is the default here
rather than something to switch on.

Examples::

    python -m ContactModelStudy.Drivers.run_episodes
    python -m ContactModelStudy.Drivers.run_episodes --n-episodes 5 --steps 200
    python -m ContactModelStudy.Drivers.run_episodes --hand-acc low --no-video
    python -m ContactModelStudy.Drivers.run_episodes --rollout-model M3
    python -m ContactModelStudy.Drivers.run_episodes --n-samples 512 --temperature 5 \\
        --results results/run.json

Results are saved by default, to ``results/run_episodes_<date>_<time>.json``;
``--results PATH`` picks the file and ``--no-results`` turns saving off. That
is the ``EpisodeRecorder`` output: the JSON (configs and one summary per
episode) and, unless ``--no-save-steps``, one ``<stem>_<id>.npy`` per episode
holding every control step's state, action, action uncertainty and planning
time. Read it back with ``ContactModelStudy.Utils.EpisodeRecorder.EpisodeReplayer``.

Offscreen rendering needs a GL context; with no DISPLAY this defaults
MUJOCO_GL to "egl".
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

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
from ContactModelStudy.Tasks.BallReorient import BallReorient  # noqa: E402
from ContactModelStudy.Tasks.CubeReorient import CubeReorient  # noqa: E402
from ContactModelStudy.Tasks.DuckReorient import DuckReorient  # noqa: E402
from ContactModelStudy.Tasks.LeapReorient import COST_WEIGHT_KEYS, LeapReorientConfig  # noqa: E402
from ContactModelStudy.Tasks.TaskBase import TaskRole  # noqa: E402
from ContactModelStudy.Utils.ContactModelPresets import (  # noqa: E402
    ALIASES,
    CONTACT_MODELS,
    GetContactModelSim,
)
from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder  # noqa: E402
from ContactModelStudy.Utils.EvalSimulators import EVAL_SIMS, makeEvalSim  # noqa: E402

#: Tasks this driver can run, by the name passed to --task.
TASKS = {"cube_reorient": CubeReorient, "duck_reorient": DuckReorient,
         "ball_reorient": BallReorient}

#: --results default: a fresh, time-stamped file under results/, chosen at start.
_AUTO_RESULTS = "auto"

#: Used when neither flag of a pair is given, as (flag to fill in, value):
#: --substeps / --ctrl-time-step and --horizon / --time-horizon. Giving either
#: flag of a pair replaces the default; giving both is an error. The flags
#: themselves default to "not given", so the default of one form never
#: collides with the other form given on the command line.
_PAIR_DEFAULTS = {
    ("substeps", "ctrl_time_step"): ("ctrl_time_step", 0.064),
    ("horizon", "time_horizon"): ("time_horizon", 0.352),
}


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
    g.add_argument("--steps", type=int, default=1000,
                   help="control steps per episode")
    g.add_argument("--timestep", type=float, default=0.0005,
                   help="eval (fine) physics timestep (s); the rollout model runs "
                        "at timestep * --eval-steps-per-rollout-step")
    g.add_argument("--eval-steps-per-rollout-step", type=int, default=8,
                   help="eval steps per rollout step (the old task used 8, with "
                        "a 0.5 ms eval timestep)")
    g.add_argument("--substeps", type=int, default=None,
                   help=f"rollout steps per control step; control rate is "
                        f"1/(rollout timestep * substeps). Give this or "
                        f"--ctrl-time-step; neither means --ctrl-time-step "
                        f"{_PAIR_DEFAULTS[('substeps', 'ctrl_time_step')][1]}")
    g.add_argument("--ctrl-time-step", type=float, default=None,
                   help="control-step duration (s), rounded down to whole rollout "
                        "timesteps; the alternative to --substeps")
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
    g.add_argument("--cost-weight", action="append", default=None, metavar="NAME=VALUE",
                   help="override one of the object's cost weights, e.g. "
                        "--cost-weight w_quat=30; repeat for more. Names: "
                        + ", ".join(COST_WEIGHT_KEYS))
    g.add_argument("--stop-on-success", action=argparse.BooleanOptionalAction, default=True,
                   help="end the episode at the first success (the old driver's "
                        "default); with --no-stop-on-success each success samples "
                        "a new goal and the episode carries on to --steps")

    g = p.add_argument_group("planner")
    g.add_argument("--rollout-model", default="M2",
                   choices=sorted(CONTACT_MODELS) + sorted(ALIASES), metavar="MODEL",
                   help="contact model the planner rolls out with: M1 (stiff MJWarp), "
                        "M2 (MJWarp soft, default), M3 (ComFree), M4 (XPBD)")
    g.add_argument("--n-samples", type=int, default=64,
                   help="N: rollout worlds, one per sampled control sequence")
    g.add_argument("--horizon", type=int, default=None,
                   help=f"H: planning horizon in control steps. Give this or "
                        f"--time-horizon; neither means --time-horizon "
                        f"{_PAIR_DEFAULTS[('horizon', 'time_horizon')][1]}")
    g.add_argument("--time-horizon", type=float, default=None,
                   help="planning horizon (s), rounded down to whole control steps; "
                        "the alternative to --horizon")
    g.add_argument("--n-iterations", type=int, default=1)
    g.add_argument("--noise-sigma", type=float, default=0.2)
    g.add_argument("--temperature", type=float, default=40.0,
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
                   help="max contacts per rollout world; omitted, the rollout "
                        "config's default is used")
    g.add_argument("--njmax", type=int, default=None,
                   help="max constraint rows per rollout world; omitted, the rollout "
                        "config's default is used (MJWarp's own choice, 64 for the "
                        "leap scenes, overflows in a grasp)")

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
    g.add_argument("--results", default="results/run_episodes.json",
                   help="save the recorded episodes: a JSON summary here, plus one "
                        ".npy of per-step data per episode beside it. Defaults to "
                        "results/run_episodes_<date>_<time>.json")
    g.add_argument("--no-results", dest="results", action="store_const", const=None,
                   help="save nothing")
    g.add_argument("--save-steps", action=argparse.BooleanOptionalAction, default=False,
                   help="save every step's state, action and uncertainty (the .npy "
                        "files); with --no-save-steps only the JSON summary is saved")
    g.add_argument("--uncertainty", action=argparse.BooleanOptionalAction, default=False,
                   help="have the planner report, and the recorder keep, the "
                        "uncertainty of each action")
    g.add_argument("--debug", action="store_true", help="per-plan MPPI diagnostics")
    return p


def _newGoal(task, eval_task, renderer, recorder) -> None:
    """Sample one goal and give it to both tasks, the renderer and the recorder.

    The two task instances load different scenes but must chase one goal, so it
    is sampled once — on the eval task, which owns the episode and its seed —
    and set on both. Each new goal is a rotation of the previous one. The
    renderer keeps its own copy of the scene, so it is told too, or the video's
    goal marker would never move.
    """
    goal = eval_task.sampleNewGoal()
    eval_task.setGoal(goal)
    task.setGoal(goal)
    recorder.recordGoal(goal)
    if renderer is not None:
        eval_task.setRendererToGoal(renderer)


def run_episode(args, task, eval_task, sim, planner, renderer, recorder) -> str:
    """Run one closed-loop episode, reporting it to ``recorder`` as it goes.

    Every control step's state, action, action uncertainty (when the planner
    reports one) and planning time go to the recorder, as do each goal adopted
    and each goal reached. The episode is left open; the caller finishes it
    once the video is saved, so the video's path can go in too. The recorder
    builds the summary; see ``EpisodeRecorder.GenerateSummary``.

    The planner sees the eval simulator's state and returns an absolute command;
    the eval simulator holds that command for ``substeps`` rollout steps.

    The episode is judged with the *eval* task's ``isSuccess`` / ``isFailure``,
    evaluated on the start-of-step state as the old driver did. A failure always
    ends it. A success ends it too unless ``--no-stop-on-success``, in which
    case a fresh goal is sampled and the episode carries on — the old driver's
    multi-goal mode.

    Returns:
        Why the episode ended: ``"success"``, ``"failure"`` or ``"timeout"``.
        A multi-goal episode that reached goals and then ran out of steps ends
        in ``"timeout"``; its summary still reports it as a success.
    """
    eval_task.setSimToInitialState(sim)
    _, _, u0 = eval_task.getInitialState()
    # Under absolute control the mean IS the command, so a zero mean would open
    # the hand and drop the object on the first plan; seed it with the task's
    # initial grasp. Under both relative modes the mean is a delta, and zero
    # correctly means "hold" — the pose, or the command.
    reset_mean = u0 if args.control_mode == "absolute" else None
    _newGoal(task, eval_task, renderer, recorder)
    planner.Reset(reset_mean)
    if renderer is not None:
        renderer.Reset()

    # One control step advances the eval simulator this many of its own (fine)
    # steps: the planner's (resolved) substeps rollout steps, each
    # eval_steps_per_rollout_step long.
    eval_steps = planner.substeps * eval_task.config.eval_steps_per_rollout_step
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
    _advance(int(round(args.settle / eval_task.timestep)))

    end_reason = "timeout"
    for _ in range(args.steps):
        if eval_task.isFailure(sim):
            end_reason = "failure"
            break
        if eval_task.isSuccess(sim):
            recorder.recordSuccess()
            if args.stop_on_success:
                end_reason = "success"
                break
            _newGoal(task, eval_task, renderer, recorder)
            planner.Reset(reset_mean)

        q, q_dot = sim.GetState()
        t0 = time.perf_counter()
        # The applied control is passed so ctrl_relative always accumulates from
        # what the hand is actually commanded — at step 0 the task's initial
        # grasp, not whatever the planner returned last episode.
        out = planner.Plan(q, q_dot, u=sim.GetControl())
        dt = time.perf_counter() - t0
        u, sigma = out if isinstance(out, tuple) else (out, None)
        recorder.recordStateAndAction(q, q_dot, u, sigma, planning_time=dt)

        sim.SetControl(u)
        _advance(eval_steps)

    # End on the final state, unless that state was itself a frame deadline and
    # is already the last frame — a duplicate would stretch playback by a frame.
    if renderer is not None and eval_clock[0] % steps_per_frame != 0:
        renderer.RenderState(sim.GetState()[0])
    return end_reason


def _printEpisode(episode: int, s: dict, args) -> None:
    """One episode's result, from the recorder's summary of it."""
    outcome = ("SUCCESS" if s["success"] else "FAILED" if s["failed"]
               else s["finish_reason"])
    where = (f" at step {s['steps_to_success']}" if s["success"]
             else f" at step {s['n_steps']}" if s["failed"] else "")
    print(f"\nepisode {episode}: {outcome}{where}  ({s['n_steps']}/{args.steps} steps)")
    print(f"  goals {' -> '.join(g or '?' for g in s['goals'])}  ({s['goals_reached']} reached)")
    e0, e1 = s.get("goal_errors_start"), s.get("goal_errors_end")
    if e0 and e1:
        print(f"  pos {e0['pos']:.4f} -> {e1['pos']:.4f} m   "
              f"quat {e0['quat']:.4f} -> {e1['quat']:.4f}   "
              f"vel {e1['vel']:.4f}")
    if s.get("plan_s_mean"):
        # Only the run's very first plan compiles kernels and captures the
        # graph; later episodes reuse both.
        note = " (compile + graph capture)" if episode == 0 else ""
        mean_ms, max_ms, first_ms = (1e3 * s[k] for k in ("plan_s_mean", "plan_s_max", "plan_s_first"))
        print(f"  plan {mean_ms:.1f} ms mean, {max_ms:.1f} ms max "
              f"({1e3 / mean_ms:.0f} Hz), first {first_ms:.0f} ms{note}")
    if s.get("video"):
        print(f"  video {s['video']}")


def parseArgs(argv=None) -> argparse.Namespace:
    """Parse and check the command line, filling in the defaults that depend on it.

    Everything ``main`` rejects before building anything is rejected here, as
    a parser error (``SystemExit``), so a caller can validate a command line
    without running it. ``args.cost_weights`` holds the parsed
    ``--cost-weight`` overrides as a dict, or ``None``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    args.cost_weights = None
    if args.cost_weight:
        args.cost_weights = {}
        for item in args.cost_weight:
            name, sep, value = item.partition("=")
            if not sep:
                parser.error(f"--cost-weight wants NAME=VALUE, got {item!r}")
            if name not in COST_WEIGHT_KEYS:
                parser.error(f"--cost-weight: unknown weight {name!r}; names are "
                             + ", ".join(COST_WEIGHT_KEYS))
            try:
                args.cost_weights[name] = float(value)
            except ValueError:
                parser.error(f"--cost-weight {name}: {value!r} is not a number")
    if args.settle < 0:
        parser.error(f"--settle must be >= 0, got {args.settle}")
    # Each pair is one setting given two ways; the rollout config rejects both
    # being set, but saying so here names the flags rather than the fields.
    for (a, b), (field, value) in _PAIR_DEFAULTS.items():
        if getattr(args, a) is not None and getattr(args, b) is not None:
            parser.error(f"give --{a.replace('_', '-')} or --{b.replace('_', '-')}, not both")
        if getattr(args, a) is None and getattr(args, b) is None:
            setattr(args, field, value)
    if args.results == _AUTO_RESULTS:
        # Time-stamped so runs never overwrite one another, and so the per-step
        # files of different runs, which sit side by side, never mix.
        args.results = str(_REPO_ROOT / "results" / f"run_episodes_{time.strftime('%Y%m%d_%H%M%S')}.json")
    return args


def main(argv=None) -> int:
    args = parseArgs(argv)

    task_cls = TASKS[args.task]
    # Two views of the same task: the scene the planner predicts with, and the
    # accurate one it is scored on.
    # Goals are sampled on the eval task (seeded, so a run is reproducible) and
    # copied to the rollout task; see _newGoal.
    k = args.eval_steps_per_rollout_step
    task = task_cls(LeapReorientConfig(
        role=TaskRole.ROLLOUT, timestep=args.timestep, hand_acc=args.hand_acc,
        obj_acc=args.obj_acc, eval_steps_per_rollout_step=k,
        goal_difficulty=args.goal_difficulty, cost_weights=args.cost_weights,
    ))
    eval_task = task_cls(LeapReorientConfig(
        role=TaskRole.EVAL, timestep=args.timestep, eval_steps_per_rollout_step=k,
        goal_difficulty=args.goal_difficulty, seed=args.seed, cost_weights=args.cost_weights,
    ))

    # Each simulator at its own task's timestep: the eval sim fine, the rollout
    # model coarse (they are equal when k == 1).
    sim = makeEvalSim(args.eval_sim, eval_task.getModelPath(), eval_task.timestep)
    # Buffer sizes are passed only when given on the command line: passing
    # None would override the config's defaults with "let MJWarp choose".
    buffers = {k: v for k, v in (("nconmax", args.nconmax), ("njmax", args.njmax))
               if v is not None}
    rollout_sim = GetContactModelSim(
        args.rollout_model, task.getModelPath(), N=args.n_samples,
        timestep=task.timestep, substeps=args.substeps, ctrl_time_step=args.ctrl_time_step,
        horizon=args.horizon, time_horizon=args.time_horizon, **buffers,
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

    rcfg = rollout_sim.config
    control_hz = 1.0 / rcfg.control_timestep
    print(f"task       {args.task}  (rollout {args.hand_acc}_{args.obj_acc})")
    print(f"eval scene {Path(eval_task.getModelPath()).name}  (on {args.eval_sim})")
    print(f"rollout    {Path(task.getModelPath()).name}  (contact model {args.rollout_model}: "
          f"{type(rollout_sim).__name__})")
    print(f"planner    {planner!r}")
    print(f"timesteps  eval {eval_task.timestep * 1e3:g} ms, rollout "
          f"{task.timestep * 1e3:g} ms ({k} eval steps per rollout step)")
    print(f"control    {control_hz:.1f} Hz: {rcfg.resolved_substeps} rollout steps = "
          f"{rcfg.control_timestep * 1e3:g} ms per control step  ({args.steps} steps = "
          f"{args.steps / control_hz:.2f} s per episode)")
    print(f"horizon    {rcfg.resolved_horizon} control steps = {rcfg.horizon_duration * 1e3:g} ms")

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
            extra = {"episode": episode, "settle_s": args.settle}
            try:
                end_reason = run_episode(args, task, eval_task, sim, planner, renderer, recorder)
                if renderer is not None:
                    extra["video"] = str(renderer.Save(video_path))
            finally:
                if renderer is not None:
                    renderer.Close()
            recorder.episodeFinished(end_reason, **extra)
            _printEpisode(episode, recorder.GenerateSummary(-1), args)

        if args.n_episodes > 1:
            b = recorder.GenerateSummary()
            n = b["n_episodes"]
            print(f"\n{n} episodes: {b['n_success']} succeeded, {b['n_failed']} failed, "
                  f"{n - b['n_success'] - b['n_failed']} timed out  "
                  f"(success rate {b['success_rate']:.0%})")
    finally:
        # Also on an error or Ctrl-C, so the episodes already finished are kept.
        # An episode cut off part-way is closed as "interrupted" rather than lost.
        if recorder.recording:
            recorder.episodeFinished("interrupted")
        if args.results and len(recorder):
            out = recorder.Save(args.results, save_steps=args.save_steps)
            what = "" if args.save_steps else ", summary only"
            print(f"\nresults {out}  ({len(recorder)} episodes{what})")

    rollout_sim.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
