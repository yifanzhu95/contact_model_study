"""Run several episodes at once: GPU planner processes serving CPU eval-sim workers.

The engine behind ``run_episodes_interwoven.py`` and ``run_episodes_pooled.py``.

Why
---
``run_episodes.py`` alternates a GPU plan with a CPU eval-sim step, so one of
the two is always idle. Running several episodes at once overlaps them: while
one episode waits for its plan, others step their eval simulators, and while
those step, the GPU plans for someone else.

How
---
* **Planner processes**, one or more per GPU. Each holds a rollout simulator
  and an MPPI planner, and serves plan requests from any episode. A planner
  keeps nothing between requests: each request carries its episode's planner
  state (``SaveState``: the mean sequence, the noise stream, the adaptive
  temperature) and its current goal, and the reply carries the updated state
  back. So any free planner can serve any episode.
* **Worker processes**, one per concurrent episode. Each owns an eval
  simulator and runs episodes with exactly the loop ``run_episodes.py`` uses
  (``runRecordedEpisode``), handing it a ``RemotePlanner`` whose ``Plan`` sends
  a request and waits for the reply.
* **The scheduler** is two shared queues. Free workers take the next episode;
  free planners take the next plan request. Nothing is assigned ahead of time,
  so a slow episode (a Pinocchio step, a long settle) never holds up the rest.
* **The main process** starts everything, collects each finished episode into
  one ``EpisodeRecorder``, prints it, and saves one results file, as
  ``run_episodes.py`` does.

Episodes are reproducible by their index: episode ``k``'s goals and planner
noise come from seeds derived from ``(--seed, k)``, not from what ran before it.
So a run's results do not depend on how many workers or planners it had, up to
MJWarp's own run-to-run nondeterminism. Unlike ``run_episodes.py``, where episode
``k``'s goals follow on from episode ``k-1``'s, results are not episode-for-
episode identical to a sequential run.

Processes are started with ``spawn``: a process holding a CUDA context must not
be forked. Everything heavy is imported inside the process functions, after each
planner process has chosen its GPU with ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass

import numpy as np

#: How often a waiting process checks whether the run has been aborted (s).
_POLL_S = 0.5


# -- seeds ---------------------------------------------------------------------
def episodeSeeds(seed: int | None, episode: int) -> tuple[int, int]:
    """``(goal seed, noise seed)`` for one episode, from the run seed and its index."""
    ss = np.random.SeedSequence([0 if seed is None else int(seed), int(episode)])
    goal, noise = ss.generate_state(2)
    return int(goal), int(noise % (2**31 - 1))


# -- resources -------------------------------------------------------------------
def visibleGpus() -> list[str]:
    """GPU ids this job may use: ``CUDA_VISIBLE_DEVICES`` if set, else every GPU found."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g.strip() for g in env.split(",") if g.strip() and g.strip() != "-1"]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
        n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
        return [str(i) for i in range(n)]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def availableCpus() -> int:
    """Cores this process may actually use (respects cgroups and SLURM_CPUS_PER_TASK)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        n = min(n, int(slurm))
    return max(1, n)


# -- stand-ins for objects that live in another process --------------------------
@dataclass
class SimStandIn:
    """Describes a simulator held by another process, for the recorder."""

    class_name: str
    config: object
    N: int | None = None
    nq: int = 0
    nv: int = 0
    nu: int = 0


class _Aborted(Exception):
    """The run was stopped while this episode waited for a plan."""


class RemotePlanner:
    """A planner living in a planner process, as ``run_episode`` sees it.

    ``Plan`` sends the state, the current goal and this episode's planner
    state, and waits for the action. ``Reset`` is sent with the next plan
    request. It exposes what the recorder reads from a planner: ``config``,
    ``sim`` and ``task`` (the rollout ones), ``class_name`` and
    ``last_min_cost``, plus ``last_plan_seconds``, the solve time in the planner
    process, so waiting for a free GPU is not counted as planning.
    """

    def __init__(self, desc: dict, task, requests, replies, worker: int, abort):
        self.config = desc["planner_config"]
        self.class_name = desc["planner_class"]
        self.sim = SimStandIn(desc["sim_class"], desc["sim_config"], N=desc["N"])
        self.task = task
        #: What run_episode reads to pace the eval sim: rollout steps per control step.
        self.substeps = desc["sim_config"].resolved_substeps
        self.horizon = desc["sim_config"].resolved_horizon
        self.last_min_cost = float("nan")
        self.last_plan_seconds = None
        self.last_action_uncertainty = None
        self.wait_s = 0.0
        self._requests, self._replies = requests, replies
        self._worker, self._abort = worker, abort
        self._state = None
        self._reset, self._mean = True, None
        self._episode, self._noise_seed = None, None

    def startEpisode(self, episode: int, noise_seed: int) -> None:
        self._episode, self._noise_seed = episode, noise_seed
        self._state, self._reset, self._mean = None, True, None

    def Reset(self, mean=None) -> None:
        self._reset, self._mean = True, None if mean is None else np.asarray(mean, dtype=float)

    def Plan(self, q, q_dot, u=None):
        if self._abort.is_set():                # stop at the next control step
            raise _Aborted()
        req = {"worker": self._worker, "episode": self._episode, "q": np.asarray(q, float),
               "q_dot": np.asarray(q_dot, float), "u": None if u is None else np.asarray(u, float),
               "goal": self.task.getGoal(), "state": self._state, "reset": self._reset,
               "mean": self._mean, "noise_seed": self._noise_seed}
        t0 = time.perf_counter()
        self._requests.put(req)
        while True:
            try:
                rep = self._replies.get(timeout=_POLL_S)
                break
            except queue.Empty:
                if self._abort.is_set():
                    raise _Aborted()
        self.wait_s += time.perf_counter() - t0
        if rep.get("error"):
            raise RuntimeError(f"planner process failed: {rep['error']}")
        self._state, self._reset = rep["state"], False
        self.last_min_cost, self.last_plan_seconds = rep["min_cost"], rep["plan_s"]
        self.last_action_uncertainty = rep["sigma"]
        return rep["action"] if rep["sigma"] is None else (rep["action"], rep["sigma"])


# -- planner process -------------------------------------------------------------
def _ignoreInterrupts() -> None:
    """Leave Ctrl-C and SIGTERM to the main process.

    A terminal sends Ctrl-C to every process in the group, and SLURM's
    scancel signals them all too. The main process turns either into an
    orderly stop (the abort event), which is what lets workers close their
    episodes as "interrupted" instead of dying mid-step.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _plannerMain(args, gpu: str | None, planner_id: int, requests, replies, events, abort) -> None:
    """Serve plan requests until a ``None`` arrives."""
    _ignoreInterrupts()
    for r in replies:            # after a stop, unread replies must not block exit
        r.cancel_join_thread()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    busy, n_plans = 0.0, 0
    try:
        from ContactModelStudy.Drivers import run_episodes as drv
        from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI
        from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
        task, _ = drv.buildTasks(args)
        sim = GetContactModelSim(args.rollout_model, task.getModelPath(), N=args.n_samples,
                                 **drv.rolloutSimOverrides(args, task))
        planner = MPPI(sim, task, drv.buildPlannerConfig(args))
        events.put(("planner_ready", planner_id, gpu))
        goal = None
        while True:
            try:
                req = requests.get(timeout=_POLL_S)
            except queue.Empty:
                if abort.is_set():
                    break
                continue
            if req is None or abort.is_set():
                break
            t0 = time.perf_counter()
            try:
                if goal is None or not np.array_equal(goal, req["goal"]):
                    goal = np.asarray(req["goal"]).copy()
                    task.setGoal(goal)
                if req["state"] is None:              # a new episode
                    planner.Reset(req["mean"])
                    planner.LoadState({**planner.SaveState(), "u_prev": None, "last_action_seq": None,
                                       "noise_seed": req["noise_seed"], "resample_count": 0})
                else:
                    planner.LoadState(req["state"])
                    if req["reset"]:                  # a new goal mid-episode: fresh mean, same noise stream
                        planner.Reset(req["mean"])
                t1 = time.perf_counter()
                out = planner.Plan(req["q"], req["q_dot"], u=req["u"])
                plan_s = time.perf_counter() - t1
                action, sigma = out if isinstance(out, tuple) else (out, None)
                replies[req["worker"]].put({"action": action, "sigma": sigma, "plan_s": plan_s,
                                            "min_cost": float(getattr(planner, "last_min_cost", np.nan)),
                                            "state": planner.SaveState()})
                n_plans += 1
            except Exception as e:
                replies[req["worker"]].put({"error": f"{type(e).__name__}: {e}"})
            busy += time.perf_counter() - t0
        sim.Close()
    except BaseException as e:
        events.put(("planner_failed", planner_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        return
    events.put(("planner_stats", planner_id, busy, n_plans))


# -- worker process --------------------------------------------------------------
def _workerMain(args, worker_id: int, desc: dict, episodes, requests, reply, events, abort) -> None:
    """Take episodes until a ``None`` arrives; run each against the remote planners."""
    _ignoreInterrupts()
    requests.cancel_join_thread()  # after a stop, an unread request must not block exit
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        from ContactModelStudy.Drivers import run_episodes as drv
        from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder
        from ContactModelStudy.Utils.EvalSimulators import makeEvalSim
        task, eval_task = drv.buildTasks(args)
        sim = makeEvalSim(args.eval_sim, eval_task.getModelPath(), eval_task.timestep)
        planner = RemotePlanner(desc, task, requests, reply, worker_id, abort)
        recorder = EpisodeRecorder(eval_task, sim, planner, cli_args=vars(args), eval_sim=args.eval_sim)
        events.put(("worker_ready", worker_id))
    except BaseException as e:
        events.put(("worker_failed", worker_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        return
    busy = 0.0
    while not abort.is_set():
        try:
            ep = episodes.get(timeout=_POLL_S)
        except queue.Empty:
            continue
        if ep is None:
            break
        goal_seed, noise_seed = episodeSeeds(args.seed, ep)
        eval_task.reseed(goal_seed)
        planner.startEpisode(ep, noise_seed)
        wait0, t0 = planner.wait_s, time.perf_counter()
        error = None
        try:
            drv.runRecordedEpisode(ep, args, task, eval_task, sim, planner, recorder)
        except _Aborted:
            recorder.episodeFinished("interrupted", episode=ep, settle_s=args.settle)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            recorder.episodeFinished("error", episode=ep, settle_s=args.settle, error=error)
        busy += (time.perf_counter() - t0) - (planner.wait_s - wait0)
        summary = recorder.GenerateSummary(-1)
        episode = recorder.episodes.pop()
        episode.extra["worker"] = worker_id
        episode.extra["goal_seed"] = goal_seed
        events.put(("episode", worker_id, ep, episode, summary, error))
    events.put(("worker_stats", worker_id, busy))


# -- main process ----------------------------------------------------------------
def describeRollout(args, task) -> dict:
    """What a worker's ``RemotePlanner`` reports about the planner, built without a GPU."""
    from ContactModelStudy.Drivers import run_episodes as drv
    from ContactModelStudy.Utils.ContactModelPresets import CONTACT_MODELS, _presetConfig, _resolveName, _simulatorClasses
    name = _resolveName(args.rollout_model)
    sim_cls, _ = _simulatorClasses(CONTACT_MODELS[name]["simulator"])
    return {"planner_config": drv.buildPlannerConfig(args), "planner_class": "MPPI",
            "sim_class": sim_cls.__name__, "N": args.n_samples,
            "sim_config": _presetConfig(args.rollout_model, **drv.rolloutSimOverrides(args, task))}


def runPool(args, n_workers: int, gpus: list[str | None], planners_per_gpu: int = 1) -> int:
    """Run ``args.n_episodes`` episodes with ``n_workers`` workers and planners on ``gpus``."""
    from ContactModelStudy.Drivers import run_episodes as drv
    from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder
    from ContactModelStudy.Utils.EvalSimulators import evalSimConfig

    # Line-buffered, so a log file (an HPC job's, say) shows each episode as it
    # finishes rather than in large blocks.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    task, eval_task = drv.buildTasks(args)
    desc = describeRollout(args, task)
    n_workers = max(1, min(n_workers, args.n_episodes))
    planner_gpus = [g for g in gpus for _ in range(planners_per_gpu)]
    drv.printHeader(args, task, eval_task, desc["sim_config"], desc["sim_class"],
                    f"MPPI(N={args.n_samples}), {len(planner_gpus)} planner process(es)")
    print(f"pool       {n_workers} eval workers ({args.eval_sim}, CPU), {len(planner_gpus)} planners "
          f"on GPU(s) {', '.join(str(g) for g in gpus)}")

    # The collected results: every episode a worker finishes lands here.
    eval_cls, eval_cfg = evalSimConfig(args.eval_sim, eval_task.timestep)
    sim_desc = SimStandIn(eval_cls, eval_cfg, nq=eval_task.nq, nv=eval_task.nv, nu=eval_task.nu)
    planner_desc = RemotePlanner(desc, task, None, None, -1, None)
    recorder = EpisodeRecorder(eval_task, sim_desc, planner_desc, cli_args=vars(args), eval_sim=args.eval_sim)

    ctx = mp.get_context("spawn")
    abort = ctx.Event()
    episodes, requests, events = ctx.Queue(), ctx.Queue(), ctx.Queue()
    replies = [ctx.Queue() for _ in range(n_workers)]
    planners = [ctx.Process(target=_plannerMain, name=f"planner{i}",
                            args=(args, g, i, requests, replies, events, abort), daemon=True)
                for i, g in enumerate(planner_gpus)]
    workers = [ctx.Process(target=_workerMain, name=f"worker{w}",
                           args=(args, w, desc, episodes, requests, replies[w], events, abort), daemon=True)
               for w in range(n_workers)]
    for ep in range(args.n_episodes):
        episodes.put(ep)
    for _ in workers:
        episodes.put(None)

    # SIGTERM (scancel, a job's time limit) stops the run like Ctrl-C does.
    def _terminate(signum, frame):
        raise KeyboardInterrupt
    previous_term = signal.signal(signal.SIGTERM, _terminate)

    t_start = time.time()
    for p in planners + workers:
        p.start()
    done: dict[int, dict] = {}
    stats = {"planner_busy": 0.0, "plans": 0, "worker_busy": 0.0, "planner_stats": 0}
    failed_procs: list[str] = []

    def handle(msg) -> None:
        """Act on one message from a planner or worker."""
        kind = msg[0]
        if kind == "episode":
            _, w, ep, episode, summary, error = msg
            if ep in done:
                return
            recorder.episodes.append(episode)
            done[ep] = summary
            drv._printEpisode(ep, summary, args)
            if error:
                print(f"  ERROR in worker {w}: {error}")
        elif kind in ("planner_failed", "worker_failed"):
            print(f"\n{kind.replace('_', ' ')} ({msg[1]}): {msg[2]}")
            failed_procs.append(f"{kind} {msg[1]}")
        elif kind == "worker_stats":
            stats["worker_busy"] += msg[2]
        elif kind == "planner_stats":
            stats["planner_busy"] += msg[2]
            stats["plans"] += msg[3]
            stats["planner_stats"] += 1

    def drainUntil(procs, deadline) -> None:
        """Keep reading messages until ``procs`` have exited or ``deadline`` passes.

        Reading while waiting matters: a child cannot exit until the messages
        it sent (a finished episode can be large) have been read.
        """
        while any(p.is_alive() for p in procs) and time.time() < deadline:
            try:
                handle(events.get(timeout=0.2))
            except queue.Empty:
                pass

    try:
        while len(done) < args.n_episodes:
            try:
                handle(events.get(timeout=1.0))
            except queue.Empty:
                if any(not p.is_alive() for p in planners) and not failed_procs:
                    failed_procs.append("a planner process exited unexpectedly")
                if all(not w.is_alive() for w in workers):
                    break
            if any(f.startswith("planner_failed") for f in failed_procs) \
                    or sum(f.startswith("worker_failed") for f in failed_procs) >= n_workers \
                    or "a planner process exited unexpectedly" in failed_procs:
                break
    except KeyboardInterrupt:
        print("\ninterrupted: closing the episodes in progress as 'interrupted' ...")
        failed_procs.append("interrupted")
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        wall = time.time() - t_start
        if failed_procs:
            abort.set()
        # Workers stop once the episode queue is empty (or at the next control
        # step after an abort); then the planners are told to stop.
        drainUntil(workers, time.time() + (30 if failed_procs else 120))
        for _ in planners:
            requests.put(None)
        drainUntil(planners, time.time() + 30)
        while True:
            try:
                handle(events.get(timeout=0.2))
            except queue.Empty:
                break
        for p in planners + workers:
            if p.is_alive():
                p.terminate()
        recorder.episodes.sort(key=lambda e: e.extra.get("episode", 0))
        if len(recorder) > 1:
            drv.printBatch(recorder)
        n = len(done)
        if n:
            print(f"\n{n} episodes in {wall:.1f} s ({wall / n:.1f} s per episode, "
                  f"{n * 3600 / wall:.0f} per hour)")
            if stats["planner_stats"]:
                print(f"GPU planners busy {100 * stats['planner_busy'] / (wall * len(planners)):.0f}% "
                      f"({stats['plans']} plans), CPU workers busy "
                      f"{100 * stats['worker_busy'] / (wall * len(workers)):.0f}%")
        drv.saveResults(args, recorder)
    if failed_procs:
        print("\nstopped early: " + "; ".join(failed_procs))
        return 1
    return 0
