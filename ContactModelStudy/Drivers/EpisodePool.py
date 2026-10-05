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
* **The main process** owns an ``EpisodePool``: it submits ``EpisodeJob``s and
  polls for finished episodes. ``runPool`` (what the two drivers call) submits
  ``--n-episodes`` jobs, collects them into one ``EpisodeRecorder``, prints
  them, and saves one results file, as ``run_episodes.py`` does.

The pool outlives a batch: a job can name its own rollout model (from the
models the pool was built for), planner params (``temperature``,
``noise_sigma``) and cost weights, which the planner process applies in place
before each plan. So a search (``experiments/run_bayes_opt.py``) keeps one pool
for its whole run instead of starting processes and capturing graphs per trial.

Episodes are reproducible by their index: episode ``k``'s goals and planner
noise come from seeds derived from ``(--seed, k)``, not from what ran before it.
So a run's results do not depend on how many workers or planners it had, up to
MJWarp's own run-to-run nondeterminism. Unlike ``run_episodes.py``, where episode
``k``'s goals follow on from episode ``k-1``'s, results are not episode-for-
episode identical to a sequential run.

**GPU eval simulators.** With ``--eval-sim M1``-``M4`` the eval simulator is
a contact model on the GPU, so each worker claims a GPU too (round-robin over
the pool's) and holds its own CUDA context (about half a GB). Its steps then
share the GPU with the planners: fewer workers than cores is usually right, and
the end-of-run busy line shows the balance.

Processes are started with ``spawn``: a process holding a CUDA context must not
be forked. Everything heavy is imported inside the process functions, after each
planner process has chosen its GPU with ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, replace

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


@dataclass
class EpisodeJob:
    """One episode for the pool to run, and the settings it runs under.

    Attributes:
        episode: The episode index. Its goals and planner noise come from
            ``(--seed, episode)`` alone, so two jobs with the same index see the
            same episode whatever their other settings.
        rollout_model: The contact model to plan with; one the pool was built
            for. ``None`` means ``--rollout-model``.
        planner_params: Planner config fields to change for this episode, from
            the planner's ``RUNTIME_FIELDS`` (``temperature``, ``noise_sigma``).
            Fields left out keep the command line's value.
        cost_weights: Cost weights to override for this episode, by name.
            Weights left out keep the object's (or ``--cost-weight``'s) value.
        key: Anything the caller wants back with the result.
    """

    episode: int
    rollout_model: str | None = None
    planner_params: dict | None = None
    cost_weights: dict | None = None
    key: object = None


@dataclass
class PoolResult:
    """A finished episode: the job, the recorded episode, its summary, any error."""

    job: EpisodeJob
    episode: object
    summary: dict
    error: str | None
    worker: int


class RemotePlanner:
    """A planner living in a planner process, as ``run_episode`` sees it.

    ``Plan`` sends the state, the current goal, this episode's planner state
    and its settings (model, planner params, cost weights), and waits for the
    action. ``Reset`` is sent with the next plan request. It exposes what the
    recorder reads from a planner: ``config``, ``sim`` and ``task`` (the
    rollout ones, as set for the current job), ``class_name`` and
    ``last_min_cost``, plus ``last_plan_seconds``, the solve time in the
    planner process, so waiting for a free GPU is not counted as planning.
    """

    def __init__(self, desc: dict, task, requests, replies, worker: int, abort):
        self.task = task
        self._desc, self._model, self._params, self._weights = None, None, None, None
        self._useDesc(desc)
        self.last_min_cost = float("nan")
        self.last_plan_seconds = None
        self.last_action_uncertainty = None
        self.wait_s = 0.0
        self._requests, self._replies = requests, replies
        self._worker, self._abort = worker, abort
        self._state = None
        self._reset, self._mean = True, None
        self._episode, self._noise_seed = None, None

    def _useDesc(self, desc: dict, params: dict | None = None) -> None:
        self._desc = desc
        self.config = desc["planner_config"] if not params else replace(desc["planner_config"], **params)
        self.class_name = desc["planner_class"]
        self.sim = SimStandIn(desc["sim_class"], desc["sim_config"], N=desc["N"])
        #: What run_episode reads to pace the eval sim: rollout steps per control step.
        self.substeps = desc["sim_config"].resolved_substeps
        self.horizon = desc["sim_config"].resolved_horizon

    def useJob(self, desc: dict, job: EpisodeJob, base_weights: dict) -> None:
        """Take on ``job``'s model and settings, so plans and the recorder see them."""
        self._useDesc(desc, job.planner_params)
        self._model, self._params = job.rollout_model, job.planner_params
        if (job.cost_weights or None) != self._weights:
            # Only on a change, so a run that never sets weights leaves the task as built.
            self.task.setCostWeights({**base_weights, **(job.cost_weights or {})})
            self._weights = job.cost_weights or None

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
               "mean": self._mean, "noise_seed": self._noise_seed, "model": self._model,
               "params": self._params, "weights": self._weights}
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


class _PlannerUnit:
    """One rollout model in a planner process: its task, simulator and planner."""

    def __init__(self, args, model: str):
        from ContactModelStudy.Drivers import run_episodes as drv
        from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI
        from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
        self.task, _ = drv.buildTasks(args)
        self.sim = GetContactModelSim(model, self.task.getModelPath(), N=args.n_samples,
                                      **drv.rolloutSimOverrides(args, self.task))
        self.planner = MPPI(self.sim, self.task, drv.buildPlannerConfig(args))
        self.base_config = self.planner.config
        self.base_weights = dict(self.task.params["cost_weights"])
        self.goal, self.settings = None, ((), ())

    def apply(self, params: dict | None, weights: dict | None) -> None:
        """Switch to a job's settings, if they differ from the last request's."""
        settings = (tuple(sorted((params or {}).items())), tuple(sorted((weights or {}).items())))
        if settings == self.settings:
            return
        # From the base each time, so a field one job set does not leak into the next.
        runtime = {f: getattr(self.base_config, f) for f in self.planner.RUNTIME_FIELDS}
        self.planner.UpdateConfig(**{**runtime, **(params or {})})
        self.task.setCostWeights({**self.base_weights, **(weights or {})})
        self.settings = settings


def _plannerMain(args, gpu: str | None, planner_id: int, models: list[str],
                 requests, replies, events, abort) -> None:
    """Serve plan requests, for any of ``models``, until a ``None`` arrives."""
    _ignoreInterrupts()
    for r in replies:            # after a stop, unread replies must not block exit
        r.cancel_join_thread()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    busy, n_plans = 0.0, 0
    try:
        units = {m: _PlannerUnit(args, m) for m in models}
        events.put(("planner_ready", planner_id, gpu))
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
                unit = units[req["model"] or models[0]]
                planner = unit.planner
                unit.apply(req["params"], req["weights"])
                if unit.goal is None or not np.array_equal(unit.goal, req["goal"]):
                    unit.goal = np.asarray(req["goal"]).copy()
                    unit.task.setGoal(unit.goal)
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
        for unit in units.values():
            unit.sim.Close()
    except BaseException as e:
        events.put(("planner_failed", planner_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        return
    events.put(("planner_stats", planner_id, busy, n_plans))


# -- worker process --------------------------------------------------------------
def _workerMain(args, worker_id: int, gpu: str | None, descs: dict, jobs, requests, reply, events,
                abort) -> None:
    """Take jobs until a ``None`` arrives; run each episode against the remote planners.

    ``gpu`` is set when the eval simulator runs on the GPU (M1-M4): the worker
    then claims that device, as a planner process does.
    """
    _ignoreInterrupts()
    requests.cancel_join_thread()  # after a stop, an unread request must not block exit
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        from ContactModelStudy.Drivers import run_episodes as drv
        from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder
        from ContactModelStudy.Utils.EvalSimulators import makeEvalSim
        task, eval_task = drv.buildTasks(args)
        base_weights = dict(task.params["cost_weights"])
        sim = makeEvalSim(args.eval_sim, eval_task.getModelPath(), eval_task.timestep)
        planner = RemotePlanner(descs[args.rollout_model], task, requests, reply, worker_id, abort)
        recorder = EpisodeRecorder(eval_task, sim, planner, cli_args=vars(args), eval_sim=args.eval_sim)
        events.put(("worker_ready", worker_id))
    except BaseException as e:
        events.put(("worker_failed", worker_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        return
    busy = 0.0
    while not abort.is_set():
        try:
            item = jobs.get(timeout=_POLL_S)
        except queue.Empty:
            continue
        if item is None:
            break
        job_id, job = item
        ep = job.episode
        goal_seed, noise_seed = episodeSeeds(args.seed, ep)
        eval_task.reseed(goal_seed)
        planner.useJob(descs[job.rollout_model or args.rollout_model], job, base_weights)
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
        events.put(("episode", worker_id, job_id, episode, summary, error))
    events.put(("worker_stats", worker_id, busy))


# -- main process ----------------------------------------------------------------
def describeRollout(args, task, rollout_model: str | None = None, planner_params: dict | None = None) -> dict:
    """What a worker's ``RemotePlanner`` reports about the planner, built without a GPU.

    ``rollout_model`` and ``planner_params`` describe a job's settings instead
    of the command line's.
    """
    from ContactModelStudy.Drivers import run_episodes as drv
    from ContactModelStudy.Utils.ContactModelPresets import CONTACT_MODELS, _presetConfig, _resolveName, _simulatorClasses
    model = rollout_model or args.rollout_model
    sim_cls, _ = _simulatorClasses(CONTACT_MODELS[_resolveName(model)]["simulator"])
    config = drv.buildPlannerConfig(args)
    if planner_params:
        config = replace(config, **planner_params)
    return {"planner_config": config, "planner_class": "MPPI",
            "sim_class": sim_cls.__name__, "N": args.n_samples,
            "sim_config": _presetConfig(model, **drv.rolloutSimOverrides(args, task))}


@contextlib.contextmanager
def sigtermAsInterrupt():
    """Inside the block, SIGTERM (scancel, a job's time limit) raises ``KeyboardInterrupt``.

    So a driver stops the same orderly way on either: it calls
    ``EpisodePool.interrupt`` and closes the episodes in flight.
    """
    def _terminate(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class EpisodePool:
    """GPU planner processes and CPU eval-sim workers that run episodes on demand.

    The processes live as long as the pool, so a caller can keep submitting
    jobs, each with its own contact model and planner settings, without paying
    process start-up and CUDA graph capture again. ``runPool`` submits one
    fixed batch; ``experiments/run_bayes_opt.py`` keeps submitting trials.

    Every planner process holds one planner per model in ``rollout_models``, so
    any free planner can serve any job. A job's planner params and cost weights
    are applied by the planner process before each plan, in place, which keeps
    the captured rollout graphs valid; see ``SamplingBasedPlannerBase.UpdateConfig``
    and ``TaskBase.setCostWeights``.

    Use::

        with EpisodePool(args, n_workers=8, gpus=["0", "1"], rollout_models=["M2", "M3"]) as pool:
            pool.submit(EpisodeJob(0, rollout_model="M3", planner_params={"temperature": 5.0}))
            while pool.pending and not pool.fatal:
                for result in pool.poll():
                    ...

    ``problems`` lists what went wrong (a process that failed, an interrupt);
    ``fatal`` is set once the pool can no longer finish its jobs.
    """

    def __init__(self, args, n_workers: int, gpus: list[str | None], planners_per_gpu: int = 1,
                 rollout_models: list[str] | None = None):
        from ContactModelStudy.Drivers import run_episodes as drv
        self.args = args
        self.models = list(dict.fromkeys(rollout_models or [args.rollout_model]))
        if args.rollout_model not in self.models:
            # The workers' default description; harmless when no job uses it.
            args = self.args = _withModel(args, self.models[0])
        self.n_workers = max(1, int(n_workers))
        self.planner_gpus = [g for g in gpus for _ in range(planners_per_gpu)]
        from ContactModelStudy.Utils.EvalSimulators import isGpuEvalSim
        #: Each worker's GPU: round-robin over the pool's GPUs for a GPU eval
        #: simulator, none for a CPU one.
        self.worker_gpus = [gpus[w % len(gpus)] if isGpuEvalSim(args.eval_sim) and gpus else None
                            for w in range(self.n_workers)]
        task, _ = drv.buildTasks(args)
        self.descs = {m: describeRollout(args, task, m) for m in self.models}
        self.problems: list[str] = []
        self.fatal = False
        self.stats = {"planner_busy": 0.0, "plans": 0, "worker_busy": 0.0, "planner_stats": 0}
        self._jobs: dict[int, EpisodeJob] = {}
        self._next_id = 0
        self._finished: list[PoolResult] = []
        self._started = self._closed = False
        self.t_start = None

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> "EpisodePool":
        ctx = mp.get_context("spawn")
        self._abort = ctx.Event()
        self._job_queue, self._requests, self._events = ctx.Queue(), ctx.Queue(), ctx.Queue()
        # Kept on the pool: Process.start drops its args, and a queue the parent
        # no longer references is gone before a spawned child can attach to it.
        self._replies = replies = [ctx.Queue() for _ in range(self.n_workers)]
        self._planners = [ctx.Process(target=_plannerMain, name=f"planner{i}",
                                      args=(self.args, g, i, self.models, self._requests, replies,
                                            self._events, self._abort), daemon=True)
                          for i, g in enumerate(self.planner_gpus)]
        self._workers = [ctx.Process(target=_workerMain, name=f"worker{w}",
                                     args=(self.args, w, self.worker_gpus[w], self.descs, self._job_queue,
                                           self._requests,
                                           replies[w], self._events, self._abort), daemon=True)
                         for w in range(self.n_workers)]
        self.t_start = time.time()
        for p in self._planners + self._workers:
            p.start()
        self._started = True
        return self

    def __enter__(self) -> "EpisodePool":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.interrupt("interrupted" if exc_type is KeyboardInterrupt else f"{exc_type.__name__}: {exc}")
        self.close()

    # -- jobs --------------------------------------------------------------------
    def submit(self, job: EpisodeJob) -> int:
        """Queue one episode; returns its job id. Workers take jobs in submission order."""
        model = job.rollout_model or self.args.rollout_model
        if model not in self.models:
            raise ValueError(f"rollout model {model!r} is not one this pool was built for: {self.models}")
        job_id = self._next_id
        self._next_id += 1
        self._jobs[job_id] = job
        self._job_queue.put((job_id, job))
        return job_id

    @property
    def pending(self) -> int:
        """Jobs submitted and not yet returned."""
        return len(self._jobs)

    def poll(self, timeout: float = 1.0) -> list[PoolResult]:
        """Episodes finished since the last call; waits up to ``timeout`` for the first."""
        deadline = time.time() + timeout
        while True:
            try:
                self._handle(self._events.get(timeout=max(0.0, min(0.2, deadline - time.time()))))
            except queue.Empty:
                self._checkAlive()
            if self._finished and self._events.empty() or time.time() >= deadline or self.fatal:
                break
        out, self._finished = self._finished, []
        return out

    def interrupt(self, reason: str = "interrupted") -> None:
        """Stop: running episodes close as "interrupted" at their next control step."""
        if reason not in self.problems:
            self.problems.append(reason)
        self.fatal = True
        if self._started:
            self._abort.set()

    def close(self) -> list[PoolResult]:
        """Stop every process; returns any episodes that finished meanwhile."""
        if not self._started or self._closed:
            out, self._finished = self._finished, []
            return out
        self._closed = True
        if self.problems:
            self._abort.set()
        # Workers stop once they read a None (or at the next control step after
        # an abort); then the planners are told to stop.
        for _ in self._workers:
            self._job_queue.put(None)
        self._drainUntil(self._workers, time.time() + (30 if self.problems else 120))
        for _ in self._planners:
            self._requests.put(None)
        self._drainUntil(self._planners, time.time() + 30)
        while True:
            try:
                self._handle(self._events.get(timeout=0.2))
            except queue.Empty:
                break
        for p in self._planners + self._workers:
            if p.is_alive():
                p.terminate()
        out, self._finished = self._finished, []
        return out

    @property
    def wall(self) -> float:
        return 0.0 if self.t_start is None else time.time() - self.t_start

    # -- messages ----------------------------------------------------------------
    def _handle(self, msg) -> None:
        """Act on one message from a planner or worker."""
        kind = msg[0]
        if kind == "episode":
            _, w, job_id, episode, summary, error = msg
            job = self._jobs.pop(job_id, None)
            if job is None:
                return
            self._finished.append(PoolResult(job, episode, summary, error, w))
        elif kind in ("planner_failed", "worker_failed"):
            print(f"\n{kind.replace('_', ' ')} ({msg[1]}): {msg[2]}")
            self.problems.append(f"{kind} {msg[1]}")
            n_worker_failed = sum(p.startswith("worker_failed") for p in self.problems)
            if kind == "planner_failed" or n_worker_failed >= self.n_workers:
                self.fatal = True
        elif kind == "worker_stats":
            self.stats["worker_busy"] += msg[2]
        elif kind == "planner_stats":
            self.stats["planner_busy"] += msg[2]
            self.stats["plans"] += msg[3]
            self.stats["planner_stats"] += 1

    def _checkAlive(self) -> None:
        if self._closed or self.fatal:
            return
        if any(not p.is_alive() for p in self._planners):
            self.problems.append("a planner process exited unexpectedly")
            self.fatal = True
        elif all(not w.is_alive() for w in self._workers):
            self.problems.append("every worker exited")
            self.fatal = True

    def _drainUntil(self, procs, deadline) -> None:
        """Keep reading messages until ``procs`` have exited or ``deadline`` passes.

        Reading while waiting matters: a child cannot exit until the messages
        it sent (a finished episode can be large) have been read.
        """
        while any(p.is_alive() for p in procs) and time.time() < deadline:
            try:
                self._handle(self._events.get(timeout=0.2))
            except queue.Empty:
                pass

    def printUtilization(self, n_done: int) -> None:
        """The end-of-run throughput line, and how busy the planners and workers were."""
        wall = self.wall
        if not n_done:
            return
        print(f"\n{n_done} episodes in {wall:.1f} s ({wall / n_done:.1f} s per episode, "
              f"{n_done * 3600 / wall:.0f} per hour)")
        if self.stats["planner_stats"]:
            print(f"GPU planners busy {100 * self.stats['planner_busy'] / (wall * len(self._planners)):.0f}% "
                  f"({self.stats['plans']} plans), eval workers busy "
                  f"{100 * self.stats['worker_busy'] / (wall * len(self._workers)):.0f}%")


def _withModel(args, model: str):
    """A copy of ``args`` with another ``rollout_model``."""
    import argparse
    return argparse.Namespace(**{**vars(args), "rollout_model": model})


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
    pool = EpisodePool(args, n_workers, gpus, planners_per_gpu)
    drv.printHeader(args, task, eval_task, desc["sim_config"], desc["sim_class"],
                    f"MPPI(N={args.n_samples}), {len(pool.planner_gpus)} planner process(es)")
    where = "GPU" if pool.worker_gpus[0] is not None else "CPU"
    print(f"pool       {n_workers} eval workers ({args.eval_sim}, {where}), {len(pool.planner_gpus)} planners "
          f"on GPU(s) {', '.join(str(g) for g in gpus)}")

    # The collected results: every episode a worker finishes lands here.
    eval_cls, eval_cfg = evalSimConfig(args.eval_sim, eval_task.timestep)
    sim_desc = SimStandIn(eval_cls, eval_cfg, nq=eval_task.nq, nv=eval_task.nv, nu=eval_task.nu)
    planner_desc = RemotePlanner(desc, task, None, None, -1, None)
    recorder = EpisodeRecorder(eval_task, sim_desc, planner_desc, cli_args=vars(args), eval_sim=args.eval_sim)
    done: dict[int, dict] = {}

    def take(results: list[PoolResult]) -> None:
        for r in results:
            ep = r.job.episode
            if ep in done:
                continue
            recorder.episodes.append(r.episode)
            done[ep] = r.summary
            drv._printEpisode(ep, r.summary, args)
            if r.error:
                print(f"  ERROR in worker {r.worker}: {r.error}")

    with sigtermAsInterrupt():
        try:
            pool.start()
            for ep in range(args.n_episodes):
                pool.submit(EpisodeJob(ep))
            while pool.pending and not pool.fatal:
                take(pool.poll(1.0))
        except KeyboardInterrupt:
            print("\ninterrupted: closing the episodes in progress as 'interrupted' ...")
            pool.interrupt()
        finally:
            wall = pool.wall
            take(pool.close())
            recorder.episodes.sort(key=lambda e: e.extra.get("episode", 0))
            if len(recorder) > 1:
                drv.printBatch(recorder)
            pool.t_start = time.time() - wall        # the line reports the run, not the shutdown
            pool.printUtilization(len(done))
            drv.saveResults(args, recorder)
    if pool.problems:
        print("\nstopped early: " + "; ".join(pool.problems))
        return 1
    return 0
