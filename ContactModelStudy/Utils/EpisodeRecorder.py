"""Recording a batch of episodes, and replaying it.

``EpisodeRecorder`` collects, per episode, every control step's state, action,
action uncertainty and planning time, plus why the episode ended, and the
configs of everything that produced it. ``Save`` writes one JSON summary for the
batch and one ``.npy`` file per episode::

    results/run.json            configs + one summary entry per episode
    results/run_3f9a1c2e.npy    that episode's per-step arrays
    results/run_b71d0e44.npy

Each ``.npy`` holds a structured array, one record per control step, with the
fields ``t``, ``q``, ``q_dot``, ``u``, ``sigma_u``, ``planning_time`` and
``planner_cost``. A missing value is stored as NaN. It loads with plain
``np.load`` and needs no pickle.

The recorder also owns each episode's summary: ``GenerateSummary`` derives it
from what was recorded (steps, goals, successes, the start and end states and
their goal errors, planning times), so a driver never assembles one itself.

``EpisodeReplayer`` reads a saved batch back: by episode, or state by state
across the whole batch.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

#: Version of the saved layout, written into the JSON, so a reader can tell a
#: file it does not understand. Version 2 added the per-step ``planner_cost``.
FORMAT_VERSION = 2
_READABLE_VERSIONS = (1, 2)

#: Per-step fields of the ``.npy`` records, besides the state and action ones.
_STEP_FIELDS = ("t", "q", "q_dot", "u", "sigma_u", "planning_time", "planner_cost")


def _jsonable(obj: Any) -> Any:
    """Convert configs and their contents into plain JSON types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return _jsonable(obj.value)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _taskEntry(task) -> dict:
    """A task's class, scene, config, and the timestep it actually runs at.

    ``timestep`` is kept apart from ``config.timestep`` because the two differ
    for a rollout task, whose scene runs at a coarser step than the config's
    (eval) one.
    """
    return {"class": type(task).__name__, "role": _jsonable(getattr(task, "role", None)),
            "model_path": task.getModelPath(), "timestep": float(task.timestep),
            "config": _jsonable(task.config)}


def _simEntry(sim) -> dict:
    """A simulator's class and config, plus the schedule its config resolves to.

    The config keeps what was asked for (``substeps`` or ``ctrl_time_step``,
    ``horizon`` or ``time_horizon``), so the whole-step values actually used
    are recorded beside it, under ``resolved``.
    """
    cfg = sim.config
    resolved = {name: _jsonable(getattr(cfg, name))
                for name in ("resolved_substeps", "control_timestep",
                             "resolved_horizon", "horizon_duration")
                if hasattr(cfg, name)}
    entry = {"class": type(sim).__name__, "config": _jsonable(cfg)}
    if hasattr(sim, "N"):
        entry["N"] = sim.N
    if resolved:
        entry["resolved"] = resolved
    return entry


def _configSnapshot(task, simulator, planner, metadata: dict) -> dict:
    """What produced an episode: both tasks, both simulators and the planner.

    The eval side is what the recorder was given. The rollout side is read off
    the planner: the task it plans against (``planner.task``) and the
    simulator it rolls out on (``planner.sim``).
    """
    snap = {
        "eval_task": _taskEntry(task),
        "eval_simulator": _simEntry(simulator),
        "planner": {"class": type(planner).__name__, "config": _jsonable(planner.config)},
    }
    rollout_task = getattr(planner, "task", None)
    if rollout_task is not None:
        snap["rollout_task"] = _taskEntry(rollout_task)
    rollout = getattr(planner, "sim", None)
    if rollout is not None:
        snap["rollout_simulator"] = _simEntry(rollout)
    if metadata:
        snap["metadata"] = _jsonable(metadata)
    return snap


@dataclass
class _Episode:
    """One episode's buffers while recording, and its result once finished."""

    id: str
    configs: dict
    t: list = field(default_factory=list)
    q: list = field(default_factory=list)
    q_dot: list = field(default_factory=list)
    u: list = field(default_factory=list)
    sigma_u: list = field(default_factory=list)
    planning_time: list = field(default_factory=list)
    planner_cost: list = field(default_factory=list)
    #: ``(step, goal, label)`` for every goal adopted during the episode.
    goals: list = field(default_factory=list)
    #: Steps at which a goal was reached.
    success_steps: list = field(default_factory=list)
    #: Goal errors of the first recorded state, against the first goal.
    start_errors: dict | None = None
    #: State and goal errors when the episode finished.
    q_end: list | None = None
    q_dot_end: list | None = None
    end_errors: dict | None = None
    finish_reason: str | None = None
    #: Caller-supplied extras (a video path, the settle time, ...).
    extra: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.q)

    def toArray(self, dims: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """The per-step buffers as one structured array, one record per step.

        ``dims`` is ``(nq, nv, nu)``, used to shape the fields of an episode
        with no steps; otherwise they are read off the recorded data.
        """
        nq, nv, nu = (len(self.q[0]), len(self.q_dot[0]), len(self.u[0])) if self.q else dims
        dtype = np.dtype([
            ("t", np.float64), ("q", np.float64, (nq,)), ("q_dot", np.float64, (nv,)),
            ("u", np.float64, (nu,)), ("sigma_u", np.float64, (nu,)),
            ("planning_time", np.float64), ("planner_cost", np.float64),
        ])
        out = np.empty(self.n_steps, dtype=dtype)
        for name in dtype.names:
            if self.n_steps:
                out[name] = np.asarray(getattr(self, name), dtype=np.float64)
        return out


class EpisodeRecorder:
    """Records a batch of episodes run by one task, simulator and planner.

    An episode starts with its first ``record...`` call and ends with
    ``episodeFinished``; one recorder holds any number of finished episodes.
    The driver reports the steps (``recordStateAndAction``) and the two events
    only it sees (``recordGoal``, ``recordSuccess``). Everything else in an
    episode's summary is read off the task, simulator and planner by the
    recorder itself; see ``GenerateSummary``.
    The configs are captured at construction and again at the start of every
    episode. An episode whose configs differ from the recorder's keeps its own
    copy in the saved summary.

    Example::

        rec = EpisodeRecorder(eval_task, sim, planner, cli_args=vars(args))
        rec.recordGoal(goal)
        for step in range(n_steps):
            q, q_dot = sim.GetState()
            t0 = time.perf_counter()
            u, sigma = planner.Plan(q, q_dot)
            rec.recordStateAndAction(q, q_dot, u, sigma, time.perf_counter() - t0)
            ...
        rec.episodeFinished("timeout")
        print(rec.GenerateSummary(-1))
        rec.Save("results/run.json")
    """

    def __init__(self, task, simulator, planner, **metadata):
        """Bind the recorder to what runs the episodes.

        Args:
            task: The eval task the episodes are scored against.
            simulator: The eval simulator the states come from.
            planner: The planner choosing the actions. The rollout task and
                rollout simulator it plans with (``planner.task``,
                ``planner.sim``) are recorded too.
            **metadata: Anything else worth keeping with the batch (CLI
                arguments, a git hash, a note), saved under ``metadata``.
        """
        self.task = task
        self.simulator = simulator
        self.planner = planner
        self.metadata = metadata
        #: The configs at construction; what ``Combine`` compares.
        self.configs = self._snapshot()
        self.episodes: list[_Episode] = []
        self._active: _Episode | None = None

    def _dims(self) -> tuple[int, int, int]:
        """``(nq, nv, nu)`` of the eval simulator."""
        sim = self.simulator
        return int(sim.nq), int(sim.nv), int(sim.nu)

    def _snapshot(self) -> dict:
        return _configSnapshot(self.task, self.simulator, self.planner, self.metadata)

    def _current(self) -> _Episode:
        """The episode being recorded, starting one if there is none."""
        if self._active is None:
            self._active = _Episode(id=uuid.uuid4().hex[:8], configs=self._snapshot())
        return self._active

    def _goalErrors(self, q, q_dot) -> dict | None:
        """The task's goal errors for a state, when the task defines them."""
        fn = getattr(self.task, "goalErrors", None)
        return None if fn is None else _jsonable(fn(q, q_dot))

    # -- recording -----------------------------------------------------------
    @property
    def recording(self) -> bool:
        """Whether an episode has started and not yet finished."""
        return self._active is not None

    def recordStateAndAction(self, q, q_dot, U, sigma_U=None, planning_time=None) -> None:
        """Append one control step to the current episode, starting one if needed.

        Args:
            q, q_dot: The state the action was planned from.
            U: The action applied, ``(nu,)``.
            sigma_U: The planner's uncertainty in ``U``, ``(nu,)``, if it
                reported one. Stored as NaN otherwise.
            planning_time: Seconds the plan took. Stored as NaN if not given.

        Also recorded, read off the objects the recorder holds: the simulator's
        clock as ``t``, and the planner's best rollout cost for this plan as
        ``planner_cost`` (NaN if the planner reports none). The goal errors of
        an episode's first step are its start errors.
        """
        ep = self._current()
        if ep.n_steps == 0:
            ep.start_errors = self._goalErrors(q, q_dot)
        u = np.asarray(U, dtype=np.float64).ravel()
        ep.t.append(float(np.asarray(getattr(self.simulator, "time", np.nan)).ravel()[0]))
        ep.q.append(np.asarray(q, dtype=np.float64).ravel())
        ep.q_dot.append(np.asarray(q_dot, dtype=np.float64).ravel())
        ep.u.append(u)
        ep.sigma_u.append(np.full(u.shape, np.nan) if sigma_U is None
                          else np.asarray(sigma_U, dtype=np.float64).ravel())
        ep.planning_time.append(np.nan if planning_time is None else float(planning_time))
        ep.planner_cost.append(float(getattr(self.planner, "last_min_cost", np.nan)))

    def recordGoal(self, goal, label: str | None = None) -> None:
        """Note that the task has adopted ``goal``, from the current step on.

        Args:
            goal: The goal, as the task's ``setGoal`` took it.
            label: A short name for it. Defaults to the task's ``goalFace()``
                when it has one (the letter a LEAP goal shows).
        """
        if label is None and hasattr(self.task, "goalFace"):
            label = self.task.goalFace(goal)
        ep = self._current()
        ep.goals.append((ep.n_steps, _jsonable(np.asarray(goal)), label))

    def recordSuccess(self) -> None:
        """Note that the current goal was reached, at the current step."""
        ep = self._current()
        ep.success_steps.append(ep.n_steps)

    def episodeFinished(self, finish_reason: str, **extra) -> None:
        """End the current episode.

        Args:
            finish_reason: Why it ended, e.g. ``"success"``, ``"failure"``,
                ``"timeout"`` or ``"interrupted"``.
            **extra: Things only the caller knows, kept in the episode's
                summary as they are (a video path, a settle time, ...).

        The simulator's state at this point is recorded as the end state, with
        its goal errors against the goal then current.

        With no step recorded since the last episode, this records an episode
        of zero steps: one that ended before its first action (it started out
        failed, or already at its goal) is still an episode.
        """
        ep = self._current()
        q, q_dot = self.simulator.GetState()
        ep.q_end = _jsonable(np.asarray(q).ravel())
        ep.q_dot_end = _jsonable(np.asarray(q_dot).ravel())
        ep.end_errors = self._goalErrors(q, q_dot)
        if ep.start_errors is None:                 # zero steps: it started here
            ep.start_errors = ep.end_errors
        ep.finish_reason = str(finish_reason)
        ep.extra = _jsonable(extra)
        self.episodes.append(ep)
        self._active = None

    # -- summaries -----------------------------------------------------------
    def GenerateSummary(self, episode: int | None = None) -> dict:
        """Summarize one finished episode, or the whole batch.

        Args:
            episode: Index of a finished episode (negative counts from the end,
                so ``-1`` is the latest). ``None`` summarizes the batch.

        An episode's summary has:

        * ``id``, ``finish_reason``, ``failed``, ``n_steps``;
        * ``goals`` (their labels, in order), ``goals_reached``,
          ``success_steps``, ``steps_to_success`` and ``success``. An episode
          succeeded if it reached any goal, however it then ended;
        * ``goal_errors_start`` / ``goal_errors_end``, when the task defines
          ``goalErrors``: the first state against the first goal, and the final
          state against the goal then current;
        * ``q_end`` / ``q_dot_end``, the state the last action led to, which
          the per-step records do not include;
        * ``plan_s_first``, ``plan_s_mean``, ``plan_s_max``: planning time.
          The first plan of a run pays for kernel compilation and graph
          capture, so mean and max leave it out when there is more than one;
        * whatever was passed to ``episodeFinished`` as ``extra``.

        The batch summary has the counts (``n_episodes``, ``finish_reasons``,
        ``n_success``, ``n_failed``, ``success_rate``) and every episode's
        summary under ``episodes``.
        """
        if episode is None:
            eps = [self.GenerateSummary(i) for i in range(len(self.episodes))]
            reasons = [e["finish_reason"] for e in eps]
            n = len(eps)
            n_ok = sum(e["success"] for e in eps)
            return {
                "n_episodes": n,
                "finish_reasons": {r: reasons.count(r) for r in sorted(set(reasons))},
                "n_success": n_ok,
                "n_failed": sum(e["failed"] for e in eps),
                "success_rate": n_ok / n if n else None,
                "episodes": eps,
            }

        ep = self.episodes[episode]
        succ = list(ep.success_steps)
        pt = np.asarray(ep.planning_time, dtype=float)
        finite = pt[np.isfinite(pt)]
        rest = finite[1:] if finite.size > 1 else finite
        summary = {
            "id": ep.id,
            "finish_reason": ep.finish_reason,
            "failed": ep.finish_reason == "failure",
            "n_steps": ep.n_steps,
            "goals": [label for _step, _goal, label in ep.goals],
            "goals_reached": len(succ),
            "success_steps": succ,
            "steps_to_success": succ[0] if succ else None,
            "success": bool(succ),
            "goal_errors_start": ep.start_errors,
            "goal_errors_end": ep.end_errors,
            "q_end": ep.q_end,
            "q_dot_end": ep.q_dot_end,
            "plan_s_first": float(finite[0]) if finite.size else None,
            "plan_s_mean": float(rest.mean()) if rest.size else None,
            "plan_s_max": float(rest.max()) if rest.size else None,
        }
        summary.update(ep.extra)
        return summary

    # -- output --------------------------------------------------------------
    def Save(self, path: str | Path, save_steps: bool = True) -> Path:
        """Write the summary JSON to ``path`` and one ``.npy`` per episode beside it.

        Each episode's arrays go to ``<stem>_<episode id>.npy`` in the same
        directory, and the JSON names each file. Only finished episodes are
        saved.

        Args:
            path: Where to write the JSON.
            save_steps: Write the per-step ``.npy`` files. With ``False`` only
                the JSON is written, with its configs and summaries, and each
                episode's ``file`` is ``null``.

        Returns:
            The path of the JSON written.

        Raises:
            RuntimeError: If an episode is still being recorded. Finish it
                first, for instance with ``episodeFinished("interrupted")``, so
                that it is not silently dropped.
        """
        if self.recording:
            raise RuntimeError(
                "an episode is still being recorded; call episodeFinished first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        batch = self.GenerateSummary()
        entries = []
        for ep, summary in zip(self.episodes, batch.pop("episodes")):
            npy = None
            if save_steps:
                npy = f"{path.stem}_{ep.id}.npy"
                np.save(path.parent / npy, ep.toArray(self._dims()))
            entry = {"id": ep.id, "file": npy, "finish_reason": ep.finish_reason,
                     "n_steps": ep.n_steps, "summary": summary}
            if ep.configs != self.configs:
                entry["configs"] = ep.configs
            entries.append(entry)
        doc = {"format_version": FORMAT_VERSION, "configs": self.configs,
               **batch, "episodes": entries}
        path.write_text(json.dumps(doc, indent=2))
        return path

    def Clear(self) -> None:
        """Drop every recorded episode, including one in progress."""
        self.episodes = []
        self._active = None

    def Combine(self, other: "EpisodeRecorder") -> "EpisodeRecorder":
        """A new recorder holding both recorders' episodes, this one's first.

        Neither recorder is changed.

        Raises:
            RuntimeError: If either recorder is in the middle of an episode.
            ValueError: If the two were built with different configs.
        """
        if self.recording or other.recording:
            raise RuntimeError("cannot combine a recorder that is recording an episode")
        if self.configs != other.configs:
            diff = sorted(k for k in set(self.configs) | set(other.configs)
                          if self.configs.get(k) != other.configs.get(k))
            raise ValueError(f"cannot combine recorders with different configs (differ in {diff})")
        out = EpisodeRecorder.__new__(EpisodeRecorder)
        out.task, out.simulator, out.planner = self.task, self.simulator, self.planner
        out.metadata = dict(self.metadata)
        out.configs = self.configs
        out.episodes = list(self.episodes) + list(other.episodes)
        out._active = None
        return out

    def __len__(self) -> int:
        """Finished episodes held."""
        return len(self.episodes)


@dataclass
class RecordedEpisode:
    """One saved episode, as ``EpisodeReplayer`` hands it back.

    Attributes:
        id: The episode's unique id.
        finish_reason: Why it ended.
        summary: The extra summary saved with it.
        configs: The configs it was run with.
        n_steps: Control steps in the episode.
        t, q, q_dot, u, sigma_u, planning_time, planner_cost: Per-step
            arrays, one row per control step, as recorded. ``planner_cost`` is
            all NaN for a version-1 file, which did not record it. All ``None``
            when the batch was saved without its steps (``save_steps=False``);
            the summary is still there.
    """

    id: str
    finish_reason: str
    summary: dict
    configs: dict
    n_steps: int
    t: np.ndarray | None
    q: np.ndarray | None
    q_dot: np.ndarray | None
    u: np.ndarray | None
    sigma_u: np.ndarray | None
    planning_time: np.ndarray | None
    planner_cost: np.ndarray | None

    @property
    def has_steps(self) -> bool:
        """Whether the per-step arrays were saved."""
        return self.q is not None

    def __len__(self) -> int:
        return self.n_steps


class EpisodeReplayer:
    """Reads back a batch saved by ``EpisodeRecorder.Save``.

    Iterating over it yields episodes. ``iterStates`` walks every step of every
    episode instead, skipping episodes saved without their steps::

        rep = EpisodeReplayer("results/run.json")
        for ep in rep:
            print(ep.id, ep.finish_reason, len(ep))
        for ep_id, step, q, q_dot, u in rep.iterStates():
            ...

    Episode arrays are loaded from disk when first asked for.
    """

    def __init__(self, path: str | Path):
        """Open a saved batch.

        Args:
            path: The JSON file ``EpisodeRecorder.Save`` wrote.

        Raises:
            ValueError: If the file is from an unknown format version.
        """
        self.path = Path(path)
        doc = json.loads(self.path.read_text())
        version = doc.get("format_version")
        if version not in _READABLE_VERSIONS:
            raise ValueError(f"{self.path} has format version {version}, "
                             f"this reader understands {_READABLE_VERSIONS}")
        #: The configs the batch was recorded with.
        self.configs: dict = doc["configs"]
        self._entries: list[dict] = doc["episodes"]
        self._cache: dict[int, RecordedEpisode] = {}

    @property
    def ids(self) -> list[str]:
        """Episode ids, in recording order."""
        return [e["id"] for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, i: int) -> RecordedEpisode:
        """The ``i``-th episode, loaded on first access."""
        i = range(len(self._entries))[i]      # normalizes negatives, raises IndexError
        if i not in self._cache:
            e = self._entries[i]
            if e.get("file") is None:                       # saved without steps
                fields = {name: None for name in _STEP_FIELDS}
            else:
                arr = np.load(self.path.parent / e["file"])
                fields = {name: (arr[name] if name in arr.dtype.names
                                 else np.full(len(arr), np.nan)) for name in _STEP_FIELDS}
            self._cache[i] = RecordedEpisode(
                id=e["id"], finish_reason=e["finish_reason"], summary=e.get("summary", {}),
                configs=e.get("configs", self.configs), n_steps=int(e["n_steps"]), **fields,
            )
        return self._cache[i]

    def episode(self, episode_id: str) -> RecordedEpisode:
        """The episode with this id."""
        return self[self.ids.index(episode_id)]

    def __iter__(self) -> Iterator[RecordedEpisode]:
        return self.iterEpisodes()

    def iterEpisodes(self) -> Iterator[RecordedEpisode]:
        """Every episode, in recording order."""
        for i in range(len(self)):
            yield self[i]

    def iterStates(self) -> Iterator[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]]:
        """``(episode_id, step, q, q_dot, u)`` for every saved step of every episode."""
        for ep in self.iterEpisodes():
            if not ep.has_steps:
                continue
            for k in range(len(ep)):
                yield ep.id, k, ep.q[k], ep.q_dot[k], ep.u[k]
