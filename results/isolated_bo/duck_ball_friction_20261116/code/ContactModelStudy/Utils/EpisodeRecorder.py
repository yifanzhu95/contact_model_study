"""Recording a batch of episodes, and replaying it.

``EpisodeRecorder`` collects, per episode, every control step's state, action,
action uncertainty and planning time, plus why the episode ended, and the
configs of everything that produced it. ``Save`` writes one JSON summary for the
batch and one ``.npy`` file per episode::

    results/run.json            configs + one summary entry per episode
    results/run_3f9a1c2e.npy    that episode's per-step arrays
    results/run_b71d0e44.npy

Each ``.npy`` holds a structured array, one record per kept control step (every
step, or every k-th: see ``StepSaving``), with the fields ``step``, ``t``,
``q``, ``q_dot``, ``u``, ``sigma_u``, ``planning_time``, ``planner_cost``, the
state that followed (``q_next``, ``q_dot_next``) and what the plan started from
(``u_prev``, ``noise_index``, ``temperature``, and with ``StepSaving.plans`` the
``(H, nu)`` ``mean_start`` and the planned ``action_seq``). A missing value is
stored as NaN. It loads with plain ``np.load`` and needs no pickle.

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
#: file it does not understand. Version 2 added the per-step ``planner_cost``;
#: version 3 keeps every k-th step (``step``), with the state that followed it
#: and what its plan started from.
FORMAT_VERSION = 3
_READABLE_VERSIONS = (1, 2, 3)

#: Per-step fields of the ``.npy`` records, in order.
_STEP_FIELDS = ("step", "t", "q", "q_dot", "u", "sigma_u", "planning_time", "planner_cost",
                "q_next", "q_dot_next", "u_prev", "noise_index", "temperature")
#: Also saved per step under ``StepSaving.plans``: the planner's ``(H, nu)`` sequences.
_PLAN_FIELDS = ("mean_start", "action_seq")


@dataclass(frozen=True)
class StepSaving:
    """Which control steps an ``EpisodeRecorder`` keeps, and what it keeps of each.

    Attributes:
        enabled: Keep per-step data at all. Off, an episode keeps only its
            summary and ``Save`` writes no ``.npy``.
        stride: Keep steps ``0, stride, 2 * stride, ...``. A kept step also
            keeps the state that followed it (``q_next``, ``q_dot_next``).
        plans: Also keep, per kept step, the ``(H, nu)`` mean its plan started
            from (``mean_start``) and the command sequence it planned
            (``action_seq``). A warm-started plan needs its mean to be replayed
            on its own; under ``warm_start="none"`` every plan starts from the
            reset mean, so it does not.
    """

    enabled: bool = True
    stride: int = 1
    plans: bool = False

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")

    def keeps(self, step: int) -> bool:
        """Whether an episode's step ``step`` is kept."""
        return self.enabled and step % self.stride == 0

    @classmethod
    def fromArgs(cls, args) -> "StepSaving":
        """From ``run_episodes.py``'s ``save_steps``, ``save_stride`` and ``warm_start``."""
        return cls(enabled=bool(args.save_steps), stride=int(getattr(args, "save_stride", 1)),
                   plans=getattr(args, "warm_start", "none") != "none")


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
    return {"class": _className(task), "role": _jsonable(getattr(task, "role", None)),
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
    entry = {"class": _className(sim), "config": _jsonable(cfg)}
    if getattr(sim, "N", None) is not None:
        entry["N"] = sim.N
    if resolved:
        entry["resolved"] = resolved
    return entry


def _className(obj) -> str:
    """``obj``'s class name, or the ``class_name`` a stand-in reports for the real object.

    A parallel driver records through stand-ins for a planner and simulator
    that live in another process; they name the class they stand in for.
    """
    return getattr(obj, "class_name", None) or type(obj).__name__


def _configSnapshot(task, simulator, planner, metadata: dict) -> dict:
    """What produced an episode: both tasks, both simulators and the planner.

    The eval side is what the recorder was given. The rollout side is read off
    the planner: the task it plans against (``planner.task``) and the
    simulator it rolls out on (``planner.sim``).
    """
    snap = {
        "eval_task": _taskEntry(task),
        "eval_simulator": _simEntry(simulator),
        "planner": {"class": _className(planner), "config": _jsonable(planner.config)},
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
    #: Control steps taken, kept or not.
    n_steps: int = 0
    #: Every step's planning time, kept or not: the summary's statistics.
    planning_times: list = field(default_factory=list)
    #: The kept steps, one dict of ``_STEP_FIELDS`` (and ``_PLAN_FIELDS``) each.
    steps: list = field(default_factory=list)
    #: The planner's noise seed for the episode, when it reports one.
    noise_seed: int | None = None
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

    def closeLastStep(self, q: np.ndarray, q_dot: np.ndarray) -> None:
        """Give the last kept step, if it was the latest step, the state that followed it."""
        if self.steps and self.steps[-1]["step"] == self.n_steps - 1:
            self.steps[-1]["q_next"], self.steps[-1]["q_dot_next"] = q, q_dot

    def toArray(self, dims: tuple[int, int, int, int] = (0, 0, 0, 0), plans: bool = False) -> np.ndarray:
        """The kept steps as one structured array, one record per step.

        ``dims`` is ``(nq, nv, nu, H)``, used to shape the fields of an episode
        with no kept steps; otherwise they are read off the recorded data.
        """
        nq, nv, nu, H = dims
        if self.steps:
            s = self.steps[0]
            nq, nv, nu = len(s["q"]), len(s["q_dot"]), len(s["u"])
            H = len(s["mean_start"]) if plans else H
        fields = [("step", np.int64), ("t", np.float64), ("q", np.float64, (nq,)),
                  ("q_dot", np.float64, (nv,)), ("u", np.float64, (nu,)), ("sigma_u", np.float64, (nu,)),
                  ("planning_time", np.float64), ("planner_cost", np.float64),
                  ("q_next", np.float64, (nq,)), ("q_dot_next", np.float64, (nv,)),
                  ("u_prev", np.float64, (nu,)), ("noise_index", np.float64), ("temperature", np.float64)]
        if plans:
            fields += [(name, np.float64, (H, nu)) for name in _PLAN_FIELDS]
        dtype = np.dtype(fields)
        out = np.empty(len(self.steps), dtype=dtype)
        for name in dtype.names:
            if self.steps:
                out[name] = np.asarray([s[name] for s in self.steps], dtype=dtype[name].base)
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
    copy in the saved summary. Which steps are kept whole, and what of them,
    is a ``StepSaving``.

    Example::

        rec = EpisodeRecorder(eval_task, sim, planner, steps=StepSaving(stride=4),
                              cli_args=vars(args))
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

    def __init__(self, task, simulator, planner, steps: StepSaving | None = None, **metadata):
        """Bind the recorder to what runs the episodes.

        Args:
            task: The eval task the episodes are scored against.
            simulator: The eval simulator the states come from.
            planner: The planner choosing the actions. The rollout task and
                rollout simulator it plans with (``planner.task``,
                ``planner.sim``) are recorded too.
            steps: Which steps to keep whole. Defaults to every step, without
                the planner's sequences.
            **metadata: Anything else worth keeping with the batch (CLI
                arguments, a git hash, a note), saved under ``metadata``.
        """
        self.task = task
        self.simulator = simulator
        self.planner = planner
        self.steps = steps if steps is not None else StepSaving()
        self.metadata = metadata
        #: The configs at construction; what ``Combine`` compares.
        self.configs = self._snapshot()
        self.episodes: list[_Episode] = []
        self._active: _Episode | None = None

    def _dims(self) -> tuple[int, int, int, int]:
        """``(nq, nv, nu)`` of the eval simulator, and the planner's horizon ``H``."""
        sim = self.simulator
        return int(sim.nq), int(sim.nv), int(sim.nu), int(getattr(self.planner, "horizon", 0))

    def _sequence(self, seq, nu: int) -> np.ndarray:
        """A planner's ``(H, nu)`` sequence as float64, or NaN when it reported none."""
        if seq is None:
            return np.full((self._dims()[3], nu), np.nan)
        return np.asarray(seq, dtype=np.float64).reshape(-1, nu)

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

        Every step counts toward ``n_steps`` and the planning-time summary; the
        steps ``self.steps`` keeps are also stored whole. A kept step also
        records, read off the objects the recorder holds: the simulator's
        clock as ``t``; the planner's best rollout cost as ``planner_cost``;
        from ``planner.last_plan_start``, the control the plan accumulated from
        (``u_prev``), its noise draw (``noise_index``) and its temperature;
        under ``StepSaving.plans``, the mean it started from (``mean_start``)
        and ``planner.last_action_seq``; and, once the next step or the end of
        the episode comes, the state that followed (``q_next``,
        ``q_dot_next``). What the planner does not report is NaN. The goal
        errors of an episode's first step are its start errors.
        """
        ep = self._current()
        q = np.asarray(q, dtype=np.float64).ravel()
        q_dot = np.asarray(q_dot, dtype=np.float64).ravel()
        if ep.n_steps == 0:
            ep.start_errors = self._goalErrors(q, q_dot)
        ep.closeLastStep(q, q_dot)
        start = getattr(self.planner, "last_plan_start", None) or {}
        if ep.noise_seed is None and start.get("noise_seed") is not None:
            ep.noise_seed = int(start["noise_seed"])
        pt = np.nan if planning_time is None else float(planning_time)
        ep.planning_times.append(pt)
        if self.steps.keeps(ep.n_steps):
            u = np.asarray(U, dtype=np.float64).ravel()
            nan = np.full(u.shape, np.nan)
            u_prev = start.get("u_prev")
            rec = {
                "step": ep.n_steps,
                "t": float(np.asarray(getattr(self.simulator, "time", np.nan)).ravel()[0]),
                "q": q, "q_dot": q_dot, "u": u,
                "sigma_u": nan if sigma_U is None else np.asarray(sigma_U, dtype=np.float64).ravel(),
                "planning_time": pt,
                "planner_cost": float(getattr(self.planner, "last_min_cost", np.nan)),
                "q_next": np.full(q.shape, np.nan), "q_dot_next": np.full(q_dot.shape, np.nan),
                "u_prev": nan if u_prev is None else np.asarray(u_prev, dtype=np.float64).ravel(),
                "noise_index": float(start.get("resample_count", np.nan)),
                "temperature": float(start.get("lam", np.nan)),
            }
            if self.steps.plans:
                rec["mean_start"] = self._sequence(start.get("U"), u.size)
                rec["action_seq"] = self._sequence(getattr(self.planner, "last_action_seq", None), u.size)
            ep.steps.append(rec)
        ep.n_steps += 1

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
        its goal errors against the goal then current, and as the state that
        followed the last step, if that step was kept.

        With no step recorded since the last episode, this records an episode
        of zero steps: one that ended before its first action (it started out
        failed, or already at its goal) is still an episode.
        """
        ep = self._current()
        q, q_dot = self.simulator.GetState()
        ep.closeLastStep(np.asarray(q, dtype=np.float64).ravel(), np.asarray(q_dot, dtype=np.float64).ravel())
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
        * ``goals`` (their labels, in order), with ``goal_values`` (each goal
          as ``setGoal`` took it) and ``goal_steps`` (the step it was adopted
          at), ``goals_reached``,
          ``success_steps``, ``steps_to_success`` and ``success``. An episode
          succeeded if it reached any goal, however it then ended;
        * ``goal_errors_start`` / ``goal_errors_end``, when the task defines
          ``goalErrors``: the first state against the first goal, and the final
          state against the goal then current;
        * ``q_end`` / ``q_dot_end``, the state the last action led to, which
          the per-step records do not include;
        * ``plan_s_first``, ``plan_s_mean``, ``plan_s_max``: planning time,
          over every step, kept or not. The first plan of a run pays for
          kernel compilation and graph capture, so mean and max leave it out
          when there is more than one;
        * ``noise_seed``: the planner's noise seed, when it reports one;
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
        pt = np.asarray(ep.planning_times, dtype=float)
        finite = pt[np.isfinite(pt)]
        rest = finite[1:] if finite.size > 1 else finite
        summary = {
            "id": ep.id,
            "finish_reason": ep.finish_reason,
            "failed": ep.finish_reason == "failure",
            "n_steps": ep.n_steps,
            "goals": [label for _step, _goal, label in ep.goals],
            "goal_values": [goal for _step, goal, _label in ep.goals],
            "goal_steps": [step for step, _goal, _label in ep.goals],
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
            "noise_seed": ep.noise_seed,
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
            save_steps: Write the per-step ``.npy`` files, of the steps
                ``self.steps`` kept. With ``False``, or when ``self.steps`` is
                not enabled, only the JSON is written, with its configs and
                summaries, and each episode's ``file`` is ``null``.

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
        save_steps = save_steps and self.steps.enabled
        entries = []
        for ep, summary in zip(self.episodes, batch.pop("episodes")):
            npy = None
            if save_steps:
                npy = f"{path.stem}_{ep.id}.npy"
                np.save(path.parent / npy, ep.toArray(self._dims(), self.steps.plans))
            entry = {"id": ep.id, "file": npy, "finish_reason": ep.finish_reason,
                     "n_steps": ep.n_steps, "n_saved_steps": len(ep.steps) if save_steps else 0,
                     "summary": summary}
            if ep.configs != self.configs:
                entry["configs"] = ep.configs
            entries.append(entry)
        doc = {"format_version": FORMAT_VERSION, "configs": self.configs,
               "step_saving": dataclasses.asdict(self.steps) if save_steps else None,
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
        if self.steps != other.steps:
            raise ValueError(f"cannot combine recorders that keep steps differently "
                             f"({self.steps} and {other.steps})")
        out = EpisodeRecorder.__new__(EpisodeRecorder)
        out.task, out.simulator, out.planner = self.task, self.simulator, self.planner
        out.steps = self.steps
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
        n_steps: Control steps in the episode, saved or not.
        step: Which control step each saved row is (``0, k, 2k, ...`` when
            saved every k-th step).
        t, q, q_dot, u, sigma_u, planning_time, planner_cost: Per-step
            arrays, one row per saved step, as recorded.
        q_next, q_dot_next: The state each saved step led to.
        u_prev, noise_index, temperature: What each saved step's plan started
            from: the control it accumulated from, its noise draw and its
            temperature.
        mean_start, action_seq: ``(n_saved, H, nu)``: the mean each saved
            step's plan started from and the command sequence it planned.
            ``None`` unless saved (``StepSaving.plans``).
        save_stride: Every how many steps a row was saved.

    Older files are filled in to the same layout: version 1 and 2 saved every
    step, so ``step`` counts up, ``q_next`` is the next row's state (the end
    state for the last), ``u_prev`` the previous row's action (NaN for the
    first), and the rest NaN; version 1 had no ``planner_cost`` (NaN). All the
    per-step arrays are ``None`` when the batch was saved without its steps
    (``save_steps=False``); the summary is still there.
    """

    id: str
    finish_reason: str
    summary: dict
    configs: dict
    n_steps: int
    step: np.ndarray | None
    t: np.ndarray | None
    q: np.ndarray | None
    q_dot: np.ndarray | None
    u: np.ndarray | None
    sigma_u: np.ndarray | None
    planning_time: np.ndarray | None
    planner_cost: np.ndarray | None
    q_next: np.ndarray | None
    q_dot_next: np.ndarray | None
    u_prev: np.ndarray | None
    noise_index: np.ndarray | None
    temperature: np.ndarray | None
    mean_start: np.ndarray | None = None
    action_seq: np.ndarray | None = None
    save_stride: int = 1

    @property
    def has_steps(self) -> bool:
        """Whether the per-step arrays were saved."""
        return self.q is not None

    @property
    def n_saved(self) -> int:
        """Rows of per-step data saved."""
        return 0 if self.step is None else len(self.step)

    @property
    def has_plan_starts(self) -> bool:
        """Whether every saved step records its plan's noise draw (version 3 on)."""
        return self.has_steps and bool(np.isfinite(self.noise_index).all())

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
        #: How the steps were kept (``StepSaving`` as a dict); every step before version 3.
        self.step_saving: dict = doc.get("step_saving") or {"enabled": True, "stride": 1, "plans": False}
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
            summary = e.get("summary", {})
            if e.get("file") is None:                       # saved without steps
                fields = dict.fromkeys(_STEP_FIELDS)
            else:
                arr = np.load(self.path.parent / e["file"])
                fields = {name: arr[name] for name in _STEP_FIELDS + _PLAN_FIELDS if name in arr.dtype.names}
                if "step" not in fields:
                    fields = _everyStepFields(fields, summary)
                for name in _STEP_FIELDS:                   # anything else not recorded
                    fields.setdefault(name, np.full(len(arr), np.nan))
            self._cache[i] = RecordedEpisode(
                id=e["id"], finish_reason=e["finish_reason"], summary=summary,
                configs=e.get("configs", self.configs), n_steps=int(e["n_steps"]),
                save_stride=int(self.step_saving.get("stride", 1)), **fields,
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
            for k in range(ep.n_saved):
                yield ep.id, int(ep.step[k]), ep.q[k], ep.q_dot[k], ep.u[k]


def _everyStepFields(fields: dict, summary: dict) -> dict:
    """A version 1/2 episode's fields, filled in to the version 3 layout.

    Those versions saved every step, so the state after a step is the next
    row's (the end state after the last) and the control before it the
    previous row's action. What they did not record is NaN.
    """
    q, q_dot, u = fields["q"], fields["q_dot"], fields["u"]
    n = len(q)
    nan = lambda *shape: np.full(shape, np.nan)  # noqa: E731
    q_end = np.asarray(summary["q_end"], dtype=float) if summary.get("q_end") is not None else nan(q.shape[1])
    v_end = (np.asarray(summary["q_dot_end"], dtype=float) if summary.get("q_dot_end") is not None
             else nan(q_dot.shape[1]))
    return {**fields,
            "step": np.arange(n),
            "planner_cost": fields.get("planner_cost", nan(n)),
            "q_next": np.concatenate([q[1:], q_end[None]]) if n else q.copy(),
            "q_dot_next": np.concatenate([q_dot[1:], v_end[None]]) if n else q_dot.copy(),
            "u_prev": np.concatenate([nan(1, u.shape[1]), u[:-1]]) if n else u.copy(),
            "noise_index": nan(n), "temperature": nan(n)}
