"""Recording a batch of episodes, and replaying it.

``EpisodeRecorder`` collects, per episode, every control step's state, action,
action uncertainty and planning time, plus why the episode ended, and the
configs of everything that produced it. ``Save`` writes one JSON summary for the
batch and one ``.npy`` file per episode::

    results/run.json            configs + one summary entry per episode
    results/run_3f9a1c2e.npy    that episode's per-step arrays
    results/run_b71d0e44.npy

Each ``.npy`` holds a structured array, one record per control step, with the
fields ``t``, ``q``, ``q_dot``, ``u``, ``sigma_u`` and ``planning_time``. A
missing ``sigma_u`` or ``planning_time`` is stored as NaN. It loads with plain
``np.load`` and needs no pickle.

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
#: file it does not understand.
FORMAT_VERSION = 1


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


def _configSnapshot(task, simulator, planner, metadata: dict) -> dict:
    """What produced an episode: every config, the classes, and the scene."""
    snap = {
        "task": {"class": type(task).__name__, "model_path": task.getModelPath(),
                 "config": _jsonable(task.config)},
        "simulator": {"class": type(simulator).__name__, "config": _jsonable(simulator.config)},
        "planner": {"class": type(planner).__name__, "config": _jsonable(planner.config)},
    }
    rollout = getattr(planner, "sim", None)
    if rollout is not None:
        snap["rollout_simulator"] = {
            "class": type(rollout).__name__, "N": getattr(rollout, "N", None),
            "config": _jsonable(rollout.config),
        }
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
    finish_reason: str | None = None
    summary: dict = field(default_factory=dict)

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
            ("planning_time", np.float64),
        ])
        out = np.empty(self.n_steps, dtype=dtype)
        for name in dtype.names:
            if self.n_steps:
                out[name] = np.asarray(getattr(self, name), dtype=np.float64)
        return out


class EpisodeRecorder:
    """Records a batch of episodes run by one task, simulator and planner.

    An episode starts with its first ``recordStateAndAction`` and ends with
    ``episodeFinished``; one recorder holds any number of finished episodes.
    The configs are captured at construction and again at the start of every
    episode. An episode whose configs differ from the recorder's keeps its own
    copy in the saved summary.

    Example::

        rec = EpisodeRecorder(eval_task, sim, planner, cli_args=vars(args))
        for step in range(n_steps):
            q, q_dot = sim.GetState()
            t0 = time.perf_counter()
            u, sigma = planner.Plan(q, q_dot)
            rec.recordStateAndAction(q, q_dot, u, sigma, time.perf_counter() - t0)
            ...
        rec.episodeFinished("success")
        rec.Save("results/run.json")
    """

    def __init__(self, task, simulator, planner, **metadata):
        """Bind the recorder to what runs the episodes.

        Args:
            task: The eval task the episodes are scored against.
            simulator: The eval simulator the states come from.
            planner: The planner choosing the actions. Its rollout simulator,
                ``planner.sim``, is recorded too.
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

        The simulator's clock is recorded alongside, as ``t``.
        """
        if self._active is None:
            self._active = _Episode(id=uuid.uuid4().hex[:8], configs=self._snapshot())
        ep = self._active
        u = np.asarray(U, dtype=np.float64).ravel()
        ep.t.append(float(np.asarray(getattr(self.simulator, "time", np.nan)).ravel()[0]))
        ep.q.append(np.asarray(q, dtype=np.float64).ravel())
        ep.q_dot.append(np.asarray(q_dot, dtype=np.float64).ravel())
        ep.u.append(u)
        ep.sigma_u.append(np.full(u.shape, np.nan) if sigma_U is None
                          else np.asarray(sigma_U, dtype=np.float64).ravel())
        ep.planning_time.append(np.nan if planning_time is None else float(planning_time))

    def episodeFinished(self, finish_reason: str, **summary) -> None:
        """End the current episode.

        Args:
            finish_reason: Why it ended, e.g. ``"success"``, ``"failure"`` or
                ``"timeout"``.
            **summary: Anything else to save in the episode's summary entry
                (goals reached, goal errors, and so on).

        With no step recorded since the last episode, this records an episode
        of zero steps: one that ended before its first action (it started out
        failed, or already at its goal) is still an episode.
        """
        if self._active is None:
            self._active = _Episode(id=uuid.uuid4().hex[:8], configs=self._snapshot())
        self._active.finish_reason = str(finish_reason)
        self._active.summary = _jsonable(summary)
        self.episodes.append(self._active)
        self._active = None

    # -- output --------------------------------------------------------------
    def Save(self, path: str | Path) -> Path:
        """Write the summary JSON to ``path`` and one ``.npy`` per episode beside it.

        Each episode's arrays go to ``<stem>_<episode id>.npy`` in the same
        directory, and the JSON names each file. Only finished episodes are
        saved.

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
        entries = []
        for ep in self.episodes:
            npy = f"{path.stem}_{ep.id}.npy"
            np.save(path.parent / npy, ep.toArray(self._dims()))
            pt = np.asarray(ep.planning_time, dtype=float)
            entry = {
                "id": ep.id,
                "file": npy,
                "finish_reason": ep.finish_reason,
                "n_steps": ep.n_steps,
                "planning_time_mean": float(np.nanmean(pt)) if np.isfinite(pt).any() else None,
                "planning_time_max": float(np.nanmax(pt)) if np.isfinite(pt).any() else None,
                "summary": ep.summary,
            }
            if ep.configs != self.configs:
                entry["configs"] = ep.configs
            entries.append(entry)
        reasons = [ep.finish_reason for ep in self.episodes]
        doc = {
            "format_version": FORMAT_VERSION,
            "configs": self.configs,
            "n_episodes": len(self.episodes),
            "finish_reasons": {r: reasons.count(r) for r in sorted(set(reasons))},
            "episodes": entries,
        }
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
        t, q, q_dot, u, sigma_u, planning_time: Per-step arrays, one row per
            control step, as recorded.
    """

    id: str
    finish_reason: str
    summary: dict
    configs: dict
    t: np.ndarray
    q: np.ndarray
    q_dot: np.ndarray
    u: np.ndarray
    sigma_u: np.ndarray
    planning_time: np.ndarray

    def __len__(self) -> int:
        return len(self.q)


class EpisodeReplayer:
    """Reads back a batch saved by ``EpisodeRecorder.Save``.

    Iterating over it yields episodes. ``iterStates`` walks every step of every
    episode instead::

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
        if version != FORMAT_VERSION:
            raise ValueError(f"{self.path} has format version {version}, "
                             f"this reader understands {FORMAT_VERSION}")
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
            arr = np.load(self.path.parent / e["file"])
            self._cache[i] = RecordedEpisode(
                id=e["id"], finish_reason=e["finish_reason"], summary=e.get("summary", {}),
                configs=e.get("configs", self.configs),
                **{name: arr[name] for name in arr.dtype.names},
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
        """``(episode_id, step, q, q_dot, u)`` for every step of every episode."""
        for ep in self.iterEpisodes():
            for k in range(len(ep)):
                yield ep.id, k, ep.q[k], ep.q_dot[k], ep.u[k]
