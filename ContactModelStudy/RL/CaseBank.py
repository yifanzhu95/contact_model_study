"""Explicit initial states and goals, reusable across backends and policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpisodeCase:
    case_id: str
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    control: tuple[float, ...]
    goal_position: tuple[float, ...]
    goal_quaternion: tuple[float, ...]

    def __post_init__(self):
        if not self.case_id:
            raise ValueError("case_id must be nonempty")
        for key, size in (("qpos", 23), ("qvel", 22), ("control", 16),
                          ("goal_position", 3), ("goal_quaternion", 4)):
            arr = np.asarray(getattr(self, key), dtype=float)
            if arr.shape != (size,) or not np.isfinite(arr).all():
                raise ValueError(f"{key} must contain {size} finite values")
            object.__setattr__(self, key, tuple(float(x) for x in arr))
        for quat in (self.qpos[19:23], self.goal_quaternion):
            if not np.isclose(np.linalg.norm(quat), 1, atol=1e-6):
                raise ValueError("Case quaternions must be unit wxyz")
        if self.qpos[18] < 0:
            raise ValueError("Initial object position is already below the failure plane")


@dataclass(frozen=True)
class CaseBank:
    cases: tuple[EpisodeCase, ...]
    task_name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "cases", tuple(self.cases))
        if not self.cases or len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("A case bank must be nonempty and have unique IDs")
        if self.task_name is not None:
            from ContactModelStudy.Tasks.Registry import TASK_MODULES
            if self.task_name not in TASK_MODULES:
                raise ValueError("Unknown case-bank task")

    def json_text(self):
        payload = {"schema": "leap_case_bank_v1", "cases": [asdict(c) for c in self.cases]}
        if self.task_name is not None:
            payload.update(schema="leap_case_bank_v2", task_name=self.task_name)
        return json.dumps(payload,
                          sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def sha256(self):
        return hashlib.sha256(self.json_text().encode()).hexdigest()

    def save(self, path: str | Path):
        Path(path).write_text(self.json_text() + "\n")

    def require_task(self, task_name):
        # Historical v1 banks were generated only for cube.
        if (self.task_name or "cube_reorient") != task_name:
            raise ValueError("Case-bank task differs from the selected environment task")

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text())
        if data.get("schema") not in ("leap_case_bank_v1", "leap_case_bank_v2"):
            raise ValueError("Unknown case bank schema")
        name = data.get("task_name") if data["schema"] == "leap_case_bank_v2" else None
        if data["schema"] == "leap_case_bank_v2" and not isinstance(name, str):
            raise ValueError("Version-2 case banks require task_name")
        return cls(tuple(EpisodeCase(**c) for c in data["cases"]), task_name=name)

    @classmethod
    def development(cls, *, count: int, seed: int, difficulty: int, task_name=None):
        """Diagnostic bank only; separate train/selection/test banks are not created here."""
        from ContactModelStudy.Tasks.Registry import make_task
        if count < 1:
            raise ValueError("count must be positive")
        task = make_task(task_name or "cube_reorient", role="eval", seed=seed, goal_difficulty=difficulty)
        q, v, u = task.getInitialState()
        start = task.getGoal()
        return cls(tuple(EpisodeCase(f"dev-{i:05d}", tuple(q), tuple(v), tuple(u),
                                    tuple(task.params["target_pos"]),
                                    tuple(task.sampleNewGoal(current_goal=start)))
                         for i in range(count)), task_name=task_name)
