"""Short GPU audit of the two-planner KL worker's ownership and input wiring.

Uses the worker's normal arguments plus --audit_report. Extra array downloads
make this a diagnostic run, not a timing benchmark. This checks explicitly
listed buffers and host inputs, not all backend-private state or bitwise
repeatability. Use separate processes to compare repeated real/null runs.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

from contact_study.evaluation import json_io
from experiments.hpc import run_kl_divergence_cell as worker


def digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()


def host_input(planner, data):
    return {"qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
            "ctrl": data.ctrl.copy(), "goal": planner.goal_wp.numpy().copy()}


def assert_same(left, right, context):
    for name in left:
        if not np.array_equal(left[name], right[name]):
            raise AssertionError(f"{context}: {name} changed")


def main():
    parser = worker.build_parser()
    parser.description = __doc__
    parser.set_defaults(n_episodes=1, max_steps=8, kl_every=1, n_samples=16)
    parser.add_argument("--audit_report", type=Path, required=True)
    args = parser.parse_args()
    args.geometry = worker.resolve_kl_geometry(args.task, args.geometry)
    worker.validate_kl_args(args)
    if not args.sync_reference_mean:
        parser.error("this audit requires synchronized reference means")
    args.record_precision = 0
    observations = []
    constructed = 0
    latest_degraded = None
    base = worker.MPPIController
    owned = ("U_wp", "V_wp", "w_wp", "_static_eps_wp", "qpos_reset", "qvel_reset", "ctrl_reset")

    class AuditedPlanner(base):
        def __init__(self, **kwargs):
            nonlocal constructed, latest_degraded
            env_before = copy.deepcopy(kwargs["rng"].bit_generator.state)
            super().__init__(**kwargs)
            if env_before != kwargs["rng"].bit_generator.state:
                raise AssertionError("Planner construction consumed the environment RNG")
            self.audit_reference = constructed % 2 == 1
            self.audit_episode = constructed // 2
            if self.audit_reference:
                self.audit_degraded = latest_degraded
                deg = self.audit_degraded
                for name in owned:
                    if getattr(self, name).ptr == getattr(deg, name).ptr:
                        raise AssertionError(f"Planners share mutable {name}")
                if self.m is deg.m or self.d is deg.d:
                    raise AssertionError("Planners share model/data objects")
            else:
                latest_degraded = self
            constructed += 1

        def plan(self, data):
            before = host_input(self, data)
            env_before = copy.deepcopy(self.rng.bit_generator.state)
            pre_mean = self.U_wp.numpy().copy()
            if self.audit_reference:
                deg = self.audit_degraded
                assert_same(deg.audit_input, before, "Reference input differs from degraded input")
                np.testing.assert_array_equal(pre_mean, deg.audit_pre_mean)
                owned_before = {name: getattr(deg, name).numpy().copy() for name in owned}
                counters_before = (deg._plan_count, deg._resample_count)
            action = super().plan(data)
            assert_same(before, host_input(self, data), "Planning mutated host inputs")
            if env_before != self.rng.bit_generator.state:
                raise AssertionError("Planning consumed the environment RNG")
            if self.audit_reference:
                assert_same(owned_before, {name: getattr(deg, name).numpy() for name in owned},
                            "Shadow mutated degraded buffers")
                if counters_before != (deg._plan_count, deg._resample_count):
                    raise AssertionError("Shadow advanced degraded sampling counters")
            else:
                self.audit_input, self.audit_pre_mean = before, pre_mean
            observations.append({"episode": self.audit_episode,
                                 "role": "reference" if self.audit_reference else "degraded",
                                 "plan_index": self._plan_count - 1,
                                 "input": {k: json_io.compact(v, precision=0) for k, v in before.items()},
                                 "pre_mean_sha256": digest(pre_mean),
                                 "noise_sha256": digest(self._static_eps_wp.numpy()),
                                 "action": json_io.compact(action, precision=0),
                                 "plan_ok": self.last_plan_ok})
            return action

    worker.MPPIController = AuditedPlanner
    try:
        worker.wp.init()
        result, _ = worker.run_cell(args)
    finally:
        worker.MPPIController = base
    args.audit_report.parent.mkdir(parents=True, exist_ok=True)
    json_io.dump(result, args.audit_report.with_suffix(".cell.json"), precision=0)
    json_io.dump({"assertions_passed": True,
                  "scope": "Host inputs, environment RNG, explicit planner buffers and sampling counters",
                  "limitations": "Backend-private state and cross-process repeatability are not certified",
                  "config": result["config"], "observations": observations}, args.audit_report, precision=0)
    print(f"Audit passed; {len(observations)} planner calls; saved {args.audit_report}")


if __name__ == "__main__":
    main()
