"""Check whether the reference Duck grasp is stable without a planner.

The evaluation simulator receives the fixed initial hand command for the settle
phase and every subsequent control period. If this test is stable but an MPPI
episode drops the Duck, the instability was introduced by planned actions, not
by an invalid initial grasp.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import warp as wp

import contact_study.tasks  # noqa: F401 - task registration
from contact_study.evaluation import json_io
from contact_study.tasks.base import get_task
from contact_study.tasks.config import EvalSimulatorKind, TaskRole


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default="duck_low_high")
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--control_dt", type=float, default=0.064)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    wp.init()  # task.load() publishes cost indices to CUDA even in this no-plan test
    task = get_task("grasp_reorient", geometry=args.geometry, role=TaskRole.EVAL)
    task.load()
    task.config.eval_sim = EvalSimulatorKind.MUJOCO
    sim = task.make_eval_simulator(render=False)
    q0, v0, u0 = task.get_inital_state(np.random.default_rng(args.seed))
    sim.reset(q0, v0)

    eval_dt = task.config.timestep
    rollout_substeps = task.config.eval_substeps_per_rollout
    rollout_dt = eval_dt * rollout_substeps
    for _ in range(int(args.settle / rollout_dt)):
        sim.apply_control(u0)
        sim.step(rollout_substeps)

    initial = sim.get_state()
    obj_qpos_adr = 16
    eval_steps_per_control = int(round(args.control_dt / eval_dt))
    zs: list[float] = []
    failed_step = None
    for step in range(args.max_steps):
        state = sim.get_state()
        z = float(state.qpos[obj_qpos_adr + 2])
        zs.append(z)
        if z < 0.0:
            failed_step = step
            break
        sim.apply_control(u0)
        sim.step(eval_steps_per_control)

    final = sim.get_state()
    initial_pos = np.asarray(initial.qpos[obj_qpos_adr : obj_qpos_adr + 3])
    final_pos = np.asarray(final.qpos[obj_qpos_adr : obj_qpos_adr + 3])
    initial_quat = np.asarray(initial.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7])
    final_quat = np.asarray(final.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7])
    quat_change = float(1.0 - np.dot(initial_quat, final_quat) ** 2)

    payload = {
        "purpose": "reference-eval Duck initial-grasp stability without planning",
        "geometry_selector": args.geometry,
        "eval_geometry": "scenes/leap/env_leap_eval_duck.xml",
        "eval_simulator": "MuJoCo",
        "control": "fixed initial hand command; no planner",
        "configuration": {
            "settle_seconds": args.settle,
            "control_dt_seconds": args.control_dt,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        "result": {
            "stable": failed_step is None,
            "failed_step": failed_step,
            "steps_run": len(zs),
            "simulated_seconds_after_settle": len(zs) * args.control_dt,
            "object_z_after_settle": zs[0],
            "object_z_final": float(final_pos[2]),
            "object_z_min": min(zs),
            "object_z_max": max(zs),
            "object_position_drift_m": float(np.linalg.norm(final_pos - initial_pos)),
            "object_quaternion_change": quat_change,
            "final_object_speed": float(np.linalg.norm(final.qvel[16:22])),
        },
        "software": {"warp": wp.__version__},
    }
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or HERE / "results" / f"duck_hold_stability_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    json_io.dump(payload, output)
    print(json_io.dumps(payload))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
