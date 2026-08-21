"""The async driver at zero latency must reproduce the synchronous driver exactly.

run_async_eval_episode with --plan_latency_ms 0 degenerates to one plan per
executor tick, applying row 0 of a plan solved at that same instant — which is
precisely run_eval_episode's plan-then-apply order. This test captures every
apply_control() from both drivers at the same seed and diffs the control tapes.

Why cart_pole and not grasp_reorient: contact-rich tasks are not reproducible
run-to-run at all. On grasp_reorient/pinocchio the SAME unmodified driver run
twice in one process diverges at step 0 (max |du| ~1.1), so no bit-exact
comparison is possible there — see the mjwarp contact-ordering discussion in
tests/README or the notes on run-to-run variance. cart_pole with the MuJoCo eval
sim has essentially no contacts and IS bit-exact, which makes it the place to
verify driver-level equivalence.

Needs a CUDA device. Run directly (the repo has no pytest):

    python tests/test_async_sync_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parents[1]))

wp.init()

import contact_study.tasks  # noqa: F401 — registers all tasks
from contact_study.contact_models.config import ContactModelConfig
from contact_study.drivers import run_async_eval_episode as async_mod
from contact_study.drivers import run_eval_episode as sync_mod
from contact_study.planners import make_planner_config
from contact_study.tasks.config import EvalSimulatorKind

TASK      = "cart_pole"
EVAL_SIM  = EvalSimulatorKind.MUJOCO
MAX_STEPS = 40
N_SAMPLES = 64
SEED      = 5


def _instrument(mod, log):
    """Patch mod.get_task so the eval sim logs every command; cap the episode."""
    orig = mod.get_task

    def patched(*a, **k):
        task = orig(*a, **k)
        task.config.max_steps = MAX_STEPS
        orig_make = task.make_eval_simulator

        def make(*aa, **kk):
            sim = orig_make(*aa, **kk)
            orig_apply = sim.apply_control

            def apply(ctrl):
                log.append(np.asarray(ctrl, dtype=float).copy())
                return orig_apply(ctrl)

            sim.apply_control = apply
            return sim

        task.make_eval_simulator = make
        return task

    mod.get_task = patched
    return orig


def _planner_cfg():
    return make_planner_config(
        "mppi", n_samples=N_SAMPLES, time_horizon=0.352, step_time=0.064,
        noise_sigma=0.2, temperature=20.0, delta_range=(None, None),
        warm_start=False, use_full_graph=True, nconmax=50, njmax=300,
        seed=1234, debug=False, resample_interval=1,
    )


def main() -> int:
    common = dict(task_name=TASK, contact_cfg=ContactModelConfig.M2(), planner="mppi",
                  settle_seconds=0.0, eval_sim=EVAL_SIM, verbose=False,
                  fin_ep_on_success=False)

    sync_log: list[np.ndarray] = []
    orig = _instrument(sync_mod, sync_log)
    r_sync = sync_mod.run_eval_episode(
        planner_cfg=_planner_cfg(), rng=np.random.default_rng(SEED), **common)
    sync_mod.get_task = orig

    async_log: list[np.ndarray] = []
    orig = _instrument(async_mod, async_log)
    # plan_warmup=0 is required: reset() deliberately does NOT rewind
    # _resample_count (it keys the noise seed), so warm-up plans would shift the
    # noise stream and the tapes would legitimately differ.
    r_async = async_mod.run_async_eval_episode(
        planner_cfg=_planner_cfg(), rng=np.random.default_rng(SEED),
        plan_latency_ms=0.0, plan_warmup=0, **common)
    async_mod.get_task = orig

    A = np.stack(sync_log) if sync_log else np.empty((0, 0))
    B = np.stack(async_log) if async_log else np.empty((0, 0))

    print(f"task={TASK}  eval_sim={EVAL_SIM.value}  max_steps={MAX_STEPS}  "
          f"n_samples={N_SAMPLES}")
    print(f"  sync {A.shape}   async {B.shape}")

    if A.shape[0] == 0:
        print("FAIL  no commands captured")
        return 1
    if A.shape != B.shape:
        print(f"FAIL  command-count mismatch: {A.shape} vs {B.shape}")
        return 1
    if not np.array_equal(A, B):
        d = np.abs(A - B)
        first = int(np.argmax(d.max(axis=1) > 0))
        print(f"FAIL  control tapes differ: max|d|={d.max():.3e}, "
              f"first differing step={first}")
        print(f"        sync [{first}] = {A[first][:4]}")
        print(f"        async[{first}] = {B[first][:4]}")
        return 1

    print("ok    control tapes identical bit-for-bit")
    if r_sync.final_cost != r_async.final_cost:
        print(f"FAIL  final_cost differs: {r_sync.final_cost!r} vs {r_async.final_cost!r}")
        return 1
    print(f"ok    final_cost identical ({r_sync.final_cost!r})")
    print("\n1/1 passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
