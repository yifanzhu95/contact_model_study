"""The recorded trajectory must be enough to replay the episode.

Four checks, in order of how much they need:

  A. replay identity   ctrl[k] must be reconstructible from qpos[k] and
                       action[k] alone, using only what trajectory["context"]
                       records. This is THE property that makes the record
                       replayable; it needs no simulator beyond the one that
                       produced the record, and it runs for all three planners
                       and both control parameterizations.

  B. open-loop replay  Re-driving a fresh eval sim from steps.qpos[0] with the
                       recorded ctrl column must reproduce the recorded states.
                       Note this correctly SKIPS the settle: step 0's state is
                       the post-settle state.

  C. planner coverage  planner_dist must be populated and well-formed for mppi,
                       cem and the predictive sampler — the last being a greedy
                       argmin with no weight vector at all, hence degenerate.

  D. end reasons       success / failed / timeout / multi-goal, on both drivers,
                       forced by patching is_success and has_failed rather than
                       by trying to make physics cooperate.

Why cart_pole with the MuJoCo eval sim: contact-rich tasks are not reproducible
run-to-run, so B can only be exact here — see the discussion at the top of
tests/test_async_sync_equivalence.py.

Needs a CUDA device. Run directly (the repo has no pytest):

    python tests/test_trajectory_replay.py
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
from contact_study.evaluation.trajectory import TrajectoryConfig
from contact_study.planners import make_planner_config
from contact_study.tasks.base import TaskRole, get_task
from contact_study.tasks.config import EvalSimulatorKind

TASK      = "cart_pole"
EVAL_SIM  = EvalSimulatorKind.MUJOCO
N_SAMPLES = 32
SEED      = 7

# precision=0 disables the significant-digit rounding: qpos/ctrl are float64 and
# would otherwise lose digits, which an exact comparison cannot tolerate.
EXACT = dict(precision=0)


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
def _patch_task(mod, max_steps, *, is_success=None, has_failed=None):
    """Cap the episode length and optionally force the task's exit conditions.

    Physics cannot be made to succeed or drop the cart-pole on cue, so the exit
    predicates are replaced outright. Both the rollout and the eval task get the
    same bound function; only the rollout task's is called by the drivers, so a
    call-counting predicate counts control steps exactly.
    """
    orig = mod.get_task

    def patched(*a, **k):
        task = orig(*a, **k)
        task.config.max_steps = max_steps
        if is_success is not None:
            task.is_success = is_success
        if has_failed is not None:
            task.has_failed = has_failed
        return task

    mod.get_task = patched
    return orig


def _cfg(planner, *, relative, n_iterations=None):
    kw = dict(n_samples=N_SAMPLES, time_horizon=0.256, step_time=0.064,
              noise_sigma=0.2, delta_range=(None, None), warm_start=False,
              use_full_graph=True, nconmax=50, njmax=300, seed=1234, debug=False,
              resample_interval=1, ctrl_relative_to_qpos=relative)
    if planner == "mppi":
        kw["temperature"] = 20.0
    if n_iterations is not None:
        kw["n_iterations"] = n_iterations
    return make_planner_config(planner, **kw)


def _run_sync(planner, *, relative, max_steps, fin_ep_on_success=True,
              is_success=None, has_failed=None):
    orig = _patch_task(sync_mod, max_steps,
                       is_success=is_success, has_failed=has_failed)
    try:
        return sync_mod.run_eval_episode(
            task_name=TASK, contact_cfg=ContactModelConfig.M2(), planner=planner,
            planner_cfg=_cfg(planner, relative=relative),
            rng=np.random.default_rng(SEED), settle_seconds=0.0,
            eval_sim=EVAL_SIM, verbose=False,
            record=TrajectoryConfig(**EXACT),
            fin_ep_on_success=fin_ep_on_success)
    finally:
        sync_mod.get_task = orig


# ---------------------------------------------------------------------------
# A. replay identity
# ---------------------------------------------------------------------------
def check_replay_identity(traj, label: str) -> bool:
    """ctrl[k] == clip(base[k] + action[k]), rebuilt from the record alone."""
    ctx    = traj["context"]
    steps  = traj["steps"]
    qpos   = np.asarray(steps["qpos"],   dtype=float)
    action = np.asarray(steps["action"], dtype=float)
    ctrl   = np.asarray(steps["ctrl"],   dtype=float)
    adr, nu = ctx["robot_qpos_adr"], ctx["nu"]
    lo, hi  = ctx["clip_lo"], ctx["clip_hi"]

    u = np.asarray(ctx["u0"], dtype=float)
    for k in range(len(ctrl)):
        u = (qpos[k][adr:adr + nu] + action[k]) if ctx["ctrl_relative_to_qpos"] \
            else (u + action[k])
        if lo is not None:
            u = np.clip(u, lo, hi)
        if not np.array_equal(u, ctrl[k]):
            print(f"FAIL  [{label}] step {k}: rebuilt {u} != recorded {ctrl[k]}")
            return False
    print(f"ok    [{label}] replay identity holds over {len(ctrl)} steps")
    return True


def test_a_replay_identity() -> bool:
    ok = True
    for planner in ("mppi", "cem", "predictive_sampler"):
        for relative in (True, False):
            r = _run_sync(planner, relative=relative, max_steps=12)
            ok &= check_replay_identity(
                r.trajectory, f"{planner} relative={relative}")
    return ok


# ---------------------------------------------------------------------------
# B. open-loop replay
# ---------------------------------------------------------------------------
def test_b_open_loop() -> bool:
    r = _run_sync("mppi", relative=True, max_steps=20)
    traj  = r.trajectory
    steps = traj["steps"]
    qpos  = np.asarray(steps["qpos"], dtype=float)
    qvel  = np.asarray(steps["qvel"], dtype=float)
    ctrl  = np.asarray(steps["ctrl"], dtype=float)
    n_eval = traj["context"]["eval_steps_per_control"]

    task = get_task(TASK, role=TaskRole.EVAL)
    task.load()
    task.config.eval_sim = EVAL_SIM
    sim = task.make_eval_simulator(video_path=None, render=False)
    # steps.qpos[0] IS the post-settle state, so replay starts there and the
    # settle is correctly skipped.
    sim.reset(qpos[0].copy(), qvel[0].copy())

    worst = 0.0
    for k in range(len(ctrl) - 1):
        sim.apply_control(ctrl[k])
        sim.step(n_eval)
        st = sim.get_state()
        worst = max(worst, float(np.abs(st.qpos - qpos[k + 1]).max()),
                           float(np.abs(st.qvel - qvel[k + 1]).max()))

    if worst > 1e-9:
        print(f"FAIL  open-loop replay drifted: max|d|={worst:.3e}")
        return False
    print(f"ok    open-loop replay reproduces {len(ctrl) - 1} states "
          f"(max|d|={worst:.3e})")
    return True


# ---------------------------------------------------------------------------
# C. planner coverage
# ---------------------------------------------------------------------------
def test_c_planner_dist() -> bool:
    ok = True
    for planner, expect_degenerate in (("mppi", False), ("cem", False),
                                       ("predictive_sampler", True)):
        r  = _run_sync(planner, relative=True, max_steps=6)
        pd = r.trajectory.get("planner_dist")
        if pd is None:
            print(f"FAIL  [{planner}] no planner_dist block")
            ok = False
            continue

        nu   = r.trajectory["context"]["nu"]
        mu   = np.asarray(pd["mu"],  dtype=float)
        cov  = np.asarray(pd["cov"], dtype=float)
        ess  = np.asarray(pd["ess"], dtype=float)
        seq  = np.asarray(pd["mean_seq"], dtype=float)
        degen = np.asarray(pd["degenerate"], dtype=bool)

        problems = []
        if pd["kind"] != planner:
            problems.append(f"kind={pd['kind']!r} != {planner!r}")
        if mu.shape[1:] != (nu,):
            problems.append(f"mu shape {mu.shape}")
        if cov.shape[1:] != (nu, nu):
            problems.append(f"cov shape {cov.shape}")
        if not np.allclose(cov, np.swapaxes(cov, 1, 2)):
            problems.append("cov not symmetric")
        if not np.isfinite(ess).all():
            problems.append("ess not finite")
        if seq.shape[1:] != (r.trajectory["context"]["horizon"], nu):
            problems.append(f"mean_seq shape {seq.shape}")
        if bool(degen.all()) != expect_degenerate:
            problems.append(f"degenerate={degen.tolist()} "
                            f"(expected all {expect_degenerate})")

        if problems:
            print(f"FAIL  [{planner}] " + "; ".join(problems))
            ok = False
        else:
            print(f"ok    [{planner}] planner_dist  n={len(mu)}  mu{mu.shape}  "
                  f"cov{cov.shape}  ess={ess[0]:.2f}  degenerate={expect_degenerate}")
    return ok


# ---------------------------------------------------------------------------
# D. end reasons
# ---------------------------------------------------------------------------
def _run_async(max_steps, *, fin_ep_on_success=True,
               is_success=None, has_failed=None):
    orig = _patch_task(async_mod, max_steps,
                       is_success=is_success, has_failed=has_failed)
    try:
        return async_mod.run_async_eval_episode(
            task_name=TASK, contact_cfg=ContactModelConfig.M2(), planner="mppi",
            planner_cfg=_cfg("mppi", relative=True),
            rng=np.random.default_rng(SEED), settle_seconds=0.0,
            eval_sim=EVAL_SIM, verbose=False, plan_latency_ms=0.0, plan_warmup=0,
            record=TrajectoryConfig(**EXACT), fin_ep_on_success=fin_ep_on_success)
    finally:
        async_mod.get_task = orig


def _at_step(n):
    """A task predicate that fires on its n-th call and every call after."""
    state = {"i": 0}

    def pred(mjd):
        state["i"] += 1
        return state["i"] > n
    return pred


def _never(mjd):
    return False


def test_d_end_reasons() -> bool:
    # The predicates count calls, so each driver run needs its own — hence a
    # factory per case rather than one shared dict.
    cases = [
        ("success",
         lambda: dict(is_success=_at_step(7), has_failed=_never),
         dict(max_steps=20), ("success", False, 7)),
        ("failed",
         lambda: dict(is_success=_never, has_failed=_at_step(5)),
         dict(max_steps=20), ("failed", False, 5)),
        ("timeout",
         lambda: dict(is_success=_never, has_failed=_never),
         dict(max_steps=6), ("timeout", True, 6)),
        ("multi-goal",
         lambda: dict(is_success=_at_step(7), has_failed=_never),
         dict(max_steps=20, fin_ep_on_success=False), ("success", True, 20)),
    ]

    ok = True
    for name, make_patches, runkw, expect in cases:
        fin = runkw.get("fin_ep_on_success", True)
        for driver in ("sync", "async"):
            if driver == "sync":
                r = _run_sync("mppi", relative=True, max_steps=runkw["max_steps"],
                              fin_ep_on_success=fin, **make_patches())
            else:
                r = _run_async(runkw["max_steps"], fin_ep_on_success=fin,
                               **make_patches())
            got   = (r.end_reason, r.time_out, r.n_steps_taken)
            steps = r.trajectory.get("steps", {}).get("step", [])
            n_rec = len(steps)
            if got != expect:
                print(f"FAIL  [{driver} {name}] {got} != {expect}")
                ok = False
            elif n_rec != r.n_steps_taken:
                print(f"FAIL  [{driver} {name}] recorded {n_rec} steps but "
                      f"n_steps_taken={r.n_steps_taken}")
                ok = False
            else:
                print(f"ok    [{driver} {name}] end_reason={got[0]!r} "
                      f"time_out={got[1]} n_steps_taken={got[2]}")
    return ok


def main() -> int:
    results = {
        "A replay identity":  test_a_replay_identity(),
        "B open-loop replay": test_b_open_loop(),
        "C planner_dist":     test_c_planner_dist(),
        "D end reasons":      test_d_end_reasons(),
    }
    n_ok = sum(results.values())
    print(f"\n{n_ok}/{len(results)} groups passed")
    for name, ok in results.items():
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
