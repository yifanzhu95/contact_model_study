#!/usr/bin/env python3
"""Cost of closed-loop trajectories against planning horizon, on an M5 world.

For every planning horizon h, an MPPI planner on the model with horizon h
controls its own copy of the eval sim for W control steps from a start state,
replanning every step (receding horizon). The start states come from one of:

- a driver (default): the model at the window length W runs episodes on an M5
  world, and every ``--every`` steps its state becomes a start state. Each is
  repeated K times with different planner noise seeds; every repeat starts from
  the same state, control and warm-start mean (the driver's, first h rows), so
  h <= W.
- ``--seeds N``: N trajectories from the task's initial pose, settled as an
  episode is, each with seed i's goal and planner noise
  (``EpisodePool.episodeSeeds(--seed, i)``: the first goal of episode i of a
  ``run_episodes.py`` run with the same ``--seed``), and the planner's own
  initial mean. The seeds are the same for every model, period and horizon, so
  the comparisons are paired. Each seed is one start state with K = 1, and h
  may exceed W. ``--goal-face F`` gives every seed one goal instead, the start
  turned to show face F on top (``LeapReorient.goalShowing``; "S" is the
  upside-down flip), so the seeds differ only in planner noise.

Two closed loops:

- sync (default): the eval sim waits for every Plan(), as ``run_episodes.py``.
- ``--async``: the loop of ``Drivers/run_episodes_async.py``. Each Plan() is
  timed on the wall clock and that much sim time passes while the executor
  plays the latest plan's rows (``--executor``, default tape), so a planner
  slower than its control period falls behind. Latency is measured, so run one
  job at a time on an otherwise idle GPU. Every planner is warmed up (compile
  and graph capture) before the first window, with its state restored.

Each trajectory is scored as the planner's rollout scores one: the task cost
after each of the W steps, the last step's being the terminal cost. Kept raw in
the .npz (``run`` (S, nh, K, W), ``term`` (S, nh, K)), so running and terminal
can be split any way afterwards, with the first step each trajectory failed
(``fail_at``) or reached the goal (``success_at``), -1 if never. The .json
holds the free energy over the K trajectories at the planner's temperature, per
start state (with K = 1, the trajectory's cost):

  f_avg   F of (sum of W - 1 running + terminal) / W
  f_run   F of the running cost per step over the first W - 1 steps
  f_runW  F of the running cost per step over all W steps
  f_term  F of the terminal cost

``cost_stats``, the mean over start states of each (the mean over the K
repeats of each start) with its standard error and 95% t-interval, the
fraction of trajectories that failed or reached the goal, and ``plan_times``,
the mean Plan() seconds per window. ``--async`` adds the async statistics per
window (``async_*``, and ``async_stats`` in the .npz).

``--noise-sigma`` is the noise std per control step at every period. Under
``--brownian-sigma`` it is the std over ``--sigma-ref-dt`` (64 ms) and scales
as a Brownian increment, sigma * sqrt(dt / 64 ms), so the noise variance per
second of plan is the same at every control period.

The eval sim's state is set from (q, q_dot); solver warm starts inside the eval
sim are not restored, so repeats differ by that as well as by planner noise.
Any other flag goes to the driver's parser (``run_episodes.py``, or
``run_episodes_async.py`` under ``--async``).

Usage:
    python experiments/horizon_closed_loop.py --model M2 --mean-cost-over-horizon \\
        --temperature 0.1 --out results/horizon_kl_sweep/cl_M2_test.json
    python experiments/horizon_closed_loop.py --async --model M2 ... --out .../acl_M2_test.json
    python experiments/horizon_closed_loop.py --model M2 --seeds 10 --window 16 --horizons 1 2 4 8 20 \\
        --mean-cost-over-horizon --temperature 0.1 --out .../cl_M2_seeds10.json
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import horizon_cl_results as hcr  # noqa: E402
from ContactModelStudy.Drivers import run_episodes as drv  # noqa: E402
from ContactModelStudy.Drivers import run_episodes_async as drv_async  # noqa: E402
from ContactModelStudy.Drivers.EpisodePool import episodeSeeds  # noqa: E402
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI  # noqa: E402
from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim  # noqa: E402
from ContactModelStudy.Utils.EvalSimulators import makeEvalSim  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10, 12, 16, 20])
ap.add_argument("--window", type=int, default=None, help="closed-loop steps W (default: max horizon)")
ap.add_argument("--model", default="M2")
ap.add_argument("--repeats", type=int, default=8, help="K closed-loop runs per start state and horizon")
ap.add_argument("--steps", type=int, default=25, help="driver steps per episode")
ap.add_argument("--episodes", type=int, default=4)
ap.add_argument("--every", type=int, default=5, help="take a start state every this many driver steps")
ap.add_argument("--mean-cost-over-horizon", action="store_true")
ap.add_argument("--async", dest="async_", action="store_true",
                help="planning time is spent as sim time (Drivers/run_episodes_async.py)")
ap.add_argument("--brownian-sigma", action="store_true",
                help="scale --noise-sigma with the control period as Brownian motion: "
                     "sigma * sqrt(dt / --sigma-ref-dt) per control step")
ap.add_argument("--sigma-ref-dt", type=float, default=0.064,
                help="control period (s) at which --noise-sigma applies unscaled under --brownian-sigma")
ap.add_argument("--seeds", type=int, default=None, metavar="N",
                help="start from the initial pose with N seeds' goals and noise, instead of a driver")
ap.add_argument("--goal-face", default=None, metavar="F",
                help="with --seeds: every seed's goal is the start turned to show face F on top (e.g. S)")
ap.add_argument("--out", required=True)
opts, rest = ap.parse_known_args()
assert opts.seeds or not opts.goal_face, "--goal-face needs --seeds"
W = opts.window or max(opts.horizons)
if opts.seeds:
    opts.repeats = 1
else:
    assert max(opts.horizons) <= W, "every planning horizon must be <= the window (the driver's warm start)"
HS = opts.horizons
H_drive = W
PARSER = drv_async.build_parser() if opts.async_ else None


def plannerArgs(h, seed):
    a = drv.parseArgs(["--rollout-model", opts.model, "--horizon", str(h), "--seed", str(seed),
                       "--eval-sim", args.eval_sim, "--no-video", "--no-results"] + rest, parser=PARSER)
    if h < 2:
        a.warm_start = "hold"          # shift needs H >= 2
    return a


def noiseSigma(sigma, control_dt):
    """The noise per control step: sigma, or under --brownian-sigma the std of a Brownian
    increment over control_dt with sigma over --sigma-ref-dt (variance linear in time)."""
    return sigma * np.sqrt(control_dt / opts.sigma_ref_dt) if opts.brownian_sigma else sigma


def make_planner(h, seed):
    a = plannerArgs(h, seed)
    s = GetContactModelSim(a.rollout_model, task.getModelPath(), N=a.n_samples,
                           task=task, **drv.rolloutSimOverrides(a, task))
    cfg = drv.buildPlannerConfig(a)
    cfg.noise_sigma = float(noiseSigma(a.noise_sigma, s.config.control_timestep))
    cfg.mean_cost_over_horizon = opts.mean_cost_over_horizon
    return MPPI(s, task, cfg), a


class NoRecorder:
    def recordGoal(self, goal):
        pass


args = drv.parseArgs(["--rollout-model", opts.model, "--horizon", str(H_drive), "--eval-sim", "M5",
                      "--no-video", "--no-results", "--no-stop-on-success"] + rest, parser=PARSER)
task, eval_task = drv.buildTasks(args)
cl = makeEvalSim(args.eval_sim, eval_task.getModelPath(), eval_task.timestep, task=eval_task)
if not opts.seeds:
    world = makeEvalSim(args.eval_sim, eval_task.getModelPath(), eval_task.timestep, task=eval_task)
    driver, _ = make_planner(H_drive, args.seed)
planners = {h: make_planner(h, args.seed + 1) for h in HS}
ref = planners[HS[0]][0] if opts.seeds else driver     # for the settings every planner shares
lam = ref.lam
eval_steps = drv.controlEvalSteps(task, eval_task, ref)
print(f"{f'{opts.seeds} seeds from the initial pose' if opts.seeds else 'driver ' + repr(driver)}\n"
      f"world/closed loop {args.eval_sim}, W={W}, K={opts.repeats}, T={lam}, "
      f"ctrl dt={ref.sim.config.control_timestep * 1e3:g} ms, sigma={ref.config.noise_sigma:.4g}"
      + (f" (Brownian: {args.noise_sigma:g} at {opts.sigma_ref_dt * 1e3:g} ms)" if opts.brownian_sigma else "")
      + (f", async executor {args.executor}" if opts.async_ else ", sync"), flush=True)
if opts.async_:
    loops = {h: drv_async.AsyncLoop(p, a, eval_steps, eval_task.timestep) for h, (p, a) in planners.items()}
    clock = drv.EvalClock(cl, None)
    for h, (p, _) in planners.items():
        w = drv_async.warmUp(args.plan_warmup, eval_task, cl, p)
        print(f"  warm-up h={h}: " + ", ".join(f"{m:.0f}" for m in w) + " ms", flush=True)


def stepCost(terminal):
    return float(eval_task.calcCosts(cl.inner, terminal=terminal).numpy()[0])


def startWindow(p, h, q0, qd0, u0, U0, noise_seed):
    """Reset the planner (to U0's first h rows as its mean, if given) and the eval sim to (q0, qd0, u0)."""
    p.Reset(None)
    p._noise_seed = int(noise_seed)
    if U0 is not None:
        p.U_wp.assign(np.ascontiguousarray(U0[:h]))
    cl.SetState(q0, qd0)
    cl.SetControl(u0)


def closedLoop(h, q0, qd0, u0, U0, noise_seed):
    """W receding-horizon steps of the horizon-h planner on the eval sim from (q0, qd0, u0).

    Returns (running cost per step (W,), terminal cost, first failed step or -1, first step at
    the goal or -1, stats); stats is (mean Plan() seconds,) in sync, STAT_KEYS in async.
    """
    p, _ = planners[h]
    startWindow(p, h, q0, qd0, u0, U0, noise_seed)
    run = np.full(W, np.nan)
    term, fail_at, success_at, t_plan = np.nan, -1, -1, 0.0
    for t in range(W):
        q, qd = cl.GetState()
        t0 = time.perf_counter()
        u = p.Plan(q, qd, u=cl.GetControl())
        t_plan += time.perf_counter() - t0
        cl.SetControl(u)
        cl.Step(eval_steps)
        run[t] = stepCost(False)
        if t == W - 1:
            term = stepCost(True)
        if fail_at < 0 and eval_task.isFailure(cl):
            fail_at = t
        if success_at < 0 and eval_task.isSuccess(cl):
            success_at = t
    return run, term, fail_at, success_at, (t_plan / W,)


def asyncClosedLoop(h, q0, qd0, u0, U0, noise_seed):
    """closedLoop, with planning time spent as sim time: W control ticks of sim time."""
    p, _ = planners[h]
    loop = loops[h]
    startWindow(p, h, q0, qd0, u0, U0, noise_seed)
    run = np.full(W, np.nan)
    fail_at, success_at = [-1], [-1]

    def check(t):
        if fail_at[0] < 0 and eval_task.isFailure(cl):
            fail_at[0] = t
        if success_at[0] < 0 and eval_task.isSuccess(cl):
            success_at[0] = t

    def on_tick(i):
        # Tick i sees the state after control step i - 1.
        if i > 0:
            run[i - 1] = stepCost(False)
            check(i - 1)
        return None

    loop.resetStats()
    loop.run(cl, clock, W, on_tick)
    run[W - 1] = stepCost(False)
    term = stepCost(True)
    check(W - 1)
    s = loop.stats()
    st = (float(np.mean(loop.plan_seconds)), s["n_plans"], s["stale_ticks"], s["held_ticks"],
          s["tape_exhausted_ticks"], s["staleness_ms_mean"] or np.nan)
    return run, term, fail_at[0], success_at[0], st


STAT_KEYS = hcr.STAT_KEYS_ASYNC if opts.async_ else hcr.STAT_KEYS_SYNC


def driverStarts():
    """Run the driver episodes; every --every steps its state becomes a start state.

    Returns the start states as arrays: ``starts`` (S, 2) (episode, step), ``q``, ``qd``, ``u``
    (the control then), ``U`` (S, W, nu) (the driver's warm-start mean), ``goal`` and ``seeds``
    (S, K) (the repeats' noise seeds).
    """
    seed_rng = np.random.default_rng(args.seed + 0xC10)
    out = {k: [] for k in ("starts", "q", "qd", "u", "U", "goal", "seeds")}
    t_start = time.time()
    for ep in range(opts.episodes):
        eval_task.setSimToInitialState(world)
        drv._newGoal(task, eval_task, None, NoRecorder())
        driver.Reset(None)
        world.Step(int(round(args.settle / eval_task.timestep)))
        for step in range(opts.steps):
            if eval_task.isFailure(world):
                print(f"  ep {ep}: driver failure at step {step}")
                break
            if eval_task.isSuccess(world):
                drv._newGoal(task, eval_task, None, NoRecorder())
                driver.Reset(None)
            q, qd = world.GetState()
            u_now = world.GetControl()
            u = driver.Plan(q, qd, u=u_now)
            if driver.last_plan_ok and step % opts.every == 0:
                for k, v in (("starts", (ep, step)), ("q", q), ("qd", qd), ("u", u_now),
                             ("U", np.asarray(driver.last_plan_start["U"], dtype=np.float32)),
                             ("goal", eval_task.getGoal()), ("seeds", seed_rng.integers(0, 2**31 - 1, opts.repeats))):
                    out[k].append(np.array(v, copy=True))
            world.SetControl(u)
            world.Step(eval_steps)
        print(f"ep {ep} driver done ({time.time() - t_start:.0f} s)", flush=True)
    return {k: np.array(v) for k, v in out.items()}


def seedStarts():
    """The --seeds start states, laid out as driverStarts' without ``U``: the initial pose, settled
    as an episode is, with seed i's goal (or the --goal-face goal) and planner noise seed."""
    out = {"starts": [], "goal": [], "seeds": []}
    for i in range(opts.seeds):
        goal_seed, noise_seed = episodeSeeds(args.seed, i)
        eval_task.setSimToInitialState(cl)          # the goal is drawn from the starting orientation
        eval_task.reseed(goal_seed)
        goal = eval_task.goalShowing(opts.goal_face) if opts.goal_face else eval_task.sampleNewGoal()
        out["starts"].append((i, 0))
        out["goal"].append(np.array(goal, copy=True))
        out["seeds"].append([noise_seed])
    eval_task.setSimToInitialState(cl)
    cl.Step(int(round(args.settle / eval_task.timestep)))
    q, qd = cl.GetState()
    state = {"q": np.array(q, copy=True), "qd": np.array(qd, copy=True), "u": np.array(cl.GetControl(), copy=True)}
    return {**{k: np.array(v) for k, v in out.items()},
            **{k: np.repeat(v[None], opts.seeds, axis=0) for k, v in state.items()}}


S0 = seedStarts() if opts.seeds else driverStarts()
if opts.goal_face:
    print(f"every seed's goal: {S0['goal'][0].round(4).tolist()}, face {eval_task.goalFace(S0['goal'][0])} up",
          flush=True)
window = asyncClosedLoop if opts.async_ else closedLoop
starts = [tuple(int(x) for x in s) for s in S0["starts"]]
RUN, TERM, FAIL, SUCC, ST = [], [], [], [], []
t_start = time.time()
for s, (ep, step) in enumerate(starts):
    eval_task.setGoal(S0["goal"][s])
    task.setGoal(S0["goal"][s])
    q, qd, u_now, seeds = (S0[k][s] for k in ("q", "qd", "u", "seeds"))
    U0 = S0["U"][s] if "U" in S0 else None
    run = np.full((len(HS), opts.repeats, W), np.nan)
    term = np.full((len(HS), opts.repeats), np.nan)
    fail = np.full((len(HS), opts.repeats), -1)
    succ = np.full((len(HS), opts.repeats), -1)
    st = np.full((len(HS), opts.repeats, len(STAT_KEYS)), np.nan)
    for i, h in enumerate(HS):
        for k in range(opts.repeats):      # repeat k uses the same noise seed at every h
            run[i, k], term[i, k], fail[i, k], succ[i, k], st[i, k] = window(h, q, qd, u_now, U0, seeds[k])
    RUN.append(run); TERM.append(term); FAIL.append(fail); SUCC.append(succ); ST.append(st)
    J = (run[:, :, :W - 1].sum(-1) + term) / W
    print(f"  {'seed' if opts.seeds else 'ep'} {ep} step {step:3d}  start {s + 1:3d}/{len(starts)}  "
          f"{time.time() - t_start:6.0f} s  "
          "mean (run+term)/W " + " ".join(f"h{h}:{J[i].mean():.3g}" for i, h in enumerate(HS))
          + "  plan ms " + " ".join(f"h{h}:{1e3 * st[i, :, 0].mean():.0f}" for i, h in enumerate(HS)),
          flush=True)

# (S, nh, K, W), (S, nh, K), ..., (S, nh, K, stats)
RUN, TERM, FAIL, SUCC, ST = map(np.array, (RUN, TERM, FAIL, SUCC, ST))
meta = {"model": opts.model, "world": args.eval_sim,
        "control_dt": ref.sim.config.control_timestep, "n_samples": args.n_samples,
        "temperature": lam, "noise_sigma": ref.config.noise_sigma,
        "sigma_scaling": "brownian" if opts.brownian_sigma else "none",
        "noise_sigma_ref": args.noise_sigma, "sigma_ref_dt": opts.sigma_ref_dt,
        "mean_cost_over_horizon": opts.mean_cost_over_horizon,
        "driver": "async" if opts.async_ else "sync",
        "start": "seeds" if opts.seeds else "driver", "seed": args.seed, "goal_face": opts.goal_face}
if opts.async_:
    meta.update({"executor": args.executor, "plan_latency_ms": args.plan_latency_ms,
                 "latency_scale": args.latency_scale, "async_shift": args.async_shift})
outp = Path(opts.out)
hcr.save(outp, meta, RUN, TERM, FAIL, ST, HS, starts, STAT_KEYS, success=SUCC)
cost = {k: np.nanmean(v, -1) for k, v in hcr.views(RUN, TERM).items()}   # mean over the K repeats: (S, nh)
n = len(starts)
print(f"\n{n} start states x K={opts.repeats}, window W={W}; mean ± standard error over start states")
print("h   cost (run+term)/W   cost run/(W-1)       cost term   fail  goal   plan ms"
      + ("   stale ticks/window" if opts.async_ else ""))


def meanSe(x):
    return f"{np.nanmean(x):7.4g} ± {np.nanstd(x, ddof=1) / np.sqrt(n) if n > 1 else np.nan:<6.3g}"


for i, h in enumerate(HS):
    print(f"{h:2d}  {meanSe(cost['f_avg'][:, i])}  {meanSe(cost['f_run'][:, i])}  {meanSe(cost['f_term'][:, i])}  "
          f"{(FAIL[:, i] >= 0).mean():4.2f}  {(SUCC[:, i] >= 0).mean():4.2f}  {1e3 * np.mean(ST[:, i, :, 0]):7.1f}"
          + (f"  {np.mean(ST[:, i, :, 2]):6.1f}" if opts.async_ else ""))
print(f"wrote {outp}")
