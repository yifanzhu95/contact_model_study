#!/usr/bin/env python3
"""Tune ComFree's (M3) stiffness and damping so its planner acts like Kamino's (M5).

Two phases:

1. **Reference.** One closed-loop episode with an MPPI planner rolling out on M5,
   scored in the M5 eval sim, exactly as ``run_episodes --rollout-model M5`` runs
   it. Before every plan the planner's state (warm-start mean, noise stream) is
   saved, with the measured state, the applied control and the goal; after it,
   the first-action distribution (``FirstActionDistribution``). ``--n-states``
   evenly spaced steps are kept.
2. **Search.** A planner rolling out on M3 is put in each saved state
   (``LoadState``: same mean, same noise samples, so only the contact model
   differs) and plans; the objective is the mean over states of
   ``KL(M5 || M3)`` between the two first-action Gaussians
   (``Utils/PlannerKLDiv.GaussianKL``, both covariances shrunk toward
   ``noise_sigma^2 I`` as ``CalcPlannerKLDiv`` does). A Gaussian-process BO
   (scikit-optimize) searches log10 of that over log-uniform stiffness and
   damping. The parameters are written into the M3 model in place, so one
   simulator and its captured graph serve every trial.

**Which temperature.** At the study's MPPI temperature (1e-5) the weights are
one-hot (ESS 1 of 256), so each "distribution" is the single cheapest sample,
and the KL only says whether two models picked the same one. Both planners
score the same samples, though, so their weights can be formed at any
temperature from the per-sample costs. By default (``--kl-ess 0.1``) each
state gets the temperature at which M5's weights have an ESS of 10% of the
samples, and both models are compared there: how alike the two models rank
the samples. The KL at the run temperature is reported too (``kl_run``);
``--kl-ess 0`` optimizes that instead.

The other rollout models (``--baselines``) are scored on the same states, as a
yardstick for how low the KL gets for a model that plans well.

Every ``run_episodes`` option applies (task, seed, planner settings, ...); the
defaults here are the ball check's::

    python test_scripts/tune_comfree_kl.py                      # both phases
    python test_scripts/tune_comfree_kl.py --reference results/comfree_kl/ball_reorient_m5_reference.pkl
    python test_scripts/tune_comfree_kl.py --steps 20 --n-states 4 --n-calls 4 \
        --n-initial-points 2 --baselines M2                    # smoke test

Writes ``<outdir>/<task>_m5_reference.pkl``, ``<task>_m5_episode.json`` and
``<task>_bo.json``. Needs the contact_kamino env.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ContactModelStudy.Drivers import run_episodes as drv  # noqa: E402
from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI  # noqa: E402
from ContactModelStudy.Utils.ContactModelPresets import CONTACT_MODELS, GetContactModelSim  # noqa: E402
from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder, StepSaving  # noqa: E402
from ContactModelStudy.Utils.EvalSimulators import makeEvalSim  # noqa: E402
from ContactModelStudy.Utils.PlannerKLDiv import MIN_VARIANCE, GaussianKL  # noqa: E402

#: KL given to a state whose M3 plan failed (every rollout NaN): worse than any real one.
FAILED_KL = 1e9
#: Seeds of the search: the M3 preset as it stands, the old study's values, and the
#: ComFreeConfig defaults as edited.
SEED_POINTS = [(CONTACT_MODELS["M3"]["params"]["stiffness"], CONTACT_MODELS["M3"]["params"]["damping"]),
               (0.2, 0.001), (100.0, 0.5)]


def buildParser():
    p = drv.build_parser()
    p.description = __doc__
    p.set_defaults(task="ball_reorient", rollout_model="M5", eval_sim="M5", seed=1, warm_start="shift",
                   goal_difficulty=4, steps=400, video=None, results=None)
    g = p.add_argument_group("tuning")
    g.add_argument("--outdir", default=str(_REPO_ROOT / "results" / "comfree_kl"))
    g.add_argument("--reference", default=None,
                   help="a saved reference (.pkl) to reuse instead of running the M5 episode")
    g.add_argument("--n-states", type=int, default=20, help="states kept from the M5 episode")
    g.add_argument("--n-calls", type=int, default=50, help="BO trials, the seeds included")
    g.add_argument("--n-initial-points", type=int, default=10, help="random trials before the GP")
    g.add_argument("--bo-seed", type=int, default=0)
    g.add_argument("--stiffness-range", type=float, nargs=2, default=(1e-2, 1e3))
    g.add_argument("--damping-range", type=float, nargs=2, default=(1e-4, 10.0))
    g.add_argument("--shrinkage", type=float, default=1e-3,
                   help="fraction of each covariance replaced by noise_sigma^2 I (as CalcPlannerKLDiv)")
    g.add_argument("--kl-ess", type=float, default=0.1,
                   help="compare the weightings at the temperature giving M5 this ESS, as a "
                        "fraction of the samples (per state); 0 uses the run's --temperature")
    g.add_argument("--baselines", nargs="*", default=["M1", "M2", "M4"],
                   help="other rollout models to score on the same states")
    return p


class RecordingPlanner:
    """A planner that keeps, at every ``Plan``, what is needed to replay it.

    Everything else is passed through to the wrapped planner, so the driver and
    the recorder see the planner itself.
    """

    def __init__(self, planner):
        self.inner = planner
        self.records = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def Plan(self, q, q_dot, u=None):
        rec = {"step": len(self.records), "q": np.array(q, float), "q_dot": np.array(q_dot, float),
               "u": None if u is None else np.array(u, float),
               "goal": np.array(self.inner.task.getGoal(), float),
               "planner_state": self.inner.SaveState()}
        out = self.inner.Plan(q, q_dot, u=u)
        g = readPlan(self.inner)
        if g is not None:
            w = self.inner.w_wp.numpy().astype(np.float64)
            g["ess"] = float(w.sum() ** 2 / (w ** 2).sum())
            rec.update(g)
        else:                           # a failed plan is not a usable reference
            rec["mu"] = None
        self.records.append(rec)
        return out


def readPlan(planner):
    """The last plan: first-action Gaussian, per-sample costs and first actions; ``None`` if it failed."""
    try:
        mu, cov = planner.FirstActionDistribution()
    except RuntimeError:
        return None
    return {"mu": mu, "cov": cov, "costs": planner.costs_wp.numpy().astype(np.float64),
            "V0": planner.V_wp.numpy()[:, 0, :].astype(np.float64)}


def weightsAt(costs, lam: float):
    """MPPI's weights at temperature ``lam``; NaN rollouts get zero."""
    c = np.where(np.isfinite(costs), costs, np.inf)
    w = np.exp(-(c - c.min()) / lam)
    return w / w.sum()


def tempForEss(costs, ess: float) -> float:
    """The temperature at which the weights have this effective sample size (bisection in log)."""
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        w = weightsAt(costs, 10.0 ** mid)
        lo, hi = (mid, hi) if 1.0 / (w ** 2).sum() < ess else (lo, mid)
    return 10.0 ** hi


def weightedGaussian(V0, w):
    mu = w @ V0
    D = V0 - mu
    cov = (w[:, None] * D).T @ D
    return mu, 0.5 * (cov + cov.T)


def shrunk(cov, sigma: float, shrinkage: float):
    """The covariance ``CalcPlannerKLDiv`` would compare: shrunk toward the proposal, floored."""
    n = cov.shape[0]
    return (1.0 - shrinkage) * cov + (shrinkage * sigma ** 2 + MIN_VARIANCE) * np.eye(n)


def makePlanner(model: str, args, task, **params):
    sim = GetContactModelSim(model, task.getModelPath(), N=args.n_samples,
                             **drv.rolloutSimOverrides(args, task), **params)
    return MPPI(sim, task, drv.buildPlannerConfig(args))


def closePlanner(planner) -> None:
    if hasattr(planner.sim, "Close"):
        planner.sim.Close()


def runReference(args, outdir: Path) -> dict:
    """Phase 1: one M5 episode; returns (and saves) the kept states."""
    task, eval_task = drv.buildTasks(args)
    sim = makeEvalSim(args.eval_sim, eval_task.getModelPath(), eval_task.timestep)
    planner = RecordingPlanner(makePlanner("M5", args, task))
    recorder = EpisodeRecorder(eval_task, sim, planner, steps=StepSaving.fromArgs(args),
                               cli_args=vars(args), eval_sim=args.eval_sim)
    drv.printHeader(args, task, eval_task, planner.sim.config, type(planner.sim).__name__, repr(planner.inner))
    t0 = time.perf_counter()
    s = drv.runRecordedEpisode(0, args, task, eval_task, sim, planner, recorder)
    drv._printEpisode(0, s, args)
    print(f"M5 episode: {s['finish_reason']} after {len(planner.records)} plans "
          f"({time.perf_counter() - t0:.0f} s)")
    recorder.Save(str(outdir / f"{args.task}_m5_episode.json"), save_steps=False)

    valid = [r for r in planner.records if r["mu"] is not None]
    if not valid:
        raise RuntimeError("every M5 plan failed; nothing to tune against")
    keep = np.unique(np.linspace(0, len(valid) - 1, min(args.n_states, len(valid))).round().astype(int))
    ref = {"task": args.task, "end_reason": s["finish_reason"], "n_plans": len(planner.records),
           "steps": [valid[i]["step"] for i in keep],
           "states": [valid[i] for i in keep]}

    # Replaying a saved state must reproduce the plan, or the comparison means nothing.
    for st in ref["states"][:2]:
        g = planOnce(planner.inner, st)
        print(f"replay check: max |cost recorded - replayed| = {np.nanmax(np.abs(g['costs'] - st['costs'])):.3g}, "
              f"|mu diff| = {np.abs(g['mu'] - st['mu']).max():.3g}")
    closePlanner(planner.inner)
    if hasattr(sim, "Close"):
        sim.Close()

    path = outdir / f"{args.task}_m5_reference.pkl"
    with open(path, "wb") as f:
        pickle.dump(ref, f)
    print(f"reference: {len(ref['states'])} states at steps {ref['steps']} -> {path}")
    return ref


def planOnce(planner, st):
    """Plan from a saved state; ``readPlan``'s dict, or ``None`` if the plan failed."""
    planner.task.setGoal(st["goal"])
    planner.LoadState(st["planner_state"])
    planner.Plan(st["q"], st["q_dot"], u=st["u"])
    return readPlan(planner)


def _kl(mu_p, cov_p, mu_q, cov_q, args) -> float:
    s = args.noise_sigma
    kl = GaussianKL(mu_p, shrunk(cov_p, s, args.shrinkage), mu_q, shrunk(cov_q, s, args.shrinkage))
    return min(kl, FAILED_KL) if math.isfinite(kl) else FAILED_KL


def scorePlanner(planner, ref: dict, args) -> dict:
    """KL(M5 || this planner) at every reference state, at the comparison and the run temperature."""
    kls, kls_run, rank = [], [], []
    for st in ref["states"]:
        g = planOnce(planner, st)
        if g is None or not np.all(np.isfinite(g["mu"])):
            kls.append(FAILED_KL); kls_run.append(FAILED_KL); rank.append(float("nan"))
            continue
        # The same mean and noise give the same samples; anything else means the replay is off.
        assert np.allclose(g["V0"], st["V0"], atol=1e-5), "samples differ from the reference's"
        kls_run.append(_kl(st["mu"], st["cov"], g["mu"], g["cov"], args))
        if args.kl_ess > 0:
            lam = st["kl_temperature"]
            kls.append(_kl(*weightedGaussian(st["V0"], weightsAt(st["costs"], lam)),
                           *weightedGaussian(g["V0"], weightsAt(g["costs"], lam)), args))
        else:
            kls.append(kls_run[-1])
        ok = np.isfinite(st["costs"]) & np.isfinite(g["costs"])
        rank.append(float(np.corrcoef(np.argsort(np.argsort(st["costs"][ok])),
                                      np.argsort(np.argsort(g["costs"][ok])))[0, 1]))
    kls = np.asarray(kls)
    return {"mean_kl": float(kls.mean()), "median_kl": float(np.median(kls)),
            "n_failed": int((kls >= FAILED_KL).sum()), "mean_kl_run": float(np.mean(kls_run)),
            "cost_rank_rho": float(np.nanmedian(rank)) if not np.all(np.isnan(rank)) else float("nan"),
            "kl": kls.tolist(), "kl_run": kls_run}


def setComFree(planner, stiffness: float, damping: float) -> None:
    # In place: comfree_warp reads these per world each step, and the captured
    # rollout graph holds these very arrays.
    planner.sim.m.comfree_stiffness.assign(np.array([stiffness], np.float32))
    planner.sim.m.comfree_damping.assign(np.array([damping], np.float32))


def runSearch(args, ref: dict, outdir: Path) -> dict:
    """Phase 2: BO over ComFree's stiffness and damping; returns (and saves) the results."""
    from skopt import Optimizer
    from skopt.space import Real

    task, _ = drv.buildTasks(args)
    for st in ref["states"]:
        st["kl_temperature"] = tempForEss(st["costs"], args.kl_ess * args.n_samples) if args.kl_ess > 0 \
            else args.temperature
    out = {"task": args.task, "kl_ess": args.kl_ess,
           "kl_temperatures": [st["kl_temperature"] for st in ref["states"]], "reference_steps": ref["steps"], "reference_end": ref["end_reason"],
           "shrinkage": args.shrinkage, "noise_sigma": args.noise_sigma, "temperature": args.temperature,
           "reference_ess_median": float(np.median([st["ess"] for st in ref["states"]])),
           "baselines": {}, "trials": []}
    path = outdir / f"{args.task}_bo.json"

    def save():
        path.write_text(json.dumps(out, indent=1))

    for model in args.baselines:
        planner = makePlanner(model, args, task)
        out["baselines"][model] = scorePlanner(planner, ref, args)
        closePlanner(planner)
        b = out["baselines"][model]
        print(f"baseline {model}: mean KL {b['mean_kl']:.4g}, median {b['median_kl']:.4g}, "
              f"run-temp KL {b['mean_kl_run']:.4g}, cost rank rho {b['cost_rank_rho']:.2f}, failed {b['n_failed']}")
        save()

    planner = makePlanner("M3", args, task)
    opt = Optimizer([Real(*args.stiffness_range, prior="log-uniform", name="stiffness"),
                     Real(*args.damping_range, prior="log-uniform", name="damping")],
                    base_estimator="GP", acq_func="gp_hedge",
                    n_initial_points=args.n_initial_points, random_state=args.bo_seed)
    seeds = [list(p) for p in SEED_POINTS]
    print(f"\n{'trial':>5s} {'stiffness':>10s} {'damping':>10s} {'mean KL':>10s} {'median KL':>10s} "
          f"{'run KL':>10s} {'rank rho':>8s} {'failed':>6s}")
    for i in range(args.n_calls):
        x = seeds.pop(0) if seeds else opt.ask()
        setComFree(planner, *x)
        r = scorePlanner(planner, ref, args)
        opt.tell(x, math.log10(r["mean_kl"] + 1e-12))
        out["trials"].append({"stiffness": float(x[0]), "damping": float(x[1]), **r})
        best = min(out["trials"], key=lambda t: t["mean_kl"])
        out["best"] = {k: best[k] for k in ("stiffness", "damping", "mean_kl", "median_kl", "mean_kl_run",
                                             "cost_rank_rho", "n_failed")}
        save()
        print(f"{i:5d} {x[0]:10.4g} {x[1]:10.4g} {r['mean_kl']:10.4g} {r['median_kl']:10.4g} "
              f"{r['mean_kl_run']:10.4g} {r['cost_rank_rho']:8.2f} {r['n_failed']:6d}", flush=True)
    closePlanner(planner)

    print(f"\nreference: {len(ref['states'])} states, median ESS of the M5 weights "
          f"{out['reference_ess_median']:.1f} of {args.n_samples}")
    print("best 5 M3 trials:")
    for t in sorted(out["trials"], key=lambda t: t["mean_kl"])[:5]:
        print(f"  stiffness {t['stiffness']:.4g}  damping {t['damping']:.4g}  "
              f"mean KL {t['mean_kl']:.4g}  median {t['median_kl']:.4g}  run KL {t['mean_kl_run']:.4g}  "
              f"rho {t['cost_rank_rho']:.2f}  failed {t['n_failed']}")
    for model, b in out["baselines"].items():
        print(f"  {model:30s}  mean KL {b['mean_kl']:.4g}  median {b['median_kl']:.4g}  "
              f"run KL {b['mean_kl_run']:.4g}  rho {b['cost_rank_rho']:.2f}  failed {b['n_failed']}")
    print(f"results -> {path}")
    return out


def main(argv=None) -> int:
    args = drv.parseArgs(argv, buildParser(), results_prefix="tune_comfree")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.reference:
        with open(args.reference, "rb") as f:
            ref = pickle.load(f)
        if ref["task"] != args.task:
            raise SystemExit(f"--reference is for {ref['task']}, not --task {args.task}")
    else:
        ref = runReference(args, outdir)
    runSearch(args, ref, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
