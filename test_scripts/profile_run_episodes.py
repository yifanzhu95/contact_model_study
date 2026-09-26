#!/usr/bin/env python3
"""Profile run_episodes.py: where each control step's time goes, per eval simulator.

Each eval simulator runs a full driver episode in its own process, with a timer
around every call the driver makes (eval-simulator step, planning, state
reads, success/failure checks, recording, rendering). The table shows
milliseconds per control step and each part's share. The first plan is left
out, since it compiles kernels and captures the CUDA graph.

Examples::

    python test_scripts/profile_run_episodes.py
    python test_scripts/profile_run_episodes.py --eval-sims mujoco drake --steps 300
    python test_scripts/profile_run_episodes.py --video             # include rendering
    python test_scripts/profile_run_episodes.py --cprofile          # + hotspots per sim
    python test_scripts/profile_run_episodes.py --pinocchio-stages  # split a Pinocchio step
    python test_scripts/profile_run_episodes.py -- --rollout-model M3 --n-samples 512

Anything after ``--`` is passed to run_episodes.py unchanged, so the profile is
of whatever configuration you give it (the driver's defaults otherwise).

cProfile adds per-call overhead to Python-heavy code, so the timing table always
comes from a run without it. ``--cprofile`` adds a separate run for hotspots.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import cProfile
import functools
import importlib
import io
import json
import os
import pstats
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_SIMS = {
    "mujoco": ("ContactModelStudy.Simulators.Mujoco", "Mujoco"),
    "pinocchio": ("ContactModelStudy.Simulators.Pinocchio", "Pinocchio"),
    "drake": ("ContactModelStudy.Simulators.Drake", "Drake"),
}
PLAN = "planner: Plan (GPU rollouts)"


# -- child: one instrumented driver run ---------------------------------------
def _profileOne(eval_sim: str, steps: int, video: bool, use_cprofile: bool,
                driver_args: list[str], out_json: str) -> None:
    from ContactModelStudy.Drivers import run_episodes as drv
    from ContactModelStudy.Renderers.MujocoVideoRenderer import MujocoVideoRenderer
    from ContactModelStudy.SamplingBasedPlanners.SamplingBasedPlannerBase import SamplingBasedPlannerBase
    from ContactModelStudy.Tasks.LeapReorient import LeapReorient
    from ContactModelStudy.Utils.EpisodeRecorder import EpisodeRecorder

    totals, calls = collections.defaultdict(float), collections.Counter()
    per_call = collections.defaultdict(list)
    state = {"on": False}

    def wrap(cls, name, label):
        f = getattr(cls, name)

        @functools.wraps(f)
        def timed(*a, **k):
            if not state["on"]:
                return f(*a, **k)
            t0 = time.perf_counter()
            r = f(*a, **k)
            dt = time.perf_counter() - t0
            totals[label] += dt
            calls[label] += 1
            per_call[label].append(dt)
            return r
        setattr(cls, name, timed)

    sim_cls = getattr(importlib.import_module(EVAL_SIMS[eval_sim][0]), EVAL_SIMS[eval_sim][1])
    for name in ("Step", "GetState", "SetControl", "GetControl"):
        wrap(sim_cls, name, f"eval sim: {name}")
    wrap(SamplingBasedPlannerBase, "Plan", PLAN)
    wrap(LeapReorient, "isSuccess", "task: isSuccess")
    wrap(EpisodeRecorder, "recordStateAndAction", "recorder: recordStateAndAction")
    wrap(MujocoVideoRenderer, "RenderState", "renderer: RenderState")

    # The control loop starts at the first isFailure check (after the settle)
    # and ends when the episode is finished.
    is_failure = LeapReorient.isFailure

    def first_failure_check(self, sim, out=None):
        if not state["on"]:
            state["on"], state["t0"] = True, time.perf_counter()
        t0 = time.perf_counter()
        r = is_failure(self, sim, out)
        totals["task: isFailure"] += time.perf_counter() - t0
        calls["task: isFailure"] += 1
        return r
    LeapReorient.isFailure = first_failure_check

    finished = EpisodeRecorder.episodeFinished

    def finish(self, *a, **k):
        if state["on"]:
            state["wall"], state["on"] = time.perf_counter() - state["t0"], False
        return finished(self, *a, **k)
    EpisodeRecorder.episodeFinished = finish

    argv = ["--eval-sim", eval_sim, "--steps", str(steps), "--no-results", *driver_args]
    tmp = tempfile.mkdtemp()
    argv += ["--video", str(Path(tmp) / f"{eval_sim}.mp4")] if video else ["--no-video"]
    prof = cProfile.Profile()
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        if use_cprofile:
            prof.enable()
        drv.main(argv)
        if use_cprofile:
            prof.disable()

    first_plan = per_call[PLAN][0] if per_call[PLAN] else 0.0
    res = {"eval_sim": eval_sim, "steps": calls[PLAN], "wall_s": state.get("wall", float("nan")),
           "first_plan_s": first_plan, "parts": {k: [totals[k], calls[k]] for k in totals}}
    if use_cprofile:
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(20)
        res["cprofile"] = s.getvalue()
    Path(out_json).write_text(json.dumps(res))


# -- Pinocchio: per-stage split of one step -----------------------------------
def _pinocchioStages(timestep: float, control_steps: int, eval_steps: int) -> None:
    """Time each stage of Pinocchio's step while the cube is held and the fingers move."""
    import numpy as np
    from ContactModelStudy.Simulators.Pinocchio import Pinocchio, PinocchioConfig
    from ContactModelStudy.Tasks.CubeReorient import CubeReorient
    from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig

    task = CubeReorient(LeapReorientConfig(timestep=timestep))
    _, _, u0 = task.getInitialState()
    s = Pinocchio(task.getModelPath(), PinocchioConfig(timestep=timestep))
    task.setSimToInitialState(s)
    s.Step(int(round(0.2 / timestep)))                         # settle
    pin, T, pc = s._pin, collections.defaultdict(float), time.perf_counter

    def step():
        model, data = s.model, s.data
        q, v, dt = s._q, s._v, s.config.timestep
        a = pc(); cms, cds = s._detectContacts(); n_contact = len(cms); T["collision + contact models"] += pc() - a
        a = pc(); jc, jd = s._jointConstraints(q); cms, cds = cms + jc, cds + jd; T["joint-limit/friction models"] += pc() - a
        a = pc()
        u = np.clip(s._u, s._ctrl_lo, s._ctrl_hi)
        tau = -s._damping * v
        tau[s._ctrl_v] += s._kp * (u - q[s._ctrl_q]) - s._kv * v[s._ctrl_v]
        T["servo torques"] += pc() - a
        a = pc(); pin.crba(model, data, q, pin.Convention.WORLD); T["crba (mass matrix)"] += pc() - a
        a = pc(); v_free = v + dt * pin.aba(model, data, q, v, tau, s._fext); T["aba (free motion)"] += pc() - a
        if not cms:
            s._q, s._v = pin.integrate(model, q, v_free * dt), v_free
            return
        a = pc()
        for cm, cd in zip(cms, cds):
            cm.calc(model, data, cd)
        chol = pin.ConstraintCholeskyDecomposition(model, data, cms, cds)
        chol.compute(model, data, cms, cds, s.config.delassus_regularization)
        delassus = chol.getDelassusOperatorCholeskyExpression()
        T["constraint calc + Delassus Cholesky"] += pc() - a
        a = pc()
        Jc = np.asarray(pin.getConstraintsJacobian(model, data, cms, cds)).reshape(-1, model.nv)
        g = Jc @ v_free
        idx = 0
        for cm, cd in zip(cms[:n_contact], cds[:n_contact]):
            size, kp = cm.residualSize(), cm.baumgarte_corrector_parameters.Kp
            if kp != 0.0:
                g[idx:idx + size] += kp * cd.extract().constraint_position_error / dt
            idx += size
        T["Jacobian + Baumgarte"] += pc() - a
        a = pc(); s._solver.solve(delassus, g, cms, cds, s._settings, s._result); T["ADMM solve"] += pc() - a
        a = pc()
        f = np.asarray(s._result.retrieveConstraintImpulses()).ravel() / dt
        v_new = v + dt * pin.aba(model, data, q, v, tau + Jc.T @ f, s._fext)
        s._q, s._v = pin.integrate(model, q, v_new * dt), v_new
        T["aba + integrate"] += pc() - a

    mjm = task.mjm
    flex = [k for k in range(16) if mjm.actuator(k).name.split("_")[1] in ("mcp", "pip", "dip", "ipl")]
    tgt = u0.copy()
    tgt[flex] += 0.3 * (mjm.actuator_ctrlrange[flex, 1] - u0[flex])
    n, w0 = 0, pc()
    for k in range(control_steps):
        s.SetControl(u0 + (0.5 - 0.5 * np.cos(2 * np.pi * k / control_steps)) * (tgt - u0))
        for _ in range(eval_steps):
            step()
            n += 1
    wall = pc() - w0
    print(f"\nPinocchio step, {n} steps at {timestep * 1e3:g} ms holding the cube: "
          f"{wall / n * 1e3:.3f} ms/step")
    for k, v in sorted(T.items(), key=lambda x: -x[1]):
        print(f"  {v / n * 1e3:7.4f} ms/step  {100 * v / wall:5.1f}%  {k}")


# -- parent: run each simulator, print the table ------------------------------
def _printTable(res: dict, video: bool) -> None:
    n, fp = res["steps"], res["first_plan_s"]
    wall = res["wall_s"] - fp
    n_rest = max(n - 1, 1)
    print(f"\n== {res['eval_sim']}{' (video)' if video else ''}: {wall / n_rest * 1e3:.1f} ms per "
          f"control step over {n_rest} steps  (first plan {fp:.2f} s excluded)")
    rows = []
    for k, (t, c) in res["parts"].items():
        if k == PLAN:
            t -= fp
        rows.append((t, k))
    for t, k in sorted(rows, reverse=True):
        if t / n_rest * 1e3 >= 0.05:
            print(f"   {k:<32} {t / n_rest * 1e3:7.2f} ms/step  {100 * t / wall:5.1f}%")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    driver_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, driver_args = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-sims", nargs="+", default=list(EVAL_SIMS), choices=list(EVAL_SIMS))
    p.add_argument("--steps", type=int, default=150, help="control steps per profiled episode")
    p.add_argument("--video", action="store_true", help="also render a video (rendering is timed)")
    p.add_argument("--cprofile", action="store_true", help="add a cProfile run per simulator for hotspots")
    p.add_argument("--pinocchio-stages", action="store_true", help="also split a Pinocchio step by stage")
    p.add_argument("--child", nargs=2, metavar=("SIM", "OUT_JSON"), help=argparse.SUPPRESS)
    p.add_argument("--child-cprofile", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args.child:
        _profileOne(args.child[0], args.steps, args.video, args.child_cprofile, driver_args, args.child[1])
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        for sim in args.eval_sims:
            for use_cprofile in ([False, True] if args.cprofile else [False]):
                out = str(Path(tmp) / f"{sim}_{int(use_cprofile)}.json")
                cmd = [sys.executable, __file__, "--child", sim, out, "--steps", str(args.steps)]
                cmd += ["--video"] if args.video else []
                cmd += ["--child-cprofile"] if use_cprofile else []
                cmd += ["--", *driver_args] if driver_args else []
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0 or not Path(out).exists():
                    print(f"\n== {sim}: FAILED\n{r.stderr[-2000:]}")
                    break
                res = json.loads(Path(out).read_text())
                if use_cprofile:
                    lines = res["cprofile"].splitlines()
                    start = next(i for i, line in enumerate(lines) if "tottime" in line)
                    print(f"\n-- {sim}: cProfile hotspots (inflated by profiling overhead)")
                    print("\n".join(line[:160] for line in lines[start:start + 18]))
                else:
                    _printTable(res, args.video)

    if args.pinocchio_stages:
        _pinocchioStages(timestep=0.0005, control_steps=60, eval_steps=128)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
