"""The GPU contact models (M1-M5) as eval simulators: SingleWorld and EvalSimulators."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from conftest import EVAL_CUBE
from ContactModelStudy.Utils.EvalSimulators import (EVAL_PRESETS, evalSimConfig, evalSimNames,
                                                    isGpuEvalSim, makeEvalSim)

DT = 0.002
CLASSES = {"M1": "VectorizedMujoco", "M2": "VectorizedMujoco", "M3": "ComFree", "M4": "XPBD", "M5": "Kamino"}
# M5 (Kamino) builds only where Newton is installed (the contact_kamino env).
MODELS = [pytest.param(m, marks=pytest.mark.skipif(importlib.util.find_spec("newton") is None,
                                                   reason="M5 needs Newton (contact_kamino env)"))
          if m == "M5" else m for m in EVAL_PRESETS]


def test_names():
    assert evalSimNames() == ["drake", "mujoco", "pinocchio", "M1", "M2", "M3", "M4", "M5"]
    assert all(isGpuEvalSim(m) for m in EVAL_PRESETS)
    assert not any(isGpuEvalSim(n) for n in ("mujoco", "pinocchio", "drake"))
    with pytest.raises(ValueError, match="unknown eval simulator"):
        makeEvalSim("M9", EVAL_CUBE, DT)
    with pytest.raises(ValueError):
        evalSimConfig("m2x", DT)


@pytest.fixture(scope="module")
def sims():
    """Each preset's eval sim, built on first use and shared across the tests."""
    built = {}

    class _Sims:
        def __getitem__(self, m):
            if m not in built:
                built[m] = makeEvalSim(m, EVAL_CUBE, DT)
            return built[m]
    return _Sims()


@pytest.mark.gpu
@pytest.mark.parametrize("model", MODELS)
def test_single_world_reads_like_a_simulator(sims, model, cube_tasks, cube_initial):
    from ContactModelStudy.Simulators.Simulator import Simulator
    from ContactModelStudy.Simulators.VectorizedSimulator import VectorizedSimulator
    sim = sims[model]
    assert isinstance(sim, Simulator) and not isinstance(sim, VectorizedSimulator)
    assert sim.class_name == CLASSES[model] and sim.timestep == pytest.approx(DT)
    name, cfg = evalSimConfig(model, DT)
    assert name == CLASSES[model] and cfg == sim.config
    q0, v0, u0 = cube_initial
    sim.SetState(q0, v0)
    sim.SetControl(u0)
    q, v = sim.GetState()
    assert q.shape == (sim.nq,) and v.shape == (sim.nv,) and sim.GetControl().shape == (sim.nu,)
    assert np.allclose(q, q0, atol=1e-6) and np.allclose(sim.GetControl(), u0, atol=1e-6)
    t0 = sim.time
    sim.Step(10)
    assert isinstance(t0, float) and sim.time == pytest.approx(t0 + 10 * DT)
    # Tasks judge it on the host, like CPU MuJoCo.
    ev = cube_tasks[1]
    ev.setSimToInitialState(sim)
    assert type(ev.isSuccess(sim)) is bool and type(ev.isFailure(sim)) is bool


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("model", MODELS)
def test_graph_steps_equal_eager_steps(model, cube_initial):
    """Graph replay agrees with eager stepping as well as eager agrees with itself.

    M1's stiff contact under 200 solver iterations is not bit-reproducible on
    MJWarp (two eager runs end ~1e-3 apart), so the bar is that run-to-run
    noise, not zero.
    """
    from ContactModelStudy.Simulators.SingleWorld import SingleWorld
    q0, v0, u0 = cube_initial
    graph, eager, eager2 = (makeEvalSim(model, EVAL_CUBE, DT) for _ in range(3))
    eager.use_graph = eager2.use_graph = False
    rng = np.random.default_rng(0)
    for s in (graph, eager, eager2):
        s.SetState(q0, v0)
    for n in (1, 5, 37, 300, 5, 37):           # repeats replay the captured graphs
        u = u0 + 0.05 * rng.standard_normal(u0.shape)
        for s in (graph, eager, eager2):
            s.SetControl(u)
            s.Step(n)
    qg, qe, qe2 = graph.GetState()[0], eager.GetState()[0], eager2.GetState()[0]
    noise = np.abs(qe2 - qe).max()
    assert np.isfinite(qg).all() and np.abs(qg - qe).max() <= max(3 * noise, 1e-4)
    assert isinstance(graph, SingleWorld) and graph._graphs and not graph._graph_failed
    assert set(graph._graphs) <= {1 << k for k in range(9)}


@pytest.mark.gpu
@pytest.mark.parametrize("model", MODELS)
def test_cube_stays_in_hand(sims, model, cube_tasks, cube_initial):
    sim, ev = sims[model], cube_tasks[1]
    ev.setSimToInitialState(sim)
    sim.Step(int(1.0 / DT))
    q, v = sim.GetState()
    assert np.isfinite(q).all() and np.isfinite(v).all() and not ev.isFailure(sim)


def test_single_world_needs_one_world():
    pytest.importorskip("warp")
    from conftest import HAS_CUDA
    if not HAS_CUDA:
        pytest.skip("no CUDA device")
    from ContactModelStudy.Simulators.SingleWorld import SingleWorld
    from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
    with pytest.raises(ValueError, match="N=1"):
        SingleWorld(GetContactModelSim("M2", EVAL_CUBE, N=2, timestep=DT))
