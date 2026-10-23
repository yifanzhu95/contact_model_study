"""Per-world goals on the task and per-world reset on the vectorized simulators.

What a training environment needs from the shared layer: each world scored
against its own goal, and finished worlds reset without disturbing the rest.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from conftest import ROLLOUT_CUBE

N = 8
MODELS = ["M1", "M2", "M3", "M4",
          pytest.param("M5", marks=pytest.mark.skipif(importlib.util.find_spec("newton") is None,
                                                      reason="Kamino needs Newton (contact_kamino env)"))]


def _sim(model, n=N):
    from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
    return GetContactModelSim(model, ROLLOUT_CUBE, N=n)


def _scattered(q0, n, seed):
    rng = np.random.default_rng(seed)
    q = np.tile(q0, (n, 1)) + rng.normal(0, 0.02, (n, q0.size))
    q[:, 19:23] /= np.linalg.norm(q[:, 19:23], axis=1, keepdims=True)
    return q, rng.normal(0, 0.1, (n, 22))


# -- per-world goals ------------------------------------------------------------
@pytest.mark.gpu
def test_goal_table_of_the_shared_goal_matches_the_shared_path(cube_tasks, cube_initial):
    import warp as wp
    ro, _ = cube_tasks
    q0, _, u0 = cube_initial
    sim = _sim("M2")
    sim.SetState(*_scattered(q0, N, 0))
    sim.SetControl(u0)
    sim.Step(3)
    table = wp.array(np.tile(ro.goal, (N, 1)), dtype=wp.float32, device=sim.config.device)
    for terminal in (False, True):
        shared = ro.calcCosts(sim, terminal=terminal).numpy().copy()
        assert np.array_equal(ro.calcCosts(sim, terminal=terminal, goals=table).numpy(), shared)
    assert np.array_equal(ro.isSuccess(sim, goals=table).numpy(), ro.isSuccess(sim).numpy())


@pytest.mark.gpu
def test_each_world_is_scored_against_its_own_goal(cube_initial):
    import warp as wp
    from ContactModelStudy.Tasks.CubeReorient import CubeReorient
    from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig
    from ContactModelStudy.Tasks.TaskBase import TaskRole
    task = CubeReorient(LeapReorientConfig(role=TaskRole.ROLLOUT, seed=3, goal_difficulty=4))
    q0, _, u0 = cube_initial
    sim = _sim("M2")
    q, v = _scattered(q0, N, 1)
    goals = [task.sampleNewGoal(current_goal=task.getGoal()) for _ in range(N)]
    # Put a few worlds exactly on their own goal, so success differs per world too.
    for k in (0, 3):
        q[k, 16:19], q[k, 19:23], v[k] = task.params["target_pos"], goals[k], 0.0
    sim.SetState(q, v)
    sim.SetControl(u0)
    table = wp.array(np.stack([task.goalRow(g) for g in goals]), dtype=wp.float32, device=sim.config.device)
    cost = task.calcCosts(sim, goals=table).numpy().copy()
    success = task.isSuccess(sim, goals=table).numpy().copy()
    for k, g in enumerate(goals):
        task.setGoal(g)
        assert cost[k] == task.calcCosts(sim).numpy()[k]
        assert success[k] == task.isSuccess(sim).numpy()[k]
    assert success[0] and success[3] and not success.all()


def test_goal_table_shape_is_checked(cube_tasks):
    pytest.importorskip("warp")
    import warp as wp
    from conftest import HAS_CUDA
    if not HAS_CUDA:
        pytest.skip("no CUDA device")
    ro, _ = cube_tasks
    sim = _sim("M2")
    with pytest.raises(ValueError, match="goals must be"):
        ro.calcCosts(sim, goals=wp.zeros((N + 1, ro.goal.size), dtype=wp.float32, device=sim.config.device))


# -- per-world reset ------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.parametrize("model", MODELS)
def test_reset_worlds_touches_only_the_selected_worlds(model, cube_initial):
    q0, v0, u0 = cube_initial
    sim = _sim(model)
    sim.SetState(*_scattered(q0, N, 2))
    sim.SetControl(u0)
    sim.Step(5)
    q_before, v_before = sim.GetState()
    u_before = sim.GetControl()
    ids = [1, 4, 6]
    rq, rv = _scattered(q0, len(ids), 3)
    ru = np.tile(u0, (len(ids), 1))
    sim.ResetWorlds(ids, rq, rv, ru)
    q_after, v_after = sim.GetState()
    others = [k for k in range(N) if k not in ids]
    assert np.array_equal(q_after[others], q_before[others])
    assert np.array_equal(v_after[others], v_before[others])
    assert np.array_equal(sim.GetControl()[others], u_before[others])
    assert np.allclose(q_after[ids], rq, atol=1e-5)
    assert np.allclose(v_after[ids], rv, atol=1e-5)
    assert np.allclose(sim.GetControl()[ids], ru, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("model", MODELS)
def test_a_reset_world_steps_like_a_fresh_one(model, cube_initial):
    """A reset world carries nothing over from its previous episode."""
    q0, v0, u0 = cube_initial
    sim = _sim(model, n=2)
    sim.SetState(np.tile(q0, (2, 1)), np.zeros((2, 22)))
    sim.SetControl(u0)
    sim.Step(40)                                   # world 1 builds up contacts and warm starts
    sim.ResetWorlds([1], q0[None], v0[None], u0[None])
    fresh = _sim(model, n=1)
    fresh.SetState(q0, v0)
    fresh.SetControl(u0)
    for _ in range(10):
        sim.Step(1)
        fresh.Step(1)
    assert np.allclose(sim.GetState()[0][1], fresh.GetState()[0][0], atol=2e-3)


@pytest.mark.gpu
def test_reset_worlds_rejects_bad_ids(cube_initial):
    q0, _, _ = cube_initial
    sim = _sim("M2")
    for ids in ([], [0, 0], [N]):
        with pytest.raises(ValueError):
            sim.ResetWorlds(ids, np.tile(q0, (len(ids), 1)))


@pytest.mark.gpu
@pytest.mark.skipif(importlib.util.find_spec("newton") is None, reason="Kamino needs Newton")
@pytest.mark.parametrize("lift, steps", [(0.3, 60), (0.0, 200)], ids=["free-spin", "in-grasp"])
def test_kamino_reports_the_object_body_state(lift, steps, cube_tasks, cube_initial):
    """The MuJoCo-layout mirror is the object's actual rigid-body pose and spin, not a stale copy."""
    from scipy.spatial.transform import Rotation
    ro, _ = cube_tasks
    q0, _, u0 = cube_initial
    oq, ov, ob = (int(ro.indices[i]) for i in (0, 1, 4))
    sim = _sim("M5", n=1)
    q, v = q0.copy(), np.zeros(22)
    q[oq + 2] += lift
    if lift:
        v[ov:ov + 6] = [0.1, 0.0, 0.5, 3.0, -2.0, 4.0]
    sim.SetState(q, v)
    sim.SetControl(u0)
    sim.Step(steps)
    qm, vm = sim.GetState()
    m = sim.model
    joint = sim.mjm.joint(int(sim.mjm.body_jntadr[ob])).name
    body = int(m.joint_child.numpy()[[str(s).rsplit("/", 1)[-1] for s in m.joint_label].index(joint)])
    pose, twist = sim.state_0.body_q.numpy()[body], sim.state_0.body_qd.numpy()[body]
    quat = pose[[6, 3, 4, 5]] * np.sign(np.dot(pose[[6, 3, 4, 5]], qm[0, oq + 3:oq + 7]))
    assert np.allclose(qm[0, oq:oq + 3], pose[:3], atol=1e-6)
    assert np.allclose(qm[0, oq + 3:oq + 7], quat, atol=1e-6)
    spin_body = Rotation.from_quat(pose[3:7]).inv().apply(twist[3:])
    assert np.allclose(vm[0, ov + 3:ov + 6], spin_body, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("model", MODELS)
def test_capacity_check_passes_in_a_normal_grasp(model, cube_initial):
    q0, _, u0 = cube_initial
    sim = _sim(model)
    sim.SetState(q0)
    sim.SetControl(u0)
    sim.Step(20)
    sim.CheckCapacity()
