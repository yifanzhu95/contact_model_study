"""Changing a built planner's settings: TaskBase.setCostWeights, UpdateConfig, a long-lived EpisodePool."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ROLLOUT_CUBE
from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig
from ContactModelStudy.Tasks.TaskBase import TaskRole


def test_set_cost_weights_updates_params_and_config():
    t = CubeReorient(LeapReorientConfig(cost_weights={"w_joint": 3.0}))
    own = CubeReorient().params["cost_weights"]
    t.setCostWeights({"w_quat": 7.0})
    assert t.params["cost_weights"]["w_quat"] == 7.0 and t.params["cost_weights"]["w_joint"] == 3.0
    assert t.weights[0] == np.float32(7.0)
    # The config describes the instance: the overrides relative to the object's own weights.
    assert t.config.cost_weights == {"w_quat": 7.0, "w_joint": 3.0}
    t.setCostWeights({"w_quat": own["w_quat"], "w_joint": own["w_joint"]})
    assert t.config.cost_weights is None
    with pytest.raises(ValueError, match="unknown cost weight"):
        t.setCostWeights({"w_nope": 1.0})


@pytest.mark.gpu
def test_update_config_only_takes_runtime_fields(cube_tasks):
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco, VectorizedMujocoConfig
    p = MPPI(VectorizedMujoco(ROLLOUT_CUBE, VectorizedMujocoConfig(horizon=4, substeps=2), N=16),
             cube_tasks[0], MPPI_Config(temperature=10.0, noise_sigma=0.1))
    p.UpdateConfig(temperature=2.0, noise_sigma=0.05)
    assert (p.config.temperature, p.config.noise_sigma, p.lam) == (2.0, 0.05, 2.0)
    with pytest.raises(ValueError, match="cannot change"):
        p.UpdateConfig(control_mode="absolute")
    with pytest.raises(ValueError, match="positive"):
        p.UpdateConfig(temperature=-1.0)


@pytest.mark.gpu
def test_new_weights_reach_a_captured_graph(cube_initial):
    """Weights changed after capture score like a planner built with them."""
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco, VectorizedMujocoConfig
    q0, v0, u0 = cube_initial
    new = {"w_quat": 400.0, "w_contact": 50.0}

    def planner(task):
        sim = VectorizedMujoco(ROLLOUT_CUBE, VectorizedMujocoConfig(horizon=4, substeps=2), N=64)
        return MPPI(sim, task, MPPI_Config(seed=0, noise_sigma=0.1, temperature=10.0))

    a = planner(CubeReorient(LeapReorientConfig(role=TaskRole.ROLLOUT)))
    a.Plan(q0, v0, u=u0)                               # captures the graph
    before = a.last_min_cost
    a.Reset()
    a.LoadState({**a.SaveState(), "resample_count": 0})
    a.task.setCostWeights(new)
    a.Plan(q0, v0, u=u0)
    b = planner(CubeReorient(LeapReorientConfig(role=TaskRole.ROLLOUT, cost_weights=new)))
    b.Plan(q0, v0, u=u0)
    assert not np.isclose(a.last_min_cost, before, rtol=1e-3)
    assert np.isclose(a.last_min_cost, b.last_min_cost, rtol=1e-3)


@pytest.mark.gpu
@pytest.mark.slow
def test_one_pool_serves_jobs_with_different_models_and_settings():
    from ContactModelStudy.Drivers import run_episodes as drv
    from ContactModelStudy.Drivers.EpisodePool import EpisodeJob, EpisodePool
    args = drv.parseArgs(["--timestep", "0.0005", "--eval-steps-per-rollout-step", "8", "--substeps", "4",
                          "--horizon", "4", "--n-samples", "32", "--settle", "0.1", "--steps", "3",
                          "--no-video", "--no-results", "--goal-difficulty", "4", "--temperature", "10"])
    jobs = [EpisodeJob(0, rollout_model="M2", key="a"),
            EpisodeJob(0, rollout_model="M3", planner_params={"temperature": 3.0},
                       cost_weights={"w_quat": 99.0}, key="b"),
            EpisodeJob(1, rollout_model="M2", planner_params={"noise_sigma": 0.05}, key="c")]
    results = {}
    with EpisodePool(args, 2, ["0"], rollout_models=["M2", "M3"]) as pool:
        for batch in (jobs[:2], jobs[2:]):              # a second batch on the same processes
            for j in batch:
                pool.submit(j)
            while pool.pending and not pool.fatal:
                for r in pool.poll():
                    results[r.job.key] = r
        with pytest.raises(ValueError):
            pool.submit(EpisodeJob(0, rollout_model="M1"))
    assert not pool.problems and set(results) == {"a", "b", "c"}
    cfg = {k: r.episode.configs for k, r in results.items()}
    assert cfg["a"]["planner"]["config"]["temperature"] == 10 and cfg["b"]["planner"]["config"]["temperature"] == 3
    assert cfg["c"]["planner"]["config"]["noise_sigma"] == 0.05
    assert (cfg["a"]["rollout_simulator"]["class"], cfg["b"]["rollout_simulator"]["class"]) == \
        ("VectorizedMujoco", "ComFree")
    assert cfg["b"]["rollout_task"]["config"]["cost_weights"] == {"w_quat": 99.0}
    assert cfg["a"]["rollout_task"]["config"]["cost_weights"] is None
    assert cfg["c"]["rollout_task"]["config"]["cost_weights"] is None     # b's weights do not leak
    # Same episode index, same goals, whatever the model and settings.
    assert results["a"].episode.extra["goal_seed"] == results["b"].episode.extra["goal_seed"]
