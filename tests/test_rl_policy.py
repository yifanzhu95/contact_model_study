"""The thin RL layer: encoding, the training environment, the training driver, and policies in run_episodes.

Needs torch (the ``rl`` extra); GPU tests also need CUDA, and the training
round trip needs skrl.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import REPO_ROOT  # noqa: E402
from ContactModelStudy.RL.Config import EnvConfig, PPOConfig  # noqa: E402

HAS_SKRL = importlib.util.find_spec("skrl") is not None
MODELS = ["M1", "M2", "M3", "M4",
          pytest.param("M5", marks=pytest.mark.skipif(importlib.util.find_spec("newton") is None,
                                                      reason="Kamino needs Newton"))]


def _smoke(model="M2", **env):
    return EnvConfig(**{"rollout_model": model, "num_envs": 4, "episode_steps": 5, "settle_s": 0.1, **env})


@pytest.fixture
def actor_file(tmp_path, cube_tasks):
    """An untrained actor exported with a smoke contract, as train_rl_policy writes one."""
    from ContactModelStudy.RL import Encoding
    from ContactModelStudy.RL.Policy import Actor, export_actor
    contract = {"config": _smoke(episode_steps=7, settle_s=0.25).to_dict(),
                "observation_profile": Encoding.OBSERVATION_PROFILE,
                "action_profile": Encoding.ACTION_PROFILE, "reward_profile": Encoding.REWARD_PROFILE}
    path = tmp_path / "actor.pt"
    export_actor(Actor(hidden_sizes=(8,)), contract, path)
    return path


# -- configs and encoding -----------------------------------------------------------
def test_configs_validate():
    assert EnvConfig().substeps == 16 and EnvConfig(settle_s=0.5).settle_steps == 125
    for bad in (dict(control_dt=0.01), dict(rollout_model="M9"), dict(num_envs=1),
                dict(task_name="spin"), dict(device="cpu"), dict(settle_s=-1.0)):
        with pytest.raises(ValueError):
            EnvConfig(**bad)
    with pytest.raises(ValueError):
        PPOConfig(hidden_sizes=())


def test_codec_limits_the_target_change(cube_tasks):
    from ContactModelStudy.RL.Encoding import ActionCodec
    mjm = cube_tasks[0].mjm
    codec = ActionCodec.fromModel(mjm, rate=2.0, control_dt=0.05)
    previous = torch.as_tensor(mjm.actuator_ctrlrange.mean(axis=1), dtype=torch.float32)[None]
    out = codec.decode(torch.full((1, 16), 50.0), previous)
    assert torch.allclose(out - previous, torch.full((1, 16), 0.1), atol=1e-6)
    assert torch.all(codec.decode(torch.full((1, 16), -1e6), codec.lower[None]) == codec.lower)


def test_observation_layout(cube_tasks, cube_initial):
    from ContactModelStudy.RL.Encoding import ActionCodec, ObservationEncoder
    ro, _ = cube_tasks
    q0, v0, u0 = cube_initial
    codec = ActionCodec.fromModel(ro.mjm, rate=1.0, control_dt=0.064)
    enc = ObservationEncoder(ro.mjm, ro.indices, codec, episode_steps=10)

    def row(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32))[None]

    obs = enc(row(q0), row(v0), row(ro.params["target_pos"]), row(ro.getGoal()), row(u0), torch.tensor([4]))
    assert obs.shape == (1, 73) and torch.isfinite(obs).all()
    assert obs[0, -1].item() == pytest.approx(0.6)
    assert torch.all(obs[0, :16].abs() <= 1.0)


# -- the policy planner and run_episodes ----------------------------------------------
def test_policy_planner_runs_its_contract(actor_file, cube_tasks, cube_initial):
    from ContactModelStudy.RL.PolicyPlanner import RLPolicyPlanner
    ro, _ = cube_tasks                                  # 2 ms timestep: 64 ms is 32 substeps
    q0, v0, u0 = cube_initial
    planner = RLPolicyPlanner(actor_file, ro)
    assert planner.substeps == 32 and planner.sim is None
    planner.Reset()
    u = planner.Plan(q0, v0, u0)
    assert u.shape == (16,) and planner.last_plan_ok
    assert np.all(np.abs(u - u0) <= planner.codec.delta + 1e-6)
    with pytest.raises(ValueError, match="applied command"):
        planner.Plan(q0, v0)


def test_policy_planner_rejects_another_task(actor_file):
    from ContactModelStudy.RL.PolicyPlanner import RLPolicyPlanner
    from ContactModelStudy.Tasks.BallReorient import BallReorient
    with pytest.raises(ValueError, match="trained on cube_reorient"):
        RLPolicyPlanner(actor_file, BallReorient())


def test_run_episodes_fits_its_options_to_the_policy(actor_file):
    from ContactModelStudy.Drivers import run_episodes as drv
    base = ["--planner", "rl", "--policy", str(actor_file), "--no-video", "--no-results"]
    args = drv.parseArgs(base, allow_rl=True)
    assert (args.steps, args.settle, args.ctrl_time_step) == (7, 0.25, None)
    for extra in (["--no-stop-on-success"], ["--steps", "8"], ["--settle", "1"], ["--substeps", "4"],
                  ["--task", "ball_reorient"], ["--uncertainty"]):
        with pytest.raises(SystemExit):
            drv.parseArgs(base + extra, allow_rl=True)
    with pytest.raises(SystemExit):
        drv.parseArgs(base)                              # pooled drivers: MPPI only
    with pytest.raises(SystemExit):
        drv.parseArgs(["--policy", str(actor_file)])
    mppi = drv.parseArgs(["--no-video", "--no-results"])
    assert (mppi.planner, mppi.steps, mppi.settle, mppi.ctrl_time_step) == ("mppi", 1000, 1.0, 0.064)


# -- the training environment -------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.parametrize("model", MODELS)
def test_train_env_steps_and_resets_finished_worlds(model):
    from ContactModelStudy.RL.TrainEnv import TrainEnv
    env = TrainEnv(_smoke(model))
    try:
        obs, _ = env.reset()
        assert obs.shape == (4, 73) and torch.isfinite(obs).all()
        assert torch.all(obs[:, -1] == 1.0)
        for k in range(5):
            obs, reward, terminated, truncated, info = env.step(torch.zeros(4, 16, device=env.device))
            assert reward.shape == terminated.shape == truncated.shape == (4, 1)
            assert torch.isfinite(reward).all() and torch.all(reward <= 0) and not truncated.any()
        assert terminated.all()                          # the 5-step deadline
        assert torch.all(obs[:, -1] == 1.0)              # and every world started over
    finally:
        env.close()


@pytest.mark.gpu
def test_each_world_gets_its_own_goal():
    from ContactModelStudy.RL.TrainEnv import TrainEnv
    env = TrainEnv(_smoke(num_envs=16, goal_difficulty=4))
    try:
        env.reset()
        quats = env._goals[:, 3:7].cpu().numpy()
        assert len({tuple(np.round(q, 4)) for q in quats}) > 1
        assert np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-5)
    finally:
        env.close()


# -- training round trip ------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not HAS_SKRL, reason="training needs skrl")
def test_train_resume_and_evaluate(tmp_path):
    from ContactModelStudy.Drivers import run_episodes, train_rl_policy
    config = REPO_ROOT / "configs" / "rl" / "cube_m2_smoke.json"
    first, second = tmp_path / "first", tmp_path / "second"
    assert train_rl_policy.main(["--config", str(config), "--timesteps", "8", "--output", str(first)]) == 0
    summary = json.loads((first / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["ppo_updates"] == 2
    assert {"agent.pt", "actor.pt", "manifest.json"} <= {p.name for p in first.iterdir()}

    assert train_rl_policy.main(["--resume", str(first / "agent.pt"), "--timesteps", "4",
                                 "--output", str(second)]) == 0
    assert json.loads((second / "manifest.json").read_text())["environment"] == \
        json.loads((first / "manifest.json").read_text())["environment"]

    results = tmp_path / "eval.json"
    assert run_episodes.main(["--planner", "rl", "--policy", str(second / "actor.pt"), "--n-episodes", "1",
                              "--no-video", "--results", str(results)]) == 0
    recorded = json.loads(results.read_text())
    assert recorded["n_episodes"] == 1
    assert recorded["configs"]["planner"]["class"] == "RLPolicyPlanner"
