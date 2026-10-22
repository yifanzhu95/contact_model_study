"""Actual fixed-version PPO update, probability/timeout contracts, and actor deployment."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ContactModelStudy.Controllers.RLPolicyController import RLPolicyController
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig, Transition
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv
from ContactModelStudy.RL.Policy import export_actor
from ContactModelStudy.RL.SkrlPPO import (PPOConfig, build_agent, collect_updates,
                                         restore_training_checkpoint, save_training_checkpoint)


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    bank = CaseBank.development(count=4, seed=5, difficulty=8)
    env = LeapVectorEnv(EnvConfig(episode_steps=4), bank)
    cfg = PPOConfig(steps_per_update=4, hidden_sizes=(16, 16))
    agent, actor = build_agent(env, cfg)
    yield env, cfg, agent, actor
    env.close()


@pytest.mark.ppo_update
def test_real_ppo_update_and_episode_reset_resume(setup, tmp_path):
    env, cfg, agent, actor = setup
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    metrics = []
    stats = collect_updates(env, agent, updates=1, on_update=lambda i, m: metrics.append((i, m)))
    assert metrics[0][0] == 1
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics[0][1].values())
    assert stats["ppo_updates"] == 1 and stats["transitions"] == 8
    assert stats["episode_resets"] == 2
    assert any(not torch.equal(v, before[k]) for k, v in actor.state_dict().items())
    for key in ("values", "returns", "advantages"):
        assert torch.isfinite(agent.memory.get_tensor_by_name(key)).all()
    save_training_checkpoint(agent, actor, env, cfg, tmp_path / "training.pt")
    restored, restored_actor = build_agent(env, cfg)
    report = restore_training_checkpoint(restored, env, cfg, tmp_path / "training.pt")
    assert report == {"resume_mode": "reset_all_episodes", "partial_rollout_restored": False}
    for key, value in actor.state_dict().items():
        assert torch.equal(restored_actor.state_dict()[key], value)
    assert restored.optimizer.state_dict()["state"]
    assert not len(restored.memory)


def test_raw_latent_log_probability_and_memory_ownership(setup):
    env, cfg, agent, actor = setup
    obs = env.reset_all([0, 1])
    with torch.no_grad():
        latent, outputs = agent.act(obs, None, timestep=0, timesteps=4)
        original = latent.clone()
        stored_log_prob = outputs["log_prob"].clone()
        _, recalculated = agent.models["policy"].act({"observations": obs, "taken_actions": latent}, role="policy")
        ratio = (recalculated["log_prob"] - stored_log_prob).exp()
        torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=1e-6, rtol=0)
        tr = env.step(latent)
        reward = tr.reward.clone()
        agent.record_transition(observations=obs, states=None, actions=latent, rewards=tr.reward.clone(),
                                next_observations=tr.next_obs, next_states=None,
                                terminated=tr.terminated, truncated=tr.truncated,
                                infos={}, timestep=0, timesteps=4)
    assert torch.equal(latent, original) and torch.equal(tr.reward, reward)
    env.reset_all([2, 3])
    assert torch.equal(agent.memory.get_tensor_by_name("actions")[0], original)
    assert torch.equal(agent.memory.get_tensor_by_name("observations")[0], obs)


def test_bootstrap_uses_final_next_observation_without_modifying_raw_reward(setup):
    env, cfg, agent, actor = setup
    obs = env.reset_all()
    # A state-dependent known critic exposes using the wrong final observation.
    class KnownValue(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]
    agent.models["value"].network = KnownValue()
    with torch.no_grad():
        latent, _ = agent.act(obs, None, timestep=0, timesteps=4)
        final_obs = obs.clone()
        final_obs[:, 0] = torch.tensor([7., 9.])
        raw = torch.full((2, 1), -1.)
        agent.record_transition(observations=obs, states=None, actions=latent, rewards=raw.clone(),
                                next_observations=final_obs, next_states=None,
                                terminated=torch.zeros(2, 1, dtype=torch.bool),
                                truncated=torch.tensor([[True], [False]]),
                                infos={}, timestep=0, timesteps=4)
    assert raw.tolist() == [[-1.], [-1.]]
    assert agent.memory.get_tensor_by_name("rewards")[0].tolist() == [[6.], [-1.]]


@pytest.mark.ppo_update
def test_mixed_episode_boundaries_and_nonterminal_rollout_bootstrap():
    """Four synthetic samples through the actual PPO/GAE implementation."""
    from gymnasium.spaces import Box
    import numpy as np
    class MixedEnv:
        num_envs, device = 2, torch.device('cpu')
        observation_space = Box(-np.inf, np.inf, shape=(73,), dtype=np.float32)
        action_space = Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)
        def reset_all(self):
            self.k, self.coordinates = 0, torch.tensor([10., 20.])
            self.case_ids, self.resets = ['initial-0', 'initial-1'], []
            return self.observe()
        def observe(self):
            obs = torch.zeros(2, 73); obs[:, 0] = self.coordinates
            return obs
        def reset_indices(self, ids):
            self.resets.append(tuple(ids))
            for i in ids:
                self.coordinates[i] = 100.*(i+1)
                self.case_ids[i] = f'reset-{i}'
        def step(self, latent):
            assert latent.shape == (2, 16) and torch.isfinite(latent).all()
            self.coordinates = torch.tensor([11., 21.]) if self.k == 0 else torch.tensor([101., 22.])
            reward = torch.tensor([[-1.], [-3.]]) if self.k == 0 else torch.tensor([[-2.], [-4.]])
            done = torch.tensor([[True], [False]]) if self.k == 0 else torch.tensor([[False], [True]])
            reasons = ('success', 'running') if self.k == 0 else ('running', 'deadline')
            self.k += 1
            return Transition(self.observe(), reward, done, torch.zeros_like(done), reasons, {})
    torch.set_num_threads(1); torch.manual_seed(13)
    env = MixedEnv()
    agent, _ = build_agent(env, PPOConfig(steps_per_update=2, learning_epochs=1,
                                         hidden_sizes=(4,), gae_lambda=1., learning_rate=1e-12))
    # Keep the real critic/optimizer parameter references. With positive inputs,
    # this ELU network exactly represents V(observation) = observation[0].
    with torch.no_grad():
        for p in agent.models['value'].parameters(): p.zero_()
        network = agent.models['value'].network
        network[0].weight[0, 0] = network[-1].weight[0, 0] = 1.
    transitions = []
    stats = collect_updates(env, agent, updates=1,
                            on_transition=lambda s, tr, ids: transitions.append((tr, ids)))
    assert stats['transitions'] == 4 and stats['ppo_updates'] == 1
    assert env.resets == [(0,), (1,)]
    torch.testing.assert_close(agent.memory.get_tensor_by_name('returns')[:, :, 0],
                               torch.tensor([[-1., -7.], [99., -4.]]), rtol=0, atol=1e-5)
    torch.testing.assert_close(agent.memory.get_tensor_by_name('observations')[:, :, 0],
                               torch.tensor([[10., 20.], [100., 21.]]), rtol=0, atol=0)
    assert transitions[0][1] == ('initial-0', 'initial-1')
    assert transitions[1][1] == ('reset-0', 'initial-1')
    assert transitions[0][0].next_obs[:, 0].tolist() == [11., 21.]
    assert not agent.tracking_data


def test_exported_controller_matches_training_command_without_skrl(setup, tmp_path):
    env, cfg, agent, actor = setup
    obs = env.reset_all([0, 1])
    with torch.no_grad():
        expected = env.codec.decode(actor(obs), env.current_target)
    path = tmp_path / "actor.pt"
    export_actor(actor, env.describe(), path)
    controller = RLPolicyController(path)
    assert torch.equal(controller.act(obs, env.current_target), expected)
    subprocess.run([sys.executable, "-c",
        "import sys; from ContactModelStudy.Controllers.RLPolicyController import RLPolicyController; "
        "RLPolicyController(sys.argv[1]); assert 'skrl' not in sys.modules", str(path)],
        cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)


def test_evaluation_records_aborted_case_instead_of_reporting_complete_success(setup, tmp_path, monkeypatch):
    from ContactModelStudy.Drivers import evaluate_rl
    from ContactModelStudy.RL.Contracts import NumericalTransitionError
    env, cfg, agent, actor = setup
    policy, cases, output = tmp_path / "actor.pt", tmp_path / "cases.json", tmp_path / "evaluation"
    export_actor(actor, env.describe(), policy)
    env.case_bank.save(cases)
    class FaultEnv(LeapVectorEnv):
        def step_command(self, command):
            raise NumericalTransitionError("injected nonfinite transition", self.backend.read_frame())
    monkeypatch.setattr(evaluate_rl, "LeapVectorEnv", FaultEnv)
    with pytest.raises(NumericalTransitionError):
        evaluate_rl.main(["--policy", str(policy), "--case-bank", str(cases), "--output", str(output)])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["execution_status"] == "failed"
    assert summary["failure_reason"] == "numerical_failure"
    assert summary["allocated_cases"] == 4 and summary["not_executed"] == 3
    assert summary["completed_cases"] == 0
    assert (output / "fault_state.npz").exists()
