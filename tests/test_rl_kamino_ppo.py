"""One real Kamino PPO update, reset boundaries, and portable actor/checkpoint checks."""
from importlib.util import find_spec
import json

import numpy as np
import pytest
import torch
import warp as wp

from ContactModelStudy.Controllers.RLPolicyController import RLPolicyController
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv
from ContactModelStudy.RL.SkrlPPO import (
    PPOConfig, build_agent, collect_updates, restore_training_checkpoint, save_training_checkpoint,
)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available() or find_spec("newton") is None,
    reason="Needs explicit CUDA Torch and Newton 1.6 runtime")]


@pytest.mark.ppo_update
def test_kamino_ppo_update_reset_export_and_restore(tmp_path, monkeypatch, record_property):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    env = LeapVectorEnv(
        EnvConfig(physics_backend="kamino", contact_preset="M5", device="cuda:0",
                  episode_steps=4, seed=9),
        CaseBank.development(count=4, seed=9, difficulty=8))
    try:
        # Observe the actual native substeps without changing the physics call.
        native_step = env.backend.sim.Step_GPU
        solver_steps = []
        def observe_step(steps=1):
            native_step(steps)
            wp.synchronize_device(str(env.device))
            status = env.backend.sim.solver.status.numpy().copy()
            solver_steps.append({name: status[name].tolist() for name in
                                 ("converged", "iterations", "r_p", "r_d", "r_c")})
        monkeypatch.setattr(env.backend.sim, "Step_GPU", observe_step)

        cfg = PPOConfig(steps_per_update=4, hidden_sizes=(16, 16))
        agent, actor = build_agent(env, cfg)
        before = {k: v.clone() for k, v in actor.state_dict().items()}
        value_before = {k: v.clone() for k, v in agent.models["value"].state_dict().items()}
        transitions, metrics = [], []
        def remember_transition(step, tr, case_ids):
            transitions.append((tr, tr.next_obs.clone(), tr.reward.clone()))
        stats = collect_updates(env, agent, updates=1, on_transition=remember_transition,
                                on_update=lambda i, values: metrics.append(values))
        assert stats["ppo_updates"] == 1 and stats["transitions"] == 8
        assert stats["episode_resets"] == 2
        assert stats["completed_episodes"] == {"success": 0, "physical_drop": 0, "deadline": 2}
        actor_delta = max((v - before[k]).abs().max().item() for k, v in actor.state_dict().items())
        assert actor_delta > 0
        assert any(not torch.equal(v, value_before[k])
                   for k, v in agent.models["value"].state_dict().items())
        assert len(metrics) == 1 and all(np.isfinite(v) for v in metrics[0].values())
        for name in ("observations", "actions", "rewards", "log_prob", "values", "returns", "advantages"):
            tensor = agent.memory.get_tensor_by_name(name)
            assert tensor.device.type == "cuda" and torch.isfinite(tensor).all(), name
        assert agent.memory.get_tensor_by_name("terminated")[-1].all()
        assert not agent.memory.get_tensor_by_name("truncated").any()
        assert len(solver_steps) == 4 * env.config.physics_steps
        assert all(np.isfinite(step[name]).all() for step in solver_steps for name in ("r_p", "r_d", "r_c"))

        save_training_checkpoint(agent, actor, env, cfg, tmp_path / "training.pt")
        cuda_rng = torch.cuda.get_rng_state(env.device).clone()
        restored, restored_actor = build_agent(env, cfg)
        report = restore_training_checkpoint(restored, env, cfg, tmp_path / "training.pt")
        assert report == {"resume_mode": "reset_all_episodes", "partial_rollout_restored": False}
        assert torch.equal(torch.cuda.get_rng_state(env.device), cuda_rng)
        assert not len(restored.memory) and restored.optimizer.state_dict()["state"]
        for name in ("policy", "value"):
            expected = agent.models[name].state_dict()
            for key, value in restored.models[name].state_dict().items():
                assert torch.equal(value, expected[key]), (name, key)
        obs = env.reset_all([0, 1])
        with torch.no_grad():
            expected_command = env.codec.decode(restored_actor(obs), env.current_target)
        controller = RLPolicyController(tmp_path / "actor.pt", device=env.device)
        command = controller.act(obs, env.current_target)
        torch.testing.assert_close(command, expected_command, rtol=0, atol=0)
        cpu_controller = RLPolicyController(tmp_path / "actor.pt")
        cpu_command = cpu_controller.act(obs.cpu(), env.current_target.cpu())
        torch.testing.assert_close(cpu_command, expected_command.cpu(), rtol=1e-5, atol=1e-6)
        assert (command >= env.codec.lower).all() and (command <= env.codec.upper).all()
        assert (command - env.current_target).abs().max() <= env.codec.delta + 1e-6
        # Neither the collector's automatic episode resets nor this later reset
        # may overwrite the terminal observations/rewards handed to PPO.
        for tr, saved_obs, saved_reward in transitions:
            assert torch.equal(tr.next_obs, saved_obs) and torch.equal(tr.reward, saved_reward)
        record_property("kamino_ppo_evidence", json.dumps({
            "stats": stats, "metrics": metrics[0], "actor_max_parameter_change": actor_delta,
            "memory_device": "cuda:0", "solver_substeps": solver_steps,
            "export_gpu_command_max_difference": (command-expected_command).abs().max().item(),
            "export_cpu_command_max_difference": (cpu_command-expected_command.cpu()).abs().max().item(),
            "resume": report,
        }, allow_nan=False))
    finally:
        env.close()
