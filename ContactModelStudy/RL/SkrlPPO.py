"""Use the fixed, existing skrl PPO implementation; only collection is project-owned."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
import math
from pathlib import Path
import warnings

import torch
from torch import nn

from ContactModelStudy.RL.Policy import Actor, export_actor, mlp


@dataclass(frozen=True)
class PPOConfig:
    steps_per_update: int = 8
    learning_epochs: int = 2
    mini_batches: int = 2
    learning_rate: float = 0.0003
    hidden_sizes: tuple[int, ...] = (64, 64)
    ratio_clip: float = 0.2
    gae_lambda: float = 0.95
    entropy_loss_scale: float = 0.0
    initial_log_std: float = -1.5

    def __post_init__(self):
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1
               for x in (self.steps_per_update, self.learning_epochs, self.mini_batches, *self.hidden_sizes)):
            raise ValueError("PPO dimensions/counts must be positive integers")
        if not self.hidden_sizes or not 0 < self.learning_rate < float("inf"):
            raise ValueError("Hidden layers and a finite positive learning rate are required")
        if (not 0 < self.ratio_clip < 1 or not 0 <= self.gae_lambda <= 1
                or not math.isfinite(self.entropy_loss_scale) or self.entropy_loss_scale < 0
                or not math.isfinite(self.initial_log_std) or not -5 <= self.initial_log_std <= 1):
            raise ValueError("Invalid PPO clip, GAE lambda, entropy scale or initial log std")


def build_agent(env, config: PPOConfig):
    if version("skrl") != "2.1.0":
        raise RuntimeError("This integration is audited for skrl==2.1.0")
    from skrl.agents.torch import ExperimentCfg
    from skrl.agents.torch.ppo import PPO, PPO_CFG
    from skrl.memories.torch import RandomMemory
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
    batch = env.num_envs * config.steps_per_update
    if batch < 2 or batch % config.mini_batches:
        raise ValueError("A PPO batch must have >=2 samples and divide evenly into mini_batches")

    class Policy(GaussianMixin, Model):
        def __init__(self):
            Model.__init__(self, observation_space=env.observation_space,
                           action_space=env.action_space, device=env.device)
            GaussianMixin.__init__(self, clip_actions=False, clip_mean_actions=False,
                                   clip_log_std=True, min_log_std=-5, max_log_std=1)
            self.actor = Actor(hidden_sizes=config.hidden_sizes)
            self.log_std = nn.Parameter(torch.full((16,), config.initial_log_std))

        def compute(self, inputs, role=""):
            return self.actor(inputs["observations"]), {"log_std": self.log_std}

    class Value(DeterministicMixin, Model):
        def __init__(self):
            Model.__init__(self, observation_space=env.observation_space,
                           action_space=env.action_space, device=env.device)
            DeterministicMixin.__init__(self, clip_actions=False)
            self.network = mlp(73, 1, config.hidden_sizes)

        def compute(self, inputs, role=""):
            return self.network(inputs["observations"]), {}

    policy, value = Policy().to(env.device), Value().to(env.device)
    cfg = PPO_CFG(rollouts=config.steps_per_update, learning_epochs=config.learning_epochs,
                  mini_batches=config.mini_batches, learning_rate=config.learning_rate,
                  discount_factor=1.0, gae_lambda=config.gae_lambda, time_limit_bootstrap=True,
                  ratio_clip=config.ratio_clip, entropy_loss_scale=config.entropy_loss_scale,
                  grad_norm_clip=0.5, value_clip=0.2, value_loss_scale=2.5,
                  kl_threshold=0.0, learning_rate_scheduler=None,
                  random_timesteps=0, learning_starts=0, mixed_precision=False,
                  observation_preprocessor=None, state_preprocessor=None, value_preprocessor=None,
                  rewards_shaper=None,
                  experiment=ExperimentCfg(write_interval=0, checkpoint_interval=0, wandb=False))
    memory = RandomMemory(memory_size=config.steps_per_update, num_envs=env.num_envs, device=env.device)
    agent = PPO(models={"policy": policy, "value": value}, memory=memory,
                observation_space=env.observation_space, state_space=None,
                action_space=env.action_space, device=env.device, cfg=cfg)
    # No trainer object is involved; disabled automatic logging/checkpointing
    # makes its optional configuration unnecessary in this fixed version.
    agent.init()
    agent.enable_training_mode(True)
    return agent, policy.actor


def collect_updates(env, agent, *, updates: int, on_transition=None, on_update=None, on_boundary=None,
                    should_stop=None):
    """No implicit resets, no physics derivatives, no locally implemented PPO/GAE."""
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError("updates must be a positive integer")
    total_steps = updates * agent.cfg.rollouts
    outcomes = {k: 0 for k in ("success", "physical_drop", "deadline")}
    reset_count = 0
    completed_steps, stop_reason = 0, None
    for step in range(total_steps):
        if step % agent.cfg.rollouts == 0 and should_stop:
            stop_reason = should_stop()
            if stop_reason:
                break
        if step == 0:
            obs = env.reset_all()
        agent.pre_interaction(timestep=step, timesteps=total_steps)
        with torch.no_grad():
            latent, outputs = agent.act(obs, None, timestep=step, timesteps=total_steps)
            if not torch.isfinite(outputs["log_prob"]).all():
                raise FloatingPointError("Nonfinite action log probability")
            tr = env.step(latent)
            if not torch.isfinite(tr.reward).all() or not torch.isfinite(tr.next_obs).all():
                raise FloatingPointError("Nonfinite transition; update aborted")
            agent.record_transition(observations=obs, states=None, actions=latent,
                                    rewards=tr.reward.clone(), next_observations=tr.next_obs,
                                    next_states=None, terminated=tr.terminated,
                                    truncated=tr.truncated & ~tr.terminated, infos={},
                                    timestep=step, timesteps=total_steps)
        # Account for accepted samples before an optimizer failure can abort
        # post_interaction. Do not reset: PPO still needs the final observation.
        if on_transition:
            on_transition(step, tr, tuple(env.case_ids))
        for reason in tr.reasons:
            if reason in outcomes:
                outcomes[reason] += 1
        # post_interaction can call backward(), so it must be outside no_grad.
        agent.post_interaction(timestep=step, timesteps=total_steps)
        if not all(torch.isfinite(p).all().item() for p in agent.models["policy"].parameters()):
            raise FloatingPointError("Nonfinite policy parameters after update")
        if not all(torch.isfinite(p).all().item() for p in agent.models["value"].parameters()):
            raise FloatingPointError("Nonfinite value parameters after update")
        if (step + 1) % agent.cfg.rollouts == 0:
            # Read the library's measurements; safety must not depend on
            # whether the caller logs them. No extra actions or local update.
            metrics = {key: float(agent.tracking_data[key][-1]) for key in
                       ("Loss / Policy loss", "Loss / Value loss", "Policy / Standard deviation")}
            if not all(math.isfinite(x) for x in metrics.values()):
                raise FloatingPointError("Nonfinite PPO update diagnostics")
            if on_update:
                on_update((step + 1) // agent.cfg.rollouts, metrics)
            # skrl's write_interval=0 disables its automatic draining. Metrics
            # have been persisted by the callback and must not grow in RAM.
            agent.tracking_data.clear()
        ids = tr.done_ids
        if ids.numel():
            reset_count += ids.numel()
            env.reset_indices(ids.tolist())
        obs = env.observe()
        completed_steps = step + 1
        # Save only after transition accounting and indexed resets. Restores
        # intentionally discard physical episodes and any rollout memory.
        if on_boundary and (step + 1) % agent.cfg.rollouts == 0:
            on_boundary((step + 1) // agent.cfg.rollouts)
    return {"vector_steps": completed_steps, "transitions": completed_steps * env.num_envs,
            "ppo_updates": completed_steps // agent.cfg.rollouts, "stop_reason": stop_reason,
            "episode_resets": reset_count, "completed_episodes": outcomes,
            "purpose": "short_interface_smoke_not_a_learning_comparison"}


def save_training_checkpoint(agent, actor, env, config, path, *, progress=None):
    """A checkpoint for episode-reset restart; no claim of exact physics continuation."""
    path = Path(path)
    agent.save(str(path))
    # The optimizer/value/policy are in skrl's native checkpoint. Additional
    # RNG/config metadata documents what can and cannot be resumed.
    metadata = {"torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(env.device) if env.device.type == "cuda" else None,
                "ppo": asdict(config),
                "environment": env.describe(), "resume_mode": "reset_all_episodes",
                "case_rng_states": [rng.bit_generator.state for rng in env._rng]}
    if progress is not None:
        metadata["progress"] = progress
        metadata["checkpoint_schema"] = "leap_training_metadata_v1"
    torch.save(metadata, path.with_suffix(".resume.pt"))
    export_actor(actor, env.describe(), path.with_name("actor.pt"))


def restore_training_checkpoint(agent, env, config, path):
    """Load weights/optimizer and RNGs, then restart episodes; never restore a partial buffer."""
    path = Path(path)
    from ContactModelStudy.RL.RunIO import checkpoint_metadata, recorded_environment_config
    metadata = checkpoint_metadata(path)
    # The scene content hash is contractual; its absolute checkout path is not.
    saved, current = dict(metadata["environment"]), dict(env.describe())
    saved.pop("scene_path", None); current.pop("scene_path", None)
    if 'scene_model_fingerprint' not in saved:
        current.pop('scene_model_fingerprint', None)
        warnings.warn('Legacy checkpoint has only a top-level XML hash; included scene and mesh identity '
                      'cannot be verified. New checkpoints record the compiled source scene.', UserWarning)
    saved["config"] = recorded_environment_config(saved).to_dict()
    current["config"] = recorded_environment_config(current).to_dict()
    if (metadata["ppo"] != asdict(config) or saved != current
            or len(metadata["case_rng_states"]) != len(env._rng)
            or metadata["resume_mode"] != "reset_all_episodes"):
        raise ValueError("Resume requires the same PPO/environment/case-bank contract")
    agent.load(str(path))
    for rng, state in zip(env._rng, metadata["case_rng_states"], strict=True):
        rng.bit_generator.state = state
    torch.set_rng_state(metadata["torch_rng"])
    if env.device.type == "cuda":
        torch.cuda.set_rng_state(metadata["cuda_rng"], env.device)
    agent.memory.reset()
    agent._rollout = 0
    return {"resume_mode": metadata["resume_mode"], "partial_rollout_restored": False}
