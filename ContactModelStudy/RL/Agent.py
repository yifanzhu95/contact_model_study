"""skrl's PPO, set up for ``TrainEnv``: a Gaussian policy around ``Policy.Actor`` and a value MLP.

The algorithm, rollout memory, logging and periodic checkpoints are skrl's
own. The only addition is a stop check after each PPO update, so a run asked to
stop (wall-time budget, SIGTERM) ends on a whole update with nothing half done.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn

from ContactModelStudy.RL.Config import PPOConfig
from ContactModelStudy.RL.Policy import Actor, mlp

SKRL_VERSION = "2.1.0"


class TrainingStopped(Exception):
    """Raised out of the trainer after a PPO update when a stop was requested."""

    def __init__(self, reason: str, timestep: int):
        super().__init__(f"training stopped after timestep {timestep}: {reason}")
        self.reason, self.timestep = reason, timestep


def build_agent(env, config: PPOConfig, *, directory: str | Path, checkpoint_interval: int = 0,
                should_stop=None):
    """skrl PPO for ``env``; returns ``(agent, actor)``, the actor being the policy's mean network.

    Args:
        env: A ``TrainEnv``.
        config: PPO settings.
        directory: Where skrl writes TensorBoard logs and periodic checkpoints
            (``<directory>/skrl``).
        checkpoint_interval: Save a checkpoint every this many PPO updates; 0
            saves none.
        should_stop: Called after each PPO update; a non-empty string stops
            training there by raising ``TrainingStopped``.

    The agent is initialized by the skrl trainer it is handed to.
    """
    if version("skrl") != SKRL_VERSION:
        raise RuntimeError(f"This integration is written against skrl=={SKRL_VERSION}, found {version('skrl')}")
    from skrl.agents.torch import ExperimentCfg
    from skrl.agents.torch.ppo import PPO, PPO_CFG
    from skrl.memories.torch import RandomMemory
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

    batch = env.num_envs * config.rollout_steps
    if batch % config.mini_batches:
        raise ValueError(f"num_envs * rollout_steps ({batch}) must divide evenly into {config.mini_batches} mini-batches")

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
            self.network = mlp(env.observation_space.shape[0], 1, config.hidden_sizes)

        def compute(self, inputs, role=""):
            return self.network(inputs["observations"]), {}

    class BoundedPPO(PPO):
        def post_interaction(self, *, timestep: int, timesteps: int) -> None:
            super().post_interaction(timestep=timestep, timesteps=timesteps)
            if should_stop is not None and not self._rollout % self.cfg.rollouts:
                reason = should_stop()
                if reason:
                    raise TrainingStopped(reason, timestep + 1)

    policy, value = Policy().to(env.device), Value().to(env.device)
    # discount_factor = 1: the task is finite and its remaining time is observed.
    cfg = PPO_CFG(rollouts=config.rollout_steps, learning_epochs=config.learning_epochs,
                  mini_batches=config.mini_batches, learning_rate=config.learning_rate,
                  discount_factor=1.0, gae_lambda=config.gae_lambda,
                  ratio_clip=config.ratio_clip, entropy_loss_scale=config.entropy_loss_scale,
                  experiment=ExperimentCfg(directory=str(directory), experiment_name="skrl",
                                           checkpoint_interval=checkpoint_interval * config.rollout_steps))
    memory = RandomMemory(memory_size=config.rollout_steps, num_envs=env.num_envs, device=env.device)
    agent = BoundedPPO(models={"policy": policy, "value": value}, memory=memory,
                       observation_space=env.observation_space, state_space=None,
                       action_space=env.action_space, device=env.device, cfg=cfg)
    return agent, policy.actor
