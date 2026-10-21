"""Load a trained actor and issue absolute actuator targets without a trainer."""
from __future__ import annotations

import torch

from ContactModelStudy.RL.Encoding import ActionCodec, ACTION_PROFILE, OBSERVATION_PROFILE
from ContactModelStudy.RL.Policy import load_actor


class RLPolicyController:
    """The caller supplies encoded observations and the actually applied previous target.

    This feed-forward controller has no hidden episode state. The environment's
    shared encoder owns history initialization. The finite-case protocol in
    run_episodes supplies it through PolicyEpisodeController. Hardware state
    estimation remains outside this simulation controller.
    """
    def __init__(self, path, device="cpu"):
        self.device = torch.device(device)
        self.actor, self.manifest = load_actor(path)
        self.actor.to(self.device)
        action = self.manifest["action"]
        if (action["profile"] != ACTION_PROFILE
                or self.manifest["observation_profile"] != OBSERVATION_PROFILE):
            raise ValueError("Unsupported policy observation/action profile")
        cfg = self.manifest["config"]
        self.codec = ActionCodec(action["lower"], action["upper"],
                                 rate=cfg["target_rate"], control_dt=cfg["control_dt"], device=self.device)

    @torch.no_grad()
    def act(self, observation, previous_target):
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        previous_target = torch.as_tensor(previous_target, dtype=torch.float32, device=self.device)
        if (observation.ndim != 2 or observation.shape[1] != 73
                or previous_target.shape != (observation.shape[0], 16)
                or not torch.isfinite(observation).all() or not torch.isfinite(previous_target).all()):
            raise ValueError("Expected finite batched observations and previous targets")
        if ((previous_target < self.codec.lower) | (previous_target > self.codec.upper)).any():
            raise ValueError("Previous target is outside the exported actuator range")
        return self.codec.decode(self.actor(observation), previous_target)
