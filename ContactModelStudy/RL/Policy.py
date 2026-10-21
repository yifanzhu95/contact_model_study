"""Inference network and export format. This module deliberately has no skrl dependency."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


def mlp(inputs, outputs, hidden_sizes):
    layers = []
    for width in hidden_sizes:
        layers.extend((nn.Linear(inputs, width), nn.ELU()))
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, observation_size=73, action_size=16, hidden_sizes=(64, 64)):
        super().__init__()
        self.spec = {"observation_size": observation_size, "action_size": action_size,
                     "hidden_sizes": list(hidden_sizes)}
        self.network = mlp(observation_size, action_size, hidden_sizes)
        # Small initial means keep the first commanded changes close to the grasp.
        nn.init.uniform_(self.network[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observation):
        return self.network(observation)


def export_actor(actor: Actor, environment: dict, path: str | Path):
    torch.save({"format": "leap_actor_v1", "network": actor.spec,
                "actor_state": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
                "environment": environment}, Path(path))


def load_actor(path: str | Path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "leap_actor_v1":
        raise ValueError("Unsupported policy artifact")
    actor = Actor(**payload["network"])
    actor.load_state_dict(payload["actor_state"], strict=True)
    actor.eval()
    return actor, payload["environment"]
