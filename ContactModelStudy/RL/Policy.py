"""The policy network and its export format; importing this needs torch but not skrl.

``actor.pt`` holds the deterministic mean network and the environment contract
it was trained under (``TrainEnv.describe``), which ``RLPolicyPlanner`` checks
before running it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

ACTOR_FORMAT = "leap_actor_v2"


def mlp(inputs: int, outputs: int, hidden_sizes) -> nn.Sequential:
    layers = []
    for width in hidden_sizes:
        layers.extend((nn.Linear(inputs, width), nn.ELU()))
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Observation to latent action mean."""

    def __init__(self, observation_size: int = 73, action_size: int = 16, hidden_sizes=(256, 128)):
        super().__init__()
        self.spec = {"observation_size": observation_size, "action_size": action_size,
                     "hidden_sizes": list(hidden_sizes)}
        self.network = mlp(observation_size, action_size, hidden_sizes)
        # Small initial means keep the first commanded changes close to the grasp.
        nn.init.uniform_(self.network[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def export_actor(actor: Actor, environment: dict, path: str | Path) -> None:
    torch.save({"format": ACTOR_FORMAT, "network": actor.spec,
                "actor_state": {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
                "environment": environment}, Path(path))


def load_actor(path: str | Path) -> tuple[Actor, dict]:
    """``(actor, environment contract)`` from an ``actor.pt``, the actor in eval mode on the CPU."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != ACTOR_FORMAT:
        raise ValueError(f"{path} is not a {ACTOR_FORMAT} actor (format {payload.get('format')!r})")
    actor = Actor(**payload["network"])
    actor.load_state_dict(payload["actor_state"], strict=True)
    actor.eval()
    return actor, payload["environment"]
