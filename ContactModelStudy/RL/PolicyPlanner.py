"""A trained actor as a planner, so ``run_episodes`` runs it exactly as it runs MPPI.

``Plan`` maps the measured state to a command through the same observation
encoding and action decoding the policy was trained with, using the actor's
mean (no sampling). It predicts nothing, so it holds no simulator; the episode
is scored by ``run_episodes``' eval simulator like any other.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ContactModelStudy.Planners.PlannerBase import PlannerBase


@dataclass
class RLPolicyConfig:
    """What the recorder keeps about a policy run: the actor file and its training contract."""

    policy: str
    policy_sha256: str
    device: str = "cpu"
    environment: dict = field(default_factory=dict)


def policyContract(path: str | Path) -> dict:
    """The ``EnvConfig`` fields a policy was trained under, without building the network on a device."""
    from ContactModelStudy.RL.Policy import load_actor
    return load_actor(path)[1]["config"]


class RLPolicyPlanner(PlannerBase):
    """Runs an ``actor.pt`` from ``train_rl_policy``.

    The task handed in is the one ``run_episodes`` sets each episode's goal on;
    the policy reads the goal from it at every plan. A step counter since
    ``Reset`` supplies the episode's remaining time, which the policy observes.
    """

    name = "RLPolicy"

    def __init__(self, policy: str | Path, task, device: str = "cpu"):
        """Load an actor and check it fits ``task``.

        Raises:
            ValueError: If the actor was trained on another task, with another
                encoding, or at a control step that is not a whole number of
                ``task.timestep``.
        """
        import torch
        from ContactModelStudy.RL import Encoding
        from ContactModelStudy.RL.Config import EnvConfig
        from ContactModelStudy.RL.Policy import load_actor
        from ContactModelStudy.Tasks.Registry import TASK_MODULES

        path = Path(policy)
        actor, contract = load_actor(path)
        for key, mine in (("observation_profile", Encoding.OBSERVATION_PROFILE),
                          ("action_profile", Encoding.ACTION_PROFILE)):
            if contract.get(key) != mine:
                raise ValueError(f"{path} was trained with {key} {contract.get(key)!r}; this code uses {mine!r}")
        env = EnvConfig(**contract["config"])
        if type(task).__name__ != TASK_MODULES[env.task_name]:
            raise ValueError(f"{path} was trained on {env.task_name}, not {type(task).__name__}")
        substeps = env.control_dt / task.timestep
        if abs(substeps - round(substeps)) > 1e-6 or round(substeps) < 1:
            raise ValueError(f"The policy's control step {env.control_dt} s is not a whole number of "
                             f"the task's {task.timestep} s timestep")

        self._torch = torch
        self.task = task
        self.substeps = int(round(substeps))
        self.env_config = env
        self.config = RLPolicyConfig(policy=str(path.resolve()),
                                     policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                     device=str(device), environment=contract)
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.codec = Encoding.ActionCodec.fromModel(task.mjm, rate=env.target_rate,
                                                    control_dt=env.control_dt, device=self.device)
        self.encoder = Encoding.ObservationEncoder(task.mjm, task.indices, self.codec,
                                                   env.episode_steps, device=self.device)
        self._steps = 0

    def Reset(self, mean: np.ndarray | None = None) -> None:
        """Start a new episode's clock. ``mean`` is for sampling-based planners and is ignored."""
        self._steps = 0
        self.last_plan_ok = False

    def Plan(self, q: np.ndarray, q_dot: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        """The actor's command for the measured state; ``u``, the command in force, is required."""
        if u is None:
            raise ValueError("RLPolicyPlanner needs the applied command u: the policy moves the target from it")
        torch = self._torch

        def row(x):
            return torch.as_tensor(np.asarray(x, dtype=np.float32).ravel(), device=self.device)[None]

        previous = row(u)
        with torch.no_grad():
            obs = self.encoder(row(q), row(q_dot), row(self.task.params["target_pos"]),
                               row(self.task.getGoal()), previous,
                               torch.tensor([self._steps], device=self.device))
            command = self.codec.decode(self.actor(obs), previous)
        self._steps += 1
        out = command[0].cpu().numpy()
        self.last_plan_ok = bool(np.isfinite(out).all())
        return out

    def __repr__(self) -> str:
        e = self.env_config
        return (f"RLPolicyPlanner({Path(self.config.policy).name}, trained on {e.rollout_model} "
                f"{e.task_name}, control {e.control_dt * 1e3:g} ms, {e.episode_steps} steps)")
