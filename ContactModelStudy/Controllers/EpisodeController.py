"""Controller boundary used by the finite-case episode driver, without PPO."""
from typing import Protocol


class EpisodeController(Protocol):
    def reset(self, case) -> None: ...
    def set_goal(self, case) -> None: ...
    def act(self, env, observation): ...
    def describe(self) -> dict: ...
    def close(self) -> None: ...


class PolicyEpisodeController:
    def __init__(self, policy):
        self.policy = policy
        self.last_command_info = {}

    def reset(self, case):
        # The feed-forward actor has no hidden state. The environment reset
        # initializes both the observation history and the applied target.
        pass

    def act(self, env, observation):
        return self.policy.act(observation, env.current_target)

    def set_goal(self, case):
        # The feed-forward actor receives the new goal in its observation.
        pass

    def describe(self):
        return {"kind": "rl", "observation_profile": self.policy.manifest["observation_profile"],
                "command": "absolute_actuator_target", "new_PPO_updates": 0}

    def close(self):
        pass
