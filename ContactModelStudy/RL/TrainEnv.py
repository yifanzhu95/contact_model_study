"""The training environment: ``num_envs`` worlds of a task on one of the study's contact models.

A thin layer over what MPPI already uses. The worlds are a
``VectorizedSimulator`` from ``GetContactModelSim``, and reward and outcomes
come from the task's own ``calcCosts`` / ``isSuccess`` / ``isFailure``, each
world scored against its own goal through a goal table. What is added here is
only what a learner needs: observation and action encoding, an episode clock
per world, and resetting worlds as they finish.

The interface is skrl's environment wrapper (``reset``, ``step``, ``state``,
``render``, ``close``, ``num_envs``, the spaces), so skrl's trainer drives it
directly. Episodes are single-goal, as in ``run_episodes``: each ends on the
first success, a drop, or the deadline, and that world is reset inside
``step``, with the next episode's first observation returned in its row.

The deadline counts as a terminal state, not a truncation: the time left is
part of the observation, so there is nothing to bootstrap past it. That also
keeps skrl from bootstrapping from the reset observation.
"""

from __future__ import annotations

import numpy as np
import torch

from ContactModelStudy.RL.Config import EnvConfig
from ContactModelStudy.RL.Encoding import (ACTION_PROFILE, OBSERVATION_PROFILE, REWARD_PROFILE,
                                           ActionCodec, ObservationEncoder, finite_reward)


def makeTrainTask(config: EnvConfig, **overrides):
    """The rollout-role task a training run uses, at the run's physics timestep."""
    from ContactModelStudy.Tasks.Registry import make_task
    options = dict(role="rollout", timestep=config.physics_dt, seed=config.seed,
                   goal_difficulty=config.goal_difficulty, hand_acc=config.hand_acc,
                   obj_acc=config.obj_acc, cost_weights=config.cost_weights)
    options.update(overrides)
    return make_task(config.task_name, **options)


class TrainEnv:
    """skrl-compatible vectorized environment; see the module docstring.

    Attributes:
        config: The run's ``EnvConfig``.
        task: The rollout-role task; its goal is never changed, goals live in
            the per-world table.
        sim: The ``VectorizedSimulator`` holding the worlds.
        start_q, start_v: The settled start state every episode begins from.
    """

    num_agents = 1
    state_space = None

    def __init__(self, config: EnvConfig):
        import gymnasium
        import warp as wp
        from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim

        self.config = config
        self.num_envs = config.num_envs
        self.device = torch.device(config.device)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (ObservationEncoder.SIZE,), np.float32)
        self.action_space = gymnasium.spaces.Box(-np.inf, np.inf, (16,), np.float32)

        self.task = makeTrainTask(config)
        self.sim = GetContactModelSim(config.rollout_model, self.task.getModelPath(), N=self.num_envs,
                                      timestep=config.physics_dt, device=str(self.device))
        mjm = self.task.mjm
        self.codec = ActionCodec.fromModel(mjm, rate=config.target_rate, control_dt=config.control_dt,
                                           device=self.device)
        self.encoder = ObservationEncoder(mjm, self.task.indices, self.codec, config.episode_steps,
                                          device=self.device)
        self._goal_start = self.task.getGoal()
        self._goals_wp = wp.array(np.tile(self.task.goal, (self.num_envs, 1)), dtype=wp.float32,
                                  device=str(self.device))
        self._goals = wp.to_torch(self._goals_wp)
        n = self.num_envs
        self._outs = {kind: wp.zeros(n, dtype=dtype, device=str(self.device))
                      for kind, dtype in (("cost", wp.float32), ("success", wp.bool), ("failure", wp.bool))}
        self._target = torch.zeros((n, 16), device=self.device)
        self._steps = torch.zeros(n, dtype=torch.int64, device=self.device)

        q0, v0, self.start_u = self.task.getInitialState()
        self.start_q, self.start_v = self._settle(q0, v0, self.start_u)

    # -- skrl interface ------------------------------------------------------------
    def reset(self):
        """Start a new episode in every world; returns ``(observations, info)``."""
        self._resetWorlds(np.arange(self.num_envs))
        return self._observe(), {}

    def step(self, actions: torch.Tensor):
        """Apply latent actions ``(num_envs, 16)``; returns skrl's five-tuple.

        Raises:
            FloatingPointError: If a world's state became non-finite.
            RuntimeError: If a world outgrew its contact or constraint buffers.
        """
        import warp as wp

        command = self.codec.decode(actions.detach().to(self.device, torch.float32), self._target)
        torch.cuda.synchronize(self.device)
        self.sim.SetControl(wp.from_torch(command.contiguous()))
        self.sim.Step_GPU(self.config.substeps)
        self.sim.CheckCapacity()
        qpos, qvel = self._state()
        bad = ~(torch.isfinite(qpos).all(dim=1) & torch.isfinite(qvel).all(dim=1))
        if bad.any():
            raise FloatingPointError(f"Non-finite state in worlds {torch.nonzero(bad).flatten().tolist()}")
        self._target.copy_(command)
        steps_before = self._steps.clone()
        self._steps += 1

        cost = wp.to_torch(self.task.calcCosts(self.sim, out=self._outs["cost"], goals=self._goals_wp))
        dropped = wp.to_torch(self.task.isFailure(self.sim, out=self._outs["failure"])).clone()
        success = wp.to_torch(self.task.isSuccess(self.sim, out=self._outs["success"], goals=self._goals_wp))
        wp.synchronize_device(str(self.device))
        success = success & ~dropped
        deadline = (self._steps >= self.config.episode_steps) & ~success & ~dropped
        reward = finite_reward(cost, dropped, steps_before, self.config.episode_steps, self.config.cost_reference)
        done = success | dropped | deadline
        info = {"episode": {"successes": success.sum().float(), "drops": dropped.sum().float(),
                            "deadlines": deadline.sum().float()}}

        ended = torch.nonzero(done).flatten()
        if ended.numel():
            self._resetWorlds(ended.cpu().numpy())
        terminated = done[:, None].clone()
        truncated = torch.zeros_like(terminated)
        return self._observe(), reward, terminated, truncated, info

    def state(self):
        return None

    def render(self, *args, **kwargs):
        return None

    def close(self) -> None:
        if self.sim is not None:
            self.sim.Close()
            self.sim = None

    # -- contract ------------------------------------------------------------------
    def describe(self) -> dict:
        """What a policy trained here assumes; exported with the actor."""
        return {"config": self.config.to_dict(), "observation_profile": OBSERVATION_PROFILE,
                "action_profile": ACTION_PROFILE, "reward_profile": REWARD_PROFILE,
                "scene": str(self.task.getModelPath()), "simulator": type(self.sim).__name__}

    # -- internals -----------------------------------------------------------------
    def _state(self):
        """Live ``(qpos, qvel)`` torch views of every world."""
        import warp as wp
        state = self.sim.DeviceState()
        wp.synchronize_device(str(self.device))
        return wp.to_torch(state.qpos), wp.to_torch(state.qvel)

    def _observe(self) -> torch.Tensor:
        qpos, qvel = self._state()
        return self.encoder(qpos, qvel, self._goals[:, 0:3], self._goals[:, 3:7], self._target, self._steps)

    def _settle(self, q0, v0, u0):
        """The start state: the initial grasp held for ``settle_s``, as ``run_episodes`` settles.

        Settled once, here, rather than at every reset: settling a world would
        step all of them, and the result is the same state each time.
        """
        if self.config.settle_steps == 0:
            return np.asarray(q0, dtype=np.float32), np.asarray(v0, dtype=np.float32)
        self.sim.SetState(q0, v0)
        self.sim.SetControl(u0)
        self.sim.Step(self.config.settle_steps)
        q, v = self.sim.GetState()
        if not (np.isfinite(q[0]).all() and np.isfinite(v[0]).all()) or self.task.isFailure(self.sim).numpy()[0]:
            raise RuntimeError(f"The initial grasp does not hold for {self.config.settle_s} s on "
                               f"{self.config.rollout_model}; the object was dropped while settling")
        return q[0].copy(), v[0].copy()

    def _resetWorlds(self, ids: np.ndarray) -> None:
        """Back to the settled start, each with a freshly sampled goal."""
        k = len(ids)
        self.sim.ResetWorlds(ids, np.tile(self.start_q, (k, 1)), np.tile(self.start_v, (k, 1)),
                             np.tile(self.start_u, (k, 1)))
        rows = np.stack([self.task.goalRow(self.task.sampleNewGoal(current_goal=self._goal_start))
                         for _ in range(k)])
        index = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        self._goals[index] = torch.as_tensor(rows, device=self.device)
        self._target[index] = torch.as_tensor(self.start_u, dtype=torch.float32, device=self.device)
        self._steps[index] = 0
        torch.cuda.synchronize(self.device)
