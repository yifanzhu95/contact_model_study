"""Tensor task core: separate goals, indexed resets, and owned transitions."""
from __future__ import annotations

import hashlib
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ContactModelStudy.RL.Backends import make_backend
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Randomization import FixedDomainLifecycle
from ContactModelStudy.RL.SceneIdentity import scene_model_fingerprint
from ContactModelStudy.RL.Contracts import EnvConfig, Frame, NumericalTransitionError, Transition
from ContactModelStudy.RL.Encoding import (ActionCodec, OBSERVATION_PROFILE, REWARD_PROFILE,
                                          encode_observation, finite_reward, task_terms)


class LeapVectorEnv:
    """Not a Gymnasium VectorEnv: GPU-facing tensor and reset semantics are explicit.

    CPU diagnostics, pinned MJWarp, and Kamino are explicit choices.
    No fallback is performed. The single-world Gym adapter below
    is the public Gymnasium-compatible entry point.
    """
    def __init__(self, config: EnvConfig, case_bank: CaseBank):
        from ContactModelStudy.Tasks.Registry import task_from_environment
        case_bank.require_task(config.task_name)
        self.config, self.case_bank = config, case_bank
        self.domain_lifecycle = FixedDomainLifecycle(config)
        self.domain_lifecycle.before_backend_create()
        self.reset_domains = {}
        self.num_envs, self.device = config.num_envs, torch.device(config.device)
        self.task = task_from_environment(config)
        self._scene_model_fingerprint = scene_model_fingerprint(self.task.getModelPath())
        self.backend = make_backend(self.task, config)
        model = self.backend.model
        if (model.nq, model.nv, model.nu) != (23, 22, 16):
            raise ValueError("The current profile requires the original LEAP reorientation layout")
        if not np.all(model.actuator_ctrllimited):
            raise ValueError("Every actuator needs an explicit control range")
        joint_ids = model.actuator_trnid[:, 0]
        if not np.all(model.jnt_limited[joint_ids]):
            raise ValueError("The current observation profile needs bounded hand joints")
        self._joint_q_ids = model.jnt_qposadr[joint_ids].copy()
        self._joint_v_ids = model.jnt_dofadr[joint_ids].copy()
        self._joint_ranges = torch.tensor(model.jnt_range[joint_ids], dtype=torch.float32, device=self.device)
        self.codec = ActionCodec(model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1],
                                 rate=config.target_rate, control_dt=config.control_dt, device=self.device)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(73,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)
        self._goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._goal_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._target = torch.zeros((self.num_envs, 16), device=self.device)
        self.steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._needs_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._evaluation_goal_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.case_ids = [""] * self.num_envs
        self._rng = [np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(self.num_envs)]
        self._faulted = False
        self._closed = False
        self._weights = torch.tensor(self.task.weights, device=self.device)
        if not torch.isfinite(self._weights).all() or (self._weights < 0).any():
            raise ValueError("Finite nonnegative weights are required by the reward contract")
        self._home = torch.tensor(self.task.params["init_ctrl"], dtype=torch.float32, device=self.device)

    @property
    def current_target(self):
        return self._target.clone()

    def read_frame(self):
        """Canonical owned state for external controllers; includes actual M5 bodies."""
        self._check_open()
        try:
            frame = self.backend.read_frame()
            self._validate_frame(frame)
        except (Exception, KeyboardInterrupt):
            self._faulted = True
            raise
        return frame

    def _check_open(self):
        if self._closed or self._faulted:
            raise RuntimeError("Environment is closed or faulted; recreate it after diagnostics")

    def _validate_frame(self, frame: Frame):
        arrays = (frame.qpos, frame.qvel, frame.sites, frame.object_twist)
        valid = all(torch.isfinite(x).all().item() for x in arrays)
        obj_q = int(self.task.indices[0])
        norm = torch.linalg.vector_norm(frame.qpos[:, obj_q+3:obj_q+7], dim=-1)
        if not valid or not torch.allclose(norm, torch.ones_like(norm), atol=1e-4, rtol=0):
            self._faulted = True
            raise NumericalTransitionError("Nonfinite state or invalid object quaternion", frame)

    def reset_all(self, case_indices=None):
        return self.reset_indices(range(self.num_envs), case_indices)

    def reset_indices(self, ids, case_indices=None):
        self._check_open()
        if isinstance(ids, torch.Tensor):
            if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("Reset IDs must be a one-dimensional integer tensor")
            ids = ids.tolist()
        ids = list(ids)
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in ids):
            raise ValueError("Reset IDs must be integers")
        ids = [int(i) for i in ids]
        if len(ids) != len(set(ids)) or any(i < 0 or i >= self.num_envs for i in ids):
            raise ValueError("Reset IDs must be unique and in range")
        if not ids:
            return None
        if case_indices is None:
            case_indices = [int(self._rng[i].integers(len(self.case_bank.cases))) for i in ids]
        case_indices = list(case_indices)
        if (len(case_indices) != len(ids)
                or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                       or i < 0 or i >= len(self.case_bank.cases) for i in case_indices)):
            raise ValueError("One valid case index is required per reset world")
        cases = [self.case_bank.cases[i] for i in case_indices]
        controls = torch.tensor([c.control for c in cases], dtype=torch.float32, device=self.device)
        if ((controls < self.codec.lower) | (controls > self.codec.upper)).any():
            raise ValueError("Case initial controls exceed actuator limits")
        domains = self.domain_lifecycle.before_reset(ids, cases)
        try:
            self.backend.reset_worlds(ids, cases)
        except (Exception, KeyboardInterrupt):
            self._faulted = True
            raise
        self.reset_domains.update({record["world_id"]: record for record in domains})
        self._goal_pos[ids] = torch.tensor([c.goal_position for c in cases], dtype=torch.float32, device=self.device)
        self._goal_quat[ids] = torch.tensor([c.goal_quaternion for c in cases], dtype=torch.float32, device=self.device)
        self._target[ids] = controls
        self.steps[ids] = 0
        self._needs_reset[ids] = False
        self._evaluation_goal_ready[ids] = False
        for i, c in zip(ids, cases, strict=True):
            self.case_ids[i] = c.case_id
        frame = self.read_frame()
        return self._encode(frame) if not self._needs_reset.any() else None

    def _encode(self, frame):
        return encode_observation(frame, self._goal_pos, self._goal_quat, self._target,
                                  self.steps, self.config, self._joint_q_ids, self._joint_v_ids,
                                  self._joint_ranges, self.codec, self.task.indices)

    def observe(self):
        self._check_open()
        if self._needs_reset.any():
            raise RuntimeError("Reset all ended/uninitialized worlds before observing a new action state")
        frame = self.read_frame()
        return self._encode(frame)

    def advance_evaluation_goal(self, case):
        """Continue a successful single-world evaluation without resetting physics.

        Only a success with time left permits this explicit operation. The PPO
        collector never calls it; ordinary terminal/reset semantics stay intact.
        Targets and remaining time retain their original encoding and bounds.
        """
        from ContactModelStudy.RL.CaseBank import EpisodeCase
        self._check_open()
        if (self.num_envs != 1 or not bool(self._evaluation_goal_ready[0])
                or not bool(self._needs_reset[0]) or int(self.steps[0]) >= self.config.episode_steps):
            raise RuntimeError('A new evaluation goal requires a successful single world with time remaining')
        if not isinstance(case, EpisodeCase) or case.case_id != self.case_ids[0]:
            raise ValueError('A continued goal must belong to the current case')
        self._goal_pos[0] = torch.tensor(case.goal_position, device=self.device)
        self._goal_quat[0] = torch.tensor(case.goal_quaternion, device=self.device)
        self._needs_reset[0] = False
        self._evaluation_goal_ready[0] = False
        return self.observe()

    def step(self, raw_action):
        raw_action = torch.as_tensor(raw_action, dtype=torch.float32, device=self.device)
        if tuple(raw_action.shape) != (self.num_envs, 16):
            raise ValueError("Expected (num_envs, 16) latent actions")
        return self.step_command(self.codec.decode(raw_action, self._target))

    def step_command(self, command):
        self._check_open()
        if self._needs_reset.any():
            raise RuntimeError("An ended/uninitialized world must be reset before the next step")
        command = torch.as_tensor(command, dtype=torch.float32, device=self.device).detach().clone()
        if (tuple(command.shape) != (self.num_envs, 16) or not torch.isfinite(command).all()
                or ((command < self.codec.lower) | (command > self.codec.upper)).any()
                or (torch.abs(command - self._target) > self.codec.delta + 1e-6).any()):
            raise ValueError("Command violates shape, finite, actuator or target-rate contract")
        before = self.steps.clone()
        try:
            self.backend.step(command, self.config.physics_steps)
        except (Exception, KeyboardInterrupt):
            self._faulted = True
            raise
        self._target.copy_(command)
        self.steps += 1
        frame = self.read_frame()
        terms = task_terms(frame, self._goal_pos, self._goal_quat, self._home,
                           self.task.params["fallen_z"], self.task.indices, self._weights)
        if not all(torch.isfinite(x).all().item() for x in terms.values()):
            self._faulted = True
            raise NumericalTransitionError("Nonfinite task terms", frame)
        drop = terms["physical_drop"]
        success = terms["success"] & ~drop
        deadline = (self.steps >= self.config.episode_steps) & ~drop & ~success
        done = drop | success | deadline
        reasons = tuple("physical_drop" if drop[i] else "success" if success[i] else
                        "deadline" if deadline[i] else "running" for i in range(self.num_envs))
        reward = finite_reward(terms["raw_cost"], drop, before, self.config)
        self._needs_reset.copy_(done)
        self._evaluation_goal_ready.copy_(success & (self.steps < self.config.episode_steps))
        return Transition(self._encode(frame), reward.clone(), done[:, None].clone(),
                          torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device), reasons,
                          {k: v.clone() for k, v in terms.items()})

    def describe(self):
        model = self.backend.model
        result = {"schema": "leap_rl_environment_v1", "config": self.config.to_dict(),
                "physics": self.backend.describe(), "case_bank_sha256": self.case_bank.sha256,
                "scene_path": str(Path(self.task.getModelPath()).resolve()),
                "scene_xml_sha256": hashlib.sha256(Path(self.task.getModelPath()).read_bytes()).hexdigest(),
                "scene_model_fingerprint": dict(self._scene_model_fingerprint),
                "observation_profile": OBSERVATION_PROFILE, "observation_size": 73,
                "observation_frame": "world", "quaternion_order": "wxyz",
                "observation_scales": {"joint_velocity": 20.0, "position_multiplier": 10.0,
                                       "linear_velocity": 1.0, "angular_velocity": 10.0},
                "joint_ranges": self._joint_ranges.tolist(),
                "actuator_names": [model.actuator(i).name for i in range(model.nu)],
                "joint_qpos_indices": self._joint_q_ids.tolist(),
                "joint_qvel_indices": self._joint_v_ids.tolist(),
                "action": self.codec.describe(), "reward_profile": REWARD_PROFILE,
                "cost_weights": self.task.params["cost_weights"],
                "task_protocol": "single_goal_finite_deadline_gamma1_v1",
                "autoreset": False, "purpose": "interface_diagnostic_not_learning_evidence"}
        # Preserve existing actor/checkpoint metadata exactly when not requested.
        if self.config.randomization is not None:
            result["randomization"] = self.domain_lifecycle.describe()
        return result

    def close(self):
        if not self._closed:
            self.backend.close()
            self._closed = True


class LeapGymEnv(gym.Env):
    """Single-world, NumPy Gymnasium adapter; the tensor core is separate."""
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig, case_bank: CaseBank):
        if config.num_envs != 1:
            raise ValueError("LeapGymEnv is single-world; use LeapVectorEnv for batches")
        self.core = LeapVectorEnv(config, case_bank)
        self.observation_space, self.action_space = self.core.observation_space, self.core.action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = (options or {}).get("case_index")
        if index is None:
            index = int(self.np_random.integers(len(self.core.case_bank.cases)))
        obs = self.core.reset_all([index])
        return obs[0].cpu().numpy().copy(), {"case_id": self.core.case_ids[0]}

    def step(self, action):
        tr = self.core.step(torch.as_tensor(action, dtype=torch.float32)[None, :])
        return (tr.next_obs[0].cpu().numpy().copy(), float(tr.reward[0, 0]),
                bool(tr.terminated[0, 0]), bool(tr.truncated[0, 0]),
                {"reason": tr.reasons[0], "raw_cost": float(tr.diagnostics["raw_cost"][0])})

    def close(self):
        self.core.close()
