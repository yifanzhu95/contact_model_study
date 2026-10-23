"""What a policy sees and how its output becomes a command, shared by training and inference.

``TrainEnv`` and ``RLPolicyPlanner`` both encode observations and decode
actions through this module, from the simulator's own ``qpos``/``qvel``, so a
policy is run exactly as it was trained. The cost and the success test are not
here: they are the task's own (``calcCosts`` / ``isSuccess`` / ``isFailure``).
"""

from __future__ import annotations

import numpy as np
import torch

OBSERVATION_PROFILE = "sim_world73_v1"
ACTION_PROFILE = "latent_target_delta_v1"
REWARD_PROFILE = "task_cost_finite_gamma1_v1"


def quaternion_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit wxyz quaternions ``(n, 4)`` to rotation matrices ``(n, 3, 3)``."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
                       dim=-1).reshape(-1, 3, 3)


def rotation6(matrix: torch.Tensor) -> torch.Tensor:
    """The first two columns of each rotation matrix, ``(n, 6)``: a continuous orientation code."""
    return matrix[..., :, :2].transpose(-2, -1).reshape(-1, 6)


def _requireLeapLayout(mjm) -> np.ndarray:
    """The actuated joint of each actuator, after checking the scene is the LEAP layout this encodes."""
    if (mjm.nq, mjm.nv, mjm.nu) != (23, 22, 16):
        raise ValueError(f"Expected the LEAP reorientation layout (nq, nv, nu) = (23, 22, 16), "
                         f"got {(mjm.nq, mjm.nv, mjm.nu)}")
    if not np.all(mjm.actuator_ctrllimited):
        raise ValueError("Every actuator needs an explicit control range")
    joints = mjm.actuator_trnid[:, 0]
    if not np.all(mjm.jnt_limited[joints]):
        raise ValueError("Every actuated joint needs a range")
    return joints


class ActionCodec:
    """Latent policy output to an absolute actuator target, rate-limited and inside the actuator range.

    ``target = clamp(previous + target_rate * control_dt * tanh(latent), lower, upper)``.
    """

    def __init__(self, lower, upper, *, rate: float, control_dt: float, device="cpu"):
        self.lower = torch.as_tensor(lower, dtype=torch.float32, device=device).clone()
        self.upper = torch.as_tensor(upper, dtype=torch.float32, device=device).clone()
        if (not torch.isfinite(self.lower).all() or not torch.isfinite(self.upper).all()
                or not torch.all(self.upper > self.lower)):
            raise ValueError("Actuator bounds must be finite with upper > lower")
        self.delta = rate * control_dt

    @classmethod
    def fromModel(cls, mjm, *, rate: float, control_dt: float, device="cpu") -> "ActionCodec":
        _requireLeapLayout(mjm)
        return cls(mjm.actuator_ctrlrange[:, 0], mjm.actuator_ctrlrange[:, 1],
                   rate=rate, control_dt=control_dt, device=device)

    def decode(self, latent: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        if latent.shape != previous.shape or not torch.isfinite(latent).all():
            raise ValueError("Action must have the expected batch shape and finite values")
        return torch.clamp(previous + self.delta * torch.tanh(latent), self.lower, self.upper)


class ObservationEncoder:
    """The 73 values a policy sees, from ``qpos``/``qvel`` and the goal.

    In order: 16 hand joint positions scaled to [-1, 1]; 16 hand joint
    velocities / 20; object position * 10; object orientation (6D); goal
    position minus object position, * 10; goal orientation in the object frame
    (6D); object linear velocity and angular velocity / 10, both in world axes;
    the previous actuator target scaled to [-1, 1]; and the fraction of the
    episode left. Simulated state, not sensor estimates.
    """

    SIZE = 73

    def __init__(self, mjm, indices, codec: ActionCodec, episode_steps: int, device="cpu"):
        joints = _requireLeapLayout(mjm)
        self.codec, self.episode_steps = codec, int(episode_steps)
        self.obj_q, self.obj_v = int(indices[0]), int(indices[1])
        self.q_ids = torch.as_tensor(mjm.jnt_qposadr[joints], dtype=torch.long, device=device)
        self.v_ids = torch.as_tensor(mjm.jnt_dofadr[joints], dtype=torch.long, device=device)
        ranges = torch.as_tensor(mjm.jnt_range[joints], dtype=torch.float32, device=device)
        self.q_low, self.q_high = ranges[:, 0], ranges[:, 1]

    def __call__(self, qpos, qvel, goal_pos, goal_quat, previous, steps) -> torch.Tensor:
        """``(n, 73)`` observations; every argument has a leading world axis, ``steps`` is ``(n,)``."""
        oq, ov = self.obj_q, self.obj_v
        pos = qpos[:, oq:oq + 3]
        rot = quaternion_matrix(qpos[:, oq + 3:oq + 7])
        joint = 2 * (qpos[:, self.q_ids] - self.q_low) / (self.q_high - self.q_low) - 1
        target = 2 * (previous - self.codec.lower) / (self.codec.upper - self.codec.lower) - 1
        # MuJoCo's free-joint velocity is linear in world axes, angular in body
        # axes; both go to the policy in world axes.
        linear = qvel[:, ov:ov + 3]
        angular = (rot @ qvel[:, ov + 3:ov + 6, None]).squeeze(-1)
        return torch.cat((joint, qvel[:, self.v_ids] / 20,
                          pos * 10, rotation6(rot), (goal_pos - pos) * 10,
                          rotation6(rot.transpose(-2, -1) @ quaternion_matrix(goal_quat)),
                          linear, angular / 10,
                          target, (1 - steps.float() / self.episode_steps)[:, None]), dim=-1)


def finite_reward(cost, dropped, steps_before, episode_steps: int, cost_reference: float) -> torch.Tensor:
    """Per-step reward of the finite, undiscounted task, ``(n, 1)``.

    The task's running cost, scaled by ``cost_reference`` and clamped to
    [0, 1], spread over the episode. A drop ends the episode and takes the
    worst penalty for every step it cut off, so dropping never pays.
    """
    scaled = (cost / cost_reference).clamp(0, 1)
    remaining = (episode_steps - steps_before).float()
    reward = torch.where(dropped, -remaining / episode_steps, -scaled / episode_steps)
    return reward[:, None]
