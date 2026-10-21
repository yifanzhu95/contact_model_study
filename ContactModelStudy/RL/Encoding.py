"""Shared training/inference encoding and legacy-compatible named task terms."""
from __future__ import annotations

import torch

from ContactModelStudy.RL.Contracts import Frame


OBSERVATION_PROFILE = "sim_world73_dev_v1"
ACTION_PROFILE = "latent_target_delta_v1"
REWARD_PROFILE = "finite_absorbing_cost_gamma1_v1"


def quaternion_matrix(q):
    """Unit wxyz quaternion to rotation matrix. Rows/columns use world axes."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=-1).reshape(-1, 3, 3)


def rotation6(matrix):
    return matrix[..., :, :2].transpose(-2, -1).reshape(-1, 6)


class ActionCodec:
    def __init__(self, lower, upper, *, rate: float, control_dt: float, device="cpu"):
        self.lower = torch.as_tensor(lower, dtype=torch.float32, device=device).clone()
        self.upper = torch.as_tensor(upper, dtype=torch.float32, device=device).clone()
        if (not torch.isfinite(self.lower).all() or not torch.isfinite(self.upper).all()
                or not torch.all(self.upper > self.lower)):
            raise ValueError("Actuator bounds must be finite with upper > lower")
        self.delta = rate * control_dt

    def decode(self, latent, previous):
        if latent.shape != previous.shape or not torch.isfinite(latent).all():
            raise ValueError("Action must have the expected batch shape and finite values")
        return torch.clamp(previous + self.delta * torch.tanh(latent), self.lower, self.upper)

    def describe(self):
        return {"profile": ACTION_PROFILE, "lower": self.lower.tolist(),
                "upper": self.upper.tolist(), "max_target_change": self.delta}


def task_terms(frame: Frame, goal_pos, goal_quat, home, fallen_z, indices, weights):
    obj_q, obj_v, robot_q, n_hand = (int(x) for x in indices[:4])
    pos = frame.qpos[:, obj_q:obj_q+3]
    quat = frame.qpos[:, obj_q+3:obj_q+7]
    diff = pos - goal_pos
    terms = {
        "quat": 1 - (quat * goal_quat).sum(-1).square(),
        "pos_x": diff[:, 0].abs(), "pos_y": diff[:, 1].abs(), "pos_z": diff[:, 2].abs(),
        "pos": torch.linalg.vector_norm(diff, dim=-1),
        "contact": torch.linalg.vector_norm(pos[:, None, :] - frame.sites[:, indices[5:9]], dim=-1).sum(-1),
        "joint": (frame.qpos[:, robot_q:robot_q+n_hand] - home).square().sum(-1),
        "joint_velo": frame.qvel[:, robot_q:robot_q+n_hand].square().sum(-1),
        "fallen": (pos[:, 2] < fallen_z).float(),
        "object_speed_legacy": torch.linalg.vector_norm(frame.qvel[:, obj_v:obj_v+6], dim=-1),
        "physical_drop": pos[:, 2] < 0,
    }
    # Preserve w_velo -> position norm, and the original absence of effort cost.
    names = ("quat", "pos_x", "pos_y", "pos_z", "pos", "contact", "joint", "joint_velo", "fallen")
    terms["raw_cost"] = sum(weights[i] * terms[key] for i, key in enumerate(names))
    terms["success"] = (terms["pos"] < 0.02) & (terms["quat"] < 0.04) & (terms["object_speed_legacy"] < 0.1)
    return terms


def encode_observation(frame, goal_pos, goal_quat, previous, steps, config,
                       joint_q_ids, joint_v_ids, joint_ranges, codec, indices):
    obj_q = int(indices[0])
    pos = frame.qpos[:, obj_q:obj_q+3]
    rot = quaternion_matrix(frame.qpos[:, obj_q+3:obj_q+7])
    goal_rot = quaternion_matrix(goal_quat)
    q_low, q_high = joint_ranges[:, 0], joint_ranges[:, 1]
    joint_position = 2 * (frame.qpos[:, joint_q_ids] - q_low) / (q_high - q_low) - 1
    target = 2 * (previous - codec.lower) / (codec.upper - codec.lower) - 1
    return torch.cat((joint_position, frame.qvel[:, joint_v_ids] / 20,
                      pos * 10, rotation6(rot), (goal_pos - pos) * 10,
                      rotation6(rot.transpose(-2, -1) @ goal_rot),
                      frame.object_twist[:, :3], frame.object_twist[:, 3:] / 10,
                      target, (1 - steps.float() / config.episode_steps)[:, None]), dim=-1).clone()


def finite_reward(raw_cost, dropped, steps_before, config):
    """Undiscounted finite task, gamma=1, with failed absorbing tail summed once."""
    cost = (raw_cost / config.cost_reference).clamp(0, 1)
    remaining = config.episode_steps - steps_before
    reward = torch.where(dropped, -remaining.float() / config.episode_steps,
                         -cost / config.episode_steps)
    return reward[:, None]
