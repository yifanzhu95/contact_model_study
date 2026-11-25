"""CPU policy inference and faithful training-interface conversion. No robot I/O.

Inputs use actuator order from interface.json. All world-frame vectors refer to
the TRAINING scene world, after the receiver's frame/motor calibration.
"""
from pathlib import Path
import json
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent


class FrozenNormalization(nn.Module):
    def __init__(self, state):
        super().__init__()
        for k in ("_mean", "_var", "_std", "count"):
            self.register_buffer(k, state["obs_normalizer." + k].clone())

    def forward(self, observation):
        # This is the trained export's exact convention, NOT sqrt(var + eps).
        return (observation - self._mean) / (self._std + 0.01)


class InferenceMLP(nn.Module):
    """Plain PyTorch fallback containing only frozen actor/normalizer tensors."""
    def __init__(self, state):
        super().__init__()
        self.obs_normalizer = FrozenNormalization(state)
        self.mlp = nn.Sequential(nn.Linear(73,512), nn.ELU(), nn.Linear(512,512),
                                 nn.ELU(), nn.Linear(512,256), nn.ELU(), nn.Linear(256,16))
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.eval()

    def forward(self, observation):
        return self.mlp(self.obs_normalizer(observation))


def _batch(value, width, name):
    x = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[1] != width or not torch.isfinite(x).all():
        raise ValueError(f"{name}: expected finite [batch,{width}] or [{width}]")
    return x


def quaternion_matrix_wxyz(q):
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    if (norm < 1e-8).any():
        raise ValueError("Quaternion norm is zero; expected w,x,y,z")
    q = q / norm
    w,x,y,z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)),dim=-1).reshape(-1,3,3)


def rotation6(matrix):
    # COLUMN order: R00,R10,R20,R01,R11,R21 (not the first six row-major entries).
    return matrix[:,:,:2].transpose(-2,-1).reshape(-1,6)


def encode_state(state, interface=None):
    """Physical state -> raw73D observation. Do NOT pre-normalize for the actor.

    object_angular_velocity_world_rad_s is in WORLD axes; raw MuJoCo angular
    qvel would first need R_object @ omega_body. Linear velocity is NOT /10.
    elapsed_steps is controls already applied:0 before the first decision.
    """
    if interface is None:
        interface = json.loads((ROOT/"interface.json").read_text())
    fields={
        "joint_position_rad":16, "joint_velocity_rad_s":16,
        "object_position_world_m":3, "object_quaternion_wxyz":4,
        "object_linear_velocity_world_m_s":3, "object_angular_velocity_world_rad_s":3,
        "goal_position_world_m":3, "goal_quaternion_wxyz":4,
        "previous_target_rad":16,
    }
    x={k:_batch(state[k],n,k) for k,n in fields.items()}
    batch=x["joint_position_rad"].shape[0]
    if any(v.shape[0]!=batch for v in x.values()):
        raise ValueError("All state fields must have the same batch size")
    steps=torch.as_tensor(state["elapsed_steps"],dtype=torch.float32).reshape(-1)
    if steps.numel()==1:
        steps=steps.expand(batch)
    if (steps.numel()!=batch or not torch.isfinite(steps).all() or
        not torch.equal(steps,steps.round()) or (steps<0).any() or (steps>=500).any()):
        raise ValueError("elapsed_steps must be integer0..499;500 ends this task")
    joints=interface["joints"]
    qlow=torch.tensor([j["joint_position_range_rad"][0] for j in joints],dtype=torch.float32)
    qhigh=torch.tensor([j["joint_position_range_rad"][1] for j in joints],dtype=torch.float32)
    ulow=torch.tensor([j["actuator_target_range_rad"][0] for j in joints],dtype=torch.float32)
    uhigh=torch.tensor([j["actuator_target_range_rad"][1] for j in joints],dtype=torch.float32)
    q=(2*(x["joint_position_rad"]-qlow)/(qhigh-qlow)-1)
    previous=2*(x["previous_target_rad"]-ulow)/(uhigh-ulow)-1
    pos=x["object_position_world_m"]
    rot=quaternion_matrix_wxyz(x["object_quaternion_wxyz"])
    grot=quaternion_matrix_wxyz(x["goal_quaternion_wxyz"])
    return torch.cat((q,x["joint_velocity_rad_s"]/20,pos*10,rotation6(rot),
                      (x["goal_position_world_m"]-pos)*10,
                      rotation6(rot.transpose(-2,-1)@grot),
                      x["object_linear_velocity_world_m_s"],
                      x["object_angular_velocity_world_rad_s"]/10,
                      previous,(1-steps/500)[:,None]),dim=-1)


def decode_action(raw_action, previous_target_rad, interface=None):
    """Raw16 -> absolute joint-position target, radians, in TRAINING joint order.

    Uses PREVIOUS ACTUATOR TARGET, not measured current joint position. The
    trained policy applies tanh exactly once. No0.15rad lead limiter was trained.
    """
    if interface is None:
        interface=json.loads((ROOT/"interface.json").read_text())
    raw=_batch(raw_action,16,"raw_action")
    previous=_batch(previous_target_rad,16,"previous_target_rad")
    if raw.shape!=previous.shape:
        raise ValueError("Raw action and previous target batch shapes differ")
    lower=torch.tensor([j["actuator_target_range_rad"][0] for j in interface["joints"]],dtype=torch.float32)
    upper=torch.tensor([j["actuator_target_range_rad"][1] for j in interface["joints"]],dtype=torch.float32)
    proposed=previous+0.064*torch.tanh(raw)
    return torch.maximum(torch.minimum(proposed,upper),lower)


class Policy:
    def __init__(self, directory=ROOT, backend="torchscript"):
        self.directory=Path(directory)
        self.interface=json.loads((self.directory/"interface.json").read_text())
        if backend=="torchscript":
            self.actor=torch.jit.load(str(self.directory/"policy_noDR_s1009_u0800.ts.pt"),map_location="cpu").eval()
        elif backend=="pytorch":
            state=torch.load(self.directory/"actor_inference_weights.pt",map_location="cpu",weights_only=True)
            self.actor=InferenceMLP(state)
        else:
            raise ValueError("backend must be torchscript or pytorch")
        self.backend=backend

    @torch.inference_mode()
    def raw_action(self, observation):
        observation=_batch(observation,73,"observation")
        # Frozen normalizer is ALREADY inside both supported actor backends.
        result=self.actor(observation)
        if result.shape!=(observation.shape[0],16) or not torch.isfinite(result).all():
            raise ValueError("Actor returned invalid raw actions")
        return result

    @torch.inference_mode()
    def infer_state(self, state):
        obs=encode_state(state,self.interface)
        raw=self.raw_action(obs)
        target=decode_action(raw,state["previous_target_rad"],self.interface)
        # No state mutation, motor commands, sockets, ROS, simulation or reset.
        return {"observation":obs,"raw_action":raw,"joint_target_rad":target}
