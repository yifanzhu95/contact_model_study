"""Real CPU physics for interface diagnostics; GPU qualification remains separate."""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from ContactModelStudy.RL.Contracts import EnvConfig, Frame, NumericalTransitionError


class CpuMujocoBatch:
    """Independent MjData objects; resetting one cannot reset another's solver state.

    Uses the existing CPU simulator to compile the scene and apply the M1/M2
    contact preset. CPU float64 MuJoCo is NOT a certified MJWarp substitute.
    """

    def __init__(self, task, config: EnvConfig):
        from ContactModelStudy.Simulators.Mujoco import Mujoco, MujocoConfig
        from ContactModelStudy.Utils.ContactModelPresets import CONTACT_MODELS, M1_SOLREF_TIMECONST_MULT
        params = dict(CONTACT_MODELS[config.contact_preset]["params"])
        if config.contact_preset == "M1":
            params["solref_timeconst"] = M1_SOLREF_TIMECONST_MULT * config.physics_dt
        self._template = Mujoco(task.getModelPath(), MujocoConfig(timestep=config.physics_dt, **params))
        self.model = self._template.mjm
        self.num_envs = config.num_envs
        self.data = [mujoco.MjData(self.model) for _ in range(self.num_envs)]
        self._readout = mujoco.MjData(self.model)
        self._object_id = int(task.indices[4])
        self._closed = False
        self._preset = config.contact_preset

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Physics backend is closed")

    def reset_worlds(self, ids, cases):
        self._check_open()
        for i, case in zip(ids, cases, strict=True):
            data = self.data[i]
            mujoco.mj_resetData(self.model, data)
            data.qpos[:] = case.qpos
            data.qvel[:] = case.qvel
            data.ctrl[:] = case.control
            mujoco.mj_forward(self.model, data)

    def step(self, target: torch.Tensor, steps: int):
        self._check_open()
        commands = target.detach().cpu().numpy()
        for i, data in enumerate(self.data):
            warnings_before = data.warning.number.copy()
            data.ctrl[:] = commands[i]
            mujoco.mj_step(self.model, data, nstep=steps)
            changed = np.flatnonzero(data.warning.number > warnings_before)
            if changed.size:
                raise NumericalTransitionError(
                    f"MuJoCo warning in world {i}, warning indices {changed.tolist()}; batch aborted",
                    self.read_frame())

    def read_frame(self):
        self._check_open()
        q, v, sites, twists = [], [], [], []
        for data in self.data:
            # Full forward on separate scratch data cannot alter live warm starts.
            mujoco.mj_copyData(self._readout, self.model, data)
            mujoco.mj_forward(self.model, self._readout)
            twist = np.empty(6)
            mujoco.mj_objectVelocity(self.model, self._readout, mujoco.mjtObj.mjOBJ_XBODY,
                                    self._object_id, twist, 0)
            q.append(data.qpos.copy())
            v.append(data.qvel.copy())
            sites.append(self._readout.site_xpos.copy())
            twists.append(np.concatenate((twist[3:], twist[:3])))
        def tensor(items):
            return torch.tensor(np.stack(items), dtype=torch.float32)
        return Frame(tensor(q), tensor(v), tensor(sites), tensor(twists))

    def describe(self):
        return {"backend": "mujoco_cpu_diagnostic", "contact_preset": self._preset,
                "physics_precision": "float64", "frame_precision": "float32",
                "indexed_reset": True, "post_step_readout": "separate_scratch_forward",
                "gpu_qualified": False, "num_envs": self.num_envs,
                "reset_protocol": "raw_initial_state_without_settling"}

    def close(self):
        self.data.clear()
        self._readout = None
        self._template = None
        self._closed = True


def make_backend(task, config):
    if config.physics_backend in ("mujoco_eval", "pinocchio_eval", "drake_eval"):
        from ContactModelStudy.RL.EvalBackend import MpcEvalBatch
        return MpcEvalBatch(task, config)
    if config.physics_backend == "mujoco_cpu_diagnostic":
        return CpuMujocoBatch(task, config)
    if config.physics_backend == "mjwarp":
        from ContactModelStudy.RL.MjwarpBackend import MjwarpBatch
        return MjwarpBatch(task, config)
    if config.physics_backend == "kamino":
        from ContactModelStudy.RL.KaminoBackend import KaminoBatch
        return KaminoBatch(task, config)
    if config.physics_backend in ("comfree", "xpbd"):
        from ContactModelStudy.RL.AlternativeBackends import ComFreeBatch, XpbdBatch
        cls = ComFreeBatch if config.physics_backend == "comfree" else XpbdBatch
        return cls(task, config)
    raise ValueError(f"Unsupported physics backend: {config.physics_backend}")
