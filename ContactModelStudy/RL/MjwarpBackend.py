"""Adapter around the project's pinned MJWarp, without replacing its integrator."""
from __future__ import annotations

from importlib.metadata import distribution, version
import json

import mujoco
import numpy as np
import torch
import warp as wp

from ContactModelStudy.RL.Contracts import Frame, NumericalTransitionError, MJWARP_NATIVE_PROFILE
from ContactModelStudy.RL.Encoding import quaternion_matrix


COMFREE_COMMIT = "98b4015dcd96961b80d6d78565b002b83178263e"


class MjwarpBatch:
    """Single-device M1/M2 integration, with explicit synchronization for correctness.

    No state arrays cross through NumPy on the step/readout path. Scalar safety
    checks and explicit stream barriers remain; throughput is not certified.
    """
    BACKEND_NAME = "mjwarp"
    ALLOWED_PRESETS = ("M1", "M2")

    def __init__(self, task, config):
        if config.physics_backend != self.BACKEND_NAME or config.contact_preset not in self.ALLOWED_PRESETS:
            raise ValueError(f"{type(self).__name__} requires {self.BACKEND_NAME}/{self.ALLOWED_PRESETS}")
        if not torch.cuda.is_available():
            raise RuntimeError("MJWarp requires working CUDA Torch; CPU fallback is disabled")
        if version("mujoco") != "3.6.0" or version("warp-lang") != "1.17.0":
            raise RuntimeError("This adapter requires MuJoCo 3.6.0 and Warp 1.17.0")
        origin = json.loads(distribution("comfree_warp").read_text("direct_url.json") or "{}")
        if origin.get("vcs_info", {}).get("commit_id") != COMFREE_COMMIT:
            raise RuntimeError("Install the exact project-pinned comfree_warp commit")
        self.num_envs, self.device = config.num_envs, torch.device(config.device)
        self._preset, self._closed = config.contact_preset, False
        self._requested_numerics = config.numerics_profile
        self._row_orderer = None
        self._object_q, self._object_v = map(int, task.indices[:2])
        self._object_body = int(task.indices[4])
        with wp.ScopedDevice(str(self.device)):
            self.sim = self._make_simulator(task, config)
            self.sim.ClearControlSequence()
            # Kinematics runs on scratch data, never on the live solver's buffers.
            self._scratch = self.sim._makeData(nconmax=self.sim.config.nconmax, njmax=self.sim.config.njmax)
        self.model = self.sim.mjm
        self._step_physics = self._select_step()
        body = self._object_body
        joint = int(self.model.body_jntadr[body])
        if (self.model.body_parentid[body] != 0 or self.model.body_jntnum[body] != 1
                or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
                or self.model.jnt_qposadr[joint] != self._object_q
                or self.model.jnt_dofadr[joint] != self._object_v
                or not np.allclose(self.model.jnt_pos[joint], 0)):
            raise ValueError("World twist readout currently requires the original world-root free object")
        wp.synchronize_device(str(self.device))

    def _make_simulator(self, task, config):
        from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
        return GetContactModelSim(config.contact_preset, task.getModelPath(), N=self.num_envs,
                                  timestep=config.physics_dt, device=str(self.device))

    def _native_step(self, model, data):
        # Use the same physics entry point as the MPC simulator's Step_GPU.
        self.sim._stepPhysics()

    def _select_step(self):
        if self._requested_numerics in (None, MJWARP_NATIVE_PROFILE):
            self._numerics_profile = MJWARP_NATIVE_PROFILE
            return self._native_step
        # Retained only for explicitly selected historical numerical variants.
        from ContactModelStudy.RL.MjwarpNumerics import make_ordered_step, NUMERICS_PROFILE
        from ContactModelStudy.RL.GpuConstraintRows import make_ordered_rows_step, NUMERICS_PROFILE as ROWS_PROFILE
        if self._requested_numerics == ROWS_PROFILE:
            self._numerics_profile = ROWS_PROFILE
            step, self._row_orderer = make_ordered_rows_step(self.sim.m, self.sim.d)
            return step
        self._numerics_profile = NUMERICS_PROFILE
        return make_ordered_step(self.sim.m)

    def _reset_solver_data(self, mask):
        self.sim._mjw.reset_data(self.sim.m, self.sim.d, mask)

    def _reset_extra_state(self, ids):
        """Subclasses clear their own per-world scratch without touching other worlds."""

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Physics backend is closed")

    def _torch_barrier(self):
        # Torch and Warp can use different streams. Explicit barriers keep their
        # shared views ordered; replacing them with events is a later optimization.
        torch.cuda.synchronize(self.device)

    def reset_worlds(self, ids, cases):
        self._check_open()
        if not ids:
            return
        if len(ids) != len(cases) or len(set(ids)) != len(ids) or any(i < 0 or i >= self.num_envs for i in ids):
            raise ValueError("One case per unique in-range world is required")
        self._torch_barrier()
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        self._torch_barrier()
        with wp.ScopedDevice(str(self.device)):
            self._reset_solver_data(wp.from_torch(mask))
        wp.synchronize_device(str(self.device))
        for name, values in (("qpos", [c.qpos for c in cases]),
                             ("qvel", [c.qvel for c in cases]),
                             ("ctrl", [c.control for c in cases])):
            wp.to_torch(getattr(self.sim.d, name))[ids] = torch.tensor(
                values, dtype=torch.float32, device=self.device)
        self._reset_extra_state(ids)
        self._torch_barrier()
        # The vendored reset also invalidates packed contact metadata. Contacts
        # are rebuilt by the next step. Do not call full forward on live worlds:
        # that would overwrite unselected warm starts and other solver caches.

    def step(self, target, steps):
        from ContactModelStudy.RL.GpuConstraintRows import ConstraintOrderingError
        self._check_open()
        if self.sim._has_control_sequence:
            raise RuntimeError("RL owns held targets; an external control sequence is not supported")
        command = target.detach().contiguous()
        self._torch_barrier()
        with wp.ScopedDevice(str(self.device)):
            self.sim.SetControl(wp.from_torch(command))
            for _ in range(steps):
                # Each backend selects its own original solver pipeline.
                try:
                    self._step_physics(self.sim.m, self.sim.d)
                except ConstraintOrderingError as exc:
                    raise NumericalTransitionError(str(exc), self.read_frame()) from exc
                wp.synchronize_device(str(self.device))
                # Counts continue beyond allocation in this fixed engine. Fail
                # closed instead of letting dropped constraints enter learning.
                contacts = int(wp.to_torch(self.sim.d.nacon)[0].item())
                rows = int(wp.to_torch(self.sim.d.nefc).max().item())
                if contacts > self.sim.d.naconmax or rows >= self.sim.d.njmax:
                    raise NumericalTransitionError(
                        f"{self.BACKEND_NAME} capacity reached: contacts={contacts}/{self.sim.d.naconmax}, "
                        f"constraint_rows={rows}/{self.sim.d.njmax}", self.read_frame())

    def read_frame(self):
        self._check_open()
        self._torch_barrier()
        with wp.ScopedDevice(str(self.device)):
            wp.copy(self._scratch.qpos, self.sim.d.qpos)
            self.sim._mjw.kinematics(self.sim.m, self._scratch)
        wp.synchronize_device(str(self.device))
        qpos = wp.to_torch(self.sim.d.qpos).clone()
        qvel = wp.to_torch(self.sim.d.qvel).clone()
        sites = wp.to_torch(self._scratch.site_xpos).clone()
        # MuJoCo free-joint translational velocity uses world axes, angular
        # velocity body axes. Convert at the free body origin (mjOBJ_XBODY).
        rot = quaternion_matrix(qpos[:, self._object_q+3:self._object_q+7])
        linear = qvel[:, self._object_v:self._object_v+3]
        angular = (rot @ qvel[:, self._object_v+3:self._object_v+6, None]).squeeze(-1)
        return Frame(qpos, qvel, sites, torch.cat((linear, angular), dim=-1))

    def describe(self):
        result = {"backend": self.BACKEND_NAME, "contact_preset": self._preset,
                "numerics_profile": self._numerics_profile,
                "native_package_modified": False,
                "device": str(self.device), "physics_precision": "float32",
                "frame_precision": "float32", "indexed_reset": True,
                "post_step_readout": "scratch_kinematics_and_free_joint_world_twist",
                "synchronization": "explicit_torch_warp_barriers",
                "comfree_commit": COMFREE_COMMIT, "num_envs": self.num_envs,
                "contact_capacity_total": self.sim.d.naconmax,
                "constraint_capacity_per_world": self.sim.d.njmax,
                "gpu_qualified": False, "qualification_scope": "bounded_interface_checks_only",
                "reset_protocol": "raw_initial_state_without_settling"}
        if self._row_orderer is not None:
            result["row_ordering"] = self._row_orderer.describe()
        return result

    def close(self):
        if not self._closed:
            self._torch_barrier()
            wp.synchronize_device(str(self.device))
            self._scratch = None
            self.sim.Close()
            self._closed = True
