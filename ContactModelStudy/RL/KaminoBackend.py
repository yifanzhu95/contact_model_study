"""RL adapter for the existing Newton 1.6 Kamino simulator, with indexed reset."""
from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version

import mujoco
import numpy as np
import torch
import warp as wp

from ContactModelStudy.RL.Contracts import Frame, NumericalTransitionError
from ContactModelStudy.RL.Encoding import quaternion_matrix


def array_fields(container, prefix=""):
    """Newton 1.6 state/control arrays, including registered namespaces."""
    from newton import Model
    for name, value in vars(container).items():
        if isinstance(value, wp.array):
            yield prefix + name, value
        elif isinstance(value, Model.AttributeNamespace):
            yield from array_fields(value, prefix + name + ".")


def body_origin_twist(body_pose_xyzw, body_twist_com, local_com):
    """Newton world COM linear/angular -> world body-origin linear/angular."""
    quat = body_pose_xyzw[:, (6, 3, 4, 5)]
    rot = quaternion_matrix(quat)
    offset = (rot @ local_com[..., None]).squeeze(-1)
    angular = body_twist_com[:, 3:]
    linear = body_twist_com[:, :3] - torch.linalg.cross(angular, offset)
    return torch.cat((linear, angular), dim=-1)


class KaminoBatch:
    """Uniform replicated LEAP worlds; no global reset-and-restore emulation.

    The original integrator/configuration is reused. Joint coordinates remain
    reduced readouts, while object pose/twist and sites use actual rigid bodies.
    """
    def __init__(self, task, config):
        if not torch.cuda.is_available():
            raise RuntimeError("Kamino requires working CUDA Torch; no CPU fallback")
        for name, expected in (("newton", "1.6.0"), ("warp-lang", "1.17.0"), ("mujoco", "3.6.0")):
            if version(name) != expected:
                raise RuntimeError(f"Kamino RL adapter requires {name}=={expected}")
        from ContactModelStudy.Simulators.Kamino import Kamino, KaminoConfig
        self.num_envs, self.device = config.num_envs, torch.device(config.device)
        self._closed = self._faulted = False
        self._object_q, self._object_v = map(int, task.indices[:2])
        with wp.ScopedDevice(str(self.device)):
            # Preserve actual source defaults, including PADMM 200 iterations.
            self.sim = Kamino(task.getModelPath(), KaminoConfig(
                timestep=config.physics_dt, device=str(self.device)), self.num_envs)
            self.sim.ClearControlSequence()
            self._read_q = wp.empty_like(self.sim.qpos_wp)
            self._read_v = wp.empty_like(self.sim.qvel_wp)
            self._read_sites = wp.empty_like(self.sim.site_xpos_wp)
        self.model = self.sim.mjm
        if (self.model.nq, self.model.nv, self.model.nu) != (23, 22, 16):
            raise ValueError("Kamino RL requires the original LEAP reorientation layout")
        mj_body = int(task.indices[4])
        mj_joint = int(self.model.body_jntadr[mj_body])
        if (self.model.body_parentid[mj_body] != 0 or self.model.body_jntnum[mj_body] != 1
                or self.model.jnt_type[mj_joint] != mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("Object readout requires the world-root free object")
        name = self.model.joint(mj_joint).name
        model = self.sim.model
        jpw = model.joint_count // self.num_envs
        local_joint = [str(label).rsplit("/", 1)[-1] for label in model.joint_label[:jpw]].index(name)
        children = model.joint_child.numpy()
        self._object_bodies = torch.tensor(children[local_joint::jpw], dtype=torch.long, device=self.device)
        if (model.body_count % self.num_envs or model.joint_coord_count % self.num_envs
                or model.joint_dof_count % self.num_envs):
            raise ValueError("Indexed copy requires uniform replicated worlds")
        # Reject layout changes rather than guessing which arrays are per-world.
        for container in (self.sim.state_0, self.sim.state_1, self.sim.control):
            for _, arr in array_fields(container):
                self._world_rows(arr)
        self._reset_mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        self._last_status = None
        wp.synchronize_device(str(self.device))

    def _world_rows(self, array):
        if array.shape[0] % self.num_envs:
            raise ValueError(f"Unexpected replicated array layout: {array.shape}")
        tensor = wp.to_torch(array)
        return tensor.reshape(self.num_envs, array.shape[0] // self.num_envs, *tensor.shape[1:])

    def _check_open(self):
        if self._closed or self._faulted:
            raise RuntimeError("Kamino backend is closed or faulted; recreate after diagnostics")

    def reset_worlds(self, ids, cases):
        self._check_open()
        if not ids:
            return
        if len(ids) != len(cases) or len(set(ids)) != len(ids) or any(i < 0 or i >= self.num_envs for i in ids):
            raise ValueError("One case per unique in-range world is required")
        from ContactModelStudy.Simulators.Kamino import _mujoco_to_newton_kernel
        sim, model = self.sim, self.sim.model
        torch.cuda.synchronize(self.device)
        wp.synchronize_device(str(self.device))
        self._reset_mask.zero_()
        self._reset_mask[ids] = True  # Final global-world bit remains False.
        for name, values in (("qpos_wp", [c.qpos for c in cases]),
                             ("qvel_wp", [c.qvel for c in cases]),
                             ("ctrl_wp", [c.control for c in cases])):
            wp.to_torch(getattr(sim, name))[ids] = torch.tensor(values, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize(self.device)
        with wp.ScopedDevice(str(self.device)):
            # Scratch joint inputs can be rebuilt for all worlds; the actual
            # state and solver caches are only reset by the native world mask.
            wp.launch(_mujoco_to_newton_kernel, dim=self.num_envs,
                inputs=[sim.qpos_wp, sim.qvel_wp, model.joint_coord_world_start, model.joint_dof_world_start,
                        sim._q_src, sim._d_src, *sim._free, sim._n_free],
                outputs=[sim._reset_q, sim._reset_qd], device=str(self.device))
            sim._reset_success.zero_()
            sim.solver.reset(sim.state_0, world_mask=wp.from_torch(self._reset_mask),
                             config=sim._reset_config, success_mask=sim._reset_success)
        wp.synchronize_device(str(self.device))
        if not wp.to_torch(sim._reset_success)[ids].all().item():
            frame = self.read_frame()
            self._faulted = True
            raise NumericalTransitionError(f"Kamino FK reset failed in selected worlds {ids}", frame)
        # Clear only the selected controls, matching Control.clear(model)
        # semantics while keeping valid quaternion targets in coord layout.
        for name, arr in array_fields(sim.control):
            rows = self._world_rows(arr)
            if name == "joint_target_q":
                rows[ids] = self._world_rows(model.joint_target_q)[ids]
            else:
                rows[ids] = 0
        local_targets = wp.to_torch(sim._targets).long()
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
        self._world_rows(sim.control.joint_target_q)[ids_tensor[:, None], local_targets[None, :]] = wp.to_torch(sim.ctrl_wp)[ids]
        source = dict(array_fields(sim.state_0))
        for name, arr in array_fields(sim.state_1):
            self._world_rows(arr)[ids] = self._world_rows(source[name])[ids]
        wp.to_torch(sim._time_wp)[ids] = 0
        wp.to_torch(sim._contact_peak)[ids] = 0
        torch.cuda.synchronize(self.device)
        with wp.ScopedDevice(str(self.device)):
            sim._refreshMirror()
        wp.synchronize_device(str(self.device))

    def step(self, target, steps):
        self._check_open()
        command = target.detach().contiguous()
        torch.cuda.synchronize(self.device)
        with wp.ScopedDevice(str(self.device)):
            self.sim.SetControl(wp.from_torch(command))
            for _ in range(steps):
                self.sim.Step_GPU(1)
                wp.synchronize_device(str(self.device))
                peak = wp.to_torch(self.sim._contact_peak)
                # These small status diagnostics cross to the host; physical
                # state, controls, observations and rewards stay on the GPU.
                self._last_status = self.sim.solver.status.numpy().copy()
                finite = all(np.isfinite(self._last_status[key]).all() for key in ("r_p", "r_d", "r_c"))
                if not finite or (peak >= self.sim.contact_capacity).any().item():
                    frame = self.read_frame()
                    self._faulted = True
                    raise NumericalTransitionError("Kamino nonfinite residual or contact capacity reached", frame)

    def read_frame(self):
        self._check_open()
        from ContactModelStudy.Simulators.Kamino import _newton_to_mujoco_kernel, _site_kernel
        sim, model = self.sim, self.sim.model
        torch.cuda.synchronize(self.device)
        with wp.ScopedDevice(str(self.device)):
            wp.launch(_newton_to_mujoco_kernel, dim=self.num_envs,
                inputs=[sim.state_0.joint_q, sim.state_0.joint_qd, model.joint_coord_world_start,
                        model.joint_dof_world_start, sim._q_dst, sim._d_dst, *sim._free, sim._n_free],
                outputs=[self._read_q, self._read_v], device=str(self.device))
            wp.launch(_site_kernel, dim=(self.num_envs, self.model.nsite),
                inputs=[sim._site_local, model.shape_world_start, model.shape_body,
                        model.shape_transform, sim.state_0.body_q],
                outputs=[self._read_sites], device=str(self.device))
        wp.synchronize_device(str(self.device))
        qpos, qvel = wp.to_torch(self._read_q).clone(), wp.to_torch(self._read_v).clone()
        pose = wp.to_torch(sim.state_0.body_q)[self._object_bodies]
        velocity = wp.to_torch(sim.state_0.body_qd)[self._object_bodies]
        com = wp.to_torch(model.body_com)[self._object_bodies]
        twist = body_origin_twist(pose, velocity, com)
        qpos[:, self._object_q:self._object_q+3] = pose[:, :3]
        quat = pose[:, (6, 3, 4, 5)]  # Newton xyzw -> canonical wxyz.
        qpos[:, self._object_q+3:self._object_q+7] = quat
        qvel[:, self._object_v:self._object_v+3] = twist[:, :3]
        qvel[:, self._object_v+3:self._object_v+6] = (
            quaternion_matrix(quat).transpose(-2, -1) @ twist[:, 3:, None]).squeeze(-1)
        return Frame(qpos, qvel, wp.to_torch(self._read_sites).clone(), twist)

    def render_body_poses(self, world=0):
        """Owned host xyz/wxyz poses for video only, including actual hand drift."""
        self._check_open()
        if not 0 <= world < self.num_envs:
            raise ValueError('Invalid render world')
        per_world = self.sim.model.body_count // self.num_envs
        names = [str(label).rsplit('/', 1)[-1] for label in self.sim.model.body_label[:per_world]]
        if len(set(names)) != len(names):
            raise ValueError('Ambiguous body labels; cannot render actual Kamino poses')
        lookup = {name: i for i, name in enumerate(names)}
        wp.synchronize_device(str(self.device))
        poses = wp.to_torch(self.sim.state_0.body_q).reshape(self.num_envs, per_world, 7)[world].cpu().numpy().copy()
        result = {}
        for body in range(1, self.model.nbody):
            if self.model.body_mocapid[body] >= 0:
                continue  # The renderer owns the current case's goal marker.
            name = self.model.body(body).name
            if name not in lookup:
                raise ValueError(f'Kamino render body {name!r} is missing')
            pose = poses[lookup[name]]
            result[body] = pose[[0, 1, 2, 6, 3, 4, 5]]
        return result

    def describe(self):
        return {"backend": "kamino", "contact_preset": "M5", "device": str(self.device),
                "newton_version": version("newton"), "physics_precision": "float32",
                "frame_precision": "float32", "num_envs": self.num_envs,
                "indexed_reset": "native_world_mask_N_plus_1_and_selected_control_state_copy",
                "post_step_readout": "actual_rigid_body_object_and_sites_reduced_hand_joints",
                "synchronization": "explicit_torch_warp_barriers",
                "solver_config": asdict(self.sim.config),
                "gpu_qualified": False, "qualification_scope": "bounded_reset_readout_checks_only",
                "reset_protocol": "raw_initial_state_without_settling"}

    def close(self):
        if not self._closed:
            torch.cuda.synchronize(self.device)
            wp.synchronize_device(str(self.device))
            self._read_q = self._read_v = self._read_sites = None
            self.sim.Close()
            self._closed = True
