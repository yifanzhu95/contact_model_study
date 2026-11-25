"""GPU planning and serial CPU reference rollouts with calibrated servo resets."""
import mujoco
import numpy as np
import warp as wp

from ContactModelStudy.Simulators.Mujoco import Mujoco
from ContactModelStudy.Simulators.VectorizedSimulator import DeviceState, VectorizedSimulator


@wp.kernel
def _broadcast(source: wp.array(dtype=wp.float32), target: wp.array2d(dtype=wp.float32)):
    n, i = wp.tid()
    target[n, i] = source[i]


@wp.kernel
def _sequence_control(sequence: wp.array3d(dtype=wp.float32), t: int,
                      command: wp.array2d(dtype=wp.float32)):
    n, i = wp.tid()
    command[n, i] = sequence[n, t, i]


@wp.kernel
def _filter_targets(command: wp.array2d(dtype=wp.float32), act: wp.array2d(dtype=wp.float32),
                    tau: wp.array(dtype=wp.float32), limits: wp.array2d(dtype=wp.float32),
                    early: wp.array(dtype=wp.int32), dt: float,
                    target: wp.array2d(dtype=wp.float32)):
    n, i = wp.tid()
    u = wp.clamp(command[n, i], limits[i, 0], limits[i, 1])
    previous = act[n, i]
    updated = u + (previous - u) * wp.exp(-dt / tau[i])
    target[n, i] = previous
    if early[i] != 0:
        target[n, i] = updated
    act[n, i] = updated


class CalibratedRollouts(VectorizedSimulator):
    """Add repeatable servo state to the backend selected by the shared factory.

    MJWarp-family backends integrate native MjData.act. Kamino exposes position
    targets instead; integrate the exact first-order target filter around each
    of its physics steps. All buffers are allocated before CUDA graph capture.
    """
    def __init__(self, backend):
        super().__init__(backend.model_path, backend.config, backend.N)
        self.backend, self.mjm = backend, backend.mjm
        dev = self.config.device
        self.activation = wp.zeros(16, dtype=wp.float32, device=dev)
        self.native = hasattr(getattr(backend, "d", None), "act")
        if not self.native:
            self.act = wp.zeros((self.N, 16), dtype=wp.float32, device=dev)
            self.command = wp.zeros((self.N, 16), dtype=wp.float32, device=dev)
            self.target = wp.zeros_like(self.command)
            self.sequence = wp.zeros((self.N, self.horizon, 16), dtype=wp.float32, device=dev)
            self.tau = wp.array(self.mjm.actuator_dynprm[:, 0], dtype=wp.float32, device=dev)
            self.limits = wp.array(self.mjm.actuator_ctrlrange, dtype=wp.float32, device=dev)
            self.early = wp.array(self.mjm.actuator_actearly, dtype=wp.int32, device=dev)
            self.sequence_active = False

    nq = property(lambda self: self.backend.nq)
    nv = property(lambda self: self.backend.nv)
    nu = property(lambda self: self.backend.nu)

    def BroadcastState(self, q, q_dot=None, u=None):
        self.backend.BroadcastState(q, q_dot, u)
        act = self.backend.d.act if self.native else self.act
        wp.launch(_broadcast, dim=(self.N, 16), inputs=[self.activation], outputs=[act],
                  device=self.config.device)
        if self.native:
            self.backend.d.qacc_warmstart.zero_()
            self.backend.d.time.zero_()
        elif hasattr(self.backend, "_time_wp"):
            self.backend._time_wp.zero_()

    def SetState(self, q, q_dot=None):
        self.backend.SetState(q, q_dot)
        act = self.backend.d.act if self.native else self.act
        wp.launch(_broadcast, dim=(self.N, 16), inputs=[self.activation], outputs=[act],
                  device=self.config.device)

    def GetState(self):
        return self.backend.GetState()

    def SetControl(self, u):
        self.backend.SetControl(u)
        if not self.native:
            wp.copy(self.command, self.backend.DeviceState().ctrl)

    def GetControl(self):
        return self.backend.GetControl() if self.native else self.command.numpy()

    def SetControlSequence(self, sequence):
        if self.native:
            self.backend.SetControlSequence(sequence)
        else:
            if isinstance(sequence, wp.array):
                wp.copy(self.sequence, sequence)
            else:
                self.sequence.assign(self._validate_control_sequence(sequence).astype(np.float32))
            self.sequence_active, self._sequence_index = True, 0

    def ClearControlSequence(self):
        self.backend.ClearControlSequence()
        if not self.native:
            self.sequence_active, self._sequence_index = False, 0

    def Step_GPU(self, steps=1):
        if self.native:
            self.backend.Step_GPU(steps)
            return
        for _ in range(steps):
            if self.sequence_active:
                t = min(self._sequence_index // self.config.resolved_substeps, self.horizon - 1)
                wp.launch(_sequence_control, dim=(self.N, 16), inputs=[self.sequence, t],
                          outputs=[self.command], device=self.config.device)
                self._sequence_index += 1
            wp.launch(_filter_targets, dim=(self.N, 16),
                      inputs=[self.command, self.act, self.tau, self.limits, self.early,
                              self.config.timestep], outputs=[self.target], device=self.config.device)
            self.backend.SetControl(self.target)
            self.backend.Step_GPU(1)

    def DeviceState(self):
        return self.backend.DeviceState()

    def Close(self):
        self.backend.Close()


class CpuRollouts(VectorizedSimulator):
    """Small, serial reference backend; uses the SAME MPPI and task as CUDA."""
    def __init__(self, path, config, N, lower, upper):
        super().__init__(path, config, N)
        self.mjm = mujoco.MjModel.from_xml_path(str(path))
        self.mjm.opt.timestep = config.timestep
        self.mjm.opt.gravity[:] = config.gravity
        Mujoco.applyContactParams(self.mjm, config)
        self.mjm.actuator_ctrlrange[:] = np.column_stack((lower, upper))
        self.data = [mujoco.MjData(self.mjm) for _ in range(N)]
        self.activation = wp.zeros(16, dtype=wp.float32, device="cpu")
        self.state = DeviceState(
            wp.zeros((N, self.nq), dtype=wp.float32, device="cpu"),
            wp.zeros((N, self.nv), dtype=wp.float32, device="cpu"),
            wp.zeros((N, self.nu), dtype=wp.float32, device="cpu"),
            wp.zeros((N, self.mjm.nsite), dtype=wp.vec3, device="cpu"))

    nq = property(lambda self: self.mjm.nq)
    nv = property(lambda self: self.mjm.nv)
    nu = property(lambda self: self.mjm.nu)

    def SetState(self, q, q_dot=None):
        q = self._to_worlds(q, self.nq, "q")
        v = self._to_worlds(np.zeros(self.nv) if q_dot is None else q_dot, self.nv, "q_dot")
        for i, data in enumerate(self.data):
            mujoco.mj_resetData(self.mjm, data)
            data.qpos[:] = q[i]
            data.qvel[:] = v[i]
            data.act[:] = self.activation.numpy()
            mujoco.mj_forward(self.mjm, data)
        self._refresh()

    def BroadcastState(self, q, q_dot=None, u=None):
        self.SetState(q.numpy(), None if q_dot is None else q_dot.numpy())
        if u is not None:
            self.SetControl(u)

    def GetState(self):
        return np.array([d.qpos for d in self.data]), np.array([d.qvel for d in self.data])

    def SetControl(self, u):
        u = self._to_worlds(u.numpy() if isinstance(u, wp.array) else u, 16, "u")
        for data, value in zip(self.data, u):
            data.ctrl[:] = value
        self.state.ctrl.assign(u.astype(np.float32))

    def GetControl(self):
        return np.array([d.ctrl for d in self.data])

    def SetControlSequence(self, sequence):
        super().SetControlSequence(sequence.numpy() if isinstance(sequence, wp.array) else sequence)

    def Step_GPU(self, steps=1):
        for _ in range(steps):
            control = self._next_control()
            if control is not None:
                self.SetControl(control)
            for data in self.data:
                mujoco.mj_step(self.mjm, data)
        self._refresh()

    def _refresh(self):
        q, v = self.GetState()
        self.state.qpos.assign(q.astype(np.float32))
        self.state.qvel.assign(v.astype(np.float32))
        self.state.ctrl.assign(self.GetControl().astype(np.float32))
        self.state.site_xpos.assign(np.asarray([d.site_xpos for d in self.data], dtype=np.float32))

    def DeviceState(self):
        return self.state


