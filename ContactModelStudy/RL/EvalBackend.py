"""MPC's original CPU evaluators behind the tensor reset/readout contract."""
from dataclasses import asdict
from types import SimpleNamespace
import mujoco
import numpy as np
import torch

from ContactModelStudy.RL.Contracts import Frame, NumericalTransitionError


class MpcEvalBatch:
    """One native evaluator per world; selected resets rebuild only that world."""
    def __init__(self, task, config):
        from ContactModelStudy.Utils.EvalSimulators import evalSimConfig
        self.name = config.physics_backend.removesuffix("_eval")
        self.sim_class, self.sim_config = evalSimConfig(self.name, config.physics_dt)
        self.config, self.num_envs = config, config.num_envs
        self.path, self._object_id = task.getModelPath(), int(task.indices[4])
        self.model = mujoco.MjModel.from_xml_path(self.path)
        self.model.opt.timestep = config.physics_dt  # Readout only; never integrated here.
        self.sims, self._scratch, self._closed = [None]*self.num_envs, mujoco.MjData(self.model), False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Native CPU evaluator is closed")

    def reset_worlds(self, ids, cases):
        from ContactModelStudy.Utils.EvalSimulators import makeEvalSim
        self._check_open()
        for i, case in zip(ids, cases, strict=True):
            fresh = makeEvalSim(self.name, self.path, self.config.physics_dt)
            try:
                fresh.SetState(np.asarray(case.qpos), np.asarray(case.qvel))
                fresh.SetControl(np.asarray(case.control))
            except BaseException:
                fresh.Close()
                raise
            previous, self.sims[i] = self.sims[i], fresh
            if previous is not None:
                previous.Close()

    def step(self, target, steps):
        self._check_open()
        for sim, command in zip(self.sims, target.detach().cpu().numpy(), strict=True):
            data = getattr(sim, "mjd", None)
            before = data.warning.number.copy() if data is not None else None
            sim.SetControl(command)
            sim.Step(steps)
            if before is not None and np.any(data.warning.number > before):
                raise NumericalTransitionError("Native MuJoCo evaluator reported a warning", self.read_frame())

    def read_frame(self):
        self._check_open()
        qs, vs, sites, twists = [], [], [], []
        for sim in self.sims:
            if sim is None:
                raise RuntimeError("Reset every world before reading native evaluator state")
            q, v = sim.GetState()
            self._scratch.qpos[:], self._scratch.qvel[:] = q, v
            mujoco.mj_forward(self.model, self._scratch)
            twist = np.empty(6)
            mujoco.mj_objectVelocity(self.model, self._scratch, mujoco.mjtObj.mjOBJ_XBODY,
                                    self._object_id, twist, 0)
            qs.append(q.copy()); vs.append(v.copy()); sites.append(self._scratch.site_xpos.copy())
            twists.append(np.r_[twist[3:], twist[:3]])
        return Frame(*(torch.tensor(np.stack(x), dtype=torch.float32) for x in (qs, vs, sites, twists)))

    def describe(self):
        return {"backend": self.config.physics_backend, "contact_preset": "native",
                "implementation": self.sim_class, "solver_config": asdict(self.sim_config),
                "num_envs": self.num_envs, "indexed_reset": True,
                "reset_protocol": "fresh_native_evaluator_for_selected_world_without_settling",
                "post_step_readout": "canonical_native_state_and_separate_scratch_forward",
                "gpu_qualified": False, "native_package_modified": False}

    def close(self):
        if self._closed:
            return
        from ContactModelStudy.RL.RunIO import close_resources
        resources = [(f"world_{i}", SimpleNamespace(close=sim.Close))
                     for i, sim in enumerate(self.sims) if sim is not None]
        self.sims, self._scratch, self._closed = [], None, True
        errors = close_resources(resources)
        if errors:
            raise RuntimeError(errors[0]["message"])
