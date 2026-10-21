"""Original M3/M4 solvers with the shared MuJoCo-layout RL reset/readout contract."""
from dataclasses import asdict

import warp as wp

from ContactModelStudy.RL.MjwarpBackend import MjwarpBatch


class _NativeContactBatch(MjwarpBatch):
    """Share layout handling, while retaining each original physics implementation."""
    EXTRA_RESET_FIELDS = ()

    def _make_simulator(self, task, config):
        from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
        return GetContactModelSim(config.contact_preset, task.getModelPath(), N=self.num_envs,
                                  timestep=config.physics_dt, device=str(self.device))

    def _native_step(self, model, data):
        # This bound method is ComFree._stepPhysics or XPBD._stepPhysics,
        # never the base simulator's MJWarp Newton step.
        self.sim._stepPhysics()

    def _select_step(self):
        self._numerics_profile = self.NUMERICS_PROFILE
        return self._native_step

    def _reset_extra_state(self, ids):
        # The pinned vendor reset handles its original data fields only.
        # M3 adds fields dynamically; M4 owns scratch on the simulator.
        for path in self.EXTRA_RESET_FIELDS:
            value = self.sim
            for part in path.split("."):
                value = getattr(value, part)
            wp.to_torch(value)[ids] = 0

    def describe(self):
        return {**super().describe(), "solver_config": asdict(self.sim.config),
                "physics_pipeline": type(self.sim).__name__,
                "extra_indexed_reset_fields": list(self.EXTRA_RESET_FIELDS)}


class ComFreeBatch(_NativeContactBatch):
    BACKEND_NAME = "comfree"
    ALLOWED_PRESETS = ("M3",)
    NUMERICS_PROFILE = "native_comfree_v1"
    EXTRA_RESET_FIELDS = ("d.qvel_smooth_pred", "d.qfrc_total", "d.efc.efc_dist", "d.efc.efc_mass")

    def _reset_solver_data(self, mask):
        self.sim._cfw.reset_data(self.sim.m, self.sim.d, mask)


class XpbdBatch(_NativeContactBatch):
    BACKEND_NAME = "xpbd"
    ALLOWED_PRESETS = ("M4",)
    NUMERICS_PROFILE = "native_xpbd_v1"
    EXTRA_RESET_FIELDS = ("_qvel_pred", "_qfrc_total", "_lambda_efc", "_dof_con_count")
