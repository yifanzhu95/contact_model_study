"""tests/test_integration.py

Integration smoke tests. Require MuJoCo + CUDA.
Run with: pytest tests/test_integration.py -v

Each test loads the push scene (smallest) and runs a few steps
to verify the full dispatch pipeline works end-to-end.
"""

import pytest
import numpy as np

try:
    import mujoco
    import warp as wp
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False

pytestmark = pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco/warp not installed")

from pathlib import Path
SCENES_DIR = Path(__file__).parent.parent / "scenes"


@pytest.fixture(scope="module")
def push_mjm():
    xml_path = SCENES_DIR / "tasks" / "push_accurate.xml"
    return mujoco.MjModel.from_xml_path(str(xml_path))


@pytest.fixture(autouse=True, scope="module")
def warp_init():
    wp.init()


# ---------------------------------------------------------------------------

class TestBackendDispatch:
    """Verify that each backend can: put_model, make_data, step N times."""

    N_STEPS = 10
    N_WORLDS = 8

    def _run_smoke(self, push_mjm, cfg):
        from contact_study.contact_models import api
        m = api.put_model(push_mjm, cfg)
        d = api.make_data(push_mjm, m, nworld=self.N_WORLDS)
        for _ in range(self.N_STEPS):
            api.step(m, d)
        wp.synchronize()
        qpos = d.qpos.numpy()
        assert qpos.shape == (self.N_WORLDS, push_mjm.nq)
        assert not np.any(np.isnan(qpos)), "NaN in qpos after stepping"

    def test_M2_smoke(self, push_mjm):
        from contact_study.contact_models.config import ContactModelConfig
        self._run_smoke(push_mjm, ContactModelConfig.M2())

    def test_M1_smoke(self, push_mjm):
        from contact_study.contact_models.config import ContactModelConfig
        self._run_smoke(push_mjm, ContactModelConfig.M1())

    def test_M3_smoke(self, push_mjm):
        from contact_study.contact_models.config import ContactModelConfig
        self._run_smoke(push_mjm, ContactModelConfig.M3())

    def test_M4_smoke(self, push_mjm):
        from contact_study.contact_models.config import ContactModelConfig
        self._run_smoke(push_mjm, ContactModelConfig.M4())

    def test_M4_damping_friction_smoke(self, push_mjm):
        from contact_study.contact_models.config import ContactModelConfig
        self._run_smoke(push_mjm, ContactModelConfig.M4(damping_friction=True))


class TestResetData:
    def test_reset_restores_initial_qpos(self, push_mjm):
        from contact_study.contact_models import api
        from contact_study.contact_models.config import ContactModelConfig
        cfg = ContactModelConfig.M2()
        m   = api.put_model(push_mjm, cfg)
        d   = api.make_data(push_mjm, m)
        q0  = d.qpos.numpy().copy()
        for _ in range(20):
            api.step(m, d)
        api.reset_data(push_mjm, m, d)
        q_reset = d.qpos.numpy()
        np.testing.assert_array_almost_equal(q_reset, q0, decimal=5)


class TestPhysicsNoisePutModel:
    def test_M7_does_not_crash(self, push_mjm):
        from contact_study.contact_models import api
        from contact_study.contact_models.config import ContactModelConfig
        cfg = ContactModelConfig.M7(friction_sigma=0.2)
        m   = api.put_model(push_mjm, cfg)
        d   = api.make_data(push_mjm, m)
        for _ in range(5):
            api.step(m, d)
        wp.synchronize()
        assert not np.any(np.isnan(d.qpos.numpy()))


class TestBatchedRollout:
    def test_fixed_sample_rollout(self, push_mjm):
        import contact_study.tasks.tasks  # noqa
        from contact_study.contact_models.config import ContactModelConfig
        from contact_study.tasks.base import get_task
        from contact_study.utils.rollout import fixed_sample_rollout

        task = get_task("push")
        rng  = np.random.default_rng(0)
        q0, v0 = task.sample_initial_state(rng)

        result = fixed_sample_rollout(
            mjm          = push_mjm,
            cfg          = ContactModelConfig.M2(),
            n_samples    = 64,
            horizon      = 10,
            cost_fn      = task.cost_fn,
            initial_qpos = q0,
            initial_qvel = v0,
            rng          = rng,
        )
        assert result["costs"].shape == (64,)
        assert result["final_qpos"].shape[1] == push_mjm.nq
        assert not np.any(np.isnan(result["costs"]))
