"""M6, ComFree-Sim (comfree_warp 0.3).

The config and preset tests run anywhere. The simulator tests need
comfree_warp >= 0.3, which only the contact_comfree2 env has, and skip
elsewhere.
"""

from __future__ import annotations

from importlib import metadata

import numpy as np
import pytest

from conftest import ROLLOUT_CUBE, matches_within_noise
from ContactModelStudy.Simulators.ComFreeV2 import ComFreeV2Config
from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Utils.ContactModelPresets import _presetConfig, getContactModel

DT = 0.004


def _has_comfree_v2() -> bool:
    try:
        version = metadata.version("comfree_warp")
    except metadata.PackageNotFoundError:
        return False
    return tuple(int(p) for p in version.split(".")[:2]) >= (0, 3)


needs_v2 = pytest.mark.skipif(not _has_comfree_v2(), reason="needs comfree_warp >= 0.3 (contact_comfree2 env)")


@pytest.mark.parametrize("kwargs", [dict(alpha=0.0), dict(alpha=1.5), dict(gamma=0.5),
                                    dict(refinement=5), dict(refinement=-1), dict(refinement=True),
                                    dict(cone="elliptic")])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        ComFreeV2Config(**kwargs)


def test_m6_preset():
    assert getContactModel("comfree2").name == "M6"
    cfg = _presetConfig("M6", task=CubeReorient(), timestep=DT)
    assert isinstance(cfg, ComFreeV2Config)
    assert (cfg.alpha, cfg.gamma, cfg.refinement) == (0.002, 10.0, 2)


@pytest.mark.gpu
@needs_v2
def test_parameters_reach_the_device_model():
    from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
    sim = GetContactModelSim("M6", ROLLOUT_CUBE, N=2, timestep=DT, alpha=0.01, gamma=4.0, refinement=3)
    assert np.allclose(sim.m.comfree_alpha.numpy(), 0.01)
    assert np.allclose(sim.m.comfree_gamma.numpy(), 4.0)
    assert sim.m.comfree_refinement == 3


@pytest.mark.gpu
@needs_v2
def test_holds_the_grasp(cube_initial):
    from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
    q0, _, u0 = cube_initial
    sim = GetContactModelSim("M6", ROLLOUT_CUBE, N=4, timestep=DT)
    sim.SetState(q0)
    sim.SetControl(u0)
    sim.Step_GPU(250)
    q = sim.GetState()[0]
    assert np.isfinite(q).all()
    # the cube stays in the hand: within 2 cm of where it started
    assert np.linalg.norm(q[:, 16:19] - q0[16:19], axis=1).max() < 0.02


@pytest.mark.gpu
@needs_v2
def test_captured_graph_reproduces_eager_rollout(cube_initial):
    # BroadcastState clears the warm start, so every replay starts alike, as eager steps do
    import warp as wp
    from ContactModelStudy.Simulators.ComFreeV2 import ComFreeV2
    q0, _, u0 = cube_initial
    H, sub, NW = 8, 4, 64
    sim = ComFreeV2(ROLLOUT_CUBE, ComFreeV2Config(timestep=DT, horizon=H, substeps=sub), N=NW)
    rng = np.random.default_rng(1)
    Uwp = wp.array((u0 + rng.normal(0, 0.2, (NW, H, u0.size))).astype(np.float32), device="cuda")
    qd = wp.array(q0.astype(np.float32), device="cuda")
    vd = wp.zeros(sim.nv, dtype=float, device="cuda")

    def body():
        sim.BroadcastState(qd, vd)
        sim.SetControlSequence(Uwp)
        sim.Step_GPU(H * sub)

    eager = []
    for _ in range(3):
        body()
        wp.synchronize()
        eager.append(sim.GetState()[0])
    with wp.ScopedCapture() as cap:
        body()
    wp.capture_launch(cap.graph)
    wp.synchronize()
    assert matches_within_noise(sim.GetState()[0], eager)
