"""GPU row ordering: conformance, failure guards, reset/readout and inference only."""
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from ContactModelStudy.Controllers.RLPolicyController import RLPolicyController
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig
from ContactModelStudy.RL.DiagnosticOrdering import permute_rows
from ContactModelStudy.RL.GpuConstraintRows import NUMERICS_PROFILE, ROW_FIELDS, ConstraintOrderingError
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv
from ContactModelStudy.RL.Policy import Actor, export_actor


def test_legacy_config_serialization_and_explicit_profile_rejection():
    legacy = EnvConfig().to_dict()
    assert "numerics_profile" not in legacy
    assert EnvConfig(**legacy).to_dict() == legacy
    with pytest.raises(ValueError):
        EnvConfig(numerics_profile=NUMERICS_PROFILE)
    with pytest.raises(ValueError):
        EnvConfig(physics_backend="mjwarp", device="cuda:0", numerics_profile="unknown")
    with pytest.raises(ValueError):
        EnvConfig(physics_backend="kamino", contact_preset="M5", device="cuda:0", numerics_profile=NUMERICS_PROFILE)


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Explicit CUDA runtime required")


def environment(preset="M2", worlds=3):
    return LeapVectorEnv(EnvConfig(physics_backend="mjwarp", device="cuda:0", contact_preset=preset,
                                  num_envs=worlds, numerics_profile=NUMERICS_PROFILE),
                         CaseBank.development(count=4, seed=9, difficulty=8))


@pytest.mark.gpu
@gpu
def test_optional_profile_does_not_patch_native_consumers():
    from comfree_warp.mujoco_warp._src import forward, solver
    before = (forward.step, solver.solve, solver._update_gradient_incremental)
    env = environment(worlds=2)
    try:
        env.reset_all([0, 1]); env.step(torch.zeros((2, 16), device=env.device))
        assert before == (forward.step, solver.solve, solver._update_gradient_incremental)
        assert env.backend.sim._mjw.step is before[0]
        assert env.describe()["physics"]["numerics_profile"] == NUMERICS_PROFILE
    finally:
        env.close()


@pytest.mark.gpu
@gpu
def test_gpu_permutation_matches_cpu_reference_including_tails_and_contact_mapping():
    env = environment()
    try:
        env.reset_all([0, 1, 2])
        d, orderer = env.backend.sim.d, env.backend._row_orderer
        counts = [4, 3, 0]
        wp.to_torch(d.nefc).copy_(torch.tensor(counts, dtype=torch.int32, device=env.device))
        for name in ROW_FIELDS:
            value = wp.to_torch(getattr(d.efc, name))
            value.fill_(-123 if not value.is_floating_point() else float("nan"))
            for world, n in enumerate(counts):
                for row in range(n):
                    if name == "type":
                        value[world, row] = (6, 2, 6, 0)[row]
                    elif name == "J":
                        value[world, row] = 0
                        value[world, row, row] = (3, -2, 1, 4)[row]
                    else:
                        value[world, row] = (row+1)*(world+1)
        wp.to_torch(d.nacon)[0] = 3
        wp.to_torch(d.contact.worldid)[:3] = torch.tensor([0, 1, 0], device=env.device)
        addresses = wp.to_torch(d.contact.efc_address)
        addresses.fill_(-1)
        addresses[:3, :2] = torch.tensor([[0, 3], [2, 0], [2, 1]], device=env.device)
        before = {name: wp.to_torch(getattr(d.efc, name)).clone() for name in ROW_FIELDS}
        before_address = addresses.clone()
        permute_rows(d, "canonical")
        expected = {name: wp.to_torch(getattr(d.efc, name)).clone() for name in ROW_FIELDS}
        expected_address = addresses.clone()
        for name, value in before.items():
            wp.to_torch(getattr(d.efc, name)).copy_(value)
        addresses.copy_(before_address)
        torch.cuda.synchronize()
        orderer.apply(d)
        wp.synchronize()
        for name, value in expected.items():
            torch.testing.assert_close(wp.to_torch(getattr(d.efc, name)), value, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(addresses, expected_address)
        for contact, world in enumerate((0, 1, 0)):
            for edge in range(2):
                old, new = int(before_address[contact, edge]), int(addresses[contact, edge])
                for name in ROW_FIELDS:
                    torch.testing.assert_close(wp.to_torch(getattr(d.efc, name))[world, new], before[name][world, old], rtol=0, atol=0)
        # Numerical tails contain NaNs deliberately: only initialized prefixes may be inspected or changed.
        for world, n in enumerate(counts):
            for name, value in before.items():
                torch.testing.assert_close(wp.to_torch(getattr(d.efc, name))[world, n:], value[world, n:],
                                           rtol=0, atol=0, equal_nan=True)
    finally:
        env.close()


@pytest.mark.gpu
@gpu
def test_nonfinite_key_and_capacity_are_rejected_before_permutation():
    env = environment(worlds=2)
    try:
        env.reset_all([0, 1])
        env.step(torch.zeros((2, 16), device=env.device))
        d, orderer = env.backend.sim.d, env.backend._row_orderer
        # The initial raw grasp can still have zero active contacts. Explicitly
        # construct one initialized active row before injecting its invalid key.
        wp.to_torch(d.nefc)[0] = 1
        for name in ROW_FIELDS:
            wp.to_torch(getattr(d.efc, name))[0, 0] = 0
        live = wp.to_torch(d.efc.pos)
        before = wp.to_torch(d.qpos).clone()
        original = live[0, 0].item()
        live[0, 0] = float("nan")
        torch.cuda.synchronize()
        with pytest.raises(ConstraintOrderingError, match="Nonfinite"):
            orderer.apply(d)
        assert torch.equal(wp.to_torch(d.qpos), before)
        live[0, 0] = original
        wp.to_torch(d.nefc)[0] = d.njmax
        torch.cuda.synchronize()
        with pytest.raises(ConstraintOrderingError, match="capacity"):
            orderer.apply(d)
    finally:
        env.close()


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("preset", ["M1", "M2"])
def test_gpu_profile_indexed_reset_future_and_nonmutating_world_readout(preset, record_property):
    import json
    envs = [environment(preset) for _ in range(3)]
    env, reference, repeat = envs
    evidence = []
    try:
        action = torch.linspace(-.2, .3, 16, device=env.device)[None]*torch.tensor([[1.], [-.4], [.7]], device=env.device)
        for e in envs:
            e.reset_all([0, 1, 2]); e.step(action)
        live = env.backend.sim.d
        keys = ("qpos", "qvel", "ctrl", "qacc_warmstart", "qM", "time")
        before = {name: wp.to_torch(getattr(live, name))[[0, 2]].clone() for name in keys}
        env.reset_indices([1], [3])
        for name in keys:
            assert torch.equal(wp.to_torch(getattr(live, name))[[0, 2]], before[name])
        for step in range(3):
            for e in envs:
                e.step(action)
            frames = [e.backend.read_frame() for e in envs]
            for name in ("qpos", "qvel"):
                a, b, c = [getattr(frame, name)[[0, 2]] for frame in frames]
                noise = (b-c).abs().max()
                bound = 3*noise+torch.finfo(b.dtype).eps*b.abs().max().clamp_min(1)
                difference = (a-b).abs().max()
                assert difference <= bound
                evidence.append({"step": step+1, "field": name, "reset_difference": float(difference),
                                 "repeat_difference": float(noise), "bound": float(bound)})
        before_read = {name: wp.to_torch(getattr(live, name)).clone() for name in keys}
        frame = env.backend.read_frame()
        cpu = mujoco.MjData(env.backend.model)
        cpu.qpos[:] = frame.qpos[0].cpu().numpy(); cpu.qvel[:] = frame.qvel[0].cpu().numpy()
        mujoco.mj_forward(env.backend.model, cpu)
        twist = np.empty(6)
        mujoco.mj_objectVelocity(env.backend.model, cpu, mujoco.mjtObj.mjOBJ_XBODY,
                                env.backend._object_body, twist, 0)
        np.testing.assert_allclose(frame.sites[0].cpu(), cpu.site_xpos, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(frame.object_twist[0].cpu(), np.r_[twist[3:], twist[:3]], rtol=2e-6, atol=2e-7)
        for name in keys:
            assert torch.equal(wp.to_torch(getattr(live, name)), before_read[name])
        record_property("gpu_rows_reset_evidence", json.dumps({"preset": preset, "future": evidence}))
    finally:
        for e in envs:
            e.close()


@pytest.mark.gpu
@gpu
def test_explicit_profile_export_inference_and_legacy_environment_contract(tmp_path):
    env = environment(worlds=2)
    try:
        obs = env.reset_all([0, 1])
        torch.manual_seed(123)
        actor = Actor().to(env.device)
        path = tmp_path/"untrained_actor.pt"
        export_actor(actor, env.describe(), path)
        gpu = RLPolicyController(path, device="cuda:0")
        cpu = RLPolicyController(path)
        assert gpu.manifest["config"]["numerics_profile"] == NUMERICS_PROFILE
        torch.testing.assert_close(gpu.act(obs, env.current_target).cpu(), cpu.act(obs.cpu(), env.current_target.cpu()), rtol=1e-5, atol=1e-6)
    finally:
        env.close()
