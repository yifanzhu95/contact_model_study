"""Bounded GPU interfaces; native trajectory repeatability is diagnostic only."""
from dataclasses import replace
from importlib.util import find_spec

import mujoco
import numpy as np
import pytest
import torch

from ContactModelStudy.Controllers.RLPolicyController import RLPolicyController
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv
from ContactModelStudy.RL.SkrlPPO import (PPOConfig, build_agent, collect_updates,
                                         restore_training_checkpoint, save_training_checkpoint)

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available() or find_spec("comfree_warp") is None,
    reason="Requires explicit CUDA Torch and pinned comfree_warp runtime")]


def make_env(preset="M2", steps=32, num_envs=2, numerics_profile=None):
    backend = {"M3": "comfree", "M4": "xpbd"}.get(preset, "mjwarp")
    return LeapVectorEnv(EnvConfig(physics_backend=backend, device="cuda:0",
                                  contact_preset=preset, episode_steps=steps, num_envs=num_envs,
                                  numerics_profile=numerics_profile),
                         CaseBank.development(count=4, seed=9, difficulty=8))


@pytest.mark.parametrize("reset_id", [0, 1])
@pytest.mark.parametrize("preset", ["M1", "M2", "M3", "M4"])
def test_indexed_gpu_reset_preserves_hidden_state_and_future(reset_id, preset, record_property):
    import warp as wp
    env, reference, replay = make_env(preset), make_env(preset), make_env(preset)
    other = 1-reset_id
    try:
        env.reset_all([0, 1])
        reference.reset_all([0, 1])
        replay.reset_all([0, 1])
        action = torch.full((2, 16), 0.2, device=env.device)
        env.step(action)
        reference.step(action)
        replay.step(action)
        data = env.backend.sim.d
        snapshot = {key: wp.to_torch(getattr(data, key))[other].clone() for key in
                    ("qpos", "qvel", "ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "qM", "time")}
        frame_before = env.read_frame()
        episode_before = (env.steps[other].clone(), env._goal_quat[other].clone(), env.current_target[other],
                          env.case_ids[other])
        rng = str(env._rng[other].bit_generator.state)
        env.reset_indices([reset_id], [2])
        for key, before in snapshot.items():
            assert torch.equal(wp.to_torch(getattr(data, key))[other], before), key
        frame_after = env.read_frame()
        for key in ('qpos', 'qvel', 'sites', 'object_twist'):
            assert torch.equal(getattr(frame_after, key)[other], getattr(frame_before, key)[other]), key
        assert torch.equal(env.steps[other], episode_before[0])
        assert torch.equal(env._goal_quat[other], episode_before[1])
        assert torch.equal(env.current_target[other], episode_before[2])
        assert env.case_ids[other] == episode_before[3]
        assert str(env._rng[other].bit_generator.state) == rng
        measured = []
        for step in range(3):
            env.step(action)
            reference.step(action)
            replay.step(action)
            actual, expected = env.backend.read_frame(), reference.backend.read_frame()
            repeated = replay.backend.read_frame()
            for key in ("qpos", "qvel"):
                a, b, c = (getattr(f, key)[other] for f in (actual, expected, repeated))
                noise = (b-c).abs().max()
                rounding = torch.finfo(b.dtype).eps * b.abs().max().clamp_min(1)
                measured.append({"control_step": step + 1, "field": key,
                                 "reset_difference": (a-b).abs().max().item(),
                                 "repeat_difference": noise.item(), "bound": (3*noise+rounding).item()})
                # Two independent replays do not estimate a guaranteed bound
                # on another native GPU trajectory. Isolation is asserted on
                # the state/cache/episode snapshots above, before integration.
                assert torch.isfinite(a).all()
        import json
        record_property("reset_evidence", json.dumps({"preset": preset, "reset_id": reset_id,
                        "numerics_profile": env.describe()["physics"]["numerics_profile"], "future": measured,
                        "future_comparison": "diagnostic_only_native_gpu_nondeterminism_accepted"}))
    finally:
        env.close()
        reference.close()
        replay.close()


@pytest.mark.parametrize("preset", ["M1", "M2", "M3", "M4"])
def test_three_world_middle_reset_with_different_commands(preset, record_property):
    import warp as wp
    import json
    envs = [make_env(preset, num_envs=3) for _ in range(3)]
    env, reference, repeat = envs
    try:
        action = torch.linspace(-0.25, 0.35, 16, device=env.device)[None, :] * torch.tensor(
            [[1.], [-0.4], [0.7]], device=env.device)
        for e in envs:
            e.reset_all([0, 1, 2])
            e.step(action)
        before = {name: wp.to_torch(getattr(env.backend.sim.d, name))[[0, 2]].clone() for name in
                  ("qpos", "qvel", "qacc_warmstart", "ctrl", "time", "qM")}
        env.reset_indices([1], [3])
        for name, expected in before.items():
            assert torch.equal(wp.to_torch(getattr(env.backend.sim.d, name))[[0, 2]], expected), name
        measured = []
        for step in range(3):
            for e in envs:
                e.step(action)
            frames = [e.backend.read_frame() for e in envs]
            for name in ("qpos", "qvel"):
                for world in (0, 2):
                    a, b, c = [getattr(f, name)[world] for f in frames]
                    noise = (b-c).abs().max()
                    rounding = torch.finfo(b.dtype).eps * b.abs().max().clamp_min(1)
                    difference, bound = (a-b).abs().max(), 3*noise + rounding
                    assert torch.isfinite(a).all()
                    measured.append({"step": step + 1, "world": world, "field": name,
                                     "reset_difference": difference.item(), "repeat_difference": noise.item(),
                                     "bound": bound.item()})
        record_property("three_world_evidence", json.dumps({"preset": preset, "future": measured,
            "future_comparison": "diagnostic_only_native_gpu_nondeterminism_accepted"}))
    finally:
        for e in envs:
            e.close()


@pytest.mark.parametrize("preset", ["M1", "M2"])
def test_default_dispatch_and_parameters_match_mpc(preset, monkeypatch):
    from dataclasses import asdict
    from ContactModelStudy.Utils.ContactModelPresets import _presetConfig
    env = make_env(preset)
    calls = []
    original = env.backend.sim._stepPhysics
    def observe_native():
        calls.append(1)
        return original()
    monkeypatch.setattr(env.backend.sim, "_stepPhysics", observe_native)
    try:
        assert asdict(env.backend.sim.config) == asdict(_presetConfig(
            preset, timestep=env.config.physics_dt, device=str(env.device)))
        assert env.describe()["physics"]["numerics_profile"] == "native_mjwarp_v1"
        assert env.describe()["config"]["numerics_profile"] == "native_mjwarp_v1"
        assert env.backend._row_orderer is None
        env.reset_all([0, 1])
        env.step(torch.zeros((2, 16), device=env.device))
        assert len(calls) == env.config.physics_steps
        assert torch.isfinite(env.backend.read_frame().qpos).all()
    finally:
        env.close()


def test_legacy_ordered_dispatch_does_not_patch_native_consumers(monkeypatch):
    import importlib
    import warp as wp
    native_solver = importlib.import_module("comfree_warp.mujoco_warp._src.solver")
    native_forward = importlib.import_module("comfree_warp.mujoco_warp._src.forward")
    original_solve, original_step = native_solver.solve, native_forward.step
    original_incremental = native_solver._update_gradient_incremental
    env = make_env("M1", numerics_profile="dense_newton_ordered_changes_v1")
    calls = []
    def observe_native(*args, **kwargs):
        calls.append(1)
        return original_incremental(*args, **kwargs)
    monkeypatch.setattr(native_solver, "_update_gradient_incremental", observe_native)
    try:
        env.reset_all([0, 1])
        env.step(torch.full((2, 16), 0.2, device=env.device))
        assert not calls  # Private RL dispatch retains its own ordered boundary.
        assert native_solver.solve is original_solve and native_forward.step is original_step
        assert env.backend.sim._mjw.step is original_step
        with wp.ScopedDevice(str(env.device)):
            env.backend.sim.Step_GPU(1)  # Existing public simulator still reaches native dispatch.
        wp.synchronize_device(str(env.device))
        assert calls
    finally:
        env.close()


@pytest.mark.parametrize("preset", ["M1", "M2", "M3", "M4"])
def test_gpu_readout_matches_fresh_cpu_kinematics_and_world_twist(preset):
    import warp as wp
    env = make_env(preset)
    try:
        case = env.case_bank.cases[0]
        qpos = list(case.qpos)
        qpos[-4:] = [np.cos(0.4), 0, 0, np.sin(0.4)]
        velocity = tuple(np.linspace(-0.2, 0.3, 22))
        moving = replace(case, case_id="moving", qpos=tuple(qpos), qvel=velocity)
        env.backend.reset_worlds([0, 1], [moving, moving])
        for _ in range(2):
            live = env.backend.sim.d
            snapshots = {name: wp.to_torch(getattr(live, name)).clone() for name in
                         ("qpos", "qvel", "qacc_warmstart", "ctrl", "qM")}
            frame = env.backend.read_frame()
            assert frame.qpos.device.type == "cuda" and frame.sites.device.type == "cuda"
            scratch = mujoco.MjData(env.backend.model)
            scratch.qpos[:] = frame.qpos[0].cpu().numpy()
            scratch.qvel[:] = frame.qvel[0].cpu().numpy()
            mujoco.mj_forward(env.backend.model, scratch)
            twist = np.empty(6)
            mujoco.mj_objectVelocity(env.backend.model, scratch, mujoco.mjtObj.mjOBJ_XBODY,
                                    env.backend._object_body, twist, 0)
            # Float32 FK vs float64 reference; fixed scales are metres and m/s, rad/s.
            np.testing.assert_allclose(frame.sites[0].cpu(), scratch.site_xpos, rtol=2e-6, atol=2e-7)
            np.testing.assert_allclose(frame.object_twist[0].cpu(), np.r_[twist[3:], twist[:3]],
                                       rtol=2e-6, atol=2e-7)
            for name, before in snapshots.items():
                assert torch.equal(wp.to_torch(getattr(live, name)), before), name
            env.backend.step(torch.tensor([case.control]*2, device=env.device), 1)
        assert env.backend.model.opt.iterations == (200 if preset == "M1" else 25)
    finally:
        env.close()


@pytest.mark.parametrize("preset", ["M1", "M2", "M3", "M4"])
@pytest.mark.ppo_update
def test_gpu_ppo_updates_exports_and_restores_without_cpu_state_roundtrip(tmp_path, preset, record_property):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    env = make_env(preset, steps=4)
    try:
        cfg = PPOConfig(steps_per_update=4, hidden_sizes=(16, 16))
        agent, actor = build_agent(env, cfg)
        before = {k: v.clone() for k, v in actor.state_dict().items()}
        metrics = []
        stats = collect_updates(env, agent, updates=1, on_update=lambda i, m: metrics.append(m))
        assert stats["transitions"] == 8 and stats["episode_resets"] == 2
        assert any(not torch.equal(v, before[k]) for k, v in actor.state_dict().items())
        for key in ("observations", "actions", "returns", "advantages"):
            values = agent.memory.get_tensor_by_name(key)
            assert values.device.type == "cuda" and torch.isfinite(values).all()
        save_training_checkpoint(agent, actor, env, cfg, tmp_path / "training.pt")
        rng = torch.cuda.get_rng_state(env.device)
        restored, restored_actor = build_agent(env, cfg)
        restore_training_checkpoint(restored, env, cfg, tmp_path / "training.pt")
        assert torch.equal(torch.cuda.get_rng_state(env.device), rng)
        for key, value in actor.state_dict().items():
            assert torch.equal(restored_actor.state_dict()[key], value)
        obs = env.reset_all([0, 1])
        expected = env.codec.decode(actor(obs), env.current_target)
        controller = RLPolicyController(tmp_path / "actor.pt", device=env.device)
        torch.testing.assert_close(controller.act(obs, env.current_target), expected)
        cpu_controller = RLPolicyController(tmp_path / "actor.pt")
        torch.testing.assert_close(cpu_controller.act(obs.cpu(), env.current_target.cpu()), expected.cpu(),
                                   rtol=1e-5, atol=1e-6)
        transition = env.step(torch.zeros(2, 16, device=env.device))
        old_obs = transition.next_obs.clone()
        env.reset_all([2, 3])
        assert torch.equal(transition.next_obs, old_obs)
        import json
        record_property("gpu_ppo_evidence", json.dumps({"preset": preset, "stats": stats,
            "metrics": metrics[0], "environment": env.describe(),
            "actor_max_parameter_change": max((v-before[k]).abs().max().item()
                                              for k, v in actor.state_dict().items())}, allow_nan=False))
    finally:
        env.close()


@pytest.mark.parametrize("preset", ["M3", "M4"])
def test_alternative_solver_dispatch_and_extra_reset_fields(preset, monkeypatch, record_property):
    import warp as wp
    from ContactModelStudy.RL import MjwarpNumerics
    from ContactModelStudy.Utils.ContactModelPresets import CONTACT_MODELS
    def reject_newton(*args, **kwargs):
        raise AssertionError("M3/M4 must not call the M1/M2 Newton dispatch")
    monkeypatch.setattr(MjwarpNumerics, "make_ordered_step", reject_newton)
    env = make_env(preset)
    try:
        monkeypatch.setattr(env.backend.sim._mjw, "step", reject_newton)
        calls = []
        native = env.backend.sim._stepPhysics
        def observe_native():
            calls.append(1)
            return native()
        monkeypatch.setattr(env.backend.sim, "_stepPhysics", observe_native)
        for key, value in CONTACT_MODELS[preset]["params"].items():
            assert getattr(env.backend.sim.config, key) == value, key
        assert type(env.backend.sim).__name__ == ("ComFree" if preset == "M3" else "XPBD")
        env.reset_all([0, 1])
        env.step(torch.full((2, 16), 0.2, device=env.device))
        assert len(calls) == env.config.physics_steps
        arrays = {}
        for path in env.backend.EXTRA_RESET_FIELDS:
            arr = env.backend.sim
            for part in path.split("."):
                arr = getattr(arr, part)
            arrays[path] = wp.to_torch(arr)
            arrays[path][0] = 7.0
            arrays[path][1] = 9.0
        torch.cuda.synchronize(env.device)
        other = {key: value[1].clone() for key, value in arrays.items()}
        env.reset_indices([0], [2])
        for key, value in arrays.items():
            assert not value[0].any(), key
            assert torch.equal(value[1], other[key]), key
        before = env.backend.read_frame()
        env.backend.reset_worlds([], [])
        after = env.backend.read_frame()
        assert torch.equal(before.qpos, after.qpos) and torch.equal(before.qvel, after.qvel)
        env.step(torch.zeros(2, 16, device=env.device))
        assert len(calls) == 2 * env.config.physics_steps
        description = env.describe()["physics"]
        assert description["numerics_profile"] == ("native_comfree_v1" if preset == "M3" else "native_xpbd_v1")
        record_property("native_dispatch_evidence", __import__("json").dumps({
            "preset": preset, "native_physics_calls": len(calls), "extra_fields": list(arrays),
            "other_world_extra_fields_preserved": True, "physics": description}, allow_nan=False))
    finally:
        env.close()
