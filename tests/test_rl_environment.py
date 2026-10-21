"""Meaningful RL contract checks using real CPU physics and the original cost kernel."""
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from ContactModelStudy.RL.CaseBank import CaseBank, EpisodeCase
from ContactModelStudy.RL.Contracts import EnvConfig, NumericalTransitionError
from ContactModelStudy.RL.Encoding import finite_reward, task_terms
from ContactModelStudy.RL.LeapVectorEnv import LeapGymEnv, LeapVectorEnv


@pytest.fixture
def bank():
    return CaseBank.development(count=4, seed=9, difficulty=8)


@pytest.fixture
def env(bank):
    e = LeapVectorEnv(EnvConfig(), bank)
    e.reset_all([0, 1])
    yield e
    e.close()


@pytest.mark.parametrize("kwargs", [
    {"num_envs": 0}, {"num_envs": 1.5}, {"episode_steps": True},
    {"physics_dt": float("nan")}, {"control_dt": 0.007},
    {"target_rate": 0}, {"physics_backend": "kamino"}, {"contact_preset": "M3"},
    {"physics_backend": "mjwarp"}, {"device": "cuda:0"},
    {"physics_backend": "mjwarp", "device": "cuda:1"},
    {"physics_backend": "kamino", "device": "cuda:0", "contact_preset": "M2"},
    {"physics_backend": "mjwarp", "device": "cuda:0", "contact_preset": "M5"},
    {"physics_backend": "comfree", "device": "cuda:0", "contact_preset": "M2"},
    {"physics_backend": "xpbd", "device": "cuda:0", "contact_preset": "M3"},
    {"physics_backend": "comfree", "device": "cpu", "contact_preset": "M3"},
    {"physics_backend": "xpbd", "device": "cpu", "contact_preset": "M4"},
])
def test_invalid_or_unimplemented_configuration_is_explicit(kwargs):
    with pytest.raises(ValueError):
        EnvConfig(**kwargs)


@pytest.mark.parametrize("backend,preset", [("comfree", "M3"), ("xpbd", "M4")])
def test_native_alternative_backend_contract_can_be_declared_without_a_gpu(backend, preset):
    cfg = EnvConfig(physics_backend=backend, contact_preset=preset, device="cuda:0")
    assert cfg.physics_backend == backend and cfg.contact_preset == preset


@pytest.mark.parametrize("preset", ["M1", "M2"])
def test_new_mjwarp_configs_record_native_mpc_profile(preset):
    config = EnvConfig(physics_backend="mjwarp", contact_preset=preset, device="cuda:0")
    assert config.numerics_profile == "native_mjwarp_v1"
    assert config.to_dict()["numerics_profile"] == "native_mjwarp_v1"
    assert EnvConfig(**config.to_dict()) == config


def test_case_bank_roundtrip_is_explicit_and_hash_stable(bank, tmp_path):
    path = tmp_path / "bank.json"
    bank.save(path)
    loaded = CaseBank.load(path)
    assert loaded == bank and loaded.sha256 == bank.sha256
    changed = replace(bank.cases[0], case_id="different")
    assert CaseBank((changed, *bank.cases[1:])).sha256 != bank.sha256
    with pytest.raises(ValueError):
        CaseBank((bank.cases[0], bank.cases[0]))
    with pytest.raises(ValueError):
        replace(bank.cases[0], goal_quaternion=(0, 0, 0, 0))


def test_cpu_m1_preset_reaches_actual_physics_instead_of_only_changing_a_label(bank):
    e = LeapVectorEnv(EnvConfig(num_envs=1, contact_preset="M1"), bank)
    try:
        e.reset_all([0])
        assert e.backend.model.opt.iterations == 200
        assert e.backend.model.opt.tolerance == pytest.approx(1e-10)
        np.testing.assert_allclose(e.backend.model.geom_solimp[:, :2], 0.9999)
        tr = e.step(torch.zeros(1, 16))
        assert torch.isfinite(tr.next_obs).all()
        assert e.describe()["physics"]["gpu_qualified"] is False
    finally:
        e.close()


def test_two_goals_are_independent_at_the_same_physical_state(bank):
    a = replace(bank.cases[0], case_id="identity", goal_quaternion=(1., 0., 0., 0.))
    b = replace(a, case_id="rotate", goal_quaternion=(0., 1., 0., 0.))
    e = LeapVectorEnv(EnvConfig(), CaseBank((a, b)))
    try:
        obs = e.reset_all([0, 1])
        assert torch.equal(obs[0, :41], obs[1, :41])
        assert not torch.equal(obs[0, 44:50], obs[1, 44:50])
        tr = e.step(torch.zeros(2, 16))
        assert tr.diagnostics["quat"][0] < tr.diagnostics["quat"][1]
        assert tr.diagnostics["raw_cost"][0] < tr.diagnostics["raw_cost"][1]
    finally:
        e.close()


def test_reset_preserves_other_world_hidden_state_rng_and_future_trajectory(env, bank):
    control = LeapVectorEnv(EnvConfig(), bank)
    try:
        control.reset_all([0, 1])
        action = torch.full((2, 16), 0.2)
        env.step(action)
        control.step(action)
        before = env.backend.data[1]
        snapshots = {key: getattr(before, key).copy() for key in
                     ("qpos", "qvel", "ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied")}
        other_rng = str(env._rng[1].bit_generator.state)
        env.reset_indices([0], [2])
        for key, value in snapshots.items():
            np.testing.assert_array_equal(getattr(env.backend.data[1], key), value)
        assert str(env._rng[1].bit_generator.state) == other_rng
        for _ in range(3):
            tr = env.step(action)
            ref = control.step(action)
            torch.testing.assert_close(tr.next_obs[1], ref.next_obs[1], rtol=0, atol=0)
        steps = env.steps.clone()
        assert env.reset_indices([]) is None
        assert torch.equal(env.steps, steps)
    finally:
        control.close()


def test_post_step_readout_matches_fresh_forward_without_mutating_physics(env):
    env.step(torch.zeros(2, 16))
    state = env.backend.data[0]
    snapshot = {key: getattr(state, key).copy() for key in ("qpos", "qvel", "qacc_warmstart", "ctrl")}
    frame = env.backend.read_frame()
    fresh = mujoco.MjData(env.backend.model)
    mujoco.mj_copyData(fresh, env.backend.model, state)
    mujoco.mj_forward(env.backend.model, fresh)
    np.testing.assert_allclose(frame.sites[0].numpy(), fresh.site_xpos, rtol=1e-6, atol=1e-7)
    for key, value in snapshot.items():
        np.testing.assert_array_equal(getattr(state, key), value)


def test_transition_storage_survives_future_steps_and_reset(env):
    tr = env.step(torch.zeros(2, 16))
    old_obs, old_reward = tr.next_obs.clone(), tr.reward.clone()
    env.step(torch.ones(2, 16))
    env.reset_indices([0], [3])
    assert torch.equal(tr.next_obs, old_obs) and torch.equal(tr.reward, old_reward)


def test_reset_indices_reject_fractional_ids_without_truncating(env):
    before = env.current_target
    for ids in ([0.7], [True], [0, 0], torch.tensor([0.5])):
        with pytest.raises(ValueError):
            env.reset_indices(ids)
    with pytest.raises(ValueError):
        env.reset_indices([0], [0.5])
    assert torch.equal(before, env.current_target)


def test_action_contract_and_actual_control(env):
    raw = torch.full((2, 16), 100.)
    before, raw_copy = env.current_target, raw.clone()
    command = env.codec.decode(raw, before)
    env.step(raw)
    assert torch.equal(raw, raw_copy)
    assert (env.current_target - before).abs().max() <= env.codec.delta + 1e-6
    np.testing.assert_allclose(env.backend.data[0].ctrl, command[0].numpy(), atol=0, rtol=0)
    steps = env.steps.clone()
    with pytest.raises(ValueError):
        env.step(torch.full((2, 16), float("nan")))
    with pytest.raises(ValueError):
        env.step_command(env.current_target + 1)
    assert torch.equal(env.steps, steps)


def test_final_step_success_precedes_deadline(bank):
    cfg = EnvConfig(num_envs=1, episode_steps=1, physics_dt=1e-6, control_dt=1e-6)
    c = bank.cases[0]
    c = replace(c, goal_position=c.qpos[16:19], goal_quaternion=c.qpos[19:23])
    e = LeapVectorEnv(cfg, CaseBank((c,)))
    try:
        e.reset_all([0])
        tr = e.step(torch.zeros(1, 16))
        assert tr.reasons == ("success",)
        assert tr.terminated.item() and not tr.truncated.item()
        assert tr.next_obs[0, -1] == 0
        with pytest.raises(RuntimeError, match="must be reset"):
            e.step(torch.zeros(1, 16))
    finally:
        e.close()


def test_deadline_is_true_finite_termination_and_drop_has_priority(bank):
    e = LeapVectorEnv(EnvConfig(num_envs=1, episode_steps=1), bank)
    try:
        e.reset_all([0])
        tr = e.step(torch.zeros(1, 16))
        assert tr.reasons == ("deadline",) and tr.terminated.item() and not tr.truncated.item()
        e.reset_all([0])
        e.backend.data[0].qpos[18] = -0.1
        tr = e.step(torch.zeros(1, 16))
        assert tr.reasons == ("physical_drop",) and tr.reward.item() == -1
    finally:
        e.close()


def test_numerical_fault_stops_environment_instead_of_becoming_drop(env):
    env.backend.data[0].qpos[0] = float("nan")
    with pytest.raises(NumericalTransitionError):
        env.observe()
    with pytest.raises(RuntimeError, match="faulted"):
        env.reset_all()


def test_mujoco_internal_recovery_warning_is_not_silently_accepted(env):
    env.backend.data[0].qvel[0] = float("nan")
    with pytest.raises(NumericalTransitionError, match="warning"):
        env.step(torch.zeros(2, 16))
    with pytest.raises(RuntimeError, match="faulted"):
        env.reset_all()


@pytest.mark.parametrize("steps_before", [0, 3, 7])
def test_failed_tail_cost_does_not_reward_early_drop(steps_before):
    cfg = EnvConfig(episode_steps=8)
    n = cfg.episode_steps - steps_before
    dropped = finite_reward(torch.tensor([0.]), torch.tensor([True]), torch.tensor([steps_before]), cfg)
    worst_survival = sum(float(finite_reward(torch.tensor([100.]), torch.tensor([False]),
                                            torch.tensor([k]), cfg))
                         for k in range(steps_before, cfg.episode_steps))
    assert dropped.item() == pytest.approx(-n / 8) == pytest.approx(worst_survival)


def test_named_cost_matches_existing_warp_kernel_on_same_state(env):
    import warp as wp
    from ContactModelStudy.Simulators.VectorizedMujoco import VectorizedMujoco
    env.reset_all([0, 0])
    env.step(torch.full((2, 16), 0.3))
    frame = env.backend.read_frame()
    env.task.setGoal(np.array(env.case_bank.cases[0].goal_quaternion))
    # Exercise the original task kernel with CPU Warp arrays, not another simulator.
    sim = object.__new__(VectorizedMujoco)
    sim._n_worlds = 2
    sim.config = SimpleNamespace(device="cpu")
    sim.d = SimpleNamespace(qpos=wp.array(frame.qpos.numpy(), dtype=wp.float32, device="cpu"),
                           qvel=wp.array(frame.qvel.numpy(), dtype=wp.float32, device="cpu"),
                           ctrl=wp.zeros((2, 16), dtype=wp.float32, device="cpu"),
                           site_xpos=wp.array(frame.sites.numpy(), dtype=wp.vec3, device="cpu"))
    # The legacy task omits launch(device=...). Its global default may be CUDA
    # even when the input/output arrays are CPU arrays, so bind it explicitly.
    with wp.ScopedDevice("cpu"):
        old_device = env.task.calcCosts(sim)
        wp.synchronize_device("cpu")
    old = old_device.numpy().copy()
    terms = task_terms(frame, env._goal_pos, env._goal_quat, env._home,
                       env.task.params["fallen_z"], env.task.indices, env._weights)
    np.testing.assert_allclose(terms["raw_cost"].numpy(), old, rtol=2e-6, atol=1e-6)


def test_gymnasium_single_world_contract(bank):
    from gymnasium.utils.env_checker import check_env
    e = LeapGymEnv(EnvConfig(num_envs=1), bank)
    try:
        check_env(e, skip_render_check=True)
    finally:
        e.close()


@pytest.mark.parametrize('operation', ['step', 'reset', 'read'])
def test_review_backend_exception_faults_environment(env, monkeypatch, operation):
    def fail(*args, **kwargs): raise RuntimeError('injected backend failure')
    if operation == 'step':
        monkeypatch.setattr(env.backend, 'step', fail)
        call = lambda: env.step(torch.zeros(2, 16))
    elif operation == 'reset':
        monkeypatch.setattr(env.backend, 'reset_worlds', fail)
        call = lambda: env.reset_indices([0], [0])
    else:
        monkeypatch.setattr(env.backend, 'read_frame', fail)
        call = env.observe
    with pytest.raises(RuntimeError, match='injected backend'): call()
    assert env._faulted
    with pytest.raises(RuntimeError, match='closed or faulted'): env.step(torch.zeros(2, 16))
