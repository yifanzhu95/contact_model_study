"""Small real-GPU checks of Kamino reset/readout; no PPO or training is invoked."""
from dataclasses import replace
from importlib.util import find_spec
import json

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig, NumericalTransitionError
from ContactModelStudy.RL.KaminoBackend import array_fields, body_origin_twist
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available() or find_spec("newton") is None,
    reason="Needs explicit CUDA Torch and Newton 1.6 runtime")]


def make_env():
    return LeapVectorEnv(EnvConfig(physics_backend="kamino", contact_preset="M5", device="cuda:0"),
                         CaseBank.development(count=4, seed=9, difficulty=8))


def selected_state(env, world):
    """Snapshot physical state/control plus the native solver's persistent caches."""
    sim, output = env.backend.sim, {}
    for label, obj in (("state0", sim.state_0), ("state1", sim.state_1), ("control", sim.control)):
        for name, arr in array_fields(obj):
            output[f"{label}.{name}"] = arr.numpy().reshape(env.num_envs, arr.shape[0]//env.num_envs, *arr.numpy().shape[1:])[world].copy()
    for name in ("qpos_wp", "qvel_wp", "ctrl_wp", "site_xpos_wp", "_time_wp", "_contact_peak"):
        output[name] = getattr(sim, name).numpy()[world].copy()
    impl = sim.solver._solver_kamino
    for name in ("time", "steps"):
        output[f"internal.{name}"] = getattr(impl._data.time, name).numpy()[world].copy()
    fd, problem = impl._solver_fd._data, impl._problem_fd.data
    start, length = int(problem.vio.numpy()[world]), int(problem.maxdim.numpy()[world])
    for name in ("lambdas", "v_plus"):
        output[f"solution.{name}"] = getattr(fd.solution, name).numpy()[start:start+length].copy()
    for name in ("status", "penalty", "linear_solver_atol"):
        output[f"fd.{name}"] = getattr(fd, name).numpy()[world].copy()
    for name, arr in vars(fd.state).items():
        if isinstance(arr, wp.array):
            output[f"fd_state.{name}"] = arr.numpy().copy()  # Native masked reset leaves this scratch state intact.
    for label, cache in (("contacts", impl._ws_contacts._cache), ("limits", impl._ws_limits._cache)):
        ids = np.flatnonzero(cache.wid.numpy() == world)
        for name, arr in vars(cache).items():
            if isinstance(arr, wp.array) and arr.shape[0] == cache.wid.shape[0]:
                output[f"warm_{label}.{name}"] = arr.numpy()[ids].copy()
    return output


@pytest.mark.parametrize("reset_id", [0, 1])
def test_native_mask_reset_preserves_other_state_controls_caches_and_future(reset_id, record_property):
    env, control, replay = make_env(), make_env(), make_env()
    other = 1-reset_id
    try:
        action = torch.full((2, 16), 0.2, device=env.device)
        for e in (env, control, replay):
            e.reset_all([0, 1])
            e.step(action)
        before = selected_state(env, other)
        assert np.any(before["solution.lambdas"] != 0), "Must exercise an active solver cache"
        assert before["warm_contacts.key"].size > 0, "Must exercise contact warm starts"
        episode = (env.steps[other].clone(), env._goal_quat[other].clone(), env.current_target[other],
                   env.case_ids[other], str(env._rng[other].bit_generator.state))
        env.reset_indices([reset_id], [2])
        after = selected_state(env, other)
        for key in before:
            np.testing.assert_array_equal(after[key], before[key], err_msg=key)
        assert torch.equal(env.steps[other], episode[0])
        assert torch.equal(env._goal_quat[other], episode[1])
        assert torch.equal(env.current_target[other], episode[2])
        assert env.case_ids[other] == episode[3] and str(env._rng[other].bit_generator.state) == episode[4]
        np.testing.assert_array_equal(env.backend._reset_mask.cpu().numpy(), [reset_id == 0, reset_id == 1, False])
        assert wp.to_torch(env.backend.sim._time_wp)[reset_id] == 0
        assert wp.to_torch(env.backend.sim.solver._solver_kamino._data.time.steps)[reset_id] == 0
        measured = []
        for step in range(3):
            env.step(action)
            control.step(action)
            replay.step(action)
            frames = [e.backend.read_frame() for e in (env, control, replay)]
            for key in ("qpos", "qvel", "sites", "object_twist"):
                a, b, c = [getattr(f, key)[other] for f in frames]
                noise = (b-c).abs().max()
                rounding = torch.finfo(b.dtype).eps * b.abs().max().clamp_min(1)
                measured.append({"step": step + 1, "field": key, "reset_difference": (a-b).abs().max().item(),
                                 "repeat_difference": noise.item(), "bound": (3*noise + rounding).item()})
                assert (a-b).abs().max() <= 3*noise + rounding, (key, noise.item(), (a-b).abs().max().item())
        record_property("reset_evidence", json.dumps({"reset_id": reset_id,
                        "unchanged_array_checks": len(before), "future": measured}))
    finally:
        for e in (env, control, replay):
            e.close()


def test_empty_and_full_reset_and_selected_ping_pong_buffers():
    env = make_env()
    try:
        env.reset_all([0, 1])
        env.step(torch.full((2, 16), 0.2, device=env.device))
        before = selected_state(env, 1)
        assert env.reset_indices([]) is None
        env.backend.reset_worlds([], [])
        after = selected_state(env, 1)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        env.reset_all([2, 3])
        sim = env.backend.sim
        assert torch.equal(env.steps, torch.zeros_like(env.steps))
        assert wp.to_torch(sim._time_wp).count_nonzero() == 0
        first, second = dict(array_fields(sim.state_0)), dict(array_fields(sim.state_1))
        for name, arr in first.items():
            np.testing.assert_array_equal(arr.numpy(), second[name].numpy(), err_msg=name)
        local = wp.to_torch(sim._targets).long()
        torch.testing.assert_close(env.backend._world_rows(sim.control.joint_target_q)[:, local], env.current_target)
        assert env.describe()["physics"]["solver_config"]["padmm_max_iterations"] == 200
    finally:
        env.close()


def test_nonzero_velocity_rotated_reset_matches_mujoco_coordinates_and_fk(record_property):
    env = make_env()
    try:
        base = env.case_bank.cases[0]
        q = list(base.qpos)
        q[-4:] = [np.cos(0.4), 0, np.sin(0.4), 0]
        velocity = tuple(np.linspace(-0.1, 0.15, 22))
        moving = replace(base, case_id="rotated-moving", qpos=tuple(q), qvel=velocity)
        env.backend.reset_worlds([0, 1], [moving, base])
        frame = env.backend.read_frame()
        np.testing.assert_allclose(frame.qpos[0].cpu(), q, atol=1e-5, rtol=0)
        np.testing.assert_allclose(frame.qvel[0].cpu(), velocity, atol=1e-5, rtol=0)
        data = mujoco.MjData(env.backend.model)
        data.qpos[:], data.qvel[:] = q, velocity
        mujoco.mj_forward(env.backend.model, data)
        twist = np.empty(6)
        mujoco.mj_objectVelocity(env.backend.model, data, mujoco.mjtObj.mjOBJ_XBODY,
                                int(env.task.indices[4]), twist, 0)
        np.testing.assert_allclose(frame.sites[0].cpu(), data.site_xpos, atol=1e-5, rtol=0)
        np.testing.assert_allclose(frame.object_twist[0].cpu(), np.r_[twist[3:], twist[:3]], atol=1e-5, rtol=0)
        record_property("reset_readout_errors", json.dumps({
            "qpos_max_abs": float(np.max(np.abs(frame.qpos[0].cpu().numpy()-q))),
            "qvel_max_abs": float(np.max(np.abs(frame.qvel[0].cpu().numpy()-velocity))),
            "sites_max_abs_m": float(np.max(np.abs(frame.sites[0].cpu().numpy()-data.site_xpos))),
            "twist_max_abs": float(np.max(np.abs(frame.object_twist[0].cpu().numpy()-np.r_[twist[3:], twist[:3]])))}))
    finally:
        env.close()


def actual_sites_numpy(sim):
    from scipy.spatial.transform import Rotation
    poses, shapes = sim.state_0.body_q.numpy(), sim.model.shape_transform.numpy()
    parents, starts, local = sim.model.shape_body.numpy(), sim.model.shape_world_start.numpy(), sim._site_local.numpy()
    out = np.zeros((sim.N, sim.mjm.nsite, 3))
    for w in range(sim.N):
        for sid, index in enumerate(local):
            shape = starts[w] + index
            parent = parents[shape]
            point = shapes[shape, :3]
            out[w, sid] = (poses[parent, :3] + Rotation.from_quat(poses[parent, 3:]).apply(point)
                           if parent >= 0 else point)
    return out


def test_post_step_readout_uses_actual_bodies_and_is_read_only(record_property):
    env = make_env()
    try:
        env.reset_all([0, 1])
        env.step(torch.full((2, 16), 0.2, device=env.device))
        sim = env.backend.sim
        before = [selected_state(env, w) for w in range(2)]
        frame = env.backend.read_frame()
        np.testing.assert_allclose(frame.sites.cpu(), actual_sites_numpy(sim), atol=2e-7, rtol=2e-6)
        record_property("post_step_sites_max_abs_m", float(np.max(np.abs(frame.sites.cpu().numpy()-actual_sites_numpy(sim)))))
        for w in range(2):
            after = selected_state(env, w)
            for key in before[w]:
                np.testing.assert_array_equal(after[key], before[w][key], err_msg=key)
        # Disturb a fingertip body only, leaving joint q unchanged. A reduced
        # FK readout would hide this difference; actual-body readout must see it.
        old_joints = wp.to_torch(sim.state_0.joint_q).clone()
        shape = int(sim._site_local.numpy()[int(env.task.indices[5])])
        body = int(sim.model.shape_body.numpy()[shape])
        wp.to_torch(sim.state_0.body_q)[body, 0] += 0.001
        changed = env.backend.read_frame()
        torch.testing.assert_close(changed.sites[0, int(env.task.indices[5]), 0],
                                   frame.sites[0, int(env.task.indices[5]), 0] + 0.001)
        assert torch.equal(wp.to_torch(sim.state_0.joint_q), old_joints)
    finally:
        env.close()


def test_world_origin_twist_accounts_for_rotation_and_nonzero_com_offset():
    # 90-degree z rotation: local COM (0.1,0,0) -> world (0,0.1,0).
    # omega=(0,0,2), v_com=(1,2,3) => v_origin=(1.2,2,3).
    pose = torch.tensor([[0, 0, 0, 0, 0, np.sqrt(0.5), np.sqrt(0.5)]], dtype=torch.float32)
    twist = body_origin_twist(pose, torch.tensor([[1., 2., 3., 0., 0., 2.]]), torch.tensor([[0.1, 0., 0.]]))
    torch.testing.assert_close(twist, torch.tensor([[1.2, 2., 3., 0., 0., 2.]]))


def test_frame_and_transition_storage_survive_future_motion_and_reset():
    env = make_env()
    try:
        env.reset_all([0, 1])
        frame = env.backend.read_frame()
        saved = {key: getattr(frame, key).clone() for key in ("qpos", "qvel", "sites", "object_twist")}
        transition = env.step(torch.zeros(2, 16, device=env.device))
        obs, reward = transition.next_obs.clone(), transition.reward.clone()
        env.reset_indices([0], [2])
        env.step(torch.zeros(2, 16, device=env.device))
        for key in saved:
            assert torch.equal(getattr(frame, key), saved[key])
        assert torch.equal(transition.next_obs, obs) and torch.equal(transition.reward, reward)
    finally:
        env.close()


def test_fk_reset_failure_faults_environment_and_prevents_next_action(monkeypatch):
    env = make_env()
    try:
        env.reset_all([0, 1])
        original = env.backend.sim.solver.reset
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            kwargs["success_mask"].zero_()
        monkeypatch.setattr(env.backend.sim.solver, "reset", fail)
        with pytest.raises(NumericalTransitionError) as exc:
            env.reset_indices([0], [2])
        assert exc.value.frame.qpos.shape == (2, 23)
        with pytest.raises(RuntimeError, match="faulted"):
            env.step(torch.zeros(2, 16, device=env.device))
    finally:
        env.close()
