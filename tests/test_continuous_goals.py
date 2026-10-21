"""Goal continuation, physical-state preservation, clocks and honest accounting."""
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ContactModelStudy.Controllers.EpisodeController import PolicyEpisodeController
from ContactModelStudy.Controllers.MPPIController import MPPIController
from ContactModelStudy.Drivers.ContinuousGoals import GoalSequenceConfig, prepare_goal_sequences
from ContactModelStudy.Drivers.ControllerEpisodes import run_controller_cases
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv


def fixture_case(task='cube_reorient', steps=4, difficulty=0):
    config = EnvConfig(num_envs=1, physics_dt=1e-6, control_dt=1e-6,
                       episode_steps=steps, goal_difficulty=difficulty, task_name=task)
    bank = CaseBank.development(count=1, seed=0, difficulty=0, task_name=task)
    case = bank.cases[0]
    return config, CaseBank((replace(case, goal_position=case.qpos[16:19],
                                    goal_quaternion=case.qpos[19:23]),), task_name=task)


def hold_controller(env):
    return PolicyEpisodeController(SimpleNamespace(manifest={'observation_profile': 'sim_world73_dev_v1'},
                                                   act=lambda obs, target: target))


@pytest.mark.parametrize('task', ['cube_reorient', 'duck_reorient', 'ball_reorient'])
def test_goal_change_preserves_real_physics_history_commands_rng_and_clock(task):
    config, bank = fixture_case(task)
    env = LeapVectorEnv(config, bank)
    try:
        with pytest.raises(RuntimeError, match='successful'):
            env.advance_evaluation_goal(bank.cases[0])
        env.reset_all([0])
        tr = env.step_command(env.current_target)
        assert tr.reasons == ('success',)
        state = env.backend.data[0]
        snapshots = {k: getattr(state, k).copy() for k in
                     ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'qfrc_applied', 'xfrc_applied')}
        target, steps, rng, domains = env.current_target, env.steps.clone(), str(env._rng[0].bit_generator.state), dict(env.reset_domains)
        changed = replace(bank.cases[0], goal_quaternion=(0., 1., 0., 0.))
        obs = env.advance_evaluation_goal(changed)
        for key, value in snapshots.items():
            np.testing.assert_array_equal(getattr(state, key), value)
        assert torch.equal(env.current_target, target) and torch.equal(env.steps, steps)
        assert str(env._rng[0].bit_generator.state) == rng and env.reset_domains == domains
        assert torch.equal(obs[:, :44], tr.next_obs[:, :44])
        assert not torch.equal(obs[:, 44:50], tr.next_obs[:, 44:50])
        assert torch.equal(obs[:, 50:], tr.next_obs[:, 50:])
        with pytest.raises(RuntimeError, match='successful'):
            env.advance_evaluation_goal(changed)
        assert env.step_command(env.current_target).reasons == ('running',)
    finally:
        env.close()


@pytest.mark.parametrize('ending', ['success_at_deadline', 'deadline', 'drop', 'running'])
def test_continuation_rejects_non_success_or_no_remaining_time(ending):
    config, bank = fixture_case(steps=1 if ending in ('success_at_deadline', 'deadline') else 4)
    if ending in ('deadline', 'running'):
        bank = CaseBank((replace(bank.cases[0], goal_quaternion=(0., 1., 0., 0.)),), task_name=bank.task_name)
    env = LeapVectorEnv(config, bank)
    try:
        env.reset_all([0])
        if ending == 'drop': env.backend.data[0].qpos[18] = -.1
        tr = env.step_command(env.current_target)
        assert tr.reasons[0] == {'success_at_deadline': 'success', 'deadline': 'deadline',
                                 'drop': 'physical_drop', 'running': 'running'}[ending]
        with pytest.raises(RuntimeError, match='successful'):
            env.advance_evaluation_goal(bank.cases[0])
    finally:
        env.close()


def test_sequence_is_precomputed_and_independent_of_training_seed_and_model():
    config, bank = fixture_case(difficulty=8)
    first = prepare_goal_sequences(config, bank, GoalSequenceConfig(4, 9))
    second = prepare_goal_sequences(replace(config, seed=900, physics_backend='kamino',
                                           contact_preset='M5', device='cuda:0'), bank, GoalSequenceConfig(4, 9))
    assert first == second
    assert first[0][0][0] == bank.cases[0]
    assert first[0][0][1].goal_quaternion != first[0][0][0].goal_quaternion


@pytest.mark.parametrize('controller_kind', ['rl', 'mppi'])
def test_shared_driver_completes_sequence_without_resetting_state_or_command(tmp_path, controller_kind, monkeypatch):
    config, bank = fixture_case()
    # Deliberately nearby targets exercise several real success boundaries
    # without requiring a trained actor. This is not rotation-learning evidence.
    from ContactModelStudy.Tasks.LeapReorient import LeapReorient, axisAngleQuat, quatMul
    monkeypatch.setattr(LeapReorient, 'sampleNewGoal', lambda self, current_goal=None:
                        quatMul(np.asarray(current_goal), axisAngleQuat(np.array([0., 0., 1.]), np.deg2rad(5))))
    observations, environments, plans = [], [], []
    def create_env(config, bank):
        env = LeapVectorEnv(config, bank)
        reset = env.backend.reset_worlds
        env.reset_calls = 0
        def counted_reset(*args):
            env.reset_calls += 1
            return reset(*args)
        env.backend.reset_worlds = counted_reset
        environments.append(env)
        return env
    @dataclass
    class Config: control_mode: str = 'ctrl_relative'
    class Planner:
        config = Config()
        last_plan_ok = True
        resets = 0
        def Reset(self): self.resets += 1
        def Plan(self, qpos, qvel, u): plans.append(u.copy()); return u+1e-7
    planner = Planner()
    def create_controller(env):
        if controller_kind == 'mppi':
            return MPPIController(planner, SimpleNamespace(params={}, setGoal=lambda goal: None),
                                  SimpleNamespace(config=Config(), N=2, Close=lambda: None))
        def act(obs, previous):
            observations.append((obs.clone(), previous.clone()))
            return previous+1e-7
        return PolicyEpisodeController(SimpleNamespace(manifest={'observation_profile': 'sim_world73_dev_v1'}, act=act))
    result = run_controller_cases(config, bank, tmp_path/'run', create_controller, identity={},
                                  env_factory=create_env, goals=GoalSequenceConfig(3), save_steps=True)
    assert result['successes'] == result['sequence_successes'] == 1
    assert result['goals_reached_completed'] == 3 and result['new_PPO_updates'] == 0
    record = json.loads((tmp_path/'run/episodes.jsonl').read_text())
    assert record['control_steps'] == 3 and record['sim_seconds'] == pytest.approx(3e-6)
    assert record['goals_reached'] == record['goals_attempted'] == 3
    assert environments[0].reset_calls == 1
    trace = [json.loads(line) for line in (tmp_path/'run/steps.jsonl').read_text().splitlines()]
    assert [r['goal_index'] for r in trace] == [0, 1, 2]
    np.testing.assert_allclose(trace[1]['command'], np.array(trace[0]['command'])+1e-7, rtol=1e-6, atol=1e-8)
    if controller_kind == 'mppi': assert planner.resets == 3
    else:
        assert [float(obs[0, -1]) for obs, _ in observations] == [1., .75, .5]


@pytest.mark.parametrize('steps', [1, 3])
def test_first_goal_success_is_not_sequence_success_or_extra_time(tmp_path, steps):
    config, bank = fixture_case(steps=steps, difficulty=9)
    result = run_controller_cases(config, bank, tmp_path/'run', hold_controller, identity={},
                                  goals=GoalSequenceConfig(3))
    record = json.loads((tmp_path/'run/episodes.jsonl').read_text())
    assert result['successes'] == 0 and result['deadlines'] == 1
    assert record['goals_reached'] == 1 and not record['sequence_complete']
    assert record['control_steps'] == steps
    assert record['goals_attempted'] == (1 if steps == 1 else 2)
    assert record['goal_results'][0]['reason'] == 'success'


def test_failure_after_a_reached_goal_retains_progress_without_claiming_sequence_success(tmp_path):
    config, bank = fixture_case()
    class FailingController(PolicyEpisodeController):
        def set_goal(self, case): raise RuntimeError('injected goal update failure')
    with pytest.raises(RuntimeError, match='goal update'):
        run_controller_cases(config, bank, tmp_path/'run',
            lambda env: FailingController(hold_controller(env).policy), identity={}, goals=GoalSequenceConfig(3))
    result = json.loads((tmp_path/'run/summary.json').read_text())
    assert result['execution_status'] == 'failed' and result['successes'] == 0
    assert result['failed_cases'] == 1 and result['active_sequence_progress']['goals_reached'] == 1
    assert '"event": "success"' in (tmp_path/'run/goals.jsonl').read_text()


def test_drop_after_first_goal_ends_sequence_as_drop(tmp_path):
    config, bank = fixture_case()
    def controller(env):
        class DroppingController(PolicyEpisodeController):
            def set_goal(self, case): env.backend.data[0].qpos[18] = -.1
        return DroppingController(hold_controller(env).policy)
    result = run_controller_cases(config, bank, tmp_path/'run', controller, identity={}, goals=GoalSequenceConfig(3))
    record = json.loads((tmp_path/'run/episodes.jsonl').read_text())
    assert result['successes'] == 0 and result['physical_drops'] == 1
    assert record['goals_reached'] == 1 and record['control_steps'] == 2
    assert [g['reason'] for g in record['goal_results']] == ['success', 'physical_drop']


@pytest.mark.parametrize('args', [['--goals-per-episode', '0'], ['--goals-per-episode', '-1'], ['--goal-seed', '2']])
def test_invalid_cli_goal_options_fail_before_loading_policy(tmp_path, args):
    from ContactModelStudy.Drivers.evaluate_rl import main
    with pytest.raises(SystemExit):
        main(['--policy', 'missing', '--case-bank', 'missing', '--output', str(tmp_path/'run'), *args])
