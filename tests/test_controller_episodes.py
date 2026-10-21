"""Shared controller dispatch and execution contracts; no PPO optimizer."""
from dataclasses import dataclass
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from ContactModelStudy.Controllers.EpisodeController import PolicyEpisodeController
from ContactModelStudy.Controllers.MPPIController import MPPIController
from ContactModelStudy.Drivers.ControllerEpisodes import run_controller_cases
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig, Frame, Transition
from ContactModelStudy.RL.Encoding import ActionCodec


@dataclass
class PlannerConfig:
    control_mode: str = 'ctrl_relative'


class TestEnvironment:
    __test__ = False
    def __init__(self, config, bank):
        self.config, self.bank, self.device = config, bank, 'cpu'
        self.codec = ActionCodec([-3.]*16, [3.]*16, rate=config.target_rate, control_dt=config.control_dt)
        self._target = torch.zeros(1, 16); self.k = 0
    def describe(self): return {'task_protocol': 'single_goal_finite_deadline_gamma1_v1'}
    @property
    def current_target(self): return self._target.clone()
    def reset_all(self, ids):
        self.case = self.bank.cases[ids[0]]; self.k = 0
        self._target = torch.tensor([self.case.control])
        return torch.zeros(1, 73)
    def read_frame(self):
        return Frame(torch.tensor([self.case.qpos]), torch.tensor([self.case.qvel]),
                     torch.zeros(1, 4, 3), torch.zeros(1, 6))
    def step_command(self, command):
        assert (command-self._target).abs().max() <= self.codec.delta+1e-6
        self._target.copy_(command); self.k += 1
        done = torch.tensor([[self.k == self.config.episode_steps]])
        return Transition(torch.zeros(1, 73), torch.tensor([[-.01]]), done,
            torch.zeros_like(done), ('deadline' if done.item() else 'running',),
            {'quat': torch.tensor([.25]), 'pos': torch.tensor([.001]), 'raw_cost': torch.tensor([1.])})
    def close(self): self.closed = True


class Planner:
    config = PlannerConfig()
    last_plan_ok = True
    def __init__(self): self.inputs = []; self.resets = 0
    def Reset(self): self.resets += 1
    def Plan(self, q, v, u):
        self.inputs.append((q.copy(), v.copy(), u.copy()))
        return u+.001


def test_rl_mppi_share_cases_clock_stopping_and_step_records(tmp_path):
    bank = CaseBank.development(count=2, seed=0, difficulty=8)
    config = EnvConfig(num_envs=1, episode_steps=2)
    plan = Planner(); goals = []; resources = SimpleNamespace(config=PlannerConfig(), N=4, Close=lambda: None)
    task = SimpleNamespace(params={}, setGoal=lambda q: goals.append(q.copy()))
    policy = SimpleNamespace(manifest={'observation_profile': 'sim_world73_dev_v1'}, act=lambda obs, previous: previous+.001)
    for name, factory in [('rl', lambda env: PolicyEpisodeController(policy)),
                          ('mppi', lambda env: MPPIController(plan, task, resources))]:
        output = tmp_path/name
        result = run_controller_cases(config, bank, output, factory, identity={'case_bank_sha256': bank.sha256},
                                      env_factory=TestEnvironment, save_steps=True)
        assert result['completed_cases'] == result['deadlines'] == 2 and result['new_PPO_updates'] == 0
        episodes = [json.loads(line) for line in (output/'episodes.jsonl').read_text().splitlines()]
        assert [r['case_id'] for r in episodes] == [c.case_id for c in bank.cases]
        assert all(r['control_steps'] == 2 and r['sim_seconds'] == .032 for r in episodes)
        trace = [json.loads(line) for line in (output/'steps.jsonl').read_text().splitlines()]
        assert len(trace) == 4 and len(trace[0]['qpos_after']) == 23
        assert np.allclose(trace[0]['command'], trace[2]['command'])  # episode history resets
    assert plan.resets == 2 and len(plan.inputs) == 4
    assert np.array_equal(goals[-1], bank.cases[-1].goal_quaternion)
    assert np.array_equal(task.params['target_pos'], bank.cases[-1].goal_position)
    assert np.allclose(plan.inputs[1][2], plan.inputs[0][2]+.001)


def test_mppi_projection_is_visible_and_bad_plan_stops(tmp_path):
    bank = CaseBank.development(count=1, seed=0, difficulty=8)
    env = TestEnvironment(EnvConfig(num_envs=1, episode_steps=2), bank); obs = env.reset_all([0])
    plan = Planner(); plan.Plan = lambda *args, **kw: np.full(16, 100.)
    ctrl = MPPIController(plan, SimpleNamespace(params={}, setGoal=lambda q: None),
                          SimpleNamespace(config=PlannerConfig(), N=4, Close=lambda: None))
    command = ctrl.act(env, obs)
    assert (command-env.current_target).abs().max() <= env.codec.delta+1e-6
    assert ctrl.last_command_info['projected_components'] == 16
    plan.last_plan_ok = False
    with pytest.raises(RuntimeError, match='valid plan'): ctrl.act(env, obs)


def test_historical_entry_routes_only_explicit_finite_protocol(monkeypatch):
    from ContactModelStudy.Drivers import run_episodes, run_controller_episodes
    assert run_episodes.parseArgs([]).protocol == 'legacy_mpc'
    assert run_episodes.parseArgs([]).ctrl_time_step == .064
    calls = []
    monkeypatch.setattr(run_controller_episodes, 'main', lambda argv: calls.append(argv))
    argv = ['--protocol', 'finite_case_bank', '--controller', 'rl']
    assert run_episodes.main(argv) == 0 and calls == [argv]
    with pytest.raises(SystemExit): run_episodes.parseArgs(['--protocol', 'finite_case_bank'])


def test_common_entry_rejects_legacy_defaults_and_mixed_controller_flags(tmp_path):
    from ContactModelStudy.Drivers.run_controller_episodes import main
    for extra in (['--controller', 'rl'], ['--controller', 'mppi', '--policy', 'unused'],
                  ['--controller', 'rl', '--policy', 'unused', '--settle', '1']):
        with pytest.raises(SystemExit): main(['--case-bank', 'unused', '--output', str(tmp_path/'run'), *extra])


def test_review_cleanup_failure_cannot_leave_completed_summary(tmp_path):
    bank = CaseBank.development(count=1, seed=0, difficulty=8)
    policy = SimpleNamespace(manifest={'observation_profile': 'sim_world73_dev_v1'}, act=lambda obs, previous: previous)
    envs = []
    def create_env(config, cases):
        env = TestEnvironment(config, cases); envs.append(env); return env
    class ClosingController(PolicyEpisodeController):
        def close(self): raise RuntimeError('controller close failed')
    with pytest.raises(RuntimeError, match='close failed'):
        run_controller_cases(EnvConfig(num_envs=1, episode_steps=1), bank, tmp_path/'run',
            lambda env: ClosingController(policy), identity={}, env_factory=create_env)
    result = json.loads((tmp_path/'run/summary.json').read_text())
    assert result['execution_status'] == 'failed'
    assert result['failure_reason'] == 'cleanup_failure'
    assert result['completed_cases'] == 1 and result['failed_cases'] == 0
    assert envs[0].closed


def test_review_cleanup_error_preserves_primary_failure(tmp_path):
    bank = CaseBank.development(count=1, seed=0, difficulty=8)
    class FailingController:
        def describe(self): return {'kind': 'test'}
        def reset(self, case): pass
        def act(self, env, obs): raise ValueError('primary action failure')
        def close(self): raise RuntimeError('secondary close failure')
    with pytest.raises(ValueError, match='primary action'):
        run_controller_cases(EnvConfig(num_envs=1, episode_steps=1), bank, tmp_path/'run',
            lambda env: FailingController(), identity={}, env_factory=TestEnvironment)
    result = json.loads((tmp_path/'run/summary.json').read_text())
    assert result['exception'] == 'ValueError'
    assert result['cleanup_errors'][0]['message'] == 'secondary close failure'


def test_review_cuda_decision_timer_waits_for_device_completion(tmp_path, monkeypatch):
    bank = CaseBank.development(count=1, seed=0, difficulty=8); events = []
    class DeviceTaggedEnv(TestEnvironment):
        def __init__(self, config, cases):
            super().__init__(config, cases); self.device = torch.device('cuda:0')
    class Controller:
        def describe(self): return {'kind': 'test'}
        def reset(self, case): pass
        def act(self, env, obs): events.append('act'); return env.current_target
        def close(self): pass
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: events.append('sync'))
    run_controller_cases(EnvConfig(num_envs=1, episode_steps=1), bank, tmp_path/'run',
        lambda env: Controller(), identity={}, env_factory=DeviceTaggedEnv)
    assert events == ['sync', 'act', 'sync']


def test_video_hook_is_shared_and_does_not_add_control_steps(tmp_path):
    from ContactModelStudy.Drivers.EvaluationVideo import VideoConfig
    bank = CaseBank.development(count=1, seed=0, difficulty=8)
    events = []
    class Video:
        def __init__(self, settings, output, env): events.append('open')
        def begin(self, case, index): events.append(('begin', case.case_id))
        def capture(self, step, sim_time): events.append(('frame', step, sim_time))
        def finish(self): events.append('finish'); return {'path': 'test.mp4'}
        def close(self): events.append('close')
    policy = SimpleNamespace(manifest={'observation_profile': 'sim_world73_dev_v1'}, act=lambda obs, previous: previous)
    result = run_controller_cases(EnvConfig(num_envs=1, episode_steps=2), bank, tmp_path/'run',
        lambda env: PolicyEpisodeController(policy), identity={}, env_factory=TestEnvironment,
        video=VideoConfig(), video_factory=Video)
    assert result['completed_cases'] == 1
    assert [e for e in events if isinstance(e, tuple) and e[0] == 'frame'] == [('frame', 1, .016), ('frame', 2, .032)]
    assert events[-2:] == ['finish', 'close']
    record = json.loads((tmp_path/'run/episodes.jsonl').read_text())
    assert record['control_steps'] == 2 and record['video']['path'] == 'test.mp4'


def test_video_failure_is_not_counted_as_a_task_failure_and_closes_resources(tmp_path):
    from ContactModelStudy.Drivers.EvaluationVideo import VideoConfig
    bank = CaseBank.development(count=1, seed=0, difficulty=8)
    events = []
    class Video:
        def __init__(self, *args): pass
        def begin(self, *args): pass
        def capture(self, *args): raise RuntimeError('encoder failure')
        def close(self): events.append('video_closed')
    policy = SimpleNamespace(manifest={'observation_profile': 'sim_world73_dev_v1'}, act=lambda obs, previous: previous)
    with pytest.raises(RuntimeError, match='encoder failure'):
        run_controller_cases(EnvConfig(num_envs=1, episode_steps=2), bank, tmp_path/'run',
            lambda env: PolicyEpisodeController(policy), identity={}, env_factory=TestEnvironment,
            video=VideoConfig(), video_factory=Video)
    result = json.loads((tmp_path/'run/summary.json').read_text())
    assert result['failure_reason'] == 'video_failure' and result['completed_cases'] == 0
    assert result['outcomes']['video_failure'] == 1 and events == ['video_closed']


def test_video_settings_require_the_explicit_video_flag(tmp_path):
    from ContactModelStudy.Drivers.run_controller_episodes import main
    with pytest.raises(SystemExit):
        main(['--controller', 'rl', '--policy', 'unused', '--case-bank', 'unused',
              '--output', str(tmp_path/'run'), '--fps', '30'])
