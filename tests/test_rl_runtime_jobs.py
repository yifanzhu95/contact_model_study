"""Single-device preflight and independent-job routing, with zero optimization."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from ContactModelStudy.Drivers import check_rl_runtime, run_rl_job
from ContactModelStudy.RL import Preflight
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig
from ContactModelStudy.RL.SkrlPPO import PPOConfig


@pytest.fixture
def job_files(tmp_path):
    config, ppo = EnvConfig(num_envs=2, episode_steps=2), PPOConfig(steps_per_update=2)
    bank = CaseBank.development(count=2, seed=4, difficulty=8)
    bank.save(tmp_path/'cases.json')
    (tmp_path/'config.json').write_text(json.dumps({'environment': config.to_dict(), 'ppo': asdict(ppo)}))
    plan = {'schema': 'rl_independent_jobs_v1', 'jobs': [
        {'id': name, 'config': 'config.json', 'case_bank': 'cases.json', 'updates': 3,
         'max_transitions': 10, 'max_seconds': 100, 'stop_margin_seconds': 10}
        for name in ('first', 'second')]}
    (tmp_path/'jobs.json').write_text(json.dumps(plan))
    return tmp_path, config, ppo, bank, plan


def test_preflight_default_checks_scene_without_creating_a_physics_backend(job_files, monkeypatch):
    tmp, config, ppo, bank, _ = job_files
    def forbidden(*args):
        raise AssertionError('Default preflight must not allocate simulation environments')
    monkeypatch.setattr(Preflight, 'LeapVectorEnv', forbidden)
    report = Preflight.runtime_preflight(config, ppo, bank)
    assert report['scene_layout'] == [23, 22, 16]
    assert report['physics_steps'] == report['training_transitions'] == report['ppo_updates'] == 0
    assert not report['backend_reset_checked']


def test_preflight_optional_reset_is_bounded_and_has_no_steps(job_files):
    _, config, ppo, bank, _ = job_files
    report = Preflight.runtime_preflight(config, ppo, bank, check_backend=True)
    assert report['backend_reset_checked'] and report['probe_num_envs'] == 2
    assert report['physics_steps'] == report['ppo_updates'] == 0


def test_wrong_dependency_version_fails_before_backend(job_files, monkeypatch):
    _, config, ppo, bank, _ = job_files
    actual = Preflight.version
    monkeypatch.setattr(Preflight, 'version', lambda name: '0.0.0' if name == 'skrl' else actual(name))
    with pytest.raises(RuntimeError, match='skrl requires audited version'):
        Preflight.runtime_preflight(config, ppo, bank)


def test_failed_preflight_writes_a_fresh_failure_report_and_nonzero_exit(tmp_path):
    output = tmp_path/'check.json'
    with pytest.raises(SystemExit) as exc:
        check_rl_runtime.main(['--config', str(tmp_path/'missing.json'), '--output', str(output)])
    assert exc.value.code == 2
    assert json.loads(output.read_text())['execution_status'] == 'failed'
    frozen = output.read_bytes()
    with pytest.raises(SystemExit):
        check_rl_runtime.main(['--config', str(tmp_path/'missing.json'), '--output', str(output)])
    assert output.read_bytes() == frozen


def test_job_plan_is_nonexecuting_and_outputs_do_not_overlap(job_files, monkeypatch):
    tmp, _, _, _, _ = job_files
    def forbidden(*args, **kwargs):
        raise AssertionError('Plan-only must not preflight a GPU or train')
    monkeypatch.setattr(run_rl_job, 'runtime_preflight', forbidden)
    monkeypatch.setattr(run_rl_job.train_rl, 'main', forbidden)
    first = run_rl_job.main(['--plan', str(tmp/'jobs.json'), '--job-index', '0', '--output-root', str(tmp/'outputs')])
    second = run_rl_job.main(['--plan', str(tmp/'jobs.json'), '--job-id', 'second', '--output-root', str(tmp/'outputs')])
    assert first['mode'] == second['mode'] == 'plan_only'
    assert first['job_directory'] != second['job_directory']
    assert first['budget']['effective_transition_limit'] == 8
    assert not (tmp/'outputs').exists()


def test_execute_routes_only_selected_job_after_preflight(job_files, monkeypatch):
    tmp, _, _, _, _ = job_files
    events = []
    def check(*args):
        events.append('preflight')
        return {'execution_status': 'passed'}
    def train(args):
        events.append(('train', args))
        return {'execution_status': 'completed', 'ppo_updates': 0}
    monkeypatch.setattr(run_rl_job, 'runtime_preflight', check)
    monkeypatch.setattr(run_rl_job.train_rl, 'main', train)
    result = run_rl_job.main(['--plan', str(tmp/'jobs.json'), '--job-id', 'second',
                             '--output-root', str(tmp/'outputs'), '--execute'])
    assert result['ppo_updates'] == 0 and events[0] == 'preflight'
    assert str(tmp/'outputs/second/run') in events[1][1]
    assert (tmp/'outputs/second/job.json').exists() and not (tmp/'outputs/first').exists()
    with pytest.raises(FileExistsError):
        run_rl_job.main(['--plan', str(tmp/'jobs.json'), '--job-id', 'second',
                        '--output-root', str(tmp/'outputs'), '--execute'])


def test_failed_preflight_prevents_training_and_preserves_job(job_files, monkeypatch):
    tmp, _, _, _, _ = job_files
    def fail(*args):
        raise RuntimeError('incompatible runtime')
    monkeypatch.setattr(run_rl_job, 'runtime_preflight', fail)
    monkeypatch.setattr(run_rl_job.train_rl, 'main', lambda *args: pytest.fail('Training after failed preflight'))
    with pytest.raises(RuntimeError, match='incompatible'):
        run_rl_job.main(['--plan', str(tmp/'jobs.json'), '--job-index', '0',
                        '--output-root', str(tmp/'outputs'), '--execute'])
    report = json.loads((tmp/'outputs/first/job_summary.json').read_text())
    assert report['execution_status'] == 'failed' and not report['automatic_retry']


def test_job_budget_includes_preflight_and_stops_without_training(job_files, monkeypatch):
    tmp, _, _, _, _ = job_files
    now = [0.]
    monkeypatch.setattr(run_rl_job.time, 'monotonic', lambda: now[0])
    def preflight(*args):
        now[0] = 95.
        return {'execution_status': 'passed'}
    monkeypatch.setattr(run_rl_job, 'runtime_preflight', preflight)
    monkeypatch.setattr(run_rl_job.train_rl, 'main', lambda *args: pytest.fail('Budget expired during preflight'))
    report = run_rl_job.main(['--plan', str(tmp/'jobs.json'), '--job-index', '0',
                             '--output-root', str(tmp/'outputs'), '--execute'])
    assert report['stop_reason'] == 'wall_time_budget_before_training'
    assert report['transitions'] == report['ppo_updates'] == 0
    assert not (tmp/'outputs/first/run').exists()


@pytest.mark.parametrize('change', [
    {'id': '../escape'}, {'updates': True}, {'max_seconds': None}, {'max_transitions': 1},
    {'checkpoint_interval': -1}, {'extra': 'ignored'}])
def test_invalid_job_cannot_create_output(job_files, change):
    tmp, _, _, _, plan = job_files
    plan['jobs'][0].update(change)
    (tmp/'jobs.json').write_text(json.dumps(plan))
    with pytest.raises((ValueError, TypeError)):
        run_rl_job.plan_job(tmp/'jobs.json', tmp/'outputs', job_index=0)
    assert not (tmp/'outputs').exists()
