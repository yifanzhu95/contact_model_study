"""Run orchestration tests with synthetic transitions: zero PPO optimization."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig, Transition
from ContactModelStudy.RL.Policy import Actor, export_actor
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv
from ContactModelStudy.RL.RunIO import (evaluation_config, checkpoint_metadata, save_run_checkpoint, sha,
                                      recorded_environment_config, validate_resume_request)
from ContactModelStudy.RL.SkrlPPO import PPOConfig, restore_training_checkpoint, save_training_checkpoint
from ContactModelStudy.Drivers import evaluate_rl, evaluate_rl_matrix, train_rl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
    config = EnvConfig(num_envs=2, episode_steps=2)
    bank = CaseBank.development(count=2, seed=0, difficulty=8)
    env = LeapVectorEnv(config, bank)
    try: manifest = env.describe()
    finally: env.close()
    bank.save(tmp_path/'cases.json')
    actor = Actor(); export_actor(actor, manifest, tmp_path/'actor.pt')
    return tmp_path, manifest, bank


class FakeEnv:
    """No simulator: enough owned state for I/O and failure-accounting tests."""
    manifest = None

    def __init__(self, config, bank):
        self.config, self.case_bank = config, bank
        self.num_envs, self.device = config.num_envs, torch.device('cpu')
        self._rng = [np.random.default_rng(100+i) for i in range(self.num_envs)]
        self.obs = torch.zeros(self.num_envs, 73); self.obs[:, 34] = 1.
        self._target = torch.tensor([bank.cases[0].control]*self.num_envs)
        self.k = 0; self.closed = False

    def describe(self):
        value = deepcopy(self.manifest)
        value.update(config=self.config.to_dict(), case_bank_sha256=self.case_bank.sha256)
        return value

    @property
    def current_target(self): return self._target.clone()
    def reset_all(self, ids=None): self.k = 0; return self.obs.clone()
    def reset_indices(self, ids): self.k = 0
    def observe(self): return self.obs.clone()
    def close(self): self.closed = True

    def step_command(self, command):
        self.k += 1; self._target.copy_(command)
        done = torch.full((self.num_envs, 1), self.k >= self.config.episode_steps)
        return Transition(self.obs.clone(), torch.full((self.num_envs, 1), -.01), done,
            torch.zeros_like(done), ('deadline' if done.all() else 'running',)*self.num_envs,
            {'raw_cost': torch.full((self.num_envs,), 5.), 'quat': torch.full((self.num_envs,), .25),
             'pos': torch.full((self.num_envs,), .001)})


@dataclass
class FakeCfg:
    rollouts: int


class FakeAgent:
    def __init__(self, actor, rollouts):
        self.actor, self.cfg = actor, FakeCfg(rollouts)
        self.memory = SimpleNamespace(reset=lambda: None)
        self.optimizer_state = {'step': torch.tensor(17)}

    def save(self, path): torch.save({'actor': self.actor.state_dict(), 'optimizer': self.optimizer_state}, path)
    def load(self, path):
        value = torch.load(path, weights_only=True)
        self.actor.load_state_dict(value['actor']); self.optimizer_state = value['optimizer']


def fake_build(env, config):
    actor = Actor(hidden_sizes=config.hidden_sizes)
    return FakeAgent(actor, config.steps_per_update), actor


def synthetic_collection(env, agent, *, updates, on_transition, on_update, on_boundary, should_stop=None):
    """Exercise orchestration callbacks without running skrl or changing weights."""
    for index in range(updates):
        if should_stop and should_stop():
            return dict(vector_steps=index*agent.cfg.rollouts, transitions=index*agent.cfg.rollouts*env.num_envs,
                        ppo_updates=index, episode_resets=0, completed_episodes={}, stop_reason=should_stop())
        for k in range(agent.cfg.rollouts):
            tr = env.step_command(env.current_target)
            if k+1 == agent.cfg.rollouts: on_update(index+1, {'synthetic_test_metric': 0.})
            on_transition(index*agent.cfg.rollouts+k, tr, ('synthetic',)*env.num_envs)
        on_boundary(index+1)
    return dict(vector_steps=updates*agent.cfg.rollouts, transitions=updates*agent.cfg.rollouts*env.num_envs,
                ppo_updates=updates, episode_resets=0, completed_episodes={})


def test_eval_models_resolve_backends_and_clear_inapplicable_numerics(inputs):
    _, manifest, _ = inputs
    manifest['config'].update(physics_backend='mjwarp', device='cuda:0', numerics_profile='dense_newton_ordered_rows_gpu_v1')
    for model, backend in [('M1', 'mjwarp'), ('M2', 'mjwarp'), ('M3', 'comfree'), ('M4', 'xpbd'), ('M5', 'kamino')]:
        c = evaluation_config(manifest, preset=model)
        assert (c.contact_preset, c.physics_backend, c.num_envs) == (model, backend, 1)
        assert c.numerics_profile == ('dense_newton_ordered_rows_gpu_v1' if backend == 'mjwarp' else None)


@pytest.mark.parametrize('profile', ['dense_newton_ordered_changes_v1', 'dense_newton_ordered_rows_gpu_v1'])
def test_legacy_implicit_profile_is_preserved_and_native_resume_rejected(inputs, profile):
    _, manifest, bank = inputs
    manifest['config'].update(physics_backend='mjwarp', device='cuda:0')
    manifest['physics'] = {'numerics_profile': profile}
    config = recorded_environment_config(manifest)
    assert config.numerics_profile == profile
    assert evaluation_config(manifest).numerics_profile == profile
    metadata = {'environment': manifest, 'ppo': {}}
    validate_resume_request(metadata, config, {}, bank)
    with pytest.raises(ValueError, match='same PPO/environment'):
        validate_resume_request(metadata, EnvConfig(**manifest['config']), {}, bank)
    del manifest['physics']['numerics_profile']
    with pytest.raises(ValueError, match='missing its numerical profile'):
        recorded_environment_config(manifest)


def test_direct_restore_preserves_legacy_implicit_mjwarp_profile(inputs, monkeypatch):
    tmp, manifest, bank = inputs
    profile = 'dense_newton_ordered_changes_v1'
    manifest['config'].update(physics_backend='mjwarp', device='cuda:0')
    manifest['physics'] = {'numerics_profile': profile}
    monkeypatch.setattr(FakeEnv, 'manifest', manifest)
    env = FakeEnv(recorded_environment_config(manifest), bank)
    cfg = PPOConfig()
    agent, actor = fake_build(env, cfg)
    path = tmp/'training.pt'
    save_training_checkpoint(agent, actor, env, cfg, path)
    metadata = torch.load(path.with_suffix('.resume.pt'), weights_only=True)
    metadata['environment']['config'].pop('numerics_profile')
    torch.save(metadata, path.with_suffix('.resume.pt'))
    restored, _ = fake_build(env, cfg)
    restore_training_checkpoint(restored, env, cfg, path)
    assert restored.optimizer_state['step'] == 17
    native = FakeEnv(EnvConfig(**manifest['config']), bank)
    with pytest.raises(ValueError, match='same PPO/environment'):
        restore_training_checkpoint(restored, native, cfg, path)


@pytest.mark.parametrize('field,value', [('control_dt', .032), ('episode_steps', 3), ('target_rate', 2.), ('cost_reference', 99.)])
def test_independent_eval_config_rejects_task_changes(inputs, field, value):
    _, manifest, _ = inputs
    config = dict(manifest['config']); config[field] = value
    with pytest.raises(ValueError, match=field): evaluation_config(manifest, environment=config)


def test_matrix_same_bank_metrics_and_all_five_cells_without_training(inputs, monkeypatch):
    tmp, manifest, bank = inputs
    FakeEnv.manifest = manifest; monkeypatch.setattr(evaluate_rl, 'LeapVectorEnv', FakeEnv)
    controller = evaluate_rl.RLPolicyController
    monkeypatch.setattr(evaluate_rl, 'RLPolicyController', lambda path, device='cpu': controller(path, device='cpu'))
    cells = evaluate_rl_matrix.main(['--policy', str(tmp/'actor.pt'), '--case-bank', str(tmp/'cases.json'), '--output', str(tmp/'matrix')])
    assert len(cells) == 5 and all(c['execution_status'] == 'completed' for c in cells)
    assert all(c['case_bank_sha256'] == bank.sha256 and c['completed_cases'] == 2 for c in cells)
    assert all(c['mean_final_angle_error_deg_completed'] == pytest.approx(60., abs=1e-4) for c in cells)
    assert len((tmp/'matrix/comparison.csv').read_text().splitlines()) == 6
    import csv
    rows = list(csv.DictReader((tmp/'matrix/comparison.csv').open()))
    assert all(row['deadlines'] == '2' and row['physical_drops'] == '0' and row['failed_cases'] == '0' for row in rows)
    assert CaseBank.load(tmp/'matrix/cases.json').sha256 == bank.sha256


def test_matrix_failure_keeps_aborted_and_unexecuted_denominators(inputs, monkeypatch):
    tmp, manifest, _ = inputs
    class FaultEnv(FakeEnv):
        def step_command(self, command): raise RuntimeError('injected interruption')
    FaultEnv.manifest = manifest
    monkeypatch.setattr(evaluate_rl, 'LeapVectorEnv', FaultEnv)
    with pytest.raises(RuntimeError, match='injected'):
        evaluate_rl_matrix.main(['--policy', str(tmp/'actor.pt'), '--case-bank', str(tmp/'cases.json'),
            '--models', 'M1', 'M2', '--output', str(tmp/'matrix')])
    result = json.loads((tmp/'matrix/summary.json').read_text())
    assert (result['failed_cells'], result['not_executed_cells']) == (1, 1)
    failed, unexecuted = result['cells']
    assert failed['allocated_cases'] == 2 and failed['completed_cases'] == 0 and failed['not_executed'] == 1
    assert failed['outcomes']['infrastructure_interrupted'] == 1
    assert failed['executed_cases'] == failed['failed_cases'] == 1
    assert unexecuted['execution_status'] == 'not_executed' and 'successes' not in unexecuted


def test_bundle_integrity_and_legacy_metadata(inputs):
    tmp, manifest, bank = inputs
    FakeEnv.manifest = manifest
    env = FakeEnv(EnvConfig(**manifest['config']), bank); cfg = PPOConfig()
    agent, actor = fake_build(env, cfg)
    progress = dict(completed_ppo_updates=7, vector_steps=56, transitions=112)
    path = save_run_checkpoint(agent, actor, env, cfg, tmp/'bundle', progress)
    assert checkpoint_metadata(path)['progress'] == progress
    restored, _ = fake_build(env, cfg)
    restore_training_checkpoint(restored, env, cfg, path)
    assert restored.optimizer_state['step'] == 17
    with pytest.raises(FileExistsError): save_run_checkpoint(agent, actor, env, cfg, tmp/'bundle', progress)
    (path.parent/'actor.pt').write_bytes(b'incomplete')
    with pytest.raises(ValueError, match='integrity'): checkpoint_metadata(path)
    legacy = tmp/'legacy/training.pt'; legacy.parent.mkdir()
    save_training_checkpoint(agent, actor, env, cfg, legacy)
    assert 'progress' not in checkpoint_metadata(legacy)


def test_cli_periodic_checkpoint_and_additional_update_resume(inputs, monkeypatch):
    tmp, manifest, _ = inputs
    FakeEnv.manifest = manifest
    monkeypatch.setattr(train_rl, 'LeapVectorEnv', FakeEnv)
    monkeypatch.setattr(train_rl, 'build_agent', fake_build)
    monkeypatch.setattr(train_rl, 'collect_updates', synthetic_collection)
    config = tmp/'config.json'
    config.write_text(json.dumps({'environment': manifest['config'], 'ppo': {'steps_per_update': 2}}))
    first = train_rl.main(['--config', str(config), '--case-bank', str(tmp/'cases.json'), '--updates', '4',
        '--checkpoint-interval', '2', '--output', str(tmp/'run')])
    assert first['cumulative_progress'] == dict(completed_ppo_updates=4, vector_steps=8, transitions=16)
    source = tmp/'run/checkpoints/update_00000002/training.pt'; frozen = sha(source)
    zero = train_rl.main(['--resume', str(source), '--updates', '0', '--output', str(tmp/'zero')])
    assert zero['ppo_updates'] == zero['transitions'] == 0 and not zero['actor_parameters_changed']
    assert zero['cumulative_progress']['completed_ppo_updates'] == 2
    resumed = train_rl.main(['--resume', str(source), '--updates', '1', '--output', str(tmp/'resumed')])
    assert resumed['ppo_updates'] == 1 and resumed['cumulative_progress']['completed_ppo_updates'] == 3
    assert sha(source) == frozen
    with pytest.raises(ValueError, match='checkpoint'):
        (source.parent/'checkpoint.json').unlink()
        checkpoint_metadata(source)


def test_collect_boundary_after_last_transition_and_reset(inputs):
    """Use the real collector with a no-op fake agent: no PPO or simulation."""
    from ContactModelStudy.RL.SkrlPPO import collect_updates
    _, manifest, bank = inputs
    events = []
    class EventEnv(FakeEnv):
        case_ids = ('a', 'b')
        def step(self, latent): return self.step_command(self.current_target)
        def reset_indices(self, ids):
            events.append('reset'); super().reset_indices(ids)
    EventEnv.manifest = manifest
    class NoOptimizerAgent:
        cfg = FakeCfg(rollouts=2)
        models = {'policy': Actor(), 'value': torch.nn.Linear(73, 1)}
        tracking_data = {k: [0.] for k in ('Loss / Policy loss', 'Loss / Value loss', 'Policy / Standard deviation')}
        def pre_interaction(self, **kw): pass
        def act(self, obs, states, **kw): return torch.zeros(2, 16), {'log_prob': torch.zeros(2, 1)}
        def record_transition(self, **kw): pass
        def post_interaction(self, **kw): events.append('post')
    collect_updates(EventEnv(EnvConfig(**manifest['config']), bank), NoOptimizerAgent(), updates=1,
        on_transition=lambda *args: events.append('transition'), on_boundary=lambda n: events.append('checkpoint'))
    assert events == ['transition', 'post', 'transition', 'post', 'reset', 'checkpoint']


def test_failure_retains_last_complete_periodic_checkpoint(inputs, monkeypatch):
    tmp, manifest, _ = inputs
    FakeEnv.manifest = manifest
    monkeypatch.setattr(train_rl, 'LeapVectorEnv', FakeEnv)
    monkeypatch.setattr(train_rl, 'build_agent', fake_build)
    def fail_after_boundary(env, agent, **kw):
        synthetic_collection(env, agent, updates=1, **{k: v for k, v in kw.items() if k != 'updates'})
        raise RuntimeError('injected partial next rollout')
    monkeypatch.setattr(train_rl, 'collect_updates', fail_after_boundary)
    config = tmp/'config.json'
    config.write_text(json.dumps({'environment': manifest['config'], 'ppo': {'steps_per_update': 2}}))
    with pytest.raises(RuntimeError, match='partial'):
        train_rl.main(['--config', str(config), '--case-bank', str(tmp/'cases.json'), '--updates', '3',
            '--checkpoint-interval', '1', '--output', str(tmp/'failed')])
    result = json.loads((tmp/'failed/summary.json').read_text())
    saved = checkpoint_metadata(result['latest_complete_checkpoint'])
    assert saved['progress']['completed_ppo_updates'] == 1
    assert not (tmp/'failed/training.pt').exists()


def test_resume_preflight_rejects_changed_settings_before_allocating_environment(inputs, monkeypatch):
    tmp, manifest, bank = inputs
    FakeEnv.manifest = manifest
    env = FakeEnv(EnvConfig(**manifest['config']), bank)
    cfg = PPOConfig(); agent, actor = fake_build(env, cfg)
    source = save_run_checkpoint(agent, actor, env, cfg, tmp/'source',
        dict(completed_ppo_updates=0, vector_steps=0, transitions=0))
    def forbidden(*a, **kw): pytest.fail('Mismatch should be rejected before simulator allocation')
    monkeypatch.setattr(train_rl, 'LeapVectorEnv', forbidden)
    from dataclasses import asdict
    for name in ('ppo', 'bank', 'environment'):
        config = {'environment': dict(manifest['config']), 'ppo': asdict(cfg)}
        extra = []
        if name == 'ppo': config['ppo']['learning_rate'] *= 2
        if name == 'environment': config['environment']['episode_steps'] += 1
        if name == 'bank':
            CaseBank(bank.cases[:1]).save(tmp/'different.json')
            extra = ['--case-bank', str(tmp/'different.json')]
        (tmp/'request.json').write_text(json.dumps(config))
        with pytest.raises(ValueError, match='same PPO/environment/case-bank'):
            train_rl.main(['--resume', str(source), '--updates', '0', '--config', str(tmp/'request.json'),
                '--output', str(tmp/name), *extra])
        assert not (tmp/name).exists()


def test_resume_allows_checkout_relocation_but_not_changed_scene_content(inputs):
    tmp, manifest, bank = inputs
    FakeEnv.manifest = manifest
    env = FakeEnv(EnvConfig(**manifest['config']), bank)
    cfg = PPOConfig(); agent, actor = fake_build(env, cfg)
    source = tmp/'training.pt'
    save_training_checkpoint(agent, actor, env, cfg, source)
    FakeEnv.manifest = deepcopy(manifest)
    FakeEnv.manifest['scene_path'] = '/another/checkout/scenes/same.xml'
    restore_training_checkpoint(agent, env, cfg, source)
    FakeEnv.manifest['scene_xml_sha256'] = 'changed'
    with pytest.raises(ValueError, match='same PPO/environment/case-bank'):
        restore_training_checkpoint(agent, env, cfg, source)


def test_resume_checks_included_scene_identity_and_warns_for_legacy_metadata(inputs, monkeypatch):
    tmp, manifest, bank = inputs
    monkeypatch.setattr(FakeEnv, 'manifest', manifest)
    env = FakeEnv(EnvConfig(**manifest['config']), bank)
    cfg = PPOConfig(); agent, actor = fake_build(env, cfg)
    source = tmp/'training.pt'
    save_training_checkpoint(agent, actor, env, cfg, source)
    changed = deepcopy(manifest)
    changed['scene_model_fingerprint']['sha256'] = 'changed_dependency'
    monkeypatch.setattr(FakeEnv, 'manifest', changed)
    with pytest.raises(ValueError, match='same PPO/environment/case-bank'):
        restore_training_checkpoint(agent, env, cfg, source)
    monkeypatch.setattr(FakeEnv, 'manifest', manifest)
    metadata_path = source.with_suffix('.resume.pt')
    metadata = torch.load(metadata_path, weights_only=True)
    del metadata['environment']['scene_model_fingerprint']
    torch.save(metadata, metadata_path)
    with pytest.warns(UserWarning, match='Legacy checkpoint'):
        restore_training_checkpoint(agent, env, cfg, source)


def test_checkpoint_rejects_unknown_bundle_schema(inputs):
    tmp, manifest, bank = inputs
    FakeEnv.manifest = manifest
    env = FakeEnv(EnvConfig(**manifest['config']), bank)
    cfg = PPOConfig(); agent, actor = fake_build(env, cfg)
    source = save_run_checkpoint(agent, actor, env, cfg, tmp/'bundle',
        dict(completed_ppo_updates=0, vector_steps=0, transitions=0))
    marker = source.parent/'checkpoint.json'
    payload = json.loads(marker.read_text()); payload['schema'] = 'unknown'
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='manifest'): checkpoint_metadata(source)


def test_review_bundle_cannot_bypass_integrity_by_losing_progress(inputs):
    tmp, manifest, bank = inputs; FakeEnv.manifest = manifest
    env = FakeEnv(EnvConfig(**manifest['config']), bank); cfg = PPOConfig()
    agent, actor = fake_build(env, cfg)
    source = save_run_checkpoint(agent, actor, env, cfg, tmp/'source',
        dict(completed_ppo_updates=0, vector_steps=0, transitions=0))
    path = source.with_suffix('.resume.pt')
    metadata = torch.load(path, weights_only=True); del metadata['progress']; torch.save(metadata, path)
    with pytest.raises(ValueError, match='checkpoint|Checkpoint'): checkpoint_metadata(source)


def test_review_direct_restore_checks_bundle_integrity(inputs):
    tmp, manifest, bank = inputs; FakeEnv.manifest = manifest
    env = FakeEnv(EnvConfig(**manifest['config']), bank); cfg = PPOConfig()
    agent, actor = fake_build(env, cfg)
    source = save_run_checkpoint(agent, actor, env, cfg, tmp/'source',
        dict(completed_ppo_updates=0, vector_steps=0, transitions=0))
    (source.parent/'actor.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='integrity'): restore_training_checkpoint(agent, env, cfg, source)


def test_review_restore_only_cleanup_failure_is_reported(inputs, monkeypatch):
    tmp, manifest, bank = inputs; FakeEnv.manifest = manifest
    env = FakeEnv(EnvConfig(**manifest['config']), bank); cfg = PPOConfig()
    agent, actor = fake_build(env, cfg)
    source = save_run_checkpoint(agent, actor, env, cfg, tmp/'source',
        dict(completed_ppo_updates=0, vector_steps=0, transitions=0))
    class ClosingEnv(FakeEnv):
        def close(self): raise RuntimeError('environment close failed')
    monkeypatch.setattr(train_rl, 'LeapVectorEnv', ClosingEnv)
    monkeypatch.setattr(train_rl, 'build_agent', fake_build)
    with pytest.raises(RuntimeError, match='close failed'):
        train_rl.main(['--resume', str(source), '--updates', '0', '--output', str(tmp/'run')])
    result = json.loads((tmp/'run/summary.json').read_text())
    assert result['execution_status'] == 'failed' and result['failure_reason'] == 'cleanup_failure'
    assert checkpoint_metadata(result['latest_complete_checkpoint'])['progress']['transitions'] == 0


def test_review_matrix_raised_call_is_failed_even_after_completed_summary(inputs, monkeypatch):
    tmp, _, bank = inputs
    def late_failure(policy, cases, output, **kwargs):
        output.mkdir()
        (output/'summary.json').write_text(json.dumps({'execution_status': 'completed',
            'completed_cases': len(cases.cases), 'failed_cases': 0, 'not_executed': 0}))
        raise RuntimeError('late evaluation exit failure')
    monkeypatch.setattr(evaluate_rl_matrix, 'run_evaluation', late_failure)
    with pytest.raises(RuntimeError, match='late evaluation'):
        evaluate_rl_matrix.main(['--policy', str(tmp/'actor.pt'), '--case-bank', str(tmp/'cases.json'),
            '--models', 'M1', 'M2', '--output', str(tmp/'matrix')])
    result = json.loads((tmp/'matrix/summary.json').read_text())
    assert result['completed_cells'] == 0 and result['failed_cells'] == 1
    assert result['not_executed_cells'] == 1 and bank.sha256 == result['case_bank_sha256']


def test_cli_sample_cap_saves_only_complete_updates_and_can_restore(inputs, monkeypatch):
    tmp, manifest, _ = inputs
    monkeypatch.setattr(FakeEnv, 'manifest', manifest)
    monkeypatch.setattr(train_rl, 'LeapVectorEnv', FakeEnv)
    monkeypatch.setattr(train_rl, 'build_agent', fake_build)
    monkeypatch.setattr(train_rl, 'collect_updates', synthetic_collection)
    config = tmp/'config.json'
    config.write_text(json.dumps({'environment': manifest['config'], 'ppo': {'steps_per_update': 2}}))
    report = train_rl.main(['--config', str(config), '--case-bank', str(tmp/'cases.json'),
        '--updates', '10', '--max-transitions', '10', '--checkpoint-interval', '0', '--output', str(tmp/'bounded')])
    assert report['stop_reason'] == 'transition_budget' and report['execution_status'] == 'stopped'
    assert report['transitions'] == 8 and report['ppo_updates'] == 2
    source = Path(report['latest_checkpoint'])
    assert checkpoint_metadata(source)['progress']['transitions'] == 8
    zero = train_rl.main(['--resume', str(source), '--updates', '0', '--output', str(tmp/'restored')])
    assert zero['transitions'] == 0 and zero['cumulative_progress']['transitions'] == 8


def test_signal_stop_saves_checkpoint_after_the_completed_update(inputs, monkeypatch):
    import signal
    tmp, manifest, _ = inputs
    monkeypatch.setattr(FakeEnv, 'manifest', manifest)
    monkeypatch.setattr(train_rl, 'LeapVectorEnv', FakeEnv)
    monkeypatch.setattr(train_rl, 'build_agent', fake_build)
    def signal_collection(env, agent, **kwargs):
        original = kwargs['on_boundary']
        def after_boundary(index):
            original(index)
            signal.raise_signal(signal.SIGTERM)
        kwargs['on_boundary'] = after_boundary
        return synthetic_collection(env, agent, **kwargs)
    monkeypatch.setattr(train_rl, 'collect_updates', signal_collection)
    config = tmp/'config.json'
    config.write_text(json.dumps({'environment': manifest['config'], 'ppo': {'steps_per_update': 2}}))
    previous = signal.getsignal(signal.SIGTERM)
    report = train_rl.main(['--config', str(config), '--case-bank', str(tmp/'cases.json'),
        '--updates', '10', '--checkpoint-interval', '0', '--output', str(tmp/'signal')])
    assert signal.getsignal(signal.SIGTERM) == previous
    assert report['stop_reason'] == 'signal:SIGTERM' and report['ppo_updates'] == 1
    assert checkpoint_metadata(report['latest_checkpoint'])['progress']['transitions'] == 4


def test_real_collector_honors_stop_only_at_complete_update_boundary(inputs):
    from ContactModelStudy.RL.SkrlPPO import collect_updates
    _, manifest, bank = inputs
    steps, boundaries = [], []
    class Env(FakeEnv):
        case_ids = ('a', 'b')
        def step(self, latent):
            steps.append('step')
            return self.step_command(self.current_target)
    Env.manifest = manifest
    class NoOptimizerAgent:
        cfg = FakeCfg(rollouts=2)
        models = {'policy': Actor(), 'value': torch.nn.Linear(73, 1)}
        tracking_data = {k: [0.] for k in ('Loss / Policy loss', 'Loss / Value loss', 'Policy / Standard deviation')}
        def pre_interaction(self, **kw): pass
        def act(self, obs, states, **kw): return torch.zeros(2, 16), {'log_prob': torch.zeros(2, 1)}
        def record_transition(self, **kw): pass
        def post_interaction(self, **kw): pass
    stop = lambda: 'requested' if steps else None
    stats = collect_updates(Env(EnvConfig(**manifest['config']), bank), NoOptimizerAgent(), updates=3,
                            should_stop=stop, on_boundary=boundaries.append)
    assert len(steps) == 2 and boundaries == [1]
    assert stats['ppo_updates'] == 1 and stats['transitions'] == 4 and stats['stop_reason'] == 'requested'


def test_zero_sample_stop_does_not_reset_environment(inputs):
    from ContactModelStudy.RL.SkrlPPO import collect_updates
    _, manifest, bank = inputs
    class UntouchedEnv(FakeEnv):
        def reset_all(self): raise AssertionError('No reset for an already-expired budget')
    UntouchedEnv.manifest = manifest
    result = collect_updates(UntouchedEnv(EnvConfig(**manifest['config']), bank),
        SimpleNamespace(cfg=FakeCfg(rollouts=2)), updates=1, should_stop=lambda: 'wall_time_budget')
    assert result['transitions'] == result['ppo_updates'] == 0


def test_matrix_video_options_reach_every_cell_without_optimization(inputs, monkeypatch):
    from ContactModelStudy.Drivers.EvaluationVideo import VideoConfig
    tmp, _, bank = inputs
    received = []
    def evaluate(policy, cases, output, **options):
        received.append(options['video'])
        return {'execution_status': 'completed', 'completed_cases': len(cases.cases),
                'allocated_cases': len(cases.cases), 'failed_cases': 0, 'not_executed': 0}
    monkeypatch.setattr(evaluate_rl_matrix, 'run_evaluation', evaluate)
    cells = evaluate_rl_matrix.main(['--policy', str(tmp/'actor.pt'), '--case-bank', str(tmp/'cases.json'),
        '--models', 'M1', 'M5', '--video', '--fps', '10', '--width', '8', '--height', '8',
        '--output', str(tmp/'video_matrix')])
    assert received == [VideoConfig(fps=10., width=8, height=8)]*2
    assert all(cell['completed_cases'] == len(bank.cases) for cell in cells)
