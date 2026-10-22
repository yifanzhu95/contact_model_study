"""Regression checks for sustained collection, failure accounting and scene identity."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import shutil

import pytest
import torch

from ContactModelStudy.Drivers import train_rl
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv
from ContactModelStudy.RL.Policy import Actor
from ContactModelStudy.RL.RunIO import validate_policy_environment
from ContactModelStudy.RL.SkrlPPO import collect_updates


@dataclass
class CollectorConfig:
    rollouts: int


class CollectorAgent:
    """Exercise the real collector callbacks with no optimization."""
    def __init__(self, rollouts=2, failure=None):
        self.cfg = CollectorConfig(rollouts=rollouts)
        self.models = {'policy': Actor(), 'value': torch.nn.Linear(73, 1)}
        self.tracking_data = defaultdict(list)
        self.failure = failure
        self.recorded = 0

    def pre_interaction(self, **kwargs):
        pass

    def act(self, obs, states, **kwargs):
        return torch.zeros(obs.shape[0], 16), {'log_prob': torch.zeros(obs.shape[0], 1)}

    def record_transition(self, **kwargs):
        self.recorded += 1

    def post_interaction(self, *, timestep, **kwargs):
        if (timestep + 1) % self.cfg.rollouts == 0:
            if self.failure:
                raise self.failure
            for key in ('Loss / Policy loss', 'Loss / Value loss', 'Policy / Standard deviation'):
                self.tracking_data[key].append(float(timestep + 1))


@pytest.mark.parametrize('with_callback', [False, True])
def test_collector_releases_consumed_metrics_at_each_update(with_callback):
    env = LeapVectorEnv(EnvConfig(num_envs=2, episode_steps=2),
                        CaseBank.development(count=2, seed=0, difficulty=8))
    agent = CollectorAgent()
    metrics = []
    retained = []
    try:
        result = collect_updates(env, agent, updates=4,
            on_update=(lambda i, m: metrics.append((i, m))) if with_callback else None,
            on_boundary=lambda i: retained.append(sum(map(len, agent.tracking_data.values()))))
        assert result['ppo_updates'] == 4
        assert retained == [0, 0, 0, 0]
        if with_callback:
            assert [m['Loss / Policy loss'] for _, m in metrics] == [2., 4., 6., 8.]
    finally:
        env.close()


def test_update_failure_counts_already_sampled_terminal_transition(tmp_path, monkeypatch):
    bank = CaseBank.development(count=2, seed=0, difficulty=8)
    bank.save(tmp_path/'cases.json')
    config = {'environment': EnvConfig(num_envs=2, episode_steps=2).to_dict(),
              'ppo': {'steps_per_update': 2}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    agent = CollectorAgent(failure=RuntimeError('injected optimizer failure'))
    monkeypatch.setattr(train_rl, 'build_agent', lambda *a: (agent, agent.models['policy']))
    with pytest.raises(RuntimeError, match='optimizer failure'):
        train_rl.main(['--config', str(tmp_path/'config.json'), '--case-bank', str(tmp_path/'cases.json'),
                       '--updates', '1', '--output', str(tmp_path/'run')])
    summary = json.loads((tmp_path/'run/summary.json').read_text())
    transitions = [json.loads(line) for line in (tmp_path/'run/transitions.jsonl').read_text().splitlines()]
    assert agent.recorded == 2
    assert summary['vector_steps'] == 2 and summary['transitions'] == 4
    assert summary['completed_ppo_updates'] == 0
    assert summary['completed_episodes']['deadline'] == 2
    assert len(transitions) == 2 and transitions[-1]['reasons'] == ['deadline', 'deadline']
    assert summary['latest_complete_checkpoint'] is None
    assert not (tmp_path/'run/checkpoint.json').exists()


def test_policy_validation_rejects_changed_scene_dependencies():
    env = LeapVectorEnv(EnvConfig(), CaseBank.development(count=2, seed=0, difficulty=8))
    try:
        training = env.describe()
    finally:
        env.close()
    training['scene_model_fingerprint'] = {'schema': 'mujoco_mjb_v1', 'mujoco_version': '3.6.0',
                                          'sha256': 'original'}
    current = deepcopy(training)
    current['scene_model_fingerprint']['sha256'] = 'changed_included_mesh'
    with pytest.raises(ValueError, match='scene_model_fingerprint'):
        validate_policy_environment(training, current)
    validate_policy_environment(training, current, allow_scene_transfer=True)


def test_scene_identity_tracks_included_physics_and_meshes_after_relocation(tmp_path):
    from ContactModelStudy.RL.SceneIdentity import scene_model_fingerprint
    scene = tmp_path/'original'; scene.mkdir()
    (scene/'root.xml').write_text('<mujoco><include file="fragment.xml"/></mujoco>')
    fragment = scene/'fragment.xml'
    fragment.write_text('<mujocoinclude><asset><mesh name="shape" file="shape.obj"/></asset>'
                        '<worldbody><body><freejoint/><geom type="mesh" mesh="shape" mass="1"/>'
                        '</body></worldbody></mujocoinclude>')
    mesh = scene/'shape.obj'
    original_mesh = ('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n'
                     'f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')
    mesh.write_text(original_mesh)
    original = scene_model_fingerprint(scene/'root.xml')
    root_hash = hashlib.sha256((scene/'root.xml').read_bytes()).hexdigest()
    moved = tmp_path/'moved'; shutil.copytree(scene, moved)
    assert scene_model_fingerprint(moved/'root.xml') == original
    fragment.write_text(fragment.read_text().replace('mass="1"', 'mass="2"'))
    assert scene_model_fingerprint(scene/'root.xml') != original
    fragment.write_text(fragment.read_text().replace('mass="2"', 'mass="1"'))
    mesh.write_text(original_mesh.replace('v 1 0 0', 'v 2 0 0'))
    assert scene_model_fingerprint(scene/'root.xml') != original
    assert hashlib.sha256((scene/'root.xml').read_bytes()).hexdigest() == root_hash
