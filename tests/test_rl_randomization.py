"""Reserved DR must be visible, disabled and fail closed; zero training."""
from dataclasses import replace
import pytest
import torch
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv


@pytest.mark.parametrize('specification', [{'enabled': True}, {'physical_parameters': {'mass': [1, 2]}},
    {'solver_parameters': {'iterations': [20, 40]}}, {'model_probabilities': {'M1': .5, 'M2': .5}}])
def test_unsupported_randomization_rejected_before_backend_allocation(specification):
    with pytest.raises(NotImplementedError, match='reserved only'): EnvConfig(randomization=specification)


def test_disabled_lifecycle_preserves_default_state_and_selected_reset_ledger():
    bank = CaseBank.development(count=2, seed=0, difficulty=8)
    config = EnvConfig(num_envs=2, episode_steps=2)
    baseline = LeapVectorEnv(config, bank)
    reserved = LeapVectorEnv(replace(config, randomization={'enabled': False}), bank)
    try:
        assert 'randomization' not in baseline.describe()
        assert 'randomization' not in config.to_dict()
        assert torch.equal(baseline.reset_all([0, 1]), reserved.reset_all([0, 1]))
        before = dict(reserved.reset_domains[0])
        reserved.reset_indices([1], [0])
        assert reserved.reset_domains[0] == before
        assert reserved.reset_domains[1]['case_id'] == bank.cases[0].case_id
        assert not reserved.reset_domains[1]['randomized']
        assert reserved.describe()['randomization']['status'] == 'reserved_only_fixed_domain'
    finally:
        baseline.close(); reserved.close()
