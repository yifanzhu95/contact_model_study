"""CPU-only scene preparation and no-launch validation for Duck/Ball BO."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'experiments'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('obj,default_mass,expected_hulls', [('duck', .05, 8), ('ball', .14, 1)])
@pytest.mark.parametrize('override_mass', [None, .07])
def test_prepare_preserves_geometry_and_scales_explicit_or_derived_inertia(
        tmp_path, obj, default_mass, expected_hulls, override_mass):
    prep = load_script('prepare_object_friction_bo')
    source = ROOT/'scenes/leap'/f'env_leap_rollout_{obj}_high_high.xml'
    before = source.read_bytes()
    mass = default_mass if override_mass is None else override_mass
    record = prep.make_scene(source, tmp_path/f'{obj}.xml', mass)
    model = mujoco.MjModel.from_xml_path(record['path'])
    original = mujoco.MjModel.from_xml_path(str(source))
    assert source.read_bytes() == before
    assert len(record['groups']['object']) == expected_hulls
    assert len(record['groups']['fingertip']) == 4
    assert model.na == 0
    np.testing.assert_allclose(model.body('obj').mass, mass)
    np.testing.assert_allclose(model.body('obj').inertia,
                               original.body('obj').inertia*mass/default_mass, rtol=2e-5)
    for group, sliding in [('object', .3), ('fingertip', .5), ('palm', .4), ('finger', .4)]:
        for name in record['groups'][group]:
            np.testing.assert_allclose(model.geom(name).friction, [sliding, .005, .0001])
    np.testing.assert_allclose(model.geom('floor').friction, original.geom('floor').friction)


def test_check_never_starts_mps_or_planners(tmp_path, monkeypatch):
    runner = load_script('run_prepared_object_bo')
    manifest = dict(source_snapshot_sha256={}, scenes={}, code=str(tmp_path), python='python',
                    csv=str(tmp_path/'searches.csv'), results=str(tmp_path/'search_results'))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr('sys.argv', ['run_search.py', '--experiment', str(tmp_path), '--check'])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[-1] == '--check'
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        return SimpleNamespace(returncode=0)

    def forbidden(*args, **kwargs):
        pytest.fail('A validation-only command tried to use GPUs or launch a process pool')

    monkeypatch.setattr(runner.subprocess, 'run', run)
    monkeypatch.setattr(runner.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(runner, 'gpu_metrics', forbidden)
    assert runner.main() == 0
    assert len(calls) == 1
    assert not (tmp_path/'single_search.lock').exists()
    assert not (tmp_path/'search_results').exists()


def test_winner_export_requires_completion_and_preserves_active_weights(tmp_path):
    runner = load_script('run_prepared_object_bo')
    folder = tmp_path/'cell';folder.mkdir()
    target = tmp_path/'DuckReorient.py';target.write_text('ACTIVE_WEIGHTS = {"w_quat": 100}\n')
    source = target.read_bytes()
    manifest = dict(results=str(tmp_path), targets={'duck': {'path': str(target)}})
    row = dict(n_calls=750, label='duck_test', w_velo=0., w_joint_velo=0., w_fallen_term=0.)
    best = dict(trial=42, objective=-.5, cost_weights={'w_quat': 20.})
    summary = folder/'bo_summary.json'
    summary.write_text(json.dumps(dict(n_trials=749, best=best)))
    with pytest.raises(RuntimeError, match='before'):
        runner.export_winner(manifest, row, 'cell')
    assert not (folder/'best_weights.json').exists()
    summary.write_text(json.dumps(dict(n_trials=750, best=best)))
    runner.export_winner(manifest, row, 'cell')
    assert target.read_bytes() == source
    data = json.loads((folder/'best_weights.json').read_text())
    assert data['cost_weights']['w_velo'] == 0
    assert data['cost_weights']['w_quat'] == 20
    assert all(line.startswith('#') for line in (folder/'best_weights_commented.txt').read_text().splitlines())
