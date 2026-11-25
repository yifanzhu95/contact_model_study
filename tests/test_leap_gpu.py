"""Physical-GPU selection and propagation, with no GPU or hardware required."""
import json
from types import SimpleNamespace

import pytest

import launch_leap_experiment as launcher
from ContactModelStudy.Hardware import GPU
from ContactModelStudy.Hardware.TrackingSession import tracking_spec


def inventory():
    return [dict(index=0, uuid='GPU-busy', memory_used_mib=100, memory_total_mib=24000, utilization_percent=90),
            dict(index=1, uuid='GPU-allocated', memory_used_mib=14000, memory_total_mib=24000, utilization_percent=0),
            dict(index=3, uuid='GPU-idle', memory_used_mib=28, memory_total_mib=24000, utilization_percent=0)]


def test_auto_samples_peak_utilization_then_memory(monkeypatch):
    samples = [inventory(), inventory(), inventory()]
    samples[0][0]['utilization_percent'] = 0
    calls = []
    monkeypatch.setattr(GPU, 'gpu_inventory', lambda: samples.pop(0))
    monkeypatch.setattr(GPU.time, 'sleep', lambda duration: calls.append(duration))
    chosen = GPU.select_gpu('auto')
    assert chosen['index'] == 3 and chosen['cuda_visible_devices'] == 'GPU-idle'
    assert chosen['local_device'] == 'cuda:0' and calls == [.2, .2] and not samples


@pytest.mark.parametrize('selection', [3, '3', 'cuda:3'])
def test_explicit_device_maps_physical_index_to_uuid(monkeypatch, selection):
    monkeypatch.setattr(GPU, 'gpu_inventory', inventory)
    assert GPU.select_gpu(selection)['uuid'] == 'GPU-idle'


def test_auto_reuses_active_tracker_gpu_until_explicit_restart(monkeypatch):
    monkeypatch.setattr(GPU, 'gpu_inventory', inventory)
    monkeypatch.setattr(GPU.time, 'sleep', lambda *a: None)
    assert GPU.select_gpu('auto', preferred_uuid='GPU-busy')['uuid'] == 'GPU-busy'
    assert GPU.select_gpu('auto')['uuid'] == 'GPU-idle'
    with pytest.raises(ValueError, match='active tracker GPU'):
        GPU.select_gpu('auto', preferred_uuid='GPU-removed')


def test_mppi_auto_excludes_tracking_card(monkeypatch):
    monkeypatch.setattr(GPU, 'gpu_inventory', inventory)
    monkeypatch.setattr(GPU.time, 'sleep', lambda *a: None)
    assert GPU.select_gpu('auto', exclude_uuids=('GPU-idle',))['uuid'] == 'GPU-allocated'
    with pytest.raises(ValueError, match='No separate GPU'):
        GPU.select_gpu('auto', exclude_uuids=tuple(row['uuid'] for row in inventory()))


@pytest.mark.parametrize('selection', ['-1', 'cpu', 'cuda', '1,3', True])
def test_invalid_gpu_fails_before_inventory(monkeypatch, selection):
    monkeypatch.setattr(GPU, 'gpu_inventory', lambda: pytest.fail('Invalid GPU queried devices'))
    with pytest.raises(ValueError, match='physical GPU index'):
        GPU.select_gpu(selection)


def test_unavailable_device_is_reported(monkeypatch):
    monkeypatch.setattr(GPU, 'gpu_inventory', inventory)
    with pytest.raises(ValueError, match='unavailable'):
        GPU.select_gpu(2)


def test_inventory_parses_nvidia_smi_without_cuda_initialization(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout='3, GPU-real-uuid, 28, 24564, 0\n')
    monkeypatch.setattr(GPU.subprocess, 'run', run)
    assert GPU.gpu_inventory() == [dict(index=3, uuid='GPU-real-uuid', memory_used_mib=28,
                                      memory_total_mib=24564, utilization_percent=0)]
    assert calls[0][0] == 'nvidia-smi'


def test_separate_gpus_reach_host_docker_and_fingertip_workers(tmp_path, monkeypatch):
    config = json.loads((launcher.ROOT/'configs/leap_experiment.json').read_text())
    config.update(init_mask_path=str(tmp_path/'mask.png'), fingertip_paths=True,
                  gpu_assignment=dict(cuda_visible_devices='GPU-idle'),
                  tracking_gpu_assignment=dict(cuda_visible_devices='GPU-vision'))
    commands = launcher.build_commands(config, tmp_path)
    for service in ('tracker', 'sam', 'mask'):
        assert 'CUDA_VISIBLE_DEVICES=GPU-vision' in commands[service]
    control = commands['controller']
    for flag in ('--device', '--fingertip-device'):
        assert control[control.index(flag)+1] == 'cuda:0'
    observed = []
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda command, **kwargs:
                        (observed.append(kwargs) or SimpleNamespace(pid=123)))
    experiment = launcher.Experiment(tmp_path, gpu_assignment=config['gpu_assignment'],
                                    tracking_gpu_assignment=config['tracking_gpu_assignment'])
    try:
        for service in ('controller', 'tracker', 'sam', 'mask', 'mujoco'):
            experiment.start(service, commands[service])
    finally:
        for child in experiment.children.values():
            child.log.close()
    assert [row['env']['CUDA_VISIBLE_DEVICES'] for row in observed] == [
        'GPU-idle', 'GPU-vision', 'GPU-vision', 'GPU-vision', 'GPU-idle']


def test_conflicting_child_device_is_rejected(tmp_path):
    config = json.loads((launcher.ROOT/'configs/leap_experiment.json').read_text())
    config.update(init_mask_path=str(tmp_path/'mask.png'), gpu_assignment=dict(cuda_visible_devices='GPU-idle'))
    with pytest.raises(ValueError, match='before --'):
        launcher.build_commands(config, tmp_path, ['--device', 'cuda:2'])


def test_tracking_signature_changes_only_when_physical_device_changes(tmp_path):
    config = json.loads((launcher.ROOT/'configs/leap_experiment.json').read_text())
    config['camera_config'] = str(launcher.local_path(config['camera_config']))
    config['gpu_assignment'] = dict(cuda_visible_devices='GPU-idle', utilization_percent=0)
    mesh = launcher.ROOT/'scenes/leap/objects/iris_cube.obj'
    first = tracking_spec(config, mesh, .035)
    config['gpu_assignment']['utilization_percent'] = 75
    assert tracking_spec(config, mesh, .035) == first
    config['gpu_assignment']['cuda_visible_devices'] = 'GPU-busy'
    assert tracking_spec(config, mesh, .035) != first


def test_mppi_gpu_changes_do_not_invalidate_running_tracking(tmp_path):
    config = json.loads((launcher.ROOT/'configs/leap_experiment.json').read_text())
    config['camera_config'] = str(launcher.local_path(config['camera_config']))
    config['tracking_gpu_assignment'] = dict(cuda_visible_devices='GPU-vision')
    config['gpu_assignment'] = dict(cuda_visible_devices='GPU-mppi-old')
    mesh = launcher.ROOT/'scenes/leap/objects/iris_cube.obj'
    before = tracking_spec(config, mesh, .035)
    config['gpu_assignment']['cuda_visible_devices'] = 'GPU-mppi-new'
    assert tracking_spec(config, mesh, .035) == before


@pytest.mark.parametrize('flag,value', [('--gpu', 'auto'), ('--device', 'cuda:3')])
def test_launcher_gpu_flag_dry_run_logs_assignment_without_services(tmp_path, monkeypatch, capsys, flag, value):
    monkeypatch.setattr(GPU, 'gpu_inventory', inventory)
    monkeypatch.setattr(GPU.time, 'sleep', lambda *a: None)
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Started an experiment service'))
    assert launcher.main(['--dry-run', '--tracking-dir', str(tmp_path/'tracking'), flag, value]) == 0
    output = capsys.readouterr().out
    assert 'GPU 3 (GPU-idle)' in output and 'CUDA_VISIBLE_DEVICES=GPU-idle' in output
    assert '--device cuda:0' in output


@pytest.mark.parametrize('reuse', [False, True])
def test_auto_pair_keeps_tracking_and_places_mppi_separately(tmp_path, monkeypatch, capsys, reuse):
    monkeypatch.setattr(GPU, 'gpu_inventory', inventory)
    monkeypatch.setattr(GPU.time, 'sleep', lambda *a: None)
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Started an experiment service'))
    tracking = tmp_path/'tracking'
    if reuse:
        tracking.mkdir()
        (tracking/'session.json').write_text(json.dumps(dict(version=1, processes={'tracker': {'pid': 123}},
            spec={'cuda_visible_devices': 'GPU-busy'})))
        monkeypatch.setattr(launcher, 'process_alive', lambda identity: True)
    assert launcher.main(['--dry-run', '--tracking-dir', str(tracking)]) == 0
    output = capsys.readouterr().out
    if reuse:
        assert 'Tracking/SAM: GPU 0 (GPU-busy)' in output
        assert 'MPPI: GPU 3 (GPU-idle)' in output
        assert 'Keeping the active tracker GPU' in output
    else:
        assert 'Tracking/SAM: GPU 3 (GPU-idle)' in output
        assert 'MPPI: GPU 1 (GPU-allocated)' in output


def test_explicit_role_flags_override_shared_gpu_without_starting_services(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(GPU, 'gpu_inventory', inventory)
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Started an experiment service'))
    assert launcher.main(['--dry-run', '--tracking-dir', str(tmp_path/'tracking'),
                          '--gpu', '0', '--tracking-gpu', '3', '--mppi-gpu', '1']) == 0
    output = capsys.readouterr().out
    assert 'Tracking/SAM: GPU 3 (GPU-idle)' in output
    assert 'MPPI: GPU 1 (GPU-allocated)' in output
