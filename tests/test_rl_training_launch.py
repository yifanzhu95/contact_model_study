"""Unsupported launch modes must fail without sampling or shared output writes."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ContactModelStudy.Drivers import train_rl


LAUNCH_KEYS = ('WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'RANK', 'LOCAL_RANK')


@pytest.mark.parametrize('launch', [
    {'WORLD_SIZE': '2', 'LOCAL_WORLD_SIZE': '2', 'RANK': '0', 'LOCAL_RANK': '0'},
    {'LOCAL_WORLD_SIZE': '2'},
    {'RANK': '1', 'LOCAL_RANK': '1'},
    {'WORLD_SIZE': 'invalid'},
])
def test_unsupported_launch_rejected_before_config_or_output(tmp_path, monkeypatch, capsys, launch):
    for key in LAUNCH_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in launch.items():
        monkeypatch.setenv(key, value)
    output = tmp_path/'run'
    with pytest.raises(SystemExit) as exc:
        train_rl.main(['--config', str(tmp_path/'missing.json'), '--output', str(output)])
    assert exc.value.code == 2
    assert not output.exists()
    assert 'single-process' in capsys.readouterr().err


def test_explicit_single_process_launch_keeps_normal_config_validation(tmp_path, monkeypatch):
    for key, value in zip(LAUNCH_KEYS, ('1', '1', '0', '0'), strict=True):
        monkeypatch.setenv(key, value)
    with pytest.raises(FileNotFoundError):
        train_rl.main(['--config', str(tmp_path/'missing.json'), '--output', str(tmp_path/'run')])
    assert not (tmp_path/'run').exists()


def test_multi_rank_cli_exits_without_entering_distributed_initialization(tmp_path):
    launch_env = dict(os.environ)
    launch_env.update(WORLD_SIZE='2', LOCAL_WORLD_SIZE='2', RANK='0', LOCAL_RANK='0')
    result = subprocess.run([
        sys.executable, '-m', 'ContactModelStudy.Drivers.train_rl',
        '--config', str(tmp_path/'missing.json'), '--output', str(tmp_path/'run'),
    ], cwd=Path(__file__).resolve().parents[1], env=launch_env,
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert 'Multi-GPU joint PPO training is not integrated' in result.stderr
    assert not (tmp_path/'run').exists()
