"""Use existing Docker membership without changing groups or shell-quoting argv."""
from types import SimpleNamespace
import shlex

import pytest

import leap_docker


@pytest.fixture
def membership(monkeypatch):
    monkeypatch.setattr(leap_docker.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(leap_docker.grp, 'getgrnam', lambda name: SimpleNamespace(gr_gid=999, gr_mem=['chetan']))
    monkeypatch.setattr(leap_docker.pwd, 'getpwuid', lambda uid: SimpleNamespace(pw_name='chetan'))
    monkeypatch.setattr(leap_docker.os, 'getgid', lambda: 1011)
    monkeypatch.setattr(leap_docker.os, 'getgroups', lambda: [1011])


def test_inactive_membership_uses_sg_and_preserves_literal_arguments(membership):
    arguments = ['exec', 'container', 'python', '-c', 'print("$HOME; `id`")', '/file with spaces']
    command = leap_docker.docker_command(arguments)
    assert command[:3] == ['/usr/bin/sg', 'docker', '-c']
    assert shlex.split(command[3]) == ['/usr/bin/docker', *arguments]


def test_active_membership_executes_docker_directly(membership, monkeypatch):
    monkeypatch.setattr(leap_docker.os, 'getgroups', lambda: [1011, 999])
    assert leap_docker.docker_command(['ps']) == ['/usr/bin/docker', 'ps']


def test_unregistered_user_is_not_granted_access(membership, monkeypatch):
    monkeypatch.setattr(leap_docker.grp, 'getgrnam', lambda name: SimpleNamespace(gr_gid=999, gr_mem=[]))
    assert leap_docker.docker_command(['ps']) == ['/usr/bin/docker', 'ps']


def test_missing_docker_is_reported(membership, monkeypatch):
    monkeypatch.setattr(leap_docker.shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError, match='Docker executable'):
        leap_docker.docker_command(['ps'])
