"""Atomic local checkpoints. Never unpickle a checkpoint from another source."""
import json
import os
import pickle
from pathlib import Path


def atomic_bytes(path, data, backup=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if backup and path.exists():
        previous = path.with_name(path.name + '.previous')
        link = previous.with_name(previous.name + '.tmp')
        link.unlink(missing_ok=True)
        os.link(path, link)
        link.replace(previous)
    temporary.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2) + '\n').encode())


def write_checkpoint(path, value):
    atomic_bytes(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), backup=True)


def read_checkpoint(path):
    path = Path(path)
    errors = []
    for candidate in (path, path.with_name(path.name + '.previous')):
        if not candidate.exists():
            continue
        try:
            return pickle.loads(candidate.read_bytes())
        except (EOFError, pickle.UnpicklingError, ValueError) as exc:
            errors.append(f'{candidate}: {exc}')
    if errors:
        raise RuntimeError('No valid local resume checkpoint: ' + '; '.join(errors))
    return None
