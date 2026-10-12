"""Small, dependency-free provenance snapshots for local and HPC runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path


_PACKAGES = ("contact_study", "mujoco", "newton", "warp-lang", "numpy")


def file_sha256(path: str | Path) -> str:
    """SHA256 of one file, streamed rather than loaded into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_provenance(root: str | Path) -> dict:
    """Exact source identity; a branch name is informational, never the key."""
    root = Path(root).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    remote = _git(root, "remote", "get-url", "origin")
    changed_paths = [] if not status else [line[3:] for line in status.splitlines()]
    return {
        "schema": "contact_model_study.git_provenance.v1",
        "root": str(root),
        "commit": commit,
        "branch": branch,
        "detached": branch is None and commit is not None,
        "dirty": None if status is None else bool(status),
        "changed_paths": changed_paths,
        "origin": remote,
    }


def scheduler_provenance(environ=None) -> dict:
    """Normalize PBS and SLURM identity without making either one mandatory."""
    env = os.environ if environ is None else environ
    if env.get("PBS_JOBID"):
        return {
            "scheduler": "pbs",
            "job_id": env.get("PBS_JOBID"),
            "array_index": env.get("PBS_ARRAY_INDEX"),
            "queue": env.get("PBS_QUEUE"),
            "project": env.get("PBS_ACCOUNT") or env.get("ALCF_PROJECT") or env.get("PROJECT"),
            "job_name": env.get("PBS_JOBNAME"),
        }
    if env.get("SLURM_JOB_ID"):
        return {
            "scheduler": "slurm",
            "job_id": env.get("SLURM_JOB_ID"),
            "array_index": env.get("SLURM_ARRAY_TASK_ID"),
            "queue": env.get("SLURM_JOB_PARTITION"),
            "project": env.get("SLURM_JOB_ACCOUNT"),
            "job_name": env.get("SLURM_JOB_NAME"),
        }
    return {
        "scheduler": "local",
        "job_id": None,
        "array_index": None,
        "queue": None,
        "project": None,
        "job_name": None,
    }


def runtime_provenance(environ=None) -> dict:
    """Versions and device-selection inputs available without importing CUDA."""
    env = os.environ if environ is None else environ
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema": "contact_model_study.runtime_provenance.v1",
        "hostname": socket.gethostname(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "environment": {
            name: env.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "PBS_JOBID", "SLURM_JOB_ID")
        },
        "scheduler": scheduler_provenance(env),
    }


def scene_provenance(paths: dict[str, str | Path]) -> dict:
    """Identity of the top-level MJCF inputs used by one run.

    Referenced-asset expansion belongs to the Polaris packaging stage; these
    hashes already prevent a top-level rollout/eval scene from changing
    silently between submission and result review.
    """
    out = {}
    for label, raw in paths.items():
        path = Path(raw).expanduser().resolve()
        out[label] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": file_sha256(path) if path.is_file() else None,
        }
    return out


def run_provenance(repo_root: str | Path, scenes: dict[str, str | Path] | None = None) -> dict:
    """One JSON-ready source/runtime/scheduler/scene snapshot."""
    return {
        "schema": "contact_model_study.run_provenance.v1",
        "source": git_provenance(repo_root),
        "runtime": runtime_provenance(),
        "scenes": scene_provenance(scenes or {}),
    }
