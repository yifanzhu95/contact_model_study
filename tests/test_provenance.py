"""Source, runtime and scheduler provenance stays JSON-ready and fail-closed."""

from __future__ import annotations

from conftest import REPO_ROOT
from ContactModelStudy.Utils.Provenance import (
    git_provenance,
    run_provenance,
    scheduler_provenance,
)


def test_git_provenance_uses_an_exact_commit():
    provenance = git_provenance(REPO_ROOT)
    assert len(provenance["commit"]) == 40
    assert provenance["branch"]
    assert isinstance(provenance["dirty"], bool)


def test_scheduler_provenance_defaults_to_local():
    assert scheduler_provenance({}) == {
        "scheduler": "local",
        "job_id": None,
        "array_index": None,
        "queue": None,
        "project": None,
        "job_name": None,
    }


def test_run_provenance_hashes_top_level_scene():
    scene = REPO_ROOT / "scenes" / "leap" / "env_leap_rollout_cube_high_high.xml"
    provenance = run_provenance(REPO_ROOT, {"rollout_scene": scene})
    entry = provenance["scenes"]["rollout_scene"]
    assert entry["exists"] and len(entry["sha256"]) == 64
