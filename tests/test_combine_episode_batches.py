"""experiments/combine_episode_batches.py: matching cells, precedence, renaming, the new folder."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys

import numpy as np

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "experiments"))
_spec = importlib.util.spec_from_file_location("combine_episode_batches",
                                               REPO_ROOT / "experiments" / "combine_episode_batches.py")
cmb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmb)
pbr = cmb.pbr


def fake_batch(d, labels, tag, status="done"):
    """A batch folder with one cell per label, two episodes each, as run_episode_batches writes it."""
    d.mkdir()
    for i, label in enumerate(labels):
        name = f"cell_{i:04d}_{label}"
        eps = []
        for k in range(2):
            ep_id = f"{tag}{i}{k}".ljust(8, "0")
            np.save(d / f"{name}_{ep_id}.npy", np.full(3, k))
            eps.append({"id": ep_id, "file": f"{name}_{ep_id}.npy", "finish_reason": "success",
                        "n_steps": 3, "n_saved_steps": 3, "summary": {"goals_reached": k}})
        (d / f"{name}.json").write_text(json.dumps(
            {"format_version": 3, "configs": {}, "n_episodes": 2, "n_success": 2, "episodes": eps}))
        (d / f"{name}.status.json").write_text(json.dumps(
            {"cell": i, "name": name, "status": status, "row": {"label": label, "rollout_model": label},
             "argv": ["--results", str(d / f"{name}.json")]}))
        (d / f"{name}.log").write_text(tag)
    return d


def test_later_folders_replace_repeated_cells(tmp_path):
    base = fake_batch(tmp_path / "base", ["M1", "M4", "M4_low"], "a")
    rerun = fake_batch(tmp_path / "rerun", ["M4", "M5"], "b")
    out = tmp_path / "out"
    assert cmb.main([str(base), str(rerun), "--out", str(out)]) == 0

    cells = pbr.listCells(out)
    assert [c["name"] for c in cells] == ["cell_0000_M1", "cell_0001_M4", "cell_0002_M4_low", "cell_0003_M5"]
    assert [(out / f"{c['name']}.log").read_text() for c in cells] == ["a", "b", "a", "b"]
    for c in cells:
        assert c["status"]["cell"] == c["index"]
        assert c["status"]["argv"] == ["--results", str(out / f"{c['name']}.json")]
        doc = json.loads((out / f"{c['name']}.json").read_text())
        for e in doc["episodes"]:
            assert e["file"].startswith(c["name"] + "_") and (out / e["file"]).exists()
    # M4's .npy came from the rerun, and M4_low's (a label M4 is a prefix of) stayed the base's
    assert sorted(p.name for p in out.glob("cell_0001_M4_*.npy")) == \
        ["cell_0001_M4_b0000000.npy", "cell_0001_M4_b0100000.npy"]
    assert sorted(p.name for p in out.glob("cell_0002_M4_low_*.npy")) == \
        ["cell_0002_M4_low_a2000000.npy", "cell_0002_M4_low_a2100000.npy"]

    with open(out / "batches.csv") as f:
        assert [r["label"] for r in csv.DictReader(f)] == ["M1", "M4", "M4_low", "M5"]
    combined = json.loads((out / "combined.json").read_text())
    assert combined["cells"][1]["replaced"] == [str(base.resolve() / "cell_0001_M4")]
    assert (out / "summary.csv").exists()


def test_a_failed_rerun_does_not_replace_a_done_cell(tmp_path):
    base = fake_batch(tmp_path / "base", ["M1", "M4"], "a")
    rerun = fake_batch(tmp_path / "rerun", ["M4"], "b", status="failed")
    out = tmp_path / "out"
    assert cmb.main([str(base), str(rerun), "--out", str(out)]) == 0
    assert (out / "cell_0001_M4.log").read_text() == "a"
    assert json.loads((out / "combined.json").read_text())["notes"]


def test_dry_run_writes_nothing(tmp_path, capsys):
    base = fake_batch(tmp_path / "base", ["M1"], "a")
    rerun = fake_batch(tmp_path / "rerun", ["M1"], "b")
    assert cmb.main([str(base), str(rerun), "--out", str(tmp_path / "out"), "--dry-run"]) == 0
    assert not (tmp_path / "out").exists()
    assert "replaces base/cell_0000_M1" in capsys.readouterr().out
