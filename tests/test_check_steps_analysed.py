"""experiments/check_steps_analysed.py: metrics recomputed from every k-th analysed step."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "experiments"))
_spec = importlib.util.spec_from_file_location("check_steps_analysed",
                                               REPO_ROOT / "experiments" / "check_steps_analysed.py")
chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)


def fake_analysed_batch(d, n_episodes=4, n_steps=40, seed=0):
    """A batch folder with one analysed cell: ``smooth`` drifts slowly within each episode, ``flat`` is
    constant per episode, ``noisy`` is independent from step to step."""
    rng = np.random.default_rng(seed)
    (d / "cell_0000_a.status.json").write_text(json.dumps({"cell": 0, "name": "cell_0000_a", "status": "done"}))
    (d / "analysis").mkdir()
    arrays = {}
    for e in range(n_episodes):
        t = np.linspace(0, 1, n_steps)
        arrays[f"smooth__ep{e}"] = e + np.sin(2 * np.pi * t)
        arrays[f"flat__ep{e}"] = np.full(n_steps, float(e))
        arrays[f"noisy__ep{e}"] = rng.normal(size=n_steps)
        arrays[f"step__ep{e}"] = np.arange(n_steps)
    np.savez(d / "analysis" / "cell_0000_a.analysis.npz", **arrays)
    return d


def test_the_baseline_has_no_shift_and_a_constant_metric_keeps_its_mean(tmp_path):
    rows = chk.checkBatch(fake_analysed_batch(tmp_path), metrics=["smooth", "flat", "noisy", "absent"], ks=[2, 4])
    assert {r["metric"] for r in rows} == {"smooth", "flat", "noisy"}       # a missing metric is skipped
    by = {(r["metric"], r["k"]): r for r in rows}
    for m in ("smooth", "flat", "noisy"):
        assert by[(m, 1)]["shift"] == 0 and by[(m, 1)]["width_ratio"] == 1
        assert [by[(m, k)]["n_steps"] for k in (1, 2, 4)] == [160, 80, 40]
    # Constant within each episode: the same mean and episode-level interval whatever is kept.
    for k in (2, 4):
        assert by[("flat", k)]["mean"] == pytest.approx(by[("flat", 1)]["mean"])
        assert by[("flat", k)]["width_ratio"] == pytest.approx(1.0)
    # Slow drift is strongly autocorrelated; independent noise is not.
    assert by[("smooth", 1)]["lag1"] > 0.9 and abs(by[("noisy", 1)]["lag1"]) < 0.4


def test_main_writes_the_table(tmp_path, capsys):
    d = fake_analysed_batch(tmp_path)
    assert chk.main([str(d), "--every", "2", "--metrics", "smooth"]) == 0
    with open(d / "steps_analysed_check.csv") as f:
        rows = list(csv.DictReader(f))
    assert [(r["metric"], r["k"]) for r in rows] == [("smooth", "1"), ("smooth", "2")]
    assert "max |shift|" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        chk.main([str(d), "--every", "1"])


def test_lag1():
    assert chk.lag1(np.arange(10.0)) == pytest.approx(1.0)
    assert chk.lag1(np.array([1.0, -1.0] * 5)) == pytest.approx(-1.0)
    assert np.isnan(chk.lag1(np.array([1.0, 2.0]))) and np.isnan(chk.lag1(np.ones(5)))
