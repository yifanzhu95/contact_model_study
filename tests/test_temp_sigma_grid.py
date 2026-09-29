"""experiments/run_temp_sigma_grid.py: grid parsing, CSV validation, running, ranking, resuming."""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from conftest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("run_temp_sigma_grid", REPO_ROOT / "experiments" / "run_temp_sigma_grid.py")
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


def write(tmp_path, text, name="g.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_parse_values():
    assert grid.parseValues("40,20,10", "t") == [40.0, 20.0, 10.0]
    assert grid.parseValues("40 20  10", "t") == [40.0, 20.0, 10.0]
    for bad, msg in (("", "empty"), ("5 -1", "> 0"), ("5 5", "twice"), ("5 x", "not a number")):
        with pytest.raises(ValueError, match=msg):
            grid.parseValues(bad, "t")


def test_grid_points_temperature_outermost():
    pts = grid.gridPoints({"temperatures": "20 5", "noise_sigmas": "0.1,0.2"})
    assert pts == [(20.0, 0.1), (20.0, 0.2), (5.0, 0.1), (5.0, 0.2)]


def test_point_argv_sets_the_grid_knobs_and_output(tmp_path):
    row = {"label": "a", "rollout_model": "M3", "temperatures": "20 5", "noise_sigmas": "0.1", "w_quat": "30"}
    argv = grid.pointArgv(row, tmp_path, 1, 5.0, 0.1)
    assert argv[argv.index("--temperature") + 1] == "5" and argv[argv.index("--noise-sigma") + 1] == "0.1"
    assert argv[argv.index("--results") + 1] == str(tmp_path / "point_001_T5_s0.1.json")
    assert "--cost-weight" in argv and "temperatures" not in " ".join(argv)


@pytest.mark.parametrize("header, message", [
    ("label,temperature,temperatures,noise_sigmas", "set by this script"),
    ("label,noise_sigma,temperatures,noise_sigmas", "set by this script"),
    ("label,n_samples", "missing required"),
    ("label,temprature,temperatures,noise_sigmas", "unknown column"),
])
def test_bad_columns(tmp_path, header, message):
    with pytest.raises(SystemExit):
        grid.main([str(write(tmp_path, header + "\n")), "--check"])


def test_check_catches_bad_rows(tmp_path, capsys):
    p = write(tmp_path, 'label,temperatures,noise_sigmas,rollout_model\n'
                        'neg,"5 -1",0.1,M2\nbad_model,5,0.1,M9\nok,"5 10","0.1 0.2",M3\n')
    assert grid.main([str(p), "--check", "--outdir", str(tmp_path / "o")]) == 1
    out = capsys.readouterr().out
    assert "BAD cell_0000_neg" in out and "BAD cell_0001_bad_model" in out and "OK  cell_0002_ok" in out


def test_example_csv_is_valid(tmp_path):
    assert grid.main([str(REPO_ROOT / "experiments" / "example_temp_sigma_grid.csv"), "--check",
                      "--outdir", str(tmp_path)]) == 0


def test_ranking():
    pts = [{"success_rate": 0.4, "mean_steps_to_success": 50},
           {"success_rate": 0.8, "mean_steps_to_success": 90},
           {"success_rate": 0.8, "mean_steps_to_success": 60},
           {"success_rate": 0.0, "mean_steps_to_success": None}]
    ranked = sorted(pts, key=grid._rankKey)
    assert [(p["success_rate"], p["mean_steps_to_success"]) for p in ranked] == \
        [(0.8, 60), (0.8, 90), (0.4, 50), (0.0, None)]


@pytest.mark.gpu
@pytest.mark.slow
def test_run_cell_same_episodes_summaries_and_resume(tmp_path):
    from ContactModelStudy.Utils.EpisodeRecorder import EpisodeReplayer
    p = write(tmp_path, "label,rollout_model,n_episodes,steps,settle,substeps,horizon,n_samples,save_steps,"
                        "stop_on_success,goal_difficulty,temperatures,noise_sigmas\n"
                        'm2,M2,2,4,0.1,4,4,32,true,false,4,"20 5",0.1\n')
    out = tmp_path / "out"
    assert grid.main([str(p), "--cell", "0", "--outdir", str(out)]) == 0
    cell = out / "cell_0000_m2"
    a, b = EpisodeReplayer(cell / "point_000_T20_s0.1.json"), EpisodeReplayer(cell / "point_001_T5_s0.1.json")
    # Every point sees the same episodes: same goal seeds and goals, different temperature.
    assert [e.summary["goal_seed"] for e in a] == [e.summary["goal_seed"] for e in b]
    assert [e.summary["goals"] for e in a] == [e.summary["goals"] for e in b]
    assert a.configs["planner"]["config"]["temperature"] == 20 and b.configs["planner"]["config"]["temperature"] == 5
    gs = json.loads((cell / "grid_summary.json").read_text())
    assert len(gs["points"]) == 2 and gs["best"] is not None and sorted(gs["ranked"]) == [0, 1]
    # Resume: nothing reruns.
    mtime = (cell / "point_000_T20_s0.1.json").stat().st_mtime
    assert grid.main([str(p), "--cell", "0", "--outdir", str(out)]) == 0
    assert (cell / "point_000_T20_s0.1.json").stat().st_mtime == mtime
    assert grid.main([str(p), "--summarize", "--outdir", str(out)]) == 0
    assert (out / "summary.csv").exists() and (out / "best.csv").exists()
