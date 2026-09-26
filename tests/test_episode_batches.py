"""experiments/run_episode_batches.py: CSV -> driver command lines, validation, running, resuming."""

from __future__ import annotations

import importlib.util
import json

import pytest

from conftest import REPO_ROOT
from ContactModelStudy.Tasks.CubeReorient import CubeReorient
from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig

spec = importlib.util.spec_from_file_location("run_episode_batches", REPO_ROOT / "experiments" / "run_episode_batches.py")
batches = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batches)


def write(tmp_path, text, name="b.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


# -- cost-weight overrides (task + driver) ------------------------------------
def test_cost_weight_override_on_the_task():
    base = CubeReorient().params["cost_weights"]
    t = CubeReorient(LeapReorientConfig(cost_weights={"w_quat": 42}))
    assert t.params["cost_weights"]["w_quat"] == 42.0
    assert {k: v for k, v in t.params["cost_weights"].items() if k != "w_quat"} == \
        {k: v for k, v in base.items() if k != "w_quat"}
    with pytest.raises(ValueError, match="unknown cost weight"):
        LeapReorientConfig(cost_weights={"w_bogus": 1})


def test_driver_cost_weight_flag():
    from ContactModelStudy.Drivers.run_episodes import parseArgs
    a = parseArgs(["--cost-weight", "w_quat=30", "--cost-weight", "w_contact=5"])
    assert a.cost_weights == {"w_quat": 30.0, "w_contact": 5.0}
    for bad in (["--cost-weight", "w_quat"], ["--cost-weight", "w_bogus=1"], ["--cost-weight", "w_quat=x"]):
        with pytest.raises(SystemExit):
            parseArgs(bad)


# -- CSV -> command line -----------------------------------------------------
def test_row_to_argv(tmp_path):
    row = {"label": "a b", "rollout_model": "M3", "uncertainty": "false", "debug": "true",
           "time_horizon": "0.2", "w_quat": "40", "video": "true"}
    name = batches.cellName(3, row)
    assert name == "cell_0003_a-b"
    argv = batches.rowToArgv(row, tmp_path, name)
    assert argv[:2] == ["--rollout-model", "M3"]
    assert "--no-uncertainty" in argv and "--debug" in argv
    assert argv[argv.index("--time-horizon") + 1] == "0.2"
    assert argv[argv.index("--cost-weight") + 1] == "w_quat=40"
    assert argv[argv.index("--results") + 1] == str(tmp_path / f"{name}.json")
    assert argv[argv.index("--video") + 1] == str(tmp_path / f"{name}.mp4")
    assert batches.checkRow(argv) is None
    assert "--no-video" in batches.rowToArgv({"label": "x"}, tmp_path, "c")


def test_blank_cells_are_left_to_the_driver(tmp_path):
    p = write(tmp_path, "label,temperature,seed\na,,3\n")
    rows = batches.readRows(p)
    assert rows == [{"label": "a", "seed": "3"}]


@pytest.mark.parametrize("header, message", [
    ("label,temprature", "unknown column"),
    ("label,results", "set by this script"),
    ("label,cost_weight", "unknown column"),
    ("label,seed,seed", "duplicate"),
])
def test_bad_columns(tmp_path, header, message):
    with pytest.raises(ValueError, match=message):
        batches.checkColumns(write(tmp_path, header + "\n"))


@pytest.mark.parametrize("row, message", [
    ({"horizon": "8", "time_horizon": "0.2"}, "not both"),
    ({"rollout_model": "M9"}, "invalid choice"),
    ({"steps": "many"}, "invalid int"),
])
def test_bad_rows_are_caught_by_check(tmp_path, row, message):
    err = batches.checkRow(batches.rowToArgv(row, tmp_path, "c"))
    assert err is not None and message in err


def test_bad_bool(tmp_path):
    with pytest.raises(ValueError, match="true/false"):
        batches.rowToArgv({"uncertainty": "maybe"}, tmp_path, "c")


def test_check_count_and_out_of_range(tmp_path, capsys):
    p = write(tmp_path, "label,seed\na,1\nb,2\n")
    assert batches.main([str(p), "--count"]) == 0 and capsys.readouterr().out.strip() == "2"
    assert batches.main([str(p), "--check", "--outdir", str(tmp_path)]) == 0
    with pytest.raises(SystemExit):
        batches.main([str(p), "--cell", "5"])


def test_example_csv_is_valid(tmp_path):
    assert batches.main([str(REPO_ROOT / "experiments" / "example_batches.csv"), "--check",
                         "--outdir", str(tmp_path)]) == 0


# -- running -------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.slow
def test_run_all_resume_and_summarize(tmp_path):
    csv_path = write(tmp_path, "label,rollout_model,n_episodes,steps,settle,substeps,horizon,n_samples,obj_acc,w_quat\n"
                               "ok,M2,2,3,0,4,4,32,,\n"
                               "missing_scene,M2,1,3,0,4,4,32,foam4,\n"
                               "weights,M3,1,3,0,4,4,32,,42\n")
    out = tmp_path / "out"
    assert batches.main([str(csv_path), "--outdir", str(out)]) == 1        # one cell fails
    status = {p.name: json.loads(p.read_text())["status"] for p in out.glob("*.status.json")}
    assert status == {"cell_0000_ok.status.json": "done", "cell_0001_missing_scene.status.json": "failed",
                      "cell_0002_weights.status.json": "done"}
    c = json.loads((out / "cell_0002_weights.json").read_text())["configs"]
    assert c["rollout_task"]["config"]["cost_weights"] == {"w_quat": 42.0}
    assert c["rollout_simulator"]["class"] == "ComFree"
    summary = json.loads((out / "summary.json").read_text())["cells"]
    assert [e["status"] for e in summary] == ["done", "failed", "done"]
    assert summary[0]["n_episodes_run"] == 2
    # Resuming skips finished cells; a single cell runs in-process.
    mtime = (out / "cell_0000_ok.json").stat().st_mtime
    assert batches.main([str(csv_path), "--outdir", str(out), "--cell", "0"]) == 0
    assert (out / "cell_0000_ok.json").stat().st_mtime == mtime
    assert batches.main([str(csv_path), "--outdir", str(out), "--cell", "0", "--overwrite"]) == 0
    assert (out / "cell_0000_ok.json").stat().st_mtime > mtime
