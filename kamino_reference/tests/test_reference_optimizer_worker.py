"""CPU-only tests for the collaborator-facing guarded worker runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kamino_feasibility.reference_optimizer_coordinator import initialize_run
from kamino_feasibility.reference_optimizer_worker import (
    build_guard_command,
    execute_guarded_optimizer,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT = REPOSITORY / "configs/optimizer_v3_20_repeat.json"
INPUTS = REPOSITORY / "configs/optimizer_v3_inputs.json"


def test_build_guard_command_preserves_explicit_safety_policy(tmp_path) -> None:
    worker = ["python", "worker.py", "--scene", "scene.xml"]
    command = build_guard_command(
        worker,
        guard_log=tmp_path / "guard.jsonl",
        lock_file=tmp_path / "lock.json",
        gpu_index=1,
        display=":9",
        maximum_runtime_s=33.0,
        allow_display_gpu=True,
        require_anydesk=False,
        require_x11=False,
        remote_window_token="token",
    )
    assert "--allow-display-gpu" in command
    assert "--no-require-anydesk" in command
    assert "--no-require-x11" in command
    assert command[command.index("--gpu-index") + 1] == "1"
    assert command[command.index("--display") + 1] == ":9"
    assert command[-len(worker) :] == worker


def test_runner_rejects_missing_scene_before_starting_gpu(tmp_path) -> None:
    run = tmp_path / "run"
    initialize_run(
        contract_path=CONTRACT,
        inputs_index_path=INPUTS,
        run_directory=run,
        budget_label="G0_4x1",
        chunk_size=4,
    )
    tasks_path = run / "iteration_00/tasks.json"
    tasks = json.loads(tasks_path.read_text())
    command = tasks["tasks"][0]["worker_command"]
    command[command.index("--scene") + 1] = str(tmp_path / "missing.xml")
    tasks_path.write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n")

    with pytest.raises(FileNotFoundError, match="reinitialize with --scene"):
        execute_guarded_optimizer(run, max_new_tasks=0)
