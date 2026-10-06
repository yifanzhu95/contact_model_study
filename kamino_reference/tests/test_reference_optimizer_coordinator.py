"""CPU-only integration checks for the resumable v2 coordinator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from kamino_feasibility.reference_optimizer_coordinator import (
    collect_current_iteration,
    initialize_run,
    run_cpu_synthetic_dry_run,
    write_cpu_synthetic_current_iteration,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT = REPOSITORY / "configs/optimizer_v3_20_repeat.json"
INPUTS = REPOSITORY / "configs/optimizer_v3_inputs.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_three_iteration_dry_run_is_complete_and_idempotent(tmp_path) -> None:
    run = tmp_path / "run"
    initialized = initialize_run(
        contract_path=CONTRACT,
        inputs_index_path=INPUTS,
        run_directory=run,
        budget_label="B2_32x20x3",
        chunk_size=4,
    )
    assert initialized["worker_execution_authorized"] is False
    tasks = json.loads((run / "iteration_00/tasks.json").read_text())
    assert tasks["task_count"] == 160
    assert tasks["gpu_workers_started"] is False
    assert all(task["must_run_through_gpu_guard"] for task in tasks["tasks"])

    result = run_cpu_synthetic_dry_run(run, synthetic_seed=4001)
    assert result["created_results"] == 480
    assert result["manifest"]["status"] == "complete"
    assert result["manifest"]["completed_iterations"] == [0, 1, 2]
    for iteration in range(3):
        summary = json.loads(
            (run / f"iteration_{iteration:02d}/iteration_result.json").read_text()
        )
        assert summary["strict_valid_cells"] == 640
        assert summary["expected_cells"] == 640
        assert summary["backend_values"] == ["cpu_synthetic_no_physics"]

    manifest_path = run / "run_manifest.json"
    before = _digest(manifest_path)
    repeated = run_cpu_synthetic_dry_run(run, synthetic_seed=4001)
    assert repeated["created_results"] == 0
    assert repeated["skipped_results"] == 0
    assert _digest(manifest_path) == before


def test_missing_chunk_is_recreated_without_overwriting_completed_chunks(tmp_path) -> None:
    run = tmp_path / "resume"
    initialize_run(
        contract_path=CONTRACT,
        inputs_index_path=INPUTS,
        run_directory=run,
        budget_label="B1_32x20",
        chunk_size=4,
    )
    first = write_cpu_synthetic_current_iteration(run, synthetic_seed=4001)
    assert first == {"created_results": 160, "skipped_results": 0}
    tasks = json.loads((run / "iteration_00/tasks.json").read_text())["tasks"]
    completed = REPOSITORY / tasks[1]["result_path"]
    completed_hash = _digest(completed)
    missing = REPOSITORY / tasks[0]["result_path"]
    missing.unlink()

    resumed = write_cpu_synthetic_current_iteration(run, synthetic_seed=4001)
    assert resumed == {"created_results": 1, "skipped_results": 159}
    assert _digest(completed) == completed_hash
    collect_current_iteration(run)
    assert json.loads((run / "run_manifest.json").read_text())["status"] == "complete"


def test_candidate_tampering_is_rejected(tmp_path) -> None:
    run = tmp_path / "tamper"
    initialize_run(
        contract_path=CONTRACT,
        inputs_index_path=INPUTS,
        run_directory=run,
        budget_label="B1_32x20",
        chunk_size=4,
    )
    write_cpu_synthetic_current_iteration(run, synthetic_seed=4001)
    task = json.loads((run / "iteration_00/tasks.json").read_text())["tasks"][0]
    result_path = REPOSITORY / task["result_path"]
    with np.load(result_path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    arrays["candidates"][0, 0, 0] += np.float32(1.0e-3)
    np.savez_compressed(result_path, **arrays)
    with pytest.raises(ValueError, match="candidate mismatch"):
        collect_current_iteration(run)


def test_chunk_size_above_remote_safety_contract_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        initialize_run(
            contract_path=CONTRACT,
            inputs_index_path=INPUTS,
            run_directory=tmp_path / "unsafe",
            budget_label="B1_32x20",
            chunk_size=5,
        )


def test_scene_override_must_exist(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="scene override does not exist"):
        initialize_run(
            contract_path=CONTRACT,
            inputs_index_path=INPUTS,
            run_directory=tmp_path / "missing-scene",
            budget_label="B1_32x20",
            chunk_size=4,
            scene_path=tmp_path / "missing.xml",
        )


def test_guarded_micro_smoke_is_not_reference_claim_eligible(tmp_path) -> None:
    run = tmp_path / "micro-smoke"
    initialize_run(
        contract_path=CONTRACT,
        inputs_index_path=INPUTS,
        run_directory=run,
        budget_label="G0_4x1",
        chunk_size=4,
    )
    run_cpu_synthetic_dry_run(run, synthetic_seed=4001)
    summary = json.loads((run / "iteration_00/iteration_result.json").read_text())
    assert summary["repeat_count"] == 1
    assert summary["strict_valid_cells"] == 4
    assert summary["reference_estimator_eligible"] is False
    assert summary["reference_claim_eligible"] is False
    assert summary["reference_claim_policy"]["estimator_contract_supported"] is True
    assert summary["reference_claim_policy"]["minimum_numerical_repeats"] == 20


def test_v3_twenty_repeat_budget_is_estimator_but_not_final_claim(tmp_path) -> None:
    run = tmp_path / "v3"
    initialize_run(
        contract_path=CONTRACT,
        inputs_index_path=INPUTS,
        run_directory=run,
        budget_label="B1_32x20",
        chunk_size=4,
    )
    run_cpu_synthetic_dry_run(run, synthetic_seed=4001)
    summary = json.loads((run / "iteration_00/iteration_result.json").read_text())
    assert summary["repeat_count"] == 20
    assert summary["strict_valid_cells"] == 640
    assert summary["reference_estimator_eligible"] is True
    assert summary["reference_claim_eligible"] is False
    assert summary["reference_claim_policy"]["single_budget_claim_allowed"] is False
