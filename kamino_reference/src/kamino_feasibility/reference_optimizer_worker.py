"""Guarded execution of externally prepared Reference Optimizer tasks.

The CPU coordinator deliberately separates planning from CUDA execution.  This
module provides the missing reusable handoff layer: it executes only the tasks
recorded in a run manifest, checks the local scene against the frozen hash,
runs every worker through the GPU guard, audits provenance, and then invokes
the strict collector.  It is resumable but never silently accepts a partial or
unguarded result.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .gpu_guard import DEFAULT_REMOTE_LOCK, audit_guard_log, load_remote_lock
from .reference_optimizer_coordinator import collect_current_iteration
from .reference_planning import file_sha256


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _resolve_recorded(path: str, repository_root: Path) -> Path:
    selected = Path(path).expanduser()
    return (
        selected.resolve()
        if selected.is_absolute()
        else (repository_root / selected).resolve()
    )


def _argument(command: list[str], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError as error:
        raise ValueError(f"worker command is missing {flag}") from error
    if index + 1 >= len(command):
        raise ValueError(f"worker command has no value after {flag}")
    return command[index + 1]


def validate_worker_task(task: dict[str, Any], repository_root: Path) -> None:
    """Validate local, immutable worker inputs before starting a subprocess."""

    command = [str(value) for value in task["worker_command"]]
    scene = _resolve_recorded(_argument(command, "--scene"), repository_root)
    if not scene.is_file():
        raise FileNotFoundError(
            f"worker scene does not exist: {scene}; reinitialize with --scene"
        )
    expected_scene_hash = str(task["scene_sha256"])
    if file_sha256(scene) != expected_scene_hash:
        raise ValueError(f"worker scene SHA-256 mismatch: {scene}")

    bundle = _resolve_recorded(_argument(command, "--bundle"), repository_root)
    if not bundle.is_file():
        raise FileNotFoundError(f"worker bundle does not exist: {bundle}")
    expected_bundle_hash = str(task["chunk_bundle_file_sha256"])
    if file_sha256(bundle) != expected_bundle_hash:
        raise ValueError(f"worker bundle SHA-256 mismatch: {bundle}")


def build_guard_command(
    worker_command: list[str],
    *,
    guard_log: Path,
    lock_file: Path = DEFAULT_REMOTE_LOCK,
    gpu_index: int = 0,
    display: str = ":1",
    maximum_runtime_s: float = 120.0,
    poll_interval_s: float = 1.0,
    minimum_free_vram_mib: int = 6144,
    maximum_temperature_c: int = 75,
    minimum_available_ram_mib: int = 6144,
    allow_display_gpu: bool = False,
    require_anydesk: bool = True,
    require_x11: bool = True,
    remote_window_token: str | None = None,
) -> list[str]:
    """Build one explicit, inspectable GPU-guard command."""

    repository_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(repository_root / "scripts/run_gpu_guarded.py"),
        "--lock-file",
        str(lock_file.expanduser().resolve()),
        "run",
        "--gpu-index",
        str(gpu_index),
        "--display",
        display,
        "--maximum-runtime-s",
        str(maximum_runtime_s),
        "--poll-interval-s",
        str(poll_interval_s),
        "--minimum-free-vram-mib",
        str(minimum_free_vram_mib),
        "--maximum-temperature-c",
        str(maximum_temperature_c),
        "--minimum-available-ram-mib",
        str(minimum_available_ram_mib),
    ]
    if allow_display_gpu:
        command.append("--allow-display-gpu")
    if not require_anydesk:
        command.append("--no-require-anydesk")
    if not require_x11:
        command.append("--no-require-x11")
    if remote_window_token is not None:
        command.extend(["--remote-window-token", remote_window_token])
    command.extend(["--log", str(guard_log.resolve()), "--", *worker_command])
    return command


def execute_guarded_optimizer(
    run_directory: str | Path,
    *,
    lock_file: str | Path = DEFAULT_REMOTE_LOCK,
    gpu_index: int = 0,
    display: str = ":1",
    maximum_runtime_s: float = 120.0,
    poll_interval_s: float = 1.0,
    minimum_free_vram_mib: int = 6144,
    maximum_temperature_c: int = 75,
    minimum_available_ram_mib: int = 6144,
    allow_display_gpu: bool = False,
    require_anydesk: bool = True,
    require_x11: bool = True,
    remote_window_token: str | None = None,
    max_new_tasks: int | None = None,
) -> dict[str, Any]:
    """Execute, audit and collect a complete resumable optimizer run."""

    run_directory = Path(run_directory).expanduser().resolve()
    manifest_path = run_directory / "run_manifest.json"
    lock_file = Path(lock_file).expanduser().resolve()
    created = 0
    skipped = 0
    completed_iterations: list[int] = []

    while True:
        manifest = _load_json(manifest_path)
        if manifest["status"] == "complete":
            return {
                "status": "complete",
                "created_tasks": created,
                "skipped_tasks": skipped,
                "completed_iterations": completed_iterations,
                "run_manifest": str(manifest_path),
            }
        repository_root = Path(manifest["repository_root"]).resolve()
        iteration = int(manifest["current_iteration"])
        tasks_path = run_directory / f"iteration_{iteration:02d}" / "tasks.json"
        task_manifest = _load_json(tasks_path)
        pending_after_limit = False

        for task in task_manifest["tasks"]:
            validate_worker_task(task, repository_root)
            output = _resolve_recorded(task["result_path"], repository_root)
            guard_directory = run_directory / f"iteration_{iteration:02d}" / "guard"
            guard_log = guard_directory / f"{task['task_id']}.jsonl"
            guard_audit = guard_directory / f"{task['task_id']}.audit.json"
            worker_stdout = guard_directory / f"{task['task_id']}.stdout.txt"

            worker_command = [str(value) for value in task["worker_command"]]
            if output.exists():
                if not guard_log.exists() or not guard_audit.exists():
                    raise RuntimeError(
                        f"result exists without complete guard provenance: {output}"
                    )
                audit = audit_guard_log(
                    guard_log,
                    expected_command=worker_command,
                    require_remote_lock_override=load_remote_lock(lock_file) is not None,
                )
                if not audit["accepted"]:
                    raise RuntimeError(f"existing guard audit failed: {guard_log}")
                skipped += 1
                continue

            if max_new_tasks is not None and created >= max_new_tasks:
                pending_after_limit = True
                break
            if guard_log.exists() or guard_audit.exists():
                raise RuntimeError(
                    "partial guard provenance requires manual inspection before resume: "
                    f"{guard_log}"
                )

            guard_directory.mkdir(parents=True, exist_ok=True)
            guard_command = build_guard_command(
                worker_command,
                guard_log=guard_log,
                lock_file=lock_file,
                gpu_index=gpu_index,
                display=display,
                maximum_runtime_s=maximum_runtime_s,
                poll_interval_s=poll_interval_s,
                minimum_free_vram_mib=minimum_free_vram_mib,
                maximum_temperature_c=maximum_temperature_c,
                minimum_available_ram_mib=minimum_available_ram_mib,
                allow_display_gpu=allow_display_gpu,
                require_anydesk=require_anydesk,
                require_x11=require_x11,
                remote_window_token=remote_window_token,
            )
            with worker_stdout.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    guard_command,
                    cwd=repository_root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"guarded worker {task['task_id']} failed with "
                    f"return code {completed.returncode}; inspect {worker_stdout}"
                )
            audit = audit_guard_log(
                guard_log,
                expected_command=worker_command,
                require_remote_lock_override=load_remote_lock(lock_file) is not None,
            )
            _write_json(guard_audit, audit)
            if not audit["accepted"]:
                raise RuntimeError(f"guard audit rejected task {task['task_id']}")
            created += 1

        if pending_after_limit:
            return {
                "status": "stopped_at_task_limit",
                "created_tasks": created,
                "skipped_tasks": skipped,
                "completed_iterations": completed_iterations,
                "run_manifest": str(manifest_path),
            }

        collect_current_iteration(run_directory)
        completed_iterations.append(iteration)
