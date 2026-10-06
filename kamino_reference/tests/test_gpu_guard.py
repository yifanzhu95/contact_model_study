import json

import pytest

from kamino_feasibility.gpu_guard import (
    GpuSafetyPolicy,
    GpuSnapshot,
    load_remote_lock,
    parse_nvidia_smi_csv,
    read_available_ram_mib,
    safety_violations,
    write_remote_lock,
    audit_guard_log,
)


def _snapshot(**changes) -> GpuSnapshot:
    values = {
        "index": 0,
        "name": "Test GPU",
        "display_active": False,
        "memory_total_mib": 16000,
        "memory_used_mib": 1000,
        "memory_free_mib": 15000,
        "utilization_gpu_percent": 0,
        "utilization_memory_percent": 0,
        "temperature_c": 35,
        "power_draw_w": 30.0,
        "power_limit_w": 300.0,
        "available_ram_mib": 20000,
        "anydesk_service_active": True,
        "x11_responsive": True,
    }
    values.update(changes)
    return GpuSnapshot(**values)


def test_parse_nvidia_smi_csv() -> None:
    parsed = parse_nvidia_smi_csv(
        "0, RTX Test, Enabled, 16303, 362, 15458, 9, 3, 32, 50.96, 300.00"
    )
    assert parsed["display_active"] is True
    assert parsed["memory_free_mib"] == 15458
    assert parsed["power_limit_w"] == pytest.approx(300.0)


def test_display_gpu_requires_explicit_allowance() -> None:
    policy = GpuSafetyPolicy()
    violations = safety_violations(
        _snapshot(display_active=True), policy, allow_display_gpu=False
    )
    assert any("active desktop" in item for item in violations)
    assert not safety_violations(
        _snapshot(display_active=True), policy, allow_display_gpu=True
    )


def test_resource_and_remote_failures_are_all_reported() -> None:
    violations = safety_violations(
        _snapshot(
            memory_free_mib=100,
            temperature_c=90,
            available_ram_mib=200,
            anydesk_service_active=False,
            x11_responsive=False,
        ),
        GpuSafetyPolicy(),
        allow_display_gpu=True,
    )
    assert len(violations) == 5


def test_remote_lock_roundtrip(tmp_path) -> None:
    path = tmp_path / "REMOTE_WORK_LOCK.json"
    written = write_remote_lock(reason="remote session critical", path=path)
    assert written == path.resolve()
    payload = load_remote_lock(path)
    assert payload is not None
    assert payload["locked"] is True
    assert payload["reason"] == "remote session critical"
    assert json.loads(path.read_text())["schema"].endswith(".v1")


def test_memavailable_parser(tmp_path) -> None:
    path = tmp_path / "meminfo"
    path.write_text("MemTotal: 100000 kB\nMemAvailable: 8192 kB\n")
    assert read_available_ram_mib(path) == 8


def _write_guard_log(
    path,
    *,
    command=None,
    returncode=0,
    terminating=False,
    require_anydesk=True,
    anydesk_active=True,
) -> None:
    snapshot = {
        "memory_free_mib": 12000,
        "memory_used_mib": 1000,
        "utilization_gpu_percent": 90,
        "utilization_memory_percent": 10,
        "temperature_c": 50,
        "power_draw_w": 100.0,
        "available_ram_mib": 16000,
        "anydesk_service_active": anydesk_active,
        "x11_responsive": True,
    }
    rows = [
        {
            "event": "preflight",
            "command": command or ["worker"],
            "policy": {
                "require_anydesk_service": require_anydesk,
                "require_x11_responsive": True,
            },
            "remote_lock_override": True,
            "violations": [],
            "snapshot": snapshot,
        },
        {"event": "started", "pid": 123},
        {"event": "sample", "violations": [], "snapshot": snapshot},
    ]
    if terminating:
        rows.append({"event": "guard_terminating", "reason": "test"})
    rows.append({"event": "completed", "returncode": returncode})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_guard_log_audit_accepts_complete_matching_run(tmp_path) -> None:
    path = tmp_path / "guard.jsonl"
    command = ["python", "worker.py"]
    _write_guard_log(path, command=command)
    report = audit_guard_log(
        path,
        expected_command=command,
        require_remote_lock_override=True,
    )
    assert report["accepted"]
    assert report["metrics"]["maximum_gpu_utilization_percent"] == 90
    assert report["requirements"]["anydesk_service"] is True


def test_guard_log_audit_accepts_inactive_anydesk_when_not_required(tmp_path) -> None:
    path = tmp_path / "guard.jsonl"
    _write_guard_log(
        path,
        require_anydesk=False,
        anydesk_active=False,
    )
    report = audit_guard_log(path, require_remote_lock_override=True)
    assert report["accepted"]
    assert report["requirements"]["anydesk_service"] is False
    assert report["checks"]["anydesk_requirement_satisfied"]


def test_guard_log_audit_rejects_wrong_command_or_termination(tmp_path) -> None:
    path = tmp_path / "guard.jsonl"
    _write_guard_log(path, command=["wrong"], returncode=70, terminating=True)
    report = audit_guard_log(
        path,
        expected_command=["expected"],
        require_remote_lock_override=True,
    )
    assert not report["accepted"]
    assert not report["checks"]["expected_command_matches"]
    assert not report["checks"]["completed_returncode_zero"]
    assert not report["checks"]["no_guard_termination"]
