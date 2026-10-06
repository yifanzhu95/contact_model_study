"""Safety primitives for CUDA work on a GPU that also drives the desktop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE_LOCK = (
    REPOSITORY_ROOT / "artifacts" / "safety" / "REMOTE_WORK_LOCK.json"
)


@dataclass(frozen=True)
class GpuSafetyPolicy:
    """Resource boundaries for one guarded GPU subprocess."""

    maximum_runtime_s: float = 120.0
    poll_interval_s: float = 1.0
    minimum_free_vram_mib: int = 6144
    maximum_temperature_c: int = 75
    minimum_available_ram_mib: int = 6144
    require_anydesk_service: bool = True
    require_x11_responsive: bool = True

    def validate(self) -> None:
        if self.maximum_runtime_s <= 0.0:
            raise ValueError("maximum_runtime_s must be positive")
        if self.poll_interval_s <= 0.0:
            raise ValueError("poll_interval_s must be positive")
        if self.minimum_free_vram_mib < 0:
            raise ValueError("minimum_free_vram_mib cannot be negative")
        if self.maximum_temperature_c <= 0:
            raise ValueError("maximum_temperature_c must be positive")
        if self.minimum_available_ram_mib < 0:
            raise ValueError("minimum_available_ram_mib cannot be negative")


@dataclass(frozen=True)
class GpuSnapshot:
    """One redacted local health sample; no network endpoint is recorded."""

    index: int
    name: str
    display_active: bool
    memory_total_mib: int
    memory_used_mib: int
    memory_free_mib: int
    utilization_gpu_percent: int
    utilization_memory_percent: int
    temperature_c: int
    power_draw_w: float
    power_limit_w: float
    available_ram_mib: int
    anydesk_service_active: bool
    x11_responsive: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_nvidia_smi_csv(line: str) -> dict[str, Any]:
    """Parse the exact no-units CSV requested by the guarded runner."""

    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) != 11:
        raise ValueError(f"expected 11 nvidia-smi fields, got {len(parts)}")
    return {
        "index": int(parts[0]),
        "name": parts[1],
        "display_active": parts[2].lower() in {"enabled", "active", "yes"},
        "memory_total_mib": int(parts[3]),
        "memory_used_mib": int(parts[4]),
        "memory_free_mib": int(parts[5]),
        "utilization_gpu_percent": int(parts[6]),
        "utilization_memory_percent": int(parts[7]),
        "temperature_c": int(parts[8]),
        "power_draw_w": float(parts[9]),
        "power_limit_w": float(parts[10]),
    }


def read_available_ram_mib(meminfo: str | Path = "/proc/meminfo") -> int:
    path = Path(meminfo)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError(f"MemAvailable is missing from {path}")


def service_is_active(service: str, *, timeout_s: float = 2.0) -> bool:
    result = subprocess.run(
        ["systemctl", "is-active", service],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_s,
        check=False,
    )
    return result.returncode == 0


def x11_is_responsive(
    *, display: str = ":1", timeout_s: float = 2.0
) -> bool:
    result = subprocess.run(
        ["xdpyinfo", "-display", display],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_s,
        check=False,
    )
    return result.returncode == 0


def collect_snapshot(
    *, gpu_index: int = 0, display: str = ":1", timeout_s: float = 3.0
) -> GpuSnapshot:
    query = (
        "index,name,display_active,memory.total,memory.used,memory.free,"
        "utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit"
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "nvidia-smi health query failed: " + result.stderr.strip()
        )
    rows = [row for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU row, got {len(rows)}")
    gpu = parse_nvidia_smi_csv(rows[0])
    return GpuSnapshot(
        **gpu,
        available_ram_mib=read_available_ram_mib(),
        anydesk_service_active=service_is_active("anydesk.service"),
        x11_responsive=x11_is_responsive(display=display),
    )


def safety_violations(
    snapshot: GpuSnapshot,
    policy: GpuSafetyPolicy,
    *,
    allow_display_gpu: bool,
) -> list[str]:
    policy.validate()
    violations = []
    if snapshot.display_active and not allow_display_gpu:
        violations.append(
            "NVIDIA GPU drives the active desktop; pass --allow-display-gpu "
            "only during an explicitly confirmed recovery window"
        )
    if snapshot.memory_free_mib < policy.minimum_free_vram_mib:
        violations.append(
            f"free VRAM {snapshot.memory_free_mib} MiB is below "
            f"{policy.minimum_free_vram_mib} MiB"
        )
    if snapshot.temperature_c > policy.maximum_temperature_c:
        violations.append(
            f"GPU temperature {snapshot.temperature_c} C exceeds "
            f"{policy.maximum_temperature_c} C"
        )
    if snapshot.available_ram_mib < policy.minimum_available_ram_mib:
        violations.append(
            f"available RAM {snapshot.available_ram_mib} MiB is below "
            f"{policy.minimum_available_ram_mib} MiB"
        )
    if policy.require_anydesk_service and not snapshot.anydesk_service_active:
        violations.append("AnyDesk service is not active")
    if policy.require_x11_responsive and not snapshot.x11_responsive:
        violations.append("X11 display is not responsive")
    return violations


def load_remote_lock(path: str | Path = DEFAULT_REMOTE_LOCK) -> dict[str, Any] | None:
    selected = Path(path)
    if not selected.exists():
        return None
    payload = json.loads(selected.read_text(encoding="utf-8"))
    if payload.get("locked") is not True:
        raise ValueError(f"invalid remote-work lock payload: {selected}")
    return payload


def write_remote_lock(
    *, reason: str, path: str | Path = DEFAULT_REMOTE_LOCK
) -> Path:
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "kamino_feasibility.remote_work_lock.v1",
        "locked": True,
        "reason": reason,
        "policy": (
            "Heavy CUDA work is denied while this file exists unless the "
            "guard receives the exact one-shot recovery-window token."
        ),
    }
    selected.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected


def audit_guard_log(
    path: str | Path,
    *,
    expected_command: list[str] | None = None,
    require_remote_lock_override: bool = False,
) -> dict[str, Any]:
    """Audit one completed guard JSONL log without touching the GPU."""

    source = Path(path).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("GPU guard log is empty")
    events = [str(row.get("event")) for row in rows]
    preflights = [row for row in rows if row.get("event") == "preflight"]
    starts = [row for row in rows if row.get("event") == "started"]
    completions = [row for row in rows if row.get("event") == "completed"]
    monitored = [
        row for row in rows if row.get("event") in {"preflight", "sample"}
    ]
    snapshots = [row["snapshot"] for row in monitored if "snapshot" in row]
    policy = preflights[0].get("policy", {}) if len(preflights) == 1 else {}
    require_anydesk = bool(policy.get("require_anydesk_service", True))
    require_x11 = bool(policy.get("require_x11_responsive", True))
    checks = {
        "first_event_is_preflight": events[0] == "preflight",
        "exactly_one_preflight": len(preflights) == 1,
        "exactly_one_started": len(starts) == 1,
        "exactly_one_completed": len(completions) == 1,
        "last_event_is_completed": events[-1] == "completed",
        "completed_returncode_zero": (
            len(completions) == 1 and completions[0].get("returncode") == 0
        ),
        "no_guard_termination": "guard_terminating" not in events,
        "all_health_samples_violation_free": bool(monitored)
        and all(not row.get("violations") for row in monitored),
        "anydesk_requirement_satisfied": bool(snapshots)
        and (
            not require_anydesk
            or all(bool(snapshot.get("anydesk_service_active")) for snapshot in snapshots)
        ),
        "x11_requirement_satisfied": bool(snapshots)
        and (
            not require_x11
            or all(bool(snapshot.get("x11_responsive")) for snapshot in snapshots)
        ),
        "expected_command_matches": (
            expected_command is None
            or (len(preflights) == 1 and preflights[0].get("command") == expected_command)
        ),
        "remote_lock_override_recorded": (
            not require_remote_lock_override
            or (
                len(preflights) == 1
                and preflights[0].get("remote_lock_override") is True
            )
        ),
    }
    metrics = None
    if snapshots:
        metrics = {
            "health_snapshot_count": len(snapshots),
            "minimum_free_vram_mib": min(
                int(snapshot["memory_free_mib"]) for snapshot in snapshots
            ),
            "maximum_used_vram_mib": max(
                int(snapshot["memory_used_mib"]) for snapshot in snapshots
            ),
            "maximum_gpu_utilization_percent": max(
                int(snapshot["utilization_gpu_percent"]) for snapshot in snapshots
            ),
            "maximum_memory_utilization_percent": max(
                int(snapshot["utilization_memory_percent"]) for snapshot in snapshots
            ),
            "maximum_temperature_c": max(
                int(snapshot["temperature_c"]) for snapshot in snapshots
            ),
            "maximum_power_draw_w": max(
                float(snapshot["power_draw_w"]) for snapshot in snapshots
            ),
            "minimum_available_ram_mib": min(
                int(snapshot["available_ram_mib"]) for snapshot in snapshots
            ),
        }
    return {
        "schema": "kamino_feasibility.gpu_guard_audit.v1",
        "guard_log_path": str(source),
        "event_count": len(rows),
        "event_sequence": events,
        "sample_event_count": events.count("sample"),
        "requirements": {
            "anydesk_service": require_anydesk,
            "x11_responsive": require_x11,
        },
        "checks": checks,
        "metrics": metrics,
        "accepted": bool(all(checks.values())),
    }
