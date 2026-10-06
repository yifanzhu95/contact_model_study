#!/usr/bin/env python3
"""Run CUDA work behind a remote-session lock and a local health watchdog."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from kamino_feasibility.gpu_guard import (
    DEFAULT_REMOTE_LOCK,
    GpuSafetyPolicy,
    collect_snapshot,
    load_remote_lock,
    safety_violations,
    write_remote_lock,
)


REMOTE_WINDOW_TOKEN = "USER_CONFIRMED_RECOVERY_WINDOW"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock-file", type=Path, default=DEFAULT_REMOTE_LOCK
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--gpu-index", type=int, default=0)
    status.add_argument("--display", default=os.environ.get("DISPLAY", ":1"))

    lock = subparsers.add_parser("lock")
    lock.add_argument("--reason", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--gpu-index", type=int, default=0)
    run.add_argument("--display", default=os.environ.get("DISPLAY", ":1"))
    run.add_argument("--maximum-runtime-s", type=float, default=120.0)
    run.add_argument("--poll-interval-s", type=float, default=1.0)
    run.add_argument("--minimum-free-vram-mib", type=int, default=6144)
    run.add_argument("--maximum-temperature-c", type=int, default=75)
    run.add_argument("--minimum-available-ram-mib", type=int, default=6144)
    run.add_argument("--allow-display-gpu", action="store_true")
    run.add_argument(
        "--no-require-anydesk",
        action="store_true",
        help="Do not make AnyDesk service health a run-stopping condition.",
    )
    run.add_argument(
        "--no-require-x11",
        action="store_true",
        help="Do not make X11 responsiveness a run-stopping condition.",
    )
    run.add_argument("--remote-window-token")
    run.add_argument("--log", required=True, type=Path)
    run.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_event(stream, event: str, **payload: Any) -> None:
    row = {"time_utc": _utc_now(), "event": event, **payload}
    stream.write(json.dumps(row, sort_keys=True) + "\n")
    stream.flush()


def _terminate_group(process: subprocess.Popen, stream, reason: str) -> None:
    _write_event(stream, "guard_terminating", reason=reason, pid=process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5.0)


def _status(args: argparse.Namespace) -> int:
    snapshot = collect_snapshot(gpu_index=args.gpu_index, display=args.display)
    payload = {
        "schema": "kamino_feasibility.gpu_guard_status.v1",
        "remote_work_lock": load_remote_lock(args.lock_file),
        "remote_work_lock_path": str(args.lock_file.expanduser().resolve()),
        "snapshot": snapshot.to_dict(),
        "heavy_gpu_work_allowed_by_default": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("run requires a command after --")

    lock = load_remote_lock(args.lock_file)
    if lock is not None and args.remote_window_token != REMOTE_WINDOW_TOKEN:
        raise RuntimeError(
            f"remote-work lock is active at {args.lock_file}; heavy GPU work "
            "is denied. An explicit safe recovery window is required."
        )
    policy = GpuSafetyPolicy(
        maximum_runtime_s=args.maximum_runtime_s,
        poll_interval_s=args.poll_interval_s,
        minimum_free_vram_mib=args.minimum_free_vram_mib,
        maximum_temperature_c=args.maximum_temperature_c,
        minimum_available_ram_mib=args.minimum_available_ram_mib,
        require_anydesk_service=not args.no_require_anydesk,
        require_x11_responsive=not args.no_require_x11,
    )
    policy.validate()
    destination = args.log.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        preflight = collect_snapshot(
            gpu_index=args.gpu_index, display=args.display
        )
        violations = safety_violations(
            preflight, policy, allow_display_gpu=args.allow_display_gpu
        )
        _write_event(
            stream,
            "preflight",
            command=command,
            policy=policy.__dict__,
            snapshot=preflight.to_dict(),
            remote_lock_override=lock is not None,
            violations=violations,
        )
        if violations:
            raise RuntimeError("GPU safety preflight failed: " + "; ".join(violations))

        def _lower_priority() -> None:
            os.nice(10)

        process = subprocess.Popen(
            command,
            start_new_session=True,
            preexec_fn=_lower_priority,
        )
        started = time.monotonic()
        _write_event(stream, "started", pid=process.pid)
        guard_reason = None
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > policy.maximum_runtime_s:
                guard_reason = (
                    f"runtime {elapsed:.3f}s exceeded "
                    f"{policy.maximum_runtime_s:.3f}s"
                )
                break
            try:
                snapshot = collect_snapshot(
                    gpu_index=args.gpu_index, display=args.display
                )
                violations = safety_violations(
                    snapshot,
                    policy,
                    allow_display_gpu=args.allow_display_gpu,
                )
                _write_event(
                    stream,
                    "sample",
                    elapsed_s=elapsed,
                    snapshot=snapshot.to_dict(),
                    violations=violations,
                )
                if violations:
                    guard_reason = "; ".join(violations)
                    break
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                guard_reason = f"health probe failed: {error}"
                break
            time.sleep(policy.poll_interval_s)

        if guard_reason is not None:
            _terminate_group(process, stream, guard_reason)
            return 70
        returncode = int(process.returncode)
        _write_event(stream, "completed", returncode=returncode)
        return returncode


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.action == "status":
            return _status(args)
        if args.action == "lock":
            selected = write_remote_lock(
                reason=args.reason, path=args.lock_file
            )
            print(selected)
            return 0
        if args.action == "run":
            return _run(args)
        raise AssertionError(args.action)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"gpu_guard: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
