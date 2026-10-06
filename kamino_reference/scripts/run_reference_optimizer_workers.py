#!/usr/bin/env python3
"""Execute and collect a prepared Reference Optimizer run through GPU guards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kamino_feasibility.gpu_guard import DEFAULT_REMOTE_LOCK
from kamino_feasibility.reference_optimizer_worker import execute_guarded_optimizer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_REMOTE_LOCK)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":1"))
    parser.add_argument("--maximum-runtime-s", type=float, default=120.0)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--minimum-free-vram-mib", type=int, default=6144)
    parser.add_argument("--maximum-temperature-c", type=int, default=75)
    parser.add_argument("--minimum-available-ram-mib", type=int, default=6144)
    parser.add_argument("--allow-display-gpu", action="store_true")
    parser.add_argument("--no-require-anydesk", action="store_true")
    parser.add_argument("--no-require-x11", action="store_true")
    parser.add_argument("--remote-window-token")
    parser.add_argument(
        "--max-new-tasks",
        type=int,
        help="Stop safely after this many newly completed tasks; resume with the same command.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = execute_guarded_optimizer(
        args.run_directory,
        lock_file=args.lock_file,
        gpu_index=args.gpu_index,
        display=args.display,
        maximum_runtime_s=args.maximum_runtime_s,
        poll_interval_s=args.poll_interval_s,
        minimum_free_vram_mib=args.minimum_free_vram_mib,
        maximum_temperature_c=args.maximum_temperature_c,
        minimum_available_ram_mib=args.minimum_available_ram_mib,
        allow_display_gpu=args.allow_display_gpu,
        require_anydesk=not args.no_require_anydesk,
        require_x11=not args.no_require_x11,
        remote_window_token=args.remote_window_token,
        max_new_tasks=args.max_new_tasks,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
