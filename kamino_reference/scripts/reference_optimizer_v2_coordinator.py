#!/usr/bin/env python3
"""Prepare, dry-run, or collect a chunked Reference Optimizer v2 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kamino_feasibility.reference_optimizer_coordinator import (
    collect_current_iteration,
    initialize_run,
    run_cpu_synthetic_dry_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    initialize = sub.add_parser("initialize")
    initialize.add_argument("--contract", required=True, type=Path)
    initialize.add_argument("--inputs-index", required=True, type=Path)
    initialize.add_argument("--budget", required=True)
    initialize.add_argument("--chunk-size", type=int, default=4)
    initialize.add_argument("--run-directory", required=True, type=Path)
    initialize.add_argument(
        "--scene",
        type=Path,
        help=(
            "Local path to the rollout MJCF. Its SHA-256 must match the "
            "frozen bundle. Strongly recommended on every new machine."
        ),
    )
    dry = sub.add_parser("dry-run")
    dry.add_argument("--run-directory", required=True, type=Path)
    dry.add_argument("--synthetic-seed", type=int, default=4001)
    collect = sub.add_parser("collect")
    collect.add_argument("--run-directory", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.action == "initialize":
        result = initialize_run(
            contract_path=args.contract,
            inputs_index_path=args.inputs_index,
            run_directory=args.run_directory,
            budget_label=args.budget,
            chunk_size=args.chunk_size,
            scene_path=args.scene,
        )
    elif args.action == "dry-run":
        result = run_cpu_synthetic_dry_run(
            args.run_directory, synthetic_seed=args.synthetic_seed
        )
    elif args.action == "collect":
        result = collect_current_iteration(args.run_directory)
    else:
        raise AssertionError(args.action)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
