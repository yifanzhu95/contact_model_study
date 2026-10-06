#!/usr/bin/env python3
"""Validate the compact, collaborator-facing Kamino handoff without simulation."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any

from kamino_feasibility.reference_planning import (
    file_sha256,
    load_reference_planning_bundle,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT = REPOSITORY / "configs/optimizer_v3_20_repeat.json"
INPUTS = REPOSITORY / "configs/optimizer_v3_inputs.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        help="Optional local rollout MJCF; checked against the frozen SHA-256.",
    )
    parser.add_argument(
        "--check-cuda",
        action="store_true",
        help="Also initialize Warp, allocate a small CUDA buffer, and verify SolverKamino.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_delivery(
    *, scene: str | Path | None = None, check_cuda: bool = False
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {
        "python": platform.python_version(),
        "packages": {},
    }
    required_packages = ("newton", "warp-lang", "mujoco", "mujoco-warp", "numpy")
    for package in required_packages:
        try:
            details["packages"][package] = version(package)
            checks[f"package_{package}_installed"] = True
        except Exception:
            details["packages"][package] = None
            checks[f"package_{package}_installed"] = False

    required = [CONTRACT, INPUTS]
    for path in required:
        checks[f"file_{path.name}_present"] = path.is_file()
    contract = _load_json(CONTRACT)
    index = _load_json(INPUTS)
    checks["contract_schema_supported"] = (
        contract.get("schema")
        == "kamino_feasibility.reference_optimizer_v3_preregistration.v1"
    )
    checks["contract_frozen"] = (
        contract.get("status") == "FROZEN_FOR_COLLABORATOR_SCALING"
    )
    checks["contract_hash_matches_index"] = (
        file_sha256(CONTRACT) == index.get("contract_sha256")
    )

    loaded_bundles = {}
    for role in ("template", "development", "held_out"):
        row = index[role]
        path = (REPOSITORY / row["path"]).resolve()
        checks[f"{role}_bundle_present"] = path.is_file()
        checks[f"{role}_file_hash_matches"] = (
            path.is_file() and file_sha256(path) == row["file_sha256"]
        )
        if path.is_file():
            bundle = load_reference_planning_bundle(path)
            loaded_bundles[role] = bundle
            if "fingerprint_sha256" in row:
                checks[f"{role}_fingerprint_matches"] = (
                    bundle.fingerprint_sha256() == row["fingerprint_sha256"]
                )

    development = loaded_bundles.get("development")
    if development is not None:
        details["frozen_problem"] = {
            "main_repo_commit": development.spec.main_repo_commit,
            "scene_sha256": development.spec.scene_sha256,
            "samples": development.spec.n_samples,
            "iterations": development.spec.n_iterations,
            "control_horizon": development.spec.control_horizon,
            "fine_steps_per_control": development.spec.physics_steps_per_control,
            "action_dimension": development.spec.nu,
        }
        checks["development_budget_capacity"] = (
            development.spec.n_samples >= 128
            and development.spec.n_iterations >= 3
        )
        if scene is not None:
            selected_scene = Path(scene).expanduser().resolve()
            checks["scene_present"] = selected_scene.is_file()
            checks["scene_hash_matches"] = (
                selected_scene.is_file()
                and file_sha256(selected_scene) == development.spec.scene_sha256
            )
            details["scene"] = str(selected_scene)
        else:
            details["scene"] = None
            details["scene_check"] = "not requested; pass --scene before a GPU run"

    if check_cuda:
        import newton
        import warp as wp

        wp.init()
        cuda_devices = [device for device in wp.get_devices() if device.is_cuda]
        checks["cuda_device_available"] = bool(cuda_devices)
        checks["solver_kamino_available"] = hasattr(newton.solvers, "SolverKamino")
        if cuda_devices:
            buffer = wp.zeros(1, dtype=wp.float32, device=cuda_devices[0])
            wp.synchronize_device(cuda_devices[0])
            checks["cuda_allocation_succeeded"] = str(buffer.device) == str(
                cuda_devices[0]
            )
            details["cuda_device"] = str(cuda_devices[0])

    return {
        "schema": "kamino_feasibility.delivery_validation.v1",
        "ready": bool(all(checks.values())),
        "checks": checks,
        "details": details,
    }


def main() -> int:
    args = _parser().parse_args()
    result = validate_delivery(scene=args.scene, check_cuda=args.check_cuda)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
