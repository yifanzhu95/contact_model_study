#!/usr/bin/env python3
"""Decide whether a completed G0 cell is eligible for later scaling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args(argv)
    outdir = args.outdir.resolve()
    preflight_path = outdir / "preflight.json"
    status_path = outdir / "cell_0000_polaris_g0_m5_cube.status.json"
    result_path = outdir / "cell_0000_polaris_g0_m5_cube.json"
    checks = []

    def check(name: str, passed: bool, detail) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    try:
        preflight = _load(preflight_path)
        check("preflight_passed", preflight.get("passed") is True, preflight_path.name)
    except (OSError, ValueError) as exc:
        preflight = {}
        check("preflight_passed", False, f"{type(exc).__name__}: {exc}")
    try:
        status = _load(status_path)
        check("execution_completed", status.get("execution_status") == "completed", status)
        check("cell_validation_passed", status.get("validation_status") == "pass", status)
        check("cell_scaling_eligible", status.get("scaling_eligible") is True, status)
        source = status.get("provenance", {}).get("source", {})
        check("status_source_sha", source.get("commit") == args.expected_sha, source)
        check("status_source_clean", source.get("dirty") is False, source)
    except (OSError, ValueError) as exc:
        status = {}
        check("status_readable", False, f"{type(exc).__name__}: {exc}")
    try:
        result = _load(result_path)
        validation = result.get("validation", {})
        check("result_validation_passed", validation.get("status") == "pass", validation)
        check("result_scaling_eligible", validation.get("scaling_eligible") is True, validation)
        source = result.get("configs", {}).get("metadata", {}).get("provenance", {}).get("source", {})
        check("result_source_sha", source.get("commit") == args.expected_sha, source)
        check("result_source_clean", source.get("dirty") is False, source)
    except (OSError, ValueError) as exc:
        check("result_readable", False, f"{type(exc).__name__}: {exc}")

    passed = bool(checks) and all(item["passed"] for item in checks)
    verdict = {
        "schema": "contact_model_study.polaris_smoke_verdict.v1",
        "passed": passed,
        "scaling_eligible": passed,
        "expected_source_sha": args.expected_sha,
        "checks": checks,
    }
    _write(outdir / "smoke_verdict.json", verdict)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
