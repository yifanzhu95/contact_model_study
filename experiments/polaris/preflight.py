#!/usr/bin/env python3
"""Fail-closed source, environment, scheduler and A100 checks for Polaris G0."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_PACKAGES = {
    "mujoco": "3.6.0",
    "newton": "1.6.0",
    "warp-lang": "1.17.0",
}


def _command(*argv: str) -> tuple[int, str]:
    try:
        result = subprocess.run(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return result.returncode, result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def _check(name: str, passed: bool, detail) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_checks(args) -> dict:
    repo = args.repo.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from ContactModelStudy.Utils.Provenance import run_provenance

    checks = []
    provenance = run_provenance(
        repo,
        {
            "rollout_scene": repo / "scenes/leap/env_leap_rollout_cube_high_high.xml",
            "eval_scene": repo / "scenes/leap/env_leap_eval_cube.xml",
        },
    )
    source = provenance["source"]
    checks.append(_check("exact_source_sha", source["commit"] == args.expected_sha, source))
    checks.append(_check("clean_source_tree", source["dirty"] is False, source["changed_paths"]))
    checks.append(_check(
        "python_3_12",
        sys.version_info[:2] == (3, 12),
        {"version": sys.version, "executable": sys.executable},
    ))

    packages = {}
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages[package] = {"expected": expected, "actual": actual}
        checks.append(_check(f"package_{package}", actual == expected, packages[package]))

    scheduler = provenance["runtime"]["scheduler"]
    checks.append(_check("pbs_compute_job", scheduler["scheduler"] == "pbs", scheduler))
    checks.append(_check("pbs_project_recorded", bool(scheduler.get("project")), scheduler))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected_gpus = getattr(args, "gpus", "0")
    gpu_ids = expected_gpus.split(",")
    checks.append(_check("expected_visible_gpus", visible == expected_gpus,
                         {"CUDA_VISIBLE_DEVICES": visible, "expected": expected_gpus}))

    rc, gpu_text = _command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )
    gpu_lines = [line.strip() for line in gpu_text.splitlines() if line.strip()]
    inventory = {line.split(",", 1)[0].strip(): line for line in gpu_lines}
    a100 = rc == 0 and all("A100" in inventory.get(g, "").upper() for g in gpu_ids)
    checks.append(_check("a100_present", a100, {"returncode": rc, "gpus": gpu_lines}))

    try:
        import warp as wp
        wp.init()
        cuda_devices = [str(wp.get_cuda_device(i)) for i in range(wp.get_cuda_device_count())]
        warp_detail = {"cuda_device_count": len(cuda_devices), "devices": cuda_devices}
        warp_ok = len(cuda_devices) == len(gpu_ids)
    except Exception as exc:  # noqa: BLE001 - every import/driver failure blocks the run
        warp_detail = {"error": f"{type(exc).__name__}: {exc}"}
        warp_ok = False
    checks.append(_check("warp_sees_expected_gpus", warp_ok, warp_detail))

    eagle = str(outdir).startswith(("/eagle/", "/lus/eagle/projects/"))
    checks.append(_check("output_on_eagle", eagle, str(outdir)))
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        probe = outdir / ".preflight-write-test"
        probe.write_text("ok\n")
        probe.unlink()
        writable = True
        write_detail = str(outdir)
    except OSError as exc:
        writable = False
        write_detail = f"{type(exc).__name__}: {exc}"
    checks.append(_check("output_writable", writable, write_detail))

    scenes_ok = all(entry["exists"] and entry["sha256"] for entry in provenance["scenes"].values())
    checks.append(_check("scene_inputs_hashed", scenes_ok, provenance["scenes"]))
    passed = all(item["passed"] for item in checks)
    return {
        "schema": "contact_model_study.polaris_preflight.v1",
        "passed": passed,
        "expected_source_sha": args.expected_sha,
        "requested_gpus": expected_gpus,
        "checks": checks,
        "provenance": provenance,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", choices=("0", "0,1,2,3"), default="0")
    args = parser.parse_args(argv)
    document = run_checks(args)
    _write_json(args.output, document)
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if document["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
