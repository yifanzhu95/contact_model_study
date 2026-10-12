#!/usr/bin/env python3
"""Hash every result artifact after a Polaris G0 job, even on failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--job-exit-code", required=True, type=int)
    parser.add_argument("--exclude", action="append", default=[],
                        help="exact relative path whose writer is outside this job's control")
    args = parser.parse_args(argv)
    outdir = args.outdir.resolve()
    destination = outdir / "artifact_manifest.json"
    files = []
    for path in sorted(p for p in outdir.rglob("*") if p.is_file() and p != destination):
        if str(path.relative_to(outdir)) in args.exclude:
            continue
        files.append({
            "path": str(path.relative_to(outdir)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    document = {
        "schema": "contact_model_study.artifact_manifest.v1",
        "job_exit_code": args.job_exit_code,
        "files": files,
        "excluded_paths": sorted(set(args.exclude)),
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
