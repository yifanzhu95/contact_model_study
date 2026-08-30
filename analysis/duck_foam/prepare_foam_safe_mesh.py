#!/usr/bin/env python3
"""Create a SphereTree-valid Duck mesh with reduced preprocessing distortion.

The source STL becomes watertight after coincident vertices are merged, but SphereTree's
legacy verifier still detects intersecting faces. This script runs only the old Manifold
reconstruction at a selected resolution. It deliberately skips FOAM's later mesh
simplification and 100-iteration Humphrey smoothing stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mesh_record(mesh: trimesh.Trimesh) -> dict[str, object]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "bounds_m": mesh.bounds.tolist(),
        "surface_area_m2": float(mesh.area),
        "volume_m3": float(mesh.volume),
    }


def percent_change(original: float, candidate: float) -> float:
    return 100.0 * (candidate / original - 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foam-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifold-leaves", type=int, default=6000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    foam_root = args.foam_root.resolve()
    sys.path.insert(0, str(foam_root))

    from foam import (  # pylint: disable=import-outside-toplevel
        check_valid_for_spherization,
        load_mesh_file,
        manifold,
    )

    input_path = args.input.resolve()
    raw = trimesh.load(input_path, force="mesh", process=False)
    if not isinstance(raw, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, received {type(raw).__name__}")

    welded = raw.copy()
    welded.merge_vertices()
    if not welded.is_volume:
        raise ValueError("The vertex-welded source is not a closed consistently wound volume.")

    welded_foam_valid = check_valid_for_spherization("medial", welded)
    if welded_foam_valid:
        raise RuntimeError(
            "The welded mesh already passes SphereTree; Manifold reconstruction is unnecessary."
        )

    repaired = manifold(welded, args.manifold_leaves)
    repaired_valid = check_valid_for_spherization("medial", repaired)
    if not repaired_valid:
        raise RuntimeError("Selected Manifold resolution still fails SphereTree verification.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(args.output)

    exported = load_mesh_file(args.output)
    exported_valid = check_valid_for_spherization("medial", exported)
    if not exported_valid:
        raise RuntimeError("Exported repaired OBJ no longer passes SphereTree verification.")

    bound_delta_mm = 1000.0 * (exported.bounds - welded.bounds)
    foam_commit = subprocess.check_output(
        ["git", "-C", str(foam_root), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata = {
        "schema_version": 1,
        "operation": {
            "merge_coincident_vertices": True,
            "manifold_implementation": "FOAM manifold_old",
            "manifold_leaves": args.manifold_leaves,
            "simplification": False,
            "smoothing": False,
        },
        "foam": {
            "root": str(foam_root),
            "commit": foam_commit,
        },
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "raw_process_false": mesh_record(raw),
            "after_vertex_welding": mesh_record(welded),
            "welded_spheretree_valid": welded_foam_valid,
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "reloaded_process_false": mesh_record(exported),
            "spheretree_valid": exported_valid,
        },
        "change_relative_to_welded_source": {
            "bounds_delta_mm": bound_delta_mm.tolist(),
            "max_absolute_bounds_delta_mm": float(np.max(np.abs(bound_delta_mm))),
            "surface_area_percent": percent_change(welded.area, exported.area),
            "volume_percent": percent_change(welded.volume, exported.volume),
        },
    }

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    changes = metadata["change_relative_to_welded_source"]
    print(f"Welded source SphereTree-valid: {welded_foam_valid}")
    print(f"Repaired output SphereTree-valid: {exported_valid}")
    print(f"Output vertices/faces: {len(exported.vertices)}/{len(exported.faces)}")
    print(f"Volume change: {changes['volume_percent']:.3f}%")
    print(f"Surface-area change: {changes['surface_area_percent']:.3f}%")
    print(f"Maximum bounds change: {changes['max_absolute_bounds_delta_mm']:.3f} mm")
    print(f"Repaired OBJ written to: {args.output}")
    print(f"Metadata written to: {args.metadata}")


if __name__ == "__main__":
    main()
