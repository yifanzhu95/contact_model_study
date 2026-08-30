#!/usr/bin/env python3
"""Create a topology-welded OBJ without smoothing or changing the Duck surface.

STL stores triangles independently, so FOAM's ``process=False`` loader sees the Duck as
thousands of disconnected triangles. This script merges coincident vertices only, verifies
that bounds, area, and volume are preserved, and exports indexed OBJ geometry that remains
watertight when FOAM loads it without automatic processing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    raw = trimesh.load(input_path, force="mesh", process=False)
    if not isinstance(raw, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, received {type(raw).__name__}")

    raw_record = mesh_record(raw)
    welded = raw.copy()
    welded.merge_vertices()
    welded_record = mesh_record(welded)

    if not welded.is_watertight or not welded.is_winding_consistent or not welded.is_volume:
        raise ValueError("Vertex welding did not produce a valid closed, consistently wound mesh.")
    if not np.allclose(raw.bounds, welded.bounds, rtol=0.0, atol=1e-12):
        raise ValueError("Vertex welding changed the mesh bounds.")
    if not np.isclose(raw.area, welded.area, rtol=1e-12, atol=1e-15):
        raise ValueError("Vertex welding changed the mesh surface area.")
    if not np.isclose(raw.volume, welded.volume, rtol=1e-12, atol=1e-15):
        raise ValueError("Vertex welding changed the mesh volume.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    welded.export(args.output)

    exported = trimesh.load(args.output, force="mesh", process=False)
    exported_record = mesh_record(exported)
    if not exported.is_watertight or not exported.is_winding_consistent or not exported.is_volume:
        raise ValueError("Exported OBJ is not valid when reloaded with process=False.")
    if not np.allclose(welded.bounds, exported.bounds, rtol=0.0, atol=1e-7):
        raise ValueError("OBJ export changed the mesh bounds beyond float text precision.")
    if not np.isclose(welded.area, exported.area, rtol=1e-6, atol=1e-12):
        raise ValueError("OBJ export changed the mesh area beyond float text precision.")
    if not np.isclose(welded.volume, exported.volume, rtol=1e-6, atol=1e-12):
        raise ValueError("OBJ export changed the mesh volume beyond float text precision.")

    metadata = {
        "schema_version": 1,
        "operation": "merge coincident vertices only; no smoothing or simplification",
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "raw_process_false": raw_record,
        },
        "welded_before_export": welded_record,
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "reloaded_process_false": exported_record,
        },
        "preservation_checks": {
            "bounds_unchanged_before_export": True,
            "surface_area_unchanged_before_export": True,
            "volume_unchanged_before_export": True,
            "exported_obj_valid_without_processing": True,
        },
    }

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    print(f"Raw vertices/faces: {len(raw.vertices)}/{len(raw.faces)}")
    print(f"Welded vertices/faces: {len(welded.vertices)}/{len(welded.faces)}")
    print(f"Exported OBJ valid with process=False: {exported.is_volume}")
    print(f"Welded OBJ written to: {args.output}")
    print(f"Metadata written to: {args.metadata}")


if __name__ == "__main__":
    main()
