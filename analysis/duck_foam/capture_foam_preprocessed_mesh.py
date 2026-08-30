#!/usr/bin/env python3
"""Capture the exact mesh produced by FOAM's preprocessing decision path.

Run this with FOAM's Pixi Python. The script imports the selected FOAM checkout, follows
the same center -> validate -> repair -> validate -> optional convex-decomposition path as
``foam.spherize_mesh``, restores the original bounding-box offset, and exports the resulting
mesh for independent analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import trimesh
from trimesh.transformations import translation_matrix


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
        "bounds_m": mesh.bounds.tolist(),
    }


def foam_fit_box_scale(mesh: trimesh.Trimesh, box_size: float = 1000.0) -> float:
    """Return the C++ ``Surface::fitIntoBox`` coordinate multiplier.

    The implementation fits the longest model extent into ``[-box_size, box_size]``, so
    its full normalized length is ``2 * box_size``.
    """
    extents = mesh.bounds[1] - mesh.bounds[0]
    longest_extent = float(max(extents))
    if longest_extent <= 0.0:
        raise ValueError("Cannot normalize a mesh with a zero longest extent.")
    return 2.0 * box_size / longest_extent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foam-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--method", default="medial")
    parser.add_argument("--manifold-leaves", type=int, default=1000)
    parser.add_argument("--simplify-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    foam_root = args.foam_root.resolve()
    sys.path.insert(0, str(foam_root))

    from foam import (  # pylint: disable=import-outside-toplevel
        check_valid_for_spherization,
        load_mesh_file,
        smooth_manifold,
    )

    input_path = args.input.resolve()
    loaded = load_mesh_file(input_path).copy()
    raw_record = mesh_record(loaded)

    low_bounds, high_bounds = loaded.bounds
    original_offset = (high_bounds + low_bounds) / 2.0
    loaded.apply_transform(translation_matrix(-original_offset))

    raw_valid = check_valid_for_spherization(args.method, loaded)
    repair_path_triggered = not raw_valid
    convex_fallback = False

    if repair_path_triggered:
        loaded = smooth_manifold(
            loaded,
            manifold_leaves=args.manifold_leaves,
            ratio=args.simplify_ratio,
        )

    repaired_valid = check_valid_for_spherization(args.method, loaded)
    if not repaired_valid:
        decomposition = trimesh.decomposition.convex_decomposition(loaded)
        loaded = trimesh.util.concatenate(
            [
                trimesh.Trimesh(vertices=part["vertices"], faces=part["faces"])
                for part in decomposition
            ]
        )
        convex_fallback = True

    final_valid = check_valid_for_spherization(args.method, loaded)
    if not final_valid:
        raise RuntimeError("Captured mesh still fails FOAM's spherization validation.")

    centered_record = mesh_record(loaded)
    normalization_scale = foam_fit_box_scale(loaded)
    loaded.apply_transform(translation_matrix(original_offset))
    restored_record = mesh_record(loaded)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    loaded.export(args.output)

    metadata = {
        "schema_version": 1,
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "raw_process_false": raw_record,
        },
        "foam": {
            "root": str(foam_root),
            "method": args.method,
            "manifold_leaves": args.manifold_leaves,
            "simplify_ratio": args.simplify_ratio,
            "fit_into_box_half_extent": 1000.0,
            "normalization_scale_internal_units_per_m": normalization_scale,
            "internal_distance_unit_m": 1.0 / normalization_scale,
        },
        "decision_path": {
            "raw_valid": raw_valid,
            "repair_path_triggered": repair_path_triggered,
            "valid_after_manifold_simplify_humphrey": repaired_valid,
            "convex_decomposition_fallback": convex_fallback,
            "final_valid": final_valid,
        },
        "original_bbox_offset_m": original_offset.tolist(),
        "processed_centered": centered_record,
        "processed_restored_to_input_frame": restored_record,
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
        },
    }

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    print(f"Raw valid: {raw_valid}")
    print(f"Repair path triggered: {repair_path_triggered}")
    print(f"Valid after repair: {repaired_valid}")
    print(f"Convex decomposition fallback: {convex_fallback}")
    print(f"Processed mesh written to: {args.output}")
    print(f"Metadata written to: {args.metadata}")


if __name__ == "__main__":
    main()
