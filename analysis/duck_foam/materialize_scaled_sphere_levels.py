#!/usr/bin/env python3
"""Write selected per-level uniform scales into a new FOAM-compatible sphere JSON.

The output remains a list of sphere-tree levels so FOAM's existing visualizer can load it.
Original FOAM fitting statistics are retained only for decoder compatibility and are explicitly
marked as applying to the unscaled source levels.
"""

from __future__ import annotations

import argparse
import copy
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


def parse_level_scale(value: str) -> tuple[int, float]:
    try:
        depth_text, scale_text = value.split("=", maxsplit=1)
        depth = int(depth_text)
        scale = float(scale_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use DEPTH=SCALE, for example 2=0.96") from error
    if depth < 0 or scale <= 0.0:
        raise argparse.ArgumentTypeError("Depth must be nonnegative and scale must be positive.")
    return depth, scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-mesh", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--level-scale",
        type=parse_level_scale,
        action="append",
        required=True,
        help="Per-level scale in DEPTH=SCALE form; may be repeated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = args.reference_mesh.resolve()
    input_path = args.input.resolve()
    mesh = trimesh.load(reference_path, force="mesh", process=True)
    pivot = np.mean(mesh.bounds, axis=0)

    with input_path.open("r", encoding="utf-8") as stream:
        source_levels = json.load(stream)
    output_levels = copy.deepcopy(source_levels)
    scales = dict(args.level_scale)

    unknown_depths = sorted(set(scales) - set(range(len(output_levels))))
    if unknown_depths:
        raise ValueError(f"Requested unavailable depths: {unknown_depths}")

    level_records: list[dict[str, object]] = []
    for depth, level in enumerate(output_levels):
        scale = scales.get(depth, 1.0)
        for sphere in level["spheres"]:
            center = np.asarray(sphere["origin"], dtype=np.float64)
            sphere["origin"] = (pivot + scale * (center - pivot)).tolist()
            sphere["radius"] = scale * float(sphere["radius"])
        level["uniform_scale"] = scale
        level["scale_pivot_m"] = pivot.tolist()
        level["foam_metrics_apply_to_unscaled_source_only"] = scale != 1.0
        level_records.append(
            {
                "depth": depth,
                "sphere_count": len(level["spheres"]),
                "uniform_scale": scale,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(output_levels, stream, indent=2)
        stream.write("\n")

    metadata = {
        "schema_version": 1,
        "reference_mesh": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "bounding_box_center_pivot_m": pivot.tolist(),
        },
        "source_sphere_model": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
        },
        "transform": "center'=pivot+scale*(center-pivot); radius'=scale*radius",
        "levels": level_records,
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
        },
        "note": (
            "Retained FOAM mean/best/worst values describe the unscaled source levels only; "
            "use independent metrics for the calibrated geometry."
        ),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    print(f"Scaled sphere levels written to: {args.output}")
    print(f"Metadata written to: {args.metadata}")
    for record in level_records:
        print(
            f"Depth {record['depth']}: {record['sphere_count']} spheres, "
            f"scale={record['uniform_scale']}"
        )


if __name__ == "__main__":
    main()
