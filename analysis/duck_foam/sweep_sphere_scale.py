#!/usr/bin/env python3
"""Sweep uniform sphere-model scales against a fixed reference mesh.

Every sphere center is scaled about the reference mesh bounding-box center and every radius
is multiplied by the same factor. The same mesh samples and per-sphere directions are reused
for every scale at a given depth so that comparisons have low sampling noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_sphere_approximation import (
    closest_points_naive_chunked,
    load_reference_mesh,
    load_sphere_level,
    sample_exposed_sphere_union,
    sample_mesh_surface,
    sha256_file,
    sphere_union_sdf,
    summarize,
)


def parse_depths(value: str) -> list[int]:
    depths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not depths:
        raise argparse.ArgumentTypeError("At least one depth is required.")
    return depths


def build_scales(
    minimum: float,
    maximum: float,
    step: float,
    extras: list[float],
) -> list[float]:
    if minimum <= 0.0 or maximum <= 0.0 or step <= 0.0:
        raise ValueError("Scale bounds and step must be positive.")
    if maximum < minimum:
        raise ValueError("Maximum scale must not be smaller than minimum scale.")
    count = int(np.floor((maximum - minimum) / step + 1e-12)) + 1
    scales = [minimum + index * step for index in range(count)]
    if not np.isclose(scales[-1], maximum):
        scales.append(maximum)
    scales.extend(extras)
    scales.append(1.0)
    return sorted({round(float(scale), 9) for scale in scales})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--spheres", type=Path, required=True)
    parser.add_argument("--depths", type=parse_depths, default=parse_depths("1,2,3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-min", type=float, default=0.88)
    parser.add_argument("--scale-max", type=float, default=1.0)
    parser.add_argument("--scale-step", type=float, default=0.01)
    parser.add_argument("--extra-scale", type=float, action="append", default=[])
    parser.add_argument("--mesh-samples", type=int, default=20000)
    parser.add_argument("--sphere-surface-samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance-mm", type=float, default=0.01)
    parser.add_argument("--minimum-covered-fraction", type=float, default=0.999)
    parser.add_argument("--maximum-gap-mm", type=float, default=0.1)
    parser.add_argument("--overlap-tolerance-m", type=float, default=1e-9)
    parser.add_argument("--closest-point-chunk", type=int, default=64)
    return parser.parse_args()


def evaluate_scale(
    *,
    mesh: Any,
    mesh_points: np.ndarray,
    original_centers: np.ndarray,
    original_radii: np.ndarray,
    pivot: np.ndarray,
    scale: float,
    depth: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    centers = pivot + scale * (original_centers - pivot)
    radii = scale * original_radii
    tolerance_m = args.tolerance_mm / 1000.0

    mesh_sdf = sphere_union_sdf(mesh_points, centers, radii)
    positive_gap = np.maximum(mesh_sdf, 0.0)
    uncovered = mesh_sdf > tolerance_m
    maximum_gap_index = int(np.argmax(mesh_sdf))

    direction_rng = np.random.default_rng(args.seed + 1000 * depth)
    exposed_points, exposed_weights, exposed_sphere_ids, candidate_count = (
        sample_exposed_sphere_union(
            centers,
            radii,
            args.sphere_surface_samples,
            direction_rng,
            args.overlap_tolerance_m,
        )
    )
    closest, unsigned_distance, triangle_id = closest_points_naive_chunked(
        mesh, exposed_points, args.closest_point_chunk
    )
    normals = mesh.face_normals[triangle_id]
    signed_distance = np.einsum("ij,ij->i", exposed_points - closest, normals)
    outside = signed_distance > tolerance_m
    overreach = np.maximum(signed_distance, 0.0)
    maximum_overreach_index = int(np.argmax(signed_distance))

    covered_fraction = float(np.mean(~uncovered))
    maximum_gap_mm = float(1000.0 * np.max(positive_gap))
    eligible = (
        covered_fraction >= args.minimum_covered_fraction
        and maximum_gap_mm <= args.maximum_gap_mm
    )

    return {
        "scale": scale,
        "eligible": eligible,
        "sphere_count": int(len(radii)),
        "mesh_to_union": {
            "covered_fraction": covered_fraction,
            "uncovered_fraction": float(np.mean(uncovered)),
            "positive_gap_mm_all_samples": summarize(positive_gap, 1000.0),
            "positive_gap_mm_uncovered_only": summarize(
                positive_gap[uncovered], 1000.0
            ),
            "maximum_signed_sdf_sample": {
                "mesh_point_m": mesh_points[maximum_gap_index].tolist(),
                "signed_sdf_mm": float(1000.0 * mesh_sdf[maximum_gap_index]),
            },
        },
        "union_to_mesh": {
            "actual_candidate_samples": candidate_count,
            "exposed_sample_count": int(len(exposed_points)),
            "outside_surface_fraction": float(
                np.sum(exposed_weights[outside]) / np.sum(exposed_weights)
            ),
            "unsigned_distance_mm": summarize(
                unsigned_distance, 1000.0, exposed_weights
            ),
            "overreach_mm_all_exposed_samples": summarize(
                overreach, 1000.0, exposed_weights
            ),
            "overreach_mm_outside_only": summarize(
                overreach[outside], 1000.0, exposed_weights[outside]
            ),
            "maximum_locally_signed_distance_sample": {
                "sphere_union_point_m": exposed_points[
                    maximum_overreach_index
                ].tolist(),
                "closest_mesh_point_m": closest[maximum_overreach_index].tolist(),
                "nearest_triangle_id": int(triangle_id[maximum_overreach_index]),
                "source_sphere_index": int(
                    exposed_sphere_ids[maximum_overreach_index]
                ),
                "locally_signed_distance_mm": float(
                    1000.0 * signed_distance[maximum_overreach_index]
                ),
            },
        },
    }


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh.resolve()
    sphere_path = args.spheres.resolve()
    mesh = load_reference_mesh(mesh_path)
    pivot = np.mean(mesh.bounds, axis=0)
    scales = build_scales(
        args.scale_min,
        args.scale_max,
        args.scale_step,
        args.extra_scale,
    )

    mesh_rng = np.random.default_rng(args.seed)
    mesh_points = sample_mesh_surface(mesh, args.mesh_samples, mesh_rng)
    depth_results: list[dict[str, Any]] = []

    for depth in args.depths:
        centers, radii, _ = load_sphere_level(sphere_path, depth)
        results = [
            evaluate_scale(
                mesh=mesh,
                mesh_points=mesh_points,
                original_centers=centers,
                original_radii=radii,
                pivot=pivot,
                scale=scale,
                depth=depth,
                args=args,
            )
            for scale in scales
        ]
        eligible = [result for result in results if result["eligible"]]
        best = min(
            eligible,
            key=lambda result: result["union_to_mesh"][
                "overreach_mm_all_exposed_samples"
            ]["mean"],
            default=None,
        )
        depth_results.append(
            {
                "depth": depth,
                "sphere_count": int(len(radii)),
                "best_eligible_scale": None if best is None else best["scale"],
                "results": results,
            }
        )

    output = {
        "schema_version": 1,
        "reference_mesh": {
            "path": str(mesh_path),
            "sha256": sha256_file(mesh_path),
            "bounds_m": mesh.bounds.tolist(),
        },
        "sphere_model": {
            "path": str(sphere_path),
            "sha256": sha256_file(sphere_path),
        },
        "transform": {
            "definition": "center'=pivot+scale*(center-pivot); radius'=scale*radius",
            "pivot": "reference mesh bounding-box center",
            "pivot_m": pivot.tolist(),
        },
        "selection_constraint": {
            "minimum_covered_fraction": args.minimum_covered_fraction,
            "maximum_positive_gap_mm": args.maximum_gap_mm,
            "coverage_tolerance_mm": args.tolerance_mm,
            "objective": "minimum area-weighted mean positive overreach",
        },
        "sampling": {
            "seed": args.seed,
            "mesh_samples": args.mesh_samples,
            "sphere_surface_requested_samples": args.sphere_surface_samples,
            "scales": scales,
        },
        "depths": depth_results,
        "limitations": [
            "Results are sampled estimates, not analytic Hausdorff guarantees.",
            "The selected scale is conditional on the stated coverage and gap constraints.",
            "Uniform scale cannot restore local shape details lost during preprocessing.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")

    for depth in depth_results:
        print(
            f"Depth {depth['depth']} ({depth['sphere_count']} spheres): "
            f"best eligible scale = {depth['best_eligible_scale']}"
        )
    print(f"Scale sweep written to: {args.output}")


if __name__ == "__main__":
    main()
