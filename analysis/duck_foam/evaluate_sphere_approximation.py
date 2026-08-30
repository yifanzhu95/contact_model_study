#!/usr/bin/env python3
"""Evaluate a sphere-union approximation against a triangle mesh.

This script deliberately does not use FOAM's mean/best/worst values. It measures:

1. Mesh -> sphere union: positive analytic sphere-union SDF values are uncovered mesh
   surface points (undercoverage).
2. Exposed sphere-union boundary -> mesh: positive locally signed nearest-triangle
   distances are sphere boundary points outside the mesh (overcoverage).

The reference mesh is loaded with ``process=True`` so duplicate vertices are welded.
No additional smoothing or mesh simplification is requested by this evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Mesh scene is empty: {path}")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, received {type(mesh).__name__}")
    if not mesh.is_watertight:
        raise ValueError(
            "The mesh is still not watertight after duplicate-vertex welding; "
            "inside/outside signs would be unreliable."
        )
    if not mesh.is_winding_consistent:
        raise ValueError("The mesh winding is inconsistent; outward signs would be unreliable.")
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def load_sphere_level(path: Path, depth: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        levels = json.load(stream)

    if not isinstance(levels, list):
        raise ValueError("Sphere JSON must contain a list of sphere-tree levels.")
    if depth < 0 or depth >= len(levels):
        raise ValueError(f"Depth {depth} is outside the available range 0..{len(levels) - 1}.")

    level = levels[depth]
    spheres = level.get("spheres", [])
    if not spheres:
        raise ValueError(f"Depth {depth} contains no active spheres.")

    centers = np.asarray([sphere["origin"] for sphere in spheres], dtype=np.float64)
    radii = np.asarray([sphere["radius"] for sphere in spheres], dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Every sphere origin must have exactly three coordinates.")
    if np.any(radii <= 0):
        raise ValueError("All active sphere radii must be positive.")
    return centers, radii, level


def sample_mesh_surface(
    mesh: trimesh.Trimesh, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Area-uniform triangle-surface sampling with an explicit random generator."""
    probabilities = mesh.area_faces / mesh.area
    face_ids = rng.choice(len(mesh.faces), size=count, p=probabilities)
    triangles = mesh.triangles[face_ids]

    random_values = rng.random((count, 2))
    root_u = np.sqrt(random_values[:, 0])
    v = random_values[:, 1]
    weights_a = 1.0 - root_u
    weights_b = root_u * (1.0 - v)
    weights_c = root_u * v
    return (
        weights_a[:, None] * triangles[:, 0]
        + weights_b[:, None] * triangles[:, 1]
        + weights_c[:, None] * triangles[:, 2]
    )


def sphere_union_sdf(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Return min_i(||point-center_i||-radius_i) for each point."""
    result = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        offsets = points[start:stop, None, :] - centers[None, :, :]
        distances = np.linalg.norm(offsets, axis=2) - radii[None, :]
        result[start:stop] = distances.min(axis=1)
    return result


def allocate_surface_samples(radii: np.ndarray, total: int) -> np.ndarray:
    areas = 4.0 * math.pi * np.square(radii)
    counts = np.maximum(16, np.rint(total * areas / areas.sum()).astype(int))
    return counts


def sample_exposed_sphere_union(
    centers: np.ndarray,
    radii: np.ndarray,
    total_samples: int,
    rng: np.random.Generator,
    overlap_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sample sphere surfaces and discard samples covered by another sphere.

    Returned weights estimate represented sphere-surface area. Sampling allocation is
    proportional to individual sphere area, and internal overlap surfaces are removed.
    """
    counts = allocate_surface_samples(radii, total_samples)
    exposed_points: list[np.ndarray] = []
    exposed_weights: list[np.ndarray] = []
    exposed_sphere_ids: list[np.ndarray] = []

    for sphere_index, (center, radius, count) in enumerate(zip(centers, radii, counts)):
        directions = rng.normal(size=(int(count), 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        points = center[None, :] + radius * directions

        offsets = points[:, None, :] - centers[None, :, :]
        margins = np.linalg.norm(offsets, axis=2) - radii[None, :]
        margins[:, sphere_index] = np.inf
        is_exposed = np.min(margins, axis=1) >= -overlap_tolerance

        if np.any(is_exposed):
            exposed_points.append(points[is_exposed])
            sphere_area = 4.0 * math.pi * radius * radius
            exposed_weights.append(
                np.full(np.count_nonzero(is_exposed), sphere_area / int(count))
            )
            exposed_sphere_ids.append(
                np.full(np.count_nonzero(is_exposed), sphere_index, dtype=np.int64)
            )

    if not exposed_points:
        raise RuntimeError("No exposed sphere-union surface samples were found.")
    return (
        np.concatenate(exposed_points, axis=0),
        np.concatenate(exposed_weights, axis=0),
        np.concatenate(exposed_sphere_ids, axis=0),
        int(counts.sum()),
    )


def closest_points_naive_chunked(
    mesh: trimesh.Trimesh, points: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    closest_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    triangle_parts: list[np.ndarray] = []
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        closest, distance, triangle_id = trimesh.proximity.closest_point_naive(
            mesh, points[start:stop]
        )
        closest_parts.append(closest)
        distance_parts.append(distance)
        triangle_parts.append(triangle_id)
    return (
        np.concatenate(closest_parts, axis=0),
        np.concatenate(distance_parts, axis=0),
        np.concatenate(triangle_parts, axis=0),
    )


def weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = percentile / 100.0 * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(values) - 1)
    return float(sorted_values[index])


def summarize(
    values: np.ndarray, scale: float, weights: np.ndarray | None = None
) -> dict[str, float | None]:
    if len(values) == 0:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    if weights is None:
        mean = float(np.mean(values))
        percentiles = np.percentile(values, [50, 95, 99])
    else:
        mean = float(np.average(values, weights=weights))
        percentiles = np.asarray(
            [weighted_percentile(values, weights, percentile) for percentile in (50, 95, 99)]
        )
    return {
        "mean": mean * scale,
        "p50": float(percentiles[0]) * scale,
        "p95": float(percentiles[1]) * scale,
        "p99": float(percentiles[2]) * scale,
        "max": float(np.max(values)) * scale,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    mesh_path = args.mesh.resolve()
    sphere_path = args.spheres.resolve()
    mesh = load_reference_mesh(mesh_path)
    centers, radii, foam_level = load_sphere_level(sphere_path, args.depth)
    scale_pivot = np.mean(mesh.bounds, axis=0)
    centers = scale_pivot + args.uniform_scale * (centers - scale_pivot)
    radii = args.uniform_scale * radii

    rng = np.random.default_rng(args.seed)
    tolerance_m = args.tolerance_mm / 1000.0

    mesh_points = sample_mesh_surface(mesh, args.mesh_samples, rng)
    mesh_sdf = sphere_union_sdf(mesh_points, centers, radii)
    undercoverage_gap = np.maximum(mesh_sdf, 0.0)
    uncovered = mesh_sdf > tolerance_m

    exposed_points, exposed_weights, exposed_sphere_ids, sphere_candidates = (
        sample_exposed_sphere_union(
            centers,
            radii,
            args.sphere_surface_samples,
            rng,
            args.overlap_tolerance_m,
        )
    )
    closest, unsigned_distance, triangle_id = closest_points_naive_chunked(
        mesh, exposed_points, args.closest_point_chunk
    )
    outward_normals = mesh.face_normals[triangle_id]
    local_signed_distance = np.einsum("ij,ij->i", exposed_points - closest, outward_normals)
    outside = local_signed_distance > tolerance_m
    overreach = np.maximum(local_signed_distance, 0.0)

    maximum_mesh_sdf_index = int(np.argmax(mesh_sdf))
    maximum_signed_distance_index = int(np.argmax(local_signed_distance))

    mesh_to_union = {
        "definition": "min_i(||mesh_point-center_i||-radius_i); positive is uncovered",
        "sample_count": int(len(mesh_points)),
        "coverage_tolerance_mm": args.tolerance_mm,
        "covered_fraction": float(np.mean(~uncovered)),
        "uncovered_fraction": float(np.mean(uncovered)),
        "signed_sphere_union_sdf_mm": summarize(mesh_sdf, 1000.0),
        "undercoverage_gap_mm_all_mesh_samples": summarize(undercoverage_gap, 1000.0),
        "undercoverage_gap_mm_uncovered_only": summarize(
            undercoverage_gap[uncovered], 1000.0
        ),
        "maximum_signed_sdf_sample": {
            "mesh_point_m": mesh_points[maximum_mesh_sdf_index].tolist(),
            "signed_sphere_union_sdf_mm": float(
                1000.0 * mesh_sdf[maximum_mesh_sdf_index]
            ),
            "uncovered_at_selected_tolerance": bool(uncovered[maximum_mesh_sdf_index]),
        },
    }

    union_to_mesh = {
        "definition": (
            "Closest triangle distance for exposed sphere-union samples; sign uses the "
            "nearest triangle outward normal, positive is outside Duck"
        ),
        "requested_candidate_samples": args.sphere_surface_samples,
        "actual_candidate_samples": sphere_candidates,
        "exposed_sample_count": int(len(exposed_points)),
        "exposed_acceptance_fraction": float(len(exposed_points) / sphere_candidates),
        "estimated_exposed_union_area_m2": float(np.sum(exposed_weights)),
        "overcoverage_tolerance_mm": args.tolerance_mm,
        "outside_surface_fraction": float(
            np.sum(exposed_weights[outside]) / np.sum(exposed_weights)
        ),
        "inside_or_on_surface_fraction": float(
            np.sum(exposed_weights[~outside]) / np.sum(exposed_weights)
        ),
        "unsigned_distance_mm": summarize(unsigned_distance, 1000.0, exposed_weights),
        "locally_signed_distance_mm": summarize(
            local_signed_distance, 1000.0, exposed_weights
        ),
        "overreach_mm_all_exposed_samples": summarize(overreach, 1000.0, exposed_weights),
        "overreach_mm_outside_only": summarize(
            overreach[outside], 1000.0, exposed_weights[outside]
        ),
        "maximum_locally_signed_distance_sample": {
            "sphere_union_point_m": exposed_points[
                maximum_signed_distance_index
            ].tolist(),
            "closest_mesh_point_m": closest[maximum_signed_distance_index].tolist(),
            "nearest_triangle_id": int(triangle_id[maximum_signed_distance_index]),
            "source_sphere_index": int(
                exposed_sphere_ids[maximum_signed_distance_index]
            ),
            "locally_signed_distance_mm": float(
                1000.0 * local_signed_distance[maximum_signed_distance_index]
            ),
            "outside_at_selected_tolerance": bool(outside[maximum_signed_distance_index]),
        },
    }

    return {
        "schema_version": 1,
        "reference_mesh": {
            "path": str(mesh_path),
            "sha256": sha256_file(mesh_path),
            "load_process": True,
            "vertices_after_welding": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "is_volume": bool(mesh.is_volume),
            "bounds_m": mesh.bounds.tolist(),
            "volume_m3": float(mesh.volume),
            "surface_area_m2": float(mesh.area),
        },
        "sphere_model": {
            "path": str(sphere_path),
            "sha256": sha256_file(sphere_path),
            "depth": args.depth,
            "sphere_count": int(len(radii)),
            "uniform_scale": args.uniform_scale,
            "uniform_scale_baked_into_level": float(
                foam_level.get("uniform_scale", 1.0)
            ),
            "scale_pivot": "reference mesh bounding-box center",
            "scale_pivot_m": scale_pivot.tolist(),
            "foam_reported_mean": foam_level.get("mean"),
            "foam_reported_best": foam_level.get("best"),
            "foam_reported_worst": foam_level.get("worst"),
            "foam_reported_metrics_apply_to_unscaled_level_only": bool(
                args.uniform_scale != 1.0
                or foam_level.get(
                    "foam_metrics_apply_to_unscaled_source_only", False
                )
            ),
            "radius_m": summarize(radii, 1.0),
        },
        "sampling": {
            "seed": args.seed,
            "mesh_surface_samples": args.mesh_samples,
            "sphere_surface_requested_samples": args.sphere_surface_samples,
            "closest_point_method": "trimesh.proximity.closest_point_naive",
            "closest_point_chunk": args.closest_point_chunk,
            "sphere_overlap_tolerance_m": args.overlap_tolerance_m,
        },
        "mesh_to_sphere_union": mesh_to_union,
        "sphere_union_to_mesh": union_to_mesh,
        "limitations": [
            "Metrics are sampled estimates, not analytic Hausdorff guarantees.",
            "Sphere-union outside/inside sign uses the nearest triangle normal.",
            "The input reference mesh is welded at load time but is not additionally smoothed "
            "or simplified by this evaluator.",
            "Negative sphere-union SDF at a mesh point is a coverage margin, not an exact "
            "distance to the boundary of an overlapping sphere union.",
        ],
    }


def print_summary(result: dict[str, Any]) -> None:
    sphere = result["sphere_model"]
    under = result["mesh_to_sphere_union"]
    over = result["sphere_union_to_mesh"]
    under_stats = under["undercoverage_gap_mm_uncovered_only"]
    over_stats = over["overreach_mm_outside_only"]

    print(f"Depth: {sphere['depth']}  active spheres: {sphere['sphere_count']}")
    print(f"Uniform sphere-model scale: {sphere['uniform_scale']:.6f}")
    print(
        f"Duck surface covered: {100.0 * under['covered_fraction']:.2f}%  "
        f"uncovered: {100.0 * under['uncovered_fraction']:.2f}%"
    )
    if under_stats["max"] is not None:
        print(
            "Uncovered gap (mm, uncovered samples): "
            f"mean={under_stats['mean']:.4f} p95={under_stats['p95']:.4f} "
            f"max={under_stats['max']:.4f}"
        )
    else:
        print("Uncovered gap: no uncovered mesh samples at the selected tolerance")
    print(
        f"Exposed sphere-union surface outside Duck: "
        f"{100.0 * over['outside_surface_fraction']:.2f}%"
    )
    if over_stats["max"] is not None:
        print(
            "Outside protrusion (mm, outside samples): "
            f"mean={over_stats['mean']:.4f} p95={over_stats['p95']:.4f} "
            f"max={over_stats['max']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="Reference STL/OBJ mesh")
    parser.add_argument("--spheres", type=Path, required=True, help="FOAM sphere JSON")
    parser.add_argument("--depth", type=int, required=True, help="Sphere-tree level to evaluate")
    parser.add_argument("--output", type=Path, help="Optional output metrics JSON")
    parser.add_argument("--mesh-samples", type=int, default=20_000)
    parser.add_argument("--sphere-surface-samples", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uniform-scale", type=float, default=1.0)
    parser.add_argument("--tolerance-mm", type=float, default=0.01)
    parser.add_argument("--overlap-tolerance-m", type=float, default=1e-9)
    parser.add_argument("--closest-point-chunk", type=int, default=64)
    args = parser.parse_args()
    if args.mesh_samples <= 0 or args.sphere_surface_samples <= 0:
        parser.error("Sample counts must be positive.")
    if args.closest_point_chunk <= 0:
        parser.error("--closest-point-chunk must be positive.")
    if args.uniform_scale <= 0.0:
        parser.error("--uniform-scale must be positive.")
    return args


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print_summary(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(f"Metrics written to: {args.output}")


if __name__ == "__main__":
    main()
