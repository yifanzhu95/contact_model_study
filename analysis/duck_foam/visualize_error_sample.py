#!/usr/bin/env python3
"""Visualize a measured worst overreach sample on a sphere approximation.

The Duck reference is gray, collision spheres are translucent blue, the measured sphere-union
point is red, its closest Duck point is green, and the yellow segment is the sampled protrusion
distance. Scale and pivot are read from an evaluator metrics JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from trimesh.transformations import translation_matrix
from trimesh.viewer import SceneViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--marker-radius-mm", type=float, default=1.5)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Construct and validate the scene without opening the interactive window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.metrics.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)

    reference = metrics["reference_mesh"]
    sphere_model = metrics["sphere_model"]
    worst = metrics["sphere_union_to_mesh"][
        "maximum_locally_signed_distance_sample"
    ]

    mesh_path = Path(reference["path"])
    sphere_path = Path(sphere_model["path"])
    mesh = trimesh.load(mesh_path, force="mesh", process=True)
    mesh.visual.face_colors = [205, 205, 205, 150]

    with sphere_path.open("r", encoding="utf-8") as stream:
        levels = json.load(stream)
    level = levels[int(sphere_model["depth"])]

    scale = float(sphere_model.get("uniform_scale", 1.0))
    pivot = np.asarray(
        sphere_model.get("scale_pivot_m", np.mean(mesh.bounds, axis=0)),
        dtype=np.float64,
    )
    scene = trimesh.Scene([mesh])

    for sphere in level["spheres"]:
        center = np.asarray(sphere["origin"], dtype=np.float64)
        center = pivot + scale * (center - pivot)
        radius = scale * float(sphere["radius"])
        geometry = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        geometry.visual.face_colors = [70, 145, 255, 65]
        scene.add_geometry(geometry, transform=translation_matrix(center))

    marker_radius = args.marker_radius_mm / 1000.0
    union_point = np.asarray(worst["sphere_union_point_m"], dtype=np.float64)
    mesh_point = np.asarray(worst["closest_mesh_point_m"], dtype=np.float64)

    red_marker = trimesh.creation.icosphere(subdivisions=2, radius=marker_radius)
    red_marker.visual.face_colors = [255, 30, 30, 255]
    scene.add_geometry(red_marker, transform=translation_matrix(union_point))

    green_marker = trimesh.creation.icosphere(
        subdivisions=2, radius=0.85 * marker_radius
    )
    green_marker.visual.face_colors = [30, 230, 80, 255]
    scene.add_geometry(green_marker, transform=translation_matrix(mesh_point))

    segment = trimesh.load_path(np.vstack([union_point, mesh_point]))
    segment.colors = np.array([[255, 220, 0, 255]], dtype=np.uint8)
    scene.add_geometry(segment)
    scene.add_geometry(
        trimesh.creation.axis(origin_size=0.0007, axis_length=0.025)
    )

    print(f"Depth: {sphere_model['depth']}  spheres: {sphere_model['sphere_count']}")
    print(f"Uniform scale: {scale}")
    print(f"Worst sampled overreach: {worst['locally_signed_distance_mm']:.4f} mm")
    print(f"Sphere-union point: {union_point.tolist()}")
    print(f"Closest Duck point: {mesh_point.tolist()}")
    print("Red = sphere boundary, green = Duck surface, yellow = measured distance")
    print(f"Scene geometries: {len(scene.geometry)}")
    if not args.no_show:
        SceneViewer(scene)


if __name__ == "__main__":
    main()
