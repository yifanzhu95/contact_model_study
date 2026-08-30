#!/usr/bin/env python3
"""Validate generated Duck sphere scenes against their manifest and source JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "analysis/duck_foam/mujoco_sphere_scenes.json"
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_collision_geom_ids(model: mujoco.MjModel) -> list[int]:
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] == object_id
        and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
    ]


def parse_vector(text: str, length: int) -> np.ndarray:
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.shape != (length,):
        raise ValueError(f"Expected {length} numeric values, got {text!r}.")
    return values


def validate_physics_invariants(
    model: mujoco.MjModel,
    expected_inertial: dict[str, str],
) -> int:
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
    np.testing.assert_allclose(model.body_mass[object_id], float(expected_inertial["mass"]))
    np.testing.assert_allclose(
        model.body_inertia[object_id], parse_vector(expected_inertial["diaginertia"], 3)
    )
    np.testing.assert_allclose(
        model.body_ipos[object_id], parse_vector(expected_inertial["pos"], 3)
    )
    np.testing.assert_allclose(
        model.body_iquat[object_id], parse_vector(expected_inertial["quat"], 4)
    )
    assert model.nq == 23
    assert model.nv == 22
    assert model.nu == 16
    return object_id


def validate_record(record: dict[str, Any]) -> None:
    scene_path = REPO_ROOT / record["scene"]
    sphere_path = REPO_ROOT / record["sphere_source"]
    assert sha256_file(scene_path) == record["scene_sha256"]
    assert sha256_file(sphere_path) == record["sphere_source_sha256"]

    levels = json.loads(sphere_path.read_text(encoding="utf-8"))
    source_spheres = levels[int(record["depth"])]["spheres"]
    offset = np.asarray(record["object_geom_offset_m"], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    validate_physics_invariants(model, record["object_inertial"])
    collision_ids = object_collision_geom_ids(model)
    assert len(collision_ids) == len(source_spheres) == int(record["sphere_count"])

    for sphere_index, (geom_id, source) in enumerate(zip(collision_ids, source_spheres)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        assert name == f"obj_sphere_{sphere_index:03d}"
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
        np.testing.assert_allclose(
            model.geom_pos[geom_id],
            np.asarray(source["origin"], dtype=np.float64) + offset,
            rtol=0.0,
            atol=5e-13,
        )
        np.testing.assert_allclose(
            model.geom_size[geom_id, 0],
            float(source["radius"]),
            rtol=0.0,
            atol=5e-13,
        )
        np.testing.assert_allclose(model.geom_friction[geom_id, 0], 0.5)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.qacc).all()


def validate_eval_reference(manifest: dict[str, Any]) -> None:
    reference = manifest["fixed_eval_reference"]
    scene_path = REPO_ROOT / reference["scene"]
    assert sha256_file(scene_path) == reference["scene_sha256"]
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    generated = manifest["generated_scenes"]
    if not generated:
        raise ValueError("Manifest contains no generated rollout scenes.")
    validate_physics_invariants(model, generated[0]["object_inertial"])
    collision_ids = object_collision_geom_ids(model)
    assert len(collision_ids) == 8
    for geom_id in collision_ids:
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    generator_path = REPO_ROOT / manifest["generator"]
    assert sha256_file(generator_path) == manifest["generator_sha256"]
    records = manifest["generated_scenes"]
    for record in records:
        validate_record(record)
        print(
            f"OK {record['selector']}: {record['sphere_count']} sphere geoms, "
            "fixed mass/inertia"
        )
    validate_eval_reference(manifest)
    print("OK eval reference: 8 mesh hull geoms, fixed mass/inertia")
    print(f"Validated {len(records)} rollout scenes and one fixed eval scene.")


if __name__ == "__main__":
    main()
