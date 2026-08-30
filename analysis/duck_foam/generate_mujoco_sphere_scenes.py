#!/usr/bin/env python3
"""Generate MuJoCo Duck rollout scenes from calibrated FOAM sphere levels.

The generated scenes keep the existing Duck visual mesh, body frame, explicit mass,
center of mass, inertia, friction, and goal marker. Only the eight convex collision
hulls in a rollout template are replaced by calibrated sphere geoms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = REPO_ROOT / "scenes/leap"
EXPLORATION_DIR = REPO_ROOT / "analysis/duck_foam/exploration"
HAND_ACCURACIES = ("low", "med", "high")


@dataclass(frozen=True)
class SphereCandidate:
    object_accuracy: str
    role: str
    source_name: str
    depth: int


CANDIDATES = (
    SphereCandidate(
        object_accuracy="foam4",
        role="Low",
        source_name="duck_foam_safe_l6000_branch4_calibrated.json",
        depth=1,
    ),
    SphereCandidate(
        object_accuracy="foam16a",
        role="Medium-A",
        source_name="duck_foam_safe_l6000_branch4_calibrated.json",
        depth=2,
    ),
    SphereCandidate(
        object_accuracy="foam16b",
        role="Medium-B",
        source_name="duck_branch4_medium16_scale096_alternative.json",
        depth=2,
    ),
    SphereCandidate(
        object_accuracy="foam64",
        role="High",
        source_name="duck_foam_safe_l6000_branch4_calibrated.json",
        depth=3,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_number(value: float) -> str:
    return f"{value:.12g}"


def parse_xml_with_comments(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def remove_named_children(parent: ET.Element, prefix: str) -> int:
    removed = 0
    for child in list(parent):
        if child.get("name", "").startswith(prefix):
            parent.remove(child)
            removed += 1
    return removed


def find_object_body(root: ET.Element) -> ET.Element:
    for body in root.findall(".//body"):
        if body.get("name") == "obj":
            return body
    raise ValueError("Template has no <body name='obj'>.")


def parse_vector_attribute(
    element: ET.Element,
    attribute: str,
    length: int,
) -> tuple[float, ...]:
    raw = element.get(attribute)
    if raw is None:
        return (0.0,) * length
    values = tuple(float(value) for value in raw.split())
    if len(values) != length:
        raise ValueError(
            f"Expected {length} values in {element.tag} {attribute!r}; got {raw!r}."
        )
    return values


def update_object_comments(object_body: ET.Element) -> None:
    for child in object_body:
        if child.tag is not ET.Comment or not child.text:
            continue
        if "Explicit inertia." in child.text:
            child.text = (
                " Explicit inertia shared by every Duck collision variant. MuJoCo "
                "therefore keeps the same mass, center of mass, principal axes, and "
                "diagonal inertia regardless of the number or shape of collision geoms. "
            )
        elif "Visual-only duck" in child.text:
            child.text = (
                " Visual-only Duck: the calibrated spheres above carry contact, while "
                "this mesh shows the original shape and contributes no mass or collision. "
            )
        elif "Every duck geom is offset" in child.text:
            child.text = (
                " Every Duck collision center and the visual mesh use the same "
                "body-frame placement as the original 8-hull rollout template. "
            )


def read_sphere_level(path: Path, depth: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    levels = json.loads(path.read_text(encoding="utf-8"))
    try:
        level = levels[depth]
    except IndexError as error:
        raise ValueError(f"{path} has no sphere depth {depth}.") from error
    spheres = level.get("spheres")
    if not isinstance(spheres, list) or not spheres:
        raise ValueError(f"{path} depth {depth} has no spheres.")
    return level, spheres


def add_sphere_geoms(
    object_body: ET.Element,
    spheres: list[dict[str, Any]],
    object_geom_offset_m: tuple[float, ...],
) -> None:
    visual = object_body.find("./geom[@name='obj_visual']")
    if visual is None:
        raise ValueError("Template object body has no obj_visual geom.")
    insertion_index = list(object_body).index(visual)
    note = ET.Comment(
        " Calibrated FOAM rollout collision spheres. Positions use the same "
        "object-frame offset as the template visual mesh. The explicit inertial above "
        "fixes mass and inertia; density=0 prevents geometry-dependent dynamics. "
    )
    object_body.insert(insertion_index, note)
    insertion_index += 1

    for sphere_index, sphere in enumerate(spheres):
        origin = [float(value) for value in sphere["origin"]]
        center = [
            origin[axis] + object_geom_offset_m[axis]
            for axis in range(3)
        ]
        geom = ET.Element(
            "geom",
            {
                "name": f"obj_sphere_{sphere_index:03d}",
                "type": "sphere",
                "size": format_number(float(sphere["radius"])),
                "pos": " ".join(format_number(value) for value in center),
                "group": "3",
                "friction": "0.5",
                "density": "0",
                "rgba": "0.18 0.55 0.95 0.35",
            },
        )
        object_body.insert(insertion_index, geom)
        insertion_index += 1


def generate_scene(
    template_path: Path,
    sphere_path: Path,
    depth: int,
    selector: str,
    output_path: Path,
) -> dict[str, Any]:
    tree = parse_xml_with_comments(template_path)
    root = tree.getroot()
    root.set("model", f"right_leap_hand scene {selector}")

    asset = root.find("asset")
    if asset is None:
        raise ValueError(f"{template_path} has no <asset> section.")
    removed_assets = remove_named_children(asset, "duck_hull_")

    object_body = find_object_body(root)
    visual = object_body.find("./geom[@name='obj_visual']")
    if visual is None:
        raise ValueError("Template object body has no obj_visual geom.")
    object_geom_offset_m = parse_vector_attribute(visual, "pos", 3)
    update_object_comments(object_body)
    removed_hulls = remove_named_children(object_body, "obj_hull_")
    if removed_assets != 8 or removed_hulls != 8:
        raise ValueError(
            f"Expected eight Duck hull assets/geoms in {template_path}; "
            f"removed {removed_assets} assets and {removed_hulls} geoms."
        )

    level, spheres = read_sphere_level(sphere_path, depth)
    add_sphere_geoms(object_body, spheres, object_geom_offset_m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    xml_text = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    output_path.write_text(xml_text + "\n", encoding="utf-8")

    inertial = object_body.find("inertial")
    if inertial is None:
        raise ValueError("Template object body has no explicit inertial block.")
    return {
        "selector": selector,
        "scene": str(output_path.relative_to(REPO_ROOT)),
        "scene_sha256": sha256_file(output_path),
        "template": str(template_path.relative_to(REPO_ROOT)),
        "template_sha256": sha256_file(template_path),
        "sphere_source": str(sphere_path.relative_to(REPO_ROOT)),
        "sphere_source_sha256": sha256_file(sphere_path),
        "depth": depth,
        "sphere_count": len(spheres),
        "baked_uniform_scale": float(level.get("uniform_scale", 1.0)),
        "object_geom_offset_m": list(object_geom_offset_m),
        "object_inertial": dict(inertial.attrib),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hand-accuracy",
        action="append",
        choices=HAND_ACCURACIES,
        help="Generate only one hand accuracy; may be repeated (default: all)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "analysis/duck_foam/mujoco_sphere_scenes.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hand_accuracies = tuple(args.hand_accuracy or HAND_ACCURACIES)
    records: list[dict[str, Any]] = []

    for hand_accuracy in hand_accuracies:
        template = SCENE_DIR / f"env_leap_rollout_duck_{hand_accuracy}_high.xml"
        for candidate in CANDIDATES:
            selector = f"duck_{hand_accuracy}_{candidate.object_accuracy}"
            output = SCENE_DIR / f"env_leap_rollout_{selector}.xml"
            sphere_source = EXPLORATION_DIR / candidate.source_name
            record = generate_scene(
                template_path=template,
                sphere_path=sphere_source,
                depth=candidate.depth,
                selector=selector,
                output_path=output,
            )
            record["candidate_role"] = candidate.role
            records.append(record)
            print(
                f"Generated {selector}: {record['sphere_count']} spheres -> "
                f"{record['scene']}"
            )

    eval_scene = SCENE_DIR / "env_leap_eval_duck.xml"
    manifest = {
        "schema_version": 1,
        "generator": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "principle": (
            "Vary rollout Duck collision spheres while keeping visual geometry, "
            "mass, center of mass, inertia, friction, object frame, and eval scene fixed."
        ),
        "recommended_first_comparison": {
            "fixed_hand_accuracy": "low",
            "selectors": [
                f"duck_low_{candidate.object_accuracy}" for candidate in CANDIDATES
            ],
        },
        "fixed_eval_reference": {
            "scene": str(eval_scene.relative_to(REPO_ROOT)),
            "scene_sha256": sha256_file(eval_scene),
            "duck_collision_geometry": "8 convex hull mesh geoms",
        },
        "generated_scenes": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest written to: {args.manifest}")


if __name__ == "__main__":
    main()
