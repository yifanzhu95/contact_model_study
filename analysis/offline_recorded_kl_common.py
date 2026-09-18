"""Pure utilities shared by the recorded-log offline KL workflow."""
from __future__ import annotations

import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


SCHEMA = "per_state_zero_offline_kl_episode_v1"
DEFAULT_MAX_MEASUREMENTS = 10
DEFAULT_REFERENCE_SAMPLES = 4096
DEFAULT_REFERENCE_TEMPERATURE = 50.0
DEFAULT_CONVERGENCE_TOL = 1e-3
DEFAULT_MAX_ITERATIONS = 25
DEFAULT_SHRINKAGE = 0.001

SCIENTIFIC_RUNTIME_FILES = (
    "contact_study/contact_models/__init__.py",
    "contact_study/contact_models/api.py",
    "contact_study/contact_models/config.py",
    "contact_study/contact_models/xpbd_backend.py",
    "contact_study/drivers/run_eval_episode.py",
    "contact_study/evaluation/distributions.py",
    "contact_study/planners/__init__.py",
    "contact_study/planners/base.py",
    "contact_study/planners/mppi.py",
    "contact_study/tasks/__init__.py",
    "contact_study/tasks/base.py",
    "contact_study/tasks/config.py",
    "contact_study/tasks/grasp_reorient.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def encode_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value) -> None:
    """Write strict JSON and atomically publish it at ``path``."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=encode_json, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def mapping_digest(mapping: dict) -> str:
    """Digest a JSON-compatible mapping independently of insertion order."""
    return sha256_bytes(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def array_digest(value, dtype=None) -> str:
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def output_filename(source_json: Path | str, episode: int) -> str:
    return f"{Path(source_json).stem}_ep_{episode:03d}_zero_restart_kl.json"


def resolved_grasp_reorient_rollout_scene(repo: Path, geometry: str) -> Path:
    """Resolve the recorded grasp-reorientation rollout scene convention."""
    value = str(geometry).strip()
    if not value or value in {"accurate", "convex_hull", "primitive_union", "linearized"}:
        obj, hand_accuracy, object_accuracy = "cube", "low", "high"
    else:
        parts = value.split("_")
        if len(parts) == 1:
            obj, hand_accuracy, object_accuracy = parts[0], "low", "high"
        elif len(parts) == 3:
            obj, hand_accuracy, object_accuracy = parts
        else:
            raise ValueError(f"Unsupported grasp-reorientation geometry: {geometry!r}")
    path = Path(repo) / "scenes" / "leap" / (
        f"env_leap_rollout_{obj}_{hand_accuracy}_{object_accuracy}.xml"
    )
    require(path.is_file(), f"Resolved rollout scene does not exist: {path}")
    return path.resolve()


def _resolve_scene_reference(entrypoint: Path, source_xml: Path, reference: str,
                             candidate_directories: list[Path]) -> Path:
    path = Path(reference)
    candidates = [path] if path.is_absolute() else [
        source_xml.parent / path,
        *(directory / path for directory in candidate_directories),
        entrypoint.parent / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(
        f"Could not resolve scene asset {reference!r} referenced by {source_xml}"
    )


def scene_asset_provenance(repo: Path, entrypoint: Path) -> dict:
    """Hash the resolved MJCF entrypoint, includes, and file-backed assets."""
    repo, entrypoint = Path(repo).resolve(), Path(entrypoint).resolve()
    xml_files, parsed, pending = [], {}, [entrypoint]
    while pending:
        path = pending.pop()
        if path in parsed:
            continue
        require(path.is_file(), f"Missing scene XML: {path}")
        root = ET.parse(path).getroot()
        parsed[path] = root
        xml_files.append(path)
        for element in root.iter("include"):
            included = _resolve_scene_reference(entrypoint, path, element.attrib["file"], [])
            pending.append(included)

    directory_attributes = {"assetdir": [], "meshdir": [], "texturedir": []}
    for root in parsed.values():
        for compiler in root.iter("compiler"):
            for name in directory_attributes:
                if compiler.get(name):
                    raw = Path(compiler.get(name))
                    directory_attributes[name].append(
                        raw.resolve() if raw.is_absolute() else (entrypoint.parent / raw).resolve()
                    )

    assets = set(xml_files)
    tag_directories = {
        "mesh": directory_attributes["meshdir"] + directory_attributes["assetdir"],
        "skin": directory_attributes["meshdir"] + directory_attributes["assetdir"],
        "texture": directory_attributes["texturedir"] + directory_attributes["assetdir"],
        "hfield": directory_attributes["assetdir"],
    }
    for source_xml, root in parsed.items():
        for tag, directories in tag_directories.items():
            for element in root.iter(tag):
                if element.get("file"):
                    assets.add(_resolve_scene_reference(
                        entrypoint, source_xml, element.get("file"), directories
                    ))

    def label(path):
        try:
            return path.relative_to(repo).as_posix()
        except ValueError:
            return str(path)

    file_hashes = {label(path): sha256_file(path) for path in sorted(assets)}
    return {
        "entrypoint": label(entrypoint),
        "entrypoint_sha256": sha256_file(entrypoint),
        "resolved_file_count": len(file_hashes),
        "resolved_files_sha256": file_hashes,
        "resolved_bundle_sha256": mapping_digest(file_hashes),
    }


def external_backend_provenance() -> dict:
    """Conservatively hash the complete Python source tree of the rollout backend."""
    specification = util.find_spec("comfree_warp")
    require(specification is not None and specification.submodule_search_locations,
            "Could not locate the comfree_warp backend package")
    root = Path(next(iter(specification.submodule_search_locations))).resolve()
    files = sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )
    require(bool(files), f"No Python backend sources found under {root}")
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in files}
    return {
        "package": "comfree_warp",
        "python_file_count": len(hashes),
        "python_tree_sha256": mapping_digest(hashes),
        "scope": "complete installed comfree_warp Python source tree",
    }


def scientific_runtime_provenance(repo: Path, geometry: str, model: str) -> dict:
    """Build a conservative digest of code and assets that can affect KL values."""
    repo = Path(repo).resolve()
    local_hashes = {name: sha256_file(repo / name) for name in SCIENTIFIC_RUNTIME_FILES}
    scene = scene_asset_provenance(
        repo, resolved_grasp_reorient_rollout_scene(repo, geometry)
    )
    versions = {}
    for distribution in ("numpy", "scipy", "mujoco", "warp-lang", "comfree-warp"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    result = {
        "contact_model": model,
        "geometry": geometry,
        "repository_files_sha256": local_hashes,
        "resolved_scene_assets": scene,
        "external_backend": external_backend_provenance(),
        "scientific_package_versions": versions,
        "scope": (
            "planner, KL, task/goal reconstruction, contact-model dispatch/config/local "
            "backend, installed rollout-backend Python sources, and resolved scene assets"
        ),
    }
    result["workflow_sha256"] = mapping_digest(result)
    return result


def evenly_spaced_unique_steps(recorded_steps, max_measurements: int) -> list[int]:
    """Choose outcome-blind rows spread across the full recorded episode."""
    require(max_measurements >= 1, "max_measurements must be at least one")
    steps = np.asarray(recorded_steps, dtype=np.int64)
    require(steps.ndim == 1 and len(steps), "Recorded planner steps must be non-empty")
    require(np.all(np.diff(steps) > 0), "Recorded planner steps must be unique and increasing")
    count = min(max_measurements, len(steps))
    if count == len(steps):
        chosen = steps
    elif count == 1:
        chosen = steps[:1]
    else:
        row_indices = np.rint(np.linspace(0, len(steps) - 1, count)).astype(int)
        require(len(np.unique(row_indices)) == count, "Even spacing produced duplicate rows")
        chosen = steps[row_indices]
    require(len(np.unique(chosen)) == count, "Selected step values are not unique")
    return [int(step) for step in chosen]


def independent_gaussian_kl(mean_p, covariance_p, mean_q, covariance_q) -> float:
    """Direct linear-solve Gaussian KL used as an independent audit formula."""
    mean_p = np.asarray(mean_p, dtype=np.float64)
    mean_q = np.asarray(mean_q, dtype=np.float64)
    covariance_p = np.asarray(covariance_p, dtype=np.float64)
    covariance_q = np.asarray(covariance_q, dtype=np.float64)
    sign_p, logdet_p = np.linalg.slogdet(covariance_p)
    sign_q, logdet_q = np.linalg.slogdet(covariance_q)
    require(sign_p == sign_q == 1, "KL covariance is not positive definite")
    delta = mean_q - mean_p
    return float(
        0.5
        * (
            np.trace(np.linalg.solve(covariance_q, covariance_p))
            + delta @ np.linalg.solve(covariance_q, delta)
            - len(delta)
            + logdet_q
            - logdet_p
        )
    )


def load_cells_manifest(path: Path) -> tuple[dict, list[dict]]:
    payload = Path(path).read_bytes()
    document = json.loads(payload)
    cells = document.get("cells")
    require(isinstance(cells, list) and cells, "Input manifest must contain a non-empty cells list")
    required = {"file", "sha256", "model", "geometry", "config", "n", "success"}
    for cell in cells:
        require(required <= set(cell), f"Manifest cell lacks required fields: {cell}")
        require(Path(cell["file"]).name == cell["file"], "Cell file must be a basename")
    require(len({cell["file"] for cell in cells}) == len(cells), "Duplicate cell files in manifest")
    return document, cells
