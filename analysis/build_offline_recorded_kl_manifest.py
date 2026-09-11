"""Build a validated input manifest from recorded synchronous cell JSONs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))

from offline_recorded_kl_common import atomic_json, require, sha256_bytes


SCHEMA = "offline_recorded_kl_input_manifest_v1"

REQUIRED_CONTEXT_FIELDS = {
    "goal_difficulty", "goal_switch_steps", "warm_start", "time_constrained",
    "shrinkage", "action_source", "planner", "resample_per_iteration",
    "rollout_dt", "q0", "v0", "u0", "horizon", "substeps", "nu",
    "ctrl_relative_to_qpos", "noise_sigma",
}
REQUIRED_STATE_FIELDS = {"step", "qpos", "qvel", "ctrl"}
REQUIRED_DISTRIBUTION_FIELDS = {"step", "mu", "cov", "ess", "degenerate"}


def describe_cell(path: Path) -> dict:
    payload = path.read_bytes()
    source = json.loads(payload)
    required = {
        "driver", "task", "model", "geometry", "planner", "planner_kwargs",
        "n_episodes", "episodes", "seed", "combo_index", "full_weights",
    }
    require(required <= set(source), f"{path.name} lacks recorded-cell fields")
    require(source["driver"] == "sync", f"{path.name} is not a synchronous cell")
    require(source["task"] == "grasp_reorient",
            f"{path.name} is not a supported grasp_reorient cell")
    require(source["planner"] == "mppi", f"{path.name} is not an MPPI cell")
    episodes = source["episodes"]
    require(isinstance(episodes, list), f"{path.name}.episodes is not a list")
    require(bool(episodes), f"{path.name} contains no episodes")
    require(len(episodes) == source["n_episodes"],
            f"{path.name} episode count disagrees with n_episodes")
    require(all(type(episode.get("success")) is bool for episode in episodes),
            f"{path.name} has a missing or non-boolean episode success value")
    for index, episode in enumerate(episodes):
        require("end_reason" in episode and "trajectory" in episode,
                f"{path.name} episode {index} lacks outcome or trajectory data")
        trajectory = episode["trajectory"]
        require(isinstance(trajectory, dict),
                f"{path.name} episode {index} trajectory is not an object")
        for name, fields in (
            ("context", REQUIRED_CONTEXT_FIELDS),
            ("steps", REQUIRED_STATE_FIELDS),
            ("planner_dist", REQUIRED_DISTRIBUTION_FIELDS),
        ):
            value = trajectory.get(name)
            require(isinstance(value, dict) and fields <= set(value),
                    f"{path.name} episode {index} lacks required {name} fields")
        require(bool(trajectory["steps"]["step"]),
                f"{path.name} episode {index} contains no recorded states")
        require(bool(trajectory["planner_dist"]["step"]),
                f"{path.name} episode {index} contains no recorded planner distributions")
    successes = sum(episode["success"] for episode in episodes)
    if source.get("n_success") is not None:
        require(source["n_success"] == successes,
                f"{path.name} n_success disagrees with episode outcomes")
    return {
        "file": path.name,
        "sha256": sha256_bytes(payload),
        "task": source["task"],
        "model": source["model"],
        "geometry": source["geometry"],
        "config": source["planner_kwargs"],
        "n": len(episodes),
        "success": successes,
    }


def build_manifest(source_dir: Path, output: Path, pattern: str = "cell_*.json") -> dict:
    source_dir, output = Path(source_dir), Path(output)
    require(source_dir.is_dir(), f"Source directory does not exist: {source_dir}")
    paths = sorted(path for path in source_dir.glob(pattern) if path.is_file())
    require(bool(paths), f"No files matching {pattern!r} under {source_dir}")
    require(all(path.parent == source_dir for path in paths),
            "Source pattern must select direct children of source-dir")
    require(not output.exists(), f"Refusing to overwrite existing manifest: {output}")
    cells = [describe_cell(path) for path in paths]
    require(len({cell["file"] for cell in cells}) == len(cells),
            "Duplicate source filenames")
    identities = [json.dumps({
        "task": cell["task"], "model": cell["model"],
        "geometry": cell["geometry"], "config": cell["config"],
    }, sort_keys=True, separators=(",", ":")) for cell in cells]
    require(len(set(identities)) == len(identities),
            "Multiple files describe the same task/model/geometry/planner configuration")
    manifest = {
        "schema": SCHEMA,
        "selection": f"sorted files matching {pattern!r}",
        "cells": cells,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pattern", default="cell_*.json")
    args = parser.parse_args(argv)
    manifest = build_manifest(args.source_dir, args.out, args.pattern)
    print(json.dumps({
        "manifest": str(args.out),
        "cells": len(manifest["cells"]),
        "episodes": sum(cell["n"] for cell in manifest["cells"]),
    }, indent=2))


if __name__ == "__main__":
    main()
