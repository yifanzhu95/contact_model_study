"""Fill in the facet normals of a binary STL that stores zeros for them.

Most of the STLs under scenes/leap/objects were exported with a zero vector in
every facet's normal slot. MuJoCo does not care — it recomputes normals from the
winding at compile time — but the Pinocchio eval viewer does: Panda3d's assimp
importer hands those zeros straight to the shader, so N.L is zero everywhere and
the mesh renders with ambient light only. That is a flat, unshaded silhouette,
far darker than the same colour in MuJoCo (the rubber duck's yellow came out a
brown blob). assimp-gen-normals does not help — it only fills normals that are
ABSENT, and these are present, just degenerate.

The fix is data-side: compute each facet's normal from its own vertex winding
and write it into the file. Geometry is untouched — vertex bytes are copied
through verbatim, so collision, inertia and every other consumer see exactly the
mesh they saw before; only the normal field changes.

Idempotent: a file whose normals are already unit length is left alone (and
reported as such), so re-running over a directory is safe.

    python scenes/scripts/fix_stl_normals.py scenes/leap/objects/rubber_duck.stl
    python scenes/scripts/fix_stl_normals.py --check scenes/leap/objects/*.stl
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

# Binary STL facet record: normal, 3 vertices, attribute byte count.
FACET = np.dtype([("n", "<3f4"), ("v", "<3,3f4"), ("attr", "<u2")])
HEADER = 84  # 80-byte header + uint32 facet count


def read_facets(path: Path):
    """(header_bytes, facet_array) for a binary STL, or (None, None) if the file
    is not one (an ASCII STL, or a truncated//mislabelled file)."""
    raw = path.read_bytes()
    if len(raw) < HEADER:
        return None, None
    count = struct.unpack("<I", raw[80:84])[0]
    if HEADER + FACET.itemsize * count != len(raw):
        return None, None
    return raw[:80], np.frombuffer(raw[HEADER:], dtype=FACET).copy()


def facet_normals(facets) -> np.ndarray:
    """Unit normal per facet from its winding (right-hand rule, matching how
    MuJoCo and every mesh tool derive an STL's orientation)."""
    v = facets["v"].astype(np.float64)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    # Degenerate (zero-area) facets have no direction to point in; leave them at
    # zero rather than dividing by ~0 and writing garbage.
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, length, out=np.zeros_like(n), where=length > 0)


def fix(path: Path, check_only: bool = False) -> bool:
    """Repair `path` in place. Returns True if it needed (or would need) it."""
    header, facets = read_facets(path)
    if facets is None:
        print(f"  skip     {path}  (not a binary STL)")
        return False

    mean_len = float(np.linalg.norm(facets["n"].astype(np.float64), axis=1).mean())
    if mean_len > 0.5:                     # already unit-ish normals
        print(f"  ok       {path}  (mean |N| = {mean_len:.3f})")
        return False

    if check_only:
        print(f"  NEEDS FIX {path}  (mean |N| = {mean_len:.3g}, {len(facets)} facets)")
        return True

    facets["n"] = facet_normals(facets).astype(np.float32)
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(facets)))
        f.write(facets.tobytes())
    print(f"  fixed    {path}  ({len(facets)} facets, mean |N| = "
          f"{float(np.linalg.norm(facets['n'].astype(np.float64), axis=1).mean()):.3f})")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="binary .stl files to repair")
    p.add_argument("--check", action="store_true",
                   help="report which files have zeroed normals without writing.")
    args = p.parse_args()

    n = sum(fix(path, args.check) for path in args.paths)
    verb = "need fixing" if args.check else "fixed"
    print(f"{n} of {len(args.paths)} file(s) {verb}")


if __name__ == "__main__":
    main()
