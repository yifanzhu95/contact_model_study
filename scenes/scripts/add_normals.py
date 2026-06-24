#!/usr/bin/env python3
"""Add per-vertex normals to .obj files that don't already have them.

Computes area-weighted vertex normals from face geometry, inserts the
resulting `vn` lines after the vertex block, and rewrites each `f` line
to reference the matching normal index. Everything else in the file
(materials, texture coords, comments, ordering) is left untouched, and
the file is saved back to its original path.
"""
import argparse
import os

import numpy as np


def parse_face_token(token):
    """Parse 'v', 'v/vt', 'v/vt/vn', or 'v//vn' -> (v, vt)."""
    parts = token.split("/")
    v = int(parts[0])
    vt = int(parts[1]) if len(parts) > 1 and parts[1] != "" else None
    return v, vt


def compute_vertex_normals(vertices, faces):
    """faces: list of lists of 1-based vertex indices."""
    normals = np.zeros_like(vertices)
    for face in faces:
        idx = [i - 1 for i in face]
        pts = vertices[idx]
        n = np.zeros(3)
        for i in range(len(pts)):
            p0, p1 = pts[i], pts[(i + 1) % len(pts)]
            n[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
            n[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
            n[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])
        for i in idx:
            normals[i] += n  # Newell vector ~ proportional to face area
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths < 1e-12] = 1.0
    return normals / lengths[:, None]


def add_normals_to_obj(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    if any(line.startswith("vn ") for line in lines):
        print(f"Skipping {filepath}: normals already present")
        return

    vertices = []
    faces = []
    last_v_line_idx = -1

    for i, line in enumerate(lines):
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
            last_v_line_idx = i
        elif line.startswith("f "):
            faces.append([parse_face_token(t) for t in line.split()[1:]])

    if not faces:
        print(f"Skipping {filepath}: no faces found")
        return

    vertices = np.array(vertices)
    vertex_normals = compute_vertex_normals(vertices, [[v for v, _ in face] for face in faces])

    new_lines = []
    face_i = 0
    for i, line in enumerate(lines):
        if line.startswith("f "):
            new_tokens = []
            for v, vt in faces[face_i]:
                new_tokens.append(f"{v}/{vt}/{v}" if vt is not None else f"{v}//{v}")
            new_lines.append("f " + " ".join(new_tokens) + "\n")
            face_i += 1
        else:
            new_lines.append(line)

        if i == last_v_line_idx:
            for n in vertex_normals:
                new_lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

    with open(filepath, "w") as f:
        f.writelines(new_lines)
    print(f"Added normals to {filepath}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="A .obj file, or a directory of .obj files")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        for fname in sorted(os.listdir(args.path)):
            if fname.lower().endswith(".obj"):
                add_normals_to_obj(os.path.join(args.path, fname))
    else:
        add_normals_to_obj(args.path)


if __name__ == "__main__":
    main()
