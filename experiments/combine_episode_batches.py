#!/usr/bin/env python3
"""Combine the results folders of several episode batches into one new batch folder.

Use it when a batch was partly rerun into a folder of its own, e.g. a second
batch that redid only one model's rows. The folders are given in order of
precedence, lowest first: the first is the base, and each cell of a later
folder replaces the cell it repeats in an earlier one::

    python experiments/combine_episode_batches.py results/episode_batches_final_123 \\
        results/episode_batches_final_M4_456 --out results/episode_batches_final_combined
    python experiments/combine_episode_batches.py <base> <rerun> --out <new> --dry-run   # only print the plan
    python experiments/combine_episode_batches.py <base> <rerun> --out <new> --symlink   # link the .npy/.mp4

The new folder is a batch folder like any other, so it can be given to
``process_batch_results.py`` (or ``hpc/submit_process_batch_results.sh``).

Which cells repeat
------------------
Episode ids are random, so a repeat is found by cell, not by episode: two
cells repeat each other when their rows have the same ``label``, or, for rows
without one, the same settings. A repeating cell replaces the earlier one
whole, every episode of it. A later cell that is not ``done`` does not replace
an earlier one that is (it is reported instead). Cells found in only one folder
are kept.

Output
------
The cells are numbered again from 0: the base folder's in its order, then the
cells only later folders have, in theirs. Each cell's files (``.json``,
``.status.json``, ``.log``, videos and the per-episode ``.npy``) are copied
under the new name, and the ``.npy`` names inside the ``.json`` and the paths
in the ``.status.json`` are rewritten to match. The folder also gets:

* ``batches.csv``: the cells' rows, so ``run_episode_batches.py batches.csv
  --summarize --outdir <new>`` works on it;
* ``summary.csv`` / ``summary.json``: as ``run_episode_batches.py --summarize``
  writes them;
* ``combined.json``: which folder and cell each new cell came from and the
  cells it replaced.

An ``analysis/`` folder from processing an input is not carried over: the new
folder is analysed afresh.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import process_batch_results as pbr  # noqa: E402
import run_episode_batches as batches  # noqa: E402


def cellKey(cell: dict) -> str:
    """What makes two cells repeats: the row's label, else all the row's settings."""
    row = cell["status"].get("row") or {}
    label = (row.get("label") or "").strip()
    if label:
        return f"label:{label}"
    if row:
        return "row:" + json.dumps({k: v for k, v in row.items() if v != ""}, sort_keys=True)
    return "name:" + re.sub(r"^cell_\d+_?", "", cell["name"])


def readBatch(batch_dir: Path) -> dict[str, dict]:
    """``{key: cell}`` for a batch folder's cells, in cell order; each cell also gets its ``dir``."""
    out: dict[str, dict] = {}
    for cell in pbr.listCells(batch_dir):
        key = cellKey(cell)
        if key in out:
            raise ValueError(f"{batch_dir}: cells {out[key]['name']} and {cell['name']} have the same "
                             f"{'label' if key.startswith('label:') else 'settings'}; cannot tell which to keep")
        out[key] = {**cell, "dir": batch_dir}
    return out


def _done(cell: dict) -> bool:
    return cell["status"].get("status") == "done"


def combine(batch_dirs: list[Path]) -> tuple[list[dict], list[str]]:
    """The cells of the combined batch, in order, each with the cells it replaced; and the warnings."""
    merged: dict[str, dict] = {}
    notes: list[str] = []
    for batch_dir in batch_dirs:
        for key, cell in readBatch(batch_dir).items():
            old = merged.get(key)
            if old is None:
                merged[key] = {**cell, "replaced": []}
            elif _done(old) and not _done(cell):
                notes.append(f"kept {old['dir'].name}/{old['name']}: {batch_dir.name}/{cell['name']} is "
                             f"{cell['status'].get('status')!r}")
            else:
                merged[key] = {**cell, "replaced": old["replaced"] + [old]}
    return list(merged.values()), notes


def _place(src: Path, dst: Path, symlink: bool) -> None:
    if symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def _videos(batch_dir: Path, name: str) -> list[Path]:
    """``<name>.mp4`` and ``<name>_ep<k>.mp4``, not another cell whose label starts like this one's."""
    pattern = re.compile(re.escape(name) + r"(_ep\d+)?\.mp4")
    return sorted(p for p in batch_dir.glob(f"{name}*.mp4") if pattern.fullmatch(p.name))


def writeCell(cell: dict, index: int, out: Path, symlink: bool = False) -> str:
    """Copy one cell's files into ``out`` as cell ``index``; returns its new name."""
    src_dir, old = cell["dir"], cell["name"]
    row = cell["status"].get("row") or {}
    new = batches.cellName(index, row)

    res = src_dir / f"{old}.json"
    if res.exists():
        doc = json.loads(res.read_text())
        for e in doc.get("episodes", []):
            if e.get("file"):
                renamed = new + e["file"][len(old):] if e["file"].startswith(old + "_") else e["file"]
                _place(src_dir / e["file"], out / renamed, symlink)
                e["file"] = renamed
        (out / f"{new}.json").write_text(json.dumps(doc, indent=2))

    status = dict(cell["status"], cell=index, name=new, combined_from=str(src_dir / old))
    status["argv"] = [a.replace(str(src_dir / old), str(out / new)) for a in status.get("argv", [])]
    (out / f"{new}.status.json").write_text(json.dumps(status, indent=2))

    if (src_dir / f"{old}.log").exists():
        shutil.copy2(src_dir / f"{old}.log", out / f"{new}.log")
    for video in _videos(src_dir, old):
        _place(video, out / (new + video.name[len(old):]), symlink)
    return new


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch_dirs", type=Path, nargs="+",
                   help="batch results folders, lowest precedence first: later folders' cells replace earlier ones'")
    p.add_argument("--out", type=Path, required=True, help="the new batch folder to write (must not hold cells yet)")
    p.add_argument("--symlink", action="store_true",
                   help="link the per-episode .npy files and videos instead of copying them")
    p.add_argument("--dry-run", action="store_true", help="print which cell comes from where; write nothing")
    args = p.parse_args(argv)

    batch_dirs = [d.resolve() for d in args.batch_dirs]
    for d in batch_dirs:
        if not d.is_dir():
            p.error(f"no such folder: {d}")
        if not any(d.glob("cell_*.status.json")):
            p.error(f"{d} holds no cells (cell_*.status.json)")
    if len(set(batch_dirs)) != len(batch_dirs):
        p.error("a folder is given twice")
    out = args.out.resolve()
    if out in batch_dirs:
        p.error("--out must be a new folder, not one of the inputs")
    if out.exists() and any(out.glob("cell_*")):
        p.error(f"{out} already holds cells; pick a new --out")

    try:
        cells, notes = combine(batch_dirs)
    except ValueError as e:
        p.error(str(e))

    print(f"{len(cells)} cells from {len(batch_dirs)} folders -> {out}")
    for i, cell in enumerate(cells):
        replaced = "".join(f"  (replaces {c['dir'].name}/{c['name']})" for c in cell["replaced"])
        print(f"  {i:4d}  {cell['dir'].name}/{cell['name']}  [{cell['status'].get('status')}]{replaced}")
    for note in notes:
        print(f"  note: {note}")
    if args.dry_run:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    provenance, rows = [], []
    for i, cell in enumerate(cells):
        name = writeCell(cell, i, out, args.symlink)
        rows.append(cell["status"].get("row") or {})
        provenance.append({"cell": i, "name": name, "from": str(cell["dir"] / cell["name"]),
                           "replaced": [str(c["dir"] / c["name"]) for c in cell["replaced"]]})

    csv_path = out / "batches.csv"
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        w.writeheader()
        w.writerows(rows)
    (out / "combined.json").write_text(json.dumps(
        {"sources": [str(d) for d in batch_dirs], "notes": notes, "cells": provenance}, indent=2))
    batches.summarize(csv_path, rows, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
