"""read_temp_sigma_cell.py

Look up one row of temp_sigma_grid.slurm's cell table for bash.

The cell table (temp_sigma_cells.csv by default) has one row per SLURM array
task: an (object, model, n_iterations, n_samples) cell plus the planning
horizon and control-step duration THAT cell runs with. Bash cannot parse a CSV
safely (quoting, missing trailing newline), so the SLURM script asks this
helper for its row and reads the answer back with `IFS=$'\\t' read -r`:

    python read_temp_sigma_cell.py cells.csv 7        # one tab-separated line
    python read_temp_sigma_cell.py cells.csv --count  # number of data rows

The header is validated (exactly COLUMNS, so a typo'd column cannot silently
run a default), the row is bounds-checked (this is the authoritative range
check; the SLURM script does not re-count), and duplicate cells are rejected:
run_temp_sigma_grid.py's label and analysis/temp_sigma_grid_to_csv.py's
cell_key do not include time_horizon/step_time, so two rows with the same
(object, model, n_iterations, n_samples) would merge into one result row.
Every failure exits non-zero with a message, which `set -eo pipefail` in the
SLURM script turns into a failed task rather than a run of the wrong cell.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Column order is the output order the SLURM script reads them in.
COLUMNS = ("object", "model", "n_iterations", "n_samples",
           "time_horizon", "step_time")
CELL_KEY = ("object", "model", "n_iterations", "n_samples")


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        missing = [c for c in COLUMNS if c not in fieldnames]
        extra   = [c for c in fieldnames if c and c not in COLUMNS]
        if missing or extra:
            sys.exit(f"{csv_path}: header must be exactly {','.join(COLUMNS)}"
                     + (f"\n  missing: {', '.join(missing)}" if missing else "")
                     + (f"\n  unknown: {', '.join(extra)}" if extra else ""))
        rows = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in reader]

    seen: dict[tuple, int] = {}
    for i, r in enumerate(rows):
        blank = [c for c in COLUMNS if r[c] == ""]
        if blank:
            sys.exit(f"{csv_path}: row {i} has blank {', '.join(blank)} "
                     f"(every cell needs all six values)")
        key = tuple(r[c] for c in CELL_KEY)
        if key in seen:
            sys.exit(f"{csv_path}: rows {seen[key]} and {i} are the same cell "
                     f"{dict(zip(CELL_KEY, key))}; the driver's label and the "
                     f"analysis cell_key ignore time_horizon/step_time, so they "
                     f"would merge into one result row")
        seen[key] = i
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("row", type=int, nargs="?", default=None,
                   help="0-indexed data row (excluding the header)")
    p.add_argument("--count", action="store_true",
                   help="print the number of data rows instead of a row")
    args = p.parse_args()

    if not args.csv.is_file():
        sys.exit(f"no such file: {args.csv}")
    rows = load_rows(args.csv)

    if args.count:
        print(len(rows))
        return
    if args.row is None:
        p.error("give a row index or --count")
    if not (0 <= args.row < len(rows)):
        sys.exit(f"row {args.row} is outside {args.csv} ({len(rows)} data rows, "
                 f"valid range 0-{len(rows) - 1}); "
                 f"set #SBATCH --array=0-{len(rows) - 1}")
    print("\t".join(rows[args.row][c] for c in COLUMNS))


if __name__ == "__main__":
    main()
