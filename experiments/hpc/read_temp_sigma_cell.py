"""read_temp_sigma_cell.py

Look up one row of temp_sigma_grid.slurm's cell table for bash.

The cell table (temp_sigma_cells.csv by default) has one row per SLURM array
task: an (object, model, n_iterations, n_samples, time_horizon, step_time) cell
plus the temperature x noise_sigma grid THAT cell is searched over. Bash cannot
parse a CSV safely (quoting, missing trailing newline), so the SLURM script asks
this helper for its row and reads the answer back with `IFS=$'\\t' read -r`:

    python read_temp_sigma_cell.py cells.csv 7        # one tab-separated line
    python read_temp_sigma_cell.py cells.csv --count  # number of data rows

The header is validated (exactly COLUMNS, so a typo'd column cannot silently
run a default), the row is bounds-checked (this is the authoritative range
check; the SLURM script does not re-count), the two grid columns are parsed the
way the driver parses them, and duplicate cells are rejected: the six CELL_KEY
columns are what run_temp_sigma_grid.py's label and
analysis/temp_sigma_grid_to_csv.py's cell_key identify a cell by, so two rows
naming the same one would merge into a single result row.

Every failure exits non-zero with a message, which `set -eo pipefail` in the
SLURM script turns into a failed task rather than a run of the wrong cell.
submit_temp_sigma_grid.sh counts rows with this same helper, so a bad CSV is
rejected at submission rather than on a node 16 hours in.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Column order is the output order the SLURM script reads them in.
COLUMNS = ("object", "model", "n_iterations", "n_samples",
           "time_horizon", "step_time", "temperatures", "noise_sigmas")
# What makes a cell: the grid columns are NOT part of it, so two rows that name
# the same cell with different temperature lists are still a duplicate.
CELL_KEY = ("object", "model", "n_iterations", "n_samples",
            "time_horizon", "step_time")
# Comma- or space-separated lists of strictly positive values; expanded into the
# grid by the driver.
LIST_COLUMNS = ("temperatures", "noise_sigmas")


def check_values(raw: str, column: str) -> None:
    """Validate one grid list the way run_temp_sigma_grid.parse_values does.

    Deliberately a copy of that function's rules rather than an import: this
    helper runs under the submit script's system python3, before any
    `conda activate`, and importing the driver would pull in warp and mujoco.
    Both knobs are strictly positive (temperature divides the cost in MPPI's
    softmax; noise_sigma is a standard deviation), and a repeated value would
    run the same point twice.
    """
    tokens = [tok for tok in raw.replace(",", " ").split() if tok]
    if not tokens:
        raise ValueError(f"{column} is empty; give at least one value")
    values = []
    for tok in tokens:
        try:
            v = float(tok)
        except ValueError:
            raise ValueError(f"bad {column} value {tok!r}; expected a number") from None
        if v <= 0.0:
            raise ValueError(f"{column} values must be > 0, got {v:g}")
        values.append(v)
    if len(set(values)) != len(values):
        raise ValueError(f"{column} lists a value twice: {values}")


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
                     f"(every cell needs all {len(COLUMNS)} values)")
        for c in LIST_COLUMNS:
            try:
                check_values(r[c], c)
            except ValueError as exc:
                sys.exit(f"{csv_path}: row {i}: {exc}")
        key = tuple(r[c] for c in CELL_KEY)
        if key in seen:
            sys.exit(f"{csv_path}: rows {seen[key]} and {i} are the same cell "
                     f"{dict(zip(CELL_KEY, key))}; these six columns are what the "
                     f"driver's label and the analysis cell_key identify a cell "
                     f"by, so the two rows would merge into one result row "
                     f"(differing temperatures/noise_sigmas do not separate them "
                     f"— put every value of a cell's grid in one row)")
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
