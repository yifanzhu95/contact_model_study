#!/bin/bash
# submit_temp_sigma_grid.sh — size and submit temp_sigma_grid.slurm for a cell CSV.
#
# SLURM's #SBATCH --array bound is static in the job script; it can't read a
# CSV's row count before the file is known. This wrapper does that part:
# counts the CSV's data rows (validating it at the same time, so a bad CSV is
# rejected here rather than on every node) and passes --array on the sbatch
# command line, which overrides temp_sigma_grid.slurm's placeholder header.
#
#   ./submit_temp_sigma_grid.sh                                # default CSV
#   ./submit_temp_sigma_grid.sh cells.csv
#   ./submit_temp_sigma_grid.sh cells.csv my_label 8           # label the combine
#                                                              # job; 8 cells on
#                                                              # GPUs at once
#   OUTDIR=/abs/path/to/previous ./submit_temp_sigma_grid.sh cells.csv   # resume
#                                                              # (same CSV: cell
#                                                              # files are named
#                                                              # by row index)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV=$(realpath "${1:-$HERE/temp_sigma_cells.csv}")
TASK_LABEL="${2:-temp_sigma_grid}"
THROTTLE="${3:-12}"     # %N suffix: how many cells hold a GPU at once

if [[ ! -f "$CSV" ]]; then
    echo "no such file: $CSV" >&2
    exit 1
fi

# Counted (and validated) by the same helper the job script reads rows with,
# not `wc -l`: the latter undercounts a file with no trailing newline (Excel and
# some editors write these), which would silently drop the last cell.
N=$(python3 "$HERE/read_temp_sigma_cell.py" "$CSV" --count)

if (( N < 1 )); then
    echo "$CSV has no data rows (only a header, or is empty)" >&2
    exit 1
fi

echo "submitting $N cells from $CSV (%$THROTTLE at a time)"
# --export=ALL,... rather than a bare --export=VAR=val: the bare form propagates
# ONLY the named vars (plus SLURM_*), which strips the environment `module load`
# needs inside the job. OUTDIR, if set, rides along for a resume.
sbatch --array="0-$((N - 1))%${THROTTLE}" \
       --export=ALL,CELLS_CSV="$CSV",TASK_LABEL="$TASK_LABEL" \
       "$HERE/temp_sigma_grid.slurm"
