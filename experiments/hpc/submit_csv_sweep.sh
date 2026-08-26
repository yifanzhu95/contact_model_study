#!/bin/bash
# submit_csv_sweep.sh — size and submit run_csv_sweep.slurm for a params CSV.
#
# SLURM's #SBATCH --array bound is static in the job script; it can't read a
# CSV's row count before the file is known. This wrapper does that part:
# counts the CSV's data rows and passes --array on the sbatch command line,
# which overrides run_csv_sweep.slurm's placeholder header.
#
#   ./submit_csv_sweep.sh params.csv
#   ./submit_csv_sweep.sh params.csv my_run_label   # names the combine job's output
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <params.csv> [task_label]" >&2
    exit 1
fi

CSV=$(realpath "$1")
TASK_LABEL="${2:-csv_sweep}"

if [[ ! -f "$CSV" ]]; then
    echo "no such file: $CSV" >&2
    exit 1
fi

# Counted with the csv module, not `wc -l`: the latter undercounts a file with
# no trailing newline (Excel and some editors write these), which would silently
# drop the last experiment from the sweep, and miscounts quoted fields
# containing newlines.
N=$(python3 -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline=""))))' "$CSV")

if (( N < 1 )); then
    echo "$CSV has no data rows (only a header, or is empty)" >&2
    exit 1
fi

echo "submitting $N cells from $CSV"
# --export=ALL,... rather than a bare --export=VAR=val: the bare form propagates
# ONLY the named vars (plus SLURM_*), which strips the environment `module load`
# needs inside the job.
sbatch --array="0-$((N - 1))" \
       --export=ALL,CSV="$CSV",TASK_LABEL="$TASK_LABEL" \
       experiments/hpc/run_csv_sweep.slurm
