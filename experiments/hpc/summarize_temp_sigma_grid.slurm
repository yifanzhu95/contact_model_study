#!/bin/bash
#SBATCH --job-name=temp_sigma_grid_summary
#SBATCH --output=logs/temp_sigma_grid_summary_%j.out
#SBATCH --error=logs/temp_sigma_grid_summary_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# Builds summary.csv, summary.json and best.csv from every cell in OUTDIR, and
# refreshes each cell's grid_summary.json. Queued by submit_temp_sigma_grid.sh to
# run after the whole array (afterany, so it also runs when some cells failed).
# By hand:
#   CSV=/abs/grid.csv OUTDIR=/abs/results/temp_sigma_grid_x_123 sbatch experiments/hpc/summarize_temp_sigma_grid.slurm
# or directly: python experiments/run_temp_sigma_grid.py grid.csv --summarize --outdir <OUTDIR>

set -euo pipefail
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"
CSV="${CSV:?set CSV}"
OUTDIR="${OUTDIR:?set OUTDIR to the results folder of the grid}"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

python experiments/run_temp_sigma_grid.py "$CSV" --summarize --outdir "$OUTDIR"
