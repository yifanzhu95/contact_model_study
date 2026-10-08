#!/bin/bash
#SBATCH --job-name=temp_sigma_grid
#SBATCH --output=logs/temp_sigma_grid_%A_%a.out
#SBATCH --error=logs/temp_sigma_grid_%A_%a.err
#SBATCH --array=0-0            # placeholder: submit_temp_sigma_grid.sh sets it to the CSV's rows
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
# One array task = one CSV row (cell) = one GPU. The cell walks its whole
# temperature x noise_sigma grid, running each point's episodes interwoven (one
# planner process and two eval-sim workers, hence 4 CPUs). A cell's cost is
# (#temperatures x #noise_sigmas x n_episodes) episodes, so a wide row is a long
# row: the --time above must cover the most expensive one. Override per
# submission with SBATCH_ARGS="--time=24:00:00" (see submit_temp_sigma_grid.sh).
# A cell that hits the limit resumes point by point when resubmitted with OUTDIR.
#
# Submit with experiments/hpc/submit_temp_sigma_grid.sh. By hand:
#   sbatch --array=0-4 --export=ALL,CSV=/abs/grid.csv experiments/hpc/run_temp_sigma_grid.slurm

set -euo pipefail

# The repo root. The cluster copies this script before running it, so the path
# cannot be derived from the script's own location.
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"

CSV="${CSV:?set CSV=/abs/path/to/grid.csv (submit_temp_sigma_grid.sh does this)}"
CELL="${SLURM_ARRAY_TASK_ID:?run this as an array job}"
# Every task of the array writes into one folder, so the summary finds every cell.
# Point OUTDIR at an earlier run's folder to resume it: finished points are skipped.
OUTDIR="${OUTDIR:-$PROJ_DIR/results/temp_sigma_grid_$(basename "${CSV%.*}")_${SLURM_ARRAY_JOB_ID}}"

module load miniconda
eval "$(conda shell.bash hook)"
# The submit script exports CONDA_ENV; contact_kamino (M1-M5) otherwise.
conda activate "${CONDA_ENV:-contact_kamino}"
export MUJOCO_GL=egl

echo "cell ${CELL} of ${CSV} -> ${OUTDIR} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# The out-of-range check is in the Python script, which parses the CSV properly.
python experiments/run_temp_sigma_grid.py "$CSV" --cell "$CELL" --outdir "$OUTDIR"
