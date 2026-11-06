#!/bin/bash
#SBATCH --job-name=contact_param_sweep_summary
#SBATCH --output=logs/contact_param_sweep_summary_%j.out
#SBATCH --error=logs/contact_param_sweep_summary_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# Builds sweep_results.csv / .json in BATCH_DIR: one row per grid point of
# every swept cell, one per cell not done (with its status). Queued by
# submit_contact_param_sweep.sh to run after the whole array (afterany, so it
# also runs when some cells failed or ran out of time). By hand:
#   BATCH_DIR=/abs/results/episode_batches_x_123 sbatch experiments/hpc/summarize_contact_param_sweep.slurm
# or directly: python experiments/contact_param_sweep.py <BATCH_DIR> --summarize

set -euo pipefail
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"
BATCH_DIR="${BATCH_DIR:?set BATCH_DIR to the batch results folder}"

module load miniconda
eval "$(conda shell.bash hook)"
# The submit script exports CONDA_ENV; contact_comfree2 (M1-M6) otherwise.
conda activate "${CONDA_ENV:-contact_comfree2}"

python experiments/contact_param_sweep.py "$BATCH_DIR" --summarize
