#!/bin/bash
#SBATCH --job-name=process_batch_results_summary
#SBATCH --output=logs/process_batch_results_summary_%j.out
#SBATCH --error=logs/process_batch_results_summary_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# Builds final_results.csv / .json and final_episodes.csv in BATCH_DIR: every
# cell's settings, the batch's own results and the analysis metrics. Queued by
# submit_process_batch_results.sh to run after the whole array (afterany, so it
# also runs when some cells failed; those show their error). By hand:
#   BATCH_DIR=/abs/results/episode_batches_x_123 sbatch experiments/hpc/summarize_batch_results.slurm
# or directly: python experiments/process_batch_results.py <BATCH_DIR> --summarize

set -euo pipefail
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"
BATCH_DIR="${BATCH_DIR:?set BATCH_DIR to the batch results folder}"

module load miniconda
eval "$(conda shell.bash hook)"
# The submit script exports CONDA_ENV; contact_kamino (M1-M5) otherwise.
conda activate "${CONDA_ENV:-contact_kamino}"

python experiments/process_batch_results.py "$BATCH_DIR" --summarize
