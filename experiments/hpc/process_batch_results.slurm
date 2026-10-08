#!/bin/bash
#SBATCH --job-name=process_batch_results
#SBATCH --output=logs/process_batch_results_%A_%a.out
#SBATCH --error=logs/process_batch_results_%A_%a.err
#SBATCH --array=0-0            # placeholder: submit_process_batch_results.sh sets it to the batch's cells
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# One array task = one cell of an episode-batch results folder = one GPU. It
# replays every recorded episode with the used planner and the optimal
# (eval-model) planner for the KL, and steps both simulators from every
# recorded state for the forward-simulation error. The optimal planner runs at
# the eval timestep, so a step costs a few times a normal plan: about 1-2 s per
# recorded step at the defaults. A cell with many long episodes can trade
# resolution for time with --stride (the KL at every k-th step), e.g.
#   ANALYSIS_ARGS="--stride 4" experiments/hpc/submit_process_batch_results.sh <batch dir>
# A cell already analysed is skipped, so resubmitting resumes.
#
# Submit with experiments/hpc/submit_process_batch_results.sh. By hand:
#   sbatch --array=0-4 --export=ALL,BATCH_DIR=/abs/results/episode_batches_x_123 \
#       experiments/hpc/process_batch_results.slurm

set -euo pipefail

# The repo root. The cluster copies this script before running it, so the path
# cannot be derived from the script's own location.
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"

BATCH_DIR="${BATCH_DIR:?set BATCH_DIR=/abs/path/to/the/batch/results (submit_process_batch_results.sh does this)}"
CELL="${SLURM_ARRAY_TASK_ID:?run this as an array job}"

module load miniconda
eval "$(conda shell.bash hook)"
# The submit script exports CONDA_ENV; contact_kamino (M1-M5) otherwise.
conda activate "${CONDA_ENV:-contact_kamino}"
export MUJOCO_GL=egl

echo "cell ${CELL} of ${BATCH_DIR} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ANALYSIS_ARGS is deliberately unquoted: it holds extra flags.
python experiments/process_batch_results.py "$BATCH_DIR" --cell "$CELL" ${ANALYSIS_ARGS:-}
