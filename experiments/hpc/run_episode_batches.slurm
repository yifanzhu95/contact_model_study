#!/bin/bash
#SBATCH --job-name=episode_batches
#SBATCH --output=logs/episode_batches_%A_%a.out
#SBATCH --error=logs/episode_batches_%A_%a.err
#SBATCH --array=0-0            # placeholder: submit_episode_batches.sh sets it to the CSV's rows
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
# One array task = one CSV row (cell) = one GPU. The resources above must cover
# the most expensive row: sbatch options on the command line override them,
# e.g. `sbatch --time=24:00:00 ...` for rows with long episodes or Pinocchio/Drake
# eval sims (see test_scripts/profile_run_episodes.py for per-step costs).
#
# Submit with experiments/hpc/submit_episode_batches.sh, which validates the CSV,
# sizes --array and queues the summary job. By hand:
#   sbatch --array=0-5 --export=ALL,CSV=/abs/batches.csv experiments/hpc/run_episode_batches.slurm

set -euo pipefail

# The repo root. The cluster copies this script before running it, so the path
# cannot be derived from the script's own location.
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"

CSV="${CSV:?set CSV=/abs/path/to/batches.csv (submit_episode_batches.sh does this)}"
CELL="${SLURM_ARRAY_TASK_ID:?run this as an array job}"
# Every task of the array writes into one folder, so the summary finds every cell.
# Point OUTDIR at an earlier run's folder to resume it: finished cells are skipped.
OUTDIR="${OUTDIR:-$PROJ_DIR/results/episode_batches_$(basename "${CSV%.*}")_${SLURM_ARRAY_JOB_ID}}"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling
export MUJOCO_GL=egl

echo "cell ${CELL} of ${CSV} -> ${OUTDIR} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# The out-of-range check is in the Python script, which parses the CSV properly.
python experiments/run_episode_batches.py "$CSV" --cell "$CELL" --outdir "$OUTDIR"
