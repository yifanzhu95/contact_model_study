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
# One array task = one cell of an episode-batch results folder = one GPU. At
# every analysed step of every recorded episode it plans with the used planner
# and the optimal (eval-model) planner, steps the action taken and the optimal
# action on both simulators, saves all of it per episode
# (analysis/<cell>/<episode id>.npz), and then computes the KLs and the
# forward-simulation errors from those files. The optimal planner runs at the
# eval timestep, so a step costs a few times a normal plan: about 1-2 s per
# analysed step at the defaults. A cell with many long episodes can trade
# resolution for time with STRIDE below.
# A cell already analysed is skipped, so resubmitting resumes; to rerun with
# another STRIDE or OPTIMAL_ARGS, add ANALYSIS_ARGS="--overwrite".
#
# Submit with experiments/hpc/submit_process_batch_results.sh. By hand:
#   sbatch --array=0-4 --export=ALL,BATCH_DIR=/abs/results/episode_batches_x_123 \
#       experiments/hpc/process_batch_results.slurm

# -- analysis settings ---------------------------------------------------------
# The optimal planner's own settings, the same for every cell: run_episodes.py
# planner flags applied on top of each cell's recorded ones. Set to the
# driver's defaults, so every cell is compared with the same optimal planner
# whatever its own planner ran with. It may also set --horizon (instead of
# --time-horizon), --delta, --cost-weight NAME=VALUE, --(no-)graph, --nconmax
# and --njmax; empty, each cell's own planner is used on the eval model.
OPTIMAL_ARGS="--n-samples 1024 --n-iterations 5 --noise-sigma 0.1 --temperature 1.0 --no-adaptive-temp --warm-start shift"
# Analyse every STRIDE-th saved step of each episode (plans and forward
# simulation). Warm-started results that did not save each plan's starting
# mean must be replayed through every step: they fail with STRIDE > 1.
STRIDE=2

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

echo "stride ${STRIDE}; optimal planner args: ${OPTIMAL_ARGS:-none, as each cell ran}"
ARGS=(--stride "$STRIDE")
# With '=': the value itself starts with --.
if [[ -n "$OPTIMAL_ARGS" ]]; then
    ARGS+=("--optimal-args=$OPTIMAL_ARGS")
fi
# ANALYSIS_ARGS is deliberately unquoted: it holds extra flags.
python experiments/process_batch_results.py "$BATCH_DIR" --cell "$CELL" "${ARGS[@]}" ${ANALYSIS_ARGS:-}
