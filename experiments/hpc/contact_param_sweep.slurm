#!/bin/bash
#SBATCH --job-name=contact_param_sweep
#SBATCH --output=logs/contact_param_sweep_%A_%a.out
#SBATCH --error=logs/contact_param_sweep_%A_%a.err
#SBATCH --array=0-0            # placeholder: submit_contact_param_sweep.sh sets it to the cells
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00          # an M3 cell of 30 x 20 steps: ~3.5 h expected; M6: ~19 h; see below
# One array task = one cell of an episode-batch results folder = one GPU. At
# every analysed step of the episodes chosen, it plans once with the zero-gap
# planner (the cell's planner on the eval model), plans again with the cell's
# planner at every point of a grid over its rollout model's contact
# parameters, and scores every plan's free energy on the eval model; see
# contact_param_sweep.py. The grid sets the cost: 25 points for M1-M3, 30 for
# M4 and 125 for M6, each a plan per step plus a share of an eval-model
# rollout (SCORE_BLOCK - 1 points per rollout). On an RTX 4090 with M5 as the
# eval model (Nov 2026, 256 samples, 64 ms control, 352 ms horizon), a step
# took ~1.2 s for the zero-gap plan, 0.34 s (M3) / 0.53 s (M6) per grid
# point's plan and ~1.2 s per score block: ~20 s a step for M3 (9 blocks) and
# ~115 s for M6 (42 blocks). So 30 episodes x 20 steps is ~3.5 h for an M3
# cell and ~19 h for an M6 one; fewer EPISODES or N_STEPS_ANALYSED, or a
# larger SCORE_BLOCK, bring M6 down. Time a cell on the cluster's GPU first
# (--episodes 0 --n-steps-analysed 2 prints per-unit times) and set --time
# with SBATCH_ARGS when submitting. The cell's checkpoint is saved after each
# grid point and each score block, so a task cut off by the time limit resumes
# where it stopped when resubmitted; a cell already done is skipped (add
# SWEEP_ARGS="--overwrite" to redo it).
#
# Submit with experiments/hpc/submit_contact_param_sweep.sh. By hand:
#   sbatch --array=0-4 --export=ALL,BATCH_DIR=/abs/results/episode_batches_x_123 \
#       experiments/hpc/contact_param_sweep.slurm

# -- sweep settings ----------------------------------------------------------------
# The episodes of each cell to analyse: indices, inclusive ranges and episode
# ids, comma-separated, e.g. "0-4,9". Empty: every episode. Set EPISODES when
# submitting to replace it.
EPISODES="${EPISODES-}"
# Analyse this many saved steps of each episode, spaced evenly from its first
# to its last (every step of a shorter episode).
N_STEPS_ANALYSED=20
# Means scored per eval-model rollout, U_A included, in SCORE_BLOCK times the
# planner's worlds: larger is fewer, bigger rollouts, if GPU memory allows.
SCORE_BLOCK=4
# With a Kamino (M5) eval model: the zero-gap planner's contact buffer per
# world and PADMM iteration cap, as in process_batch_results.slurm.
KAMINO_MAX_CONTACTS=96
KAMINO_PADMM_ITERATIONS=100

set -euo pipefail

# The repo root. The cluster copies this script before running it, so the path
# cannot be derived from the script's own location.
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"

BATCH_DIR="${BATCH_DIR:?set BATCH_DIR=/abs/path/to/the/batch/results (submit_contact_param_sweep.sh does this)}"
CELL="${SLURM_ARRAY_TASK_ID:?run this as an array job}"

module load miniconda
eval "$(conda shell.bash hook)"
# The submit script exports CONDA_ENV; contact_comfree2 (M1-M6) otherwise.
conda activate "${CONDA_ENV:-contact_comfree2}"
export MUJOCO_GL=egl

echo "cell ${CELL} of ${BATCH_DIR} on $(hostname); episodes: ${EPISODES:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ARGS=(--n-steps-analysed "$N_STEPS_ANALYSED" --score-block "$SCORE_BLOCK"
      --kamino-max-contacts "$KAMINO_MAX_CONTACTS" --kamino-padmm-iterations "$KAMINO_PADMM_ITERATIONS")
if [[ -n "$EPISODES" ]]; then
    ARGS+=(--episodes "$EPISODES")
fi
# SWEEP_ARGS is deliberately unquoted: it holds extra flags.
python experiments/contact_param_sweep.py "$BATCH_DIR" --cell "$CELL" "${ARGS[@]}" ${SWEEP_ARGS:-}
