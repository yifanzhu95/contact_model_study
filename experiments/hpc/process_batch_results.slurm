#!/bin/bash
#SBATCH --job-name=process_batch_results
#SBATCH --output=logs/process_batch_results_%A_%a.out
#SBATCH --error=logs/process_batch_results_%A_%a.err
#SBATCH --array=0-0            # placeholder: submit_process_batch_results.sh sets it to cells x shards
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00          # one shard of a 30-episode cell: ~1.1 h expected; see below
# One array task = one shard of one cell of an episode-batch results folder =
# one GPU. With N_SHARDS shards per cell, task t analyses the episodes
# t % N_SHARDS, t % N_SHARDS + N_SHARDS, ... of cell t / N_SHARDS, and the
# task that saves a cell's last episode computes its metrics. At
# every analysed step of every recorded episode it plans with the used planner
# and the optimal (eval-model) planner, steps the action taken and the optimal
# action on both simulators, saves all of it per episode
# (analysis/<cell>/<episode id>.npz), and then computes the KLs and the
# forward-simulation errors from those files. The optimal planner runs at the
# eval timestep, so a step costs many normal plans: about 22 s per analysed step
# on an RTX 5000 Ada with M5 as the eval model (Nov 2026 batch), 84% of it the
# optimal planner, and an estimated ~13 s with the Kamino settings below. A
# cell with many long episodes can trade resolution for time with STRIDE or
# N_STEPS_ANALYSED below.
# The time limit fits the defaults: 30 episodes per cell over N_SHARDS=5 is 6
# episodes x 50 steps x ~13 s = ~65 min per task, and 4 h leaves room for
# startup and an estimate scaled from a desktop GPU. Fewer shards, more
# episodes, STRIDE instead of N_STEPS_ANALYSED, or Kamino's own settings take
# longer: raise it with SBATCH_ARGS="--time=HH:MM:SS" when submitting.
# Each episode is saved as soon as it is analysed, and a cell already analysed
# is skipped, so resubmitting resumes episode by episode, even after a task hit
# the time limit. To rerun a done cell with another STRIDE, N_STEPS_ANALYSED,
# OPTIMAL_ARGS or Kamino setting, add ANALYSIS_ARGS="--overwrite".
#
# Submit with experiments/hpc/submit_process_batch_results.sh. By hand:
#   sbatch --array=0-4 --export=ALL,BATCH_DIR=/abs/results/episode_batches_x_123,N_SHARDS=1 \
#       experiments/hpc/process_batch_results.slurm

# -- analysis settings ---------------------------------------------------------
# The optimal planner's own settings, the same for every cell: run_episodes.py
# planner flags applied on top of each cell's recorded ones, so every cell is
# compared with the same optimal planner whatever its own planner ran with:
# iterate to a convergence ratio of 2, capped at 20 iterations, with wide
# noise and a near-greedy temperature. It may also set --horizon (instead of
# --time-horizon), --delta, --cost-weight NAME=VALUE, --(no-)graph, --nconmax
# and --njmax; empty, each cell's own planner is used on the eval model. Set
# OPTIMAL_ARGS when submitting to replace this (make_final_batch_csv.py prints
# the line for a tuned optimal planner).
OPTIMAL_ARGS="${OPTIMAL_ARGS-"--n-samples 1024 --convergence-ratio 2 --max-iterations 20 --noise-sigma 0.38 --temperature 0.001 --no-adaptive-temp"}"
# Analyse every STRIDE-th saved step of each episode (plans and forward
# simulation). Warm-started results that did not save each plan's starting
# mean must be replayed through every step: they fail with STRIDE > 1.
STRIDE=4
# Or analyse this many saved steps of each episode, spaced evenly from its
# first to its last (every step of a shorter episode), so every episode gets
# the same number however long it ran. Set, it overrides STRIDE; empty, STRIDE
# is used. 50, not 100: episodes differ far more than steps within one do, so
# halving the steps widened the cells' 95% intervals by only 3.5-9% on the
# Nov 2026 batch, for half the cost.
N_STEPS_ANALYSED=50
# With a Kamino (M5) eval model: the optimal and zero-gap planners' contact
# buffer per world and PADMM iteration cap (Kamino's own: 128 and 200). The
# cap at 100 cuts ~37% of a step; 96 contacts, not 64, because 64 overflowed on a
# cube cell. See DEFAULT_KAMINO in process_batch_results.py for what each costs.
# Kamino warns "contact capacity exceeded" when a world
# overflows the buffer: grep the logs for it.
KAMINO_MAX_CONTACTS=96
KAMINO_PADMM_ITERATIONS=100

set -euo pipefail

# The repo root. The cluster copies this script before running it, so the path
# cannot be derived from the script's own location.
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"

BATCH_DIR="${BATCH_DIR:?set BATCH_DIR=/abs/path/to/the/batch/results (submit_process_batch_results.sh does this)}"
TASK="${SLURM_ARRAY_TASK_ID:?run this as an array job}"
# Shards per cell; the submit script exports it.
N_SHARDS="${N_SHARDS:-1}"
CELL=$((TASK / N_SHARDS))
SHARD=$((TASK % N_SHARDS))

module load miniconda
eval "$(conda shell.bash hook)"
# The submit script exports CONDA_ENV; contact_kamino (M1-M5) otherwise.
conda activate "${CONDA_ENV:-contact_kamino}"
export MUJOCO_GL=egl

echo "cell ${CELL}, shard ${SHARD}/${N_SHARDS}, of ${BATCH_DIR} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ -n "$N_STEPS_ANALYSED" ]]; then
    echo "${N_STEPS_ANALYSED} steps analysed per episode; optimal planner args: ${OPTIMAL_ARGS:-none, as each cell ran}"
    ARGS=(--n-steps-analysed "$N_STEPS_ANALYSED")
else
    echo "stride ${STRIDE}; optimal planner args: ${OPTIMAL_ARGS:-none, as each cell ran}"
    ARGS=(--stride "$STRIDE")
fi
# With '=': the value itself starts with --.
if [[ -n "$OPTIMAL_ARGS" ]]; then
    ARGS+=("--optimal-args=$OPTIMAL_ARGS")
fi
echo "Kamino eval model: ${KAMINO_MAX_CONTACTS} contacts per world, PADMM cap ${KAMINO_PADMM_ITERATIONS}"
ARGS+=(--kamino-max-contacts "$KAMINO_MAX_CONTACTS" --kamino-padmm-iterations "$KAMINO_PADMM_ITERATIONS")
# ANALYSIS_ARGS is deliberately unquoted: it holds extra flags.
python experiments/process_batch_results.py "$BATCH_DIR" --cell "$CELL" --shard "$SHARD" --n-shards "$N_SHARDS" \
    "${ARGS[@]}" ${ANALYSIS_ARGS:-}
