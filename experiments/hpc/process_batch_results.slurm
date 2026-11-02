#!/bin/bash
#SBATCH --job-name=process_batch_results
#SBATCH --output=logs/process_batch_results_%A_%a.out
#SBATCH --error=logs/process_batch_results_%A_%a.err
#SBATCH --array=0-0            # placeholder: submit_process_batch_results.sh sets it to cells x shards / SHARDS_PER_GPU
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4      # per shard: the submit script multiplies these two by SHARDS_PER_GPU
#SBATCH --mem=16G
#SBATCH --time=30:00:00
# One array task = one GPU running SHARDS_PER_GPU shards of an episode-batch
# results folder at once, each its own process. A plan of ~1000 worlds is
# bound by kernel launches, not GPU throughput, so two processes share a GPU
# for little slowdown each (2 planners per GPU ran the BO ~1.5x faster).
# Shard g (counting every cell's shards in turn, g = task * SHARDS_PER_GPU + j)
# is shard g % N_SHARDS of cell g / N_SHARDS: it analyses that cell's episodes
# shard, shard + N_SHARDS, ..., and the shard that saves a cell's last episode
# computes its metrics. With SHARDS_PER_GPU > 1 each shard logs to
# logs/process_batch_results_<job>_<task>_<j>.out. At
# every analysed step of every recorded episode it plans with the used planner
# and the optimal (eval-model) planner, steps the action taken and the optimal
# action on both simulators, saves all of it per episode
# (analysis/<cell>/<episode id>.npz), and then computes the KLs and the
# forward-simulation errors from those files. The optimal planner runs at the
# eval timestep, so a step costs a few times a normal plan: about 1-2 s per
# analysed step at the defaults. A cell with many long episodes can trade
# resolution for time with STRIDE or N_STEPS_ANALYSED below.
# Each episode is saved as soon as it is analysed, and a cell already analysed
# is skipped, so resubmitting resumes episode by episode, even after a task hit
# the time limit. To rerun a done cell with another STRIDE, N_STEPS_ANALYSED or
# OPTIMAL_ARGS, add ANALYSIS_ARGS="--overwrite".
#
# Submit with experiments/hpc/submit_process_batch_results.sh. By hand:
#   sbatch --array=0-4 --export=ALL,BATCH_DIR=/abs/results/episode_batches_x_123,N_SHARDS=1 \
#       experiments/hpc/process_batch_results.slurm
# (one shard per GPU: SHARDS_PER_GPU defaults to 1 here; set it, with
# --cpus-per-task and --mem to match, to pack more)

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
# is used.
N_STEPS_ANALYSED=100

set -euo pipefail

# The repo root. The cluster copies this script before running it, so the path
# cannot be derived from the script's own location.
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"

BATCH_DIR="${BATCH_DIR:?set BATCH_DIR=/abs/path/to/the/batch/results (submit_process_batch_results.sh does this)}"
TASK="${SLURM_ARRAY_TASK_ID:?run this as an array job}"
# Shards per cell and shards run at once on this GPU; the submit script exports both.
N_SHARDS="${N_SHARDS:-1}"
SHARDS_PER_GPU="${SHARDS_PER_GPU:-1}"

module load miniconda
eval "$(conda shell.bash hook)"
# The submit script exports CONDA_ENV; contact_kamino (M1-M5) otherwise.
conda activate "${CONDA_ENV:-contact_kamino}"
export MUJOCO_GL=egl
# The shards share the task's CPUs.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-4} / SHARDS_PER_GPU > 0 ? ${SLURM_CPUS_PER_TASK:-4} / SHARDS_PER_GPU : 1 ))
# The submit script exports the cell count; submitted by hand, it is counted here.
N_CELLS="${N_CELLS:-$(python experiments/process_batch_results.py "$BATCH_DIR" --count)}"
TOTAL=$((N_CELLS * N_SHARDS))

echo "task ${TASK}: ${SHARDS_PER_GPU} shard(s) of ${BATCH_DIR} (${N_CELLS} cells x ${N_SHARDS} shards) on $(hostname)"
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

# One shard: run_shard <cell> <shard>. ANALYSIS_ARGS is deliberately unquoted: it holds extra flags.
run_shard() {
    python experiments/process_batch_results.py "$BATCH_DIR" --cell "$1" --shard "$2" --n-shards "$N_SHARDS" \
        "${ARGS[@]}" ${ANALYSIS_ARGS:-}
}

PIDS=()
LABELS=()
for ((j = 0; j < SHARDS_PER_GPU; j++)); do
    G=$((TASK * SHARDS_PER_GPU + j))
    (( G < TOTAL )) || break
    CELL=$((G / N_SHARDS))
    SHARD=$((G % N_SHARDS))
    LABELS+=("cell ${CELL}, shard ${SHARD}/${N_SHARDS}")
    if (( SHARDS_PER_GPU > 1 )); then
        LOG="logs/process_batch_results_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${TASK}_${j}.out"
        echo "${LABELS[-1]} -> ${LOG}"
        run_shard "$CELL" "$SHARD" >"$LOG" 2>&1 &
    else
        echo "${LABELS[-1]}"
        run_shard "$CELL" "$SHARD" &
    fi
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "${LABELS[$i]}: done"
    else
        echo "${LABELS[$i]}: FAILED (exit $?)" >&2
        FAILED=1
    fi
done
exit "$FAILED"
