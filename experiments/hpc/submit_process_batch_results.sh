#!/bin/bash
# submit_process_batch_results.sh: analyse every cell of an episode-batch results folder as a job array.
#
#   experiments/hpc/submit_process_batch_results.sh results/episode_batches_my_123456
#   experiments/hpc/submit_process_batch_results.sh results/episode_batches_my_123456 8   # at most 8 cells at once
#   ANALYSIS_ARGS="--overwrite --max-episodes 5" experiments/hpc/submit_process_batch_results.sh <batch dir>
#   N_SHARDS=10 experiments/hpc/submit_process_batch_results.sh <batch dir>   # 10 shards per cell
#   SHARDS_PER_GPU=1 experiments/hpc/submit_process_batch_results.sh <batch dir>   # one shard per GPU, as before
#   OPTIMAL_ARGS="--n-samples 1024 ... --temperature 0.02" experiments/hpc/submit_process_batch_results.sh <batch dir>
#
# The stride (or steps per episode) and the optimal planner's settings are set in process_batch_results.slurm's
# header (OPTIMAL_ARGS set here replaces the latter); other flags go in ANALYSIS_ARGS. Each cell's episodes are
# split over N_SHARDS shards (default 5), and each array task runs SHARDS_PER_GPU of them at once on its one GPU
# (default 2), so the array has cells x N_SHARDS / SHARDS_PER_GPU tasks, each with SHARDS_PER_GPU x 4 CPUs and
# x 16G memory; the max concurrent count caps those tasks (GPUs).
#
# Run from the repo root. It:
#   1. counts the folder's cells (its cell_*.status.json files),
#   2. submits process_batch_results.slurm with --array sized to them times N_SHARDS over SHARDS_PER_GPU,
#   3. queues summarize_batch_results.slurm to build final_results.csv once the array has finished.
# Cells already analysed, and episodes already saved, are skipped, so resubmitting resumes. Extra sbatch
# options can go in SBATCH_ARGS, e.g. SBATCH_ARGS="--time=24:00:00".
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <batch results dir> [max concurrent cells]" >&2
    exit 1
fi
[[ -d "$1" ]] || { echo "no such folder: $1" >&2; exit 1; }
BATCH_DIR=$(realpath "$1")
THROTTLE="${2:-}"
PROJ_DIR="${PROJ_DIR:-$(pwd)}"
cd "$PROJ_DIR"
mkdir -p logs

# The count below runs Python on the login node, so it needs the same conda
# env the jobs use. `module` is a shell function set up by login shells; a
# script run with `bash` may not have it, so load the module system first when
# it is missing. Both setups reference unset variables, so strict mode is
# paused around them. CONDA_ENV picks another env (default contact_kamino, which
# runs M1-M5); it is exported, so the jobs activate the same one.
set +u
if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        [[ -f "$init" ]] && { source "$init"; break; }
    done
fi
module load miniconda
eval "$(conda shell.bash hook)"
export CONDA_ENV="${CONDA_ENV:-contact_kamino}"
conda activate "$CONDA_ENV"
set -u

N=$(python experiments/process_batch_results.py "$BATCH_DIR" --count)
(( N >= 1 )) || { echo "$BATCH_DIR has no cells (no cell_*.status.json); is it an episode-batch results folder?" >&2; exit 1; }

N_SHARDS="${N_SHARDS:-5}"
(( N_SHARDS >= 1 )) || { echo "N_SHARDS must be >= 1, got $N_SHARDS" >&2; exit 1; }
SHARDS_PER_GPU="${SHARDS_PER_GPU:-2}"
(( SHARDS_PER_GPU >= 1 )) || { echo "SHARDS_PER_GPU must be >= 1, got $SHARDS_PER_GPU" >&2; exit 1; }
N_TASKS=$(( (N * N_SHARDS + SHARDS_PER_GPU - 1) / SHARDS_PER_GPU ))
ARRAY="0-$((N_TASKS - 1))${THROTTLE:+%$THROTTLE}"
# OPTIMAL_ARGS reaches the jobs through ALL: it holds spaces and may hold commas, which --export cannot.
[[ -n "${OPTIMAL_ARGS+x}" ]] && export OPTIMAL_ARGS && echo "optimal planner args: ${OPTIMAL_ARGS:-none}"
EXPORT="ALL,BATCH_DIR=$BATCH_DIR,PROJ_DIR=$PROJ_DIR,N_CELLS=$N,N_SHARDS=$N_SHARDS,SHARDS_PER_GPU=$SHARDS_PER_GPU"
EXPORT+="${ANALYSIS_ARGS:+,ANALYSIS_ARGS=$ANALYSIS_ARGS}"
# --export=ALL,... keeps the environment `module load` needs inside the job. The CPUs and memory are the
# slurm header's per shard times the shards per GPU; SBATCH_ARGS comes after, so it can override them.
JOB=$(sbatch --parsable --array="$ARRAY" --export="$EXPORT" \
      --cpus-per-task=$((4 * SHARDS_PER_GPU)) --mem=$((16 * SHARDS_PER_GPU))G ${SBATCH_ARGS:-} \
      experiments/hpc/process_batch_results.slurm)
JOB=${JOB%%;*}
echo "submitted $N cells x $N_SHARDS shards of $BATCH_DIR, $SHARDS_PER_GPU per GPU, as array job $JOB (--array=$ARRAY)"

sbatch --parsable --dependency=afterany:"$JOB" --export="ALL,BATCH_DIR=$BATCH_DIR,PROJ_DIR=$PROJ_DIR" \
    experiments/hpc/summarize_batch_results.slurm >/dev/null \
    && echo "summary job queued after $JOB -> $BATCH_DIR/final_results.csv"
