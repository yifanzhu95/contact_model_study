#!/bin/bash
# submit_process_batch_results.sh: analyse every cell of an episode-batch results folder as a job array.
#
#   experiments/hpc/submit_process_batch_results.sh results/episode_batches_my_123456
#   experiments/hpc/submit_process_batch_results.sh results/episode_batches_my_123456 8   # at most 8 cells at once
#   ANALYSIS_ARGS="--stride 4" experiments/hpc/submit_process_batch_results.sh <batch dir>
#
# Run from the repo root. It:
#   1. counts the folder's cells (its cell_*.status.json files),
#   2. submits process_batch_results.slurm with --array sized to them,
#   3. queues summarize_batch_results.slurm to build final_results.csv once the array has finished.
# Cells already analysed are skipped, so resubmitting resumes. Extra sbatch
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

ARRAY="0-$((N - 1))${THROTTLE:+%$THROTTLE}"
EXPORT="ALL,BATCH_DIR=$BATCH_DIR,PROJ_DIR=$PROJ_DIR${ANALYSIS_ARGS:+,ANALYSIS_ARGS=$ANALYSIS_ARGS}"
# --export=ALL,... keeps the environment `module load` needs inside the job.
JOB=$(sbatch --parsable --array="$ARRAY" --export="$EXPORT" ${SBATCH_ARGS:-} \
      experiments/hpc/process_batch_results.slurm)
JOB=${JOB%%;*}
echo "submitted $N cells of $BATCH_DIR as array job $JOB (--array=$ARRAY)"

sbatch --parsable --dependency=afterany:"$JOB" --export="ALL,BATCH_DIR=$BATCH_DIR,PROJ_DIR=$PROJ_DIR" \
    experiments/hpc/summarize_batch_results.slurm >/dev/null \
    && echo "summary job queued after $JOB -> $BATCH_DIR/final_results.csv"
