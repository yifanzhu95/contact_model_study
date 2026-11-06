#!/bin/bash
# submit_contact_param_sweep.sh: sweep every cell of an episode-batch results folder as a job array.
#
#   experiments/hpc/submit_contact_param_sweep.sh results/episode_batches_my_123456
#   experiments/hpc/submit_contact_param_sweep.sh results/episode_batches_my_123456 8   # at most 8 cells at once
#   EPISODES="0-4,9" experiments/hpc/submit_contact_param_sweep.sh <batch dir>          # these episodes of each cell
#   SWEEP_ARGS="--multipliers stiffness=0.5,1,2" experiments/hpc/submit_contact_param_sweep.sh <batch dir>
#   SBATCH_ARGS="--time=48:00:00" experiments/hpc/submit_contact_param_sweep.sh <batch dir>
#
# The steps per episode, score block and Kamino settings are set in contact_param_sweep.slurm's header;
# other flags go in SWEEP_ARGS. One array task per cell (an M5 cell is skipped at once).
#
# Run from the repo root. It:
#   1. counts the folder's cells (its cell_*.status.json files),
#   2. submits contact_param_sweep.slurm with --array sized to them,
#   3. queues summarize_contact_param_sweep.slurm to build sweep_results.csv once the array has finished.
# Cells already done are skipped and a cell cut off resumes from its checkpoint, so resubmitting resumes.
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
# env the jobs use; see submit_process_batch_results.sh for why the module
# system is set up by hand and strict mode paused around it.
set +u
if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        [[ -f "$init" ]] && { source "$init"; break; }
    done
fi
module load miniconda
eval "$(conda shell.bash hook)"
export CONDA_ENV="${CONDA_ENV:-contact_comfree2}"
conda activate "$CONDA_ENV"
set -u

N=$(python experiments/contact_param_sweep.py "$BATCH_DIR" --count)
(( N >= 1 )) || { echo "$BATCH_DIR has no cells (no cell_*.status.json); is it an episode-batch results folder?" >&2; exit 1; }

ARRAY="0-$((N - 1))${THROTTLE:+%$THROTTLE}"
# EPISODES and SWEEP_ARGS reach the jobs through ALL: they may hold commas, which --export cannot.
[[ -n "${EPISODES+x}" ]] && export EPISODES && echo "episodes: ${EPISODES:-all}"
[[ -n "${SWEEP_ARGS+x}" ]] && export SWEEP_ARGS && echo "sweep args: ${SWEEP_ARGS:-none}"
# --export=ALL,... keeps the environment `module load` needs inside the job.
JOB=$(sbatch --parsable --array="$ARRAY" --export="ALL,BATCH_DIR=$BATCH_DIR,PROJ_DIR=$PROJ_DIR" ${SBATCH_ARGS:-} \
      experiments/hpc/contact_param_sweep.slurm)
JOB=${JOB%%;*}
echo "submitted $N cells of $BATCH_DIR as array job $JOB (--array=$ARRAY)"

sbatch --parsable --dependency=afterany:"$JOB" --export="ALL,BATCH_DIR=$BATCH_DIR,PROJ_DIR=$PROJ_DIR" \
    experiments/hpc/summarize_contact_param_sweep.slurm >/dev/null \
    && echo "summary job queued after $JOB -> $BATCH_DIR/sweep_results.csv"
