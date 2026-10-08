#!/bin/bash
# submit_episode_batches.sh: validate a batches CSV and submit it as a job array.
#
#   experiments/hpc/submit_episode_batches.sh experiments/example_batches.csv
#   experiments/hpc/submit_episode_batches.sh my.csv 8        # at most 8 cells at once
#   OUTDIR=/abs/results/episode_batches_my_123 \
#       experiments/hpc/submit_episode_batches.sh my.csv      # resume an earlier run
#
# Run from the repo root. It:
#   1. validates every row against run_episodes.py (so typos fail here, not on a node),
#   2. submits run_episode_batches.slurm with --array sized to the CSV's rows,
#   3. queues summarize_batches.slurm to run once the whole array has finished.
# Extra sbatch options can go in SBATCH_ARGS, e.g. SBATCH_ARGS="--time=24:00:00".
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <batches.csv> [max concurrent cells]" >&2
    exit 1
fi
CSV=$(realpath "$1")
THROTTLE="${2:-}"
[[ -f "$CSV" ]] || { echo "no such file: $CSV" >&2; exit 1; }
PROJ_DIR="${PROJ_DIR:-$(pwd)}"
cd "$PROJ_DIR"
mkdir -p logs

# The checks below run Python on the login node, so it needs the same conda
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

python experiments/run_episode_batches.py "$CSV" --check --outdir /tmp/episode_batches_check > /tmp/episode_batches_check.$$ \
    || { cat /tmp/episode_batches_check.$$ >&2; echo "CSV has invalid rows; nothing submitted" >&2; exit 1; }
N=$(python experiments/run_episode_batches.py "$CSV" --count)
(( N >= 1 )) || { echo "$CSV has no data rows" >&2; exit 1; }

ARRAY="0-$((N - 1))${THROTTLE:+%$THROTTLE}"
EXPORT="ALL,CSV=$CSV,PROJ_DIR=$PROJ_DIR${OUTDIR:+,OUTDIR=$OUTDIR}"
# --export=ALL,... keeps the environment `module load` needs inside the job.
JOB=$(sbatch --parsable --array="$ARRAY" --export="$EXPORT" ${SBATCH_ARGS:-} \
      experiments/hpc/run_episode_batches.slurm)
JOB=${JOB%%;*}
OUT="${OUTDIR:-$PROJ_DIR/results/episode_batches_$(basename "${CSV%.*}")_${JOB}}"
echo "submitted $N cells from $CSV as array job $JOB (--array=$ARRAY)"
echo "results -> $OUT"

sbatch --parsable --dependency=afterany:"$JOB" --export="ALL,CSV=$CSV,PROJ_DIR=$PROJ_DIR,OUTDIR=$OUT" \
    experiments/hpc/summarize_batches.slurm >/dev/null \
    && echo "summary job queued after $JOB -> $OUT/summary.csv"
