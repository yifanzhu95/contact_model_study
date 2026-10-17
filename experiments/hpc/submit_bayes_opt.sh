#!/bin/bash
# submit_bayes_opt.sh: validate a Bayesian-optimization CSV and submit it, each row sized to its work.
#
#   experiments/hpc/submit_bayes_opt.sh experiments/example_bayes_opt.csv
#   experiments/hpc/submit_bayes_opt.sh my.csv 8        # at most 8 cells of each size at once
#   OUTDIR=/abs/results/bayes_opt_my_123 \
#       experiments/hpc/submit_bayes_opt.sh my.csv      # resume: finished trials are kept
#
# Run from the repo root. It:
#   1. validates every row (search space, seeds) against the driver (so typos fail here, not on a node),
#   2. sizes each row from its episodes per BO step (models x n_episodes x trials_in_flight): GPUs
#      (PLANNERS_PER_GPU planners each, 2 by default; at most MAX_GPUS, 4), 2 eval workers per
#      planner, 1 when M5 is a rollout model (never more than the episodes), a core per process and ~2.5 GB of memory each
#      (run_bayes_opt.py --resources prints this),
#   3. submits run_bayes_opt.slurm once per allocation, as a job array over the rows that share it,
#      all writing into one results folder,
#   4. queues summarize_bayes_opt.slurm to run once every array has finished.
# GPU_TYPE picks the GPU (default a5000). Extra sbatch options go in SBATCH_ARGS and win over the
# sizing, e.g. SBATCH_ARGS="--time=24:00:00" or SBATCH_ARGS="--gpus=rtx_5000_ada:4".
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <bo.csv> [max concurrent cells]" >&2
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

python experiments/run_bayes_opt.py "$CSV" --check --outdir /tmp/bayes_opt_check > /tmp/bayes_opt_check.$$ \
    || { cat /tmp/bayes_opt_check.$$ >&2; echo "CSV has invalid rows; nothing submitted" >&2; exit 1; }
N=$(python experiments/run_bayes_opt.py "$CSV" --count)
(( N >= 1 )) || { echo "$CSV has no data rows" >&2; exit 1; }

PLANNERS_PER_GPU="${PLANNERS_PER_GPU:-2}"
GPU_TYPE="${GPU_TYPE:-a5000}"
# One line per row: cell episodes gpus planners_per_gpu workers cpus mem_gb.
SIZES=$(python experiments/run_bayes_opt.py "$CSV" --resources --planners-per-gpu "$PLANNERS_PER_GPU" \
            --max-gpus "${MAX_GPUS:-4}" --outdir /tmp/bayes_opt_check | grep -E '^[0-9]+( [0-9]+){6}$')
(( $(wc -l <<< "$SIZES") == N )) || { echo "could not size every row:" >&2; echo "$SIZES" >&2; exit 1; }

# Rows that need the same allocation share a job array.
declare -A CELLS
ORDER=()
while read -r cell eps gpus ppg workers cpus mem; do
    echo "  cell $cell: $eps episodes per BO step -> $gpus GPU(s) x $ppg planners, $workers workers, $cpus CPUs, ${mem}G"
    key="$gpus $ppg $workers $cpus $mem"
    [[ -n "${CELLS[$key]:-}" ]] || ORDER+=("$key")
    CELLS[$key]="${CELLS[$key]:+${CELLS[$key]},}$cell"
done <<< "$SIZES"

JOBS=()
OUT="${OUTDIR:-}"
for key in "${ORDER[@]}"; do
    read -r gpus ppg workers cpus mem <<< "$key"
    ARRAY="${CELLS[$key]}${THROTTLE:+%$THROTTLE}"
    # --export=ALL,... keeps the environment `module load` needs inside the job. Every array
    # after the first writes into the first one's folder, so the summary finds every cell.
    EXPORT="ALL,CSV=$CSV,PROJ_DIR=$PROJ_DIR,PLANNERS_PER_GPU=$ppg,WORKERS=$workers${OUT:+,OUTDIR=$OUT}"
    # --nodes=1: --gpus is a total for the job, and SLURM may otherwise split it over nodes.
    JOB=$(sbatch --parsable --array="$ARRAY" --export="$EXPORT" --nodes=1 --ntasks=1 --gpus="$GPU_TYPE:$gpus" \
          --cpus-per-task="$cpus" --mem="${mem}G" ${SBATCH_ARGS:-} experiments/hpc/run_bayes_opt.slurm)
    JOB=${JOB%%;*}
    JOBS+=("$JOB")
    OUT="${OUT:-$PROJ_DIR/results/bayes_opt_$(basename "${CSV%.*}")_${JOB}}"
    echo "submitted cells $ARRAY as array job $JOB: $gpus x $GPU_TYPE, $cpus CPUs, ${mem}G"
done
echo "results -> $OUT"

DEPS=$(IFS=:; echo "${JOBS[*]}")
sbatch --parsable --dependency=afterany:"$DEPS" --export="ALL,CSV=$CSV,PROJ_DIR=$PROJ_DIR,OUTDIR=$OUT" \
    experiments/hpc/summarize_bayes_opt.slurm >/dev/null \
    && echo "summary job queued after ${JOBS[*]} -> $OUT/summary.csv and best.csv"
