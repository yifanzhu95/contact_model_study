#!/bin/bash
# submit_period_sweep.sh: run the period sweep as a job array, one model (one GPU) per task.
#
#   experiments/hpc/submit_period_sweep.sh                    # sync: M1-M6, six GPUs
#   experiments/hpc/submit_period_sweep.sh 3                  # at most 3 models at once
#   MODELS="M1 M3" PERIODS="16 64" SEEDS=3 experiments/hpc/submit_period_sweep.sh
#   OUTDIR=/abs/results/my_sweep experiments/hpc/submit_period_sweep.sh
#   experiments/hpc/submit_period_sweep.sh --async            # async instead: M1-M4 and M6
#   TAG=_bsig SWEEP_ARGS="--brownian-sigma" SBATCH_ARGS="--exclusive" \
#       experiments/hpc/submit_period_sweep.sh --async
#
# Sync by default (planning time free; the eval sim waits for every plan); --async, first, submits
# the async sweep (planning time spent as sim time), which measures latency and so wants nodes to
# itself (SBATCH_ARGS="--exclusive"). The runs are period_sweep_jobs.py's: one per model and control
# period (16-128 ms every 8 ms; log-spaced horizons up to 1.28 s, every step up to 8; each ~1 s from
# the initial pose toward the "S" goal (the cube flipped) with the same 10 seeds' planner noise,
# scored by its cost). MODELS, PERIODS (ms) and SEEDS narrow them. SWEEP_ARGS goes to every
# horizon_closed_loop.py run, and TAG is appended to every run's name so a variant does not skip on
# or overwrite the base runs. Results go to OUTDIR (default results/horizon_kl_sweep).
#
# Run from the repo root. It submits:
#   1. period_sweep.slurm: one array task per model, each running all of that model's runs on its GPU;
#   2. for sync, bench_plan_time.slurm (solo plan times: the sync plot's real-time cutoff);
#   3. plot_period_sweep.slurm, once those have finished (afterany): merges each run's chunks, then plots.
# Anything already in OUTDIR is skipped, so resubmitting reruns only what is missing.
set -euo pipefail

MODE=sync
if [[ ${1:-} == --async ]]; then MODE=async; shift; fi
if [[ $# -gt 1 || ( $# -eq 1 && ! $1 =~ ^[0-9]+$ ) ]]; then
    echo "usage: $0 [--async] [max concurrent models]" >&2
    exit 1
fi
THROTTLE="${1:-}"
PROJ_DIR="${PROJ_DIR:-$(pwd)}"
cd "$PROJ_DIR"
mkdir -p logs
OUTDIR=$(realpath -m "${OUTDIR:-$PROJ_DIR/results/horizon_kl_sweep}")
mkdir -p "$OUTDIR"

# period_sweep_jobs.py needs only the standard library, so the login node's python3 can list them.
# MODELS is deliberately unquoted: it holds a list.
MODEL_LIST=$(python3 experiments/period_sweep_jobs.py "$MODE" --stage models ${MODELS:+--models $MODELS})
N=$(wc -w <<< "$MODEL_LIST")
(( N >= 1 )) || { echo "no $MODE models" >&2; exit 1; }

# Everything below reaches the jobs through ALL: MODELS, PERIODS and SWEEP_ARGS may hold spaces
# or commas, which --export cannot.
export MODE OUTDIR PROJ_DIR TAG="${TAG:-}" CONDA_ENV="${CONDA_ENV:-contact_comfree2}"
export MODELS=$(echo $MODEL_LIST)
echo "$MODE models: $MODELS"
[[ -n "${PERIODS+x}" ]] && export PERIODS && echo "periods: $PERIODS ms"
[[ -n "${SEEDS+x}" ]] && export SEEDS && echo "seeds: $SEEDS"
[[ -n "${SWEEP_ARGS+x}" ]] && export SWEEP_ARGS && echo "sweep args: ${SWEEP_ARGS:-none}"

TIME=24:00:00
[[ $MODE == async ]] && TIME=06:00:00
# --export=ALL keeps the environment `module load` needs inside the job; SBATCH_ARGS comes after
# --time, so its own --time wins.
SWEEP=$(sbatch --parsable --job-name="period_sweep_$MODE" --array="0-$((N - 1))${THROTTLE:+%$THROTTLE}" \
        --export=ALL --time="$TIME" ${SBATCH_ARGS:-} experiments/hpc/period_sweep.slurm)
SWEEP=${SWEEP%%;*}
echo "submitted $N $MODE model tasks (one GPU each) as array job $SWEEP"
echo "results -> $OUTDIR"

AFTER="afterany:$SWEEP"
if [[ $MODE == sync ]]; then
    BENCH=$(sbatch --parsable --export=ALL experiments/hpc/bench_plan_time.slurm)
    BENCH=${BENCH%%;*}
    AFTER="$AFTER:$BENCH"
    echo "plan-time benchmark job $BENCH -> $OUTDIR/plan_time_solo.json"
fi

PFX=cl
[[ $MODE == async ]] && PFX=acl
sbatch --parsable --dependency="$AFTER" --export=ALL experiments/hpc/plot_period_sweep.slurm >/dev/null \
    && echo "merge and plot job queued after ${AFTER#afterany:} -> $OUTDIR/${PFX}_period${TAG}_onM5_T0.1_meancost_*.png"
