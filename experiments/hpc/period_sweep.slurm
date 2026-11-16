#!/bin/bash
#SBATCH --job-name=period_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-0            # placeholder: submit_period_sweep.sh sets it to the models
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00          # sync, set by M5; submit_period_sweep.sh asks for 6 h with --async
# One array task = one model of the period sweep = one GPU: task i takes model i
# of MODELS (period_sweep_jobs.py --stage models) and runs every one of its
# runs, one after another (run_period_sweep.sh, STAGE=chunks): 15 control
# periods, each with its 10-21 log-spaced horizons, each horizon run for ~1 s
# from the initial pose toward the "S" goal with each of the 10 seeds' planner
# noise. MODE (sync, or async with --async), MODELS, PERIODS, SEEDS, TAG,
# OUTDIR and SWEEP_ARGS come from the submit script.
#
# Scaled from the M2 preview on an RTX 4090 (Nov 2026), a sync task takes
# ~1.3-2.2 h for M1-M4 and M6 and ~7 h for M5, which plans several times
# slower; an async task ~1 h, since planning time is spent as sim time. Expect
# slower on the cluster's GPU. Each run is saved when it finishes and a run
# already saved is skipped, so a task cut off by the time limit resumes where
# it stopped when resubmitted.
#
# Async runs measure latency on the GPU they get, so every one should run on
# the same GPU type (the --gpus line above) and on an otherwise quiet node:
# submit with SBATCH_ARGS="--exclusive" if other jobs share the nodes.
#
# Submit with experiments/hpc/submit_period_sweep.sh. By hand:
#   sbatch --array=0-5 --export=ALL,MODE=sync,OUTDIR=/abs/results/horizon_kl_sweep \
#       experiments/hpc/period_sweep.slurm

set -euo pipefail

# The repo root. The cluster copies this script before running it, so the path
# cannot be derived from the script's own location.
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"

MODE="${MODE:-sync}"
TASK="${SLURM_ARRAY_TASK_ID:?run this as an array job}"
export STAGE=chunks
export OUTDIR="${OUTDIR:-$PROJ_DIR/results/horizon_kl_sweep}"
# This task's model. MODELS is deliberately unquoted: it holds a list.
MODELS=$(python3 experiments/period_sweep_jobs.py "$MODE" --stage models ${MODELS:+--models $MODELS} --job "$TASK")
export MODELS

module load miniconda
eval "$(conda shell.bash hook)"
# The submit script exports CONDA_ENV; contact_comfree2 (M1-M6) otherwise.
conda activate "${CONDA_ENV:-contact_comfree2}"
export MUJOCO_GL=egl

echo "$MODE $MODELS (task ${TASK}) -> ${OUTDIR} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ASYNC=()
[[ $MODE == async ]] && ASYNC=(--async)
# SWEEP_ARGS is deliberately unquoted: it holds extra horizon_closed_loop.py flags.
exec experiments/run_period_sweep.sh ${ASYNC[@]+"${ASYNC[@]}"} ${SWEEP_ARGS:-}
