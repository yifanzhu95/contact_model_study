#!/bin/bash
#SBATCH --job-name=bayes_opt
#SBATCH --output=logs/bayes_opt_%A_%a.out
#SBATCH --error=logs/bayes_opt_%A_%a.err
#SBATCH --array=0-0            # placeholder: submit_bayes_opt.sh sets it to the CSV's rows
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# One array task = one CSV row (cell) = one whole Bayesian optimization. The cell
# runs every trial's episodes on one pool sized from this allocation: a planner
# process per GPU and an eval-sim worker per remaining core, with several trials
# in flight so they all stay busy. Each spawned process re-imports warp and
# mujoco, hence the memory. A cell's cost is n_calls x models x n_episodes
# episodes, so --time must cover the largest row. Change the allocation per
# submission with SBATCH_ARGS, e.g.
#   SBATCH_ARGS="--gpus=rtx_5000_ada:4 --cpus-per-task=32 --time=48:00:00"
# A cell that hits the limit keeps its finished trials, and resumes from them
# when resubmitted with OUTDIR (see submit_bayes_opt.sh).
#
# Submit with experiments/hpc/submit_bayes_opt.sh. By hand:
#   sbatch --array=0-2 --export=ALL,CSV=/abs/bo.csv experiments/hpc/run_bayes_opt.slurm

set -euo pipefail

# The repo root. The cluster copies this script before running it, so the path
# cannot be derived from the script's own location.
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"

CSV="${CSV:?set CSV=/abs/path/to/bo.csv (submit_bayes_opt.sh does this)}"
CELL="${SLURM_ARRAY_TASK_ID:?run this as an array job}"
# Every task of the array writes into one folder, so the summary finds every cell.
# Point OUTDIR at an earlier run's folder to resume it: finished trials are kept.
OUTDIR="${OUTDIR:-$PROJ_DIR/results/bayes_opt_$(basename "${CSV%.*}")_${SLURM_ARRAY_JOB_ID}}"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling
export MUJOCO_GL=egl

echo "cell ${CELL} of ${CSV} -> ${OUTDIR} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} CPUs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# The pool takes every GPU SLURM granted (CUDA_VISIBLE_DEVICES) and the CPUs of
# this task; the out-of-range check is in the Python script.
python experiments/run_bayes_opt.py "$CSV" --cell "$CELL" --outdir "$OUTDIR"
