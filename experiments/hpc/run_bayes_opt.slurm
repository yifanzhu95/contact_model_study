#!/bin/bash
#SBATCH --job-name=bayes_opt
#SBATCH --output=logs/bayes_opt_%A_%a.out
#SBATCH --error=logs/bayes_opt_%A_%a.err
#SBATCH --array=0-0            # placeholder: submit_bayes_opt.sh sets it to the CSV's rows
#SBATCH --partition=gpu
#SBATCH --gpus=a5000:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=48:00:00
# One array task = one CSV row (cell) = one whole Bayesian optimization. The cell
# runs every trial's episodes on one pool: PLANNERS_PER_GPU planner processes per
# GPU and WORKERS eval-sim workers (see below), with several trials in flight so
# they all stay busy. Each spawned process re-imports warp and
# mujoco, hence the memory. A cell's cost is n_calls x models x n_episodes
# episodes, so --time must cover the largest row. submit_bayes_opt.sh sizes the
# GPUs, CPUs, memory, PLANNERS_PER_GPU and WORKERS per row from its episodes per
# BO step; the values here are for submitting by hand. Override with SBATCH_ARGS, e.g.
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
# The submit script exports CONDA_ENV; contact_kamino (M1-M5) otherwise.
conda activate "${CONDA_ENV:-contact_kamino}"
export MUJOCO_GL=egl

echo "cell ${CELL} of ${CSV} -> ${OUTDIR} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} CPUs"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# compute_mode must be Default: Exclusive_Process lets one process per GPU in, and the
# pool puts several planners and workers on each.
nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader || true

# Every GPU_LOG_S seconds (300), each GPU's utilization, memory and process count, so
# an idle GPU shows in the .out log, with whether the pool has processes on it.
gpuLog() {
    while sleep "${GPU_LOG_S:-300}"; do
        local apps line
        apps=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader 2>/dev/null)
        line=$(nvidia-smi --query-gpu=index,pci.bus_id,utilization.gpu,memory.used --format=csv,noheader,nounits \
               2>/dev/null | while IFS=', ' read -r i bus util mem; do
                   echo -n "  gpu$i ${util}% ${mem}MiB $(grep -c "$bus" <<< "$apps") procs;"
               done)
        echo "[gpus $(date +%H:%M:%S)]$line"
    done
}
gpuLog &
GPU_LOG_PID=$!
trap 'kill $GPU_LOG_PID 2>/dev/null' EXIT

# MPPI debug: one [MPPI] line per plan in the .out log. On by default; submit with
# DEBUG=0 to turn it off.
EXTRA=()
[[ "${DEBUG:-1}" == 1 ]] && EXTRA+=(--debug)
# Pool size. 2 planners per GPU ran M5 ~1.5x faster than 1 (3 added little), and
# 8 planners + 12 workers fit the 24 CPUs above, at ~1.8 GB of memory each.
# Override per submission, e.g. PLANNERS_PER_GPU=1 WORKERS=8.
EXTRA+=(--planners-per-gpu "${PLANNERS_PER_GPU:-2}" --workers "${WORKERS:-12}")

# The pool takes every GPU SLURM granted (CUDA_VISIBLE_DEVICES) and the CPUs of
# this task; the out-of-range check is in the Python script.
python experiments/run_bayes_opt.py "$CSV" --cell "$CELL" --outdir "$OUTDIR" ${EXTRA[@]+"${EXTRA[@]}"}
