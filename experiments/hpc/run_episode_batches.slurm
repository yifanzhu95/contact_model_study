#!/bin/bash
#SBATCH --job-name=episode_batches
#SBATCH --output=logs/episode_batches_%A_%a.out
#SBATCH --error=logs/episode_batches_%A_%a.err
#SBATCH --array=0-0            # placeholder: submit_episode_batches.sh sets it to the CSV's rows
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=11
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300    # 5 min before the limit: stop and save the finished episodes
# One array task = one CSV row (cell) = one GPU. By default the cell's episodes
# run at once (run_episode_batches.py --pool): PLANNERS_PER_GPU planner
# processes (2) serving WORKERS eval-sim workers (8), each worker one episode,
# with a core per process (hence 11 CPUs) and ~2 GB of memory each. With a GPU
# eval sim (M1-M5) every worker also holds a CUDA context on the GPU. Submit
# with POOL=0 to run the episodes one after another instead, as run_episodes.py
# does (2 CPUs and 16G are then enough). The resources above must cover
# the most expensive row: sbatch options on the command line override them,
# e.g. `sbatch --time=24:00:00 ...` for rows with long episodes or Pinocchio/Drake
# eval sims (see test_scripts/profile_run_episodes.py for per-step costs).
#
# Submit with experiments/hpc/submit_episode_batches.sh, which validates the CSV,
# sizes --array and queues the summary job. By hand:
#   sbatch --array=0-5 --export=ALL,CSV=/abs/batches.csv experiments/hpc/run_episode_batches.slurm

set -euo pipefail

# The repo root. The cluster copies this script before running it, so the path
# cannot be derived from the script's own location.
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"

CSV="${CSV:?set CSV=/abs/path/to/batches.csv (submit_episode_batches.sh does this)}"
CELL="${SLURM_ARRAY_TASK_ID:?run this as an array job}"
# Every task of the array writes into one folder, so the summary finds every cell.
# Point OUTDIR at an earlier run's folder to resume it: finished cells are skipped.
OUTDIR="${OUTDIR:-$PROJ_DIR/results/episode_batches_$(basename "${CSV%.*}")_${SLURM_ARRAY_JOB_ID}}"

module load miniconda
eval "$(conda shell.bash hook)"
# The submit script exports CONDA_ENV; contact_kamino (M1-M5) otherwise.
conda activate "${CONDA_ENV:-contact_kamino}"
export MUJOCO_GL=egl

echo "cell ${CELL} of ${CSV} -> ${OUTDIR} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

POOL_ARGS=()
if [[ "${POOL:-1}" == 1 ]]; then
    POOL_ARGS=(--pool --planners-per-gpu "${PLANNERS_PER_GPU:-2}" --workers "${WORKERS:-8}")
fi
# The out-of-range check is in the Python script, which parses the CSV properly.
# exec: Python is then the batch shell, so --signal=B:TERM reaches it directly.
exec python experiments/run_episode_batches.py "$CSV" --cell "$CELL" --outdir "$OUTDIR" \
    ${POOL_ARGS[@]+"${POOL_ARGS[@]}"}
