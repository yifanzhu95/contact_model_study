#!/bin/bash
#SBATCH --job-name=bench_plan_time
#SBATCH --output=logs/bench_plan_time_%j.out
#SBATCH --error=logs/bench_plan_time_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00          # 381 horizons per model; resubmit to finish the rest
# The solo Plan() time of every model, control period and horizon of the
# period sweep (bench_plan_time.py), into OUTDIR/plan_time_solo.json, which the
# sync plot uses as its real-time cutoff. It has the GPU to itself, so run it on
# the same GPU type as the sweep (the --gpus line above). Keys already in the
# file are skipped, so a rerun only adds what is missing. M5 is left out: the
# plot takes its plan times from its own runs. Queued by submit_period_sweep.sh
# with the sync sweep. By hand:
#   OUTDIR=/abs/results/horizon_kl_sweep sbatch experiments/hpc/bench_plan_time.slurm

set -euo pipefail
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"
OUTDIR="${OUTDIR:-$PROJ_DIR/results/horizon_kl_sweep}"
mkdir -p "$OUTDIR"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-contact_comfree2}"
export MUJOCO_GL=egl

MODELS_ARGS=()
for m in ${MODELS:-M1 M2 M3 M4 M6}; do
    [[ $m == M5 ]] || MODELS_ARGS+=("$m")
done
(( ${#MODELS_ARGS[@]} )) || { echo "no models to benchmark (only M5)"; exit 0; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# PERIODS and BENCH_ARGS are deliberately unquoted: each holds a list.
exec python experiments/bench_plan_time.py --models "${MODELS_ARGS[@]}" ${PERIODS:+--periods $PERIODS} \
    --out "$OUTDIR/plan_time_solo.json" ${BENCH_ARGS:-}
