#!/bin/bash
#SBATCH --job-name=period_sweep_plots
#SBATCH --output=logs/period_sweep_plots_%j.out
#SBATCH --error=logs/period_sweep_plots_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
# Merges each run's horizon chunks into OUTDIR/NAME.{json,npz}
# (run_period_sweep.sh, STAGE=merge; a run with chunks missing is left out and
# listed), then plots the runs in OUTDIR (plot_period_sweep.py): the line plots,
# the heatmaps and the frontiers, as
# OUTDIR/{cl,acl}_period<TAG>_onM5_T0.1_meancost_*.png. Queued by
# submit_period_sweep.sh to run after the chunk array (afterany, so it also
# plots what finished when some tasks failed) and, for sync, after the
# plan-time benchmark. Rerun it once the missing chunks are done. By hand:
#   OUTDIR=/abs/results/horizon_kl_sweep sbatch experiments/hpc/plot_period_sweep.slurm   # MODE=async for async

set -euo pipefail
PROJ_DIR="${PROJ_DIR:-/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study}"
cd "$PROJ_DIR"
MODE="${MODE:-sync}"
export OUTDIR="${OUTDIR:-$PROJ_DIR/results/horizon_kl_sweep}"
export TAG="${TAG:-}"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-contact_comfree2}"

# MODELS, PERIODS, TAG and OUTDIR come from the environment, as the sweep had them.
ASYNC=()
[[ $MODE == async ]] && ASYNC=(--async)
STAGE=merge experiments/run_period_sweep.sh ${ASYNC[@]+"${ASYNC[@]}"} || echo "some runs were not merged (see above)"

ARGS=(--tag "$TAG")
PREFIX="$OUTDIR/cl_period${TAG}_onM5_T0.1_meancost"
if [[ $MODE == async ]]; then
    ARGS+=(--async)
    PREFIX="$OUTDIR/acl_period${TAG}_onM5_T0.1_meancost"
fi
RC=0
for VIEW in "" --heatmap --frontiers; do
    python experiments/plot_period_sweep.py "${ARGS[@]}" $VIEW "$PREFIX" "$OUTDIR" || RC=1
done
exit $RC
