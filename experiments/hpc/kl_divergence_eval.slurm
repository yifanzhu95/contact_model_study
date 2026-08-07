#!/bin/bash
#SBATCH --job-name=cont_modeling_kl_divergence_eval
#SBATCH --output=logs/cont_modeling_kl_divergence_eval%A_%a.out
#SBATCH --error=logs/cont_modeling_kl_divergence_eval%A_%a.err
#SBATCH --array=0-15          # must equal (#N_SAMPLES * #N_ITERS * #NULL) - 1
                              # default grid: 4 * 2 * 2 = 16  ->  0-15
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# KL divergence vs success rate, as a SLURM job array — same pattern as the
# other sweeps in this directory. Each array task is one degraded-planner
# config; each produces ONE point on the final scatter plot of
# (average KL divergence) vs (success rate).
#
# What is being measured
# ----------------------
# Two MPPI planners run on the SAME contact model and the SAME state:
#   * the DEGRADED planner (this cell's n_samples / n_iterations) plans every
#     control step and drives the eval sim;
#   * the REFERENCE ("optimal") planner is a shadow with a large sample count
#     and several iterations. It never controls.
# Each planner's weighted particle set {(V_n, w_n)} is an importance-sampling
# approximation of its induced optimal control distribution q*. Both are
# moment-matched to a Gaussian on the FIRST action and compared with the
# analytic Gaussian KL. See run_kl_divergence_cell.py's docstring for why the
# induced distribution (rather than the proposal) is the right object, and why
# the first action rather than the whole horizon.
#
# This is an OPTIMIZER-quality axis, not a model-fidelity axis: both planners
# use the same rollout model, so it is orthogonal to the M1-M4 results and
# should be labelled as such.
#
# The null-control axis
# ---------------------
# The degraded planner is degraded by having fewer samples, and fewer samples
# also make its moment estimates noisier — so KL rises with degradation partly
# for statistical rather than substantive reasons. Every config is therefore run
# TWICE: once normally, and once with --null_control, which rebuilds the
# reference planner with the degraded planner's own settings (different noise
# seed only). The null cell is that config's estimator-noise floor. A config
# whose real KL is not clearly above its own null floor is not reporting a
# meaningful difference, and its scatter point should be dropped.
# Null cells are also much cheaper, since both planners are then small.
#
# Cost control
# ------------
# --kl_every is the dominant cost knob: the reference planner is a shadow, so
# it only runs every N control steps. At the default 20 a 4000-step episode
# still yields ~200 KL samples, which is ample for an episode mean, while
# cutting the reference planner's cost 20x. Without this the sweep does not
# fit in any reasonable walltime. --max_steps is the other lever.
#
# WALLTIME IS UNVALIDATED. 16 h is an estimate extrapolated from a measured
# 80 ms/plan at 256 samples; the reference planner at 4096 samples x 4
# iterations was never timed. Run one cell interactively first (or the local
# sanity checks below), read the printed per-episode step time, and re-tune
# --time, --kl_every, --max_steps and N_EPISODES before submitting the array.
#
# Sanity checks to run BEFORE trusting any of this (locally, one episode):
#   1. Null floor is small:
#        python experiments/hpc/run_kl_divergence_cell.py --outdir /tmp/kl \
#            --n_samples 256 --n_iterations 1 --null_control \
#            --n_episodes 1 --max_steps 200 --kl_every 10 --debug
#   2. A deliberately terrible planner is far above that floor:
#        python experiments/hpc/run_kl_divergence_cell.py --outdir /tmp/kl \
#            --n_samples 4 --n_iterations 1 \
#            --n_episodes 1 --max_steps 200 --kl_every 10 --debug
#   If those two do not separate cleanly, the metric is not measuring planner
#   quality and more episodes will not fix it.
#   3. Check the printed ESS. The moment-matched KL needs non-uniform weights;
#      if ESS sits at ~N the induced distribution has collapsed back onto the
#      proposal and the covariance terms carry no signal.
#
# Keep #SBATCH --array in sync with the grid: it must be
# 0-((#N_SAMPLES_SWEEP * #N_ITER_SWEEP * #NULL_SWEEP) - 1). Default 4x2x2 = 16.
#
# Submit (from the repo root, so logs/results land at the project root -
# #SBATCH output paths are resolved relative to the sbatch submission dir):
#     mkdir -p logs results
#     sbatch experiments/hpc/kl_divergence_eval.slurm
#
# Results land in
# results/kl_divergence_eval_<arrayjobid>/<model>_n<samples>_i<iters>[_null].json
# ---------------------------------------------------------------------------

# Hardcoded project root so results always land in <root>/results regardless
# of where `sbatch` was invoked from.
PROJ_DIR="/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study"
ROOT_DIR="$PROJ_DIR"
cd "$PROJ_DIR"

# ---------------------------------------------------------------------------
# 1. Fixed settings + the grids to sweep.
# ---------------------------------------------------------------------------
TASK="grasp_reorient"
MODEL="M3"
N_EPISODES=30                   # episodes per cell (drives the success-rate SE)

# The swept degraded-planner compute. These two axes together define how far
# the approximation is from the reference.
N_SAMPLES_SWEEP=("16" "64" "256" "1024")
N_ITER_SWEEP=("1" "2")
# "no" = real comparison against the reference planner; "yes" = null control.
NULL_SWEEP=("no" "yes")

# The reference ("optimal") planner. Held fixed across the whole sweep so every
# cell is measured against the same target. Ignored in null cells.
REF_N_SAMPLES=4096
REF_N_ITERATIONS=4

# KL estimator knobs.
KL_EVERY=20                     # reference-planner duty cycle (cost knob)
KL_SHRINKAGE=0.001              # covariance shrinkage toward sigma^2 I
KL_DIRECTION="forward"          # headline only; both directions are stored
EXECUTE="mean"                  # "mean" matches the rest of the study

# Shared MPPI / eval knobs, matching run_eval_episode.py's CLI defaults. Both
# planners MUST agree on these — the KL is only defined if the two
# distributions live in the same space, and the worker raises if the resolved
# schedules differ.
TIME_HORIZON=0.352
STEP_TIME=0.064
TEMPERATURE=1.0
NOISE_SIGMA=0.025
EVAL_SIM="none"                 # "none" uses the task default
SETTLE=1.0
GEOMETRY="accurate"
SEED=0

# ---------------------------------------------------------------------------
# 2. Map this task's ID to one index per axis (mixed-radix decode).
#    Peel off the lowest axis first (n_samples), then divide it out.
# ---------------------------------------------------------------------------
NS=${#N_SAMPLES_SWEEP[@]}; NI=${#N_ITER_SWEEP[@]}; NN=${#NULL_SWEEP[@]}

T="${SLURM_ARRAY_TASK_ID:-0}"
S_IDX=$(( T % NS )); T=$(( T / NS ))
I_IDX=$(( T % NI )); T=$(( T / NI ))
N_IDX=$(( T % NN ))

# ---------------------------------------------------------------------------
# 3. Select the actual values for this cell.
# ---------------------------------------------------------------------------
SEL_N_SAMPLES=${N_SAMPLES_SWEEP[$S_IDX]}
SEL_N_ITERS=${N_ITER_SWEEP[$I_IDX]}
SEL_NULL=${NULL_SWEEP[$N_IDX]}

if [ "$SEL_NULL" = "yes" ]; then
    NULL_FLAG="--null_control"
else
    NULL_FLAG="--no-null_control"
fi

OUTDIR="${OUTDIR:-$ROOT_DIR/results/kl_divergence_eval_${SLURM_ARRAY_JOB_ID:-manual}}"
mkdir -p "$OUTDIR"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

echo "cell ${SLURM_ARRAY_TASK_ID:-0}: model=$MODEL n_samples=$SEL_N_SAMPLES "\
     "n_iterations=$SEL_N_ITERS null_control=$SEL_NULL "\
     "ref=${REF_N_SAMPLES}x${REF_N_ITERATIONS} kl_every=$KL_EVERY "\
     "n_episodes=$N_EPISODES -> $OUTDIR"

python experiments/hpc/run_kl_divergence_cell.py \
    --outdir           "$OUTDIR" \
    --task             "$TASK" \
    --model            "$MODEL" \
    --n_samples        "$SEL_N_SAMPLES" \
    --n_iterations     "$SEL_N_ITERS" \
    --ref_n_samples    "$REF_N_SAMPLES" \
    --ref_n_iterations "$REF_N_ITERATIONS" \
    $NULL_FLAG \
    --n_episodes       "$N_EPISODES" \
    --kl_every         "$KL_EVERY" \
    --kl_shrinkage     "$KL_SHRINKAGE" \
    --kl_direction     "$KL_DIRECTION" \
    --execute          "$EXECUTE" \
    --time_horizon     "$TIME_HORIZON" \
    --step_time        "$STEP_TIME" \
    --temperature      "$TEMPERATURE" \
    --noise_sigma      "$NOISE_SIGMA" \
    --eval_sim         "$EVAL_SIM" \
    --settle           "$SETTLE" \
    --geometry         "$GEOMETRY" \
    --seed             "$SEED" \
    --debug
