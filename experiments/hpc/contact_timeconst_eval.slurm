#!/bin/bash
#SBATCH --job-name=cont_modeling_contact_timeconst_eval
#SBATCH --output=logs/cont_modeling_contact_timeconst_eval%A_%a.out
#SBATCH --error=logs/cont_modeling_contact_timeconst_eval%A_%a.err
#SBATCH --array=0-4           # must equal (#MODELS * #TIMECONST_SWEEP) - 1
                              # default grid: 1 * 5 = 5  ->  0-4
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Contact-time-constant sweep as a SLURM job array — same pattern as
# cntrl_freq_eval.slurm, but the swept axis is the ROLLOUT model's contact
# stiffness instead of the control period. The grid (MODELS x
# TIMECONST_SWEEP) is defined inline; this script maps $SLURM_ARRAY_TASK_ID to
# one (model, timeconst) cell, then runs N_EPISODES episodes for that cell via
# run_contact_timeconst_cell.py, which calls run_eval_episode. One array task =
# one GPU = one cell. No videos.
#
# What the axis means
# -------------------
# A MuJoCo contact row's solref is (timeconst, dampratio). timeconst governs how
# fast position-level penetration is driven out: smaller = stiffer contact,
# larger = softer. The worker sets it through
# ContactModelConfig.mujoco.solref_timeconst, which patches every geom/pair
# solref at put_model time and leaves solimp untouched — so this is a pure
# contact-stiffness axis, not the coupled stiffness+regularizer change that the
# M1 hard-contact preset makes.
#
# Why M2 only
# -----------
# M2's default is MuJoCo's compiler default solref = "0.02 1" (timeconst 20 ms):
# the leap scenes declare no solref of their own — the only such line, in
# scenes/leap/leap_right_hand.xml, is commented out. So 0.02 s is a real,
# meaningful centre for the sweep. M1's timeconst is instead derived as
# max(mult*dt, 2*dt) and hard-clamped at 2*dt, so M1 cannot be swept stiffer
# than its default without editing that clamp in contact_models/api.py. M3
# (ComFree) and M4 (XPBD) do not use solref at all.
#
# The grid
# --------
# x2 and /2 twice around the 0.02 s default would put the stiffest point at
# 0.005 s, below the 2*rollout_dt = 0.008 s floor where the semi-implicit
# integrator rings or diverges. The grid is therefore shifted up one notch —
# still five points at 2x spacing, all stable, with 0.01 s one notch stiffer
# than default and three notches softer above it. The worker warns (and runs
# anyway) if a cell lands under the floor, so re-centring the grid later is
# safe.
#
# Only the ROLLOUT model changes. The eval simulator stays whatever the task
# defaults to (Pinocchio for grasp_reorient; EVAL_SIM="none" keeps that
# default), so this measures how much rollout-model contact fidelity buys in
# closed-loop success against a fixed ground truth.
#
# The control period is held FIXED at STEP_TIME for every cell (unlike
# cntrl_freq_eval.slurm, which sweeps it), so the only thing changing across
# the grid is contact stiffness.
#
# Keep #SBATCH --array in sync with the grid: it must be
# 0-((#MODELS * #TIMECONST_SWEEP) - 1). The default 1x5 = 5 -> 0-4. The decode
# below keeps the model axis even though there is only one model, so adding M1
# back is a one-line change here plus the array bound.
#
# Submit (from the repo root, so logs/results land at the project root -
# #SBATCH output paths are resolved relative to the sbatch submission dir):
#     mkdir -p logs results
#     sbatch experiments/hpc/contact_timeconst_eval.slurm
#
# Results land in
# results/contact_timeconst_eval_<arrayjobid>/<model>_tc<microseconds>.json.
# Plot them all together with:
#     python analysis/plot_contact_timeconst_sweep_dir.py \
#         results/contact_timeconst_eval_<id>
# ---------------------------------------------------------------------------

# Hardcoded project root so results always land in <root>/results regardless
# of where `sbatch` was invoked from.
PROJ_DIR="/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study"
ROOT_DIR="$PROJ_DIR"
cd "$PROJ_DIR" #cd "$PROJ_DIR/experiments/hpc"

# ---------------------------------------------------------------------------
# 1. Fixed settings + the grids to sweep.
#    Edit the arrays below and keep #SBATCH --array in sync (the product of
#    their lengths above).
# ---------------------------------------------------------------------------
TASK="grasp_reorient"
N_EPISODES=30                                   # episodes per (model, timeconst) cell

MODELS=("M2")
# The swept contact time constants, in SECONDS. M2's default is 0.02 s;
# 2x spacing, shifted up one notch to clear the 2*rollout_dt = 0.008 s floor.
TIMECONST_SWEEP=("0.01" "0.02" "0.04" "0.08" "0.16")

# Shared MPPI / eval knobs (single values, not swept). Match
# cntrl_freq_eval.slurm's nominal cell so the results are comparable to it.
STEP_TIME=0.032                 # control period in seconds (FIXED here)
N_SAMPLES=256
TIME_HORIZON=0.256              # planning horizon in seconds
TEMPERATURE=1.0
NOISE_SIGMA=0.025
EVAL_SIM="none"                 # "none" uses the task default; "mujoco" is faster
SETTLE=1.0
GEOMETRY="accurate"
SEED=0

# ---------------------------------------------------------------------------
# 2. Map this task's ID to one index per axis (mixed-radix decode).
#    Peel off the lowest axis first (model), then divide it out for the next.
# ---------------------------------------------------------------------------
NM=${#MODELS[@]}; NT=${#TIMECONST_SWEEP[@]}

T="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_IDX=$(( T % NM )); T=$(( T / NM ))
TC_IDX=$(( T % NT ))

# ---------------------------------------------------------------------------
# 3. Select the actual values for this cell.
# ---------------------------------------------------------------------------
SEL_MODEL=${MODELS[$MODEL_IDX]}
SEL_TIMECONST=${TIMECONST_SWEEP[$TC_IDX]}

# All array tasks share one output dir (keyed by the array job id) so the plot
# script can merge the whole directory of per-cell JSONs.
OUTDIR="${OUTDIR:-$ROOT_DIR/results/contact_timeconst_eval_${SLURM_ARRAY_JOB_ID:-manual}}"
mkdir -p "$OUTDIR"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate contact_modeling

echo "cell ${SLURM_ARRAY_TASK_ID:-0}: model=$SEL_MODEL "\
     "solref_timeconst=${SEL_TIMECONST}s step_time=${STEP_TIME}s "\
     "time_horizon=${TIME_HORIZON}s n_episodes=$N_EPISODES -> $OUTDIR"

python experiments/hpc/run_contact_timeconst_cell.py \
    --outdir           "$OUTDIR" \
    --task             "$TASK" \
    --model            "$SEL_MODEL" \
    --solref_timeconst "$SEL_TIMECONST" \
    --n_episodes       "$N_EPISODES" \
    --step_time        "$STEP_TIME" \
    --n_samples        "$N_SAMPLES" \
    --time_horizon     "$TIME_HORIZON" \
    --temperature      "$TEMPERATURE" \
    --noise_sigma      "$NOISE_SIGMA" \
    --eval_sim         "$EVAL_SIM" \
    --settle           "$SETTLE" \
    --geometry         "$GEOMETRY" \
    --seed             "$SEED" \
    --debug
