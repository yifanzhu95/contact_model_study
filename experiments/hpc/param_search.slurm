#!/bin/bash
#SBATCH --job-name=cont_modeling_param_search
#SBATCH --output=logs/cont_modeling_param_search%A_%a.out
#SBATCH --error=logs/cont_modeling_param_search%A_%a.err
#SBATCH --array=0-323          # must equal (#MODELS * #W_QUAT * #W_POS * #W_CONTACT * #W_JOINT) - 1
                               # default grid: 4 * 3 * 3 * 3 * 3 = 324  ->  0-323
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

# Hardcoded project root so results always land in <root>/results regardless
# of where `sbatch` was invoked from.
PROJ_DIR="/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study"
ROOT_DIR="$PROJ_DIR"
cd "$PROJ_DIR" #cd "$PROJ_DIR/experiments/hpc"

# ---------------------------------------------------------------------------
# 1. Fixed settings + the parameter grids to sweep.
#    Edit the arrays below and keep #SBATCH --array in sync (see the product
#    of their lengths above).
# ---------------------------------------------------------------------------
TASK="grasp_reorient"
N_EPISODES=5          # episodes per weight set -> success rate

MODELS=("M1" "M2" "M3" "M4")
W_QUAT=("100" "75" "50")
W_POS=("500" "400" "300")
W_CONTACT=("1000" "750" "500")
W_JOINT=("8" "6" "4")

# Shared MPPI / eval knobs (single values, not swept).
N_SAMPLES=256
HORIZON=8
TEMPERATURE=10.0
NOISE_SIGMA=0.01
DELTA=0.1
SUBSTEPS=8
EVAL_SIM="none" #"none"
SETTLE=1.0
GEOMETRY="accurate"
SEED=0

# MPPI noise resampling / time-constrained planning. Both are opt-in; the
# values below reproduce the original behavior exactly.
RESAMPLE_INTERVAL=""     # plan steps between noise resamples (1=every step); empty=never
TIME_CONSTRAINED=0       # 1 = stop rollouts once PLAN_BUDGET_MS elapses
PLAN_BUDGET_MS=""        # rollout budget per plan in ms; required when TIME_CONSTRAINED=1

# ---------------------------------------------------------------------------
# 2. Map this task's ID to one index per axis (mixed-radix decode).
#    Peel off the lowest axis first, then divide it out for the next.
# ---------------------------------------------------------------------------
NM=${#MODELS[@]}; NQ=${#W_QUAT[@]}; NP=${#W_POS[@]}; NC=${#W_CONTACT[@]}; NJ=${#W_JOINT[@]}

T=$SLURM_ARRAY_TASK_ID
MODEL_IDX=$((   T % NM )); T=$(( T / NM ))
QUAT_IDX=$((    T % NQ )); T=$(( T / NQ ))
POS_IDX=$((     T % NP )); T=$(( T / NP ))
CONTACT_IDX=$(( T % NC )); T=$(( T / NC ))
JOINT_IDX=$((   T % NJ ))

# ---------------------------------------------------------------------------
# 3. Select the actual values for this cell.
# ---------------------------------------------------------------------------
SEL_MODEL=${MODELS[$MODEL_IDX]}
SEL_W_QUAT=${W_QUAT[$QUAT_IDX]}
SEL_W_POS=${W_POS[$POS_IDX]}
SEL_W_CONTACT=${W_CONTACT[$CONTACT_IDX]}
SEL_W_JOINT=${W_JOINT[$JOINT_IDX]}

# All array tasks share one output dir (keyed by the array job id) so the
# combine job can gather every cell_*.json.
OUTDIR="${OUTDIR:-$ROOT_DIR/results/param_search_${SLURM_ARRAY_JOB_ID:-manual}}"

module load miniconda

eval "$(conda shell.bash hook)"

conda activate contact_modeling

echo "cell ${SLURM_ARRAY_TASK_ID}: model=$SEL_MODEL "\
     "w_quat=$SEL_W_QUAT w_pos=$SEL_W_POS w_contact=$SEL_W_CONTACT w_joint=$SEL_W_JOINT"\
     "-> $OUTDIR"

# The first task queues the combine job to run once the whole array finishes
# ok, so a single `sbatch param_search.slurm` produces the merged results too.

# Only pass the opt-in MPPI flags when they are actually set, so the default
# configuration invokes run_param_cell.py exactly as before.
EXTRA_ARGS=()
[[ -n "$RESAMPLE_INTERVAL" ]] && EXTRA_ARGS+=(--resample_interval "$RESAMPLE_INTERVAL")
[[ "$TIME_CONSTRAINED" == "1" ]] && EXTRA_ARGS+=(--time_constrained --plan_budget_ms "$PLAN_BUDGET_MS")

python experiments/hpc/run_param_cell.py \
    --cell_id     "$SLURM_ARRAY_TASK_ID" \
    --outdir      "$OUTDIR" \
    --task        "$TASK" \
    --model       "$SEL_MODEL" \
    --n_episodes  "$N_EPISODES" \
    --weights     "w_quat=$SEL_W_QUAT" "w_pos=$SEL_W_POS" \
                  "w_contact=$SEL_W_CONTACT" "w_joint=$SEL_W_JOINT" \
    --n_samples   "$N_SAMPLES" \
    --horizon     "$HORIZON" \
    --temperature "$TEMPERATURE" \
    --noise_sigma "$NOISE_SIGMA" \
    --delta       "$DELTA" \
    --substeps    "$SUBSTEPS" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --geometry    "$GEOMETRY" \
    --seed        "$SEED" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    --debug
