#!/bin/bash
#SBATCH --job-name=cont_modeling_param_search
#SBATCH --output=logs/cont_modeling_param_search%A_%a.out
#SBATCH --error=logs/cont_modeling_param_search%A_%a.err
#SBATCH --array=0-80          # must equal (#MODELS * #W_QUAT * #W_POS * #W_CONTACT * #W_JOINT) - 1
                               # default grid: 4 * 3 * 3 * 3 * 3 = 324  ->  0-323
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

# Resolve paths from the script's own location, not the submission directory
# (SLURM_SUBMIT_DIR is wherever `sbatch` was invoked from, which may not be
# experiments/hpc), so results always land in <root>/results regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 1. Fixed settings + the parameter grids to sweep.
#    Edit the arrays below and keep #SBATCH --array in sync (see the product
#    of their lengths above).
# ---------------------------------------------------------------------------
TASK="grasp_reorient"
N_EPISODES=5          # episodes per weight set -> success rate

MODELS=("M2")
W_QUAT=("75" "50" "25")
W_POS=("600" "400" "200")
W_CONTACT=("750" "500" "250")
W_JOINT=("100" "50" "5")

# Shared MPPI / eval knobs (single values, not swept).
N_SAMPLES=256
HORIZON=48
TEMPERATURE=1.0
NOISE_SIGMA=0.01
DELTA=0.1
SUBSTEPS=16
EVAL_SIM="none"
SETTLE=1.0
GEOMETRY="accurate"
SEED=0

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

python run_param_cell.py \
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
    --seed        "$SEED"
