#!/bin/bash
#SBATCH --job-name=cont_modeling_param_search
#SBATCH --output=logs/cont_modeling_param_search%A_%a.out
#SBATCH --error=logs/cont_modeling_param_search%A_%a.err
#SBATCH --array=0-161         # must equal (#MODELS * #W_QUAT * #W_POS_X * #W_POS_Y * #W_POS_Z * #W_CONTACT * #W_JOINT) - 1
                               # default grid: 4 * 3 * 3 * 3 * 3 * 3 * 3 = 2916  ->  0-2915
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00

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

MODELS=("M2" "M3") #("M1" "M2" "M3" "M4")
W_QUAT=("20.0" "10.0" "5.0")
# Position error is weighted separately per axis (X is down the fingers,
# Y across the fingers, Z vertical) -> three independent search axes.
W_POS_X=("15" "7.5" "3.75")
W_POS_Y=("30" "15" "7.5")
W_POS_Z=("15" "7.5" "3.75")
W_CONTACT=("5.0")
W_JOINT=("0.6")

# Shared MPPI / eval knobs (single values, not swept).
N_SAMPLES=256
HORIZON=8
TEMPERATURE=0.01
NOISE_SIGMA=0.02
DELTA=0.1
SUBSTEPS=8
EVAL_SIM="none" #"none"
SETTLE=1.0
GEOMETRY="accurate"
SEED=0

# ---------------------------------------------------------------------------
# 2. Map this task's ID to one index per axis (mixed-radix decode).
#    Peel off the lowest axis first, then divide it out for the next.
# ---------------------------------------------------------------------------
NM=${#MODELS[@]}; NQ=${#W_QUAT[@]}
NPX=${#W_POS_X[@]}; NPY=${#W_POS_Y[@]}; NPZ=${#W_POS_Z[@]}
NC=${#W_CONTACT[@]}; NJ=${#W_JOINT[@]}

T=$SLURM_ARRAY_TASK_ID
MODEL_IDX=$((   T % NM  )); T=$(( T / NM  ))
QUAT_IDX=$((    T % NQ  )); T=$(( T / NQ  ))
POS_X_IDX=$((   T % NPX )); T=$(( T / NPX ))
POS_Y_IDX=$((   T % NPY )); T=$(( T / NPY ))
POS_Z_IDX=$((   T % NPZ )); T=$(( T / NPZ ))
CONTACT_IDX=$(( T % NC  )); T=$(( T / NC  ))
JOINT_IDX=$((   T % NJ  ))

# ---------------------------------------------------------------------------
# 3. Select the actual values for this cell.
# ---------------------------------------------------------------------------
SEL_MODEL=${MODELS[$MODEL_IDX]}
SEL_W_QUAT=${W_QUAT[$QUAT_IDX]}
SEL_W_POS_X=${W_POS_X[$POS_X_IDX]}
SEL_W_POS_Y=${W_POS_Y[$POS_Y_IDX]}
SEL_W_POS_Z=${W_POS_Z[$POS_Z_IDX]}
SEL_W_CONTACT=${W_CONTACT[$CONTACT_IDX]}
SEL_W_JOINT=${W_JOINT[$JOINT_IDX]}

# All array tasks share one output dir (keyed by the array job id) so the
# combine job can gather every cell_*.json.
OUTDIR="${OUTDIR:-$ROOT_DIR/results/param_search_${SLURM_ARRAY_JOB_ID:-manual}}"

module load miniconda

eval "$(conda shell.bash hook)"

conda activate contact_modeling

echo "cell ${SLURM_ARRAY_TASK_ID}: model=$SEL_MODEL "\
     "w_quat=$SEL_W_QUAT w_pos_x=$SEL_W_POS_X w_pos_y=$SEL_W_POS_Y w_pos_z=$SEL_W_POS_Z"\
     "w_contact=$SEL_W_CONTACT w_joint=$SEL_W_JOINT"\
     "-> $OUTDIR"

# The first task queues the combine job to run once the whole array finishes
# ok, so a single `sbatch param_search.slurm` produces the merged results too.

python experiments/hpc/run_param_cell.py \
    --cell_id     "$SLURM_ARRAY_TASK_ID" \
    --outdir      "$OUTDIR" \
    --task        "$TASK" \
    --model       "$SEL_MODEL" \
    --n_episodes  "$N_EPISODES" \
    --weights     "w_quat=$SEL_W_QUAT" \
                  "w_pos_x=$SEL_W_POS_X" "w_pos_y=$SEL_W_POS_Y" "w_pos_z=$SEL_W_POS_Z" \
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
    --debug
