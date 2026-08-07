#!/bin/bash
#SBATCH --job-name=cont_modeling_param_search
#SBATCH --output=logs/cont_modeling_param_search%A_%a.out
#SBATCH --error=logs/cont_modeling_param_search%A_%a.err
#SBATCH --array=0-17         # must equal (#MODELS * #W_QUAT * #W_POS_X * #W_POS_Y * #W_POS_Z * #W_CONTACT * #W_JOINT) - 1
                               # default grid: 4 * 3 * 3 * 3 * 3 * 3 * 3 = 2916  ->  0-2915
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

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
W_QUAT=("10.0")
# Position error is weighted separately per axis (X is down the fingers,
# Y across the fingers, Z vertical) -> three independent search axes.
W_POS_X=("3.75")
W_POS_Y=("15")
W_POS_Z=("3.75")
W_CONTACT=("10" "5.0" "2.5")
W_JOINT=("1.2" "0.6" "0.3")

# Shared MPPI / eval knobs (single values, not swept), matching run_eval_episode.py's
# CLI defaults. The rollout schedule is specified in SECONDS and quantized
# against the task's rollout timestep; for grasp_reorient (rollout_dt = 4 ms)
# STEP_TIME is 16 substeps, and TIME_HORIZON's 5.5 control steps floors to 5
# (0.320s realized).
N_SAMPLES=256
TIME_HORIZON=0.352              # planning horizon in seconds
STEP_TIME=0.064                 # control-step duration in seconds
TEMPERATURE=1.0
NOISE_SIGMA=0.025
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
    --time_horizon "$TIME_HORIZON" \
    --temperature "$TEMPERATURE" \
    --noise_sigma "$NOISE_SIGMA" \
    --step_time   "$STEP_TIME" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --geometry    "$GEOMETRY" \
    --seed        "$SEED" \
    --debug
