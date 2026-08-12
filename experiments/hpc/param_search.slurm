#!/bin/bash
#SBATCH --job-name=cont_modeling_param_search
#SBATCH --output=logs/cont_modeling_param_search%A_%a.out
#SBATCH --error=logs/cont_modeling_param_search%A_%a.err
#SBATCH --array=0-17         # must equal (product of every axis length in AXES below) - 1.
                               # Each task prints the grid total; a task id past the
                               # end exits with an error instead of silently rerunning.
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

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
N_EPISODES=5          # episodes per cell -> success rate

MODELS=("M2" "M3") #("M1" "M2" "M3" "M4")

# --- running cost weights ---------------------------------------------------
W_QUAT=("10.0")
# Position error is weighted separately per axis (X is down the fingers,
# Y across the fingers, Z vertical) -> three independent search axes.
W_POS_X=("15")
W_POS_Y=("15")
W_POS_Z=("3.75")
W_CONTACT=("10" "5.0" "2.5")
W_JOINT=("1.2" "0.6" "0.3")

# --- terminal cost weights --------------------------------------------------
# On the final horizon step these REPLACE the running cost (see
# grasp_reorient_cost_wp), so they trade off against each other, not against the
# running weights. w_pos_term uses the isotropic ||Δp||, not the per-axis terms.
W_QUAT_TERM=("900" "600" "300")
W_POS_TERM=("1500" "1000" "500")

# --- MPPI knobs that are part of the grid -----------------------------------
# Not cost weights, so they reach the worker through their own flags; listing
# them in MPPI_AXES below turns them into recorded search axes (-> CSV columns).
TEMPERATURE=("1.0")
NOISE_SIGMA=("0.025")

# Shared MPPI / eval knobs (single values, never swept), matching
# run_eval_episode.py's CLI defaults. The rollout schedule is specified in
# SECONDS and quantized against the task's rollout timestep; for grasp_reorient
# (rollout_dt = 4 ms) STEP_TIME is 16 substeps, and TIME_HORIZON's 5.5 control
# steps floors to 5 (0.320s realized).
N_SAMPLES=256
TIME_HORIZON=0.352              # planning horizon in seconds
STEP_TIME=0.064                 # control-step duration in seconds
EVAL_SIM="none" #"none"
SETTLE=1.0
GEOMETRY="accurate"
SEED=0

# ---------------------------------------------------------------------------
# 2. Map this task's ID to one value per axis (mixed-radix decode).
#    AXES names the arrays above, lowest-varying axis first; peel off the
#    lowest axis, then divide it out for the next. Add or drop an axis by
#    editing this list (and #SBATCH --array to match TOTAL).
# ---------------------------------------------------------------------------
AXES=(MODELS W_QUAT W_POS_X W_POS_Y W_POS_Z W_CONTACT W_JOINT
      W_QUAT_TERM W_POS_TERM TEMPERATURE NOISE_SIGMA)

# Non-weight axes: passed via their own flags AND named in --sweep so the
# per-cell JSON records them as grid axes and the analysis CSVs group by them.
MPPI_AXES=(TEMPERATURE NOISE_SIGMA)

TOTAL=1
for AX in "${AXES[@]}"; do
    declare -n ARR="$AX"
    TOTAL=$(( TOTAL * ${#ARR[@]} ))
done

T=${SLURM_ARRAY_TASK_ID:-0}
if (( T >= TOTAL )); then
    echo "cell $T is outside the grid (TOTAL=$TOTAL cells); set #SBATCH --array=0-$(( TOTAL - 1 ))" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Select the actual values for this cell: SEL[<AXIS_NAME>] = chosen value.
# ---------------------------------------------------------------------------
declare -A SEL
for AX in "${AXES[@]}"; do
    declare -n ARR="$AX"
    N=${#ARR[@]}
    SEL[$AX]=${ARR[$(( T % N ))]}
    T=$(( T / N ))
done

# --sweep flag + the lower-cased knob names for the non-weight axes. Keeping the
# flag itself in the array means an empty MPPI_AXES still expands cleanly.
MPPI_SWEEP=(--sweep)
for AX in "${MPPI_AXES[@]:-}"; do
    [[ -n "$AX" ]] && MPPI_SWEEP+=( "${AX,,}" )
done

# All array tasks share one output dir (keyed by the array job id) so the
# combine job can gather every cell_*.json.
OUTDIR="${OUTDIR:-$ROOT_DIR/results/param_search_${SLURM_ARRAY_JOB_ID:-manual}}"

module load miniconda

eval "$(conda shell.bash hook)"

conda activate contact_modeling

CELL_DESC=""
for AX in "${AXES[@]}"; do
    CELL_DESC+=" ${AX,,}=${SEL[$AX]}"
done
echo "cell ${SLURM_ARRAY_TASK_ID}/$(( TOTAL - 1 )):${CELL_DESC} -> $OUTDIR"

# The first task queues the combine job to run once the whole array finishes
# ok, so a single `sbatch param_search.slurm` produces the merged results too.

python experiments/hpc/run_param_cell.py \
    --cell_id     "$SLURM_ARRAY_TASK_ID" \
    --outdir      "$OUTDIR" \
    --task        "$TASK" \
    --model       "${SEL[MODELS]}" \
    --n_episodes  "$N_EPISODES" \
    --weights     "w_quat=${SEL[W_QUAT]}" \
                  "w_pos_x=${SEL[W_POS_X]}" "w_pos_y=${SEL[W_POS_Y]}" "w_pos_z=${SEL[W_POS_Z]}" \
                  "w_contact=${SEL[W_CONTACT]}" "w_joint=${SEL[W_JOINT]}" \
                  "w_quat_term=${SEL[W_QUAT_TERM]}" "w_pos_term=${SEL[W_POS_TERM]}" \
    "${MPPI_SWEEP[@]}" \
    --n_samples   "$N_SAMPLES" \
    --time_horizon "$TIME_HORIZON" \
    --temperature "${SEL[TEMPERATURE]}" \
    --noise_sigma "${SEL[NOISE_SIGMA]}" \
    --step_time   "$STEP_TIME" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --geometry    "$GEOMETRY" \
    --seed        "$SEED" \
    --debug
