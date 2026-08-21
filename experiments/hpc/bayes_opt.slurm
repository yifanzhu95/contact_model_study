#!/bin/bash
#SBATCH --job-name=cont_modeling_bayes_opt
#SBATCH --output=logs/cont_modeling_bayes_opt_%A_%a.out
#SBATCH --error=logs/cont_modeling_bayes_opt_%A_%a.err
#SBATCH --array=0-15%8       # (#OBJECTS * #MODELS) - 1; the %N suffix caps how many
                             # cells hold a GPU at once. Each task prints the grid
                             # total, and an id past the end errors out instead of
                             # silently rerunning cell 0.
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00      # one cell = N_CALLS trials x N_EPISODES episodes,
                             # each episode ~2 min for grasp_reorient. Budget
                             # accordingly; --resume (below) picks a killed cell up.

set -euo pipefail

# Hardcoded project root so results always land in <root>/results regardless
# of where `sbatch` was invoked from.
PROJ_DIR="/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study"
ROOT_DIR="$PROJ_DIR"
cd "$PROJ_DIR"

# ===========================================================================
# 1. THE GRID: one independent Bayesian optimization per (object, contact model)
#    cell. Unlike param_search.slurm these axes are not a search space -- each
#    cell runs its own gp_minimize over the cost weights and planner knobs, so
#    every object/model pair ends up with its own tuned weight set.
# ===========================================================================
TASK="grasp_reorient"

# Objects: the keys of _OBJ_PARAMS in contact_study/tasks/grasp_reorient.py.
# Each needs scenes/leap/env_leap_rollout_{obj}_{HAND_ACC}_{OBJ_ACC}.xml (and
# env_leap_eval_{obj}.xml when EVAL_SIM is not "none").
OBJECTS=("cube" "duck" "spam" "tomato")

# Contact models: the keys of MODEL_FACTORIES in run_eval_episode.py.
MODELS=("M1" "M2" "M3" "M4")

# Collision fidelity halves of the scene variant. The full --geometry string is
# assembled below as "<obj>_<HAND_ACC>_<OBJ_ACC>".
HAND_ACC="low"
OBJ_ACC="high"

# ---------------------------------------------------------------------------
# 2. Episode settings (mirror run_eval_episode.py's flags 1:1, so a winning
#    trial replays by pasting them into that driver).
# ---------------------------------------------------------------------------
PLANNER="mppi"                  # mppi | cem | predictive_sampler
N_EPISODES=2                    # episodes per trial; more averages out episode
                                # luck at a proportional cost in wall time
SEED=42                         # THE constant episode seed (same episodes every trial)
N_SAMPLES=256
TIME_HORIZON=0.352              # planning horizon in SECONDS (quantized to control steps)
STEP_TIME=0.064                 # control-step duration in SECONDS (quantized to rollout steps)
EVAL_SIM="none"                 # none | mujoco | drake | pinocchio
SETTLE=1.0
NCONMAX=50
NJMAX=300
RESAMPLE_INTERVAL=1             # plan steps between noise resamples (1 = every step)
WARM_START=0                    # 1 -> --warm_start, 0 -> --no-warm_start
USE_FULL_GRAPH=1                # 1 -> --use_full_graph (one mega CUDA graph per plan())
DEBUG=1                         # 1 -> --debug (verbose per-step diagnostics)

# Optional episode knobs -- leave EMPTY to keep the driver's own default
# (which for these means "the planner's / the task's default", not a number).
HORIZON=""                      # planning horizon in control STEPS (ignored w/ TIME_HORIZON)
SUBSTEPS=""                     # rollout substeps per control step
EVAL_SUBSTEPS=""                # eval steps per rollout step
N_ITERATIONS=""                 # optimizer iterations per plan()
DELTA=""                        # per-step delta clip magnitude; empty disables the clamp

# ---------------------------------------------------------------------------
# 3. Search space. WEIGHTS entries are 'name' (bracketed multiplicatively at
#    x4 around THIS object's default weight, so the space auto-adapts per
#    object) or 'name:lo:hi' for an explicit bracket. Leave the array empty to
#    accept the driver's DEFAULT_OPT_WEIGHTS.
# ---------------------------------------------------------------------------
OPT_WEIGHTS=("w_quat" "w_pos_x" "w_pos_y" "w_pos_z" "w_contact" "w_joint")

OPT_NOISE_SIGMA=1               # 1 -> search noise_sigma, 0 -> pin it to NOISE_SIGMA
OPT_TEMPERATURE=1               # 1 -> search temperature, 0 -> pin it to TEMPERATURE
NOISE_SIGMA_RANGE=(0.001 1.0)   # LO HI, log-uniform
TEMPERATURE_RANGE=(0.01 50.0)   # LO HI, log-uniform
NOISE_SIGMA=0.2                 # fixed value when OPT_NOISE_SIGMA=0
TEMPERATURE=30.0                # fixed value when OPT_TEMPERATURE=0

# ---------------------------------------------------------------------------
# 4. Objective:  J = -(W_SUCCESS * success_rate) + W_COST * norm_goal_err
#    Both terms live in [0, 1], so the two weights are directly comparable.
# ---------------------------------------------------------------------------
W_SUCCESS=1.0
W_COST=0.1
ERR_CLIP=10.0                   # goal error clipped here, then divided by it

# ---------------------------------------------------------------------------
# 5. Optimizer (gp_minimize).
# ---------------------------------------------------------------------------
N_CALLS=60                      # total trial budget per cell, INCLUDING resumed trials
N_INITIAL_POINTS=10             # random trials before the GP takes over
ACQ_FUNC="gp_hedge"             # EI | LCB | PI | gp_hedge
GP_NOISE="gaussian"             # "gaussian" or a float
BO_SEED=0                       # random_state for gp_minimize itself (not SEED)

# 1 -> if this cell's outdir already holds a bo_state.json, seed the optimizer
# with those trials instead of starting over. Combined with a fixed OUTDIR
# (below) this makes a requeued/extended cell continue where it stopped.
RESUME=1

# All cells of one submission share a parent dir keyed by the array job id;
# export OUTDIR_ROOT to a FIXED path when you intend to --resume a previous
# submission (the array job id changes on every sbatch).
OUTDIR_ROOT="${OUTDIR_ROOT:-$ROOT_DIR/results/bayes_opt_${SLURM_ARRAY_JOB_ID:-manual}}"

# ===========================================================================
# 6. Map this task's ID to one value per axis (mixed-radix decode, lowest-
#    varying axis first -- consecutive task ids walk the objects).
# ===========================================================================
AXES=(OBJECTS MODELS)

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

declare -A SEL
for AX in "${AXES[@]}"; do
    declare -n ARR="$AX"
    N=${#ARR[@]}
    SEL[$AX]=${ARR[$(( T % N ))]}
    T=$(( T / N ))
done

OBJ="${SEL[OBJECTS]}"
MODEL="${SEL[MODELS]}"
GEOMETRY="${OBJ}_${HAND_ACC}_${OBJ_ACC}"

# One directory per cell, named by what it contains rather than by task id, so
# the winners are readable without cross-referencing the array indices.
OUTDIR="$OUTDIR_ROOT/${TASK}_${OBJ}_${MODEL}_${PLANNER}"
mkdir -p "$OUTDIR"

# ===========================================================================
# 7. Assemble the optional / boolean flags.
# ===========================================================================
EXTRA_ARGS=()
if [[ -n "$HORIZON"       ]]; then EXTRA_ARGS+=(--horizon       "$HORIZON");       fi
if [[ -n "$SUBSTEPS"      ]]; then EXTRA_ARGS+=(--substeps      "$SUBSTEPS");      fi
if [[ -n "$EVAL_SUBSTEPS" ]]; then EXTRA_ARGS+=(--eval_substeps "$EVAL_SUBSTEPS"); fi
if [[ -n "$N_ITERATIONS"  ]]; then EXTRA_ARGS+=(--n_iterations  "$N_ITERATIONS");  fi
if [[ -n "$DELTA"         ]]; then EXTRA_ARGS+=(--delta         "$DELTA");         fi

if (( WARM_START ));      then EXTRA_ARGS+=(--warm_start);       else EXTRA_ARGS+=(--no-warm_start);       fi
if (( USE_FULL_GRAPH ));  then EXTRA_ARGS+=(--use_full_graph);   else EXTRA_ARGS+=(--no-use_full_graph);   fi
if (( OPT_NOISE_SIGMA )); then EXTRA_ARGS+=(--opt_noise_sigma);  else EXTRA_ARGS+=(--no-opt_noise_sigma);  fi
if (( OPT_TEMPERATURE )); then EXTRA_ARGS+=(--opt_temperature);  else EXTRA_ARGS+=(--no-opt_temperature);  fi
if (( DEBUG ));           then EXTRA_ARGS+=(--debug);            fi

# Keeping the flag inside the array means an empty OPT_WEIGHTS expands to
# nothing at all, which is how the driver's own default subset is requested.
WEIGHT_ARGS=()
if (( ${#OPT_WEIGHTS[@]} )); then WEIGHT_ARGS=(--opt_weights "${OPT_WEIGHTS[@]}"); fi

# A checkpoint from a previous run of THIS cell. gp_minimize rebases --n_calls
# and --n_initial_points against the restored trials, so N_CALLS keeps meaning
# "total trials for this cell".
STATE="$OUTDIR/bo_state.json"
if (( RESUME )) && [[ -f "$STATE" ]]; then
    EXTRA_ARGS+=(--resume "$STATE")
    echo "resuming from $STATE"
fi

# ===========================================================================
# 8. Run.
# ===========================================================================
module load miniconda

eval "$(conda shell.bash hook)"

conda activate contact_modeling

echo "cell ${SLURM_ARRAY_TASK_ID:-0}/$(( TOTAL - 1 )): task=$TASK obj=$OBJ model=$MODEL"\
     "planner=$PLANNER geometry=$GEOMETRY n_calls=$N_CALLS n_episodes=$N_EPISODES -> $OUTDIR"

python -u -m contact_study.drivers.run_bayes_opt \
    --task              "$TASK" \
    --geometry          "$GEOMETRY" \
    --model             "$MODEL" \
    --planner           "$PLANNER" \
    --n_episodes        "$N_EPISODES" \
    --seed              "$SEED" \
    --n_samples         "$N_SAMPLES" \
    --time_horizon      "$TIME_HORIZON" \
    --step_time         "$STEP_TIME" \
    --resample_interval "$RESAMPLE_INTERVAL" \
    --nconmax           "$NCONMAX" \
    --njmax             "$NJMAX" \
    --eval_sim          "$EVAL_SIM" \
    --settle            "$SETTLE" \
    "${WEIGHT_ARGS[@]}" \
    --noise_sigma_range "${NOISE_SIGMA_RANGE[@]}" \
    --temperature_range "${TEMPERATURE_RANGE[@]}" \
    --noise_sigma       "$NOISE_SIGMA" \
    --temperature       "$TEMPERATURE" \
    --w_success         "$W_SUCCESS" \
    --w_cost            "$W_COST" \
    --err_clip          "$ERR_CLIP" \
    --n_calls           "$N_CALLS" \
    --n_initial_points  "$N_INITIAL_POINTS" \
    --acq_func          "$ACQ_FUNC" \
    --gp_noise          "$GP_NOISE" \
    --bo_seed           "$BO_SEED" \
    --outdir            "$OUTDIR" \
    "${EXTRA_ARGS[@]}"
