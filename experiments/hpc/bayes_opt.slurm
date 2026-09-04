#!/bin/bash
#SBATCH --job-name=cont_modeling_bayes_opt
#SBATCH --output=logs/cont_modeling_bayes_opt_%A_%a.out
#SBATCH --error=logs/cont_modeling_bayes_opt_%A_%a.err
#SBATCH --array=0-2       # (#OBJECTS * #MODELS) - 1; the %N suffix caps how many
                             # cells hold a GPU at once. Each task prints the grid
                             # total, and an id past the end errors out instead of
                             # silently rerunning cell 0.
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=20    # >= N_WORKERS below: each worker drives one eval
                             # sim on its own core.
#SBATCH --mem=32G            # 4 SPAWNED interpreters, each re-importing warp +
                             # mujoco + pinocchio/coal and carrying a CUDA
                             # context's host allocations. 16G OOM-kills a worker.
#SBATCH --time=12:00:00      # one cell = N_CALLS trials x N_EPISODES episodes.
                             # Episodes within a trial now run CONCURRENTLY (see
                             # N_WORKERS), so a trial costs about as long as its
                             # SLOWEST episode rather than their sum. Budget
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
# OBJECTS=("cube" "duck" "spam" "tomato")
OBJECTS=("cube" "duck" "ball")

# Goal difficulty PER OBJECT: which goal sampler this cell's episodes draw from
# (grasp_reorient levels: 0 fixed 90 deg spin, 1 +/-90 deg spin, 2 +/-90 deg
# about a random axis, 3 adjacent face + twist, 4 any other face + twist,
# 5 180 deg flip, 6 adjacent face no twist, 7 roll to "O", 8 roll either way,
# 9 roll to "B").
#
# The object is the grid axis, so one cell = one object = one level; the driver
# takes a single --goal_difficulty and applies it to every trial and every
# model, which is what keeps the per-model objectives MODEL_AGG folds
# like-for-like. This map is the whole per-object story.
#
# It exists because the objects are not equally hard at a given level: a level
# the cube reorients reliably can be one the duck never completes, and a BO cell
# whose every trial fails scores a flat objective and teaches the GP nothing.
# Drop that object to a level it can sometimes finish and the search gets a
# gradient to follow again.
#
# Leave an object's entry EMPTY (or drop the key) to fall back to
# GOAL_DIFFICULTY_ALL; leave that empty too for the task's own default. Keys
# must be OBJECTS entries -- an unmatched key is silently unused, since only
# this cell's object is ever read.
declare -A GOAL_DIFFICULTY=(
    [cube]="8"
    [duck]="8"
    [ball]="1"
)
GOAL_DIFFICULTY_ALL=""          # level for any object with no entry above;
                                # empty -> the task's own default

# Contact models: the keys of MODEL_FACTORIES in run_eval_episode.py.
MODELS=("M1" "M2" "M3" "M4")

# 0 -> MODELS is a GRID AXIS: one independent BO per (object, model), each
#      tuning weights for that contact model alone. This is the original layout.
# 1 -> MODELS is ONE SEARCH: a single BO per object scores every trial's weight
#      vector on ALL of MODELS at the same episode seeds and folds them with
#      MODEL_AGG, yielding one weight set that has to hold across the models.
#      The grid collapses to the objects, so set --array=0-$((#OBJECTS - 1)),
#      and each trial costs #MODELS x N_EPISODES episodes instead of N_EPISODES.
#      BUDGET FOR IT: with the defaults that is 16 episodes/trial against 4
#      --cpus-per-task, i.e. 4 sequential batches per trial where the split grid
#      needed 1. Either raise --cpus-per-task (and --mem, and N_WORKERS with
#      them) toward #MODELS x N_EPISODES, or roughly quadruple --time. The
#      trade is real: 2 joint cells replace 8 split ones, so the TOTAL GPU-hours
#      are similar — it is the per-cell wall time that grows.
JOINT_MODELS=1
MODEL_AGG="mean"                # mean | worst. 'worst' (minimax) tunes the
                                # weakest model; 'mean' lets a strong model
                                # carry a weak one. JOINT_MODELS=1 only.

# Collision fidelity halves of the scene variant. The full --geometry string is
# assembled below as "<obj>_<HAND_ACC>_<OBJ_ACC>".
HAND_ACC="high"
OBJ_ACC="high"

# ---------------------------------------------------------------------------
# 2. Episode settings (mirror run_eval_episode.py's flags 1:1, so a winning
#    trial replays by pasting them into that driver).
# ---------------------------------------------------------------------------
PLANNER="mppi"                  # mppi | cem | predictive_sampler
N_EPISODES=2                    # episodes per trial; more averages out episode
                                # luck at a proportional cost in wall time
N_WORKERS="16"                    # episodes to run CONCURRENTLY, one process each
                                # (own planner + own eval sim). Empty = one per
                                # episode in the trial (#CELL_MODELS x N_EPISODES),
                                # capped by --cpus-per-task. 1 = serial, the old
                                # behavior. THIS IS THE VRAM KNOB: each worker
                                # holds its own CUDA context + MJWarp Data at
                                # N_SAMPLES worlds. Keep <= --cpus-per-task and
                                # raise --mem with it. The objective is unchanged
                                # either way; this is a pure throughput knob.
                                # Worth ~N_WORKERS-fold with EVAL_SIM below
                                # resolving to pinocchio (CPU-bound, GPU idle 98%
                                # of the time) and ~1.5x on mujoco (GPU-bound).
SEED=64                         # THE constant episode seed (same episodes every trial)
N_SAMPLES=256
TIME_HORIZON=0.352              # planning horizon in SECONDS (quantized to control steps)
STEP_TIME=0.064                 # control-step duration in SECONDS (quantized to rollout steps)
EVAL_SIM="mujoco"                 # none | mujoco | drake | pinocchio
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
OPT_WEIGHTS=("w_quat:1.0:50.0" "w_pos_x:1.0:50.0" "w_pos_y:1.0:50.0" "w_pos_z:1.0:50.0" "w_contact:0.1:100.0" "w_joint:0.01:20.0" "w_fallen:100.0:300.0" "w_quat_term:100.0:300.0" "w_pos_term:100.0:300.0")

OPT_NOISE_SIGMA=0               # 1 -> search noise_sigma, 0 -> pin it to NOISE_SIGMA
OPT_TEMPERATURE=1               # 1 -> search temperature, 0 -> pin it to TEMPERATURE

# JOINT_MODELS=1 only. 0 -> the models share ONE searched temperature.
# 1 -> each model gets its OWN temperature dimension over the same
# TEMPERATURE_RANGE, so a model whose cost scale differs can run a different
# MPPI softmax sharpness while the COST WEIGHTS stay shared -- the search is
# still looking for one weight set that holds across the models. Costs
# (#MODELS - 1) extra search dimensions and NO extra episodes, so wall time per
# trial is unchanged; give the GP more N_CALLS for the bigger space. Needs
# OPT_TEMPERATURE=1. NOTE: this renames the search dimensions (temperature ->
# temperature_M1, ...), so flipping it makes any existing bo_state.json
# unresumable -- start such a cell in a fresh directory.
PER_MODEL_TEMPERATURE=0
NOISE_SIGMA_RANGE=(0.001 1.0)   # LO HI, log-uniform
TEMPERATURE_RANGE=(0.001 25.0)   # LO HI, log-uniform
NOISE_SIGMA=0.1                 # fixed value when OPT_NOISE_SIGMA=0
TEMPERATURE=30.0                # fixed value when OPT_TEMPERATURE=0

# ---------------------------------------------------------------------------
# 4. Objective:  J = -(W_SUCCESS * success_rate) + W_COST * norm_goal_err
#    Both terms live in [0, 1], so the two weights are directly comparable.
# ---------------------------------------------------------------------------
W_SUCCESS=1.0
W_COST=0.75
ERR_CLIP=250.0                   # goal error clipped here, then divided by it

# ---------------------------------------------------------------------------
# 5. Optimizer (gp_minimize).
# ---------------------------------------------------------------------------
N_CALLS=750                      # total trial budget per cell, INCLUDING resumed trials
N_INITIAL_POINTS=20             # random trials before the GP takes over
ACQ_FUNC="gp_hedge"             # EI | LCB | PI | gp_hedge
GP_NOISE="gaussian"             # "gaussian" or a float
BO_SEED=0                       # random_state for gp_minimize itself (not SEED)

# 1 -> if this cell's outdir already holds a bo_state.json, seed the optimizer
# with those trials instead of starting over. Combined with a fixed OUTDIR
# (below) this makes a requeued/extended cell continue where it stopped.
RESUME=1

# All cells of one submission share a parent dir keyed by the array job id;
# export OUTDIR_ROOT to a FIXED path when you intend to --resume a previous
# submission WHOLESALE (the array job id changes on every sbatch), e.g.
#
#   sbatch --export=ALL,OUTDIR_ROOT=$ROOT_DIR/results/bayes_opt_1234567 \
#          --array=0-1 experiments/hpc/bayes_opt.slurm
#
OUTDIR_ROOT="${OUTDIR_ROOT:-$ROOT_DIR/results/bayes_opt_${SLURM_ARRAY_JOB_ID:-manual}}"

# PER-CELL resume dirs, for when the cells you are picking up do NOT all live
# under one OUTDIR_ROOT (different submissions, a renamed dir, one object that
# needs more trials than the rest). A hit here REPLACES $OUTDIR for that cell,
# so the run continues writing cell_*.json and bo_state.json into the existing
# directory. Keys are tried in this order, first hit wins:
#
#   1. the array task id                      [0]="/abs/path"
#   2. the cell name, <task>_<obj>_<model>_<planner>
#                                             [grasp_reorient_duck_M1+M2+M3+M4_mppi]="/abs/path"
#   3. the object name                        [duck]="/abs/path"
#
# Leave it empty for the normal "everything under one root" case. Note that the
# array ids that get a directory here are the ONLY thing tying an index to a
# run -- the grid decode below still determines which object/model that index
# actually optimizes, so the path you paste must belong to the same cell.
declare -A RESUME_DIRS=(
    [cube]=""
    [ball]=""
    [duck]=""
)

# ===========================================================================
# 6. Map this task's ID to one value per axis (mixed-radix decode, lowest-
#    varying axis first -- consecutive task ids walk the objects).
# ===========================================================================
if (( JOINT_MODELS )); then
    AXES=(OBJECTS)              # every model runs inside each cell
else
    AXES=(OBJECTS MODELS)       # one cell per (object, model)
fi

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
GEOMETRY="${OBJ}_${HAND_ACC}_${OBJ_ACC}"

if (( JOINT_MODELS )); then
    CELL_MODELS=("${MODELS[@]}")
    MODEL=$(IFS=+; echo "${MODELS[*]}")     # "M1+M2+M3+M4", for the dir name
else
    CELL_MODELS=("${SEL[MODELS]}")
    MODEL="${SEL[MODELS]}"
fi

# One directory per cell, named by what it contains rather than by task id, so
# the winners are readable without cross-referencing the array indices.
CELL_NAME="${TASK}_${OBJ}_${MODEL}_${PLANNER}"
OUTDIR="$OUTDIR_ROOT/$CELL_NAME"

# ...unless this cell was pinned to an existing directory above.
for KEY in "${SLURM_ARRAY_TASK_ID:-0}" "$CELL_NAME" "$OBJ"; do
    if [[ -n "${RESUME_DIRS[$KEY]:-}" ]]; then
        OUTDIR="${RESUME_DIRS[$KEY]}"
        echo "RESUME_DIRS[$KEY] -> $OUTDIR"
        break
    fi
done
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
if [[ -n "$N_WORKERS"     ]]; then EXTRA_ARGS+=(--n_workers     "$N_WORKERS");     fi
if (( JOINT_MODELS ));      then EXTRA_ARGS+=(--model_agg     "$MODEL_AGG");      fi

if (( WARM_START ));      then EXTRA_ARGS+=(--warm_start);       else EXTRA_ARGS+=(--no-warm_start);       fi
if (( USE_FULL_GRAPH ));  then EXTRA_ARGS+=(--use_full_graph);   else EXTRA_ARGS+=(--no-use_full_graph);   fi
if (( OPT_NOISE_SIGMA )); then EXTRA_ARGS+=(--opt_noise_sigma);  else EXTRA_ARGS+=(--no-opt_noise_sigma);  fi
if (( OPT_TEMPERATURE )); then EXTRA_ARGS+=(--opt_temperature);  else EXTRA_ARGS+=(--no-opt_temperature);  fi

# Only meaningful for a joint-models cell with a searched temperature; the
# driver rejects --per_model_temperature alongside --no-opt_temperature.
if (( PER_MODEL_TEMPERATURE && JOINT_MODELS && OPT_TEMPERATURE )); then
    EXTRA_ARGS+=(--per_model_temperature)
else
    EXTRA_ARGS+=(--no-per_model_temperature)
fi
if (( DEBUG ));           then EXTRA_ARGS+=(--debug);            fi

# This cell's goal difficulty: its object's entry, else GOAL_DIFFICULTY_ALL,
# else no flag at all -- which is how the task's own default is requested.
CELL_DIFFICULTY="${GOAL_DIFFICULTY[$OBJ]:-}"
CELL_DIFFICULTY="${CELL_DIFFICULTY:-$GOAL_DIFFICULTY_ALL}"
if [[ -n "$CELL_DIFFICULTY" ]]; then
    EXTRA_ARGS+=(--goal_difficulty "$CELL_DIFFICULTY")
fi

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
     "planner=$PLANNER geometry=$GEOMETRY n_calls=$N_CALLS n_episodes=$N_EPISODES"\
     "episodes/trial=$(( ${#CELL_MODELS[@]} * N_EPISODES ))"\
     "goal_difficulty=${CELL_DIFFICULTY:-task-default} -> $OUTDIR"

# Per-control-step trajectory + planner-distribution recording (see
# contact_study/evaluation/trajectory.py). OFF for sweeps: at grasp_reorient's
# max_steps it is megabytes per episode, which this many cells cannot carry.
# Drop the --no- prefixes (or raise --planner_dist_every) to record a cell.
RECORD_ARGS=(--no-record_trajectory --no-record_planner_dist)

python -u -m contact_study.drivers.run_bayes_opt \
    "${RECORD_ARGS[@]}" \
    --task              "$TASK" \
    --geometry          "$GEOMETRY" \
    --models            "${CELL_MODELS[@]}" \
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
