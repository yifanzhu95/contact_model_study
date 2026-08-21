#!/bin/bash
#SBATCH --job-name=cont_modeling_eval
#SBATCH --output=logs/cont_modeling_eval_%j.out
#SBATCH --error=logs/cont_modeling_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00

set -euo pipefail

# Hardcoded project root so videos always land in <root>/videos regardless of
# where `sbatch` was invoked from.
PROJ_DIR="/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study"
ROOT_DIR="$PROJ_DIR"
cd "$PROJ_DIR" #cd "$PROJ_DIR/experiments/hpc"

# ---------------------------------------------------------------------------
# 1. Single run: one model + one weight set, N_EPISODES episodes, rendered.
#    Unlike param_search.slurm there is no grid — just edit the values below.
# ---------------------------------------------------------------------------
TASK="grasp_reorient"
MODEL="M3"
PLANNER="mppi"         # mppi | cem | predictive_sampler
N_EPISODES=10          # episodes to run for this one weight set

# Shared planner / eval knobs, matching run_eval_episode.py's CLI defaults. The
# rollout schedule is specified in SECONDS and quantized against the task's
# rollout timestep; for grasp_reorient (rollout_dt = 4 ms) STEP_TIME is 16
# substeps, and TIME_HORIZON's 5.5 control steps floors to 5 (0.320s realized).
N_SAMPLES=256
TIME_HORIZON=0.352              # planning horizon in seconds
STEP_TIME=0.064                 # control-step duration in seconds
NOISE_SIGMA=0.25
EVAL_SIM="none"
SETTLE=1.0
SEED=0

# Planner-specific knobs, appended below by the case block.
TEMPERATURE=20.0                 # mppi: softmax temperature
ELITE_FRAC=0.1                  # cem:  elite fraction of N_SAMPLES
CEM_ALPHA=0.1                   # cem:  refit smoothing

case "$PLANNER" in
  mppi)
    PLANNER_ARGS=(--temperature "$TEMPERATURE")
    ;;
  cem)
    PLANNER_ARGS=(--elite_frac "$ELITE_FRAC" --cem_alpha "$CEM_ALPHA")
    ;;
  predictive_sampler|ps)
    # Greedy best-of-N; n_samples and noise_sigma are the only knobs it needs.
    PLANNER_ARGS=(--include_nominal)
    ;;
  *)
    echo "unknown PLANNER=$PLANNER" >&2
    exit 1
    ;;
esac

# Videos go to <root>/videos. run_eval_episode renders one per episode, naming
# them <stem>_ep000.gif, <stem>_ep001.gif, ... when N_EPISODES > 1.
VIDEO_DIR="/home/kg878/scratch_pi_amd78/kg878/contact_modeling_videos" # "$ROOT_DIR/videos"
mkdir -p "$VIDEO_DIR"
VIDEO_PATH="$VIDEO_DIR/${TASK}_${MODEL}_${PLANNER}_eval.gif"

# EGL for headless (offscreen) rendering on the GPU node.
#export MUJOCO_GL="egl"

module load miniconda
conda activate contact_modeling

echo "eval: task=$TASK model=$MODEL planner=$PLANNER n_episodes=$N_EPISODES "\
     "-> videos in $VIDEO_DIR"

python -u -m contact_study.drivers.run_eval_episode \
    --task        "$TASK" \
    --model       "$MODEL" \
    --planner     "$PLANNER" \
    --n_episodes  "$N_EPISODES" \
    --n_samples   "$N_SAMPLES" \
    --time_horizon "$TIME_HORIZON" \
    --noise_sigma "$NOISE_SIGMA" \
    --step_time   "$STEP_TIME" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --seed        "$SEED" \
    --video       "$VIDEO_PATH" \
    "${PLANNER_ARGS[@]}" \
    --debug
