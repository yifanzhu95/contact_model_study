#!/bin/bash
#SBATCH --job-name=cont_modeling_eval
#SBATCH --output=logs/cont_modeling_eval_%j.out
#SBATCH --error=logs/cont_modeling_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

set -euo pipefail

# Resolve paths from the script's own location, not the submission directory
# (SLURM_SUBMIT_DIR is wherever `sbatch` was invoked from, which may not be
# experiments/hpc), so videos always land in <root>/videos regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 1. Single run: one model + one weight set, N_EPISODES episodes, rendered.
#    Unlike param_search.slurm there is no grid — just edit the values below.
# ---------------------------------------------------------------------------
TASK="grasp_reorient"
MODEL="M2"
N_EPISODES=2          # episodes to run for this one weight set

# Shared MPPI / eval knobs.
N_SAMPLES=256
HORIZON=48
TEMPERATURE=10.0
NOISE_SIGMA=0.01
DELTA=0.1
SUBSTEPS=16
EVAL_SIM="none"
SETTLE=1.0
SEED=0

# Videos go to <root>/videos. run_eval_episode renders one per episode, naming
# them <stem>_ep000.gif, <stem>_ep001.gif, ... when N_EPISODES > 1.
VIDEO_DIR="$ROOT_DIR/videos"
mkdir -p "$VIDEO_DIR"
VIDEO_PATH="$VIDEO_DIR/${TASK}_${MODEL}_eval.gif"

# EGL for headless (offscreen) rendering on the GPU node.
#export MUJOCO_GL="egl"

module load miniconda
conda activate contact_modeling

echo "eval: task=$TASK model=$MODEL n_episodes=$N_EPISODES "\
     "-> videos in $VIDEO_DIR"

python -u -m contact_study.drivers.run_eval_episode \
    --task        "$TASK" \
    --model       "$MODEL" \
    --n_episodes  "$N_EPISODES" \
    --n_samples   "$N_SAMPLES" \
    --horizon     "$HORIZON" \
    --temperature "$TEMPERATURE" \
    --noise_sigma "$NOISE_SIGMA" \
    --delta       "$DELTA" \
    --substeps    "$SUBSTEPS" \
    --eval_sim    "$EVAL_SIM" \
    --settle      "$SETTLE" \
    --seed        "$SEED" \
    --video       "$VIDEO_PATH" \
    --debug
