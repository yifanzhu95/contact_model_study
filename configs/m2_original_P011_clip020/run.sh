#!/usr/bin/env bash
set -euo pipefail
cd /home/chetan/contact_model_study
exec /home/chetan/miniconda3/envs/mj_env/bin/python /home/chetan/contact_model_study/launch_leap_experiment.py --config /home/chetan/contact_model_study/configs/m2_original_P011_clip020/experiment.json --n-episodes 12 --log-dir results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/original_P011_shift512_clip020 --restart-tracking "$@"
