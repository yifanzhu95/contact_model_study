Prepared Duck/Ball friction BO. No search has been started.

Frozen Cube code; original uncalibrated high/high hands; M1-M4, worst-model objective.
Object-specific initial poses, targets, COM handling, thresholds and contact-model settings retained.
Mass (kg): {'duck': 0.05, 'ball': 0.14}. Sliding friction object/tips/palm/fingers: .3/.5/.4/.4.
Each cell: 750 trials; 120 initial points (1 object-default seed + 119 random);
4 fixed episodes per model; 16 episodes per trial; 1000 steps maximum.
256 samples; sigma=.05; 1 iteration; warm-start=none; dt=.004; control dt=.064.
Horizon matches Cube exactly: requested .352, resolves to 5 actions / .32 seconds.
Separate shared M1-M4 weight set at every fixed temperature.
The two Cube masses are NOT assigned to these objects; masses above come from their scenes.
No GPU allocations or MPS services occur during preparation or --check.
The 20-planner Cube memory test is not a Duck/Ball memory benchmark.

CPU validation only:
CUDA_VISIBLE_DEVICES='' /home/chetan/miniconda3/envs/mj_env/bin/python /home/chetan/contact_model_study/results/isolated_bo/duck_ball_friction_20261116/run_search.py --check

Future starts (only after requested; one selected cell per invocation):
cell 0: duck_mass50g_T0.1
  /home/chetan/miniconda3/envs/mj_env/bin/python /home/chetan/contact_model_study/results/isolated_bo/duck_ball_friction_20261116/run_search.py --cell 0
cell 1: ball_mass140g_T0.1
  /home/chetan/miniconda3/envs/mj_env/bin/python /home/chetan/contact_model_study/results/isolated_bo/duck_ball_friction_20261116/run_search.py --cell 1
cell 2: duck_mass50g_T0.05
  /home/chetan/miniconda3/envs/mj_env/bin/python /home/chetan/contact_model_study/results/isolated_bo/duck_ball_friction_20261116/run_search.py --cell 2
cell 3: ball_mass140g_T0.05
  /home/chetan/miniconda3/envs/mj_env/bin/python /home/chetan/contact_model_study/results/isolated_bo/duck_ball_friction_20261116/run_search.py --cell 3
cell 4: duck_mass50g_T0.075
  /home/chetan/miniconda3/envs/mj_env/bin/python /home/chetan/contact_model_study/results/isolated_bo/duck_ball_friction_20261116/run_search.py --cell 4
cell 5: ball_mass140g_T0.075
  /home/chetan/miniconda3/envs/mj_env/bin/python /home/chetan/contact_model_study/results/isolated_bo/duck_ball_friction_20261116/run_search.py --cell 5
cell 6: duck_mass50g_T0.15
  /home/chetan/miniconda3/envs/mj_env/bin/python /home/chetan/contact_model_study/results/isolated_bo/duck_ball_friction_20261116/run_search.py --cell 6
cell 7: ball_mass140g_T0.15
  /home/chetan/miniconda3/envs/mj_env/bin/python /home/chetan/contact_model_study/results/isolated_bo/duck_ball_friction_20261116/run_search.py --cell 7
cell 8: duck_mass50g_T0.2
  /home/chetan/miniconda3/envs/mj_env/bin/python /home/chetan/contact_model_study/results/isolated_bo/duck_ball_friction_20261116/run_search.py --cell 8
cell 9: ball_mass140g_T0.2
  /home/chetan/miniconda3/envs/mj_env/bin/python /home/chetan/contact_model_study/results/isolated_bo/duck_ball_friction_20261116/run_search.py --cell 9
