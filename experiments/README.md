# Experiments: batches, grids, Bayesian optimization, and analysing results

`run_episode_batches.py` runs batches of `run_episodes.py` episodes described by
a CSV, one row per batch (a *cell*). It works the same on a workstation and on
the HPC:

| | Command |
| --- | --- |
| Check a CSV (runs nothing) | `python experiments/run_episode_batches.py my.csv --check` |
| Run every cell, in order | `python experiments/run_episode_batches.py my.csv` |
| Run one cell (0-based row) | `python experiments/run_episode_batches.py my.csv --cell 3` |
| Rebuild the summary | `python experiments/run_episode_batches.py my.csv --summarize --outdir <results dir>` |
| Submit to the HPC | `experiments/hpc/submit_episode_batches.sh my.csv [max concurrent]` |

`example_batches.csv` is a template.

## The CSV

Each column is one setting. A blank cell, or a missing column, keeps the
driver's default.

- **Driver options.** Any `run_episodes.py` flag, with underscores instead of
  dashes. For example: `task`, `n_episodes`, `steps`, `rollout_model`,
  `eval_sim`, `hand_acc`, `obj_acc`, `timestep`, `ctrl_time_step` or
  `substeps`, `time_horizon` or `horizon`, `n_samples`, `noise_sigma`,
  `temperature`, `control_mode`, `settle`, `seed`, `goal_difficulty`,
  `nconmax`, `njmax`. `run_episodes.py --help` lists them all.
  `eval_sim` takes a CPU simulator (`mujoco`, `pinocchio`, `drake`) or a GPU
  contact model (`M1`–`M5`; M5, Kamino, needs the `contact_kamino` env).
  `plan_on_eval` (true/false) makes the planner roll out on the eval scene.
- **On/off options.** These take `true` or `false`: `stop_on_success`,
  `warm_start`, `uncertainty`, `save_steps`, `graph`, `debug`, ...
- **Cost weights.** `w_quat`, `w_pos_x`, ..., `w_fallen_term` override the
  object's tuned weights.
- **`label`.** Names the row in the output files.
- **`video`.** `true` records each episode's video.

Any other column is an error, so a misspelled one can't silently run a whole
sweep at the default. `--check`, which the submit script always runs first,
also catches bad values: an unknown model, both `horizon` and `time_horizon`,
a non-number.

## Output

Each run gets one folder, `results/episode_batches_<csv name>_<job id or time>/`
(set it with `--outdir`, or `OUTDIR=` on the HPC). For each cell it holds:

- **`cell_<row>_<label>.json`:** the `EpisodeRecorder` results, with one `.npy`
  of per-step data per episode beside it unless the row sets
  `save_steps=false`. Read it with `EpisodeReplayer`.
- **`cell_<row>_<label>.status.json`:** `done` or `failed`, the exact driver
  command line, the wall time and any error.
- **`cell_<row>_<label>.log`:** the driver's output, when run locally.
- **`summary.csv` / `summary.json`:** every row's settings next to its status,
  success rate, successes, mean steps to success and mean planning time.

**Resuming.** A cell marked `done` is skipped when the same CSV is run again
into the same folder, so a run that stopped part-way picks up where it left
off. `--overwrite` redoes cells. Keep the CSV's row order when resuming: cells
are named by row number.

**Failures.** Run locally, each cell runs in its own process, so a failing cell
is marked `failed` and the rest carry on (`--stop-on-error` stops instead).

## On the HPC (`hpc/`)

| File | Role |
| --- | --- |
| `submit_episode_batches.sh` | **What you run**, from the repo root, on the login node. It loads `miniconda` and activates the `contact_modeling` env (`CONDA_ENV=` picks another) so it can run Python there, then validates the CSV, submits the array sized to its rows (an optional second argument caps how many cells run at once), and queues the summary. Extra `sbatch` options go in `SBATCH_ARGS`, for example `SBATCH_ARGS="--time=24:00:00"`. |
| `run_episode_batches.slurm` | The job array: task *i* runs `--cell i` on its own GPU node. Every task writes into one folder. |
| `summarize_batches.slurm` | Builds `summary.csv` after the array. It is queued with `afterany`, so it also runs when some cells fail. |

```bash
mkdir -p logs
experiments/hpc/submit_episode_batches.sh experiments/example_batches.csv
# resume an earlier run: finished cells are skipped
OUTDIR=/abs/path/results/episode_batches_example_batches_123456 \
    experiments/hpc/submit_episode_batches.sh experiments/example_batches.csv
```

The resources in `run_episode_batches.slurm` (one `rtx_5000_ada`, 16 GB, 12 h)
apply to every cell, so they must cover the most expensive row. To estimate a
row's cost, run `test_scripts/profile_run_episodes.py -- <that row's flags>`.
The project path defaults to the cluster checkout; set `PROJ_DIR` to use
another.

# Temperature × noise-sigma grid search

`run_temp_sigma_grid.py` searches MPPI's `temperature` and `noise_sigma` for
each cell of a CSV. The right temperature scales with the size of the task
cost, which differs per contact model, object and horizon, so each row names
its own grid.

For every grid point, the cell runs `n_episodes` episodes through
`run_episodes_interwoven.py`: while one episode plans on the GPU, the other
steps its eval simulator on the CPU. Every point sees the same episodes,
because goals and planner noise depend only on `(seed, episode)`. So points
differ only in their two settings. Points are ranked by success rate, ties
broken by mean steps to success.

| | Command |
| --- | --- |
| Check a CSV (runs nothing) | `python experiments/run_temp_sigma_grid.py grid.csv --check` |
| Every cell, in order | `python experiments/run_temp_sigma_grid.py grid.csv` |
| One cell | `python experiments/run_temp_sigma_grid.py grid.csv --cell 2` |
| Rebuild the summaries | `python experiments/run_temp_sigma_grid.py grid.csv --summarize --outdir <results dir>` |
| Submit to the HPC | `experiments/hpc/submit_temp_sigma_grid.sh grid.csv [max concurrent]` |

`example_temp_sigma_grid.csv` is a template.

**The CSV.** It takes the same columns as the batch CSV (driver options,
`w_*` weights, `label`, `video`), plus two required columns:

- `temperatures`: the values to search, separated by spaces or commas
  (quote a comma list), all positive, no repeats. They are the outer loop.
- `noise_sigmas`: the same format. One value makes it a 1-D temperature sweep.

`temperature` and `noise_sigma` themselves can't be columns, because the grid
sets them.

**Output**, in `results/temp_sigma_grid_<csv>_<job id or time>/`:

- **One folder per cell** (`cell_<row>_<label>/`), containing:
  - `point_<i>_T<t>_s<σ>.json` (and `.npy` files if `save_steps`) and
    `.status.json` for each point;
  - `grid_summary.json`, with the points ranked and the best one.
- **At the top:**
  - `summary.csv`: every point with its cell's settings and its rank;
  - `best.csv`: each cell's best point.

Each finished cell also prints its ranked table and a ready-to-paste
`run_episodes.py` command for its best point.

**Resuming.** A point marked done is skipped on a rerun into the same folder,
so a cell that hits the wall clock picks up point by point. Resubmit with
`OUTDIR=<that folder>`, and keep the same CSV.

**On the HPC.** `hpc/submit_temp_sigma_grid.sh`, `hpc/run_temp_sigma_grid.slurm`
and `hpc/summarize_temp_sigma_grid.slurm` mirror the batch scripts: one array
task per row on its own GPU, validation first, and a summary job queued
`afterany`.

- **CPUs:** each task asks for 4, for the interwoven driver's planner and two
  workers.
- **Time:** a cell costs `#temperatures × #noise_sigmas × n_episodes`
  episodes, so the time limit has to cover the widest row. Raise it with
  `SBATCH_ARGS="--time=24:00:00"`.

# Bayesian optimization of weights, temperature and noise

`run_bayes_opt.py` runs one Bayesian optimization (BO) per CSV row. It uses
scikit-optimize's Gaussian process to search the cost weights, MPPI's
`temperature` and `noise_sigma`. Each *trial* is one point of the search,
scored by running `n_episodes` episodes on every model in `rollout_models`, with
the same weights for all of them.

Every episode of a cell runs on one long-lived pool (`EpisodePool`, the engine
of `run_episodes_pooled.py`): a planner process per GPU, an eval-sim worker per
remaining core, and several trials in flight at once. While trials are
pending, the next point is chosen as if each pending trial had scored the best
J so far (a *constant liar*), so concurrent trials don't pile onto one spot.
A trial's settings travel with its episodes, and the planner processes apply
them in place, so nothing restarts between trials.

Every trial and every model sees the same episodes, because goals and planner
noise depend only on `(seed, episode)`.

| | Command |
| --- | --- |
| Check a CSV (runs nothing) | `python experiments/run_bayes_opt.py bo.csv --check` |
| Every cell, in order | `python experiments/run_bayes_opt.py bo.csv` |
| One cell, sized pool | `python experiments/run_bayes_opt.py bo.csv --cell 0 --gpus 0,1 --workers 12` |
| Rebuild the summaries | `python experiments/run_bayes_opt.py bo.csv --summarize --outdir <results dir>` |
| Submit to the HPC | `experiments/hpc/submit_bayes_opt.sh bo.csv [max concurrent]` |

`example_bayes_opt.csv` is a template, and `example_bo_seeds.csv` is an example
seed file.

**Objective** (minimized), as in the old `run_bayes_opt.py`:

```
J_model = -w_success * success_rate + w_cost * mean normalized final goal error
```

The goal error divides each final error by the task's success threshold, sums
them, clips at `err_clip` and divides by it, so both terms are in [0, 1]. An
episode that raised scores as a failure with the worst error. `model_agg`
folds the models' J: `mean`, or `worst` (the largest).

**The CSV.** It takes the batch CSV's columns (driver options, `w_*` to pin a
weight, `label`), except `rollout_model` and `video`, plus:

| Column | Meaning | Blank |
| --- | --- | --- |
| `rollout_models` | Models scored with the same weights, e.g. `M1 M2 M3 M4` | the driver default |
| `model_agg` | `mean` or `worst` | `mean` |
| `opt_weights` | Weights to search: `w_quat:1:50 w_contact` (bare name: x/4 to x4 around the object's value), or `none` | the old study's nine weights and bounds, widened to contain the object's own values |
| `temperature_range`, `noise_sigma_range` | `lo hi` to search | pinned to the `temperature`/`noise_sigma` column or the driver default |
| `per_model_temperature` | `true`: a `temperature_<model>` dimension per model | `false` |
| `n_calls` | Trials in all, including seeds and resumed trials | 100 |
| `n_initial_points` | Random trials before the GP, less one per seed | 10 |
| `acq_func`, `bo_seed` | `gp_hedge`, `EI`, `LCB` or `PI`; the optimizer's seed | `gp_hedge`, 0 |
| `w_success`, `w_cost`, `err_clip` | The objective | 1, 0.1, 250 |
| `trials_in_flight` | Trials evaluated at once | enough to give every worker an episode |

Every dimension is log-uniform. `--check` rejects:

- an empty search space;
- a knob that is both pinned and searched;
- an unknown weight or model;
- a seed outside the search box.

**Seeding with known settings.** Seeds run before the random and GP trials,
count toward `n_calls`, and each replaces one random trial.

- `seed_defaults` (on unless `false`): the object's own weights, with the
  pinned temperature and noise. The default temperature must then lie inside
  `temperature_range`. Turn this off if it doesn't.
- `seed_points`: a CSV of known settings (path relative to the batch CSV),
  with one row per point and columns named after the dimensions (`w_quat`,
  `temperature`, `noise_sigma`, ...). Anything a row leaves out takes its
  default. A value for a pinned setting must equal the pinned value.
- `seed_from` + `seed_top_k` (5): an earlier cell folder. Its best trials
  are **re-run** here, so their scores come from this row's episodes and
  models. Dimensions that don't exist here are ignored, and values outside the
  box are clipped into it.

**Output**, in `results/bayes_opt_<csv>_<job id or time>/`:

- **One folder per cell** (`cell_<row>_<label>/`):
  - `trial_<i>/<model>.json` (and `.npy` files if `save_steps`): the trial's
    episodes on that model, readable with `EpisodeReplayer`. The recorded
    configs carry the trial's temperature, noise and weights.
  - `trial_<i>/trial.json`: the point, its source (`defaults`,
    `seed_points:<row>`, `seed_from:<trial>`, `random`, `gp`), its J and the
    per-model scores.
  - `bo_state.json`: the space and every point and J.
  - `bo_summary.json`: the trials ranked, the best-so-far trace, the best
    trial and the commands to replay it.
- **At the top:**
  - `summary.csv`: every trial, with its cell's settings;
  - `best.csv`: each cell's best trial.

**Resuming.** Rerun into the same folder, with `OUTDIR=` on the HPC. Finished
trials are told to the optimizer again, and the search continues to
`n_calls`; raise `n_calls` to extend a finished search. Trials that were still
running are redone. The search space may be widened, but a changed set of
dimensions, or a narrowed range that leaves earlier trials outside it, is
refused. `--overwrite` starts the cell over.

**Stopping.** Ctrl-C or `scancel` stops at the next control step. Finished
trials are kept.

**On the HPC.** `hpc/submit_bayes_opt.sh`, `hpc/run_bayes_opt.slurm` and
`hpc/summarize_bayes_opt.slurm` mirror the other pipelines: one array task per
row, validation first, and a summary job queued `afterany`.

- **Resources:** each task asks for 2 GPUs, 16 CPUs, 32 GB and 24 h. The pool
  uses whatever SLURM grants. Change it per submission, for example
  `SBATCH_ARGS="--gpus=rtx_5000_ada:4 --cpus-per-task=32 --time=48:00:00"`.
- **Cost:** a cell is `n_calls × models × n_episodes` episodes.

# Processing batch results: KL to the optimal planner, forward-simulation error

`process_batch_results.py` takes a results folder written by
`run_episode_batches.py` and analyses every cell:

1. **KL to the optimal planner.**
   - **The used planner** is rebuilt from the cell's recorded command line.
   - **The optimal planner** is the same planner (sample count, temperature,
     noise, weights, and the same control step and horizon as durations)
     planning with the eval model: the eval scene at the eval timestep, with
     the eval simulator's contact model.
   - **The replay:** both walk every recorded episode, planning from the
     recorded states with the recorded previous controls and each step's goal.
     Each planner keeps its own warm state and resets where the episode did.
     Both draw the same noise.
   - **The measure:** each step's first actions become Gaussians, and both
     `KL(used‖opt)` and `KL(opt‖used)` are kept, along with the distance
     between their means.
2. **Forward-simulation error.** From every recorded state, with that step's
   action, the rollout model (rollout scene and timestep) and the eval
   simulator (eval scene and timestep) each take one control step. All the
   steps of an episode run at once, as the worlds of one GPU simulator.
   - **The comparison:** object position (m) and rotation (rad), hand joints
     (RMS rad), and the full `qpos`/`qvel`.
   - **A sanity column (`eval_vs_recorded_*`):** the eval prediction against
     what the episode actually did next. This shows the cost of restarting
     from a recorded state; it is about 0.04 mm.

**Requirements.** The cell must have been run with `save_steps=true`, and its
`eval_sim` must be a GPU contact model (`M1`–`M5`): the optimal planner and the
forward step need a vectorized eval model. Anything else fails that cell with
the reason.

| | Command |
| --- | --- |
| Count the cells | `python experiments/process_batch_results.py <batch dir> --count` |
| Analyse every cell, then the final tables | `python experiments/process_batch_results.py <batch dir>` |
| One cell | `python experiments/process_batch_results.py <batch dir> --cell 2` |
| Only the final tables | `python experiments/process_batch_results.py <batch dir> --summarize` |
| Submit to the HPC | `experiments/hpc/submit_process_batch_results.sh <batch dir> [max concurrent]` |

**Options:**

- `--stride k` computes the KL only every k-th step. Each planner keeps its
  warm state between the steps it plans.
- `--max-episodes` limits the episodes analysed per cell.
- `--shrinkage` is the KL's covariance shrinkage (1e-3).
- `--overwrite` redoes cells already analysed.

**Output**, written into the batch folder:

- **`analysis/<cell>.analysis.json`:** each metric's mean, median and p90,
  per episode and per cell, plus how the optimal planner was built.
- **`analysis/<cell>.analysis.npz`:** the per-step arrays.
- **`analysis/<cell>.analysis_status.json`:** `done` or `failed`, with the
  error.
- **`final_results.csv` (and `.json`):** one row per cell. It holds the cell's
  settings, the batch's own results as in `summary.csv` (status, success rate,
  steps to success, plan time), the analysis status, and every analysis
  metric.
- **`final_episodes.csv`:** one row per episode, the same way.

**Goals.** New results save each goal's quaternion (`goal_values`,
`goal_steps`). Older ones are rebuilt by replaying the eval task's goal stream
from the recorded seed, and checked against the saved goal labels.

**Cost.** The optimal planner runs at the eval timestep, so each recorded step
costs a few plans: about 1–2 s at 0.5 ms eval and 64 ms control steps.

**On the HPC.** `submit_process_batch_results.sh <batch dir> [max concurrent]`
counts the cells, submits `process_batch_results.slurm` (one GPU per cell,
12 h), and queues `summarize_batch_results.slurm` `afterany` to build
`final_results.csv`.

- **Extra flags** go in `ANALYSIS_ARGS`, e.g. `ANALYSIS_ARGS="--stride 4"`.
- **Resubmitting** skips the cells already done.
