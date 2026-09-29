# Experiments: batches of episodes from a CSV

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
