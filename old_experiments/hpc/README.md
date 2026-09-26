# HPC sweeps

Three ways to run a batch of experiments as a SLURM **job array**, all ending
in a merged `combine_results.py` output:

- **Grid search** (below): the grid (models × weight combos) is hardcoded as
  bash arrays in `param_search.slurm`, and each array task evaluates **one**
  point of it.
- **[CSV-driven sweep](#csv-driven-sweep)**: every experiment is one row of a
  CSV instead — no bash grid to edit, and each row can pick its own
  planner/model.
- **[Temperature / noise-sigma grid](#temperature--noise-sigma-grid)**: the
  array assigns each GPU a contact model and an object, and the cell itself
  walks a whole temperature × `noise_sigma` grid with its episodes running
  concurrently in a pool.

## HPC weight grid search

Runs the cost-weight sweep as a SLURM **job array**: the grid (models × weight
combos) lives in `param_search.slurm`, which maps `$SLURM_ARRAY_TASK_ID` to one
value per axis, runs a few episodes for that one weight set, and writes its own
JSON. When the array finishes, a combine job merges every cell.

## Files

| File | Role |
|------|------|
| `param_search.slurm` | The array job. Defines the parameter grids inline, decodes the task id into one value per axis, runs the cell, and (from task 0) queues the combine job. **This is the only thing you submit.** |
| `run_param_cell.py` | Worker. Runs `--n_episodes` episodes for one `--model` + `--weights` set (via `run_eval_episode`) and writes `cell_<id>.json`. |
| `bayes_opt.slurm` | Array job over **objects × contact models**. Each cell runs its own `contact_study.drivers.run_bayes_opt` (scikit-optimize GP search over the cost weights + `noise_sigma`/`temperature`) into its own `results/bayes_opt_<arrayjobid>/<task>_<obj>_<model>_<planner>/`. Not a grid *search* — the array axes just fan out one independent optimization per pair. Needs `scikit-optimize` in the env; re-submitting with `OUTDIR_ROOT` pointing at a previous run resumes each cell from its `bo_state.json`. |
| `temp_sigma_grid.slurm` | Array job over the rows of a **cell CSV** (`temp_sigma_cells.csv`: object, contact model, `n_iterations`, `n_samples`, `time_horizon`, `step_time`, plus the `temperatures`/`noise_sigmas` to search), one cell per GPU. Each cell runs `contact_study.drivers.run_temp_sigma_grid` over that row's temperature × `noise_sigma` grid. Submit with `submit_temp_sigma_grid.sh`. See [its section](#temperature--noise-sigma-grid). |
| `combine.slurm` | Runs the combiner after the array (queued automatically as an `afterok` dependency). |
| `combine_results.py` | Merges all `cell_*.json` into `<prefix>_rich.json` + `<prefix>_agg.json` and prints a ranked top-N table. |

## Usage

1. In `param_search.slurm`, edit the grids (`MODELS`, `W_QUAT`, `W_POS`,
   `W_CONTACT`, `W_JOINT`), `N_EPISODES`, and the shared MPPI knobs.
2. **Keep `#SBATCH --array` in sync** with the grid: it must be
   `0-(product of the array lengths − 1)`. The default `4×3×3×3×3 = 324` → `0-323`.
3. Submit:
   ```bash
   cd experiments/hpc
   mkdir -p logs
   sbatch param_search.slurm
   ```
   One job runs per weight set; task 0 also queues the combine job to run once
   the whole array succeeds.

Results land in `results/param_search_<arrayjobid>/`:
- `cell_00000.json … cell_00323.json` — one per weight set (success rate + per-episode detail)
- `combined_<task>_rich.json` / `combined_<task>_agg.json` — merged
- the combine job's log prints the ranked top configs.

### Adding or removing a swept axis

The grids are decoded lowest-axis-first (model, then quat, pos, contact, joint).
To add an axis, add its array, a matching `IDX` line in the mixed-radix decode,
a `SEL_*` selection, and another `name=value` token in the `--weights` call — then
update `#SBATCH --array` to the new product. Weight names must match the task's
`cost_weights` keys.

### Combining by hand

If the auto-queued combine job didn't run:
```bash
OUTDIR=results/param_search_<id> TASK=grasp_reorient sbatch combine.slurm
# or directly:
python combine_results.py --indir results/param_search_<id>
```

## CSV-driven sweep

An alternative to editing bash grids in `param_search.slurm`: put every
experiment as one row of a CSV and let a job array run it. SLURM has no
native "one job per CSV row" feature — array bounds are static in the job
script — so a tiny submit wrapper sizes `--array` to the file first.

| File | Role |
|------|------|
| `run_csv_cell.py` | Worker. Runs one CSV row (`--n_episodes` episodes via `run_eval_episode`, or `run_async_eval_episode` when the row sets `driver=async`) and writes `cell_<row>.json`. |
| `run_csv_sweep.slurm` | The array job. Reads `$CSV`, guards `$SLURM_ARRAY_TASK_ID` against the row count, runs the worker, and (from row 0) queues the combine job. |
| `submit_csv_sweep.sh` | Counts `$CSV`'s data rows and submits `run_csv_sweep.slurm` with a matching `--array`. **This is what you run.** |
| `example_params.csv` | Template covering mppi/cem/predictive_sampler rows and mixed models. |

### CSV columns

One row = one experiment. All columns are optional except `task`.

- **Reserved** (experiment-level, not planner knobs): `task`, `model`,
  `planner`, `n_episodes`, `seed`, `geometry`, `hand_acc`, `obj_acc`,
  `eval_sim`, `settle`, `eval_substeps`, `goal_difficulty`, `record_trajectory`,
  `record_planner_dist`, `planner_dist_every`, `driver`, `plan_latency_ms`,
  `latency_scale`, `plan_warmup`, `executor`, `async_shift`. A blank/missing cell falls
  back to `run_csv_cell.py`'s own `--model` / `--planner` /
  `--record_trajectory` / ... default (see `build_parser`). These go to
  `run_eval_episode` directly — `eval_substeps` in particular is *not* a
  planner-config field, so it has to be handled here rather than forwarded.
  The three recording columns mirror
  `contact_study/evaluation/trajectory.py`'s CLI flags, letting most rows run
  lean (`false,false`) while a handful opt into full per-step recording for
  inspection.
  `hand_acc`/`obj_acc` split the hand and object collision-geometry fidelity
  (`SceneVariant` in `contact_study/tasks/config.py`, same axes
  `bayes_opt.slurm` sweeps as `HAND_ACC`/`OBJ_ACC`) out of `geometry` so they
  can vary per row: when either is set, `geometry` is read as the bare
  object name and the three are joined into `"<geometry>_<hand_acc>_<obj_acc>"`
  (a blank side falls back to that axis's task default). Leave both blank to
  use `geometry` exactly as before — a bare object name or an
  already-composed `"<obj>_<hand_acc>_<obj_acc>"` string.
  `goal_difficulty` picks which goal the task asks for, for tasks that have
  levels (`grasp_reorient`: 0 fixed 90 deg spin, 1 +/-90 deg spin, 2 +/-90 deg
  about a random object axis, 3 adjacent face + random twist, 4 any other face
  + twist, 5 180 deg flip, 6 adjacent face no twist, 7 roll to "O", 8 roll
  either way, 9 roll to "B" — see `GraspReorientTask`'s class docstring). A
  blank cell keeps the task's own default; a level on a task with no difficulty levels is an
  error rather than a silent no-op. Like `driver`, it is folded into the cell
  label when the column is present, so rows differing only in difficulty stay
  separate rows in `combine_results.py`.
  `driver` picks the control loop: `sync` (the default, and what every row
  did before this column existed) runs `run_eval_episode`, which freezes the
  eval simulator for the duration of `plan()`; `async` runs
  `contact_study/drivers/run_async_eval_episode.py`, which keeps the sim
  running and charges the planning latency as *simulated* time, so a slower
  contact model no longer looks free. The five async knobs —
  `plan_latency_ms` (impose a latency in ms instead of measuring it; the most
  sweepable axis), `latency_scale` (multiply the measured latency),
  `plan_warmup`, `executor` (`zoh`/`tape`/`time`) and `async_shift` — are
  arguments of that driver rather than planner-config fields, which is why
  they are reserved here. They are **rejected** on a `driver=sync` row rather
  than silently ignored.
- **`w_<name>`**: cost-weight overrides, one column per weight (e.g.
  `w_quat`, `w_pos_x`, `w_quat_term`) — any key of the task's
  `cost_weights`.
- **Everything else**: forwarded straight to
  `make_planner_config(planner, **row)` (`contact_study/planners/__init__.py`),
  which only keeps the fields the selected planner's config declares — so a
  column irrelevant to a row's planner (e.g. `alpha` on an mppi row) is
  silently ignored, and a blank cell leaves that field at its own default.
  Use the planner dataclass field names directly: `n_samples`,
  `n_iterations`, `time_horizon`, `step_time`, `noise_sigma`, `resample_interval`,
  `warm_start`, `nconmax`, `njmax`, `time_constrained`, `plan_budget_ms`;
  MPPI's `temperature`; CEM's `n_elites`/`elite_frac`/`alpha`/`min_sigma`;
  predictive sampler's `include_nominal`. `delta` is special-cased to a
  single clip magnitude, expanded to `delta_range=(-delta, delta)`.
  `time_constrained` is special-cased too, in that setting it alone is
  enough: `PlannerConfig` otherwise rejects it unless `plan_budget_ms > 0`
  and `use_full_graph=false` agree with it, so `resolve_time_constraint`
  defaults the budget to the row's `step_time` (in ms — the control period
  the cell is running at, the same rule `run_cntrl_freq_cell.py` uses) and
  forces the step-graph path, printing what it filled in. Give
  `plan_budget_ms` explicitly to pin the budget instead. Combined with
  `driver=async` this is the anytime planner: latency capped by truncating
  the horizon.

Because each row names its own `planner`/`model`, one CSV can compare
planners or contact models side by side — something the bash-grid sweep
can't do without duplicating itself.

A column that is none of the three (i.e. not a field of *any* planner config)
is rejected up front with the list of valid names. That guard matters because
`make_planner_config` ignores keys the planner doesn't declare — which is
what lets one CSV mix planners, but would otherwise let a misspelled
`temprature` column quietly run a whole sweep at the default.

### Usage

Run from the **repo root** — `submit_csv_sweep.sh` passes
`experiments/hpc/run_csv_sweep.slurm` to `sbatch` as a root-relative path, and
the job's `#SBATCH --output=logs/...` resolves against the submission dir, so
the logs land in `<repo-root>/logs` alongside the combine job's.

```bash
mkdir -p logs
./experiments/hpc/submit_csv_sweep.sh experiments/hpc/example_params.csv
# or, with your own file (the CSV path may be relative or absolute):
./experiments/hpc/submit_csv_sweep.sh path/to/params.csv my_run_label
```

Results land in `results/csv_sweep_<arrayjobid>/`, same layout as the grid
search (`cell_00000.json ...`, `combined_<label>_rich.json` /
`_agg.json`) — `combine_results.py` needs no changes since `run_csv_cell.py`
writes the same per-cell schema `run_param_cell.py` does.

To run a single row locally (e.g. to sanity-check a CSV before submitting):
```bash
python experiments/hpc/run_csv_cell.py \
    --csv experiments/hpc/example_params.csv --row 0 --outdir /tmp/csv_test
```

## Temperature / noise-sigma grid

A third array job, for the two planner knobs that do **not** transfer between
contact models and objects. MPPI's `temperature` divides the cost inside the
softmax weighting, so the useful λ scales with the magnitude of the task cost —
which differs per contact model, per object and per horizon. The readable
picture is therefore one grid per cell, not a single global optimum — and each
cell names the grid worth searching in it, as two columns of its own CSV row.
The array also fans `n_iterations`, `n_samples` and the planning horizon /
control-step duration out across cells — see the bullets below.

| File | Role |
|------|------|
| `temp_sigma_cells.csv` | The cell table: one row per array task, columns `object,model,n_iterations,n_samples,time_horizon,step_time,temperatures,noise_sigmas` (all required). The last two are **lists** — space- or comma-separated (`100 50 25` or a quoted `"100,50,25"`) — and are the grid searched inside that one cell. Ships with the 48-cell `cube × M1–M4 × {1,2,3} × {16,64,256,1024}` grid at `0.352 s / 0.064 s`, every row searching `T ∈ {100, 50, 25, 12.5, 6.25} × σ ∈ {0.1}`; edit rows freely. |
| `submit_temp_sigma_grid.sh` | Counts (and validates) the CSV's rows, then `sbatch`es `temp_sigma_grid.slurm` with `--array=0-(N-1)%throttle`. **This is what you run.** |
| `temp_sigma_grid.slurm` | The array job. Reads row `$SLURM_ARRAY_TASK_ID` of the CSV (via `read_temp_sigma_cell.py`) and runs one cell. Knobs that are genuinely global — task, `HAND_ACC`/`OBJ_ACC`, episode settings, `USE_CONVERGENCE` — still live here; the temperature / `noise_sigma` lists do **not**, they are per row in the CSV. |
| `read_temp_sigma_cell.py` | Helper the job script reads its row with: validates the header, bounds-checks the row, checks each grid list is positive numbers with no repeats, and rejects two rows naming the same `(object, model, n_iterations, n_samples, time_horizon, step_time)` (see below). |
| `contact_study/drivers/run_temp_sigma_grid.py` | The cell. Walks that row's temperature × `noise_sigma` grid, running episodes through the same `EpisodePool` as `run_bayes_opt.py`, and writes one `cell_<array index>_<point>.json` per grid point. |

How it differs from the two sweeps above:

- Unlike `param_search.slurm`, the SLURM script does **not** own the grid — it
  owns the *cells*. The grid lives in the CSV row's `temperatures` /
  `noise_sigmas` lists, passed straight to the driver, so bash never needs to
  know how many points they expand to, and one array task keeps a GPU busy for a
  whole grid instead of a single point. Per row and not per submission because
  the useful λ scales with the cost magnitude a cell actually sees: a cheap cell
  can afford a wide sweep while an expensive one searches the three values worth
  trying. The cost of a cell is its own `#temperatures × #noise_sigmas ×
  N_EPISODES`, so a wide row is a long row.
- `n_iterations`, `n_samples`, `time_horizon` and `step_time` are *cell* axes,
  not grid axes, because they change what an episode **costs** rather than only
  how it scores: one cell per value keeps a cell's wall clock and VRAM constant
  instead of multiplying the inner grid. They live in a CSV rather than bash
  arrays because the rows are not a full cross product: a horizon worth
  searching at `n_samples=1024` may not be worth searching at 16, and the grid
  each cell deserves differs too — a mixed-radix decode of independent lists can
  express none of that, while a table is just the rows you want. Note the
  `#SBATCH` resource request is uniform across the
  array, so `--gpus`/`--mem`/`--time` must cover the most expensive row —
  `N_WORKERS × max n_samples` worlds, at the largest
  `n_iterations × time_horizon / step_time` (rollout steps per `plan()`).
- Two rows with the same `(object, model, n_iterations, n_samples,
  time_horizon, step_time)` are rejected up front: those six columns are what
  the driver's cell label and `analysis/temp_sigma_grid_to_csv.py`'s cell key
  identify a cell by, so such rows would merge into one result row. Differing
  `temperatures`/`noise_sigmas` do **not** separate them — put every value of a
  cell's grid in that cell's one row. Several horizons for one
  `(object, model, n_iterations, n_samples)` are fine, and are the reason the
  schedule is part of the key.
- `USE_CONVERGENCE=true` switches MPPI to its convergence-terminated mode
  (`--convergence_tol`/`--max_iterations`): `plan()` iterates until the returned
  action settles instead of running a fixed count, so the CSV's `n_iterations`
  column is ignored — drop the rows that differ only in it, or they rerun the
  same cell. Calibrate `CONVERGENCE_TOL` on one cell with `--debug`
  first (the planner prints `converged in k/cap iterations` per `plan()` call):
  a tolerance the update never reaches just runs every call to `MAX_ITERATIONS`
  at the cost of an extra device→host read per iteration.
- Unlike `bayes_opt.slurm`, the search is an even grid rather than a GP, and it
  needs no `scikit-optimize` (the driver deliberately imports nothing from
  `run_bayes_opt.py`, which pulls in `skopt` at module scope).

### Usage

```bash
experiments/hpc/submit_temp_sigma_grid.sh                          # temp_sigma_cells.csv
experiments/hpc/submit_temp_sigma_grid.sh my_cells.csv my_label 8  # label, %8 throttle
```

Edit the cell CSV (or write a new one and pass its path); the wrapper sizes
`#SBATCH --array` from the row count, so there is nothing to keep in sync. It
also validates the CSV before submitting, so a typo'd header, a blank value, a
grid value that is not a positive number (or is listed twice) and a duplicate
cell are all caught on the login node. A hand
`sbatch --array=... --export=ALL,CELLS_CSV=... temp_sigma_grid.slurm` still
works; an id past the end exits with a message naming the correct range rather
than silently rerunning cell 0.

Every array task writes into **one** shared directory,
`results/temp_sigma_grid_<arrayjobid>/`, each point named for its row and its
place in that row's grid (`cell_<array index>_<point>.json`), so cells searching
differently sized grids cannot collide and a single combine job merges every
cell. Task 0 queues that combine
automatically as an `afterok` dependency; by hand it is the usual

```bash
OUTDIR=/abs/path/to/results/temp_sigma_grid_1234567 TASK=temp_sigma_grid \
    sbatch experiments/hpc/combine.slurm
```

Each cell also writes `grid_summary_<array index>.json` — its own points ranked
by success rate, ties broken by mean steps-to-success — and prints that table
plus a ready-to-paste `run_eval_episode` replay command for the winner.

**Resuming.** A grid point whose `cell_*.json` already exists is skipped, so a
cell that hit the wall clock picks up where it stopped. Point the resubmission
at the old directory:

```bash
OUTDIR=/abs/path/to/results/temp_sigma_grid_1234567 \
    experiments/hpc/submit_temp_sigma_grid.sh my_cells.csv
```

Resume with the **same CSV**: cell files are named by row index, so a reordered
or shortened CSV would resume the wrong cells. (A directory written before the
files were named `cell_<row>_<point>.json` re-runs its points; its old
`cell_<id>.json` files are still read by `combine_results.py` and the analysis
script, which glob `cell_*.json`.)

To check one cell locally before submitting (2 points, 2 episodes):
```bash
python -m contact_study.drivers.run_temp_sigma_grid \
    --task grasp_reorient --geometry duck_low_high --model M2 \
    --temperatures "20 40" --noise_sigmas 0.1 --n_episodes 2 \
    --n_samples 64 --n_iterations 1 --time_horizon 0.352 --step_time 0.064 \
    --n_workers 2 --outdir /tmp/tsgrid --no-record_trajectory --no-record_planner_dist
# or the convergence-terminated variant (--n_iterations is then ignored):
#   ... --convergence_tol 1e-4 --max_iterations 10
```
