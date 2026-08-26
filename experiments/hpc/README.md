# HPC sweeps

Two ways to run a batch of experiments as a SLURM **job array**, both ending in
a merged `combine_results.py` output:

- **Grid search** (below): the grid (models × weight combos) is hardcoded as
  bash arrays in `param_search.slurm`.
- **[CSV-driven sweep](#csv-driven-sweep)**: every experiment is one row of a
  CSV instead — no bash grid to edit, and each row can pick its own
  planner/model.

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
| `run_csv_cell.py` | Worker. Runs one CSV row (`--n_episodes` episodes via `run_eval_episode`) and writes `cell_<row>.json`. |
| `run_csv_sweep.slurm` | The array job. Reads `$CSV`, guards `$SLURM_ARRAY_TASK_ID` against the row count, runs the worker, and (from row 0) queues the combine job. |
| `submit_csv_sweep.sh` | Counts `$CSV`'s data rows and submits `run_csv_sweep.slurm` with a matching `--array`. **This is what you run.** |
| `example_params.csv` | Template covering mppi/cem/predictive_sampler rows and mixed models. |

### CSV columns

One row = one experiment. All columns are optional except `task`.

- **Reserved** (experiment-level, not planner knobs): `task`, `model`,
  `planner`, `n_episodes`, `seed`, `geometry`, `hand_acc`, `obj_acc`,
  `eval_sim`, `settle`, `eval_substeps`, `record_trajectory`,
  `record_planner_dist`, `planner_dist_every`. A blank/missing cell falls
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
- **`w_<name>`**: cost-weight overrides, one column per weight (e.g.
  `w_quat`, `w_pos_x`, `w_quat_term`) — any key of the task's
  `cost_weights`.
- **Everything else**: forwarded straight to
  `make_planner_config(planner, **row)` (`contact_study/planners/__init__.py`),
  which only keeps the fields the selected planner's config declares — so a
  column irrelevant to a row's planner (e.g. `alpha` on an mppi row) is
  silently ignored, and a blank cell leaves that field at its own default.
  Use the planner dataclass field names directly: `n_samples`,
  `time_horizon`, `step_time`, `noise_sigma`, `resample_interval`,
  `warm_start`, `nconmax`, `njmax`, `time_constrained`, `plan_budget_ms`;
  MPPI's `temperature`; CEM's `n_elites`/`elite_frac`/`alpha`/`min_sigma`;
  predictive sampler's `include_nominal`. `delta` is special-cased to a
  single clip magnitude, expanded to `delta_range=(-delta, delta)`.

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
