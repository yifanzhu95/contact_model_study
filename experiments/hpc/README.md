# HPC weight grid search

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
