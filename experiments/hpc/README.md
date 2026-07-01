# HPC weight grid search

Runs `experiments/run_param_search.py`'s sweep as a SLURM **job array**: the grid
(models × weight combos) is split so each array task runs **one** weight set for a
few episodes, writes its own JSON, and a final job combines them.

## Files

| File | Role |
|------|------|
| `search_config.sh` | **Single source of truth** — task, models, the list of each weight, episode count, MPPI knobs. Edit this to change the sweep. |
| `run_param_cell.py` | Worker. Rebuilds the grid from CLI args and runs the one cell given by `--combo_index` (via `run_eval_episode`), writing `cell_<i>.json`. |
| `combine_results.py` | Merges all `cell_*.json` into `<prefix>_rich.json` + `<prefix>_agg.json` and prints a ranked top-N table. |
| `param_search.slurm` | Array job: one task per cell, `--combo_index $SLURM_ARRAY_TASK_ID`. |
| `combine.slurm` | Runs the combiner after the array finishes. |
| `submit_param_search.sh` | Convenience: sizes the array, submits it, and chains the combine job (`afterok`). |

## Usage

1. Edit `search_config.sh` (grid + MPPI knobs) and fill in the `TODO` cluster
   lines (`--partition`, `--account`, env activation) in the two `.slurm` files.
2. Submit everything:
   ```bash
   cd experiments/hpc
   ./submit_param_search.sh          # or: MAX_CONCURRENT=40 ./submit_param_search.sh
   ```
   This computes the number of cells, submits the array (one job per weight set),
   and queues the combine job to run automatically once the array succeeds.
3. Results land in `results/param_search_<timestamp>/`:
   - `cell_00000.json … cell_NNNNN.json` — one per weight set (success rate + per-episode detail)
   - `combined_<task>_rich.json` / `combined_<task>_agg.json` — merged
   - the combine job's log prints the ranked top configs.

### Running without the submit helper

`sbatch param_search.slurm` works too, but set `#SBATCH --array=0-<N-1>` yourself
(`N = source search_config.sh; num_cells`) and export `OUTDIR`, e.g.
`OUTDIR=results/run1 sbatch --array=0-323 --export=ALL,OUTDIR param_search.slurm`.

### Running locally (no SLURM)

The worker runs the whole grid serially when `--combo_index` is omitted:
```bash
python run_param_cell.py --outdir results/local_run --n_episodes 3 \
    --models M2 M3 --weights w_quat=15,20,25 w_pos=15,20,25
python combine_results.py --indir results/local_run
```
Count the cells first with `--num_combos` (add `--models`/`--weights` to match).
