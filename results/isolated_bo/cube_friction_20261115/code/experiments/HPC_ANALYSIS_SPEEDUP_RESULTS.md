# Results of `HPC_ANALYSIS_SPEEDUP.md`, run on the HPC (2026-11-02/03)

Reply to `experiments/HPC_ANALYSIS_SPEEDUP.md`. Batch analysed:
`results/episode_batches_csv_run_11_2_26_31710923` (15 cells, M4 rollout / M5 eval, cube and ball,
30 episodes per cell in the CSV; 7 cells' episode runs had failed, so most cells have 5–6 episodes per shard).
Array job 31720242, `SHARDS_PER_GPU=2`, `N_SHARDS=5`, `N_STEPS_ANALYSED=100`, default `OPTIMAL_ARGS`
(`--n-samples 1024 --convergence-ratio 2 --max-iterations 20 --noise-sigma 0.38 --temperature 0.001 --no-adaptive-temp`).
GPU: NVIDIA RTX 5000 Ada, 32 GB.

## Summary

1. **Two shards per GPU gave no throughput gain.** A paired shard is 2.0x slower per step than one alone
   on its GPU, so throughput per GPU is ~1.0x (the doc expected 1.3–1.8x). One shard already showed 100 %
   GPU utilisation in `nvidia-smi`. Memory is not the issue (~4.9 GB per shard of 32 GB).
2. **The optimal planner is ~85 % of each analysed step, and the iteration cap is not the cost.** It
   converges in ~4 iterations (p90 6–8, cap 20), ~100 % of the time. The cost is per iteration:
   ~3.7 s/iteration with the GPU to itself (1024 samples, M5 on `env_leap_eval_cube.xml`, 16 x 4 ms
   substeps per control step). This is case 2 of the doc's step 3.
3. **`N_STEPS_ANALYSED` should stay 100.** At k = 2 every metric's `max |shift|` exceeds 0.25 and regret's
   width ratio is 1.12. Lag-1 is only 0.1–0.2, so analysed steps are nearly independent: fewer steps loses
   real information.
4. **A bug was found and fixed:** concurrent shards raced on a shared temp file (below).

## Bug: shared temp-file name

`_writeJson` and `_saveNpz` in `experiments/process_batch_results.py` wrote through
`path.with_name(path.name + ".tmp")`. Every shard of a cell writes `analysis/<cell>/_cell.json` at start,
and with shards of a cell starting together one shard's `os.replace` removed the tmp file under the
others:

```
FileNotFoundError: .../analysis/cell_0000_.../_cell.json.tmp -> .../_cell.json
```

13 of 75 shards died this way (cells 0–4), no other failures. Fix (uncommitted in the HPC working tree,
tests still pass): the temp name holds the pid, `path.with_name(f"{path.name}.{os.getpid()}.tmp")`.
Tasks started after the fix (13–37) had no failures. The race existed with one shard per GPU too, just
less likely. `git diff` is in the zip as `process_batch_results_fix.diff`.

## Per-step time, alone vs paired (from shard logs of job 31720242)

Shards whose partner died at start in the race ran alone on their GPU, giving a same-job baseline.
Medians over episodes:

| | episodes | per-step total | opt plan | opt iters | opt s/iteration |
| --- | --- | --- | --- | --- | --- |
| alone on GPU (tasks 1, 3, 4, 6, 8, 9, 10, 37) | 44 | 19.5 s | 15.9 s | 4.2 | 3.73 |
| 2 per GPU (tasks 2, 7, 12–36) | 270 | 39.3 s | 32.0 s | 4.2 | 7.82 |

Per-step total = used + opt + zero plans + regret rollouts (forward sim, ~1 s, not included).
Example lone-shard line: `per step: plans used 0.05 / opt 15.08 / zero 1.22 s, regret 2.40 s, opt 3.9
iterations, converged 100%; forward sim 1.0 s`. The doc's estimate of 1–2 s per step was ~10x low.

## Timing columns of `final_results.csv` (cells 5–14; cells 0–4 incomplete, see bug)

All measured with 2 shards per GPU, so roughly 2x the lone-GPU cost. Seconds per plan.

| cell | used | opt | zero | regret | n_iter_opt mean / p90 | converged_opt |
| --- | --- | --- | --- | --- | --- | --- |
| 5 cube low 1e-3 | 0.07 | 33.8 | 2.45 | 4.89 | 4.27 / 7 | 0.9996 |
| 6 cube high 0.1 | 0.09 | 38.9 | 2.40 | 4.82 | 4.82 / 8 | 1.0 |
| 7 cube med 0.1 | 0.07 | 34.1 | 2.49 | 4.96 | 4.21 / 7 | 1.0 |
| 8 cube low 0.1 | 0.07 | 33.9 | 2.50 | 4.99 | 4.15 / 7 | 0.9997 |
| 9 ball high 1e-5 | 0.08 | 27.3 | 2.26 | 4.52 | 3.70 / 7 | 1.0 |
| 10 ball med 1e-5 | 0.06 | 31.5 | 2.28 | 4.54 | 4.23 / 7 | 0.9995 |
| 11 ball low 1e-5 | 0.06 | 30.8 | 2.27 | 4.50 | 4.17 / 7 | 1.0 |
| 12 ball high 1e-3 | 0.07 | 24.8 | 2.15 | 4.28 | 3.53 / 6 | 1.0 |
| 13 ball med 1e-3 | 0.06 | 30.0 | 2.27 | 4.50 | 4.07 / 7 | 1.0 |
| 14 ball low 1e-3 | 0.05 | 27.8 | 1.96 | 3.88 | 4.34 / 7 | 1.0 |

Note the zero-gap planner (also M5) costs ~2.3 s vs the used planner's ~0.07 s, and one optimal
iteration costs ~3x a zero-gap plan (alone: 3.7 vs 1.2 s; 1024 vs 256 samples), so M5 on the eval
model is the expensive part throughout.

## `check_steps_analysed.py --every 2 4` (10 analysed cells, 2580 steps per cell at k = 1)

| metric | k | max \|shift\| | width ratio | lag-1 |
| --- | --- | --- | --- | --- |
| kl_used_opt | 2 | 0.416 | 1.037 | 0.19 |
| kl_used_opt | 4 | 0.474 | 1.024 | 0.08 |
| kl_opt_used | 2 | 0.413 | 1.043 | 0.19 |
| kl_opt_used | 4 | 0.465 | 1.021 | 0.13 |
| regret | 2 | 0.486 | 1.124 | 0.10 |
| regret | 4 | 1.015 | 1.279 | 0.02 |
| fsim_obj_pos_err | 2 | 0.694 | 1.052 | 0.09 |
| fsim_obj_pos_err | 4 | 0.646 | 1.067 | 0.01 |
| fsim_obj_rot_err | 2 | 0.260 | 0.987 | 0.13 |
| fsim_obj_rot_err | 4 | 0.663 | 1.017 | 0.04 |
| eval_used_vs_opt_obj_pos_err | 2 | 0.328 | 1.038 | 0.20 |
| eval_used_vs_opt_obj_pos_err | 4 | 0.537 | 1.144 | 0.15 |

Lag-1 at k = 1: 0.11–0.21. Full table: `steps_analysed_check.csv`.

## Open questions for offline work

- Why ~3.7 s per optimal iteration? Profile one optimal plan (1024 worlds of M5 on the eval model):
  kernel time vs Python/launch overhead vs host-device copies, and whether the solver is iterating far
  (the logs are full of `Warning: opt.ccd_iterations, currently set to 35, needs to be increased.`,
  i.e. the collision solver hits its own cap).
- Since one shard already saturates the GPU, `SHARDS_PER_GPU=1` gives the same throughput with half
  the wall time per shard; 2 only helps when GPUs are scarce.
- Options the user has not yet decided on: fewer optimal samples, planning several steps per rollout,
  a cheaper optimal-planner timestep.
- Minor: the optimal planner's log line repeats the flags (defaults + `OPTIMAL_ARGS`); argparse takes the
  last, values are identical, so harmless.

## Zip contents (`results/speedup_handoff_11_2_26.zip`)

- `HPC_ANALYSIS_SPEEDUP_RESULTS.md` (this file), `HPC_ANALYSIS_SPEEDUP.md`, `process_batch_results_fix.diff`
- `batch/`: the batch folder without raw episodes: `analysis/` (per-episode `.npz`, per-cell
  `analysis.npz`, status JSONs), `final_results.csv/json`, `final_episodes.csv`,
  `steps_analysed_check.csv`, every `cell_*.json` and `cell_*.status.json`
- `batch/`: raw episode `.npy` files for cell 5 only (`cell_0005_cube_M4_M5_low_T_1e-3_*.npy`, 23 files,
  61 MB), enough to re-run and profile the analysis of one cell offline. All 393 raw episodes are 928 MB
  and were left on the HPC.
- `logs/`: all task, shard and summary logs of job 31720242 / 31720243
