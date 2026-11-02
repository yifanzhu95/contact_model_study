# Speeding up `process_batch_results` on the HPC: what changed and what to check

You are working on the HPC copy of the `contact_model_study` repo (default
`PROJ_DIR=/home/kg878/project_pi_amd78/kg878/Projects/contact_model_study`,
conda env `contact_kamino`). Run everything from the repo root.

## Background

`experiments/process_batch_results.py` analyses an episode-batch results folder.
At every analysed step it plans three times (the used, optimal and zero-gap
planners) and runs two regret rollouts, one after another, at roughly 1–2 s per
step. The optimal planner (1024 samples, up to 20 convergence iterations, eval
timestep) is probably most of that. Three changes were made to cut the cost:

1. **Several shards per GPU.** Each array task now runs `SHARDS_PER_GPU` shards
   (default **2**) at once on its one GPU, as separate processes. A plan of
   ~1000 worlds leaves the GPU mostly idle; the BO notes 2 planners per GPU ran
   ~1.5× faster than 1.
2. **Timing is recorded.** Every analysed step saves each planner's plan time and
   iteration count, whether the optimal planner converged before its cap, and
   the regret rollouts' time. These appear in `final_results.csv` like any
   metric.
3. **A check for how many steps to analyse.** `experiments/check_steps_analysed.py`
   recomputes an analysed batch's metrics from every k-th analysed step, on the
   CPU, to decide whether `N_STEPS_ANALYSED=100` can be lowered.

None of this changes the analysis results or the key episodes are saved under:
episodes already analysed stay valid and are not redone.

## 0. Preflight

Confirm the changes are on this machine (they may need a `git pull` first):

```bash
ls experiments/check_steps_analysed.py tests/test_check_steps_analysed.py
grep -n SHARDS_PER_GPU experiments/hpc/process_batch_results.slurm experiments/hpc/submit_process_batch_results.sh
grep -n TIMING_KEYS experiments/process_batch_results.py
bash -n experiments/hpc/process_batch_results.slurm && bash -n experiments/hpc/submit_process_batch_results.sh
```

If any of these is missing, stop and report it rather than editing the files.

Optionally run the CPU tests (on the login node or a short CPU job):

```bash
python -m pytest tests/test_process_batch_results.py tests/test_check_steps_analysed.py -m "not slow and not gpu" -q
```

Six failures in `tests/test_bayes_opt.py` (`temperature_M2=1e-05 is outside the
search range`) are known, predate these changes, and are unrelated. Do not try to
fix them.

## 1. Choose `N_STEPS_ANALYSED` (CPU only, needs an analysed batch)

Find a batch folder that has already been analysed (it has
`analysis/<cell>.analysis.npz` files), then:

```bash
python experiments/check_steps_analysed.py results/<analysed batch dir> --every 2 4
```

It prints, per metric and k, `max |shift|`, `width ratio` and `lag-1`, and writes
the full table to `<batch dir>/steps_analysed_check.csv`. What the columns mean:

- **`shift`**: how far the cell's mean moved from using every analysed step
  (k = 1), measured in half-widths of k = 1's 95% interval over episodes.
- **`width ratio`**: how much that interval widened.
- **`lag-1`**: how correlated neighbouring kept steps still are (near 1 means
  they are redundant).

Decision rule:

- **k = 2 has `max |shift|` < 0.25 and `width ratio` < ~1.1 for every metric:**
  `N_STEPS_ANALYSED=50` is safe and halves the cost. If k = 4 passes as well,
  25 is safe too.
- **Otherwise:** keep 100.

Do **not** edit `N_STEPS_ANALYSED` in `experiments/hpc/process_batch_results.slurm`
yourself. Report the numbers and the recommendation. Changing it mid-batch makes
the episodes already saved under 100 steps get redone.

If no analysed batch exists yet, skip this step and come back to it after step 2.

## 2. Run the analysis with two shards per GPU

Submit as before. `SHARDS_PER_GPU` defaults to 2:

```bash
experiments/hpc/submit_process_batch_results.sh results/<batch dir> [max concurrent GPUs]
```

The submit script prints the array size, now cells × `N_SHARDS` (5) / 2 tasks.
Each task requests 8 CPUs and 32 GB. The optional second argument caps concurrent
tasks, i.e. GPUs. `SHARDS_PER_GPU=1 experiments/hpc/submit_process_batch_results.sh …`
restores one shard per GPU.

Logs:

- `logs/process_batch_results_<job>_<task>.out`: the task's own log, with the
  `nvidia-smi` line, which (cell, shard) pairs it runs, and each shard's done or
  FAILED.
- `logs/process_batch_results_<job>_<task>_<j>.out`: each shard's own output,
  one per shard of the task.

While the first tasks run, check:

1. **GPU memory.** The task log prints the GPU's total memory. Run `ssh <node>
   nvidia-smi` (or `srun --jobid <job> --overlap nvidia-smi`) to confirm both
   processes fit with headroom. If a shard dies with a CUDA out-of-memory error,
   resubmit with `SHARDS_PER_GPU=1`. Resubmitting resumes where it stopped.
2. **Per-step time.** Each episode's line in a shard log now ends with, e.g.,
   `per step: plans used 0.12 / opt 1.40 / zero 0.20 s, regret 0.15 s, opt 11.3 iterations, converged 85%; forward sim 0.4 s`.
   Compare the total per analysed step against older logs that ran one shard
   per GPU. With 2 shards per GPU each shard may be a bit slower; the win is that
   two run per GPU. Throughput per GPU should be ~1.3–1.8× the old.
3. **Failures.** If a shard FAILED, read its `_<j>.out` traceback. Report
   anything that is not out-of-memory or the time limit.

Do **not**:

- pass `--overwrite` (it redoes finished cells);
- pass `--max-episodes` on a real batch folder (the cell is then marked done with
  only those episodes);
- change `OPTIMAL_ARGS`.

## 3. Read the timing in the results

When the summary job has written `<batch dir>/final_results.csv`, pull these
columns per cell (all new):

| Column | Meaning |
| --- | --- |
| `plan_s_used_mean`, `plan_s_opt_mean`, `plan_s_zero_mean` | seconds per plan, each planner |
| `regret_s_mean` | seconds for the two regret rollouts per step |
| `n_iter_opt_mean`, `n_iter_opt_p90` | optimal planner's iterations per plan |
| `converged_opt_mean` | fraction of optimal plans that converged before the 20-iteration cap |

For example:

```bash
python - results/<batch dir>/final_results.csv <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
cols = ["name", "plan_s_used_mean", "plan_s_opt_mean", "plan_s_zero_mean", "regret_s_mean",
        "n_iter_opt_mean", "n_iter_opt_p90", "converged_opt_mean"]
print("\t".join(cols))
for r in rows:
    print("\t".join(r.get(c, "") for c in cols))
EOF
```

How to read it:

- **`plan_s_opt_mean` dominates and `converged_opt_mean` is low (many plans hit
  the cap, `n_iter_opt_p90` ≈ 20):** the 20-iteration cap is what costs. Report
  this; lowering `--max-iterations` in `OPTIMAL_ARGS` is a separate decision for
  the user.
- **The optimal planner mostly converges early:** the cap is not the cost. The
  next speedup would be planning several steps per rollout, a larger code
  change for the user to decide on.

Cells analysed before this change show these columns empty or NaN. Only newly
analysed episodes have timing.

## 4. Report back

Reply with:

1. Preflight: files present, tests run (pass/fail counts).
2. Step 1: the printed `check_steps_analysed.py` table (or "no analysed batch
   yet") and your `N_STEPS_ANALYSED` recommendation, per the rule above.
3. Step 2: job id, `SHARDS_PER_GPU`, GPU model and memory used per task, any
   failed shards and why, and the per-step time vs the old one-shard-per-GPU
   logs, if you could find them.
4. Step 3: the timing table and which of the two cases above it falls into.

Change no code or settings. Report back and let the user decide.
