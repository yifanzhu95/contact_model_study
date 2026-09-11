# Offline KL analysis for recorded trajectory cells

This workflow compares the acting MPPI distributions already stored in completed synchronous trajectory-cell JSON files with a newly evaluated higher-compute MPPI reference. It never reruns the acting controller or changes the recorded task outcome.

At each of up to 10 evenly spaced recorded planner states, the reference control-sequence mean starts at zero. The reference retains its updated mean only between optimization iterations within that one solve; the final mean is discarded before the next state. Defaults are 4096 reference samples, T=50, a maximum of 25 iterations, a squared-L2 first-action stopping threshold of 1e-3, and covariance shrinkage 0.001.

## Inputs

The source directory contains one recorded `cell_*.json` file per configuration. The input manifest contains a `cells` list. Each entry must provide:

- `file`: source JSON basename;
- `sha256`: exact source-file hash;
- `model`: contact-model label;
- `geometry`: recorded scene-variant label;
- `config`: recorded acting-planner configuration;
- `n`: source episode count;
- `success`: source success count.

Generate this manifest directly from a directory of project-produced cell
files; no report-specific preprocessing is required:

```bash
python analysis/build_offline_recorded_kl_manifest.py \
  --source-dir /path/to/source_cells \
  --out /path/to/data_manifest.json
```

The builder checks the supported synchronous `grasp_reorient`/MPPI protocol,
episode outcomes, and the trajectory/context/state/distribution fields required
by offline replay before any GPU work is scheduled. It also rejects empty cells
and duplicate configuration identities.

## Run or resume the episode batch

```bash
python analysis/run_offline_recorded_kl_batch.py \
  --source-dir /path/to/source_cells \
  --manifest /path/to/data_manifest.json \
  --outdir /path/to/new_kl_batch \
  --workers 1
```

When all manifest cells contain the same number of episodes, the default target is every episode in every cell. If counts differ, set `--episodes-per-cell` explicitly to choose a balanced prefix. Episode indices are scheduled in balanced order: episode 0 for all cells, then episode 1 for all cells, and so on. `--max-new-episodes` and `--launch-budget-seconds` can further bound a run.

Each completed episode is an atomic `*_zero_restart_kl.json` file. Running the same command again validates and reuses compatible completed episodes. An interrupted episode is archived and restarted from its first selected state because a JSON checkpoint does not contain the complete GPU random/device state. Cached outputs from a different worker, common analysis module, scientific workflow digest, source hash, or reference preset are rejected instead of mixed. The workflow digest covers the planner, KL implementation, task/goal reconstruction, contact-model API/config/local backend, installed rollout-backend Python sources, and resolved MJCF entrypoint/includes/file-backed assets.

Multiple workers share the selected CUDA device. More workers do not imply linear scaling on one GPU and can exceed GPU memory; profile a conservative worker count on the target machine.

## Plot the largest balanced completed prefix

```bash
python analysis/plot_offline_recorded_kl.py \
  --source-dir /path/to/source_cells \
  --manifest /path/to/data_manifest.json \
  --batch-dir /path/to/new_kl_batch \
  --outdir /path/to/new_plot_export
```

The plotter includes only the largest episode prefix complete in every manifest cell. `--episodes-per-cell N` can request a smaller prefix. It validates source and worker hashes, original outcomes, state membership, exact zero resets, fresh-noise counters, unchanged acting moments, covariance regularization, convergence flags, and both Gaussian KL directions before plotting. Worker and plot exports record the repository commit, dirty-worktree status, and exact analysis-file hashes needed to interpret an uncommitted diagnostic run.

The combined figure contains:

- all finite valid KL measurements;
- converged-reference measurements only.

Both panels first average within each episode and then give each eligible episode equal weight. Success rate always uses every original outcome in the corresponding source cell. The export includes PNG, SVG, a validation JSON, a short README, and a byte-verified ZIP archive. Shared metadata uses portable input names and content hashes rather than machine-specific absolute paths.

## Interpretation limits

The higher-compute reference is not a certificate of global optimality. Converged-only filtering may remove difficult states non-randomly. Acting temperature and visited trajectory can differ between configurations. The goal is reconstructed from the recorded seed flow because it is not explicitly serialized in the recorded logs. Gaussian KL summarizes the first action distribution, not the full planned trajectory distribution.
