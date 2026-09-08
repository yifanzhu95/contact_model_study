# Cube high/high plotting example

This is a small integration pilot: M3, cube high/high, 16 x 1 and 256 x 1
degraded compute, each with a separate real/null cell and three episodes.
The real reference uses 4096 samples and four iterations. Episodes retain the
normal 1000-command maximum. Root seed is 20260907; the full scientific
configuration and per-episode seeds are in each JSON.

The PNG/PDF illustrate the workflow. Three episodes per cell do not establish
a configuration ranking or a KL-success relationship. Null is a separate
closed-loop run, not a state-matched noise correction. Horizontal whiskers
are between-episode KL standard errors; vertical whiskers are Wilson 95%
success intervals.

| Degraded budget | Real successes | Independent null successes | Real mean KL | Null mean KL |
|---|---:|---:|---:|---:|
| 16 x 1 | 0/3 | 0/3 | 3449.58 | 2071.95 |
| 256 x 1 | 1/3 | 2/3 | 863.41 | 26.90 |

All KL means here weight episodes equally. There were 459 valid measured
transitions and no invalid KL records. The `sensitivity/` figure and summary
were computed on the complete local raw moments. Their substantial change
with alpha is part of the interpretation, not a retuning of the experiment.

The four compact cell JSONs preserve scalar per-step KL/ESS, episode outcomes,
seeds, timings and scientific settings. Only raw covariance/mean blocks were
omitted to keep the example small. Each file records its original source
filename and SHA-256 digest. These files reproduce the KL-vs-SR aggregation
without running a simulator or requiring GPU access. They are copies of the
same experiments, not additional independent observations: do not merge them
with their original raw files.

From the repository root with NumPy and Matplotlib available:

```bash
python -m analysis.plot_kl_divergence_dir \
    analysis/examples/kl_cube_high_high_20260907 \
    --title_note 'Integration pilot only: 3 episodes per cell' \
    --out /tmp/kl_cube_high_high_pilot.pdf
```

For fresh experiments and fixed-particle shrinkage analysis, follow
[`analysis/README_kl_divergence.md`](../../README_kl_divergence.md).
The compact plotting JSONs cannot reconstruct the omitted covariance blocks.
The validation report describes the full local-moment analysis and the
repeatability limits: [`kl_delivery_validation_20260907.md`](../../kl_delivery_validation_20260907.md).
