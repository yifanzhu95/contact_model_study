# High/high KL workflow compatibility validation

Date: 2026-09-06. Branch: `kl_divergence_analysis`.

Status: local implementation and short execution checks completed. No commit,
push, cluster submission, or full success-rate sweep was performed in this
validation. This report does not establish a KL-versus-success relationship.

## Protocol implemented

- Supported objects: cube, duck, ball, spam and tomato.
- Both planners use the selected object's project-defined `high_high` rollout
  scene. The fixed per-object evaluation scene is retained unchanged.
- Object shorthand selects high/high; legacy aliases and lower fidelities are
  rejected by the KL grasp-reorient worker only.
- Null remains an independent closed-loop run with equal planner compute and
  separate sampling. Its states need not match the real run.
- `ccd_iterations=35` is unchanged. Raw collision warnings are retained.
- New result records identify geometry, object, actual evaluation simulator,
  realized timing, cost/contact settings, goal difficulty, reference comparison
  budget and episode seeds. UUID filenames avoid cross-object/rerun overwrites.
- Plotting separates incompatible scientific settings and objects, pairs only
  compatible nulls, and rejects duplicate episode seed identities. Missing
  legacy geometry is labelled unknown, never assumed to be high/high.
- A single valid episode has no estimated between-episode KL standard error
  and cannot pass the default mean-based null screening rule. The console
  reports this as unavailable, not as zero uncertainty.

No changes were made to the shared scene assets, task goal generation, control
conversion, physical constants, MPPI optimizer implementation or collision
iteration limit. The legacy Drake hand-only path is rejected for this
all-object KL workflow rather than being presented as per-object support.
Pinocchio was used for the actual execution checks below.

## Asset and interface checks

All ten required MuJoCo scene files (five high/high rollout scenes and five
evaluation scenes) load successfully. Each has `nq=23`, `nv=22`, `nu=16`, object
position/velocity offsets of 16, consistent actuator order and valid fingertip
sites. All report the unchanged collision iteration limit of 35.

CPU construction of the task's Pinocchio simulator and reset/readback checks
also passed for all five objects. Maximum qpos reset discrepancy was below
`4.3e-10`; velocity readback discrepancy was zero. These CPU checks set up
scene metadata directly and do not substitute for GPU planner execution.

The high/high rollout hand and evaluation hand are different project assets;
this validation did not attempt to make them geometrically identical. The
common difficulty-1 setting rotates the canonical target, not each object's
settled pose, so it does not imply equal task difficulty across objects.

## Short all-object execution check

Settings: M3, Pinocchio evaluation, degraded `16 samples x 1 iteration`,
reference `64 samples x 1 iteration`, one episode/cell, two control steps,
KL every step, root seed `20260906`, settle 1 s, goal difficulty 1,
temperature 1.0, noise sigma 0.025, shrinkage 0.001. Null uses `16 x 1` on
both sides while retaining the intended `64 x 1` comparison budget in metadata.

The requested horizon of 0.352 s resolves to five control intervals of 0.064 s,
or 0.320 s. Rollout dt is 0.004 s and evaluation dt is 0.0005 s.

| Object | Real run | Independent null | Valid KL measurements | Invalid KL |
|---|---|---|---:|---:|
| cube | Completed | Completed | 2 + 2 | 0 |
| duck | Completed | Completed | 2 + 2 | 0 |
| ball | Completed | Completed | 2 + 2 | 0 |
| spam | Completed | Completed | 2 + 2 | 0 |
| tomato | Completed | Completed | 2 + 2 | 0 |

Ten self-describing JSONs were produced and correctly grouped into five
separate object/configuration figures, each with its own null. Each run ended
in timeout at the deliberately imposed two-step limit. This is a wiring
check, not evidence that a normal-duration task would fail.

Local data and logs:
`results/kl_high_high_smoke_20260906_M54e4C/`.
The figures are named `smoke_only_<geometry>_<configuration_id>.png`.

## Intended reference-budget spot check

An additional real/null pair used `duck_high_high`, a `256 x 1` degraded
planner, and the intended `4096 x 4` high-compute reference. The null used
`256 x 1` on both sides. Other shared settings matched the short check above.

Both two-control-step runs completed, each with two finite KL comparisons and
zero invalid measurements. The high-compute reference averaged approximately
584 ms per measured call in this short check. Only two calls were measured;
this is not a stable runtime benchmark or a cluster resource estimate.

Local data, logs and figure:
`results/kl_high_high_reference_check_20260906_u0D7IJ/`.

In total, the twelve short runs produced 24 valid KL measurements. No full
episode success-rate estimate is reported from these truncated runs.

## Automated and submission-template checks

The following CPU suite passed all 61 tests:

```bash
python -m pytest -q \
    tests/test_kl_analysis.py \
    tests/test_kl_worker_config.py \
    tests/test_kl_plot_grouping.py \
    tests/test_kl_scene_assets.py
```

It covers arithmetic, geometry selection, invalid-input rejection, output
identity, scene interfaces, incompatible-setting isolation, duplicate
protection, legacy compatibility and actual plotting CLI output.

The shell template passes `bash -n`. Dry-run validation covers all 80 unique
cells, both endpoints, invalid-index rejection and refusal to accidentally
run locally without an array index. The 80-cell template represents 2400
episodes at its default 30 episodes/cell; none were submitted.

## Remaining interpretation limits

- Independent null runs are not a calibrated matched-state noise floor.
- Repeated-run divergence observed in the earlier pilot remains unexplained;
  no repeatability guarantee is claimed here.
- GPU MuJoCo-Warp convex GJK/EPA warnings still occur. Keeping the iteration
  limit at 35 preserves the requested setting but does not establish collision
  convergence. This warning is not the Pinocchio ADMM warning.
- Small-sample Gaussian covariance, ESS and shrinkage remain relevant to KL
  interpretation. Sixteen first-action samples in sixteen dimensions have
  rank-deficient raw covariance before shrinkage.
- Long-duration performance, all-contact-model coverage and statistical
  KL-success relationships were not validated by these M3 compatibility runs.
- Raw logs/results are local ignored artifacts, not automatically included in
  a future code commit. The English workflow guide contains reproduction steps.

Next delivery step: review the code diff and reproduction instructions, then
commit/push only when requested. A larger scientific sweep is a separate step.
