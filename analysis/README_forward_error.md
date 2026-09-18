# One-step forward-error analysis

## Purpose

The forward-error tools compare a rollout contact model (M1--M4) with a
Pinocchio reference over one recorded control interval. They operate offline on
trajectory JSON files produced by the contact-model study and relate the
resulting local prediction error to the source cell's episode success rate.

The primary metric is the unweighted sum of object-position L2 error in metres
and object-orientation SO(3) geodesic error in radians. Both components, object
velocity errors, hand-state diagnostics, aligned traces, and Pinocchio solver
diagnostics are retained separately.

## Input contract

Each input must be a per-cell JSON produced with trajectory recording enabled.
The current implementation requires:

- a synchronous (`sync`) driver;
- a source model named M1, M2, M3, or M4;
- a task and geometry registered by `contact_study.tasks`;
- per-step `qpos`, `qvel`, `action`, `ctrl`, and `t` arrays;
- timing and settling metadata stored in `trajectory.context`;
- the Leap-hand layout used by the current manipulation tasks: hand joints
  followed by one free-joint object.

`history_aligned`, the default Pinocchio reference mode, is intended for source
episodes recorded with Pinocchio evaluation. It preserves within-episode
contact-solver history while teacher-forcing the recorded physical state before
each transition. `fresh` constructs a new Pinocchio instance at every sampled
transition and is mainly useful for sensitivity tests or data not originally
recorded with a continuous Pinocchio evaluator.

Old result files without complete recorded trajectories and asynchronous
trajectory logs are not accepted. The program fails with an explanatory error
instead of silently applying a mismatched interpretation.

## Batch analysis

Run from the repository root. A shell wildcard can pass every cell in one
source run to the same analysis invocation:

```bash
python analysis/run_forward_error_cells.py \
  results/SOURCE_RUN/cell_*.json \
  --outdir results/SOURCE_RUN_forward_error \
  --stride 10
```

`--stride 10` selects control transitions 0, 10, 20, and so on in every
episode. The final state row is never selected because it has no following
transition. For an inexpensive pipeline check, add
`--max-samples-per-cell 1`; the limited samples are spread across each cell's
eligible transitions.

The batch command writes:

- one `*_forward_error.json` for each source cell;
- one `forward_error_summary.csv` containing cell-level statistics;
- step-weighted and episode-balanced summaries;
- per-episode summaries and sample-level aligned traces.

Each output records the repository commit, a repository-relative source path,
and the source file's SHA-256 checksum. This allows a reported value to be
traced to the exact input without embedding a contributor's local directory.

## Plotting

Step-weighted view:

```bash
python analysis/plot_forward_error_dir.py \
  results/SOURCE_RUN_forward_error \
  --aggregation step \
  --xerr sd \
  --out results/SOURCE_RUN_forward_error/error_vs_success_step.pdf
```

Episode-balanced view:

```bash
python analysis/plot_forward_error_dir.py \
  results/SOURCE_RUN_forward_error \
  --aggregation episode \
  --xerr sd \
  --out results/SOURCE_RUN_forward_error/error_vs_success_episode.pdf
```

Step weighting describes all sampled local transitions but gives longer
episodes more influence. Episode balancing first averages error within each
episode and then gives episode means equal weight. Both views are useful, but
the episode-balanced view avoids automatically overweighting long timeout
episodes.

## Single-sample diagnostic

`one_step_forward_error.py` performs a more expensive two-run repeatability
check for one selected transition:

```bash
python analysis/one_step_forward_error.py \
  --input results/SOURCE_RUN/cell_00000.json \
  --episode 0 \
  --step 0 \
  --tested-model M1
```

This diagnostic uses a fresh Pinocchio reference and is intended for checking
state extraction, control semantics, timing, quaternion error, and backend
repeatability. The batch program with history-aligned Pinocchio is the primary
analysis path.

## Interpretation limits

- The combined primary metric mixes metres and radians and is descriptive, not
  dimensionless. Position and orientation components should remain visible.
- Transitions sampled from the same episode are correlated and are not
  independent experimental trials.
- Each model's recorded controller trajectory is on-policy, so different
  models can visit different states. A matched-state comparison is a separate
  counterfactual analysis.
- Pinocchio nonconverged substeps are retained as reference-quality diagnostics
  and should be reported alongside the error result.
- Success-rate conclusions require enough independent episode outcomes; a
  small local run validates the pipeline but does not establish model ranking.
