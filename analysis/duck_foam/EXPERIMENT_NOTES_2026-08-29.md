# Duck FOAM local experiment notes — 2026-08-29

## Scope

These runs establish a reproducible workflow on an RTX 5070 Ti. They do not
reproduce the FOAM paper's 5090 wall-clock numbers and do not yet establish a
Duck task-success ranking. The measurements were collected from project
revision `064cc3c`; later publication of the code does not change that provenance.

The controlled comparison changes only the Duck collision model used by GPU
rollouts. Every closed-loop episode evaluates in the same
`scenes/leap/env_leap_eval_duck.xml` reference world, which contains the
original eight convex collision hulls.

## Geometry candidates

| Selector | Collision model | Calibrated sampled protrusion mean / P95 / max |
|---|---|---:|
| `duck_low_foam4` | 4 spheres | 7.437 / 13.672 / 15.071 mm |
| `duck_low_foam16a` | 16 spheres, reduced-distortion repair | 4.090 / 7.070 / 10.738 mm |
| `duck_low_foam16b` | 16 spheres, default repair | 4.805 / 7.693 / 9.449 mm |
| `duck_low_foam64` | 64 spheres | 2.467 / 4.059 / 5.360 mm |

All candidates passed the calibrated coverage rule. Object mass, inertia,
friction, visual mesh, initial pose and reference evaluator remain fixed.

## Fixed-state computation result

Authoritative file: `results/mppi_static_n256_repeat20_5070ti.json`.

Protocol: M2, 256 parallel samples, five-step horizon, 16 rollout substeps per
control step, three discarded warm-up plans and 20 timed plans at one fixed
state. This isolates geometry cost from different closed-loop trajectories.

| Rollout geometry | Median plan time | Approx. speedup vs 8 hulls |
|---|---:|---:|
| 8-hull baseline | 70.03 ms | 1.00× |
| FOAM 4 | 41.79 ms | 1.68× |
| FOAM 16A | 45.71 ms | 1.53× |
| FOAM 16B | 47.48 ms | 1.47× |
| FOAM 64 | 60.83 ms | 1.15× |

The wider 64–1024 sample capacity sweep is in
`results/mppi_static_capacity_5070ti.json`. All 25 cells completed on the 16 GB
5070 Ti. These are local relative comparisons; absolute 5090 times require a
separate run on that GPU.

## Closed-loop diagnostic result

Authoritative held-out diagnostic:
`results/duck_geometry_heldout_diagnostic_n5_s300.json`.

Protocol: five seeds not used during manual development, M2, 256 samples,
temperature 20, noise sigma 0.15, original Duck cost weights, two discarded
warm-up plans, at most 300 control steps. The parameter set was frozen after a
small baseline-only development sweep.

| Rollout geometry | End reasons | Weighted mean plan time | Median final pos / quat / vel error |
|---|---|---:|---:|
| 8-hull baseline | 5 timeout | 101.0 ms | 0.0418 / 0.5523 / 0.202 |
| FOAM 4 | 5 timeout | 50.1 ms | 0.0493 / 0.7855 / 0.007 |
| FOAM 16A | 4 timeout, 1 drop | 60.7 ms | 0.0517 / 0.5362 / 0.064 |
| FOAM 16B | 4 timeout, 1 drop | 58.0 ms | 0.0384 / 0.6856 / 0.105 |
| FOAM 64 | 3 timeout, 2 drop | 95.1 ms | 0.0499 / 0.6914 / 0.494 |

No geometry succeeded. `timeout` here means the Duck remained in the task for
300 steps without satisfying all three strict thresholds; it does not mean
success. Therefore this table may be used to discuss runtime, drop behavior and
continuous diagnostic errors, but not task success ranking.

## Controller and repeatability diagnostics

- With no planner and the fixed initial hand command, the reference Duck grasp
  remained stable for 1,000 control steps (64 simulated seconds). The initial
  grasp itself is therefore not the source of MPPI drops. Reproducible tool:
  `check_duck_hold_stability.py`.
- Temperature 1 caused frequent drops. The Cube-derived temperature 7.187 also
  dropped all three development episodes. Temperature 20 stabilized all three
  at 300 steps but did not solve orientation.
- At temperature 20, raising noise sigma to 0.4 dropped all three episodes;
  lowering it to 0.15 preserved all three and improved continuous position and
  orientation error, but still produced no success.
- Multiplying the Duck continuous and terminal quaternion weights by 2 or 4
  made the grasp less stable and did not improve aggregate orientation error.
  Further hand tuning was stopped to avoid overfitting three development seeds.
- Exact same-seed repeatability is not guaranteed. Five repeats of the same
  baseline configuration/seed produced four 300-step timeouts and one drop at
  step 245, with materially different final goal errors. Record:
  `results/duck_baseline_repeatability_same_seed_r5.json`.

The likely mechanism is that parallel floating-point/contact reductions are not
bitwise deterministic and the closed-loop contact system amplifies tiny numeric
differences. A formal experiment must therefore include more than one random
seed and, ideally, repeated executions nested within seeds. Seed pairing alone
does not remove this variation.

## Required next gate

Do not spend the final test seeds until the Duck controller achieves nonzero
success on separate development seeds. Use a predeclared, equal tuning budget
per geometry (or tune one shared controller across all geometries), then freeze
all cost/planner parameters. A publication-oriented test should use at least:

1. disjoint development and test seeds;
2. repeated executions within a subset of seeds to quantify numeric variance;
3. success rate with confidence intervals plus drop/timeout counts;
4. continuous position, quaternion and velocity errors;
5. fixed-state latency/throughput and closed-loop latency reported separately;
6. the same reference evaluator, capacities, software versions and warm-up rule.

Until that gate is met, the strongest supported project result is the local
speed/fidelity tradeoff: FOAM 4/16 substantially reduce planning latency on the
5070 Ti, while finer sampled geometric agreement does not automatically imply
better closed-loop stability.
