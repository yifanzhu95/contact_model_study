# Duck FOAM geometry study

This directory records exploratory and final spherical approximations of the Duck collision
geometry. The current files are an **exploration baseline**, not the final Low/Medium/High model
set.

## Geometry roles

- `scenes/leap/objects/rubber_duck.stl` is the provisional canonical source and the end-to-end
  geometry-analysis reference. FOAM may fit a repaired derivative rather than this exact surface.
- FOAM sphere levels are candidate rollout collision models.
- The existing eight convex hulls remain the fixed project evaluation/reference collision model
  unless the experiment owner explicitly changes that definition.
- The STL visual geometry and the collision geometry are separate in MuJoCo.

## Current baseline

`exploration/duck_branch8_depth2.json` was generated with upstream FOAM commit
`116928f71aaa7c40356d79c84d3c9ff1f4497d90` using:

```bash
pixi run python scripts/generate_spheres.py /tmp/rubber_duck.stl \
  --output=/tmp/duck_foam_depth2.json --depth=2
```

The result contains alternative sphere-tree levels, not cumulative geometry:

| Depth | Active spheres | FOAM mean | FOAM best | FOAM worst |
|---:|---:|---:|---:|---:|
| 0 | 1 | 716.618 | 716.618 | 716.618 |
| 1 | 8 | 111.036 | 87.722 | 142.207 |
| 2 | 64 | 17.099 | 0.000 | 32.793 |

Selecting depth 2 means using 64 spheres, not `1 + 8 + 64` spheres.

## Important implementation findings

1. The raw STL loaded without vertex welding has 32,880 vertices and 10,960 disconnected
   triangle components. It fails the watertight check. Loading the same surface while only
   processing/welding duplicate vertices gives 5,482 vertices, one component, consistent winding,
   and a watertight volume without changing bounds, area, or volume. However, SphereTree's legacy
   verifier still reports 76 intersecting `Bad Faces`, so vertex welding alone is not a safe way to
   bypass repair. Because FOAM runs this verification before generation, the baseline run entered
   FOAM's repair path.
2. Replaying the wrapper's validation path showed that manifold reconstruction, simplification
   with ratio `0.2`, and 100 Humphrey smoothing iterations produced a valid mesh with 374 vertices
   and 744 faces. The second validation succeeded, so convex-decomposition fallback did not run.
   The repaired bounds are larger than the welded raw STL bounds; preprocessing is therefore a
   material part of the end-to-end geometry error, not merely a topology repair.
3. `merge=true` and `expand=true` were forwarded to the medial SphereTree generator;
   `burst=false`.
4. Although the Python command exposes `optimise=true`, the current wrapper only forwards the
   optional simplex optimiser for non-medial methods. The effective optional optimiser for this
   medial run was therefore `none`; `maxOptLevel` and `balExcess` did not activate one.
5. The Python `--eval` argument is not forwarded. A same-name C++ evaluator pointer shadows the
   intended Boolean switch, so fitting statistics are still produced. This FOAM evaluator is
   unrelated to the project's `env_leap_eval_duck.xml` evaluation world.

The machine-readable record is in `exploration/duck_branch8_depth2.metadata.json`.

`capture_foam_preprocessed_mesh.py` reproduces FOAM's preprocessing decision path without patching
the external FOAM checkout. Its captured output lets us separate:

- sphere fitting error relative to the surface FOAM actually processed; and
- preprocessing plus sphere fitting error relative to the original Duck surface.

Reproduce the captured preprocessing target with:

```bash
FOAM_ROOT=/path/to/foam
"$FOAM_ROOT/.pixi/envs/default/bin/python" \
  analysis/duck_foam/capture_foam_preprocessed_mesh.py \
  --foam-root "$FOAM_ROOT" \
  --input scenes/leap/objects/rubber_duck.stl \
  --output analysis/duck_foam/exploration/rubber_duck_foam_preprocessed.obj \
  --metadata \
    analysis/duck_foam/exploration/rubber_duck_foam_preprocessed.metadata.json
```

The captured OBJ is an analysis artifact, not a replacement for the project's visual or fixed
evaluation geometry.

## Independent geometry metrics

`evaluate_sphere_approximation.py` measures two different failure directions:

1. **Undercoverage:** uniformly sample the Duck surface and evaluate the analytic signed-distance
   field of the sphere union. A positive value means a Duck surface point lies outside every
   sphere.
2. **Overcoverage:** sample the exposed boundary of the sphere union, find its closest triangle on
   the welded original Duck surface, and use the outward nearest-triangle normal to classify the
   sample as outside or inside. A positive value means the sphere boundary protrudes outside Duck.

This is intentionally independent of FOAM's `mean`, `best`, and `worst` fields. At each tree level,
FOAM samples 42 points on every sphere (`testerLevels=2`), records that sphere's largest sampled
outside distance, and then reports the minimum, maximum, and average across spheres. These fields do
not measure missing Duck coverage. They are also written in FOAM's normalized internal units, not
metres or millimetres.

For this processed Duck, FOAM's C++ `fitIntoBox(1000)` scales the longest mesh extent to 2000
internal units. The scale is `19730.226 internal units/m`, so the JSON depth-2 values convert to:

| FOAM field | JSON value | Approximate physical distance |
|---|---:|---:|
| best | 0.000 | 0.000 mm |
| mean | 17.099 | 0.867 mm |
| worst | 32.793 | 1.662 mm |

These remain sparse per-sphere samples, not guaranteed global minimum/average/maximum distances.

The evaluator uses `trimesh.proximity.closest_point_naive` in small chunks. It is slower than an
R-tree query but requires no unrecorded package installation and computes closest points directly
against triangles.

Run it with the Python environment created by FOAM:

```bash
FOAM_ROOT=/path/to/foam
"$FOAM_ROOT/.pixi/envs/default/bin/python" \
  analysis/duck_foam/evaluate_sphere_approximation.py \
  --mesh scenes/leap/objects/rubber_duck.stl \
  --spheres analysis/duck_foam/exploration/duck_branch8_depth2.json \
  --depth 2 \
  --output analysis/duck_foam/results/duck_branch8_depth2_level2.metrics.json
```

The script treats coordinates as metres and reports distances in millimetres. Sampling counts and
the random seed are stored in every output file.

### Baseline sampled results

These results use seed 0, 20,000 original-Duck surface samples, approximately 8,000 sphere-surface
candidate samples, and a 0.01 mm classification tolerance.

| Depth | Spheres | Duck surface covered | Mean outside protrusion | P95 | Maximum |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 100.00% | 17.915 mm | 33.607 mm | 40.761 mm |
| 1 | 8 | 100.00% | 7.674 mm | 10.908 mm | 12.284 mm |
| 2 | 64 | 100.00% | 4.712 mm | 6.264 mm | 8.399 mm |

A depth-2 repeat with seed 1, 100,000 Duck samples, and approximately 32,000 sphere candidates
gave mean 4.746 mm and maximum 8.510 mm, so the initial sampled estimate is stable. All three
levels are conservative covers of the sampled original surface; increasing sphere count reduces
but does not eliminate protrusion. Part of the remaining protrusion comes from FOAM's expanded,
smoothed preprocessing target.

### Separating fitting error from preprocessing error

Running the same evaluator against the captured FOAM-preprocessed mesh gives:

| Depth | Spheres | Processed surface covered | Mean outside protrusion | P95 | Maximum |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 100.00% | 14.219 mm | 30.119 mm | 37.188 mm |
| 1 | 8 | 99.98% | 3.837 mm | 6.759 mm | 7.860 mm |
| 2 | 64 | 99.65% | 0.869 mm | 1.565 mm | 2.145 mm |

The 64-sphere result therefore has two different, both valid, interpretations:

- **Sphere fitting only:** relative to the surface FOAM actually fitted, sampled mean protrusion is
  about 0.87 mm and maximum protrusion is about 2.14 mm. A small 0.35% of processed-surface samples
  miss the sphere union by more than the 0.01 mm tolerance; the largest sampled gap is 0.332 mm.
- **End to end:** relative to the welded original project STL, sampled mean protrusion is about
  4.71 mm and maximum protrusion is about 8.40 mm, with no uncovered original-surface sample in the
  baseline run.

The difference is evidence that preprocessing materially expands and smooths the target before
sphere generation. It must not be attributed to the 64-sphere fit alone. Conversely, the 100%
original-surface coverage is not automatically good news: here it is partly achieved because the
preprocessed target is larger than the original Duck.

The larger seed-1 repeat gave 99.62% processed-surface coverage, a 0.068 mm mean uncovered gap
among uncovered samples, 0.843 mm mean protrusion, and 2.404 mm maximum sampled protrusion. The
means and coverage estimates are stable; the observed maximum grows slightly when more boundary
points are tested, as expected for a sampled maximum.

As another measure of the preprocessing change, the welded original STL has volume
`0.0002043 m^3` and area `0.02137 m^2`, whereas the captured processed target has volume
`0.0002977 m^3` and area `0.02572 m^2`: increases of about 45.7% and 20.3%, respectively.

## Branch-4 Low/Medium/High candidates

The next exploratory tree was generated with the same FOAM version and parameters except for
`branch=4` and `depth=3`:

```bash
pixi run python scripts/generate_spheres.py \
  <project>/scenes/leap/objects/rubber_duck.stl \
  --output=<project>/analysis/duck_foam/exploration/duck_branch4_depth3.json \
  --depth=3 --branch=4
```

It contains independent levels of 1, 4, 16, and 64 spheres. The following are the higher-sample
seed-1 results (100,000 mesh samples and approximately 32,000 sphere candidates):

| Provisional role | Spheres | Original covered | Original mean/max protrusion | Processed covered | Processed mean/max protrusion |
|---|---:|---:|---:|---:|---:|
| Low | 4 | 100.00% | 9.520 / 18.105 mm | 99.97% | 5.777 / 13.034 mm |
| Medium | 16 | 100.00% | 6.263 / 10.030 mm | 99.92% | 2.409 / 5.000 mm |
| High | 64 | 100.00% | 4.781 / 8.384 mm | 99.77% | 0.892 / 2.371 mm |

The 1-sphere level is retained in the JSON as the tree root but is not a practical Low candidate:
its original-Duck mean/max protrusion is about 17.9/40.8 mm.

At 64 spheres, branch 4 and the earlier branch 8 tree are effectively equivalent at the current
sampling resolution. Branch 4 is the more useful experimental hierarchy because it also provides
4- and 16-sphere levels from the same tree. Low/Medium/High are still **provisional geometry
labels**; final labels require MuJoCo throughput and fixed-world task-success measurements.

## Reduced-distortion preprocessing experiment

`prepare_welded_mesh.py` first demonstrated that coincident-vertex welding alone preserves the
original surface exactly, but the welded surface still fails SphereTree because of 76 intersecting
faces. The welded OBJ is therefore a diagnostic artifact, not a valid FOAM generation target.

ManifoldPlus depth 5 through 9 preserved the original volume and bounds much better, but its
outputs still failed SphereTree's legacy intersecting-face check. High-resolution old-Manifold
outputs and the available simplifiers had the same problem. Disabling verification was rejected
because a medial-axis algorithm should not be run knowingly on a surface that its own verifier
considers unusable.

The best verified compromise found in the local toolchain is produced by
`prepare_foam_safe_mesh.py`:

```bash
FOAM_ROOT=/path/to/foam
"$FOAM_ROOT/.pixi/envs/default/bin/python" \
  analysis/duck_foam/prepare_foam_safe_mesh.py \
  --foam-root "$FOAM_ROOT" \
  --input scenes/leap/objects/rubber_duck.stl \
  --output analysis/duck_foam/exploration/rubber_duck_foam_safe_l6000.obj \
  --metadata analysis/duck_foam/exploration/rubber_duck_foam_safe_l6000.metadata.json \
  --manifold-leaves 6000
```

It merges coincident vertices and runs only FOAM's old Manifold reconstruction at the highest
tested resolution that passes SphereTree. It deliberately skips mesh simplification and all
Humphrey smoothing. Compared with the welded original surface:

| Preprocessing target | Faces | Volume change | Area change | Maximum bounds change |
|---|---:|---:|---:|---:|
| FOAM default repair | 744 | +45.7% | +20.3% | 5.09 mm |
| Reduced-distortion repair | 14,012 | +27.1% | +14.4% | 2.54 mm |

This is a material improvement, although it is not a zero-distortion repair. The resulting
branch-4 tree is `exploration/duck_foam_safe_l6000_branch4_depth3.json`. Generation took about
244 seconds because the fitting target contains far more triangles than the default 744-face
target.

Higher-sample results against the original project STL are:

| Provisional role | Spheres | Original covered | Default-repair mean/max | Reduced-distortion mean/max |
|---|---:|---:|---:|---:|
| Low | 4 | 100.00% | 9.520 / 18.105 mm | 8.988 / 15.928 mm |
| Medium | 16 | 100.00% | 6.263 / 10.030 mm | 5.502 / 11.414 mm |
| High | 64 | 100.00% | 4.781 / 8.384 mm | 3.629 / 6.171 mm |

The reduced-distortion tree improves Low's mean and maximum and improves High's mean and maximum
by about 24% and 26%, respectively. Medium's mean improves by about 12%, but its sampled maximum
becomes about 14% worse. The location and task relevance of that local Medium outlier should be
examined before selecting the final 16-sphere model.

Relative to the reduced-distortion surface that FOAM actually fitted, the new 64-sphere level has
99.99% sampled coverage, about 1.214 mm mean protrusion, and 3.848 mm maximum sampled protrusion.
Its fitting-only error is higher than the default-repair tree because the 14,012-face target
retains more geometric detail, but its end-to-end error against the original Duck is substantially
lower. End-to-end agreement with the project geometry is the relevant geometric comparison.

Visualize the reduced-distortion 64-sphere candidate over the original Duck with:

```bash
cd /path/to/foam
pixi run python scripts/visualize_spheres.py \
  "<project>/scenes/leap/objects/rubber_duck.stl" \
  "<project>/analysis/duck_foam/exploration/duck_foam_safe_l6000_branch4_depth3.json" \
  --depth=3
```

## Global scale calibration

Uniform post-scaling was tested rather than rejected on theoretical grounds. Every sphere center
was scaled about the original Duck bounding-box center, and every radius was multiplied by the
same factor. `sweep_sphere_scale.py` evaluates each scale with common sample points and accepts a
candidate only when sampled coverage is at least 99.9% and the maximum positive gap is no more
than 0.1 mm.

Matching preprocessing-target volume would require linear scales of about `0.8820` for the
default repair and `0.9231` for the reduced-distortion repair. The latter volume-matching scale
produced only 90.17%, 92.05%, and 67.42% original-Duck coverage for the 4-, 16-, and 64-sphere
levels. Equal target volume is therefore not a usable collision-coverage criterion.

A coarse sweep followed by 0.001-scale refinement found boundary values near 0.957, 0.956, and
0.962 for the reduced-distortion 4/16/64 levels. Higher-sample checks used slightly more
conservative scales with no observed uncovered original-surface sample:

| Candidate | Source | Scale | Covered | Mean / P95 / max protrusion |
|---|---|---:|---:|---:|
| Low | reduced-distortion 4 spheres | 0.960 | 100.00% | 7.437 / 13.672 / 15.071 mm |
| Medium-A | reduced-distortion 16 spheres | 0.960 | 100.00% | 4.090 / 7.070 / 10.738 mm |
| Medium-B | default-repair 16 spheres | 0.960 | 100.00% | 4.805 / 7.693 / 9.449 mm |
| High | reduced-distortion 64 spheres | 0.965 | 100.00% | 2.467 / 4.059 / 5.359 mm |

`exploration/duck_foam_safe_l6000_branch4_calibrated.json` materializes Low, Medium-A, and High
in a single FOAM-visualizer-compatible file. Medium-B is retained separately in
`exploration/duck_branch4_medium16_scale096_alternative.json`. Original FOAM mean/best/worst
fields in these materialized files describe the unscaled source and must not be used as calibrated
metrics; the independent result JSON files are authoritative.

Medium-A has better mean and P95 agreement, while Medium-B reduces the sampled maximum. Their
worst samples occur in nearly the same central side region of the Duck, which indicates a stable
local geometric limitation rather than random sample noise. `visualize_error_sample.py` marks the
sphere point red, the closest Duck point green, and their measured distance yellow. For example:

```bash
cd "<project>"
FOAM_ROOT=/path/to/foam
"$FOAM_ROOT/.pixi/envs/default/bin/python" \
  analysis/duck_foam/visualize_error_sample.py \
  --metrics \
    analysis/duck_foam/results/duck_foam_safe_l6000_branch4_calibrated_level2_seed1_high_sample.metrics.json
```

The next MuJoCo integration stage should carry four rollout collision candidates—Low, Medium-A,
Medium-B, and High—while keeping the existing eight-hull evaluation/reference model fixed. This
allows task behavior to resolve the Medium mean-versus-maximum tradeoff.

## Decision rule for Low/Medium/High

Sphere count alone does not define fidelity. A candidate level must be considered using all of:

- undercoverage fraction and maximum uncovered gap;
- overcoverage surface fraction and protrusion distances;
- number of active sphere geoms;
- MuJoCo step/rollout throughput;
- task success in the fixed evaluation/reference world.

The current preferred candidates are the calibrated Low, Medium-A/Medium-B, and High models above.
Those labels should not become final until the runtime and task-success measurements are complete.

## MuJoCo rollout integration

The calibrated candidates are now materialized as MuJoCo sphere geoms. Object accuracy labels map
to collision models as follows:

| Label | Candidate | Sphere count | Source level |
|---|---|---:|---|
| `foam4` | Low | 4 | reduced-distortion, scale 0.960 |
| `foam16a` | Medium-A | 16 | reduced-distortion, scale 0.960 |
| `foam16b` | Medium-B | 16 | default repair, scale 0.960 |
| `foam64` | High | 64 | reduced-distortion, scale 0.965 |

`generate_mujoco_sphere_scenes.py` creates all four candidates for each `low`, `med`, and `high`
hand model. For example, `duck_low_foam16a` resolves to
`scenes/leap/env_leap_rollout_duck_low_foam16a.xml`. The same selector resolves eval to
`scenes/leap/env_leap_eval_duck.xml`, because eval depends only on the first `duck` token.

The generator reads the current rollout template's visual-mesh body-frame offset and applies that
same offset to every sphere center. This keeps the FOAM collision geometry aligned when the base
project changes the Duck body frame (the current `planner_dev` templates use a zero offset). Each
MuJoCo radius is the calibrated JSON radius. The Duck's explicit `0.05 kg` mass, center of mass,
principal axes, diagonal inertia, friction, visual mesh, freejoint, spawn pose, and goal marker are
inherited unchanged from the corresponding template. Sphere `density=0` prevents collision
geometry from changing dynamics; contact remains enabled through MuJoCo's default contype and
conaffinity.

Regenerate and validate the scenes with:

```bash
cd "<project>"
python analysis/duck_foam/generate_mujoco_sphere_scenes.py
python analysis/duck_foam/validate_mujoco_sphere_scenes.py
```

The validator compiles all 12 rollout scenes, checks every sphere center and radius against its
source JSON, checks the fixed object physics, runs `mj_forward`, and verifies that eval still has
exactly eight mesh collision hulls. Reproducibility hashes and mappings are recorded in
`mujoco_sphere_scenes.json`.

Inspect one candidate interactively with:

```bash
python -m mujoco.viewer --mjcf scenes/leap/env_leap_rollout_duck_low_foam16a.xml
```

### Short closed-loop GPU check

Before running the full multi-seed study, compare the four object models with a
small, reproducible MPPI run:

```bash
WARP_CACHE_PATH=/tmp/contact_study_warp_cache \
python analysis/duck_foam/benchmark_mppi_sphere_scenes.py \
    --n_samples 64 --max_steps 30 --nconmax 200 --njmax 1000
```

The script holds the rollout hand at `low`, uses M2 and the same random seed for
all four sphere models, and always evaluates in `env_leap_eval_duck.xml` (the
reference eight-hull Duck).  It performs a one-step warm-up before timing and
writes JSON plus a latency plot to `analysis/duck_foam/results/`.

The larger capacities are intentional.  In the 64-sphere scene the previous
driver defaults (`nconmax=50`, `njmax=300`) overflowed, which discards contacts
and invalidates both dynamics and timing.  The short run is only a functional
and engineering comparison; it is not a success-rate result.

In the viewer's **Group enable** panel, enable **Geom 3** to show collision geometry. The first
controlled comparison should hold the hand at `low` and vary only `foam4`, `foam16a`, `foam16b`,
and `foam64`. A 1,000-step MuJoCo smoke test for these four low-hand scenes stayed finite and
produced contacts for every candidate. This checks basic collision integration only; full rollout
throughput and task-success measurements had not yet been run at that stage. The controlled
measurements below now provide runtime and diagnostic closed-loop data, but still no successful
Duck episode.

### Controlled performance and closed-loop studies

Use two separate tools because computation time and task behavior answer
different questions:

```bash
# Same Duck state on every timed call: latency, throughput and GPU memory only.
python analysis/duck_foam/benchmark_mppi_static_state.py

# Paired seeds in the fixed reference-eval world: task behavior and latency.
python analysis/duck_foam/run_geometry_closed_loop_study.py
```

The static benchmark launches every geometry/sample cell in a fresh process,
warms up CUDA, and records mean/median/P95 latency, world-step throughput, and
resident GPU memory. The closed-loop study rotates geometry order by seed,
checkpoints every episode, supports exact resume through a protocol hash, and
reports Wilson success intervals, failure modes, final goal errors, and
mean/median/P95 planning latency. It disables trajectory/distribution recording
so GPU readback does not perturb the timing.

Before attributing a drop to the planner or rollout geometry, verify that the
reference initial grasp is stable under its fixed hand command:

```bash
python analysis/duck_foam/check_duck_hold_stability.py
```

The local 2026-08-29 experiment record, including limitations and file paths,
is in `EXPERIMENT_NOTES_2026-08-29.md`. In particular, current Duck MPPI tuning
did not produce a successful episode, so the closed-loop outputs are diagnostic
stability/error results and must not be presented as a final success comparison.

The reported RTX 5070 Ti timing and closed-loop diagnostics were collected on
project revision `064cc3c`, before later `planner_dev` changes to the Duck body
frame and controller. The geometric approximation metrics remain tied to the
unchanged source STL, but runtime and closed-loop measurements should be rerun
after integration with a newer project revision.
