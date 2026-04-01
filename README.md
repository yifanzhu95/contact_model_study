# Contact Model Study

> **What matters about contact models in dexterous manipulation?**
> A systematic empirical study of contact model fidelity for sampling-based MPC.

---

## Overview

This repo implements the experimental study from the attached PDF. It evaluates
10 contact model variants (M1–M10) across three manipulation tasks of increasing
contact complexity, under two experimental conditions:

- **Condition A** — fixed computation budget (models with lower cost get more samples)
- **Condition B** — fixed sample count (isolates approximation error from sample count)

### Contact Model Variants

| ID  | Description |
|-----|-------------|
| M1  | Anitescu / pyramidal-cone Newton solver (MJWarp) |
| M2  | MuJoCo default soft contact / PGS — **baseline M*** |
| M3  | Jin 2024 complementarity-free model |
| M4  | Decoupled XPBD-style penalty (Coulomb friction) |
| M4d | M4 with friction relaxed to viscous damping |
| M5  | Degraded geometry + accurate contact (M2) |
| M6  | Degraded geometry + approximate contact (M4) |
| M7  | Inaccurate physical parameters + accurate contact |
| M8  | Inaccurate physical parameters + M4 |
| M9  | Degraded geometry + inaccurate physics + M2 |
| M10 | Degraded geometry + inaccurate physics + M4 |

### Tasks

| Task | Complexity | Key contact challenge |
|------|------------|----------------------|
| Push | LOW | 1–2 quasi-static contacts |
| Grasp & Reorient | MEDIUM | ~4 contacts, dynamic lifting |
| Peg-in-Hole | HIGH | tight clearance, multi-contact insertion |

---

## Repository Structure

```
contact_study/
├── src/
│   ├── contact_models/
│   │   ├── config.py           # ContactModelConfig + all Mk factory methods
│   │   ├── api.py              # Unified dispatch surface (put_model/step/forward)
│   │   ├── xpbd_backend.py     # M4: decoupled XPBD contact solver
│   │   └── benchmarks.py       # Speed and approximation error measurement
│   ├── planners/
│   │   ├── mppi.py             # MPPI controller
│   │   └── cem.py              # CEM controller
│   ├── tasks/
│   │   ├── base.py             # BaseTask, TaskSpec, task registry
│   │   └── tasks.py            # PushTask, GraspReorientTask, PegInHoleTask
│   ├── evaluation/
│   │   └── metrics.py          # EpisodeResult, AggregatedResult, serialization
│   └── utils/
│       └── rollout.py          # batch_rollout, fixed_budget_rollout, fixed_sample_rollout
│
├── scenes/
│   └── tasks/
│       ├── push_{accurate,convex_hull,primitive_union}.xml
│       ├── grasp_reorient_{accurate,convex_hull,primitive_union}.xml
│       └── peg_in_hole_{accurate,convex_hull,primitive_union}.xml
│
├── experiments/
│   ├── run_experiment.py       # Main study runner (Conditions A & B)
│   ├── benchmark_speed.py      # Throughput benchmark vs batch size
│   └── measure_approx_error.py # Approximation error vs horizon
│
├── analysis/
│   └── plot_results.py         # All paper figures
│
├── tests/
│   ├── test_config.py          # Unit tests (no GPU required)
│   └── test_integration.py     # Integration smoke tests (GPU required)
│
├── results/                    # JSON result files (gitignored)
├── figures/                    # PDF figures (gitignored)
└── pyproject.toml
```

### Dependency on `comfree_warp`

This repo depends on the `comfree_warp` package (Yifan's code), which must be
on your `PYTHONPATH`. It provides:
- **M1 / M2**: via `comfree_warp.mujoco_warp` (vendored MJWarp)
- **M3**: via `comfree_warp` directly (Jin complementarity-free solver)
- **M4**: implemented in `src/contact_models/xpbd_backend.py`, using MJWarp's
  collision detection and kinematics but replacing the contact resolution

---

## Installation

```bash
# 1. Install the comfree_warp package (adjust path as needed)
pip install -e /path/to/comfree_warp

# 2. Install this package
pip install -e ".[dev]"

# 3. Verify
python -c "import comfree_warp; import src.contact_models.api"
```

---

## Usage

### 1. Run speed benchmark first

```bash
python experiments/benchmark_speed.py \
    --task push \
    --models M1 M2 M3 M4 \
    --batch_sizes 64 256 1024 4096 \
    --horizon 50
```

### 2. Measure approximation error

```bash
python experiments/measure_approx_error.py \
    --tasks push grasp_reorient peg_in_hole \
    --models M1 M3 M4 M5 M7 \
    --horizons 5 10 20 40 \
    --n_states 50
```

### 3. Run the full study

```bash
# Quick smoke run
python experiments/run_experiment.py \
    --tasks push \
    --models M1 M2 M3 M4 \
    --conditions A B \
    --n_episodes 5 \
    --budget_seconds 0.05

# Full study (takes hours on GPU)
python experiments/run_experiment.py \
    --tasks push grasp_reorient peg_in_hole \
    --models M1 M2 M3 M4 M4d M5 M6 M7 M8 M9 M10 \
    --conditions A B \
    --n_episodes 20 \
    --budget_seconds 0.1 \
    --n_samples_b 1024
```

### 4. Generate figures

```bash
python analysis/plot_results.py results/experiment_TIMESTAMP.json
```

### 5. Run unit tests

```bash
pytest tests/test_config.py -v           # no GPU needed
pytest tests/test_integration.py -v      # needs GPU + MuJoCo
```

---

## Adding a New Contact Model

1. Add a `Backend` enum value in `src/contact_models/config.py`
2. Add a parameter dataclass (e.g., `MyModelParams`) in the same file
3. Add a factory classmethod `ContactModelConfig.Mk()`
4. Implement `put_model / make_data / step / forward` in a new file
   `src/contact_models/my_backend.py`, following the pattern in `xpbd_backend.py`
5. Add dispatch cases in `src/contact_models/api.py`
6. Add the factory to `MODEL_FACTORIES` in `experiments/run_experiment.py`

---

## Notes on Geometry Variants

Geometry variants (M5, M6, M9, M10) are handled entirely in XML — no code
changes are needed. The `GeometryVariant` enum value is substituted into the
XML path template:

```
scenes/tasks/{task_name}_{geometry_variant}.xml
```

For non-trivial objects (e.g., a mug or T-shaped peg), prepare variants by:
1. **Convex hull**: run CoACD or V-HACD, export as mesh XML
2. **Primitive union**: manually fit spheres/capsules/boxes in the XML
3. **Linearized**: use MuJoCo's built-in mesh linearization options

---

## Citation

If you use this code, please cite:

```
[Your paper citation here]
```

and the relevant contact model papers:
- Jin 2024 (M3): arXiv:2408.07855
- Macklin et al. 2019 (M4): ACM SCA 2019
- Anitescu 2006 (M1): underlying model for MJWarp Newton/CG solver
