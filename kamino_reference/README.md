# Kamino offline reference

This is an isolated Newton 1.6 / SolverKamino environment for offline full-NCP
reference experiments.  It is intentionally not installed into the main
`contact_study` environment because the two stacks require incompatible
MuJoCo and Warp versions.

From this directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/validate_delivery.py \
  --scene ../scenes/leap/env_leap_rollout_cube_high_high.xml \
  --check-cuda
.venv/bin/python -m pytest -q
```

The lock-file step is required for reproducible GPU behavior.  In particular,
the frozen baseline uses Warp 1.17.0; an unconstrained Warp 1.18.0 install was
observed to exceed the 120-second G0 guard limit on the validation host.

The repository-level comparison entry point is:

```bash
../.venv/bin/python ../experiments/run_contact_reference_comparison.py --help
```

See [the collaborator handoff](docs/collaborator_handoff.md) before running a
large budget.
