# Validation scope

I separate interface correctness, bounded runtime checks and learning performance.
Passing interface checks does not establish that PPO has learned the task.

## Checked behavior

- Indexed resets preserve other worlds and their hidden physics/control history.
- Owned state and transition tensors survive subsequent steps and resets.
- Goal-conditioned observations, action decoding, command limits and original task costs agree.
- Success/drop/deadline priority, finite reward and final-observation handling are explicit.
- Checkpoint markers, hashes, optimizer/config/RNG restart and partial-write rejection are checked.
- Backend faults lock the environment; cleanup errors preserve the original exception.
- RL and MPPI share case execution, scoring and synchronized action timing.
- Continuous evaluation preserves physical state, commands and elapsed time at target changes.
- Partial sequence completion is retained without counting it as sequence success.
- Video follows simulation/control clocks and updates target markers without physics steps.
- Unimplemented randomization/distributed launch requests fail explicitly.
- Consumed PPO diagnostics are released after each update, including runs without a logging callback.
- An optimizer fault preserves already accepted samples and terminal outcomes in failure accounting.
- Compiled source-scene fingerprints detect included XML/mesh changes and survive checkout relocation.

Native GPU reset checks require exact preservation of the unselected worlds'
state, control history and relevant caches at the reset boundary. Subsequent
cross-run trajectory differences are recorded as diagnostics. Two replay runs
do not define a guaranteed noise bound; trajectory repeatability is not a pass
requirement for the native models. No contact model, solver tolerance or constraint
ordering is changed to meet these checks.

## Recorded evidence

The pre-training audit passed 259 core tests with two optional Pinocchio/Drake
tests skipped. This run included the GPU tests, all 24 scene choices, native
M1–M5 object/FOAM interface cases and six real PPO-update tests. Each update test
used two environments and four control steps (eight training samples).

A separate actual M2 driver check ran two updates, resumed one periodic checkpoint
for one additional update, restored without sampling and evaluated two independent
cases. It used 12 training samples in total. Per-update diagnostics were finite,
actor parameters changed, complete checkpoint counters/hashes agreed and the
restore-only actor matched exactly. This tests consecutive updates and the full
launch/save/restart/evaluation path; it does not measure learning progress.

The audit corrected three issues: retained library diagnostics grew with the
number of updates; an optimizer failure omitted its last accepted sample from
failure counts; and top-level XML checks did not cover changed included assets.
Regression tests cover each issue. Native GPU future-trajectory thresholds were
also replaced by diagnostic-only reporting, while exact reset-boundary isolation
checks were retained and strengthened. The initial failing assertion and all
before/after results are retained in the local audit evidence.

The initial framework release had 184 passing selected checks in working
and clean imported copies; two optional CPU-engine checks were skipped because
their runtime dependencies were unavailable. Real PPO-update and GPU-marked tests
were excluded from that structural suite.

A separate bounded interface check used 13 tiny evaluations with 33 explicit
control steps, covering cube/duck/ball on CPU, M1–M5 on GPU, native MPPI and matrix
routing. Ten target switches preserved measured state, commands and clocks.
Two continuous video clips decoded and their terminal images were inspected.
These used untrained hold actors and easy first goals with 1-microsecond physical
and control steps. The next targets used the original 90-degree sampling rule.
This checks success/target/deadline boundaries; it is not a realistic task-duration,
rotation-learning, sustained-throughput or multi-GPU experiment.

Earlier bounded checks connected all five native models to PPO, export and
reset-based restart. They do not establish a reliable learning baseline.

## Not established

I have not established converged learning, repeatable task success across training
seeds, fair model-performance rankings, cross-object generalization, sustained
training capacity, multi-GPU scaling, target-HPC runtime qualification or hardware
transfer. Pinocchio/Drake adapters require their optional runtimes before actual
execution can be qualified.

The shared evaluator does not make MPPI prediction costs identical to the PPO
objective. Restart begins new physical episodes and does not reproduce a partly
completed episode. GPU nondeterminism is accepted; exact cross-device trajectory
or RNG continuation is not promised.
