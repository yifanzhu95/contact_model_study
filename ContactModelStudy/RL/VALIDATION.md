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

## Recorded evidence

The unchanged runtime implementation has 184 passing selected checks in working
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
