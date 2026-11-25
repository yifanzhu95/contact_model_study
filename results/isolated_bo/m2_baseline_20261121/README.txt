M2 BASELINE BAYESIAN OPTIMIZATION — PREPARED, NOT STARTED
======================================================

This is one isolated 500-trial search for the user-designated second 12-episode
M2 baseline. No hardware, camera, LCM, optimizer trials, GPU planners, or MPS
services have been started by preparation. Wait for the user's explicit nudge.

BASELINE
--------
Frozen baseline.json points to the exact 6-success / 5-drop / 1-timeout batch.
M2 only; calibrated hand dynamics; high/high geometry; 70g, 70mm cube.
512 samples, one MPPI iteration, 0.004s physics step, 0.064s control step,
0.256s / four-action horizon, temperature 0.1, noise sigma 0.05, shifted warm
start, centered noise, no adaptive temperature or within-iteration resampling.
The goal remains [0.025,0.035,0.075]m and identity WXYZ quaternion.
Position success <20mm, quaternion metric <0.04 (~23.07deg), velocity norm <10.
Cost drop height 65mm, failure height 0. P011 is the joint-cost reference.
Sliding friction cube/tips/palm/other fingers: .3/.5/.4/.4; torsional .005,
rolling .0001. Calibrated hand dynamics and the baseline measured joint ranges
are embedded in frozen rollout/evaluation XML; all referenced mesh/texture
assets are copied and hashed inside this directory.

SEARCH SPACE — 11 DIMENSIONS
----------------------------
Log-uniform, matching the previous BO bounds:
  w_quat, w_pos_x, w_pos_y, w_pos_z, w_joint: 0.01 to 100
  w_fallen:                                 10 to 500
  w_quat_term, w_pos_term:                   10 to 1000

Uniform, newly included execution/penalty parameters:
  action_rate_weight:  0 to 50       (baseline 10)
  joint_limit_weight:  0 to 1000     (baseline 100)
  plan_blend_alpha:    0 to 1        (baseline .75)

Fixed cost terms:
  w_contact=80; w_velo=0; w_joint_velo=0; w_fallen_term=0.
The previous completed 750-trial BO started and ended with w_contact=100;
per the instruction to keep unchanged terms constant, this search retains the
CURRENT baseline contact value 80. Contact varied during that older search;
"unchanged" here specifically means initial seed versus winning candidate.
See previous_bo_audit.json. The three zero terms were already pinned before.

No search of temperature, noise, sample count, horizon, clipping bounds,
friction, mass, goal/success thresholds, calibrated dynamics, or joint limits.
The user explicitly confirmed searching both penalty weights and blending alpha.

INITIALIZATION AND EVALUATION
-----------------------------
Each trial evaluates the same 12 RECORDED baseline initial q/q_dot states,
covering six signed approximately 90-degree rotations twice each. The recorded
off-center placement is retained. Noise seeds are reproducibly derived from
seed 0 and episode index, independently of the pool worker/planner chosen.
Initial actuator targets/activation use robot_qref, as held before hardware
episodes. No synthetic object-drop settling interval is inserted. The logged
hardware --settle option was inactive during its operator preparation.

The eval plant is CPU MuJoCo with calibrated dynamics, 4ms timestep and M2's
Newton/25/1e-6/pyramidal settings. Planners use GPU M2 with native calibrated
activation reset. Sampled controls are not clipped using hardware ctrl_range.
The planner's raw first delta is optionally blended with the old time-aligned
plan, added to the updated simulated joints, clipped +/-0.2, then clipped to
the fixed measured joint bounds. Warm-start shifting is not changed.

Timing uses deterministic virtual deadlines: while a plan is pending, the
plant advances 64ms under its held target before the new command is applied.
GPU queue/computation wall time does not change the physics clock. There are
no wall-clock pacing sleeps. USB/camera delays, pose noise, and tracking errors
are NOT replayed; the recorded initial states already use Method-5 calibration.
This is a model-based optimization, not a prediction of hardware success rates.

Static CPU contact inspection found initial model/object penetration up to
5.772mm. It is retained and reported in initial_geometry_validation.json;
initial positions are not silently changed. Some recorded preparation joints
also lie outside the later preferred ranges, as in the actual baseline where
preparation bypasses the MPPI-only absolute clamp. No dynamics episode was run
to validate those starting contacts during this preparation.

OPTIMIZER AND RESOURCES
-----------------------
500 total trials including 120 initial points: the baseline seed, three
single-parameter ablations (rate=0, limit-cost=0, blend-alpha=1), 116 random
points; then 380 GP-guided points. GP hedge acquisition, optimizer seed 0.
Original objective retained:
  J = -success_rate + .1 * mean(min(sum(final_error/threshold),250)/250).
Success/failure/timeouts remain valid observations. Resource failures such as
CUDA OOM stop the pool instead of scoring an optimizer candidate as bad.
12 episodes/trial = 6000 evaluations maximum; 1000 control cycles each.
Previous BO observations are not reused because this setup changes the
physics, reference, clipping, horizon and optimization dimensions.

GPUs 2 and 3, identified by their saved UUIDs. Preferred 20 planner processes
per GPU under private MPS, fallback 16 on allocation/guard failure; 48 CPU eval
workers, at most 3 trials in flight. With 12 episodes/trial, at most 36 episode
workers have trial work concurrently; do not interpret 48 as 48 busy workers.
The historical 20-planner test did not benchmark this larger 512-sample model.
GPU/MPS startup and concurrency remain untested until the nudge.
Start refuses a busy primary, defers a busy secondary, and monitors >=3GiB
headroom and <83C.
No other user's process is stopped.

ADAPTIVE GPU WATCHDOG
---------------------
GPU 2 is the permanent primary; GPU 3 is the only optional secondary. GPU 0
and GPU 1 are monitored, never used by this search. A separate watchdog
process samples all GPUs every 5 seconds. If either monitored GPU remains
at >=20% utilization for 10 seconds, GPU 3 is released. It is restored only
after BOTH monitored GPUs stay below 10% for 60 seconds, and GPU 3 itself is
available (<15% utilization, <1000MiB used, >=3GiB free, <83C).
The search starts on GPU 2 while the watchdog establishes the idle interval.
Thresholds and dwell times are in manifest.json -> gpu_watchdog.

GPU 3 clients finish their current request and exit; their private MPS server
then stops to release its resources too. GPU 2 and the shared CPU workers,
episode states, request queues and BO scheduler are not restarted. Restoring
GPU 3 recreates ONLY those clients/graphs. No trial is paused or discarded by
a load transition. One GPU stays allocated while the search is healthy; this
does not imply 100% utilization during optimizer fitting or CPU evaluation.
Ordinary load transitions do not trigger the global 20->16 resource fallback.
A watchdog crash triggers restart; missing/stale telemetry yields GPU 3 while
GPU 2 continues. Primary allocation/thermal/CUDA faults can still stop the run.

watchdog_state.json / gpu_metrics.jsonl: samples and load policy decisions.
gpu_control.json / gpu_ack.json: requested mode and actual secondary release.
gpu_events.jsonl / status.json: transitions and current search progress.
These files/processes are created only by --start; none exists from preparation.

RESUMING AFTER AN INTERRUPTION
------------------------------
Every candidate launch, completed episode, and optimizer observation commits
resume.pkl using fsync + atomic rename, retaining the prior checkpoint too.
It includes the optimizer/RNG state, pending trial candidates, completed
episode results within unfinished trials, remaining seeds and random count.
Completed trial.json records are committed atomically and reconciled if a
kill occurs between writing the result and updating the optimizer checkpoint.
No completed trial or saved completed episode is intentionally rerun.

Run the SAME run.py --start command to resume. An interrupted, unfinished
episode is replayed from its fixed initial state and seed; arbitrary
mid-episode physics/GPU state is not checkpointed. A result not yet durably
committed at SIGKILL may need recomputation. Trials retain their candidates.
Linux parent-death signals stop orphaned pool children; the independent
watchdog also cleans up this search's registered process group/private MPS
if the supervisor is killed. No other user's processes are terminated.

ISOLATION
---------
code/ is a private source snapshot. GPU penalty weights use a two-float runtime
buffer so trial changes do not require graph recapture or a planner per trial.
Both kernels are captured once; weight zero gives exactly zero penalty.
The 48-worker, 20/16-planner pool is adapted from the previous BO workflow.
No active CubeReorient.py weights, hardware configs or calibration are edited.
Completion exports best_parameters.json and best_weights_commented.txt here;
it never activates them automatically.

CPU-only checks, safe before authorization to optimize:
  /home/chetan/miniconda3/envs/mj_env/bin/python <this-directory>/run.py --check

Future explicit start/resume — DO NOT RUN until the user nudges:
  /home/chetan/miniconda3/envs/mj_env/bin/python <this-directory>/run.py --start

Review manifest.json, search.csv, baseline.json, previous_bo_audit.json,
initial_states.json and initial_geometry_validation.json for exact values.
