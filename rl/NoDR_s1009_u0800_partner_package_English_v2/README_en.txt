NO-DR LEAP CUBE POLICY — SEED1009, FINAL800 PPO UPDATES
Package release2 — English-only handoff; inference interface version1

1. Identity and scope

This is the user-confirmed no-domain-randomization policy, seed1009, final800
PPO updates. It is the common800-update parent used by the later DR study.
It is not a DR policy or a best-checkpoint selection. Its learning history is
the original188-update prefix plus612 continuation updates,209,715,200 total
training transitions. Physical episodes restarted at188; model/optimizer and
normalizer state were restored for continuation.

The TorchScript file is an unchanged byte-for-byte copy of the original actor.
All exported actor and normalizer tensors exactly match the complete800-update
training checkpoint. Critic, optimizer and training RNG files are not required
for inference and are not included in this receiver bundle.

This package runs inference only. It does not contain a robot driver, motor
communication, a simulator, a hardware reset routine, or a ready-to-run robot
control loop. Receiver-specific sensing and motor calibration are unresolved.

2. Quick offline check

Use an environment with Python and PyTorch. No CUDA, MuJoCo, Warp or RSL-RL is
required by portable_policy.py or self_test.py. Runtime actually checked here:
Python3.12.3, PyTorch2.14.1+cu130, executing on CPU. An installed CUDA-enabled
wheel was used on CPU; this does not mean a GPU is required. Other environments
must run the supplied self-test rather than assume binary compatibility.

After extracting the archive, from this package directory:

    python self_test.py
    python example_offline.py

The first command checks checksums and112 archived observation/action cases
using both backends. Expected status: PASS_OFFLINE_INTERFACE_PARITY.
The second prints a saved simulation example; it never sends motor commands.
Neither is a new rollout or a hardware performance test.

TorchScript is the format already exported by training. Current PyTorch marks
TorchScript as deprecated, so a plain-PyTorch inference-only tensor state and
equivalent network implementation are included as a fallback:

    python self_test.py --backend pytorch

There is no need to install the original training project. See runtime_tested.json.

3. Files and minimal Python interface

policy_noDR_s1009_u0800.ts.pt : original normalized actor, default backend
actor_inference_weights.pt : same inference tensors for plain-PyTorch fallback
portable_policy.py         : CPU actor, state encoder and action decoder
interface.json            : exact observation/action convention and model ranges
joint_order.csv           : policy joint and actuator ordering
hardware_mapping_template.json : fields to populate on the receiving system
reference_cases.json      : archived simulation inputs and expected outputs
self_test.py              : checksum and offline parity check, no robot I/O
example_offline.py         : one archived inference example, no robot I/O
evaluation_summary.json   : existing M2/M5 evaluation results, limitations retained
preparation_QA.json        : actual source/encoder/action consistency checks
SHA256SUMS.json            : receiver-file checksums

    import json
    from portable_policy import Policy
    policy = Policy(backend="torchscript")  # or backend="pytorch"
    output = policy.infer_state(state)      # state schema below
    targets = output["joint_target_rad"]   # [batch,16], radians

This returns values only. The receiver owns any subsequent hardware adapter.

If the receiver already constructs the complete raw73D observation:

    raw = policy.raw_action(observation)    # [batch,73] -> [batch,16]

The actor includes frozen observation normalization:
    (observation - saved_mean) / (saved_std +0.01)
Do not normalize it a second time. Do not update statistics on hardware data.
The actor is deterministic and returns the Gaussian policy's mean latent action;
training exploration noise is not sampled at inference.

4. Required physical state

state is a dict containing the following keys. Values may describe one sample
or a batch; all fields must have the same batch size. Every vector must be finite.

joint_position_rad                    16 angles, policy joint order, radians
joint_velocity_rad_s                  16 velocities, same order, radians/second
object_position_world_m                3 positions in TRAINING world, metres
object_quaternion_wxyz                 4 quaternion values, w,x,y,z
object_linear_velocity_world_m_s       3 linear velocities, training-world axes
object_angular_velocity_world_rad_s    3 angular velocities, training-world axes
goal_position_world_m                  3 target positions, training world, metres
goal_quaternion_wxyz                   4 target quaternion values, w,x,y,z
previous_target_rad                   16 preceding accepted actuator targets
elapsed_steps                         completed64ms control periods, integer0..499

At the first decision, elapsed_steps=0. After one accepted/applied64ms control
period it is1. This task ends at500 controls (32 seconds). The supplied encoder
rejects500 or greater; it does not implement automatic physical resets.

Object pose and linear/angular velocity were directly available in simulation.
Real deployment needs a suitable state estimator (for example tracked object
pose plus a velocity estimator). Joint feedback alone is not enough for this
policy. Omitting object velocity or replacing it by zero changes its inputs.

5. Exact73-value raw observation layout (zero-based, stop exclusive)

[0:16]   joint positions: 2*(q-q_joint_lower)/(q_joint_upper-q_joint_lower)-1
[16:32]  joint velocities /20
[32:35]  object world position *10
[35:41]  object orientation rotation6
[41:44]  (goal_position-object_position)*10
[44:50]  rotation6(R_world_object.T @ R_world_goal)
[50:53]  object LINEAR world velocity, unscaled (NOT divided by10)
[53:56]  object ANGULAR world velocity /10
[56:72]  previous actuator targets scaled using ACTUATOR control ranges
[72:73]  time left:1-elapsed_steps/500

Quaternion order is w,x,y,z. R_world_object maps body vectors into world axes.
rotation6 takes the first two COLUMNS, stored in this order:
    R00,R10,R20,R01,R11,R21
It is not the first six entries of row-major matrix flattening.

The world is the original training scene world, not automatically the camera,
robot-base or palm frame. interface.json includes the fixed scene palm transform
as a reference, not a measured real-system transform. The receiver must register
its estimator to that world and rotate both velocity vectors accordingly.
MuJoCo's raw free-joint angular qvel is body-frame; archived examples convert it
using omega_world=R_world_object @ omega_body. Do not rotate a velocity that is
already reported in world axes a second time.

Joint-position normalization ranges and actuator-target ranges are different!
For example, training MCP joint-position lower bound is0 but actuator target
lower bound is-0.314rad. portable_policy.py preserves these separate scalings.
The encoder does not silently clip joint-state observations to[-1,1].

6. Exact action meaning and control timing

The16 raw network outputs are NOT joint positions, torques or motor velocities.
Apply exactly one tanh and use:

    u_next = clamp(u_previous +0.064*tanh(raw_action), ctrl_lower, ctrl_upper)

u_next is an absolute joint POSITION target in radians and policy actuator order.
u_previous is the preceding accepted actuator TARGET, not measured current q.
The output contains no trained0.15rad lead limiter. Do not apply tanh twice or
replace the accumulator with q_current+delta without documenting an interface change.

Training policy period:64ms (15.625Hz). In simulation, the same target was held
for16 physics steps of4ms, with a position servo. If the hardware low-level loop
is faster, retain the target between policy decisions; do not add0.064*tanh(a)
again on every faster motor tick. The64ms policy timing and episode clock belong
to the trained interface.

Initial previous_target_rad must match the actual target the receiver has
established for that grasp. Do not reset it to an arbitrary zero vector each tick.
If the hardware adapter clips or changes the target, keep a record of the applied
target and document the discrepancy from the trained action interface.

7. Joint mapping and low-level control

Policy order:
  0 if_mcp    1 if_rot    2 if_pip    3 if_dip
  4 mf_mcp    5 mf_rot    6 mf_pip    7 mf_dip
  8 rf_mcp    9 rf_rot   10 rf_pip   11 rf_dip
 12 th_cmc   13 th_axl   14 th_mcp   15 th_ipl

These are training actuator/joint indices, NOT verified hardware motor IDs.
Populate hardware_mapping_template.json with the receiver's IDs, signs, zero
offsets, position limits and world-frame registration before any physical use.
Interface ranges/gains are simulation training references, not calibrated robot
limits or a recommendation to copy simulator gains into a motor controller.

8. Task and existing evidence

Training task: LEAP-hand cube reorientation on M2/high-high geometry, fixed
canonical localX -90 or+90 degree goals,500 controls; success does not reset.
Goal position in training world:[0.02,0.035,0.085] metres.
Goal quaternions are listed in interface.json. They are canonical fixed goals,
not automatically “rotate90 degrees relative to the current estimated pose.”
The policy has not been trained for arbitrary objects or arbitrary target poses.
Reference cases describe simulated grasps and are NOT physical initialization
commands to replay on a robot.

Original128-trial fixed-policy evaluation:
            strict arrival   strict final   last150 continuous hold   drops
 M2             128              128                 128                 0
 M5              49               41                   0                 6

Both are simulation results. M5 is a cross-model simulator evaluation, not a
physical robot. This research policy is supplied for integration/experiments;
robust real-world transfer has not been established. These limitations have not
been removed from the handoff.

9. What has actually been checked

Original actor copied without byte changes; actor/normalizer tensors compared
exactly with complete800 checkpoint. Standalone encoder compared with original
training encoder and saved training observations.112 decision cases from the
saved M2 and M5 evaluations cover both goal directions and early/late steps.
Each case uses state at step k-1 to reconstruct the action recorded at step k.
Both inference backends are checked by self_test.py. preparation_QA.json records
maximum errors and predefined tolerances. CPU inference verification does not
include sensor latency, estimator quality, motor calibration or end-to-end robot
timing. No new physics, training or physical actuation was performed.

10. Official format/runtime references

https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
https://docs.pytorch.org/tutorials/beginner/saving_loading_models
