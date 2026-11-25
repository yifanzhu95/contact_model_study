# Contact Model Study

> **What matters about contact models in dexterous manipulation?**
> A systematic empirical study of contact model fidelity for sampling-based MPC.

---

## Overview

This repo implements (in progress) the experimental study that evaluates different contact
models across manipulation tasks at a fixed sample count, which isolates
approximation error from sample count.

### Study axes (kept orthogonal in code)

| Axis                    | Lives in                                     | How to vary                                 |
|-------------------------|----------------------------------------------|---------------------------------------------|
| Contact model (M1..M4)  | `ContactModelConfig` (`config.py`)           | `ContactModelConfig.M1()` … `.M4()`         |
| Geometry fidelity       | XML files in `scenes/leap/*.xml`             | `get_task(name, geometry="duck_low_high")` |
| Physics parameter noise | `contact_study.utils.physics_noise`          | `apply_physics_noise(mjm, PhysicsNoiseParams(...))` |

The 4 contact models stay in `ContactModelConfig`. Geometry and physics-parameter
degradations are **not** fields on that config — they are applied at MjModel load
time in the benchmark script, so any of the 4 contact models can be paired with
any geometry and any noise level without touching the core code.

### Scene variants (`--geometry`)

`--geometry` names a **scene variant**: which object is manipulated, and at what
collision fidelity the planner's hand and the object are modelled.

    --geometry <object>_<hand_acc>_<obj_acc>     e.g. duck_low_high
    --geometry <object>                          object at default fidelities

Scene files are found by convention (see `contact_study.tasks.config.SceneVariant`):

    rollout:  scenes/leap/env_leap_rollout_{obj}_{hand_acc}_{obj_acc}.xml
    eval:     scenes/leap/env_leap_eval_{obj}.xml

The eval scene carries no accuracy suffix — it *is* the reference fidelity. Only
the planner's model is degraded, which is the axis the study varies. Hand rungs
are an `<include>` swap (`low` / `med` / `high` -> the corresponding
`leap_right_hand_{accuracy}.xml`); all hand XMLs are kinematically identical, so
rollout and eval scenes differ only in collision geometry.

Available convex-hull Duck baselines are `duck_low_high`, `duck_med_high`, and
`duck_high_high`, where the middle token selects hand accuracy and `high` means
the existing eight-hull Duck collision model. The FOAM study adds four object
accuracy labels for every hand accuracy:

| Object label | Duck rollout collision model |
|--------------|------------------------------|
| `foam4`      | calibrated 4-sphere Low      |
| `foam16a`    | calibrated 16-sphere Medium-A |
| `foam16b`    | calibrated 16-sphere Medium-B |
| `foam64`     | calibrated 64-sphere High    |

For the first object-only comparison, keep the hand fixed and run
`duck_low_foam4`, `duck_low_foam16a`, `duck_low_foam16b`, and
`duck_low_foam64`. All four selectors still resolve eval to the same
`env_leap_eval_duck.xml` eight-hull reference. The generated scene manifest,
source geometry metrics, and regeneration instructions live in
`analysis/duck_foam/README.md`.

The retired `GeometryVariant` names (`accurate`, `convex_hull`,
`primitive_union`, `linearized`) are still accepted and map to the default
variant, so existing SLURM scripts keep working.

The [KL-vs-success workflow](analysis/README_kl_divergence.md) intentionally
uses a stricter selector: all five current objects (`cube`, `duck`, `ball`,
`spam`, `tomato`) use `high_high` rollout geometry with their fixed evaluation
scenes. In that worker only, object shorthand selects `high_high`, lower
fidelities and legacy aliases are rejected, and plots separate object/config
families. Other drivers retain the general scene-selection behavior above.

### Contact model variants

| ID  | Description                                                        |
|-----|--------------------------------------------------------------------|
| M1  | Wanted an Anitescu model, for now just use MuJoCo but with hard contact |
| M2  | MuJoCo default soft contact                                        |
| M3  | Jin 2024 complementarity-free model (`comfree_warp_v1`)            |
| M4  | XPBD-style penalty model (`contact_models/xpbd_backend.py`)        |

### Old M5..M10 mapping

The old hardcoded M5..M10 combinations are replaced by CLI flags on the
benchmark scripts:

| Old ID | New invocation                                                        |
|--------|-----------------------------------------------------------------------|
| M5     | `--models M2 --geometry cube_high_high`                                  |
| M6     | `--models M4 --geometry cube_high_high`                                  |
| M7     | `--models M2 --friction_sigma 0.2 --mass_sigma 0.1`                   |
| M8     | `--models M4 --friction_sigma 0.2 --mass_sigma 0.1`                   |
| M9     | `--models M2 --geometry cube_high_high --friction_sigma 0.2 --mass_sigma 0.1` |
| M10    | `--models M4 --geometry cube_high_high --friction_sigma 0.2 --mass_sigma 0.1` |

## Calibrated LEAP hardware runner

For a manual USB joint test, run `python run_leap_joint_viewer.py`. The MuJoCo
window displays measured hardware positions without stepping simulation.
It starts with torque disabled; joints outside the calibrated model limits
are listed on screen. Reposition those joints gently by hand and press Enter
to hold the measured pose and enable the right-hand **Control** actuator
sliders. Space, Q, Escape, window close or Ctrl-C stops and disables torque.
The viewer uses `manual_viewer_limits` in the hardware JSON for its speed,
velocity and tracking-error bounds. These settings apply only to the viewer;
manipulation uses the delta-only command path described below. Motor registers must already match
`configs/leap_hardware.json`; the viewer verifies them without changing gains.

`run_leap_hardware.py` shares the CLI defaults, task builder, MPPI config
builder and rollout scheduling with `ContactModelStudy/Drivers/run_episodes.py`.
`--task` selects cube, ball or duck costs and per-object `CONTACT_PARAMS`.
`--hand-acc` and `--obj-acc` select the same scene variants, including Duck FOAM
geometry. `--plan-on-eval` selects the accurate scene for planning. Each selected
scene receives the calibrated hand's joint/servo parameters, body inertias and
mounting from `scenes/leap/hardware/calibrated_hand.xml`; selected collision
geometry and object properties stay intact. The results embed generated scene
XML and resolved configs for reproducibility.

The available rollout combinations are:

| Task | Hand accuracy | Object accuracy |
|---|---|---|
| `cube_reorient` | `low`, `med`, `high` | `high` |
| `ball_reorient` | `low`, `med`, `high` | `high` |
| `duck_reorient` | `low`, `med`, `high` | `high`, `foam4`, `foam16a`, `foam16b`, `foam64` |

Unavailable combinations report the selected task and geometry instead of
falling back to a different object. The `--plan-on-eval` option uses the eval
scene regardless of the two accuracy selectors, as in the simulation driver.

`--rollout-model M1` through `M6` use the shared `GetContactModelSim` factory.
Parameters resolve in the same order: contact preset, object's task tuning,
then explicit overrides. For example, `--contact-param stiffness=0.5` overrides
M3's task tuning; repeat the flag for other contact/solver fields. `--nconmax`
and `--njmax` remain available. CPU rollouts support M1/M2 with the same solver
and contact overrides; other backends require their CUDA dependencies. Kamino
uses an external exact first-order target filter because it exposes servo
targets rather than MuJoCo activation buffers; its other physics approximations
remain those of the existing study backend.

The earlier `scenes/leap/hardware/cube.xml` remains available as an explicit
`--model` override, which replaces both geometry-selected scenes. Its cube is
**7 cm and 140 g**, matching the reference cube. `cube_tracking.obj` supplies
the same centered cube frame for FoundationPose. The copied tracker project is
under `iris-foundationpose-plusplus/`; its original checkout is untouched.

Three processes exchange LCM messages:

| Process | Receives | Publishes |
|---|---|---|
| `iris-foundationpose-plusplus/src/online_tracking2ctrl.py` | Camera frames | `POSE`: generated `pose_types.Pose`, 4×4 object transform |
| `leap_hand_lcm_bridge.py` | `LEAP_ACTION_DELTA`: 16 joint increments | `LEAP_STATE`: timestamp, 16 joint positions, 16 velocities |
| `run_leap_hardware.py --hardware` | `POSE` and `LEAP_STATE` | `LEAP_ACTION_DELTA` |

The bridge publishes MuJoCo joint order/radians; the controller applies no
second permutation or π offset. It assembles `qpos[23]` and `qvel[22]` from the
measured hand and object, with object velocity estimated using MuJoCo's free
joint convention. Publishers use epoch microseconds; synchronize system clocks
when running across machines. Feedback age and inter-stream skew are checked.
If otherwise fresh streams briefly exceed the skew limit after planning, the
controller retries the full feedback snapshot while the original plan and its
input measurements remain within their existing age/time limits. It publishes
only after synchronization recovers; persistent skew, expired deadlines and
invalid/stale packets still stop control. Retry counts, wait duration and the
rejected pair's timestamps/ages are saved under `command_feedback_sync` in the
step record. No synchronization wait is added to normal iterations.
The copied tracker timestamps frames before inference, replacing the previous
use of RealSense device milliseconds. Channels and URL are configurable; the
default URL is `udpm://239.255.76.67:7667?ttl=1`.

Camera intrinsics and extrinsics are configured together in
`configs/leap_pose_transform.json`:

- `camera_intrinsics`: 3×3 color-image pixel matrix K, read from the active RealSense profile.
- `camera_serial`, `camera_fps`: select the calibrated device and stream rate.
- `T_tag_from_camera`: camera pose in the tag frame, exactly the matrix printed
  by `tag_calib.py`; it is already inverted from the detector's tag-in-camera pose.
- `T_base_from_tag`: world pose of the calibrated hand model's fixed `tag` site,
  including the palm's mounting translation and rotation.
- `camera_to_published`: the derived camera-to-MuJoCo-base matrix
  `T_base_from_tag @ T_tag_from_camera`. The shared loader computes it from
  those two fields, so tag calibration replaces the previous hard-coded transform.
- `image_size`: `[width, height]` for the camera streams and viewer window.
- `object_translation_adjust_camera_m`: object translation correction in camera
  XYZ meters, applied before conversion to the base frame. `[0, 0, -0.015]`
  moves the published object 15 mm toward the camera, normal to its image plane.
  It leaves the measured tag calibration and raw camera overlays unchanged.
- `T_model_from_published`, `translation_adjust_m`, `rotation_adjust_deg`:
  model-frame alignment, including small XYZ translation/Euler corrections.
- `T_tracked_from_object`: tracked mesh origin/axes to simulated object frame
  (the older `T_tracked_from_cube` key is also accepted).

The tracker and `iris-foundationpose-plusplus/viewer/receiver.py` both load this
file with `--camera-config`. The tracker configures its streams from `image_size`;
the viewer's optional `--draw-axes` projects object axes using the configured K.
Existing `--cam_K` and tracker `--camera-to-published` arguments override the file.
The controller accepts `--camera-config` as an alias for `--pose-transform`.
Restart affected processes after editing their calibration. The full pose map is
`T_adjust @ T_model_from_published @ T_published_from_tracked @ T_tracked_from_object`.
Camera extrinsics are applied once by the tracker; they are not applied a second
time by the controller. The published frame is the calibrated MuJoCo world/base.
The complete camera/object mapping is
`T_base_from_tag @ T_tag_from_camera @ T_camera_adjust @ T_camera_from_object`, where
`T_camera_adjust` is the translation above (identity when absent), matching
`pose_to_mujoco_frame` in `~/comfree_dex/test_leap_hardware.py`.

Calibrate before tracking when the camera or tag mount changes:

```bash
# Query the active RGB profile and update intrinsics/device metadata:
python calibrate_leap_camera.py --intrinsics-only

# Capture 30 stable frames of the base-mounted tag and save both transforms:
python calibrate_leap_camera.py --tag-id 7 --tag-size 0.04 --once
```

The tag size is the **black square's edge length in meters** (40 mm in the
reference script). Use the ID and size of the actual fixed tag. Without
`--once`, the calibration preview continuously prints the camera-in-tag matrix;
**S** saves a stable measurement, and Q/Escape closes the camera. The host
calibration environment needs `pyrealsense2`, `pupil-apriltags`, OpenCV, NumPy
and MuJoCo. This session used the matching RealSense 2.56.5 binding from
`/tmp/leap_camera_calibration_packages`; it was not installed into existing environments.
Calibration snapshots, the previous JSON, and an annotated image are saved
under `results/camera_calibration_*.json` / `.png`.

The tool reads the exact color stream used by tracking, uses its K for AprilTag
estimation, derives `T_base_from_tag` from `scenes/leap/hardware/calibrated_hand.xml`,
and saves the composed transform at full precision. Mask capture and tracking
pin the serial/profile; RGB-aligned depth remains in meters. Native profile
checks catch a different camera, pixel grid or changed factory intrinsics,
while explicit K overrides remain available. Nonzero lens distortion requires
rectified image/depth input to this pinhole pipeline; this camera reports zero coefficients.
All calibrated object/geometry scenes preserve the same tag mount, and hardware
startup checks it against the saved calibration. A started but unfinished tag
calibration is marked pending and blocks tracking/control until saved. Restart
tracking and the controller after calibration; the camera is released on exit.

A short CPU simulation accesses neither LCM nor hardware:

```bash
python run_leap_hardware.py --device cpu --n-samples 8 --horizon 2 --steps 3 --settle 0
```

Start live operation in separate terminals, using the vision environment for
the tracker and a compatible planner/hardware environment for the other two:

```bash
python iris-foundationpose-plusplus/src/online_tracking2ctrl.py --init_mask_path /path/to/current_cube_mask.jpg
python leap_hand_lcm_bridge.py
python run_leap_hardware.py --hardware --observe-only --task cube_reorient --rollout-model M2
```

For a single supervised launch, use `launch_leap_experiment.py` instead. Its
defaults use `irislab_fppp:leap_hardware`, derived from the original Docker
image with LCM installed permanently. SAM-HQ, mask capture and tracking run in that container;
`receiver.py`, the LEAP bridge and MPPI run on the host. Run the launcher from
a host desktop with a working GUI/display for the viewer.

At bridge startup, all motors must have torque disabled and match
`expected_registers.model_number` and `expected_registers.mode` in
`configs/leap_hardware.json`. After checking the entire hand and its health,
the bridge loads any differing configured P/I/D gains and goal-current values,
verifies each write by reading it back, and logs the old/new values. It does
not change operating mode. It then reads the current pose and preloads that
pose before enabling torque. These initialization writes occur only at startup;
the manipulation loop remains unpaced.

If an earlier failed run left torque enabled, first ensure its bridge has
exited, then recover without starting tracking, MPPI or reference motion:

```bash
python leap_hand_lcm_bridge.py --disable-torque-only
```

This command disables torque on every motor, reads back the torque registers
and exits. Success prints `Torque disabled and verified on all 16 motors.`
Bridge SIGINT/SIGTERM handlers request shutdown between serial operations,
without raising into the upstream SDK. Cleanup also recovers an abandoned
SDK busy flag and clears the receive buffer before attempting torque disable.

The local `mj_env` now contains the host hardware and M1–M6 planning stack:
Python 3.12, MuJoCo/MJWarp 3.14, Warp 1.17 and Newton 1.6. Activate it without
the previous temporary-package overrides:

```bash
conda activate mj_env
unset PYTHONPATH  # remove earlier /tmp package overrides if you exported them
python launch_leap_experiment.py --n-episodes 5
```

The repository's existing `pyproject.toml` is the host dependency source of
truth. Install its hardware and Kamino extras with
`python -m pip install -e ".[hardware,kamino]"`. It pins MuJoCo/MJWarp 3.14.0,
Newton 1.6.0 and Warp 1.17.0 (the Kamino extra), and pins M3's source commit.
There was no root conda environment YAML or requirements lockfile. The copied
FoundationPose project has its own requirements, served by the Docker image.
The exact installed host versions are saved as a snapshot under
`results/mj_env_setup/`, rather than a competing requirements specification.
M6 is installed with `--no-deps` to retain Warp 1.17 for M5; `pip check` reports
its upstream `warp-lang==1.15.0` pin as the one expected override. No host
Torch/SAM installation is needed when using Docker. Rebuild the small vision
image if needed:

```bash
./leap_docker.py build -f iris-foundationpose-plusplus/docker/leap-hardware.Dockerfile \
  -t irislab_fppp:leap_hardware iris-foundationpose-plusplus/docker
```

The previous Python 3.10 environment files are retained at
`~/miniconda3/envs/mj_env_before_leap_20261110` for rollback to the original prefix;
that directory is a file backup, not a separately configured runnable environment.
Installation records and backend checks are under `results/mj_env_setup/`.
The environment check passed offline episodes for M1–M4 and M6, plus default
graph-based M2 MPPI. M5 imports and resets successfully, but its first physics
step on the calibrated hand produced non-finite state and the MPPI warmup
failed. This numerical issue remains with the repository-pinned versions;
installation success does not imply that calibrated M5 manipulation is validated.

```bash
python launch_leap_experiment.py --bbox 440 272 103 113
```

Set the bounding box to enclose the actual object in the camera image (X, Y,
width, height in pixels). The example tightly encloses the current cube placement; exclude the hand and support rails.
Configuration lives in `configs/leap_experiment.json`. Set
`interpreters.control` to your host MPPI Python and `interpreters.viewer` to a
host Python with OpenCV, pyzmq and trimesh; null means the launcher's Python.
These also have `--control-python` and `--viewer-python` overrides. For vision
outside Docker, use `--no-docker --vision-python /path/to/python`.
The launcher automatically selects separate physical GPUs for vision and MPPI
(`tracking_gpu: "auto"`, `mppi_gpu: "auto"` in the experiment JSON). For a new
tracking session, SAM/mask/tracking get the lowest load across three short
`nvidia-smi` samples, breaking ties by allocated memory. MPPI and optional
fingertip prediction then get the lowest-load remaining card. With an existing
tracker, its card is kept and excluded from MPPI's automatic selection.
Use `--tracking-gpu 3 --mppi-gpu 0` **before** `--` to choose the cards explicitly.
`--gpu auto` selects the automatic split; a numeric `--gpu 1` or `--device cuda:1`
places both roles on that card unless overridden by a role-specific flag.
Each process receives its assigned GPU UUID through `CUDA_VISIBLE_DEVICES`;
its internal device is therefore `cuda:0`. Both assignments and sampled loads
are logged in `launch.json`. Interactive windows continue using the desktop's OpenGL display.
An existing tracker stays on its original GPU: changing its selection requires
`--restart-tracking`, which also recaptures the mask. Changing only the MPPI GPU
allows a later experiment to keep tracking without a reset. Automatic splitting
requires two GPUs; on a single-GPU host, explicitly select that card for both roles.
This is launch-time device
selection, not live GPU migration or a reservation against other users.
Docker commands default to `./leap_docker.py`. If your account already belongs
to the `docker` group but the launching shell has not refreshed its groups,
the wrapper uses `sg docker` for that subprocess. It changes no account or
socket permissions. The same wrapper handles container startup, vision
commands and cleanup; it does not depend on a temporary `/tmp` script.

The launcher reuses the tracking session that it left running by default. A
first launch, or an explicit `--restart-tracking`, performs these stages:

1. Start its own uniquely named vision container using the copied
   `docker/run_container.sh` in noninteractive mode, preserving the GPU, USB,
   host-network and IPC options used by the original script.
2. Start SAM-HQ and wait for its model/server health endpoint. Configure
   `sam_hq.checkpoint_path` or `--sam-checkpoint` if the image stores the
   checkpoint somewhere other than the server's existing default. Set
   `sam_hq.start_server=false` to use an already-running server.
3. Open the host receiver and run `camera_mask.py --once --preview`. The
   receiver shows the **live camera image with a yellow rectangle for the
   configured mask-selection box**, so the operator can position the object.
   Press **Space or Enter in the receiver** when placement is correct. No
   tracking or hardware control starts while waiting for placement. The default
   `placement_timeout=0` allows unlimited placement time; Q/Escape cancels.
4. Capture the current RGB and aligned depth frame, then request a SAM-HQ mask
   for the box. The receiver shows the captured image with a **green mask**.
   Check that green covers only the intended object. **Enter accepts; R returns
   to placement for a new capture; Q/Escape cancels.** Keep the cube still until
   tracking starts. Only after acceptance does capture release the camera and
   start tracking with `mask.png` and its exact `registration_frame.npz` RGB-D
   snapshot. Registration no longer applies an old mask to a later live frame.
   The same receiver window stays open and switches to the tracked mesh bounds
   and object axes. Shared absolute paths let capture and SAM access the same
   files. The tracker publishes the camera-frame pose and image for supervision,
   separately from the base-frame LCM pose used by MPPI.
5. Open a synchronized MuJoCo window using the controller's Python, calibrated
   hand, object mesh, pose transform and LCM channels. Wait until that window
   renders and the receiver displays five distinct fresh tracking frames after
   registration and first-track kernel warmup, then start the bridge and
   controller. Display readiness checks stream freshness; the
   operator judges whether the tracking itself is correct.

The MuJoCo window shows measured joints and the tracked object in the calibrated
base frame, updating as LCM feedback arrives. It never steps physics or sends
motor commands. Missing or stale measurements hide the affected geometry; the
controller's feedback age/skew limits also apply to this display. The initial
free-camera view uses the RealSense camera's calibrated position, viewing
direction and vertical field of view from `configs/leap_pose_transform.json`.
Left-drag rotates the view, right-drag pans and the scroll wheel zooms; the
live updates preserve those changes. MuJoCo's free camera uses world-up, so
sensor roll and the off-center principal point are not reproduced exactly.
A separate textured copy of the object shows the current
goal orientation in the upper-right corner. It follows the screen corner as
the view or window size changes and is kept small to clear the setup.
Its position is shifted only for display; the overlay lists the actual target
XYZ in the base frame and quaternion in WXYZ order. The controller writes its
adopted goal to `goal.json` before the Enter gate and updates it when targets
change, including measured-position overrides. The viewer does not sample its
own goal. To omit this
window, pass `--no-mujoco-viewer` or set `mujoco_viewer=false` in the experiment
config. A standalone `view_leap_tracking.py --goal-offset X Y Z` retains the
optional placement at a fixed world offset instead of the default corner.

When enabled, the MuJoCo window shows the predicted paths of all
four fingertips over MPPI's horizon: index red, middle green, ring cyan and
thumb yellow. A small sphere marks each start and a larger sphere its endpoint.
These positions come from replaying the final weighted mean control sequence
on an independent, one-world copy of the selected calibrated rollout model,
including its contact parameters, geometry and servo dynamics, followed by
forward kinematics. Prediction starts from the same measured state and servo
state used for that plan. It follows the planner's control mode; it does not
apply the bridge's final `ctrl_range` bounds, so these are the planner's
predictions rather than a guarantee of the hardware's executed motion.

The preview runs asynchronously at up to 5 Hz with only the latest pending
plan retained. Paths disappear when their plan is older than 0.5 s, live
feedback is invalid, or manipulation ends. `fingertip_paths.json` in the run
directory carries this display data, including when analysis logging is off.
Fingertip prediction and rendering are disabled by default. To enable them,
pass `--fingertip-paths` before `--`, or set `fingertip_paths=true` in the
experiment config. To change the prediction rate, pass
`--fingertip-path-hz 2` after `--`. Prediction uses the planning device by
default and adds computation there; `--fingertip-device cuda:1` after `--`
selects a separate device when available. Neither prediction nor its viewer
decorations send motor commands or modify MPPI's controls.

Live pose consumers project small tracker rotation drift onto the nearest
proper rotation: each rotation singular value must be within `1e-3` of one.
Translation stays unchanged. Non-finite data, reflections, bad homogeneous
rows and larger rotation/scale errors are rejected, with offending matrix
details in the error. Calibration matrices retain their stricter validation.
Pose and joint-message errors are separate; a bad object frame does not poison
joint-only preparation/reset feedback. A newer valid packet clears its
stream's error, and feedback metadata records `rejected_poses`. No invalid
packet refreshes the accepted pose timestamp or bypasses freshness checks.

Tracking and its receiver/SAM/container have a separate lifetime from the hand
experiment. Completing episodes, Ctrl-C, a MuJoCo window close or a controller
fault stops the hand controller and bridge but leaves the vision session running.
The next launch skips mask capture, registration, SAM/container startup and
tracker startup. It checks that the same physical mesh, camera calibration,
tracking options, channels and endpoints are in use, and waits for distinct
fresh tracked frames before starting hardware. Changing rollout models, horizon,
samples or hand/object rollout accuracy does not itself restart tracking.

```bash
# Reuse tracking; first-ever invocation starts it once:
python launch_leap_experiment.py --n-episodes 10

# Explicitly stop old tracking, recapture/review a mask, and track again:
python launch_leap_experiment.py --restart-tracking --n-episodes 10

# Release the camera, receiver, SAM and owned container; start no hardware:
python launch_leap_experiment.py --stop-tracking
```

The stop-only command checks the isolated `camera_hand_calibration/runtime/tracking`
registry if the default `.runtime/leap_tracking` has no saved session. An explicit
`--tracking-dir` is always honored. It prints the registry being stopped, or
reports that no saved session was found instead of claiming services stopped.

Closing the camera receiver stops manipulation but leaves the tracker running;
the next launch reopens only the receiver. If the tracker dies, its stream is
stale, or its configuration changes, the launcher does not recapture on its own:
use `--restart-tracking` when reinitialization is needed. Mask/bounding-box/SAM
capture options are ignored on reuse and take effect on an explicit restart.
The managed
session is recorded under `tracking_dir` (default `.runtime/leap_tracking`;
override with `--tracking-dir`). Only one manipulation launcher can use a
session at a time. Restart/stop targets the recorded process identities and
owned container ID, never unrelated camera or Docker services.

To reuse a mask and skip SAM-HQ/mask generation, pass
`--mask iris-foundationpose-plusplus/tmp/mask.jpg`. The live placement preview
still opens; Space/Enter starts tracking after validating the existing mask,
without overwriting it. New masks use lossless PNG;
tracking thresholds JPEG masks so compression artifacts do not become object
foreground. With `mesh=null` and `mesh_scale=null`, the final controller
`--task` selects the corresponding physical visual mesh and metric scale for
both tracker and viewer: 7 cm cube, 9.7 cm diameter ball, or the reference duck.
Tracking uses the eval object's body frame and accurate visual mesh regardless
of the rollout's hand/FOAM approximation. The mesh's material/texture files are
retained. Override with `--mesh` and `--mesh-scale` (or the matching JSON keys)
for a different tracking asset; its units, origin and axes must match the
selected object, or be mapped with `T_tracked_from_object`. Custom controller
`--model` scenes require an explicit `--mesh`.

```bash
# Preview exact commands without starting Docker, camera or motors:
python launch_leap_experiment.py --dry-run --bbox 440 272 103 113

# Plan from measured states without publishing actions:
python launch_leap_experiment.py --bbox 440 272 103 113 --observe-only

# Forward the usual study parameters after --:
python launch_leap_experiment.py --bbox 440 272 103 113 -- --rollout-model M3 --hand-acc low

# Run five episodes, with a supervised reference/start/success/zero cycle:
python launch_leap_experiment.py --n-episodes 5

# Select duck costs/contact tuning, its physical tracking mesh, and FOAM rollouts:
python launch_leap_experiment.py --bbox 440 272 103 113 -- --task duck_reorient --rollout-model M3 --hand-acc med --obj-acc foam16a
```

Press **Ctrl-C in the launcher** to stop MPPI, gradually return all joints to
zero, and then disable torque. The launcher keeps the bridge and feedback
running during this return; no Enter confirmation or object tracking is needed.
The return uses the preparation speed and ±0.1 rad delta bounds, with a
30-second deadline (`episode_preparation.interrupt_timeout_s` in the hardware
config). A second Ctrl-C, stale joint feedback, or failure to reach zero ends
the return and requests immediate torque disable. **Q/Escape or closing either
viewer**, a failed child, or a stalled supervision stream also requests immediate
torque disable. The persistent vision services remain available for the next
experiment; use `--stop-tracking` to stop them. A bridge cleanup failure produces
a nonzero launch exit status.
The launcher does not stop an unrelated external SAM server/container.

Each run has a new `results/leap_experiment_.../` directory containing the
captured frame, generated mask and mask-overlay preview, paired registration
RGB-D frame, per-process logs, `controller.json`, `goal.json`, `mujoco_status.json` with the
latest displayed state and freshness status, and
`launch.json` with resolved commands/config, stop reason and exit codes.
It also identifies whether tracking was reused, which services this invocation
started, and the persistent tracking workspace/logs. On reuse, mask and paired
registration-frame provenance are copied into the new logged experiment folder.
Logging is enabled by default. `--log-dir PATH` creates a new timestamped run
folder inside PATH, while `--run-dir PATH` selects the exact new run folder.

To continue an interrupted batch, pass its timestamped run folder:

```bash
python launch_leap_experiment.py --continue-run results/path/to/leap_experiment_TIMESTAMP
```

The launcher restores the saved controller options and effective cost weights,
counts successful, dropped and timed-out episodes as completed, and runs only
the remaining episodes in a new sibling timestamped folder. Aborted attempts
are retried from reference preparation and the normal Enter prompt; an action
sequence is never resumed midway. Original logs and videos remain untouched.
Use the newest continuation folder with `--continue-run` after another abort;
the complete ancestry is counted once. A completed batch exits without starting
services. `--dry-run` previews the restored command. GPU overrides such as
`--mppi-gpu 2 --tracking-gpu 3`, `--restart-tracking`, and a new output directory
remain available; omit controller options, `--n-episodes` and `--overwrite-logs`.
Changed hardware/calibration/joint-reference inputs are rejected rather than
silently mixed into the batch. This restores recorded run settings, not an old
checkout of task/planner source code; start a separate batch after code changes.
Each new process keeps local episode/video numbers. `continuation.json` maps
them to batch episode numbers and records total completed/remaining episodes
and known aborted attempts. Normal configuration logs append only the new
process's episodes, so earlier episodes are not duplicated.

The controller also writes these analysis files in that folder:

- `steps.csv`: per-action position error (m), quaternion error (`1-dot^2` and
  angle in radians), object velocity norm, MPPI wall time, whole-iteration wall
  time, elapsed manipulation time, feedback ages and action publication status.
- `episodes.csv`: outcome and success, completed action count, final pose errors,
  time to success, manipulation duration, full episode-cycle duration, actual
  control frequency and MPPI timing statistics. CSV episode numbers start at 1;
  the original `controller.json` episode indices start at 0.
- `summary.json`: requested/completed/aborted episodes, success counts and rates
  over completed and requested episodes, aggregate MPPI times, controller wall
  time, resolved arguments, weights and planner/rollout configs.

`controller.json` retains complete measured states, commands, goals, calibrated
scene XML, hardware/camera settings, original controller arguments, per-step
metrics and partial results on interruption/error. A rejected plan is recorded
as `failed_step`, including the timing and state-age limits when freshness
checks reject it. If MPPI has no usable rollout costs, `failed_step` also records
the measured input state, activation estimate, invalid-cost counts and final
rollout-state validity. A failure-only `failed_plan_eXX_sXXXXX.npz` snapshot
preserves planner warm-start/noise state, sampled deltas, costs and final rollout
states for offline diagnosis. These device copies/file writes occur only after
a rejected plan; the successful planning path is unchanged.
Warmup time is separate from MPPI iteration timings.
Manipulation duration excludes reference preparation, user waits and resets;
cycle duration includes them. Controller wall time includes initialization and
cleanup up to the final results flush; `launch.json` additionally records the
launcher's wall time including vision startup and process shutdown.
The 0.064 s control step means 15.625 Hz in the prediction model; hardware
execution remains unpaced and its measured frequency is logged separately.
Background checkpoints use incremental JSON encoding while the hand holds;
the final flush runs after controller cleanup.

The launcher also maintains cumulative results in
`<log-dir>/configurations/<configuration-id>/` (under `results/` when
`--log-dir` is omitted). `parameters.json` identifies the configuration;
`runs.jsonl`, `episodes.csv`, `steps.csv` and `summary.json` combine its runs.
Each CSV row includes a run ID and the original run directory. The accompanying
`results.sqlite3` stores transactional copies of these records.

Matching configurations append by default, creating their logs if absent.
Model, object, geometry accuracy, planning parameters, weights, calibrated
scene, hardware/camera settings and tracking configuration determine the match.
Batch size (`--n-episodes`) and output paths do not split configurations.
Add `--overwrite-logs` **before** the `--` separator to replace the matching
configuration's cumulative results with the current run. Other configurations,
timestamped raw run folders and active tracking files remain available.
For example:

```sh
python launch_leap_experiment.py --logging --overwrite-logs \
  --log-dir results/planned/hardware/cube_reorient/hand-high_obj-high/M1 \
  --n-episodes 10 -- \
  --task cube_reorient --rollout-model M1 --hand-acc high --obj-acc high \
  --n-samples 256 --n-iterations 1 --ctrl-time-step 0.064 \
  --time-horizon 0.192 --noise-sigma 0.05 --temperature 1.0
```

The cumulative log is updated after controller/bridge shutdown, including
available partial results on interruption or error. Earlier runs made before
this cumulative logging feature remain in their raw folders and are not
automatically imported.

For a testing run, add `--no-logging` before the `--` separator. It disables
result collection, preparation traces and process logs. Camera captures and
supervision/status/goal files still work. Per-experiment temporary files are
removed after shutdown; a temporary workspace backing an active persistent
tracker is retained until `--stop-tracking` or `--restart-tracking` releases it.
Omit `--log-dir`/`--run-dir` in this mode. Direct
`run_leap_hardware.py` invocations also accept `--no-logging`.
`--overwrite-logs` requires logging to be enabled.

To save camera video, add `--record-video` **before** the launcher's `--`
separator (off by default):

```sh
python launch_leap_experiment.py --record-video --n-episodes 12 -- \
  --task cube_reorient --rollout-model M2 --n-samples 512 \
  --warm-start shift --ctrl-time-step 0.064 --time-horizon 0.32 \
  --temperature 0.1 --noise-sigma 0.05
```

Each run saves one `video/episode_01.mkv`, `video/episode_02.mkv`, etc., with
matching `episode_01.frames.csv` timestamp files and a shared
`video/recording.json` summary in its timestamped experiment folder. The clips contain
the raw RGB supervision feed, without the receiver's bounding-box/axis overlays.
The separate host recorder subscribes to the existing stream, so it works with
reused tracking without reopening the camera or recapturing a mask. Recording
includes one second holding the reference before the control loop and one second
holding the reached hand pose after it finishes. Preparation, Enter waits, the
remaining success/failure hold, and zero returns are not saved. There is no whole-run
`camera.mkv`. The recorder process starts after tracking readiness and remains
idle between episodes; persistent tracking remains running after it exits.

The controller includes episode windows in its existing 10 Hz viewer telemetry.
The recorder saves only camera timestamps and presentation times that the
controller has already confirmed as manipulation or stationary recording time. Newer frames wait
in bounded memory until confirmed or are discarded; this recorder-side delay
does not delay commands. Normal completion closes the window after the stationary
tail. Ctrl-C and feedback errors close it immediately, before supervised recovery/reset;
the normal one-second tail does not delay fault cleanup. Manipulation start/end
timestamps and timing metrics exclude both padding holds; separate
`recording_started_at_utc` / `recording_finished_at_utc` fields describe the video
window. Runs without recording have no added holds. Each clip has its own time
origin, so gaps between episodes do not generate repeated frames on disk.

FFmpeg on the host is required (`sudo apt install ffmpeg` on Ubuntu). It copies
the existing JPEG packets into a Motion JPEG Matroska video using `-c:v copy`:
there is no video re-encoding, image decoding or GPU work. The recorder runs at
reduced CPU priority and idle I/O priority when `ionice` is available. A bounded
eight-frame writer queue and two-message subscriber buffer discard frames under
load. Recorder errors are reported without stopping manipulation.

Playback defaults to 30 fps; `--video-fps 15` records at 15 fps to reduce disk
traffic. Source timestamps select output slots: missing slots repeat the last
JPEG and excess arrivals are subsampled, preserving elapsed playback time.
Each episode's `*.frames.csv` maps every video frame to its original
capture/receive timestamps, source sequence and repeat flag. `recording.json`
reports per-episode paths/outcomes/counts and aggregate received/written,
subsampled, queue-dropped, excluded and repeated frame counts. Transport-level drops
cannot be counted from this stream. A source timestamp gap exceeding five
seconds stops only the recorder, avoiding a large duplicate-frame burst.

Recording requires logging enabled; `--no-record-video` overrides a configured
default. The config equivalents are `record_video` and `video_fps` in
`configs/leap_experiment.json`. Shared CPU/disk contention can still affect
performance; compare recorded planning times and pose ages when enabling it.

For older whole-run recordings, `python split_leap_video.py <run-directory>` writes
one manipulation-only clip per completed episode under `video/episodes/`.
Preparation, operator waits, success holds and resets are excluded using the
logged episode start/finish and original camera timestamps. Clips retain the
MJPEG packets without re-encoding; each packet is verified against its source
SHA256 hash. Per-clip timestamp CSVs and `manifest.json` record the boundaries.
The original recording remains available. Use `--output-dir` to choose another
new destination; existing clip directories are not overwritten.

Adjust `startup_timeout`, `placement_timeout`, `supervision_timeout` and `shutdown_timeout` in the
experiment config as needed. Extra host weight/data paths can be shared with
the vision container using `docker.extra_mounts`; the full repository and
run directory are mounted at the same absolute paths inside and outside.
`tracking_ready_frames` (default 5) and `tracking_ready_max_age` (default 0.3 s)
govern the startup gate; a stale frame or a gap restarts its count. The bridge
is not started during slow initial registration/first-track inference. After
startup, the existing 2-second supervision timeout still applies. Automatic or
explicit role assignments override the legacy `docker.cuda_visible_devices`
fallback for SAM, mask capture and tracking.

Remove `--observe-only` to publish commands. Hardware episodes keep the object's
default target orientation fixed across all episodes: for the cube, WXYZ
`[1, 0, 0, 0]`. `--goal-quat W X Y Z` overrides that fixed orientation. Place
the cube in a different starting pose; the controller receives that pose from
tracking after the Enter gate. Simulation runs retain their goal sampler.
Position targets, drop penalties and cost weights retain the
object's task defaults. The hardware joint-home cost uses `robot_qref` in
`configs/leap_hardware.json`. `--goal-position X Y Z` overrides the position goal;
`--measured-reference` adopts measured starting hand/object positions as
home/target. Noise sampling, control mode, convergence, adaptive temperature,
uncertainty, substeps/control period, horizon/time horizon and cost-weight
flags come directly from the simulation driver.

For a per-run cube friction override, pass all three MuJoCo coefficients in
sliding/torsional/rolling order:

```bash
python launch_leap_experiment.py --n-episodes 10 -- \
  --task cube_reorient --rollout-model M2 --cube-friction 0.7 0.005 0.0001
```

Omitting `--cube-friction` retains the selected scenes' current friction values.
The override changes the cube's colliding geometry in the temporary calibrated
rollout and evaluation scenes, including when using `--model`; source XML files
and tracking remain unchanged. The arguments and generated XML
are logged, and different friction settings get separate configuration logs.
Hand collision surfaces have independent overrides, in the same three-value order:

```bash
--fingertip-friction 0.5 0.005 0.0001 \
--palm-friction 1.0 0.005 0.0001 \
--finger-friction 1.0 0.005 0.0001
```

Pass these after the launcher's `--`. Each flag is optional and leaves the other
groups at their selected scene values. `--fingertip-friction` selects all four
fingertip pads, including their capsule approximations at low/medium hand accuracy;
`--finger-friction` selects other finger and thumb collision surfaces and excludes
those pads. Palm, cube, floor, and noncolliding visuals are separate groups; the
hand flags do not change the cube, floor, or visuals. Hand overrides also work with
other object tasks. All overrides apply to both generated rollout/evaluation scenes.
The backend still applies its normal contact mixing and `condim`: in the current
MuJoCo/MJWarp cube scenes, equal-priority geoms use the larger friction values,
so lowering only the cube below the fingertips' sliding friction may have no
effect; `condim=3` activates sliding friction only.

Before **every hardware episode**, the controller gradually moves the hand to
the configured `robot_qref` (MuJoCo joint order/radians), holds it, and waits for
**Enter in the launching terminal**. It then requires fresh synchronized joint
and object poses before starting MPPI. `episode_preparation` configures the
reference motion: configured speed 0.35 rad/s, base tolerance 0.06 rad (3.44 degrees),
and command rate 50 Hz. `joint_tolerance_rad` overrides individual joints;
`if_mcp` uses 0.08 rad (4.58 degrees), covering its measured 0.0494–0.0617 rad
steady variation across episodes. Other joints keep the base tolerance.
`timeout_s=0` disables the preparation deadline: contact and placement nudges
keep the reference command active while the hand settles, rather than causing
a 30-second shutdown. A positive timeout remains available as an explicit
diagnostic limit. Acceptance additionally requires 0.3 s with
all joint speeds below 0.08 rad/s; this is a settling condition, not a command
velocity limit. The exact reference
vector and hardware gains are unchanged. The same settings govern each gradual return to zero; MPPI remains
unpaced with target increments bounded by `ctrl_range` (default ±0.1 rad).
Reference moves, reference holds, returns to zero and Ctrl-C zero returns use
absolute position waypoints, independently of `ctrl_range`. Their gradual
trajectory is governed by `episode_preparation.speed_rad_s` and `command_hz`.
While waiting for the start Enter and fresh object tracking, absolute commands
continue to maintain the reference target. The controller does not replace it
with the measured pose, which could cause another relaxation under load.
Nudges during the Enter wait do not start MPPI or restart pose preparation.
The next episode's automatic reference motion is labeled `Preparing episode`;
MPPI starts only after a separate Enter at its ready prompt. The bridge action
watchdog, feedback freshness checks and device fault checks remain active.
Before cube control starts, the terminal identifies the tracked top face, the
faces toward world +X/+Y, and the nearest signed 90-degree starting rotation
from the goal. The two I faces are distinguished as blue (+X) and pink (-X).
It also prints completed-start counts for the current run; aborted attempts do
not count. Off-protocol and edge-balanced poses are flagged. The description
is saved per episode as `starting_orientation` and uses the tracked quaternion,
not independent image recognition. With video enabled it refreshes after the
stationary lead-in. These labels assume the adjusted 70-mm cube mesh frame.
Per-joint preparation progress and timeout errors appear in the terminal,
`episode_status.json`, and `preparation_trace.jsonl` for diagnosis.

On reaching the selected task's success thresholds, hardware MPPI stops and
the bridge latches its measured joint pose once. Hold heartbeats keep that
same absolute joint target, rather than continually rebasing zero increments.
Press Enter after success to return gradually to the all-zero joint pose. The
next episode then moves gradually from zero to `robot_qref` and waits for Enter
again. After the final episode's zero return, an explicit stop packet shuts
down the bridge and verifies torque is disabled. Set the count with
`--n-episodes N` on either the launcher or the hardware runner (default 1).
If an episode times out or fails without a feedback/device fault, its hand
pose is held for Enter before returning to zero as well. A first Ctrl-C returns
to zero and disables torque. A second interrupt, viewer-close stop, and
device faults request immediate torque disable without a reset motion.
With `hold_on_non_motor_failure: true` (the current hardware default), planner
exceptions, controller process failures, and action timeouts retain the bridge's
last commanded target. The terminal asks you to remove the object and press Enter
before returning gradually to zero and disabling torque. Motor current,
temperature, hardware-error and USB checks continue during this hold; a motor or
communication fault still disables torque immediately. Recovery requires fresh
joint feedback. The launcher prints a short crash reason, with full details kept
in the process logs. Set this hardware option to `false` to restore immediate
shutdown for software failures.
The success summary is saved before waiting. Hardware always
holds on success; `--no-stop-on-success` retains its original simulation-only
behavior. Observe-only mode skips reference motion and operator gates.
The launcher prints phase prompts from `episode_status.json`; run it with
interactive stdin so Enter reaches the controller. Closing the supervision
viewer or interrupting the launcher still stops manipulation and performs bridge
torque cleanup. Tracking remains running for reuse.

Simulation mode uses a separate calibrated evaluation scene, the reference
settle duration and fine evaluation steps per control period. Hardware command
constraints apply only to live commands. This runner retains its hardware JSON
log, including partial runs, rather than the simulation driver's video/NPY
output options. `--no-results` disables saving.

The local bridge keeps the state wire format in
`~/comfree_dex/leap_hand_lcm_bridge.py`, using
`~/leap_agentic_sysid/third_party/LEAP_Hand_Hardware` (override `--hardware-api`).
It verifies controller registers in `configs/leap_hardware.json`, preloads the
measured pose before enabling torque, and checks device current, temperature
and hardware errors. During the serial loop it polls one health register every
1/32 second, alternating error and temperature for each motor. Joint feedback
runs between those transactions; both registers on all 16 motors are still
checked approximately once per second. Expired
queued action packets are discarded without application or watchdog renewal;
packet transport age consumes the remaining watchdog budget. After the first
action, a 0.5-second action timeout latches the software-failure hold described
above (or disables torque when that option is false). The bridge also disables
torque on exit. These faults stop manipulation;
they do not reshape commands. The older external bridge accepts absolute
targets and cannot execute the new delta command protocol.

The shared hardware connection temporarily sets the FTDI USB `latency_timer`
to **1 ms**, then restores **16 ms** after the port and motor cleanup finish.
These values are `usb_latency_timer_ms` and `usb_latency_restore_ms` in
`configs/leap_hardware.json`. The restore value is explicit: an adapter left
at 1 ms by a previous manual command still returns to 16 ms. Set
`usb_latency_timer_ms` to `null` to leave the adapter setting untouched.
The launcher starts a sysfs-only guard before detached hardware services;
standalone hardware utilities use the same guard through `HardwareHand`.
Closing the guard's private pipe restores the timer, including when its owner
is killed before running Python cleanup. The guard keeps the original device
file descriptor and never changes a replacement device after unplug/replug.
It performs no periodic polling or work in the motor loop, and makes no udev
or other persistent configuration changes. `launch.json` records its startup
value, requested timer and verified restoration.

If the timer requires root permission, run `sudo -v` in the same terminal
before the normal launch command. The helper uses noninteractive `sudo -n`
only when direct access is denied. Permission failure stops startup before
torque is enabled. `--dry-run` and `--stop-tracking` never change USB latency.

Add `--bridge-telemetry` to the launcher for joint/target/current and serial
timing diagnostics in `bridge_telemetry.jsonl` (requires logging). The bridge
sends samples through bounded, nonblocking local IPC to a separate low-priority
recorder process. JSON encoding, flushing and disk writes run there; a slow or
failed recorder drops diagnostic samples rather than blocking motor feedback.
`bridge_telemetry.status.json` reports accepted, written and dropped samples
and recorder errors after shutdown. Recording adds no serial reads, and does
not change planner or feedback freshness limits. Use `--no-bridge-telemetry`
to disable it. Recorder startup completes before connecting/enabling the hand;
its final flush runs after torque cleanup.
The controller owns no serial connection or torque control. Both motion and
hold heartbeats use versioned packets on `LEAP_ACTION_DELTA`; the bridge
distinguishes the 80-byte delta packet from the 16-byte `LEAPHLD1` hold packet
and `LEAPSTP1` shutdown packet.
The launcher passes the controller's `--calibrated-hand` file to the bridge
as well. Physical joint/actuator bounds remain in the planning model and the
manual viewer; manipulation commands are bounded only in delta space.

For a display-only transform check, start `leap_hand_lcm_bridge.py --state-only`
and `view_leap_tracking.py` alongside the tracker/receiver. The state-only bridge
reads joints without changing torque/registers and has no action subscription.
The MuJoCo viewer never steps physics or publishes motor commands. It displays
measured joints and the base-frame `POSE`, hides stale objects, and hides the
simulation floor so it cannot obscure a measured object on the physical table.
Use `view_leap_tracking.py --mesh PATH --mesh-scale SCALE` with the same OBJ as
the tracker; its raw origin, axes and MTL texture are preserved in the display.

`configs/leap_experiment.json` has a per-task `tracking_meshes` override for
`cube_reorient`, using
`iris-foundationpose-plusplus/meshes/cube_70_mm_2_adjusted/cube_70_mm_2.obj`
at scale **1.0** (the scan is already in meters). Ball/duck choices retain
their own meshes. Explicit `--mesh`/`--mesh-scale` take precedence.
The default vision image's baked FoundationPose weights/extensions live under
`/FoundationPose-plus-plus/FoundationPose`; `docker.foundationpose_root` passes
this path through `FOUNDATIONPOSE_ROOT`, while this checkout provides the
calibrated camera and LCM integration. Its SAM-HQ checkpoint is configured
under `sam_hq.checkpoint_path`.

MPPI incremental hardware bounds are configurable in `configs/leap_hardware.json`:

```json
"ctrl_range": {
  "lower": 0.1,
  "upper": 0.1
}
```

Both fields are **nonnegative delta magnitudes in radians**. The base for each
increment is selected by one reversible hardware setting:

```json
"delta_reference": "previous_target"
```

The checked-in configuration enables `previous_target`:
`next_target = previous_target + clip(raw_delta, -lower, +upper)`.
Zero delta preserves the commanded target even if measured joints sag. This
bounds each target increment, rather than the target's distance from measured
joints; no additional tracking-error or command-velocity cap is introduced.
Reference waypoints seed the target before each episode, zero returns reset
it, and explicit success/operator holds latch measured joints once as before.
MPPI uses the standard planner's
`ctrl_relative` mode, with the bridge's reported target as its starting control.

To restore the previous behavior, change only this setting to
`"delta_reference": "measured_position"` and relaunch. That mode uses
`next_target = measured_position + clip(raw_delta, -lower, +upper)` and the
standard planner's `pos_relative` mode. Omitting the setting also selects this
legacy mode. Hardware mode is resolved from this shared configuration; an
explicit conflicting `--control-mode` is rejected so bridge and rollout
semantics cannot silently differ. Simulation retains its normal CLI modes.

In either mode, the controller sends the raw planner delta, without applying
`ctrl_range`. The bridge forms `requested_target = base + raw_delta`, then
clips only the final target to `[base - lower, base + upper]`. The controller
predicts that final target for its servo observer/logs; its prediction does
not change the outgoing delta. For example, with a base of `1.7` and ±0.1
limits, a requested target of `2.0` becomes `1.8`.

Each side accepts a scalar for all joints or a list of 16 nonnegative finite
values in MuJoCo order: index, middle, ring `{mcp, rot, pip, dip}`, then thumb
`{cmc, axl, mcp, ipl}`. Missing/null values default to `0.1`; zero freezes that
side of a joint's increment. Negative magnitudes are rejected.
These bounds apply only to final hardware targets during MPPI manipulation.
They do not constrain
reference preparation, reference holds, zero resets, Ctrl-C zero returns or
manual joint-slider control. Manual control retains its own
`manual_viewer_limits` and physical joint/actuator bounds.

Live hardware uses the unchanged standard MPPI planner with the configured
relative-control mode. `ctrl_range` does not affect sampling, optimization or rollout control
sequences. The existing planner option `--delta`, if explicitly passed, retains
its standard symmetric sampling bound independently of hardware limits.
There is no hardware-specific planner wrapper or merging of these limits.
LCM carries the raw delta, and the bridge clips the final absolute hardware
target using the selected reference. Cached bridge commands remain absolute
targets, so repeated USB writes never accumulate another delta. Duplicate and
out-of-order action timestamps are ignored; only the newest queued delta is used.
`LEAP_ACTION_DELTA` packets are 80 bytes: the eight-byte `LEAPDLT1` marker,
an int64 little-endian epoch-microsecond timestamp and 16 float32 little-endian
deltas in MuJoCo order. The bridge rejects the old 72-byte absolute-action
packet. The launcher configures the same delta channel for both processes.
Preparation/reset uses an explicit 80-byte `LEAPPOS1` packet on that same
ordered action channel: an epoch-microsecond timestamp and 16 absolute joint
positions. The bridge writes these waypoints directly, without applying MPPI
delta bounds or rebasing on measured joints. Waypoints share action timestamp,
ordering, watchdog and device-fault checks with manipulation commands.

The step log stores the raw `action_delta`, the unbounded `requested_target`,
the planning state, and the controller's latest
`feedback.command_qpos`/timestamps. Its `u` is an estimated final clipped
bridge target for the servo observer; the bridge can have newer measurements
when it applies the delta. `action_representation="joint_delta"` identifies
the transmitted command, and `u_source="estimated_bridge_target"` labels the estimate.
In previous-target mode the bridge also publishes its held target each serial
loop, after any requested write, on `<state_channel>_TARGET` (default
`LEAP_STATE_TARGET`). Its 80-byte `LEAPTGT1` packet contains a report timestamp
and 16 target positions; it stays fresh during controller warmup too.
The original joint-state packet is unchanged. The controller requires fresh
target feedback before planning, uses it as the control-relative base, and
records `plan_base` plus target feedback in each step. Dropped commands are
therefore reconciled from the bridge's target rather than blindly integrating
a controller estimate. This feedback adds no serial reads or writes.

By default the controller publishes each completed plan immediately, without a
control-period sleep. Add `--pace-control` after the launcher's `--` separator
to limit new targets to `1 / ctrl_time_step` (15.625 Hz for `0.064` seconds).
Fast plans wait for the remaining interval before reading fresh command feedback;
overruns publish immediately and start a new interval, without catch-up bursts.
Pacing preserves plan/state freshness limits. Logs record `hardware_loop_paced`,
the frequency limit, each step's `pacing_wait_s`, and its publication timestamp.
Omit the flag (or use `--no-pace-control`) for maximum throughput.

Optional temporal blending compares overlapping optimized **delta** sequences
from two consecutive plans. Pass controller flags after the launcher's `--`:

```bash
--plan-blend-alpha 0.6
```

This publishes `alpha * new_delta[0] + (1-alpha) * old_delta[k]` before the
existing bridge clipping. `k=floor(elapsed/control_period)` uses elapsed
monotonic time since the old plan's publication, including current planning and
pacing time. Rows are zero-order-held for the resolved rollout control period.
This is approximate command-time alignment, not a prediction of the changing
measured state or serial application time. An expired old horizon contributes
nothing; the first plan of each episode is also unblended. The old sequence is
the raw prior optimizer result, not a recursively filtered command. Multiple
historical plans and exponential weighting are not implemented.

The default `--plan-blend-alpha 1` bypasses blending entirely. Add
`--plan-blend-metrics` to collect an unblended baseline: step JSON then records
`plan_blending.within_plan_diff_l2`, `aligned_plan_diff_l2`, the raw mean,
overlap age/slot, transmitted delta, and estimated clipping counts/magnitudes
before and after blending. Enabled blending records these automatically when
logging is on. These norms sum across joints in delta radians; they are not
absolute-target differences or velocities. Clipping metrics use the controller's
fresh measured/held base and are labelled estimates because the bridge may read
newer joints. No additional serial transactions, LCM messages or physics
rollouts are added; optional NumPy work/logging still has CPU overhead.

Warm starting, sampling noise, planner costs and optimizer output remain
unchanged. In particular, `shift` still shifts the previous mean by **one**
control step per planning call; its initialization can differ from blending's
wall-time alignment when planning overruns. Future absolute targets in
`last_action_seq` assume fixed joints in position-relative mode, so they are
not used for blending. Exact absolute-target blending would need a rollout of
the final weighted mean. Fingertip previews continue to display the raw
optimizer's mean, not the blended command. Blending is experimental: it changes
the executed action after scoring and can suppress useful corrections as well
as stochastic changes. It does not compensate for planning delay or guarantee
less clipping or better manipulation. Hardware command blending/metrics require
`--hardware` without `--observe-only`; safety bounds are unchanged.

The bridge has no `--rate` timer: it reads feedback, handles the newest
action and writes the held target as fast as the serial transactions complete.
There is no software command-speed cap, measured-velocity cutoff or tracking-error
cap. MPPI targets first receive the configured delta clamp around the selected
reference, then the optional measured joint-range clamp described below. Actual throughput is
set by MPPI computation and serial/LCM/device transactions. The configured control
period defines rollout physics and horizon timing in both modes, and optionally
limits hardware target updates with `--pace-control`. The manual viewer retains
its separate 50 Hz and 0.5 rad/s defaults.

Model XML is not overwritten with delta magnitudes.
`effective_ctrl_range` in the results records positive per-joint target margins,
`reference="previous_target"` or `"current_joint_position"`, and
`application="final_hardware_target"`;
`model_ctrl_range` records the project's joint/actuator-range intersection;
it is distinct from native actuator ctrlrange and from measured joint bounds.
Hardware settings apply to live/observation mode, while
simulation retains the reference driver's control parameterization and limits.
The logged `control_policy` distinguishes this final-target policy from older
runs that clipped planner samples, including in configuration-specific logs.

`configs/leap_hardware.json` now enables `joint_limits` using
`scenes/leap/hardware/calibrated_hand_measured_limits.xml`. Its capture record
must indicate all 16 joint ranges were saved, with the matching XML hash.
The hardware workflow transplants these joint ranges into generated rollout/eval
scenes, retaining the selected calibrated dynamics and actuator ctrlranges.
Original model XML files are unchanged.

The added planner cost is `weight * sum_hj (max(lower_j-target_hj, 0)^2 +
max(target_hj-upper_j, 0)^2)`, using raw requested absolute targets at every
horizon step, including the last, before actuator filtering. The default
`joint_limits.cost_weight` is `100.0`; controller `--joint-limit-weight VALUE`
overrides it. Through the launcher, put this option after `--`. A weight of zero
disables only the cost; the bridge still clamps MPPI delta targets to the measured
joint ranges after `ctrl_range`. Setting the entire `joint_limits` entry to
`null` disables both features and measured-range transplantation. Sim-to-sim
defaults remain unchanged.

The final absolute clamp takes priority if the measured pose is already outside
its interval, so that correction can exceed the delta bound. Preparation/reset
position waypoints and measured-pose hold commands retain their existing behavior:
zero and parts of `robot_qref` are outside the captured manipulation ranges.
Controller logs record the bounds, cost definition and estimated per-step
absolute clipping; optional bridge telemetry records the target after each clamp.

Rollouts reset servo activation from an observer driven by estimated applied targets;
the initial estimate assumes a holding pose. Differences between controller and
bridge measurements, packet timing and target-feedback latency can still
introduce observer error. CPU simulation uses its exact activation. MJWarp-family CUDA
backends use their required pyramidal friction cone.

The `hardware` dependency extra adds `lcm` and `dynamixel-sdk`. Vision setup
remains documented in `iris-foundationpose-plusplus/Install.md`. Packages and
camera models are not installed automatically. Use compatible backend versions.
Regenerate the original calibrated hand/cube export with
`python prepare_leap_hardware_scene.py` in MuJoCo **3.6.0**, compatible with the
supplied binary. It accepts `--archive`, `--hand-source`, `--output`, `--cube-size`
and `--cube-mass`. The generated XML removes the binary-version dependency
from normal runs; `provenance.json` records calibration hashes/checks. Normal
runs apply this hand calibration to selected study scenes without changing
their source templates.

Cube rollout scenes (all hand accuracies), the evaluation scene, and the
standalone hardware scene use the measured cube mass of **0.07 kg (70 g)**.
The scene generator also defaults to 0.07 kg. Mass and inertia are inferred
from the mass-carrying box geom; the visual mesh has zero density. These
rollout/evaluation templates are shared by the hardware and sim-to-sim drivers.
Earlier recorded experiments used 0.14 kg; their saved scenes remain unchanged.

## Repository Structure

```
contact_study/
├── contact_study/
│   ├── contact_models/
│   │   ├── config.py           # ContactModelConfig + M1..M4 factories
│   │   ├── api.py              # Unified dispatch surface (put_model/step/forward)
│   │   ├── xpbd_backend.py     # M4: XPBD-style contact model
│   │   └── benchmarks.py       # Speed and approximation error measurement
│   ├── planners/
│   │   ├── mppi.py             # MPPI controller
│   │   └── cem.py              # CEM controller
│   ├── tasks/
│   │   ├── base.py             # BaseTask, TaskSpec, task registry
│   │   └── tasks.py            # PushTask, GraspReorientTask, PegInHoleTask
│   ├── evaluation/
│   │   ├── metrics.py          # EpisodeResult, AggregatedResult, serialization
│   │   ├── trajectory.py       # per-control-step state / control / planner-belief recording
│   │   ├── distributions.py    # first-action moments of a planner, Gaussian KL
│   │   └── json_io.py          # JSON writer that keeps bulk arrays on one line
│   └── utils/
│       └── physics_noise.py    # PhysicsNoiseParams + apply_physics_noise
│
├── scenes/
│   └── leap/                   # scene variants, named by convention
│       ├── env_leap_eval_{cube,duck}.xml            # reference fidelity
│       ├── env_leap_rollout_{obj}_{hand}_{obj}.xml  # degraded planner scenes
│       └── leap_right_hand{,_eval,_capsules}.xml    # hand fidelity rungs
│
├── experiments/
│   ├── run_experiment.py       # Main study runner (tasks × models)
│   ├── benchmark_speed.py      # Throughput benchmark vs batch size
│   └── measure_approx_error.py # Approximation error vs horizon
│
├── analysis/
│   ├── plot_results.py         # All paper figures
│   └── README_kl_divergence.md # KL-vs-success workflow and diagnostics
│
└── tests/
    ├── test_allegro.py
    └── test_primitives.py
```


## Installation and system setup

### What the machine needs

- **Linux with an NVIDIA GPU and its driver.**
  - The rollouts (M1–M6) all run on the GPU through Warp. Warp ships its own
    CUDA runtime, so only the driver is needed, not the CUDA toolkit.
  - Used on an RTX 4090 (driver 580, CUDA 13.0) and on the HPC's RTX 5000 Ada.
  - Without a GPU, only the CPU parts run, and the tests skip the rest.
- **conda** (Miniconda or Anaconda). Pinocchio and coal come from conda-forge.
- **ffmpeg** on the `PATH` for videos (mediapy calls it). The recipes below
  install it into the env. Without it, pass `--no-video`.
- **On a headless machine**, MuJoCo renders through EGL, which comes with the
  NVIDIA driver.
  - The episode drivers set `MUJOCO_GL=egl` themselves when there is no
    `DISPLAY`.
  - Set it yourself for anything else that renders: `export MUJOCO_GL=egl`.

### Which environment

On this branch every model, M1–M6, runs in one env, `contact_comfree2`:

| Env | Python | Warp | MuJoCo | Runs |
| --- | --- | --- | --- | --- |
| `contact_comfree2` | 3.12 | 1.17 | 3.14 | M1–M6, CPU MuJoCo, Pinocchio, Drake |

- **Each model's package:** M1, M2 and M4 run on the standalone `mujoco-warp`
  3.14, M3 on `comfree_warp_v1` (M3's ComFree 0.1, renamed), M5 on Newton 1.6,
  and M6 on `comfree_warp` 0.3 (ComFree-Sim).
- **Why MuJoCo 3.14:** M6 needs it. Python 3.12 and Warp 1.17 are for M5:
  Newton's Kamino code does not import on Python 3.10, and Newton needs
  Warp ≥ 1.17.
- **main's envs don't run this branch.** `contact_kamino` and
  `contact_modeling` (MuJoCo 3.6) have neither `mujoco_warp` nor
  `comfree_warp_v1`. `results/bo_check` came from `contact_kamino`;
  `results/bo_check_one_env` is the same runs in `contact_comfree2`.

### Building `contact_comfree2`

From the repo root:

```bash
conda create -n contact_comfree2 -c conda-forge python=3.12 pinocchio=4.0.0 coal=3.0.2 numpy=2.2 ffmpeg
conda activate contact_comfree2
pip install drake==1.51.1 mediapy pytest
pip install -e ".[kamino]"     # MuJoCo 3.14, MJWarp 3.14, Warp 1.17, Newton 1.6, comfree_warp_v1 (pinned), skopt, ...
gh repo clone asu-iris/comfree-warp-simbenchmark /path/to/comfree-warp-simbenchmark
pip install --no-deps /path/to/comfree-warp-simbenchmark   # comfree_warp 0.3.0, for M6
```

- **Warp stays at 1.17.** ComFree-Sim pins 1.15, but Newton (M5) needs 1.17,
  and ComFree-Sim runs on 1.17 unchanged. `--no-deps` keeps pip from
  downgrading it, and `pip check` reports the mismatch.
- **The env on the lab's 4090 was upgraded in place** from a clone of
  `contact_kamino`, not built this way: `pip uninstall comfree_warp`, then
  `pip install mujoco==3.14.0 absl-py "etils[epath]"`,
  `pip install --no-deps mujoco-warp==3.14.0`, and the two ComFree packages.
- Pinocchio and Drake are needed only for those eval simulators. Without
  them, the tests that use them skip.

### What MuJoCo 3.14 changes

- **M5 is unchanged.** Kamino's trajectories on all three eval scenes under
  hold, shake and random-walk controls are bit-identical to `contact_kamino`'s.
- **M1–M4 are not.** MuJoCo 3.14 compiles the mesh scenes slightly
  differently: geom orientations move by up to 0.25°, sizes by about 0.04 mm,
  and the duck's hull is rebuilt. M1, M2 and M4 also move to MJWarp 3.14's
  collision and solver.
  - Over 200 steps of random hand motion, the cube and the duck end
    0.1–4 mm (mean) from where `contact_kamino` puts them, against
    run-to-run noise of 0.06–0.26 mm.
  - The ball compiles identically, and M3 and M4 are bit-identical on it.
  - So compare M1–M4 results only within one env.
- **M4 (XPBD) was ported to MJWarp 3.14.** 3.14 stores M as CSR and factors
  it per block, packed, so XPBD densifies M̂ and factors it itself for the
  per-row effective mass. It still uses MJWarp's factor for the per-world
  solves.

### The ComFree dependencies

- **`comfree_warp_v1` (M3)** is installed from this repository's
  `comfree-warp-v1` branch. That is upstream `asu-iris/comfree_warp@ba8b996`
  with:
  - a fix to its vendored MJWarp `sensor.py`, without which Warp ≥ 1.16 does
    not compile it (the `comfree-warp1.17` branch, which main pins);
  - the package renamed to `comfree_warp_v1`;
  - `EnableBit.MULTICCD` falling back to 0, since MuJoCo 3.14 dropped
    `mjENBL_MULTICCD`.
- **`comfree_warp` 0.3 (M6)** is the private
  `asu-iris/comfree-warp-simbenchmark`, tested at `0a8b385`.

The `comfree-warp*` branches stand apart from the study code: they share only
the early comfree history. **Never merge them into the study branches.**

To edit M3's comfree, check that branch out somewhere else and install it over
the pinned one:

```bash
git clone -b comfree-warp-v1 https://github.com/yifanzhu95/contact_model_study.git /path/to/comfree_warp_v1
pip install -e /path/to/comfree_warp_v1 --no-deps
```

### Checking the install

```bash
python -c "import mujoco, mujoco_warp, warp, comfree_warp_v1, comfree_warp; print(mujoco.__version__, mujoco_warp.__version__, warp.__version__, comfree_warp.__version__)"   # 3.14.0 3.14.0 1.17.0 0.3.0
python -c "import warp; warp.init()"         # lists the CUDA devices Warp can see
python -m pytest tests -m "not slow"         # a few minutes; the full suite is longer
```

- On an RTX 4090, `python -m pytest tests` took about 4.5 minutes in
  `contact_comfree2`, with the Warp kernels already compiled.
- The first run of each GPU model also spends a while compiling Warp kernels.
  They are cached afterwards.

### On the HPC

- Build the env once on the login node, the same way, after
  `module load miniconda`.
- The submit scripts in `experiments/hpc/` and their `.slurm` jobs activate
  `contact_comfree2` by default. Set `CONDA_ENV` when submitting to use
  another env, for example main's `contact_kamino`:
  `CONDA_ENV=contact_kamino bash experiments/hpc/submit_episode_batches.sh ...`.
  The submit script exports it, so the jobs activate the same env.
- See `experiments/README.md` for the submission workflow.

## What has been implemented and tested so far

1. Contact models M1-M4, with throughput checks on primitives and Allegro scenes.
2. Duck `grasp_reorient` MPPI closed-loop smoke tests on the four FOAM sphere
   scenes, using the fixed eight-hull Duck as the eval model.
3. An isolated Newton 1.6 / SolverKamino offline-reference subproject under
   `kamino_reference/`.  It consumes the same fingerprinted state, goal, cost,
   timing and candidate tape as M1--M4 without mixing the incompatible Warp and
   MuJoCo dependency stacks.  Start with
   `kamino_reference/docs/collaborator_handoff.md`.

## What needs to be done next

1. Tune the Duck MPPI/controller on development seeds until it has nonzero
   success, then freeze the parameters and run the predeclared multi-seed test.
2. Repeat a subset of seeds to quantify non-bitwise-deterministic contact
   variation, then test physics parameter noise (geometry fidelity is wired).


---
## Quick Tests
### Test throughtput of different models with and without the viewer in the Allegro Hand Cube Scene

Run tests/test_allegro.py, see file for options

### Test the viewer and throughtput of different models of the primitives scene
Run tests/test_primitives.py, see file for options


## Usage for benchmarks (Not Tested Yet)

### 1. Speed benchmark (clean)

```bash
python experiments/benchmark_speed.py \
    --task push \
    --models M1 M2 M3 M4 \
    --batch_sizes 64 256 1024 4096 \
    --horizon 50
```

### 2. Speed benchmark with a higher-fidelity rollout hand + noisy physics (old "M10")

```bash
python experiments/benchmark_speed.py \
    --task grasp_reorient \
    --models M4 \
    --geometry cube_high_high \
    --friction_sigma 0.2 --mass_sigma 0.1
```

### 3. Approximation error

```bash
python experiments/measure_approx_error.py \
    --tasks push grasp_reorient peg_in_hole \
    --models M1 M3 M4 \
    --horizons 5 10 20 40 \
    --n_states 50
```

### 4. Full study, clean baseline

```bash
python experiments/run_experiment.py \
    --tasks push grasp_reorient peg_in_hole \
    --models M1 M2 M3 M4 \
    --n_episodes 20 \
    --n_samples 1024
```

### 5. Full study cell: high-fidelity rollout hand + friction noise

```bash
python experiments/run_experiment.py \
    --models M1 M2 M3 M4 \
    --geometry cube_high_high \
    --friction_sigma 0.2 --mass_sigma 0.1 \
    --output results/cell_cube_high_high_noisy.json
```

To sweep over the full old-M1..M10 grid, wrap this invocation in an outer shell
loop over `--geometry` and `--friction_sigma` values.

### 6. Figures

```bash
python analysis/plot_results.py results/experiment_TIMESTAMP.json
```
