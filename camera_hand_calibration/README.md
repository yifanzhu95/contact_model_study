# Marker-free camera / LEAP calibration workspace

Everything created for this study lives in this directory: code, frozen model
and mesh assets, configuration, raw data, annotations, dependency pins, logs,
test fixtures, caches, fit candidates and plots. Production camera, hand and
controller files are never modified. Run with the existing **mj_env** Python.

Camera-only capture and the smooth articulated sweep were explicitly authorized.
The sweep completed and all 16 motors were verified torque-off afterward.
Any additional motion must remain within the operator's authorized scope.
Candidate transforms remain isolated and have not replaced production values.

The newest independent experiment is indexed in
`results/articulated_latest.json` and `results/rigid_surface_latest.json`.
It combines fresh fixed-palm natural housing correspondences, aligned depth,
and measured encoder poses of moving rigid housings. The palm paper tag is
left in place, explicitly masked before processing, and never detected or used
as a landmark. No covering or removal is needed.

```bash
# Complete offline fit and comparison; no hardware access or motion:
python camera_hand_calibration/rigid_surface_fit.py \
  --config configs/rigid_surface_fit.json
```

The joint RGB-D candidate improves the independent housing image/depth checks,
but narrowly misses the predeclared 3 mm held-out moving-depth criterion
(about 3.05 mm). It is saved as `joint_candidate.json` for review, with
`production_ready: false`. Do not treat it as established absolute accuracy.
Reports include rejected surface-only, depth-bias and palm-only alternatives.
The existing object-only -15 mm tracking correction is excluded from hand
calibration and would require separate object validation before deployment.

Raw motion data are in `datasets/articulated_20261119T033633_480690Z`:
27 settled waypoints, 1,982 saved RGB-D frames, measured joint/current/target
logs, timing and torque-cleanup status. `articulated_audit.py` recreates the
timing/clearance/current audit without hardware access. An earlier acquisition
attempt failed during initialization, before torque enable; it is retained.

The geometry audit corrected three inner-notch versus outer-corner
correspondences in the earlier stationary study and an opposite-face choice
in the initial moving-housing diagnostics. Old artifacts remain intact;
use the latest report rather than interpreting a low old residual as proof
of calibration accuracy. The final fitter verifies the visible face against
the actual frozen mesh and camera direction. Current camera/model/config
hashes still match the original snapshots.

## Live object tracking with calibration comparison

```bash
python camera_hand_calibration/live_tracking.py
# Explicit mask recapture/reinitialization:
python camera_hand_calibration/live_tracking.py --restart-tracking
# Stop this workspace's persistent vision services:
python camera_hand_calibration/live_tracking.py --stop-tracking
```

The launcher runs a strictly read-only encoder source, the existing supervised
mask/registration/tracking pipeline, and an interactive MuJoCo hand/cube viewer.
It starts no MPPI controller, subscribes to no motor action channel, and never
enables torque. A fresh launch displays the bounding-box preview; Space/Enter
captures the mask, Enter accepts it, and R returns to placement. SAM-HQ is
released after mask acceptance. Vision services persist when the comparison
viewer closes; the read-only hand source closes its serial connection.

With the **MuJoCo window focused**:

| Key | Display |
|---|---|
| F6 | Original production calibration |
| F7 | Corrected stationary camera + depth candidate |
| F8 | Joint camera + depth + measured-pose candidate |
| F9 | Toggle the existing camera-axis object-only correction, uniformly across modes |
| F10 | Method 4 known-cube anchor, when available; starts with object correction OFF |
| Close window | Close the comparison viewer |

Mouse orbit, pan and zoom stay interactive and are not reset when switching.
The selector transforms the **same live published pose** locally: it undoes
the tracker's original publication transform, then applies the selected
camera-to-world transform. It does not republish object poses or alter tracking,
production configs, or hand feedback. The original mode reproduces the current
pipeline, including its -15 mm object correction by default. Use F9 to turn it off to
inspect the raw calibration without that object correction. Mode 2 uses the
corrected fixed-palm candidate, not the earlier candidate with the audited CAD
correspondence mistake. Both candidate modes remain provisional.

Numbers, Tab, O, Q and Escape retain their native MuJoCo actions; the comparison
handler responds to F6--F9, and F10 selects method 4 after its
known-cube experiment is completed. See `method4/README.md` and
`method4/latest_capture.json` for the complete supervised capture and report.
Method 4 currently fits its cube reference but worsens the independent rigid-palm
check; it remains an experimental candidate and has not replaced production.
Escape restores the free camera instead of closing this viewer.
The object geom is explicitly a textured MESH,
not a BOX with a mesh's texture atlas.

`configs/live_tracking.json` pins the calibration candidate files, object mesh,
camera settings, image bounding box and channels. It uses the hardware
experiment's original `cube_70_mm_2_adjusted/cube_70_mm_2.obj` and materials
read-only, at scale 1.0, with the same `[440, 272, 103, 113]` placement box.
All copied object assets,
compiled viewer cache, logs, runtime registry and status files stay here.
`runtime/live_tracking_latest.json` points to the current run; its
`mujoco_status.json` records the selected calibration, correction flag and live
world-frame object/hand state. GPU selection defaults to the least-loaded card;
`--gpu 2` selects a physical GPU explicitly.

## Layout

```text
calibrate.py                 CLI entry point; configures local caches
workspace.py                 confined paths, hashes, frozen inputs
capture.py                   RGB-D acquisition + optional passive LCM states
geometry.py                  projection and robust six-DOF pose estimation
study.py                     CAD catalog, annotation, fitting and validation
quality.py                   camera timing / profile / depth plots
configs/experiment.json      editable acquisition, matching and fitting options
configs/source_camera.json  immutable copy of the existing camera configuration
configs/current_transform.json  effective prior T_world_from_camera
configs/source_hardware.json    provenance only; never opens its serial port
model/                      frozen XML, MJB, actual meshes, SHA-256 manifest
datasets/<UTC session>/      lossless RGB-D, per-frame clocks, optional joints
results/<fit>/              candidate transform, residuals, overlays, recipe
logs/                       every CLI invocation, exceptions and test output
tests/                      hardware-free synthetic and protocol tests
cache/                      local matplotlib / pytest / other caches
requirements.txt            versions present in mj_env when bootstrapped
```

Session names and metadata use UTC (`Z`). User-facing local time is Phoenix.
All normal CLI paths are relative to this directory, even when launched from
the repository root. Absolute paths and symlinks resolving outside it are
rejected. Only `bootstrap` reads original repository inputs, once; it refuses
to overwrite a completed initialization. Do not edit the frozen model/assets:
hash verification will fail. Start a new workspace snapshot for a new model.
`bootstrap --model-source ... --camera-source ... --hardware-source ...` can
choose different source files when initializing a fresh copy of these scripts.

## Camera-only commands (safe before hardware confirmation)

From the repository root:

```bash
conda activate mj_env
python camera_hand_calibration/calibrate.py probe
python camera_hand_calibration/calibrate.py capture \
  --frames 600 --pose-id camera_only_check
```

The command prints a session path such as `datasets/20261118T065346_240673Z`.
Use the path printed for your actual run:

```bash
python camera_hand_calibration/calibrate.py inspect \
  --session datasets/20261118T065346_240673Z
python camera_hand_calibration/calibrate.py overlay \
  --frame datasets/20261118T065346_240673Z/frame_00000.npz
```

Headless capture is the default; `--preview` opens a camera window. The camera
must be available: do not run concurrent tracker/mask capture. The program
does not stop existing services, reset devices or change exposure/white
balance presets. It temporarily enables SDK global timestamps if supported,
records all camera settings and restores the original global-time setting.

Frames are depth-aligned to RGB by the RealSense SDK. RGB stays RGB8; depth
stays lossless aligned Z16 with its measured meters-per-unit scale. The stream
profile includes serial, firmware, both intrinsic models, raw depth-to-color
extrinsics, distortion, actual FPS and sensor options. Camera intrinsics are
read from the active profile, rather than blindly taken from the prior.

The capture loop queues lossless frame writes and JSONL logging to a worker.
It logs source frame-number gaps and clock/skew metrics. A full writer queue
or writer error fails the run; it does not silently omit requested records.
`frames_saved` and `writer_frames` must agree. Capture does not depend on SAM,
FoundationPose, Docker, a GPU, a tag detector, or any hand device.

`session.json` records the effective configuration, package versions, code
hashes, camera profile and frozen model manifest. New sessions also copy the
code and effective configuration. `timing.jsonl` records every received frame;
`frame_*.npz` files are sampled at `save_every_n_frames` (default 12, about 5 Hz).
Sampling saves disk and supplies separate static views; it does not mean that
the camera itself runs at 5 Hz.

The first smoke recording predates the final source-copy enhancement; its
original code hashes and configuration remain recorded. It is retained as an
acquisition test, not a calibration dataset.

## Frame conventions

The candidate is **T_world_from_camera**, in meters, where the camera is the
RGB OpenCV optical frame: +X right, +Y down, +Z forward. World is the frozen
MuJoCo model world, including the hand's mounting transform. Projection uses
the inverse transform. Intrinsics are fixed during extrinsic fitting.

The initial prior includes the production camera-to-published transform,
published-to-model transform, and model-frame rotation/translation adjustments.
It explicitly **excludes the -15 mm object-only camera-axis correction**.
That offset is a tracker/object correction and is never applied to the hand.

`--initial scratch` estimates extrinsics without using the old extrinsic
transform. `--initial current` starts local refinement from the frozen prior.
The existing tag-derived prior is only a historical input. No AprilTag or
ChArUco detection is implemented or used. Leave the physical tag paper in place
and exclude its pixels when identifying natural CAD features; paper corners
are not hand landmarks.

Current geometry fitting requires pinhole/rectified RGB with zero distortion.
The connected camera meets this condition. Capture retains nonzero distortion
metadata, but annotation/fitting rejects such frames until explicit
rectification is added; it never silently applies the wrong camera model.

## Natural landmarks and model-based fitting

The original `calibrate.py fit` optimizer uses **human-confirmed natural 3D/2D features** and
robust PnP/least squares. Aligned depth provides independent residual checks;
it is not used to pull the pose toward unreliable boundary depth. Fully
automatic silhouette matching / dense ICP is not included in this first toolset.
The separate stationary RGB-D fitter below also uses interior depth patches
in its objective; it has its own explicit spatial hold-outs and limitations.

Start with the rigid palm. Its model pose does not depend on finger angles, so
fixed-palm annotation can be done from camera-only recordings. To include a
finger, fitting requires actual settled measured joints saved with that frame.
The code refuses to substitute nominal angles for missing finger measurements.

```bash
# Already prepared for this workspace; a second catalog command will preserve it.
python camera_hand_calibration/calibrate.py catalog --body palm --count 16
```

`configs/landmarks_palm.json` and `results/landmarks_palm.png` describe candidate
vertices on the **actual visual mesh**, in body-local meters. These are not
automatically detected features. Use only vertices that you can identify
unambiguously on the real surface, or author more useful known feature centers
in a separate catalog using the same `id/body/xyz_body_m` schema. Joint centers
inside plastic, fabric boundaries and collision approximations are not reliable
visible feature correspondences.

```bash
python camera_hand_calibration/calibrate.py annotate \
  --frame datasets/<session>/frame_00000.npz \
  --catalog configs/landmarks_palm.json --split train
```

The window shows the actual RGB image on the left and candidate projections
on the right. Select a candidate on the right, or use `[` / `]`; click its
physical counterpart on the left. `U` removes that correspondence, `S` saves,
and Escape/Q cancels. Inspect the catalog's three body-frame views to identify
features. The projected prior is only an annotation aid, not an observation.
The GUI requires a working desktop display; all other operations are headless.

Observations save frame/model hashes, body-local and world positions, image
pixels, measured-state provenance, pose/split labels and conservative local
depth statistics. Annotating a frame/feature again replaces that observation.
Do not call the same physical pose both train and validation. The fitter checks
pose-ID separation and the hashes; it cannot detect dishonest or inaccurate
manual feature identification.

```bash
python camera_hand_calibration/calibrate.py fit \
  --observations datasets/observations.json --initial scratch
# Or separately, retaining both candidates:
python camera_hand_calibration/calibrate.py fit \
  --observations datasets/observations.json --initial current
python camera_hand_calibration/calibrate.py report --result results/<fit>
```

Fitting needs at least six usable training correspondences. Prefer many
well-spread features across visible faces/depths, with multiple training poses
and independent held-out poses. Collinear configurations are rejected;
planar ambiguity and Jacobian conditioning are reported. RANSAC and robust
loss handle some annotation outliers, not systematic CAD errors.

Each fit stores a standalone candidate, copied observations/code, complete
recipe/environment, before/after image overlays, pixel and depth residuals,
plots in PNG/PDF, and an HTML report. A candidate with insufficient held-out
evidence stays `candidate_unvalidated`. Passing image checks requires a
full-rank fit, at least two training pose IDs and six held-out points within
the configured P95 pixel threshold. This is **not absolute metrological
accuracy** and does not establish correct finger kinematics or mounting angle.
Synthetic test outputs are explicitly marked `synthetic_test_only`.
Fit folders also copy the annotated raw frames, so deleting a temporary capture
or test fixture does not silently invalidate the saved result.

Nothing exports to or overwrites the production transform automatically.

Regenerate the workspace landing page and machine-readable inventory with:

```bash
python camera_hand_calibration/calibrate.py status
```

## Hand-data phase — prepared, NOT executed

After the operator confirms readiness, use an **already authorized** bridge
publishing measured `LEAP_STATE` positions/velocities in the configured order.
This workspace does not start the bridge or enable torque. Its only hand-data
interface is a passive LCM subscriber; it never publishes actions, opens a
serial device, or changes motor registers.

```bash
# Prepared command; wait for operator confirmation before this phase.
python camera_hand_calibration/calibrate.py capture \
  --joints-lcm --frames 300 --pose-id train_open_01 --split train
```

The existing state packet is `<q32f`: epoch-microsecond int64, 16 float32 joint
positions and 16 float32 velocities (136 bytes). Malformed states fail capture.
All received states are recorded with receipt times. Each saved camera frame
gets the nearest state, signed timestamp skew, settle-window statistics and an
explicit `usable` flag.

Default eligibility requires SDK epoch timestamps, ≤30 ms skew, ≥0.3 s of a
0.4 s settle window, ≥5 samples, ≤0.01 rad joint variation and ≤0.03 rad/s
reported velocity. These thresholds are configurable. Host-arrival-only
camera times are labelled approximate and rejected for articulated fitting.
Feedback gaps over 50 ms also invalidate the settle window.
Bridge report time still has serial/read uncertainty, so static captures are
preferred. Never reinterpret joint targets as measured poses.

Suggested physical protocol: empty hand, fixed camera and hand mount, natural
rigid features visible, good lighting, no operator occlusion. First verify the
palm; then record 5–8 settled finger poses, holding out separate poses. Any
motor motion/torque procedure is outside these scripts and awaits confirmation.

## Reproduce checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q camera_hand_calibration/tests \
  --basetemp camera_hand_calibration/cache/pytest \
  -o cache_dir=camera_hand_calibration/cache/pytest_cache
```

Run from the repository root. Tests use synthetic correspondences and fake LCM,
never a camera or motor. The end-to-end test leaves a clearly labelled
synthetic fit and overlays under `results/synthetic_test_*`. Do not treat that
as an actual camera-calibration result.

Acquisition follows the [RealSense alignment example](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py).
Pose estimation follows the [OpenCV PnP convention](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).
The workspace's pinned versions, configuration and raw observations—not current
online documentation—are the reproducibility record.

## Stationary RGB-D comparison, no hand access

`stationary_rgbd.py` fits **T_world_from_camera from scratch**, using explicit
natural feature annotations and depth on the fixed palm CAD. It uses no tags,
finger angles, LCM, serial devices, torque or motor motion. Camera factory
intrinsics and depth scale remain fixed. It initializes from both planar PnP
branches and SQPnP, then refines with robust pixel and point-to-plane residuals.
The previous extrinsics are loaded only after the fit for comparison.

```bash
python -B camera_hand_calibration/stationary_rgbd.py \
  --config configs/stationary_rgbd.json
python camera_hand_calibration/calibrate.py status
```

All paths are relative to this workspace. The input configuration explicitly
lists the CAD vertices, manually selected original-image pixel coordinates,
depth patch polygons, training/held-out regions and captured-frame splits.
Two outer palm servo housings provide training corners; the middle housing's
corners are held out. A separate spatial palm patch and a later camera-only
recording provide depth checks without refitting. No paper/tag features are
used, despite the physical tag remaining in the image. Only rigid palm mesh
geometry is used; no nominal joint configuration is substituted for feedback.

The frozen binary model is hash checked and loaded to avoid rebuilding its
meshes. The read-only source production camera configuration, hardware config
and calibrated hand XML are hashed before and after the run. No candidate is
exported into those files.

The latest completed report location is recorded in
`results/stationary_rgbd_latest.json`. Each result includes standalone
`candidate.json`, `comparison.json`, `report.md`, an HTML report, source code
and recipe snapshots, natural-feature/CAD overlays, residual plots in PNG/PDF,
per-frame later-capture checks, bootstrap transforms, and observed cube-surface
clouds in all compared world frames. The raw RGB-D frames remain in this
workspace and are referenced by SHA-256 hashes.

The object comparison separately shows (1) the original geometric extrinsics,
(2) the existing pipeline including its -15 mm object-only camera-Z tweak,
and (3) the new RGB-D candidate without any extra object correction. Observed
surface centroids are explicitly **not** labelled physical cube centres.

This is a single physical view with approximately planar features. A held-out
housing and repeated captures do not establish absolute extrinsic accuracy.
Bootstrap annotations/depth biases quantify sensitivity, and a separate
fixed-extrinsics, depth-bias-only fit checks whether sensor/model mismatch can
explain the discrepancy. Fitting is not proof that the camera moved or the old
calibration was wrong. The current parameters remain unchanged.
