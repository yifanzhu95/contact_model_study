# Method 5: manual cube reference

All code, configuration snapshots, measured joints, key events, images and fit
results stay in `camera_hand_calibration/`. Production calibration is unchanged.

Start with an empty hand:

```bash
python camera_hand_calibration/method5.py --empty-hand-confirmed
```

The hardware process screens the approach and gradually moves to the experiment's
`robot_qref`, then continuously holds that target with the experiment's gains.
The normal hardware tracker starts with a fresh mask and the normal bounding box
`[440, 272, 103, 113]`. Capture and accept the mask in its receiver. Place the cube
on the held hand, moving slowly enough for tracking to follow.

The dedicated interactive MuJoCo window shows the measured hand and translucent
cyan tracked cube. The gold reference uses the same mesh and texture. Its pose
is a manual estimate of the real cube pose relative to the modeled hand.

| Key | Action |
| --- | --- |
| Numpad 3 | Initialize/reset gold reference from the fresh tracked cube |
| Numpad 5 | Switch translation / rotation |
| Numpad 4 / 6 | Negative / positive world X |
| Numpad 2 / 8 | Negative / positive world Y |
| Numpad 1 / 7 | Negative / positive world Z |
| Numpad 0 | Cycle step size: 0.25, 1, 5 mm or degrees |
| Numpad 9 | Undo last reference edit |
| Numpad Enter | Fit calibration to recent stable tracking; apply locally |
| F11 | Compare original method-3 and fitted calibration |
| F12 | Request shutdown; continue holding for cube removal |
| Numpad / | After shutdown request, confirm cube removed and disable torque |
| Ctrl+F12 | Request finish at zero; after removing cube, Numpad / closes editor, gradually returns to zero, then disables torque |

Use the mouse to orbit, pan and zoom. These numpad bindings belong to this custom
window, rather than the standard MuJoCo viewer key handler. Enable Num Lock if
your keyboard maps the numpad to navigation keys.

The fit estimates six camera extrinsic parameters from the last three seconds
of stable raw camera-frame object poses and the edited reference. It excludes
the old object-only -15 mm correction. The hand hold and tracking continue in
separate processes throughout fitting. Every fit is preserved as `fit_NNN/` with
its reference, raw observations, image, transform and residuals. A fit cannot run
without fresh tracking, a fresh holding heartbeat, and torque enabled.

Numpad Enter also writes `manual_reference.json` before attempting a fit, so a
rejected calibration does not prevent saving the manually aligned pose. Fit
inputs are retained under `fit_attempt_*/`. Reopening the editor with
`--resume-held-pose` restores the latest saved reference from `editor_status.json`.

Limits are checked separately: at most 50 mm displacement of the observed cube,
15 degrees rotation, and 200 mm displacement of the camera origin. A rotation
about a cube 0.6 m away can shift the implied camera origin substantially while
barely moving that cube; the origin shift alone is not its position correction.

A single manually aligned pose supplies six constraints but is not an independent
accuracy validation. Fit residuals measure agreement with that reference, not
verified global accuracy. Check additional placements before adopting a candidate.

Closing the editor requests shutdown but leaves torque on until cube removal is
confirmed. If the editor is closed, use:

```bash
python camera_hand_calibration/method5.py --request-stop
# Physically remove the cube and clear the hand before this acknowledgement:
python camera_hand_calibration/method5.py --object-removed
```

Tracking stays running after the hold ends. Hardware faults still trigger cleanup.

To finish at zero from the terminal instead:

```bash
python camera_hand_calibration/method5.py --finish
# Remove the cube completely and clear your hands, then:
python camera_hand_calibration/method5.py --object-removed
```

The zero return preserves gains, screens paths from both measured and held joint
positions, uses the smooth preparation speed/acceleration, waits for settling,
then verifies torque off. This requires a hold process started with the updated
code; `--finish` refuses older active holds rather than claiming they can return
to zero. An older hold can still be stopped with `--request-stop` followed by
`--object-removed`.

## Select calibration for MPPI

Choose one option before the launcher's `--` separator:

```bash
python launch_leap_experiment.py --camera-calibration original --n-episodes 12 -- ...
python launch_leap_experiment.py --camera-calibration latest --n-episodes 12 -- ...
```

Keep the usual experiment configuration and MPPI arguments in place of `...`.
`original` uses the frozen original calibration plus the camera-axis -15 mm
object correction. `latest` resolves `method5/latest_fit.json` and uses the saved
manual-fit transform with no extra object correction. The runner itself also
accepts `--camera-calibration original|latest` when launched directly; its
`--camera-config` must describe the tracker publication it receives.

The tracking/mask camera configuration is unchanged. The controller and MuJoCo
viewer share an immutable consumer-adapter configuration that removes the
incoming publication transform/offset exactly once and applies the selected
calibration. Thus switching this option does not require a tracker restart or a
new mask. The render camera uses the selected physical camera extrinsic directly.
Source JSON files are never overwritten. Selected transforms, source paths and
hashes are captured in run logs and `runtime/selections/`.

## Standalone recovery after an abrupt script exit

Remove the object, clear your hands, and run from the repository root:

```bash
conda activate mj_env
python shutdown_leap_hand.py --empty-hand-confirmed
```

This utility needs no running calibration process, bridge, tracking, Docker or
GPU. It accepts torque left enabled after a crash, preserves the existing motor
gains/current settings, preloads the measured pose, screens a collision-free
return, moves with a smooth trajectory (maximum 0.3 rad/s and 0.35 rad/s²), waits
for zero to settle, and verifies torque off on all 16 motors. Processes holding
the configured USB port are terminated first (SIGTERM, then SIGKILL after three
seconds if needed). It rescans for owners before opening USB and checks process
identities before signalling. It never deliberately opens a concurrent serial
connection. Permission failures or an owner that cannot be stopped are reported.
`--hardware-config PATH` and `--port PATH` override device configuration when needed.

To skip movement and request torque off only:

```bash
python shutdown_leap_hand.py --empty-hand-confirmed --torque-off-only
```

Ctrl-C interrupts the move and attempts verified torque-off cleanup. Feedback,
health, clearance and following-error failures also attempt torque off. A failed
or disconnected USB bus can prevent verification; the terminal and session log
report that explicitly. Return failure produces a nonzero exit code even when
torque-off cleanup succeeds. Logs stay under `results/hand_recovery_*/` inside
the calibration workspace. No firmware settings or calibration files are changed.
