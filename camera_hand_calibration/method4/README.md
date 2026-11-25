# Method 4: known cube placement relative to the hand

This experiment starts with method 3's camera-to-world transform. It estimates
one translation correction from a physically constrained 70 mm cube placement.
It preserves method 3's rotation and separately verifies the resting-face normal
and rear-edge orientation, modulo the cube's 24 geometric symmetries.

The reference is defined by physical contact, not by positioning the cube to
match a projection generated from the calibration being evaluated:

- The three index/middle/ring fingers are straight, mutually parallel and level.
- Cube Y is centered over the middle finger's proximal housing.
- Both lower rear cube corners contact the exposed fixed gray index/ring
  motor-housing faces at the palm/finger junction.
- The cube's bottom face rests on the upper proximal finger housing surfaces.
  Its normal follows the **measured finger-link plane**, which may be tilted
  relative to the table. A horizontal cube face is not assumed in the final fit.

The frozen model supplies the finite palm contact faces and upper finger
surfaces. Commanded positions are not substituted for measurements. Reference
positions are recomputed from actual settled joint measurements under load.
The mesh's geometric-center offset is included when comparing tracked mesh
origins with the known physical cube center. Fabric compression, imperfect
placement, CAD discrepancies and tracker bias remain reference uncertainties.

`config.json` pins the method-3 parent, cube size, geometry and quality limits.
`latest_plan.json` points to the offline plan. `active.json` points to an active
supervised capture, with owned process identities. Each capture directory has
the copied source/config, complete joint/command/current log, raw camera poses,
matched joint timestamps, JPEG frames, operator acknowledgement files and
torque-cleanup status. Failed attempts are retained separately.

The capture subscribes to the existing tracker's raw camera-frame ZMQ stream;
it neither opens the camera a second time nor uses the published object's
-15 mm camera-axis correction. The normal read-only USB publisher must be
stopped before the capture driver takes exclusive ownership of the serial
device. The capture publishes measured `LEAP_STATE` for the viewer, and sends
no MPPI/LCM manipulation commands. Runtime PID gains remain unchanged.

The physical stages are explicit:

1. Operator confirms hand empty and clear.
2. Screen the fresh measured start and approach. If resting fingers are already
   inside the conservative 4 mm margin, an initial outward separation is allowed
   only above 2.5 mm, with sampled clearance nondecreasing. Later segments must
   satisfy 4 mm; live measured clearance below 2 mm aborts.
3. Move slowly to the level pose. A small bounded MCP target trim levels the
   measured supports while empty, without changing hardware gains.
4. Hold the final target fixed and request object placement. There is no
   integration of manual contact while the operator handles the cube.
5. After `object_placed.trigger`, settle and collect 60 unique stable tracked
   frames. Check timestamp match, measured support flatness and tracking jitter.
6. Keep holding and request cube removal. Never infer removal from a timeout.
7. After `object_removed.trigger`, disable torque and verify all motors off.
8. Only then compute and save the isolated method-4 translation candidate.

`method4_refine.py --run-dir ...` performs the final contact calculation only
after verified torque off. It follows the measured middle-link normal, aligns
the rear edge with the intersection of that support plane and the palm stop
plane, then solves for the cube center from contact and lateral centering.
An initial acquisition-time calculation with a horizontal-face assumption is
retained separately as `candidate_horizontal_assumption.json`; the final
`candidate.json` uses the measured-link plane. Raw measurements are unchanged.
This intermediate artifact applies to the first acquisition, during which the
operator clarified the tilted-plane constraint. The current capture code uses
the measured-link reference directly for subsequent acquisitions.

The first completed method-4 measurement fitted 60 frames over 3.29 seconds.
Reference position RMS changed from 8.887 mm (method 3) to 0.461 mm (method 4),
with method-3 rotation preserved. Resting-face normal error was 1.614 degrees
RMS relative to the measured links. However, the independent rigid-palm image
check worsened from 1.55 px to 8.68 px; the camera/palm image moved only about
0.32 px. This is evidence that the object anchor may absorb tracking/contact
or placement error. It is not established as a better global camera extrinsic.
The viewer exposes it on F10 for comparison with object correction OFF.

The fit uses robust per-frame translations, rejects outliers and reports both
world- and camera-axis changes. Rotation verification does not identify printed
faces, and it does not silently modify the prior rotation. A low residual at
this single placement is a fitting residual; independent placements are still
needed to establish accuracy throughout the workspace. Production calibration
files are never replaced by this script.

Offline commands (no hardware access):

```bash
python camera_hand_calibration/method4_reference.py --plan
MUJOCO_GL=egl python camera_hand_calibration/method4_reference.py --plan --preview
python -m pytest -q camera_hand_calibration/tests/test_method4_reference.py
```

The supervised hardware entry point is `method4_capture.py --run --run-dir ...`.
It requires a **new** workspace-local directory with the operator's empty-hand
acknowledgement. Placement/removal acknowledgements are written only after the
operator confirms the corresponding physical action. A first interrupt while
an object may be present requests removal while holding; a forced second
interrupt or device fault performs immediate torque cleanup.
