# LEAP cube hardware deployment

This is a complete reference configuration for the **M2 Trial 358 baseline**.
Runtime paths are relative to the checkout. It needs no `/home/chetan` path,
old experiment data, calibration-history pointer or current task weight defaults.
The existing workstation files under `configs/` are unchanged.

## What is provided

| File | Contents |
| --- | --- |
| `experiment.json` | Launcher configuration and explicit controller parameters, including all twelve cost weights |
| `hardware.reference.json` | USB/baud, native motor gains/current, mounting quaternion, preparation pose, clipping, captured limits, recovery and latency settings |
| `camera.reference.json` | Source camera intrinsics and Method 5 camera-to-world transform, zero object correction |
| `joint_reference.json` | P011 cube joint-cost reference, distinct from preparation pose |
| `host_versions.json`, `constraints-host.txt` | Observed host versions and core installation constraints |
| `external_assets.json` | Pinned LEAP API commit/file hashes and exact source Docker image IDs |
| `bundle_manifest.json`, `verify.py` | Reference-asset integrity checks; no device access |

The hand/limit XMLs, mesh assets and active cube-tracking texture are already in
the checkout. The manifest includes their runtime dependencies and task source.

Baseline: M2, high/high geometry, **512 samples, one iteration, 0.064 s control
period, 0.256 s/four-step horizon, paced control, temperature 0.1, noise 0.05,
shifted warm start**. Action-rate weight 50; joint-limit weight
817.6259701873483; blending alpha 1. Measured-position increments use ±0.2 rad
clipping followed by captured absolute bounds. Cube: 70 mm, 70 g. Sliding
friction cube/tips/palm/other fingers: 0.3/0.5/0.4/0.4. Mounting uses the source
30° convention encoded by the full palm quaternion.

Goal: `[0.025, 0.035, 0.075]` m and `[1, 0, 0, 0]` wxyz. Current task success:
position <20 mm, quaternion metric <0.04 (~23.07°), existing velocity criterion,
no consecutive-step hold. Preparation and zero return are separate from MPPI.

## Software and external assets

Use Linux, Python 3.12, an NVIDIA GPU/compatible driver, Docker with NVIDIA
Container Toolkit, a graphical desktop, RealSense D435i and the 16-motor LEAP
hand. Install ffmpeg through the system or Conda package manager for video.
Host Torch is unnecessary when tracking runs in Docker. From the checkout:

```bash
python -m pip install -c deployment/leap_cube/constraints-host.txt -e '.[hardware]'
git clone https://github.com/asu-iris/LEAP_Hand_Hardware.git third_party/LEAP_Hand_Hardware
git -C third_party/LEAP_Hand_Hardware checkout 4f0ae3a7f951fe8c89e18caae0d8d772718545cd
```

The project metadata separately pins M3's ComFree-v1 source. M6's private package
and Kamino are unnecessary for this M1/M2 profile; see the root README if adding
them. Version records describe the source host, not a tested fresh installation
on the destination.

The bridge expects motor IDs 0–15, model number 1240 and mode 5, at 3,000,000
baud. It initializes/verifies configured P/I/D and goal current with torque off.
USB latency is 1 ms during operation and restored to 16 ms. `sudo -v` in the
launch terminal authorizes the existing sysfs helper when needed; run the
experiment as the ordinary user, with the machine's serial and Docker access.

**The prepared vision runtime and network weights are not in Git.** The small
hardware Dockerfile depends on a locally prepared base image, so it cannot
recreate the entire runtime alone on an empty host. Transfer the final working
image separately. On the source:

```bash
./leap_docker.py save irislab_fppp:leap_hardware | gzip > /external/path/leap-hardware-image.tar.gz
```

On the destination:

```bash
gzip -dc /external/path/leap-hardware-image.tar.gz | docker load
docker image inspect irislab_fppp:leap_hardware --format '{{.Id}}'
```

Expected source image ID:
`sha256:a683622dfce5b9e4201ee07a0e2510df21e36d91b6311c447d5866ad4e2fa431`.
It contains the prepared FoundationPose/SAM stack. Container-internal paths:
`/FoundationPose-plus-plus/FoundationPose` and
`/FoundationPose-plus-plus/sam-hq/pretrained_checkpoints/sam_hq_vit_h.pth`.
The host checkout is mounted at its actual location. Alternatively prepare a
compatible base using `iris-foundationpose-plusplus/Install.md`, then build the
hardware Dockerfile; a rebuilt image needs independent validation. SAM exits
after mask acceptance. Image archives/checkpoints remain outside Git.

## Configure the destination rig

Create editable local copies, ignored by Git:

```bash
cp deployment/leap_cube/hardware.reference.json deployment/leap_cube/hardware.local.json
cp deployment/leap_cube/camera.reference.json deployment/leap_cube/camera.local.json
cp deployment/leap_cube/experiment.json deployment/leap_cube/experiment.local.json
```

In `experiment.local.json`, point `hardware_config` and `camera_config` to the
corresponding `deployment/leap_cube/*.local.json` files. Configure these fields:

- **USB:** `hardware.local.json` → `port`; use the new adapter's stable
  `/dev/serial/by-id/...` path instead of the reference `/dev/ttyUSB0`.
- **Camera:** actual serial, stream profile, measured/factory intrinsics. The
  reference is camera `310522070297`, 640×480 at 60 Hz. The existing pipeline
  checks serial/profile/factory intrinsics. Identical cameras can differ here.
- **World transform:** measure `camera_to_published` if camera/base placement
  differs. Identical hardware does not imply identical extrinsics. Use the
  included calibration tools. This profile publishes directly in MuJoCo world:
  keep `T_model_from_published` identity, all correction vectors zero, and
  launcher `camera_calibration` null. Do not add −15 mm or a second Method 5
  adapter. The saved direct transform is equivalent to the source baseline's
  publication-plus-consumer-adapter chain.
- **Hand:** verify mounting/zero conventions, calibrated response, gains,
  joint bounds and P011 on the new unit. These include per-unit measurements.
  If recaptured, change `joint_limits.model` and `--joint-reference-file` to
  the new relative file paths. `robot_qref` prepares the hand; P011 supplies
  `w_joint`. They are separate poses.
- **GPU:** automatic placement expects two GPUs. For one GPU use
  `--tracking-gpu 0 --mppi-gpu 0`; for multiple cards use `auto` or physical
  indices. Each process uses local `cuda:0` after selection.
- **Mask box:** adapt source `[440, 272, 103, 113]` to the actual cube placement
  in the image. Tracking uses the adjusted 70 mm cube mesh at scale 1.0.

To refresh only the destination camera's intrinsics/profile (camera access,
no motor access), use:

```bash
python calibrate_leap_camera.py --camera-config deployment/leap_cube/camera.local.json \
  --serial DESTINATION_CAMERA_SERIAL --intrinsics-only
```

This does not establish the new camera-to-world transform. Source calibration
datasets and the compiled workspace MJB cache are excluded from Git; historical
camera-workspace commands depending on those captures/cache need reinitializing
on the new rig. The MPPI deployment profile and explicit-XML recovery below do
not depend on that cache.

## Verify and run

These commands open no camera/USB connection and start no experiment services:

```bash
python deployment/leap_cube/verify.py
python deployment/leap_cube/verify.py --profile deployment/leap_cube/experiment.local.json --check-environment
python launch_leap_experiment.py --config deployment/leap_cube/experiment.local.json \
  --tracking-gpu 0 --mppi-gpu 0 --dry-run
```

`--check-environment` checks package metadata, external API files and executable
availability. It does not query Docker images, GPUs or devices. Launcher dry run
does query GPU selection but starts no services. Verify the live hand/object
overlay on a newly calibrated rig before commanding manipulation.

Launch the complete baseline with logging/video:

```bash
sudo -v
python launch_leap_experiment.py --config deployment/leap_cube/experiment.local.json \
  --tracking-gpu 0 --mppi-gpu 0 --n-episodes 12 --restart-tracking \
  --log-dir results/deployment/cube_M2_high_trial358
```

First launch captures a mask; `--restart-tracking` forces a new one. Omit it to
reuse an existing tracker. MPPI waits for Enter after preparation. Ctrl-C uses
the existing zero-return/torque-off workflow. Software-error hold requires object
removal and operator confirmation. Logging uses append-by-configuration defaults.

The accepted geometry comparison used these horizons with the same shared
weights/settings (one step = 0.064 s):

| Model | High | Medium | Low |
| --- | --- | --- | --- |
| M1 | 3 / 0.192 s | 3 / 0.192 s | 3 / 0.192 s |
| M2 | 4 / 0.256 s | 5 / 0.320 s | 5 / 0.320 s |

Append overrides after `--`, for example:
`-- --rollout-model M1 --hand-acc med --time-horizon 0.192`.
Actual control frequency/performance depends on destination compute and contact
states. Logs record effective settings; success rates are not guaranteed to transfer.

To recover an empty hand after an abrupt script exit using these settings:

```bash
python shutdown_leap_hand.py --empty-hand-confirmed \
  --hardware-config deployment/leap_cube/hardware.local.json \
  --hardware-api third_party/LEAP_Hand_Hardware \
  --model scenes/leap/hardware/calibrated_hand.xml
```
