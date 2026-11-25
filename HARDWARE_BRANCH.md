# Hardware experiment branch

`hardware-experiments` was created from local `main` at
`dfa0f42` (`added a contact param varying script.`), whose upstream is
`origin/main` at `https://github.com/yifanzhu95/contact_model_study.git`.

This branch captures the LEAP hardware runner and LCM bridge, persistent object
tracking and video-recording integration, calibration tools, joint-reference
and joint-limit capture, recovery/shutdown utilities, hardware models and
configurations, RL evaluation tools, and their tests. See `README.md` for setup
and experiment commands.

Source and configuration files are versioned, including small saved calibration
fits used by the current calibration selector and the mesh/texture assets used
by the hand and active cube tracker. Experiment/analysis source scripts stored
inside ignored result folders are explicitly included, along with small BO
setup/configuration snapshots. Existing local experiment outputs have not been
deleted or moved.

The following remain local and are excluded from commits:

- Experiment data and logs under `results/` and other existing output roots;
  the explicitly tracked source/setup files are exceptions.
- Videos, RGB-D arrays, calibration capture datasets, telemetry streams, and
  process/runtime state.
- Downloaded policy/network checkpoints, archives, compiled MuJoCo models,
  virtual environments, and optional large vision demonstration media/textures.

Analysis scripts require the corresponding local recordings, and resuming an
existing BO requires its local checkpoint/results. Those data are not included
in the source commit. New source files placed inside ignored output folders
must also be explicitly staged if they should be versioned.

The vision model checkpoints and RL policy weights must be supplied separately
when setting up another machine. The source checkout does not include these
downloaded binaries. The CPU RL runtime can be prepared with the setup script
inside `rl/NoDR_s1009_u0800_partner_package_English_v2/`.

Hardware experiment tests can be run without commanding the hand:

```bash
python -m pytest tests/test_leap_*.py tests/test_action_rate.py \
  tests/test_object_friction_bo_preparation.py -m 'not gpu and not slow'
```

The initial branch commit is prepared locally. Publishing the branch requires
a separate push after review.

## Validation of this snapshot

Validated on 24 November 2026 without physical hand commands:

| Suite | Result |
| --- | --- |
| Hardware tests, action-rate tests, BO preparation; CPU only | 709 passed, 1 skipped, 8 GPU/slow cases deselected |
| Camera/hand calibration tests | 59 passed |
| RL hardware-evaluation tests in the package's CPU runtime | 17 passed |

The hardware suite needs local IPC sockets and a writable Warp cache. In a
restricted agent sandbox those operations may be blocked; the passing run used
CPU-only settings outside that sandbox, with a temporary Warp cache. Calibration
and RL tests enforce their workspace boundaries, so place pytest temporary files
inside the corresponding workspace:

```bash
python -m pytest camera_hand_calibration/tests \
  --basetemp=camera_hand_calibration/cache/branch_pytest -q

rl/NoDR_s1009_u0800_partner_package_English_v2/runtime/venv/bin/python \
  -m pytest rl/NoDR_s1009_u0800_partner_package_English_v2/tests \
  --basetemp=rl/NoDR_s1009_u0800_partner_package_English_v2/runtime/branch_pytest -q
```

The GPU/slow cases and live hardware behavior were not retested for this commit.
