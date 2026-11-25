# Independent articulated camera calibration

No new calibration accepted: independent fixed-palm and moving-finger checks conflict.

The paper tag is left in place and masked from RGB and depth. It is not detected, measured, or used as a feature. Current calibration is read only after all new fits finish. Production calibration remains unchanged.

Fixed-housing image checks use the middle palm housing, excluded from training. Moving-finger checks use the middle finger, excluded from fitting; its neutral RGB observation anchors an unknown tangential feature position once. Only its other three poses are scored. These are internal geometric consistency tests, not absolute accuracy estimates.

| Candidate | Fixed housing image (px) | Moving middle finger image (px) | Fixed housing depth (mm) | Palm depth (mm) | Moving depth (mm) |
|---|---:|---:|---:|---:|---:|
| surface | 8.17 | 1.14 | 1.69 | 17.28 | 2.32 |
| surface_bias | 8.74 | 1.15 | 19.67 | 3.18 | 2.26 |
| palm | 1.38 | 7.15 | 0.44 | 0.59 | 18.13 |
| current | 4.63 | 7.73 | 20.17 | 22.11 | 2.27 |

The surface-only candidate explains the selected moving features, but its fixed-palm checks must also pass. The depth-bias fit is a nuisance diagnostic and does not alter factory depth scale or camera settings. Each model/annotation assumption can bias an apparently good local fit.

An audit of the earlier stationary analysis found that three intended outer lower-right housing corners were assigned inner-notch CAD coordinates, 4.752 mm away. This experiment uses newly annotated pixels and corrected outer vertices. Earlier artifacts remain intact; their reported residuals are not proof of absolute calibration accuracy.

Likely contributors requiring separation: natural-feature/CAD correspondence, CAD-to-real rigid housing placement, encoder zero / kinematic geometry, aligned-depth bias and missing depth on small surfaces. The present dataset does not establish which is responsible. No joint offsets or hardware gains were fitted/applied to compensate.

Reproduce the complete offline comparison:

```bash
python camera_hand_calibration/rigid_surface_fit.py --config configs/rigid_surface_fit.json
```

Raw data and motion safety/torque status: `datasets/articulated_20261119T033633_480690Z`. All output and provenance stay in this workspace. No additional hardware motion is part of this command.
