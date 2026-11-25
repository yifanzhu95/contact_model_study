# Independent articulated camera calibration

No candidate passes every cross-check; the joint fit is a provisional review candidate, not a production calibration.

The paper tag is left in place and masked from RGB and depth. It is not detected, measured, or used as a feature. Current calibration is read only after all new fits finish. Production calibration remains unchanged.

Fixed-housing image checks use the middle palm housing, excluded from training. Moving-finger checks use the middle finger, excluded from fitting; its neutral RGB observation anchors an unknown tangential feature position once. Only its other three poses are scored. These are internal geometric consistency tests, not absolute accuracy estimates.

| Candidate | Fixed housing image (px) | Moving middle finger image (px) | Fixed housing depth (mm) | Palm depth (mm) | Moving depth (mm) |
|---|---:|---:|---:|---:|---:|
| surface | 3.33 | 1.13 | 2.59 | 0.95 | 2.20 |
| surface_bias | 3.65 | 1.13 | 20.49 | 22.92 | 2.15 |
| joint | 1.55 | 1.13 | 1.09 | 1.27 | 3.05 |
| palm | 1.38 | 1.31 | 0.44 | 0.59 | 5.53 |
| current | 4.63 | 1.15 | 20.17 | 22.11 | 23.72 |

The surface-only candidate explains the selected moving features, but its fixed-palm checks must also pass. The depth-bias fit is a nuisance diagnostic and does not alter factory depth scale or camera settings. Each model/annotation assumption can bias an apparently good local fit.

An audit of the earlier stationary analysis found that three intended outer lower-right housing corners were assigned inner-notch CAD coordinates, 4.752 mm away. This experiment uses newly annotated pixels and corrected outer vertices. The exploratory moving-housing fit also initially used the opposite face; the final code verifies the visible extremal face (-23.1 mm in each proximal body) against the frozen mesh and camera viewing direction. Earlier artifacts remain intact as rejected diagnostics; their reported residuals are not proof of absolute calibration accuracy.

Likely contributors requiring separation: natural-feature/CAD correspondence, CAD-to-real rigid housing placement, encoder zero / kinematic geometry, aligned-depth bias and missing depth on small surfaces. The present dataset does not establish which is responsible. No joint offsets or hardware gains were fitted/applied to compensate.

Reproduce the complete offline comparison:

```bash
python camera_hand_calibration/rigid_surface_fit.py --config configs/rigid_surface_fit.json
```

The existing -15 mm camera-axis object-only tracking correction is excluded from all hand fits. A hand calibration cannot determine whether that tracker correction should be retained: object pose validation would be needed before deploying a replacement transform.

Raw data and motion safety/torque status: `datasets/articulated_20261119T033633_480690Z`. All output and provenance stay in this workspace. No additional hardware motion is part of this command.
