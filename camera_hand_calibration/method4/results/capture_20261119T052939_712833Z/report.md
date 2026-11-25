# Method 4: known-cube translation anchor

Captured 60 unique stable frames over 3.29 s. Cube removal was confirmed before all 16 motors were verified torque-off. Refinement ran after torque cleanup.

The cube rests on the measured finger-link plane, not an assumed horizontal/table plane. Lateral centering follows the middle finger; both lower rear corners touch the modeled fixed palm housing faces. No paper-tag observation is used.

| Calibration | Position RMS at reference (mm) | Resting-face normal RMS (deg) | Cube-axis alignment RMS (deg) | Rigid palm image RMS (px) |
|---|---:|---:|---:|---:|
| Method 1, raw | 25.439 | 1.961 | 5.654 | 4.630 |
| Method 1, -15 mm | 14.576 | 1.961 | 5.654 | 4.630 |
| Method 2, raw | 6.780 | 1.510 | 2.489 | 1.383 |
| Method 3, raw | 8.887 | 1.614 | 2.918 | 1.552 |
| Method 4, raw | 0.461 | 1.614 | 2.918 | 8.678 |

Method 4 preserves method 3 rotation exactly. Translation change in world XYZ (mm): +7.760, -1.163, -4.146.
The same change in camera XYZ (mm): -1.031, +8.769, -0.894.

All raw camera poses exclude the old -15 mm object-only correction. Only the explicitly labeled method-1 comparison row includes it. Method 4 needs no additional object-only offset.

The low method-4 residual is at the placement used to fit its translation. It is not an independent accuracy estimate. Manual centering/contact, nominal cube size, CAD surface differences and tracker bias can be absorbed into translation. Additional placements are needed to establish workspace accuracy. Rotation is verified modulo cube symmetry, not printed face identity.

The independent rigid-palm check changes from 1.55 px for method 3 to 8.68 px for method 4. A masked natural-feature image check found 0.32 px median palm displacement after the experiment. Method 4 may be absorbing cube-tracker bias, physical placement error or contact-model geometry error rather than correcting the true camera extrinsic. It remains an experimental object-reference candidate, not a globally validated camera calibration.

The acquisition-time horizontal-plane calculation is retained separately for provenance; `candidate.json` is the final measured-link-plane result. Production calibration files remain unchanged.

Reproduce after torque is off:

```bash
python camera_hand_calibration/method4_refine.py --run-dir method4/results/capture_20261119T052939_712833Z
python camera_hand_calibration/method4_report.py --run-dir method4/results/capture_20261119T052939_712833Z
```
