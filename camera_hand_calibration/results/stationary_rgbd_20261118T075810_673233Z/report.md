# Stationary RGB-D calibration comparison

**Status: candidate only; no production parameters changed.**

The scratch fit uses manually identified natural corners on the three rigid palm servo housings, sparse aligned depth, and the fixed palm CAD. No tag detector, finger geometry, joint reads, serial access, or motor commands. The current transform is loaded only after the independent fit.

| Metric | Current | Scratch RGB-D |
|---|---:|---:|
| Training corner RMS (px) | 6.098 | 2.382 |
| Held-out middle-housing RMS (px) | 6.675 | 1.597 |
| if_front, train, depth RMS (mm) | 19.610 | 1.429 |
| rf_front, train, depth RMS (mm) | 23.692 | 1.156 |
| mf_front, validation, depth RMS (mm) | 22.640 | 1.780 |
| palm_flat_left, train, depth RMS (mm) | 21.016 | 1.127 |
| palm_flat_right, validation, depth RMS (mm) | 22.213 | 0.535 |

Camera-centre difference: 50.16 mm. Relative rotation: 4.526 degrees.
Camera-centre change and rotation are coupled; this is a difference between estimates, not a measured correction to the real camera.

Same visible cube-surface centroid, scratch minus current **effective object pipeline**, world XYZ mm: [1.071, -12.999, -1.121].
The current effective pipeline already shifts the tracked object -15 mm along camera Z. That correction is excluded from hand calibration and explicitly included in the object comparison. The new candidate has no additional object offset.

## Limits

* Manual corner correspondence uncertainty and CAD-to-real mismatch
* Single fixed hand/camera pose, near-planar landmarks
* Held-out spatial features and adjacent frames, not independent poses
* Factory intrinsics and depth scale held fixed
* Spatial hold-outs and repeated static frames support consistency checks, not independent multi-view validation.
* RealSense depth bias, uncertain feature identities, and CAD-to-physical housing differences can masquerade as camera extrinsic error. No absolute ground-truth accuracy claim.
* Perturbing annotations by 1.5 px and correlated patch depths by 2.0 mm gives a P95 rotation variation of 7.62 degrees.
* Do not promote this candidate automatically. The current calibration and object correction remain untouched.

## Reproduce

```bash
/home/chetan/miniconda3/envs/mj_env/bin/python -B camera_hand_calibration/stationary_rgbd.py --config results/stationary_rgbd_20261118T075810_673233Z/config.json
```

All numerical data are in comparison.json, candidate.json and fit_data.npz; image/depth source hashes are retained. The stationary image annotations are explicit in config.json.
