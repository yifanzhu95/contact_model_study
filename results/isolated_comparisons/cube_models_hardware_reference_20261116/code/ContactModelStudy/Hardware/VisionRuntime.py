"""Small runtime adapters for the checkout and Docker FoundationPose copies."""
from contextlib import contextmanager
import faulthandler
from functools import wraps
import time

import numpy as np


@contextmanager
def startup_stage(name):
    """Make slow native initialization visible without verbose per-frame output."""
    start = time.monotonic()
    print(f"[tracking startup] {name}...", flush=True)
    faulthandler.dump_traceback_later(30, repeat=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
    print(f"[tracking startup] {name} ready in {time.monotonic() - start:.3f}s", flush=True)


def sampled_mesh_diameter(model_pts, n_sample=1000, block_size=512):
    """Same sampled maximum distance, with at most block_size² float64 workspace.

    Upstream broadcasts N×N×3 differences before taking their norm. At its
    10,000-point initialization setting those temporaries occupy several GB.
    Keep its random sampling policy and evaluate the upper triangle in blocks.
    """
    from scipy.spatial.distance import cdist

    pts = np.asarray(model_pts)
    if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts) or not np.isfinite(pts).all():
        raise ValueError("Mesh vertices must be a nonempty finite N-by-3 array")
    if n_sample is not None:
        ids = np.random.choice(len(pts), size=min(n_sample, len(pts)), replace=False)
        pts = pts[ids]
    maximum_squared = 0.0
    for i in range(0, len(pts), block_size):
        for j in range(i, len(pts), block_size):
            squared = cdist(pts[i:i + block_size], pts[j:j + block_size], metric='sqeuclidean')
            maximum_squared = max(maximum_squared, float(squared.max()))
    return float(np.sqrt(maximum_squared))


def install_bounded_mesh_diameter(estimator_module):
    """Patch the estimator's imported helper, including an image-owned copy.

    Do not edit FoundationPose inside a Docker image: the production estimator
    can live outside this checkout. Its unused mesh= branch remains upstream.
    """
    original = estimator_module.compute_mesh_diameter
    if getattr(original, '_leap_bounded_workspace', False):
        return

    @wraps(original)
    def diameter(model_pts=None, mesh=None, n_sample=1000):
        if mesh is not None:
            return original(model_pts=model_pts, mesh=mesh, n_sample=n_sample)
        return sampled_mesh_diameter(model_pts, n_sample)

    diameter._leap_bounded_workspace = True
    estimator_module.compute_mesh_diameter = diameter
