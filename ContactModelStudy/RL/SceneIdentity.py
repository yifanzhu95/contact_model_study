"""Identity of the compiled source scene, including included XML and loaded assets."""
import hashlib

import mujoco
import numpy as np


def scene_model_fingerprint(path):
    """Hash the source model before any backend-specific contact changes.

    MuJoCo's binary serialization includes the compiled geometry, meshes and
    textures without the absolute checkout path. Compile afresh so edits to an
    included file cannot be hidden by a path-based cache. This runs once per
    environment construction, outside the physics/collection loop. Clear only
    the compiler's asset cache first: file edits within its timestamp resolution
    can otherwise return stale mesh data. Live model/solver arrays are unaffected.
    """
    mujoco.mj_clearCache(mujoco.mj_getCache())
    model = mujoco.MjModel.from_xml_path(str(path))
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return {'schema': 'mujoco_mjb_v1', 'mujoco_version': mujoco.__version__,
            'sha256': hashlib.sha256(buffer).hexdigest()}
