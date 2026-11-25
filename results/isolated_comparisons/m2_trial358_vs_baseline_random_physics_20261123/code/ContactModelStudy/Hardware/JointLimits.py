"""Load the completed manual joint-limit capture shared by planning and bridge."""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ContactModelStudy.Hardware.Scene import JOINTS, ROOT


@dataclass(frozen=True)
class JointLimits:
    model: Path
    model_sha256: str
    lower: np.ndarray
    upper: np.ndarray
    weight: float

    def metadata(self):
        return dict(model=str(self.model), model_sha256=self.model_sha256,
                    joint_order=list(JOINTS), lower=self.lower.tolist(), upper=self.upper.tolist(),
                    cost_weight=self.weight, application='MPPI delta targets only, after ctrl_range')


def load_joint_limits(config):
    """None preserves legacy behavior. Reject unfinished or externally edited captures."""
    settings = config.get('joint_limits')
    if settings is None:
        return None
    if (not isinstance(settings, dict) or 'model' not in settings
            or set(settings) - {'model', 'cost_weight'}):
        raise ValueError('joint_limits must contain model and cost_weight, or be null')
    path = Path(settings['model']).expanduser()
    path = (path if path.is_absolute() else ROOT/path).resolve()
    weight = float(settings.get('cost_weight', 100.))
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('joint_limits.cost_weight must be finite and nonnegative')
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    record = json.loads(path.with_name('measured_joint_limits_capture.json').read_text())
    if (record.get('xml_status') != 'all_16_joint_ranges_saved'
            or record.get('output_sha256') != digest or record.get('joint_order') != list(JOINTS)):
        raise ValueError('Joint limits require a completed, saved capture matching the model XML')
    joints = {j.get('name'): j for j in ET.fromstring(payload).findall('./worldbody//joint')}
    bounds = []
    for name in JOINTS:
        joint = joints.get(name)
        if joint is None or joint.get('limited') != 'true':
            raise ValueError(f'Missing explicit limited joint: {name}')
        pair = np.asarray([float(x) for x in joint.get('range', '').split()])
        if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] >= pair[1]:
            raise ValueError(f'Invalid measured range: {name}')
        saved = record['ranges_rad'][name]
        if not np.allclose(pair, saved, rtol=1e-11, atol=1e-11):
            raise ValueError(f'XML and saved capture disagree for {name}')
        bounds.append(pair)
    lower, upper = np.asarray(bounds).T
    return JointLimits(path, digest, lower.copy(), upper.copy(), weight)


def configure_limit_cost(planner_config, limits, weight=None):
    if limits is None:
        if weight not in (None, 0):
            raise ValueError('Joint limit penalty requires joint_limits in the hardware config')
        return planner_config
    return replace(planner_config, joint_limit_weight=limits.weight if weight is None else weight,
                   joint_limit_lower=tuple(limits.lower), joint_limit_upper=tuple(limits.upper))


def clip_absolute_target(target, limits):
    return target if limits is None else np.clip(target, limits.lower, limits.upper)
