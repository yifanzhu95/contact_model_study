"""Comfree engine API built on top of comfree_core patches."""

import numpy as np
import warp as wp

from .comfree_core._src.forward import forward_comfree as forward
from .comfree_core._src.forward import step_comfree as step
from .comfree_core._src.types import Data as Data
from .comfree_core._src.types import Model as Model

from . import mujoco_warp as _mjwarp


def _ensure_comfree_fields(d):
  # Constraint comfree fields
  shape = d.efc.pos.shape
  if not hasattr(d.efc, "efc_dist"):
    d.efc.efc_dist = wp.zeros(shape, dtype=float)
  if not hasattr(d.efc, "efc_mass"):
    d.efc.efc_mass = wp.zeros(shape, dtype=float)

  # Data comfree fields
  if not hasattr(d, "qvel_smooth_pred"):
    d.qvel_smooth_pred = wp.zeros(d.qvel.shape, dtype=float)
  if not hasattr(d, "qfrc_total"):
    d.qfrc_total = wp.zeros(d.qvel.shape, dtype=float)

  return d


def put_model(*args, **kwargs):

  comfree_stiffness = kwargs.pop("comfree_stiffness", 0.2)
  comfree_damping = kwargs.pop("comfree_damping", 0.001)

  m = _mjwarp.put_model(*args, **kwargs)
  device = m.opt.timestep.device
  m.comfree_stiffness = wp.array(np.atleast_1d(comfree_stiffness), dtype=wp.float32, device=device)
  m.comfree_damping = wp.array(np.atleast_1d(comfree_damping), dtype=wp.float32, device=device)
  return m


def put_data(*args, **kwargs):
  d = _mjwarp.put_data(*args, **kwargs)
  return _ensure_comfree_fields(d)


def make_data(*args, **kwargs):
  d = _mjwarp.make_data(*args, **kwargs)
  return _ensure_comfree_fields(d)


def get_data_into(*args, **kwargs):
  d = _mjwarp.get_data_into(*args, **kwargs)
  return _ensure_comfree_fields(d)


def reset_data(*args, **kwargs):
  return _mjwarp.reset_data(*args, **kwargs)

__all__ = [
  "step",
  "forward",
  "Model",
  "Data",
  "get_data_into",
  "make_data",
  "put_data",
  "put_model",
  "reset_data",
]
