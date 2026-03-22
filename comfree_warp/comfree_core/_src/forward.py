# Copyright (c) 2026 ASU IRIS
# Licensed for noncommercial academic research use only.
# See comfree_warp/comfree_core/LICENSE for terms.
# -----------------------------------------------------------------------------
import warp as wp

from comfree_warp.mujoco_warp._src import collision_driver
from comfree_warp.mujoco_warp._src import sensor
from comfree_warp.mujoco_warp._src import smooth
from comfree_warp.mujoco_warp._src.forward import euler
from comfree_warp.mujoco_warp._src.forward import fwd_acceleration
from comfree_warp.mujoco_warp._src.forward import fwd_actuation
from comfree_warp.mujoco_warp._src.forward import fwd_velocity
from comfree_warp.mujoco_warp._src.forward import implicit
from comfree_warp.mujoco_warp._src.warp_util import event_scope

from . import constraint
from .types import Data
from .types import EnableBit
from .types import IntegratorType
from .types import Model
from .types import DisableBit

wp.set_module_options({"enable_backward": False})

@wp.kernel
def _advance_vel(
  # Model:
  opt_timestep: wp.array(dtype=float),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  qacc_smooth_in: wp.array2d(dtype=float),
  # Data out:
  qvel_out: wp.array2d(dtype=float),
  qfrc_constraint: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_smooth_in[worldid, dofid] * timestep
  qfrc_constraint[worldid, dofid] = 0.0


@wp.kernel
def _compute_qfrc_constraint(
  # Model:
  opt_timestep: wp.array(dtype=float),
  # comfree model parameter
  comfree_stiffness: wp.array(dtype=float),
  comfree_damping: wp.array(dtype=float),
  # Data in: 
  J: wp.array3d(dtype=float),
  efc_dist: wp.array2d(dtype=float),
  efc_mass: wp.array2d(dtype=float),
  qvel_smooth_pred: wp.array2d(dtype=float),
  nv: int,
  nefc: wp.array(dtype=int),
  # Out: 
  efc_force: wp.array2d(dtype=float),
  qfrc_constraint: wp.array2d(dtype=float),
):
  worldid, efcid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  if efcid >= nefc[worldid]:
    return
  
  efc_vel = float(0.0)
  for i in range(nv):
    efc_vel += J[worldid, efcid, i] * qvel_smooth_pred[worldid, i]

  # key steps in the comfree paper
  stiffness = comfree_stiffness[worldid % comfree_stiffness.shape[0]] / timestep
  damping = comfree_damping[worldid % comfree_damping.shape[0]] / timestep
  # predictive penetration with smoothing velocity
  efc_penetration = efc_vel * timestep + efc_dist[worldid, efcid]
  # efc_acc = -damping * efc_b[worldid, efcid] * efc_vel - stiffness * efc_k[worldid, efcid] * efc_penetration
  efc_acc = -damping *  efc_vel - stiffness *  efc_penetration
  efc_frc=  efc_mass[worldid, efcid] * efc_acc
  efc_frc= wp.max(efc_frc, 0.0)

  # output
  efc_force[worldid, efcid] =efc_frc
  for i in range(nv):
    # qfrc_constraint[worldid][i] +=  J[worldid, efcid, i] * efc_frc
    wp.atomic_add(qfrc_constraint, worldid, i, J[worldid, efcid, i] * efc_frc)



@wp.kernel
def _compute_qfrc_total(
  qfrc_smooth: wp.array2d(dtype=float),
  qfrc_constraint: wp.array2d(dtype=float),
  qfrc_total: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  qfrc_total[worldid, dofid] = qfrc_smooth[worldid, dofid] + qfrc_constraint[worldid, dofid]


@event_scope
def compute_qfrc_total(m: Model, d: Data):

  wp.launch(
    _advance_vel,
    dim=(d.nworld, m.nv),
    inputs=[
      m.opt.timestep,
      d.qvel,
      d.qacc_smooth,
    ],
    outputs=[
      d.qvel_smooth_pred,
      d.qfrc_constraint,
    ],
  )

  wp.launch(
    _compute_qfrc_constraint,
    dim=(d.nworld, d.efc.J.shape[1]),
    inputs=[
      m.opt.timestep,
      m.comfree_stiffness,
      m.comfree_damping,  
      d.efc.J,
      d.efc.efc_dist,
      d.efc.efc_mass,
      d.qvel_smooth_pred,
      m.nv,
      d.nefc,
    ],
    outputs=[
      d.efc.force,
      d.qfrc_constraint,
    ]
  )

  wp.launch(
    _compute_qfrc_total,
    dim=(d.nworld, m.nv),
    inputs=[
      d.qfrc_smooth,
      d.qfrc_constraint,
    ],
    outputs=[
      d.qfrc_total,
    ],
  )


@event_scope
def forward_comfree(m: Model, d: Data, factorize: bool = True):
  """Forward dynamics with complementarity-free model."""

  # forward position
  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  smooth.camlight(m, d)
  smooth.flex(m, d)
  smooth.tendon(m, d)
  smooth.crb(m, d)
  smooth.tendon_armature(m, d)
  if factorize:
    smooth.factor_m(m, d)
  if m.opt.run_collision_detection:
    collision_driver.collision(m, d)
  constraint.make_constraint(m, d) 
  smooth.transmission(m, d)

  d.sensordata.zero_()
  sensor.sensor_pos(m, d)
  energy = m.opt.enableflags & EnableBit.ENERGY
  if energy:
    if m.sensor_e_potential == 0:  # not computed by sensor
      sensor.energy_pos(m, d)
  else:
    d.energy.zero_()

  # forward velocity
  fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  if energy:
    if m.sensor_e_kinetic == 0:  # not computed by sensor
      sensor.energy_vel(m, d)

  # forward actuation and smooth acceleration
  if not (m.opt.disableflags & DisableBit.ACTUATION):
    if m.callback.control:
      m.callback.control(m, d)
  fwd_actuation(m, d)
  fwd_acceleration(m, d, factorize=True)

  # call comfree model to resolve constraints
  if d.njmax == 0 or m.nv == 0:
    wp.copy(d.qacc, d.qacc_smooth)
  else:
    compute_qfrc_total(m, d)
    smooth.solve_m(m, d, d.qacc, d.qfrc_total)

  sensor.sensor_acc(m, d)


@event_scope
def step_comfree(m: Model, d: Data):
  """Advance simulation with complementarity-free model."""
  forward_comfree(m, d)

  if m.opt.integrator == IntegratorType.EULER:
    wp.copy(d.efc.Ma, d.qfrc_total)
    euler(m, d)
  elif m.opt.integrator == IntegratorType.IMPLICITFAST:
    wp.copy(d.efc.Ma, d.qfrc_total)
    implicit(m, d)
  else:
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")
