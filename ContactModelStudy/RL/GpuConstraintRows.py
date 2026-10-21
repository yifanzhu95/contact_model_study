"""Optional GPU row ordering for pinned dense Newton/pyramidal M1/M2."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import warp as wp

from ContactModelStudy.RL.MjwarpNumerics import _private_bindings, make_ordered_step


NUMERICS_PROFILE = "dense_newton_ordered_rows_gpu_v1"
ROW_FIELDS = ("type", "id", "J", "pos", "margin", "D", "vel", "aref", "frictionloss", "force", "state")


class ConstraintOrderingError(RuntimeError):
    pass


@wp.kernel(enable_backward=False)
def _build_keys(counts: wp.array(dtype=int), nacon: wp.array(dtype=int), capacity: int, contact_capacity: int,
                kind: wp.array2d(dtype=int), jacobian: wp.array3d(dtype=float),
                pos: wp.array2d(dtype=float), margin: wp.array2d(dtype=float), mass: wp.array2d(dtype=float),
                vel: wp.array2d(dtype=float), aref: wp.array2d(dtype=float), friction: wp.array2d(dtype=float),
                keys: wp.array3d(dtype=float), invalid: wp.array(dtype=int)):
    world, row = wp.tid()
    n = counts[world]
    if n < 0 or n >= capacity or nacon[0] < 0 or nacon[0] > contact_capacity:
        wp.atomic_max(invalid, world, 2)
        return
    if row >= n:
        return
    keys[world, row, 0] = float(kind[world, row])
    width = jacobian.shape[2]
    for col in range(width):
        value = jacobian[world, row, col]
        keys[world, row, col+1] = value
        if not wp.isfinite(value):
            wp.atomic_max(invalid, world, 1)
    keys[world, row, width+1] = pos[world, row]
    keys[world, row, width+2] = margin[world, row]
    keys[world, row, width+3] = mass[world, row]
    keys[world, row, width+4] = vel[world, row]
    keys[world, row, width+5] = aref[world, row]
    keys[world, row, width+6] = friction[world, row]
    for col in range(width+1, width+7):
        if not wp.isfinite(keys[world, row, col]):
            wp.atomic_max(invalid, world, 1)


@wp.func
def _less(keys: wp.array3d(dtype=float), world: int, a: int, b: int) -> bool:
    for col in range(keys.shape[2]):
        ka = keys[world, a, col]
        kb = keys[world, b, col]
        if ka < kb:
            return True
        if ka > kb:
            return False
    return False


@wp.kernel(enable_backward=False)
def _sort_rows(counts: wp.array(dtype=int), keys: wp.array3d(dtype=float),
               order: wp.array2d(dtype=int), inverse: wp.array2d(dtype=int)):
    world = wp.tid()
    n = counts[world]
    # Small per-world active lists. Equal physical rows retain their relative order.
    for row in range(n):
        order[world, row] = row
    for row in range(1, n):
        value = order[world, row]
        j = row-1
        while j >= 0:
            if not _less(keys, world, value, order[world, j]):
                break
            order[world, j+1] = order[world, j]
            j -= 1
        order[world, j+1] = value
    for row in range(n):
        inverse[world, order[world, row]] = row


@wp.kernel(enable_backward=False)
def _gather(counts: wp.array(dtype=int), order: wp.array2d(dtype=int), reorder: int,
            kind: wp.array2d(dtype=int), ids: wp.array2d(dtype=int), jacobian: wp.array3d(dtype=float),
            pos: wp.array2d(dtype=float), margin: wp.array2d(dtype=float), mass: wp.array2d(dtype=float),
            vel: wp.array2d(dtype=float), aref: wp.array2d(dtype=float), friction: wp.array2d(dtype=float),
            force: wp.array2d(dtype=float), status: wp.array2d(dtype=int),
            out_kind: wp.array2d(dtype=int), out_ids: wp.array2d(dtype=int), out_jacobian: wp.array3d(dtype=float),
            out_pos: wp.array2d(dtype=float), out_margin: wp.array2d(dtype=float), out_mass: wp.array2d(dtype=float),
            out_vel: wp.array2d(dtype=float), out_aref: wp.array2d(dtype=float), out_friction: wp.array2d(dtype=float),
            out_force: wp.array2d(dtype=float), out_status: wp.array2d(dtype=int)):
    world, row = wp.tid()
    if row >= counts[world]:
        return
    source = row
    if reorder != 0:
        source = order[world, row]
    out_kind[world, row] = kind[world, source]
    out_ids[world, row] = ids[world, source]
    for col in range(jacobian.shape[2]):
        out_jacobian[world, row, col] = jacobian[world, source, col]
    out_pos[world, row] = pos[world, source]
    out_margin[world, row] = margin[world, source]
    out_mass[world, row] = mass[world, source]
    out_vel[world, row] = vel[world, source]
    out_aref[world, row] = aref[world, source]
    out_friction[world, row] = friction[world, source]
    out_force[world, row] = force[world, source]
    out_status[world, row] = status[world, source]


@wp.kernel(enable_backward=False)
def _remap_contacts(nacon: wp.array(dtype=int), counts: wp.array(dtype=int), worlds: wp.array(dtype=int),
                    inverse: wp.array2d(dtype=int), addresses: wp.array2d(dtype=int)):
    contact, edge = wp.tid()
    if contact >= nacon[0]:
        return
    world = worlds[contact]
    old = addresses[contact, edge]
    if world >= 0 and world < counts.shape[0] and old >= 0 and old < counts[world]:
        addresses[contact, edge] = inverse[world, old]


class GpuConstraintRows:
    def __init__(self, data):
        if data.efc.J_rownnz.shape[1] or data.efc.J_colind.shape[1]:
            raise ValueError("GPU row ordering requires the pinned dense Jacobian layout")
        self.capacity = data.njmax
        self.shape = (data.nworld, self.capacity)
        self.device = data.qpos.device
        self.keys = wp.empty((*self.shape, data.efc.J.shape[2]+7), dtype=float, device=self.device)
        self.order = wp.empty(self.shape, dtype=int, device=self.device)
        self.inverse = wp.empty(self.shape, dtype=int, device=self.device)
        self.invalid = wp.zeros(data.nworld, dtype=int, device=self.device)
        self.scratch = [wp.empty_like(getattr(data.efc, name)) for name in ROW_FIELDS]

    def apply(self, data):
        if (data.nworld, data.njmax) != self.shape:
            raise ValueError("GPU ordering buffers belong to a different Data layout")
        self.invalid.zero_()
        wp.launch(_build_keys, dim=self.shape, inputs=[data.nefc, data.nacon, self.capacity, data.naconmax,
            data.efc.type, data.efc.J, data.efc.pos, data.efc.margin, data.efc.D, data.efc.vel,
            data.efc.aref, data.efc.frictionloss, self.keys, self.invalid], device=self.device)
        wp.synchronize_device(self.device)
        # Only one safety scalar crosses to the host; keys/rows stay on the GPU.
        invalid = int(wp.to_torch(self.invalid).max().item())
        if invalid:
            raise ConstraintOrderingError("Invalid constraint count/capacity" if invalid == 2 else "Nonfinite constraint ordering key")
        wp.launch(_sort_rows, dim=data.nworld, inputs=[data.nefc, self.keys, self.order, self.inverse], device=self.device)
        source = [getattr(data.efc, name) for name in ROW_FIELDS]
        wp.launch(_gather, dim=self.shape, inputs=[data.nefc, self.order, 1, *source, *self.scratch], device=self.device)
        wp.launch(_gather, dim=self.shape, inputs=[data.nefc, self.order, 0, *self.scratch, *source], device=self.device)
        wp.launch(_remap_contacts, dim=data.contact.efc_address.shape,
                  inputs=[data.nacon, data.nefc, data.contact.worldid, self.inverse, data.contact.efc_address], device=self.device)

    def describe(self):
        return {"profile": NUMERICS_PROFILE, "key_order": ["type", "J", "pos", "margin", "D", "vel", "aref", "frictionloss"],
                "sorting_device": str(self.device), "host_transfer": "one_validation_scalar_per_physics_step",
                "buffer_reuse": True, "active_prefix_only": True, "contact_addresses_remapped": True}


def make_ordered_rows_step(model, data):
    from comfree_warp.mujoco_warp._src import forward
    previous = make_ordered_step(model)
    previous_solver = inspect.unwrap(previous).__globals__["solver"].solve
    ordering = GpuConstraintRows(data)
    def solve(m, d):
        ordering.apply(d)
        previous_solver(m, d)
    private = _private_bindings(forward, ("forward", "step"), solver=SimpleNamespace(solve=solve))
    return private["step"], ordering
