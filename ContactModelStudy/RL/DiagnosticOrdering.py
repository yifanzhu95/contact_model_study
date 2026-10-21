"""CPU-assisted constraint permutations for diagnosis only; never selected by backends."""
from dataclasses import fields
import inspect
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from ContactModelStudy.RL.MjwarpNumerics import _private_bindings


ROW_FIELDS = ("type", "id", "J", "pos", "margin", "D", "vel", "aref", "frictionloss", "force", "state")
KEY_FIELDS = ("type", "J", "pos", "margin", "D", "vel", "aref", "frictionloss")


def row_order(values, reverse=False):
    count = len(values["type"])
    if reverse:
        # Preserve the engine's equality/friction/limit/contact category prefixes.
        return np.asarray(sorted(range(count), key=lambda i: (int(values["type"][i]), -i)), dtype=np.int64)
    def key(i):
        return tuple(float(x) for name in KEY_FIELDS for x in np.asarray(values[name][i]).reshape(-1))
    return np.asarray(sorted(range(count), key=key), dtype=np.int64)


def constraint_rows(data):
    wp.synchronize()
    counts = wp.to_torch(data.nefc).cpu().numpy()
    return [{name: wp.to_torch(getattr(data.efc, name))[world, :int(count)].cpu().numpy().copy()
             for name in KEY_FIELDS} for world, count in enumerate(counts)]


def permute_rows(data, mode):
    """Same read/sync work for read_only; preserve dense row metadata and addresses."""
    if mode not in ("read_only", "canonical", "reverse"):
        raise ValueError(mode)
    rows = constraint_rows(data)
    if wp.to_torch(data.efc.J_rownnz).shape[1]:
        raise ValueError("Diagnostic permutation only supports dense Jacobians")
    orders = [row_order(values, reverse=mode == "reverse") for values in rows]
    if mode == "read_only":
        return orders
    worlds = wp.to_torch(data.contact.worldid)
    addresses = wp.to_torch(data.contact.efc_address)
    nacon = int(wp.to_torch(data.nacon)[0])
    for world, order in enumerate(orders):
        n = len(order)
        if not n:
            continue
        index = torch.tensor(order, dtype=torch.int64, device=data.qpos.device.alias)
        inverse = torch.empty_like(index)
        inverse[index] = torch.arange(n, device=index.device)
        for name in ROW_FIELDS:
            tensor = wp.to_torch(getattr(data.efc, name))
            tensor[world, :n] = tensor[world, :n].index_select(0, index)
        mask = (worlds[:nacon] == world)[:, None] & (addresses[:nacon] >= 0) & (addresses[:nacon] < n)
        addresses[:nacon][mask] = inverse[addresses[:nacon][mask].long()].to(addresses.dtype)
    torch.cuda.synchronize()
    return orders


def diagnostic_step(backend, *, mode=None, capture=None):
    """Bind the original dispatch to private hooks; original numerical kernels retained."""
    from comfree_warp.mujoco_warp._src import forward
    existing = inspect.unwrap(backend._step_physics).__globals__
    original_solver = existing["solver"].solve
    def wrap(name, function):
        def hooked(*args, **kwargs):
            result = function(*args, **kwargs)
            wp.synchronize()
            if capture:
                capture(name, args[1])
            return result
        return hooked
    def solve(m, d):
        if mode:
            permute_rows(d, mode)
        if capture:
            capture("pre_solver", d)
        result = original_solver(m, d)
        wp.synchronize()
        if capture:
            capture("solver", d)
        return result
    # fwd_position needs its own private globals to intercept the two earlier phases.
    position_scope = _private_bindings(forward, ("fwd_position",),
        collision_driver=SimpleNamespace(collision=wrap("collision", forward.collision_driver.collision)),
        constraint=SimpleNamespace(make_constraint=wrap("constraints", forward.constraint.make_constraint)))
    overrides = {name: wrap(name, getattr(forward, name)) for name in
                 ("fwd_velocity", "fwd_actuation", "fwd_acceleration", "euler", "implicit")}
    overrides.update(fwd_position=position_scope["fwd_position"], solver=SimpleNamespace(solve=solve))
    return _private_bindings(forward, ("forward", "step"), **overrides)["step"]
