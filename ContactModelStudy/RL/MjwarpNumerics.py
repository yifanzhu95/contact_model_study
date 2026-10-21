"""Fixed ordering of changed constraints before dense Newton Hessian updates."""
from __future__ import annotations

import inspect
from types import FunctionType, SimpleNamespace

import warp as wp


NUMERICS_PROFILE = "dense_newton_ordered_changes_v1"


@wp.kernel(enable_backward=False)
def _sort_changed_constraints(ids: wp.array2d(dtype=int), counts: wp.array(dtype=int)):
    world = wp.tid()
    # One thread owns each world's short list. Only the initialized prefix
    # participates; the unused tail may contain arbitrary allocation contents.
    for i in range(1, counts[world]):
        value = ids[world, i]
        j = i - 1
        while j >= 0:
            if ids[world, j] <= value:
                break
            ids[world, j + 1] = ids[world, j]
            j -= 1
        ids[world, j + 1] = value


def order_changed_constraints(data, context):
    wp.launch(_sort_changed_constraints, dim=data.nworld,
              inputs=[context.changed_efc_ids, context.changed_efc_count],
              device=data.qpos.device)


def _private_bindings(module, names, **overrides):
    """Bind the pinned library's dispatch bytecode to an instance-private scope.

    The numeric kernels and original function bodies are reused. Module globals
    stay untouched, so native simulators in the same process keep their behavior.
    Preserve the library's optional event timing wrappers as well.
    """
    scope = dict(vars(module))
    scope.update(overrides)
    for name in names:
        decorated = getattr(module, name)
        source = inspect.unwrap(decorated)
        if not isinstance(source, FunctionType) or source.__closure__ is not None:
            raise RuntimeError(f"Unexpected pinned MJWarp dispatch function: {name}")
        bound = FunctionType(source.__code__, scope, source.__name__, source.__defaults__)
        bound.__kwdefaults__ = source.__kwdefaults__
        bound.__qualname__, bound.__doc__ = source.__qualname__, source.__doc__
        scope[name] = module.event_scope(bound) if decorated is not source else bound
    return scope


def make_ordered_step(model):
    """Original step/forward/Newton dispatch plus one ordered-list boundary.

    This is an explicitly versioned numerical variant of the pinned backend,
    not a new contact law. It changes the summation order of the dense Newton
    incremental Hessian update, never tolerances, iteration caps or forces.
    """
    from comfree_warp.mujoco_warp._src import forward, solver, types
    if (solver.SPARSE_CONSTRAINT_JACOBIAN or model.opt.solver != types.SolverType.NEWTON
            or model.opt.cone != types.ConeType.PYRAMIDAL):
        raise ValueError("Ordered changes currently supports dense-J Newton/pyramidal only")
    native_incremental = solver._update_gradient_incremental

    def ordered_incremental(m, d, context):
        order_changed_constraints(d, context)
        native_incremental(m, d, context)

    private_solver = _private_bindings(solver, ("solve", "_solve", "_solver_iteration"),
                                      _update_gradient_incremental=ordered_incremental)
    private_forward = _private_bindings(forward, ("forward", "step"),
                                       solver=SimpleNamespace(solve=private_solver["solve"]))
    return private_forward["step"]
