"""Capture-safe cumulative diagnostics for integrated Kamino rollouts.

This module imports Newton's private ``PADMMStatus`` type, so it must only be
imported after the optional Newton dependency is known to be available.  The
public ``KaminoConfig`` remains importable in environments without Newton.

The accumulator is the integrated counterpart of the device-side diagnostics
in ``kamino_reference/batched_target_rollout.py``.  Every operation below is a
Warp kernel launch or an in-place device clear, so it is safe inside a CUDA
graph.  Host downloads happen only after a rollout has finished.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from newton._src.solvers.kamino._src.solvers.padmm.types import PADMMStatus


@wp.kernel
def _accumulate_solver_diagnostics_kernel(
    status: wp.array(dtype=PADMMStatus),
    active_contacts: wp.array(dtype=wp.int32),
    primal_tolerance: float,
    dual_tolerance: float,
    complementarity_tolerance: float,
    converged_steps: wp.array(dtype=wp.int32),
    contact_fine_steps: wp.array(dtype=wp.int32),
    primal_failure_steps: wp.array(dtype=wp.int32),
    dual_failure_steps: wp.array(dtype=wp.int32),
    complementarity_failure_steps: wp.array(dtype=wp.int32),
    max_iterations: wp.array(dtype=wp.int32),
    max_active_contacts: wp.array(dtype=wp.int32),
    max_primal_residual: wp.array(dtype=wp.float32),
    max_dual_residual: wp.array(dtype=wp.float32),
    max_complementarity_residual: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    current = status[world]
    if current.converged == 1:
        converged_steps[world] = converged_steps[world] + 1
    if active_contacts[world] > 0:
        contact_fine_steps[world] = contact_fine_steps[world] + 1
    if not wp.isfinite(current.r_p) or current.r_p > primal_tolerance:
        primal_failure_steps[world] = primal_failure_steps[world] + 1
    if not wp.isfinite(current.r_d) or current.r_d > dual_tolerance:
        dual_failure_steps[world] = dual_failure_steps[world] + 1
    if not wp.isfinite(current.r_c) or current.r_c > complementarity_tolerance:
        complementarity_failure_steps[world] = complementarity_failure_steps[world] + 1
    if current.iterations > max_iterations[world]:
        max_iterations[world] = current.iterations
    if active_contacts[world] > max_active_contacts[world]:
        max_active_contacts[world] = active_contacts[world]
    if wp.isfinite(current.r_p) and current.r_p > max_primal_residual[world]:
        max_primal_residual[world] = current.r_p
    if wp.isfinite(current.r_d) and current.r_d > max_dual_residual[world]:
        max_dual_residual[world] = current.r_d
    if wp.isfinite(current.r_c) and current.r_c > max_complementarity_residual[world]:
        max_complementarity_residual[world] = current.r_c


@wp.kernel
def _strict_rollout_validity_kernel(
    reset_success: wp.array(dtype=wp.bool),
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    running_costs: wp.array(dtype=wp.float32),
    terminal_costs: wp.array(dtype=wp.float32),
    converged_steps: wp.array(dtype=wp.int32),
    primal_failure_steps: wp.array(dtype=wp.int32),
    dual_failure_steps: wp.array(dtype=wp.int32),
    complementarity_failure_steps: wp.array(dtype=wp.int32),
    max_iterations: wp.array(dtype=wp.int32),
    max_active_contacts: wp.array(dtype=wp.int32),
    expected_fine_steps: int,
    iteration_limit: int,
    contact_capacity: int,
    valid: wp.array(dtype=wp.bool),
    finite_state: wp.array(dtype=wp.bool),
    finite_cost: wp.array(dtype=wp.bool),
):
    world = wp.tid()
    # Seed from device data so Warp treats this as a mutable dynamic boolean;
    # a constant ``True`` cannot be reassigned inside a dynamic loop.
    state_ok = wp.isfinite(qpos[world, 0])
    for i in range(qpos.shape[1]):
        if not wp.isfinite(qpos[world, i]):
            state_ok = False
    for i in range(qvel.shape[1]):
        if not wp.isfinite(qvel[world, i]):
            state_ok = False
    cost_ok = wp.isfinite(running_costs[world]) and wp.isfinite(terminal_costs[world])
    solver_ok = (
        converged_steps[world] == expected_fine_steps
        and primal_failure_steps[world] == 0
        and dual_failure_steps[world] == 0
        and complementarity_failure_steps[world] == 0
        and max_iterations[world] < iteration_limit
    )
    contacts_ok = contact_capacity <= 0 or max_active_contacts[world] < contact_capacity
    ok = reset_success[world] and state_ok and cost_ok and solver_ok and contacts_ok
    valid[world] = ok
    finite_state[world] = state_ok
    finite_cost[world] = cost_ok
    if not ok:
        # MPPI treats NaN costs as zero-weight candidates.  Mark both pieces so
        # any later cost folding stays NaN and cannot influence the action.
        running_costs[world] = wp.nan
        terminal_costs[world] = wp.nan


class KaminoRolloutDiagnostics:
    """Device accumulators and compact host summaries for one rollout."""

    _COUNTERS = (
        "converged_steps",
        "contact_fine_steps",
        "primal_failure_steps",
        "dual_failure_steps",
        "complementarity_failure_steps",
        "max_iterations",
        "max_active_contacts",
    )
    _RESIDUALS = (
        "max_primal_residual",
        "max_dual_residual",
        "max_complementarity_residual",
    )

    def __init__(self, world_count: int, device):
        self.world_count = int(world_count)
        self.device = device
        for name in self._COUNTERS:
            setattr(self, name, wp.zeros(world_count, dtype=wp.int32, device=device))
        for name in self._RESIDUALS:
            setattr(self, name, wp.zeros(world_count, dtype=wp.float32, device=device))
        self.valid = wp.zeros(world_count, dtype=wp.bool, device=device)
        self.finite_state = wp.zeros(world_count, dtype=wp.bool, device=device)
        self.finite_cost = wp.zeros(world_count, dtype=wp.bool, device=device)

    def reset(self) -> None:
        """Clear counters on device; safe when called inside graph capture."""
        for name in self._COUNTERS + self._RESIDUALS:
            getattr(self, name).zero_()
        self.valid.zero_()
        self.finite_state.zero_()
        self.finite_cost.zero_()

    def accumulate(self, status, active_contacts, tolerance: float) -> None:
        wp.launch(
            _accumulate_solver_diagnostics_kernel,
            dim=self.world_count,
            inputs=[
                status,
                active_contacts,
                tolerance,
                tolerance,
                tolerance,
                self.converged_steps,
                self.contact_fine_steps,
                self.primal_failure_steps,
                self.dual_failure_steps,
                self.complementarity_failure_steps,
                self.max_iterations,
                self.max_active_contacts,
                self.max_primal_residual,
                self.max_dual_residual,
                self.max_complementarity_residual,
            ],
            device=self.device,
        )

    def invalidate_costs(
        self,
        *,
        reset_success,
        qpos,
        qvel,
        running_costs,
        terminal_costs,
        expected_fine_steps: int,
        iteration_limit: int,
        contact_capacity: int,
    ) -> None:
        """Mark invalid candidate costs NaN before the planner weights them."""
        wp.launch(
            _strict_rollout_validity_kernel,
            dim=self.world_count,
            inputs=[
                reset_success,
                qpos,
                qvel,
                running_costs,
                terminal_costs,
                self.converged_steps,
                self.primal_failure_steps,
                self.dual_failure_steps,
                self.complementarity_failure_steps,
                self.max_iterations,
                self.max_active_contacts,
                int(expected_fine_steps),
                int(iteration_limit),
                int(contact_capacity),
            ],
            outputs=[self.valid, self.finite_state, self.finite_cost],
            device=self.device,
        )

    def summary(self, expected_fine_steps: int, iteration_limit: int, contact_capacity: int) -> dict:
        """Download a compact, JSON-ready strict-validity summary."""
        arrays = {name: getattr(self, name).numpy() for name in self._COUNTERS + self._RESIDUALS}
        valid = self.valid.numpy().astype(bool)
        finite_state = self.finite_state.numpy().astype(bool)
        finite_cost = self.finite_cost.numpy().astype(bool)

        def worlds(mask) -> int:
            return int(np.count_nonzero(mask))

        blocking = {
            "nonconverged_worlds": worlds(arrays["converged_steps"] != expected_fine_steps),
            "primal_residual_worlds": worlds(arrays["primal_failure_steps"] > 0),
            "dual_residual_worlds": worlds(arrays["dual_failure_steps"] > 0),
            "complementarity_residual_worlds": worlds(
                arrays["complementarity_failure_steps"] > 0
            ),
            "iteration_limit_worlds": worlds(arrays["max_iterations"] >= iteration_limit),
            "contact_capacity_worlds": (
                worlds(arrays["max_active_contacts"] >= contact_capacity)
                if contact_capacity > 0 else 0
            ),
            "nonfinite_state_worlds": worlds(~finite_state),
            "nonfinite_cost_worlds": worlds(~finite_cost),
        }
        return {
            "schema": "contact_model_study.kamino_rollout_diagnostics.v1",
            "world_count": self.world_count,
            "expected_fine_steps": int(expected_fine_steps),
            "valid_worlds": worlds(valid),
            "invalid_worlds": worlds(~valid),
            "all_worlds_valid": bool(valid.all()),
            "converged_steps_min": int(arrays["converged_steps"].min()),
            "converged_steps_max": int(arrays["converged_steps"].max()),
            "contact_fine_steps_max": int(arrays["contact_fine_steps"].max()),
            "max_iterations": int(arrays["max_iterations"].max()),
            "max_active_contacts": int(arrays["max_active_contacts"].max()),
            "contact_capacity": int(contact_capacity),
            "max_primal_residual": float(arrays["max_primal_residual"].max()),
            "max_dual_residual": float(arrays["max_dual_residual"].max()),
            "max_complementarity_residual": float(
                arrays["max_complementarity_residual"].max()
            ),
            "blocking_reasons": {name: count for name, count in blocking.items() if count},
        }
