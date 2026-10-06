"""Device-resident multi-world rollouts for the real Leap MJCF scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import newton
import numpy as np
import warp as wp
from newton._src.solvers.kamino._src.solvers.padmm.types import PADMMStatus

from .mjcf_probe import ScenePathResolver, infer_position_target_indices
from .reference_planning import (
    ACTION_SEMANTICS as REFERENCE_ACTION_SEMANTICS,
    COST_SEMANTICS as REFERENCE_COST_SEMANTICS,
    ReferencePlanningBundle,
    file_sha256,
)
from .target_adapter import (
    SiteRecordingModelBuilder,
    _free_joint_q_starts,
    mujoco_qpos_to_newton,
    newton_qpos_to_mujoco,
)


@wp.kernel
def _broadcast_joint_q_kernel(
    measured_q: wp.array(dtype=wp.float32),
    joint_coord_world_start: wp.array(dtype=wp.int32),
    joint_q: wp.array(dtype=wp.float32),
):
    world, local_coord = wp.tid()
    joint_q[joint_coord_world_start[world] + local_coord] = measured_q[local_coord]


@wp.kernel
def _broadcast_joint_qd_kernel(
    measured_qd: wp.array(dtype=wp.float32),
    joint_dof_world_start: wp.array(dtype=wp.int32),
    joint_qd: wp.array(dtype=wp.float32),
):
    world, local_dof = wp.tid()
    joint_qd[joint_dof_world_start[world] + local_dof] = measured_qd[local_dof]


@wp.kernel
def _write_position_targets_kernel(
    step_index: int,
    action_sequences: wp.array3d(dtype=wp.float32),
    control_target_indices: wp.array2d(dtype=wp.int32),
    joint_target_q: wp.array(dtype=wp.float32),
):
    world, actuator = wp.tid()
    target_index = control_target_indices[world, actuator]
    joint_target_q[target_index] = action_sequences[step_index, world, actuator]


@wp.kernel
def _write_relative_position_targets_kernel(
    step_index: int,
    action_sequences: wp.array3d(dtype=wp.float32),
    joint_coord_world_start: wp.array(dtype=wp.int32),
    robot_qpos_address: int,
    joint_q: wp.array(dtype=wp.float32),
    control_target_indices: wp.array2d(dtype=wp.int32),
    joint_target_q: wp.array(dtype=wp.float32),
):
    """Apply the main planner's bounded servo parameterization on device."""

    world, actuator = wp.tid()
    source_index = (
        joint_coord_world_start[world] + robot_qpos_address + actuator
    )
    target_index = control_target_indices[world, actuator]
    joint_target_q[target_index] = (
        joint_q[source_index] + action_sequences[step_index, world, actuator]
    )


@wp.kernel
def _compute_batched_site_transforms_kernel(
    site_shape_indices: wp.array2d(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transformf),
    body_q: wp.array(dtype=wp.transformf),
    site_world: wp.array(dtype=wp.transformf),
    site_count: int,
):
    world, site = wp.tid()
    shape_index = site_shape_indices[world, site]
    body_index = shape_body[shape_index]
    local_transform = shape_transform[shape_index]
    output_index = world * site_count + site
    if body_index >= 0:
        site_world[output_index] = wp.transform_multiply(
            body_q[body_index], local_transform
        )
    else:
        site_world[output_index] = local_transform


@wp.kernel
def _accumulate_grasp_reorient_cost_kernel(
    terminal: bool,
    joint_coord_world_start: wp.array(dtype=wp.int32),
    joint_dof_world_start: wp.array(dtype=wp.int32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    site_world: wp.array(dtype=wp.transformf),
    fingertip_site_local_indices: wp.array(dtype=wp.int32),
    site_count: int,
    goal: wp.array(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    costs: wp.array(dtype=wp.float32),
):
    """Source-faithful grasp-reorient cost in Newton coordinate storage.

    Newton stores a free quaternion as ``xyzw`` while the shared goal and the
    main project store it as ``wxyz``.  The object starts at local qpos 16 and
    qvel 16; the sixteen manipulated hand joints start at zero.
    """

    world = wp.tid()
    q_start = joint_coord_world_start[world]
    qd_start = joint_dof_world_start[world]
    object_q_start = q_start + 16
    object_qd_start = qd_start + 16

    px = joint_q[object_q_start]
    py = joint_q[object_q_start + 1]
    pz = joint_q[object_q_start + 2]
    # Newton object quaternion: q[19:23] == (x, y, z, w).
    dot = (
        goal[3] * joint_q[object_q_start + 6]
        + goal[4] * joint_q[object_q_start + 3]
        + goal[5] * joint_q[object_q_start + 4]
        + goal[6] * joint_q[object_q_start + 5]
    )
    quaternion_cost = 1.0 - dot * dot
    dx = px - goal[0]
    dy = py - goal[1]
    dz = pz - goal[2]
    position_l2 = wp.sqrt(dx * dx + dy * dy + dz * dz)

    joint_home_cost = float(0.0)
    joint_velocity_cost = float(0.0)
    for joint in range(16):
        joint_error = joint_q[q_start + joint] - goal[7 + joint]
        joint_home_cost = joint_home_cost + joint_error * joint_error
        joint_velocity = joint_qd[qd_start + joint]
        joint_velocity_cost = (
            joint_velocity_cost + joint_velocity * joint_velocity
        )

    fingertip_distance = float(0.0)
    for fingertip in range(4):
        local_site = fingertip_site_local_indices[fingertip]
        site_position = wp.transform_get_translation(
            site_world[world * site_count + local_site]
        )
        tip_dx = site_position[0] - px
        tip_dy = site_position[1] - py
        tip_dz = site_position[2] - pz
        fingertip_distance = fingertip_distance + wp.sqrt(
            tip_dx * tip_dx + tip_dy * tip_dy + tip_dz * tip_dz
        )

    fallen = float(0.0)
    if pz < goal[23]:
        fallen = 1.0

    # Keep these reads visible in the source contract.  The live main-project
    # cost computes object velocity but currently does not use it: weights[4]
    # multiplies position_l2.  The reference path must preserve that behavior.
    object_velocity_cost = float(0.0)
    for component in range(6):
        value = joint_qd[object_qd_start + component]
        object_velocity_cost = object_velocity_cost + value * value

    cost = (
        weights[0] * quaternion_cost
        + weights[1] * wp.abs(dx)
        + weights[2] * wp.abs(dy)
        + weights[3] * wp.abs(dz)
        + weights[4] * position_l2
        + weights[5] * fingertip_distance
        + weights[6] * joint_home_cost
        + weights[7] * joint_velocity_cost
        + weights[8] * fallen
        + 0.0 * object_velocity_cost
    )
    if terminal:
        cost = (
            weights[9] * quaternion_cost
            + weights[10] * position_l2
            + weights[11] * fallen
        )
    costs[world] = costs[world] + cost


@wp.kernel
def _accumulate_target_stage_cost_kernel(
    step_index: int,
    action_sequences: wp.array3d(dtype=wp.float32),
    body_world_start: wp.array(dtype=wp.int32),
    object_body_local_index: int,
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    site_world: wp.array(dtype=wp.transformf),
    target_object_position: wp.array(dtype=wp.vec3f),
    object_position_weight: float,
    object_velocity_weight: float,
    fingertip_distance_weight: float,
    control_weight: float,
    dt: float,
    costs: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    object_body = body_world_start[world] + object_body_local_index
    object_position = wp.transform_get_translation(body_q[object_body])
    object_velocity = body_qd[object_body]
    position_error = object_position - target_object_position[0]

    site_offset = world * 5
    fingertip_cost = float(0.0)
    for site in range(5):
        site_position = wp.transform_get_translation(site_world[site_offset + site])
        site_error = site_position - object_position
        fingertip_cost = fingertip_cost + wp.dot(site_error, site_error)

    control_cost = float(0.0)
    for actuator in range(16):
        value = action_sequences[step_index, world, actuator]
        control_cost = control_cost + value * value

    linear_speed_sq = (
        object_velocity[0] * object_velocity[0]
        + object_velocity[1] * object_velocity[1]
        + object_velocity[2] * object_velocity[2]
    )
    angular_speed_sq = (
        object_velocity[3] * object_velocity[3]
        + object_velocity[4] * object_velocity[4]
        + object_velocity[5] * object_velocity[5]
    )
    costs[world] = costs[world] + dt * (
        object_position_weight * wp.dot(position_error, position_error)
        + object_velocity_weight * (linear_speed_sq + angular_speed_sq)
        + fingertip_distance_weight * fingertip_cost
        + control_weight * control_cost
    )


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
    current_status = status[world]
    if current_status.converged == 1:
        converged_steps[world] = converged_steps[world] + 1
    if active_contacts[world] > 0:
        contact_fine_steps[world] = contact_fine_steps[world] + 1
    if current_status.r_p > primal_tolerance:
        primal_failure_steps[world] = primal_failure_steps[world] + 1
    if current_status.r_d > dual_tolerance:
        dual_failure_steps[world] = dual_failure_steps[world] + 1
    if current_status.r_c > complementarity_tolerance:
        complementarity_failure_steps[world] = (
            complementarity_failure_steps[world] + 1
        )
    if current_status.iterations > max_iterations[world]:
        max_iterations[world] = current_status.iterations
    if active_contacts[world] > max_active_contacts[world]:
        max_active_contacts[world] = active_contacts[world]
    if current_status.r_p > max_primal_residual[world]:
        max_primal_residual[world] = current_status.r_p
    if current_status.r_d > max_dual_residual[world]:
        max_dual_residual[world] = current_status.r_d
    if current_status.r_c > max_complementarity_residual[world]:
        max_complementarity_residual[world] = current_status.r_c


@wp.kernel
def _record_trace_joint_q_kernel(
    fine_step: int,
    world_count: int,
    nq: int,
    joint_coord_world_start: wp.array(dtype=wp.int32),
    joint_q: wp.array(dtype=wp.float32),
    trace_joint_q: wp.array(dtype=wp.float32),
):
    world, local_coord = wp.tid()
    source = joint_coord_world_start[world] + local_coord
    target = (fine_step * world_count + world) * nq + local_coord
    trace_joint_q[target] = joint_q[source]


@wp.kernel
def _record_trace_joint_qd_kernel(
    fine_step: int,
    world_count: int,
    nv: int,
    joint_dof_world_start: wp.array(dtype=wp.int32),
    joint_qd: wp.array(dtype=wp.float32),
    trace_joint_qd: wp.array(dtype=wp.float32),
):
    world, local_dof = wp.tid()
    source = joint_dof_world_start[world] + local_dof
    target = (fine_step * world_count + world) * nv + local_dof
    trace_joint_qd[target] = joint_qd[source]


@wp.kernel
def _record_trace_status_kernel(
    fine_step: int,
    world_count: int,
    status: wp.array(dtype=PADMMStatus),
    active_contacts: wp.array(dtype=wp.int32),
    trace_converged: wp.array(dtype=wp.int32),
    trace_iterations: wp.array(dtype=wp.int32),
    trace_primal_residual: wp.array(dtype=wp.float32),
    trace_dual_residual: wp.array(dtype=wp.float32),
    trace_complementarity_residual: wp.array(dtype=wp.float32),
    trace_active_contacts: wp.array(dtype=wp.int32),
):
    world = wp.tid()
    target = fine_step * world_count + world
    current = status[world]
    trace_converged[target] = current.converged
    trace_iterations[target] = current.iterations
    trace_primal_residual[target] = current.r_p
    trace_dual_residual[target] = current.r_d
    trace_complementarity_residual[target] = current.r_c
    trace_active_contacts[target] = active_contacts[world]


@wp.kernel
def _record_trace_contacts_kernel(
    fine_step: int,
    world_count: int,
    max_contacts_per_world: int,
    model_active_contacts: wp.array(dtype=wp.int32),
    contact_wid: wp.array(dtype=wp.int32),
    contact_cid: wp.array(dtype=wp.int32),
    contact_gid_ab: wp.array(dtype=wp.vec2i),
    contact_bid_ab: wp.array(dtype=wp.vec2i),
    contact_key: wp.array(dtype=wp.uint64),
    contact_mode: wp.array(dtype=wp.int32),
    contact_gapfunc: wp.array(dtype=wp.vec4f),
    contact_reaction: wp.array(dtype=wp.vec3f),
    contact_velocity: wp.array(dtype=wp.vec3f),
    shape_world_start: wp.array(dtype=wp.int32),
    body_world_start: wp.array(dtype=wp.int32),
    trace_contact_active: wp.array(dtype=wp.int32),
    trace_contact_gid_ab: wp.array(dtype=wp.vec2i),
    trace_contact_bid_ab: wp.array(dtype=wp.vec2i),
    trace_contact_key: wp.array(dtype=wp.uint64),
    trace_contact_mode: wp.array(dtype=wp.int32),
    trace_contact_gap: wp.array(dtype=wp.float32),
    trace_contact_reaction: wp.array(dtype=wp.vec3f),
    trace_contact_velocity: wp.array(dtype=wp.vec3f),
):
    model_contact = wp.tid()
    if model_contact >= model_active_contacts[0]:
        return
    world = contact_wid[model_contact]
    local_contact = contact_cid[model_contact]
    if (
        world < 0
        or world >= world_count
        or local_contact < 0
        or local_contact >= max_contacts_per_world
    ):
        return
    target = (
        (fine_step * world_count + world) * max_contacts_per_world
        + local_contact
    )
    gids = contact_gid_ab[model_contact]
    bids = contact_bid_ab[model_contact]
    shape_start = shape_world_start[world]
    body_start = body_world_start[world]
    local_gid_a = gids[0]
    local_gid_b = gids[1]
    local_bid_a = bids[0]
    local_bid_b = bids[1]
    if local_gid_a >= 0:
        local_gid_a = local_gid_a - shape_start
    if local_gid_b >= 0:
        local_gid_b = local_gid_b - shape_start
    if local_bid_a >= 0:
        local_bid_a = local_bid_a - body_start
    if local_bid_b >= 0:
        local_bid_b = local_bid_b - body_start
    trace_contact_active[target] = 1
    trace_contact_gid_ab[target] = wp.vec2i(local_gid_a, local_gid_b)
    trace_contact_bid_ab[target] = wp.vec2i(local_bid_a, local_bid_b)
    trace_contact_key[target] = contact_key[model_contact]
    trace_contact_mode[target] = contact_mode[model_contact]
    trace_contact_gap[target] = contact_gapfunc[model_contact][3]
    trace_contact_reaction[target] = contact_reaction[model_contact]
    trace_contact_velocity[target] = contact_velocity[model_contact]


@wp.kernel
def _record_trace_cost_kernel(
    control_step: int,
    world_count: int,
    costs: wp.array(dtype=wp.float32),
    trace_control_costs: wp.array(dtype=wp.float32),
):
    world = wp.tid()
    trace_control_costs[control_step * world_count + world] = costs[world]


@wp.kernel
def _terminal_cost_and_extract_kernel(
    body_world_start: wp.array(dtype=wp.int32),
    object_body_local_index: int,
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    target_object_position: wp.array(dtype=wp.vec3f),
    terminal_position_weight: float,
    terminal_velocity_weight: float,
    costs: wp.array(dtype=wp.float32),
    final_object_q: wp.array(dtype=wp.transformf),
    final_object_qd: wp.array(dtype=wp.spatial_vectorf),
):
    world = wp.tid()
    object_body = body_world_start[world] + object_body_local_index
    pose = body_q[object_body]
    velocity = body_qd[object_body]
    position_error = wp.transform_get_translation(pose) - target_object_position[0]
    velocity_sq = float(0.0)
    for component in range(6):
        velocity_sq = velocity_sq + velocity[component] * velocity[component]
    costs[world] = costs[world] + (
        terminal_position_weight * wp.dot(position_error, position_error)
        + terminal_velocity_weight * velocity_sq
    )
    final_object_q[world] = pose
    final_object_qd[world] = velocity


@wp.kernel
def _extract_final_object_state_kernel(
    body_world_start: wp.array(dtype=wp.int32),
    object_body_local_index: int,
    body_q: wp.array(dtype=wp.transformf),
    body_qd: wp.array(dtype=wp.spatial_vectorf),
    final_object_q: wp.array(dtype=wp.transformf),
    final_object_qd: wp.array(dtype=wp.spatial_vectorf),
):
    world = wp.tid()
    object_body = body_world_start[world] + object_body_local_index
    final_object_q[world] = body_q[object_body]
    final_object_qd[world] = body_qd[object_body]


@dataclass(frozen=True)
class LeapBatchRolloutResult:
    """One summary download after a complete target-scene horizon."""

    costs: np.ndarray
    final_object_q: np.ndarray
    final_object_qd: np.ndarray
    final_qpos: np.ndarray
    final_qvel: np.ndarray
    final_status: np.ndarray
    converged_steps: np.ndarray
    contact_fine_steps: np.ndarray
    primal_failure_steps: np.ndarray
    dual_failure_steps: np.ndarray
    complementarity_failure_steps: np.ndarray
    max_iterations: np.ndarray
    max_active_contacts: np.ndarray
    max_primal_residual: np.ndarray
    max_dual_residual: np.ndarray
    max_complementarity_residual: np.ndarray
    reset_success: np.ndarray
    site_world: np.ndarray

    def physical_flattened(self) -> np.ndarray:
        return np.concatenate(
            (
                self.costs.reshape(-1),
                self.final_object_q.reshape(-1),
                self.final_object_qd.reshape(-1),
                self.final_qpos.reshape(-1),
                self.final_qvel.reshape(-1),
            )
        )

    def diagnostic_flattened(self) -> np.ndarray:
        return np.concatenate(
            (
                self.converged_steps.astype(np.float64),
                self.contact_fine_steps.astype(np.float64),
                self.primal_failure_steps.astype(np.float64),
                self.dual_failure_steps.astype(np.float64),
                self.complementarity_failure_steps.astype(np.float64),
                self.max_iterations.astype(np.float64),
                self.max_active_contacts.astype(np.float64),
                self.max_primal_residual.astype(np.float64),
                self.max_dual_residual.astype(np.float64),
                self.max_complementarity_residual.astype(np.float64),
            )
        )

    def flattened(self) -> np.ndarray:
        """Continuous physical outputs used for replay and finite checks."""

        return self.physical_flattened()


@dataclass(frozen=True)
class LeapRolloutProfile:
    """CUDA-event timing for one reset plus one complete horizon."""

    reset_gpu_ms: float
    action_dispatch_gpu_ms: float
    solve_gpu_ms: float
    observation_cost_gpu_ms: float
    total_gpu_ms: float
    compute_wall_ms: float
    summary_download_wall_ms: float
    end_to_end_wall_ms: float


@dataclass(frozen=True)
class LeapFineStepTrace:
    """Device-recorded trajectory downloaded only after the full horizon."""

    joint_q: np.ndarray
    joint_qd: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    primal_residual: np.ndarray
    dual_residual: np.ndarray
    complementarity_residual: np.ndarray
    active_contacts: np.ndarray
    contact_active: np.ndarray
    contact_gid_ab: np.ndarray
    contact_bid_ab: np.ndarray
    contact_key: np.ndarray
    contact_mode: np.ndarray
    contact_gap: np.ndarray
    contact_reaction: np.ndarray
    contact_velocity: np.ndarray
    control_costs: np.ndarray


class KaminoBatchedLeapRollout:
    """Replicate the real Leap scene and advance all candidates in one solver."""

    def __init__(
        self,
        scene: str | Path,
        *,
        world_count: int,
        horizon_steps: int,
        device: str = "cuda:0",
        dt: float = 0.002,
        physics_steps_per_control: int = 1,
        action_semantics: str = "absolute_joint_target",
        cost_semantics: str = "legacy_simple",
        contact_gap_m: float = 0.0,
        collision_detection: bool = True,
        padmm_tolerance: float = 1.0e-4,
        padmm_max_iterations: int = 200,
        padmm_rho0: float = 1.0,
        padmm_penalty_update_method: str = "fixed",
        dynamics_storage: str = "dense",
        sparse_linear_solver: str = "CRF",
        object_body_name: str = "obj",
        object_position_weight: float = 10.0,
        object_velocity_weight: float = 0.1,
        fingertip_distance_weight: float = 0.2,
        control_weight: float = 1.0e-3,
        terminal_position_weight: float = 50.0,
        terminal_velocity_weight: float = 1.0,
    ) -> None:
        if world_count < 1:
            raise ValueError("world_count must be positive")
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if physics_steps_per_control < 1:
            raise ValueError("physics_steps_per_control must be positive")
        if action_semantics not in (
            "absolute_joint_target",
            REFERENCE_ACTION_SEMANTICS,
        ):
            raise ValueError(f"Unsupported action_semantics {action_semantics!r}")
        if cost_semantics not in ("legacy_simple", REFERENCE_COST_SEMANTICS):
            raise ValueError(f"Unsupported cost_semantics {cost_semantics!r}")
        if contact_gap_m < 0.0 or not np.isfinite(contact_gap_m):
            raise ValueError("contact_gap_m must be finite and non-negative")
        if padmm_tolerance <= 0.0 or not np.isfinite(padmm_tolerance):
            raise ValueError("padmm_tolerance must be finite and positive")
        if padmm_max_iterations < 1:
            raise ValueError("padmm_max_iterations must be positive")
        if padmm_rho0 <= 0.0 or not np.isfinite(padmm_rho0):
            raise ValueError("padmm_rho0 must be finite and positive")
        if padmm_penalty_update_method not in ("fixed", "balanced"):
            raise ValueError(
                "padmm_penalty_update_method must be 'fixed' or 'balanced'"
            )
        if dynamics_storage not in ("dense", "sparse"):
            raise ValueError("dynamics_storage must be 'dense' or 'sparse'")
        if sparse_linear_solver not in ("CR", "CRF"):
            raise ValueError("sparse_linear_solver must be 'CR' or 'CRF'")

        self.scene = Path(scene).expanduser().resolve()
        if not self.scene.is_file():
            raise FileNotFoundError(f"MJCF scene does not exist: {self.scene}")
        self.world_count = int(world_count)
        self.horizon_steps = int(horizon_steps)
        self.dt = float(dt)
        self.physics_steps_per_control = int(physics_steps_per_control)
        self.action_semantics = action_semantics
        self.cost_semantics = cost_semantics
        self.contact_gap_m = float(contact_gap_m)
        self.collision_detection = bool(collision_detection)
        self.padmm_tolerance = float(padmm_tolerance)
        self.padmm_max_iterations = int(padmm_max_iterations)
        self.padmm_rho0 = float(padmm_rho0)
        self.padmm_penalty_update_method = padmm_penalty_update_method
        self.dynamics_storage = dynamics_storage
        self.sparse_linear_solver = sparse_linear_solver
        self.object_position_weight = float(object_position_weight)
        self.object_velocity_weight = float(object_velocity_weight)
        self.fingertip_distance_weight = float(fingertip_distance_weight)
        self.control_weight = float(control_weight)
        self.terminal_position_weight = float(terminal_position_weight)
        self.terminal_velocity_weight = float(terminal_velocity_weight)

        wp.init()
        self.path_resolver = ScenePathResolver(self.scene)
        with wp.ScopedDevice(device):
            self.device = wp.get_device(device)
            template = SiteRecordingModelBuilder(up_axis=newton.Axis.Z)
            template.rigid_gap = self.contact_gap_m
            newton.solvers.SolverKamino.register_custom_attributes(template)
            template.add_mjcf(
                str(self.scene),
                path_resolver=self.path_resolver,
            )
            if len(template.recorded_sites) != 5:
                raise RuntimeError(
                    f"Expected five Leap sites, got {len(template.recorded_sites)}"
                )

            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            builder.rigid_gap = self.contact_gap_m
            newton.solvers.SolverKamino.register_custom_attributes(builder)
            builder.replicate(template, world_count=self.world_count)
            self.model = builder.finalize()

            if self.model.world_count != self.world_count:
                raise RuntimeError("Final model world count does not match request")
            self.nq = self.model.joint_coord_count // self.world_count
            self.nv = self.model.joint_dof_count // self.world_count
            self.body_count_per_world = self.model.body_count // self.world_count
            self.shape_count_per_world = self.model.shape_count // self.world_count
            if (self.nq, self.nv, self.body_count_per_world) != (23, 22, 19):
                raise RuntimeError(
                    "Unexpected per-world Leap dimensions: "
                    f"nq={self.nq}, nv={self.nv}, bodies={self.body_count_per_world}"
                )

            all_target_indices = infer_position_target_indices(self.model)
            if len(all_target_indices) % self.world_count != 0:
                raise RuntimeError("Position-target count is not divisible by world count")
            self.nu = len(all_target_indices) // self.world_count
            if self.nu != 16:
                raise RuntimeError(f"Expected 16 actuators per world, got {self.nu}")
            target_index_matrix = np.asarray(
                all_target_indices, dtype=np.int32
            ).reshape(self.world_count, self.nu)
            self.control_target_indices = wp.array(
                target_index_matrix,
                dtype=wp.int32,
                device=self.device,
            )

            suffix = f"/{object_body_name}"
            object_matches = [
                index
                for index, label in enumerate(template.body_label)
                if label == object_body_name or label.endswith(suffix)
            ]
            if len(object_matches) != 1:
                raise RuntimeError(
                    f"Expected one template body named {object_body_name!r}, "
                    f"got {object_matches}"
                )
            self.object_body_local_index = object_matches[0]
            self.site_names = tuple(
                label.rsplit("/", 1)[-1]
                for label, _, _ in template.recorded_sites
            )
            local_site_shape_indices = np.asarray(
                [shape_index for _, shape_index, _ in template.recorded_sites],
                dtype=np.int32,
            )
            shape_starts = self.model.shape_world_start.numpy()[: self.world_count]
            site_index_matrix = (
                shape_starts[:, np.newaxis] + local_site_shape_indices[np.newaxis, :]
            ).astype(np.int32)
            self.site_shape_indices = wp.array(
                site_index_matrix,
                dtype=wp.int32,
                device=self.device,
            )
            self.site_count = len(self.site_names)
            required_fingertips = ("if_tip", "mf_tip", "rf_tip", "th_tip")
            missing_fingertips = sorted(
                set(required_fingertips) - set(self.site_names)
            )
            if missing_fingertips:
                raise RuntimeError(
                    f"Missing required fingertip sites: {missing_fingertips}"
                )
            fingertip_site_indices = np.asarray(
                [self.site_names.index(name) for name in required_fingertips],
                dtype=np.int32,
            )
            self.fingertip_site_local_indices = wp.array(
                fingertip_site_indices,
                dtype=wp.int32,
                device=self.device,
            )

            config = newton.solvers.SolverKamino.Config.from_model(self.model)
            config.use_fk_solver = True
            config.use_collision_detector = self.collision_detection
            config.collision_detector.default_gap = self.contact_gap_m
            config.sparse_jacobian = self.dynamics_storage == "sparse"
            config.sparse_dynamics = self.dynamics_storage == "sparse"
            if self.dynamics_storage == "sparse":
                config.dynamics.linear_solver_type = self.sparse_linear_solver
            config.padmm.primal_tolerance = self.padmm_tolerance
            config.padmm.dual_tolerance = self.padmm_tolerance
            config.padmm.compl_tolerance = self.padmm_tolerance
            config.padmm.max_iterations = self.padmm_max_iterations
            config.padmm.rho_0 = self.padmm_rho0
            config.padmm.penalty_update_method = (
                self.padmm_penalty_update_method
            )
            self.solver = newton.solvers.SolverKamino(self.model, config=config)
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()

            self.measured_joint_q = wp.array(
                self.model.joint_q.numpy()[: self.nq],
                dtype=wp.float32,
                device=self.device,
            )
            self.measured_joint_qd = wp.array(
                self.model.joint_qd.numpy()[: self.nv],
                dtype=wp.float32,
                device=self.device,
            )
            self.initial_joint_q = wp.empty(
                self.model.joint_coord_count,
                dtype=wp.float32,
                device=self.device,
            )
            self.initial_joint_qd = wp.empty(
                self.model.joint_dof_count,
                dtype=wp.float32,
                device=self.device,
            )
            from_q = newton.solvers.SolverKamino.ResetConfig.FromJointQ(
                self.initial_joint_q
            )
            from_qd = newton.solvers.SolverKamino.ResetConfig.FromJointU(
                self.initial_joint_qd
            )
            self.reset_config = newton.solvers.SolverKamino.ResetConfig(
                body_poses=from_q,
                body_velocities=from_qd,
                base_pose=from_q,
                base_velocity=from_qd,
            )
            self.reset_success = wp.zeros(
                self.world_count, dtype=wp.bool, device=self.device
            )
            self.action_sequences = wp.zeros(
                (self.horizon_steps, self.world_count, self.nu),
                dtype=wp.float32,
                device=self.device,
            )
            self.target_object_position = wp.zeros(
                1, dtype=wp.vec3f, device=self.device
            )
            self.grasp_reorient_goal = wp.zeros(
                24, dtype=wp.float32, device=self.device
            )
            self.grasp_reorient_weights = wp.zeros(
                12, dtype=wp.float32, device=self.device
            )
            self._grasp_reorient_cost_configured = False
            self.site_world = wp.empty(
                self.world_count * self.site_count,
                dtype=wp.transformf,
                device=self.device,
            )
            self.costs = wp.zeros(
                self.world_count, dtype=wp.float32, device=self.device
            )
            self.final_object_q = wp.empty(
                self.world_count, dtype=wp.transformf, device=self.device
            )
            self.final_object_qd = wp.empty(
                self.world_count, dtype=wp.spatial_vectorf, device=self.device
            )
            self.converged_steps = wp.zeros(
                self.world_count, dtype=wp.int32, device=self.device
            )
            self.contact_fine_steps = wp.zeros(
                self.world_count, dtype=wp.int32, device=self.device
            )
            self.primal_failure_steps = wp.zeros(
                self.world_count, dtype=wp.int32, device=self.device
            )
            self.dual_failure_steps = wp.zeros(
                self.world_count, dtype=wp.int32, device=self.device
            )
            self.complementarity_failure_steps = wp.zeros(
                self.world_count, dtype=wp.int32, device=self.device
            )
            self.max_iterations = wp.zeros(
                self.world_count, dtype=wp.int32, device=self.device
            )
            self.max_active_contacts = wp.zeros(
                self.world_count, dtype=wp.int32, device=self.device
            )
            self.max_primal_residual = wp.zeros(
                self.world_count, dtype=wp.float32, device=self.device
            )
            self.max_dual_residual = wp.zeros(
                self.world_count, dtype=wp.float32, device=self.device
            )
            self.max_complementarity_residual = wp.zeros(
                self.world_count, dtype=wp.float32, device=self.device
            )

            all_free_starts = _free_joint_q_starts(self.model)
            self.free_joint_q_starts_local = tuple(
                start for start in all_free_starts if start < self.nq
            )
            self.reset_device()
            wp.synchronize_device(self.device)

    def set_initial_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Upload one measured MuJoCo-order state for later GPU broadcast."""

        qpos_array = np.asarray(qpos, dtype=np.float64)
        qvel_array = np.asarray(qvel, dtype=np.float64)
        if qpos_array.shape != (self.nq,):
            raise ValueError(f"qpos must have shape {(self.nq,)}, got {qpos_array.shape}")
        if qvel_array.shape != (self.nv,):
            raise ValueError(f"qvel must have shape {(self.nv,)}, got {qvel_array.shape}")
        if not np.isfinite(qpos_array).all() or not np.isfinite(qvel_array).all():
            raise ValueError("Initial state contains NaN or Inf")
        self.measured_joint_q.assign(
            mujoco_qpos_to_newton(qpos_array, self.free_joint_q_starts_local)
        )
        self.measured_joint_qd.assign(qvel_array.astype(np.float32))

    def set_target_object_position(self, position: np.ndarray) -> None:
        """Upload one world-frame object-position target shared by all worlds."""

        position_array = np.asarray(position, dtype=np.float32)
        if position_array.shape != (3,) or not np.isfinite(position_array).all():
            raise ValueError("Target object position must be a finite 3-vector")
        self.target_object_position.assign(position_array.reshape(1, 3))

    def set_grasp_reorient_cost(
        self,
        goal: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Upload the main task's exact goal layout and twelve cost weights."""

        goal_array = np.asarray(goal, dtype=np.float32)
        weights_array = np.asarray(weights, dtype=np.float32)
        if goal_array.shape != (24,) or not np.isfinite(goal_array).all():
            raise ValueError("grasp-reorient goal must be a finite 24-vector")
        if weights_array.shape != (12,) or not np.isfinite(weights_array).all():
            raise ValueError("grasp-reorient weights must be a finite 12-vector")
        self.grasp_reorient_goal.assign(goal_array)
        self.grasp_reorient_weights.assign(weights_array)
        self._grasp_reorient_cost_configured = True

    def configure_reference_bundle(
        self,
        bundle: ReferencePlanningBundle,
    ) -> None:
        """Validate and upload one exact first-iteration reference problem.

        The portable candidate layout is ``(sample, control, actuator)``;
        Kamino's rollout storage is ``(control, world, actuator)``.  The only
        data transformation here is that explicit transpose.
        """

        bundle.validate()
        spec = bundle.spec
        mismatches: list[str] = []
        checks = (
            ("scene_sha256", file_sha256(self.scene), spec.scene_sha256),
            ("nq", self.nq, spec.nq),
            ("nv", self.nv, spec.nv),
            ("nu", self.nu, spec.nu),
            ("n_samples/world_count", self.world_count, spec.n_samples),
            ("control_horizon", self.horizon_steps, spec.control_horizon),
            (
                "physics_steps_per_control",
                self.physics_steps_per_control,
                spec.physics_steps_per_control,
            ),
            ("action_semantics", self.action_semantics, spec.action_semantics),
            ("cost_semantics", self.cost_semantics, spec.cost_semantics),
        )
        for name, actual, expected in checks:
            if actual != expected:
                mismatches.append(f"{name}: rollout={actual!r}, bundle={expected!r}")
        if not np.isclose(self.dt, spec.physics_dt_s, rtol=0.0, atol=1.0e-12):
            mismatches.append(
                f"physics_dt_s: rollout={self.dt!r}, bundle={spec.physics_dt_s!r}"
            )
        if mismatches:
            raise ValueError(
                "Reference bundle is incompatible with this rollout:\n- "
                + "\n- ".join(mismatches)
            )

        self.set_initial_state(bundle.qpos, bundle.qvel)
        self.set_grasp_reorient_cost(bundle.goal, bundle.cost_weights)
        candidates = bundle.first_iteration_candidates()
        self.upload_action_sequences(np.transpose(candidates, (1, 0, 2)))

    def upload_action_sequences(self, action_sequences: np.ndarray) -> None:
        """Make one host-to-device action-sequence copy before rollout."""

        action_array = np.asarray(action_sequences, dtype=np.float32)
        expected = (self.horizon_steps, self.world_count, self.nu)
        if action_array.shape != expected:
            raise ValueError(
                f"action_sequences must have shape {expected}, got {action_array.shape}"
            )
        if not np.isfinite(action_array).all():
            raise ValueError("action_sequences contains NaN or Inf")
        self.action_sequences.assign(action_array)

    def reset_device(self) -> None:
        """Broadcast one state, run FK reset, and clear rollout accumulators."""

        self.control.clear(self.model)
        wp.launch(
            _broadcast_joint_q_kernel,
            dim=(self.world_count, self.nq),
            inputs=[
                self.measured_joint_q,
                self.model.joint_coord_world_start,
                self.initial_joint_q,
            ],
            device=self.device,
        )
        wp.launch(
            _broadcast_joint_qd_kernel,
            dim=(self.world_count, self.nv),
            inputs=[
                self.measured_joint_qd,
                self.model.joint_dof_world_start,
                self.initial_joint_qd,
            ],
            device=self.device,
        )
        self.reset_success.zero_()
        self.solver.reset(
            self.state_0,
            config=self.reset_config,
            success_mask=self.reset_success,
        )
        self.costs.zero_()
        self.converged_steps.zero_()
        self.contact_fine_steps.zero_()
        self.primal_failure_steps.zero_()
        self.dual_failure_steps.zero_()
        self.complementarity_failure_steps.zero_()
        self.max_iterations.zero_()
        self.max_active_contacts.zero_()
        self.max_primal_residual.zero_()
        self.max_dual_residual.zero_()
        self.max_complementarity_residual.zero_()

    def _dispatch_actions_device(self, step_index: int) -> None:
        if self.action_semantics == REFERENCE_ACTION_SEMANTICS:
            wp.launch(
                _write_relative_position_targets_kernel,
                dim=(self.world_count, self.nu),
                inputs=[
                    step_index,
                    self.action_sequences,
                    self.model.joint_coord_world_start,
                    0,
                    self.state_0.joint_q,
                    self.control_target_indices,
                    self.control.joint_target_q,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _write_position_targets_kernel,
                dim=(self.world_count, self.nu),
                inputs=[
                    step_index,
                    self.action_sequences,
                    self.control_target_indices,
                    self.control.joint_target_q,
                ],
                device=self.device,
            )

    def _solve_step_device(self) -> None:
        self.state_0.clear_forces()
        self.solver.step(
            self.state_0,
            self.state_1,
            self.control,
            None,
            self.dt,
        )
        self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(
            _accumulate_solver_diagnostics_kernel,
            dim=self.world_count,
            inputs=[
                self.solver.status,
                self.solver._contacts_kamino.world_active_contacts,
                self.padmm_tolerance,
                self.padmm_tolerance,
                self.padmm_tolerance,
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

    def _observation_cost_step_device(
        self,
        step_index: int,
        *,
        terminal: bool = False,
    ) -> None:
        wp.launch(
            _compute_batched_site_transforms_kernel,
            dim=(self.world_count, self.site_count),
            inputs=[
                self.site_shape_indices,
                self.model.shape_body,
                self.model.shape_transform,
                self.state_0.body_q,
                self.site_world,
                self.site_count,
            ],
            device=self.device,
        )
        if self.cost_semantics == REFERENCE_COST_SEMANTICS:
            if not self._grasp_reorient_cost_configured:
                raise RuntimeError(
                    "Call set_grasp_reorient_cost() before an exact-cost rollout"
                )
            wp.launch(
                _accumulate_grasp_reorient_cost_kernel,
                dim=self.world_count,
                inputs=[
                    terminal,
                    self.model.joint_coord_world_start,
                    self.model.joint_dof_world_start,
                    self.state_0.joint_q,
                    self.state_0.joint_qd,
                    self.site_world,
                    self.fingertip_site_local_indices,
                    self.site_count,
                    self.grasp_reorient_goal,
                    self.grasp_reorient_weights,
                    self.costs,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _accumulate_target_stage_cost_kernel,
                dim=self.world_count,
                inputs=[
                    step_index,
                    self.action_sequences,
                    self.model.body_world_start,
                    self.object_body_local_index,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.site_world,
                    self.target_object_position,
                    self.object_position_weight,
                    self.object_velocity_weight,
                    self.fingertip_distance_weight,
                    self.control_weight,
                    self.dt,
                    self.costs,
                ],
                device=self.device,
            )

    def _terminal_device(self) -> None:
        if self.cost_semantics == REFERENCE_COST_SEMANTICS:
            wp.launch(
                _extract_final_object_state_kernel,
                dim=self.world_count,
                inputs=[
                    self.model.body_world_start,
                    self.object_body_local_index,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.final_object_q,
                    self.final_object_qd,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _terminal_cost_and_extract_kernel,
                dim=self.world_count,
                inputs=[
                    self.model.body_world_start,
                    self.object_body_local_index,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.target_object_position,
                    self.terminal_position_weight,
                    self.terminal_velocity_weight,
                    self.costs,
                    self.final_object_q,
                    self.final_object_qd,
                ],
                device=self.device,
            )

    def rollout_device(self) -> tuple[wp.array, wp.array, wp.array]:
        """Advance the complete horizon without a host-side state download."""

        for step_index in range(self.horizon_steps):
            self._dispatch_actions_device(step_index)
            for _ in range(self.physics_steps_per_control):
                self._solve_step_device()
            self._observation_cost_step_device(
                step_index,
                terminal=step_index == self.horizon_steps - 1,
            )
        self._terminal_device()
        return self.costs, self.final_object_q, self.final_object_qd

    def download_result(self) -> LeapBatchRolloutResult:
        """Download the post-horizon summary and five site poses per world."""

        final_newton_q = self.state_0.joint_q.numpy().reshape(
            self.world_count, self.nq
        )
        final_qpos = np.stack(
            [
                newton_qpos_to_mujoco(row, self.free_joint_q_starts_local)
                for row in final_newton_q
            ]
        )

        return LeapBatchRolloutResult(
            costs=self.costs.numpy().copy(),
            final_object_q=self.final_object_q.numpy().copy(),
            final_object_qd=self.final_object_qd.numpy().copy(),
            final_qpos=final_qpos,
            final_qvel=self.state_0.joint_qd.numpy()
            .copy()
            .reshape(self.world_count, self.nv),
            final_status=self.solver.status.numpy().copy(),
            converged_steps=self.converged_steps.numpy().copy(),
            contact_fine_steps=self.contact_fine_steps.numpy().copy(),
            primal_failure_steps=self.primal_failure_steps.numpy().copy(),
            dual_failure_steps=self.dual_failure_steps.numpy().copy(),
            complementarity_failure_steps=(
                self.complementarity_failure_steps.numpy().copy()
            ),
            max_iterations=self.max_iterations.numpy().copy(),
            max_active_contacts=self.max_active_contacts.numpy().copy(),
            max_primal_residual=self.max_primal_residual.numpy().copy(),
            max_dual_residual=self.max_dual_residual.numpy().copy(),
            max_complementarity_residual=(
                self.max_complementarity_residual.numpy().copy()
            ),
            reset_success=self.reset_success.numpy().copy(),
            site_world=self.site_world.numpy().copy().reshape(
                self.world_count, self.site_count, 7
            ),
        )

    def profile_rollout(self) -> tuple[LeapRolloutProfile, LeapBatchRolloutResult]:
        """Time reset, action, solve, and observation/cost with CUDA events."""

        if not self.device.is_cuda:
            raise RuntimeError("CUDA event profiling requires a CUDA device")
        event_count = 3 + self.horizon_steps * (
            self.physics_steps_per_control + 2
        )
        events = [
            wp.Event(self.device, enable_timing=True) for _ in range(event_count)
        ]
        stream = wp.get_stream(self.device)
        wp.synchronize_device(self.device)
        wall_start = time.perf_counter()
        stream.record_event(events[0])
        self.reset_device()
        stream.record_event(events[1])

        cursor = 1
        action_pairs: list[tuple[int, int]] = []
        solve_pairs: list[tuple[int, int]] = []
        observation_pairs: list[tuple[int, int]] = []
        for step_index in range(self.horizon_steps):
            self._dispatch_actions_device(step_index)
            stream.record_event(events[cursor + 1])
            action_pairs.append((cursor, cursor + 1))
            cursor += 1
            for _ in range(self.physics_steps_per_control):
                self._solve_step_device()
                stream.record_event(events[cursor + 1])
                solve_pairs.append((cursor, cursor + 1))
                cursor += 1
            self._observation_cost_step_device(
                step_index,
                terminal=step_index == self.horizon_steps - 1,
            )
            stream.record_event(events[cursor + 1])
            observation_pairs.append((cursor, cursor + 1))
            cursor += 1

        self._terminal_device()
        stream.record_event(events[cursor + 1])
        terminal_pair = (cursor, cursor + 1)
        cursor += 1
        wp.synchronize_event(events[cursor])
        compute_wall_ms = 1.0e3 * (time.perf_counter() - wall_start)

        def elapsed(pair: tuple[int, int]) -> float:
            return float(
                wp.get_event_elapsed_time(
                    events[pair[0]], events[pair[1]], synchronize=False
                )
            )

        action_gpu_ms = sum(elapsed(pair) for pair in action_pairs)
        solve_gpu_ms = sum(elapsed(pair) for pair in solve_pairs)
        observation_gpu_ms = sum(elapsed(pair) for pair in observation_pairs)
        observation_gpu_ms += elapsed(terminal_pair)
        reset_gpu_ms = elapsed((0, 1))
        total_gpu_ms = float(
            wp.get_event_elapsed_time(events[0], events[cursor], synchronize=False)
        )

        download_start = time.perf_counter()
        result = self.download_result()
        download_wall_ms = 1.0e3 * (time.perf_counter() - download_start)
        profile = LeapRolloutProfile(
            reset_gpu_ms=reset_gpu_ms,
            action_dispatch_gpu_ms=action_gpu_ms,
            solve_gpu_ms=solve_gpu_ms,
            observation_cost_gpu_ms=observation_gpu_ms,
            total_gpu_ms=total_gpu_ms,
            compute_wall_ms=compute_wall_ms,
            summary_download_wall_ms=download_wall_ms,
            end_to_end_wall_ms=compute_wall_ms + download_wall_ms,
        )
        return profile, result

    def trace_rollout(self) -> tuple[LeapBatchRolloutResult, LeapFineStepTrace]:
        """Record every fine step on device, then perform one final download.

        The additional recording kernels remain on the same CUDA stream and do
        not synchronize with the host inside the horizon.  This is a diagnostic
        execution path, not a throughput benchmark.
        """

        fine_step_count = self.horizon_steps * self.physics_steps_per_control
        contacts = self.solver._contacts_kamino
        allocated_contacts_per_world = max(contacts.world_max_contacts_host)
        max_contacts_per_world = min(allocated_contacts_per_world, 32)
        model_max_contacts = contacts.model_max_contacts_host
        if max_contacts_per_world < 1 or model_max_contacts < 1:
            raise RuntimeError("Fine-step tracing requires allocated contacts")

        state_size = fine_step_count * self.world_count
        trace_joint_q = wp.empty(
            state_size * self.nq, dtype=wp.float32, device=self.device
        )
        trace_joint_qd = wp.empty(
            state_size * self.nv, dtype=wp.float32, device=self.device
        )
        trace_converged = wp.empty(
            state_size, dtype=wp.int32, device=self.device
        )
        trace_iterations = wp.empty(
            state_size, dtype=wp.int32, device=self.device
        )
        trace_primal_residual = wp.empty(
            state_size, dtype=wp.float32, device=self.device
        )
        trace_dual_residual = wp.empty(
            state_size, dtype=wp.float32, device=self.device
        )
        trace_complementarity_residual = wp.empty(
            state_size, dtype=wp.float32, device=self.device
        )
        trace_active_contacts = wp.empty(
            state_size, dtype=wp.int32, device=self.device
        )
        contact_size = state_size * max_contacts_per_world
        trace_contact_active = wp.zeros(
            contact_size, dtype=wp.int32, device=self.device
        )
        trace_contact_gid_ab = wp.full(
            contact_size, value=wp.vec2i(-1, -1), dtype=wp.vec2i,
            device=self.device,
        )
        trace_contact_bid_ab = wp.full(
            contact_size, value=wp.vec2i(-1, -1), dtype=wp.vec2i,
            device=self.device,
        )
        trace_contact_key = wp.zeros(
            contact_size, dtype=wp.uint64, device=self.device
        )
        trace_contact_mode = wp.full(
            contact_size, value=-1, dtype=wp.int32, device=self.device
        )
        trace_contact_gap = wp.zeros(
            contact_size, dtype=wp.float32, device=self.device
        )
        trace_contact_reaction = wp.zeros(
            contact_size, dtype=wp.vec3f, device=self.device
        )
        trace_contact_velocity = wp.zeros(
            contact_size, dtype=wp.vec3f, device=self.device
        )
        trace_control_costs = wp.empty(
            self.horizon_steps * self.world_count,
            dtype=wp.float32,
            device=self.device,
        )

        self.reset_device()
        fine_step = 0
        for control_step in range(self.horizon_steps):
            self._dispatch_actions_device(control_step)
            for _ in range(self.physics_steps_per_control):
                self._solve_step_device()
                wp.launch(
                    _record_trace_joint_q_kernel,
                    dim=(self.world_count, self.nq),
                    inputs=[
                        fine_step,
                        self.world_count,
                        self.nq,
                        self.model.joint_coord_world_start,
                        self.state_0.joint_q,
                        trace_joint_q,
                    ],
                    device=self.device,
                )
                wp.launch(
                    _record_trace_joint_qd_kernel,
                    dim=(self.world_count, self.nv),
                    inputs=[
                        fine_step,
                        self.world_count,
                        self.nv,
                        self.model.joint_dof_world_start,
                        self.state_0.joint_qd,
                        trace_joint_qd,
                    ],
                    device=self.device,
                )
                wp.launch(
                    _record_trace_status_kernel,
                    dim=self.world_count,
                    inputs=[
                        fine_step,
                        self.world_count,
                        self.solver.status,
                        contacts.world_active_contacts,
                        trace_converged,
                        trace_iterations,
                        trace_primal_residual,
                        trace_dual_residual,
                        trace_complementarity_residual,
                        trace_active_contacts,
                    ],
                    device=self.device,
                )
                wp.launch(
                    _record_trace_contacts_kernel,
                    dim=model_max_contacts,
                    inputs=[
                        fine_step,
                        self.world_count,
                        max_contacts_per_world,
                        contacts.model_active_contacts,
                        contacts.wid,
                        contacts.cid,
                        contacts.gid_AB,
                        contacts.bid_AB,
                        contacts.key,
                        contacts.mode,
                        contacts.gapfunc,
                        contacts.reaction,
                        contacts.velocity,
                        self.model.shape_world_start,
                        self.model.body_world_start,
                        trace_contact_active,
                        trace_contact_gid_ab,
                        trace_contact_bid_ab,
                        trace_contact_key,
                        trace_contact_mode,
                        trace_contact_gap,
                        trace_contact_reaction,
                        trace_contact_velocity,
                    ],
                    device=self.device,
                )
                fine_step += 1
            self._observation_cost_step_device(
                control_step,
                terminal=control_step == self.horizon_steps - 1,
            )
            wp.launch(
                _record_trace_cost_kernel,
                dim=self.world_count,
                inputs=[
                    control_step,
                    self.world_count,
                    self.costs,
                    trace_control_costs,
                ],
                device=self.device,
            )
        self._terminal_device()
        wp.synchronize_device(self.device)
        result = self.download_result()
        if int(np.max(result.max_active_contacts)) > max_contacts_per_world:
            raise RuntimeError(
                "Fine-step contact trace overflowed its compact diagnostic "
                f"capacity of {max_contacts_per_world} contacts per world"
            )
        contact_shape = (
            fine_step_count,
            self.world_count,
            max_contacts_per_world,
        )
        trace = LeapFineStepTrace(
            joint_q=trace_joint_q.numpy().reshape(
                fine_step_count, self.world_count, self.nq
            ),
            joint_qd=trace_joint_qd.numpy().reshape(
                fine_step_count, self.world_count, self.nv
            ),
            converged=trace_converged.numpy().reshape(
                fine_step_count, self.world_count
            ),
            iterations=trace_iterations.numpy().reshape(
                fine_step_count, self.world_count
            ),
            primal_residual=trace_primal_residual.numpy().reshape(
                fine_step_count, self.world_count
            ),
            dual_residual=trace_dual_residual.numpy().reshape(
                fine_step_count, self.world_count
            ),
            complementarity_residual=(
                trace_complementarity_residual.numpy().reshape(
                    fine_step_count, self.world_count
                )
            ),
            active_contacts=trace_active_contacts.numpy().reshape(
                fine_step_count, self.world_count
            ),
            contact_active=trace_contact_active.numpy().reshape(contact_shape),
            contact_gid_ab=trace_contact_gid_ab.numpy().reshape(
                *contact_shape, 2
            ),
            contact_bid_ab=trace_contact_bid_ab.numpy().reshape(
                *contact_shape, 2
            ),
            contact_key=trace_contact_key.numpy().reshape(contact_shape),
            contact_mode=trace_contact_mode.numpy().reshape(contact_shape),
            contact_gap=trace_contact_gap.numpy().reshape(contact_shape),
            contact_reaction=trace_contact_reaction.numpy().reshape(
                *contact_shape, 3
            ),
            contact_velocity=trace_contact_velocity.numpy().reshape(
                *contact_shape, 3
            ),
            control_costs=trace_control_costs.numpy().reshape(
                self.horizon_steps, self.world_count
            ),
        )
        return result, trace
