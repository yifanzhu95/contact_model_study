"""One-world state, control, and site-observation adapter for SolverKamino.

The public boundary uses MuJoCo's generalized-coordinate convention because
that is the convention used by ``contact_model_study``.  Internally the adapter
converts free-joint quaternions to Newton's convention and asks Kamino's FK
reset path to construct a consistent maximal-coordinate state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import newton
import warp as wp

from .mjcf_probe import ScenePathResolver, infer_position_target_indices


@dataclass(frozen=True)
class SiteRecord:
    """Import-time mapping from an MJCF site to its Newton shape."""

    name: str
    label: str
    shape_index: int
    body_index: int


@dataclass(frozen=True)
class EngineObservation:
    """Engine-neutral host observation in MuJoCo coordinate order."""

    qpos: np.ndarray
    qvel: np.ndarray
    control: np.ndarray
    site_names: tuple[str, ...]
    site_xpos: np.ndarray
    site_xmat: np.ndarray


@dataclass(frozen=True)
class FineStepDiagnostic:
    """Host-side snapshot from the deliberately synchronized debug path."""

    fine_step: int
    state_finite: bool
    converged: int
    iterations: int
    r_p: float
    r_d: float
    r_c: float
    candidate_points: int
    candidate_pairs: int

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "fine_step": self.fine_step,
            "state_finite": self.state_finite,
            "converged": self.converged,
            "iterations": self.iterations,
            "r_p": self.r_p,
            "r_d": self.r_d,
            "r_c": self.r_c,
            "candidate_points": self.candidate_points,
            "candidate_pairs": self.candidate_pairs,
        }


class SiteRecordingModelBuilder(newton.ModelBuilder):
    """Record the shape index returned by every MJCF ``site`` import."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.recorded_sites: list[tuple[str, int, int]] = []

    def add_site(self, body: int, **kwargs: Any) -> int:
        shape_index = super().add_site(body, **kwargs)
        label = str(kwargs.get("label") or f"site_{shape_index}")
        self.recorded_sites.append((label, shape_index, body))
        return shape_index


@wp.kernel
def _compute_site_world_transforms(
    site_shape_indices: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transformf),
    body_q: wp.array(dtype=wp.transformf),
    site_world: wp.array(dtype=wp.transformf),
):
    site_index = wp.tid()
    shape_index = site_shape_indices[site_index]
    body_index = shape_body[shape_index]
    local_transform = shape_transform[shape_index]
    if body_index >= 0:
        site_world[site_index] = wp.transform_multiply(
            body_q[body_index], local_transform
        )
    else:
        site_world[site_index] = local_transform


def _joint_spans(model: Any) -> list[tuple[int, int, int, int]]:
    """Return ``(q_begin, q_end, qd_begin, qd_end)`` per Newton joint."""

    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    spans: list[tuple[int, int, int, int]] = []
    for joint_index in range(model.joint_count):
        q_begin = int(q_start[joint_index])
        q_end = (
            int(q_start[joint_index + 1])
            if joint_index + 1 < len(q_start)
            else int(model.joint_coord_count)
        )
        qd_begin = int(qd_start[joint_index])
        qd_end = (
            int(qd_start[joint_index + 1])
            if joint_index + 1 < len(qd_start)
            else int(model.joint_dof_count)
        )
        spans.append((q_begin, q_end, qd_begin, qd_end))
    return spans


def _free_joint_q_starts(model: Any) -> tuple[int, ...]:
    """Identify 7-coordinate/6-DOF free joints without relying on enum values."""

    return tuple(
        q_begin
        for q_begin, q_end, qd_begin, qd_end in _joint_spans(model)
        if q_end - q_begin == 7 and qd_end - qd_begin == 6
    )


def mujoco_qpos_to_newton(
    qpos: np.ndarray, free_joint_q_starts: tuple[int, ...]
) -> np.ndarray:
    """Convert free quaternions from MuJoCo ``wxyz`` to Newton ``xyzw``."""

    result = np.asarray(qpos, dtype=np.float32).copy()
    if result.ndim != 1:
        raise ValueError("qpos must be a one-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError("qpos contains NaN or Inf")
    for start in free_joint_q_starts:
        w, x, y, z = result[start + 3 : start + 7].copy()
        result[start + 3 : start + 7] = (x, y, z, w)
    return result


def newton_qpos_to_mujoco(
    joint_q: np.ndarray, free_joint_q_starts: tuple[int, ...]
) -> np.ndarray:
    """Convert free quaternions from Newton ``xyzw`` to MuJoCo ``wxyz``."""

    result = np.asarray(joint_q, dtype=np.float64).copy()
    if result.ndim != 1:
        raise ValueError("joint_q must be a one-dimensional array")
    for start in free_joint_q_starts:
        x, y, z, w = result[start + 3 : start + 7].copy()
        result[start + 3 : start + 7] = (w, x, y, z)
    return result


def _quat_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm((x, y, z, w))
    if norm == 0.0:
        raise ValueError("Cannot convert a zero quaternion")
    x, y, z, w = np.asarray((x, y, z, w)) / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class KaminoOneWorldAdapter:
    """Minimal one-world adapter for the target MJCF integration spike."""

    def __init__(
        self,
        scene: str | Path,
        *,
        device: str = "cuda:0",
        dt: float = 0.002,
        enable_contacts: bool = True,
        contact_gap_m: float = 0.0,
        dynamics_storage: str = "dense",
        sparse_linear_solver: str = "CRF",
        padmm_tolerance: float | None = None,
        padmm_max_iterations: int | None = None,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if not np.isfinite(contact_gap_m) or contact_gap_m < 0.0:
            raise ValueError("contact_gap_m must be finite and non-negative")
        if dynamics_storage not in ("dense", "sparse"):
            raise ValueError("dynamics_storage must be 'dense' or 'sparse'")
        if sparse_linear_solver not in ("CR", "CRF"):
            raise ValueError("sparse_linear_solver must be 'CR' or 'CRF'")
        if padmm_tolerance is not None and (
            not np.isfinite(padmm_tolerance) or padmm_tolerance <= 0.0
        ):
            raise ValueError("padmm_tolerance must be finite and positive")
        if padmm_max_iterations is not None and padmm_max_iterations < 1:
            raise ValueError("padmm_max_iterations must be positive")
        self.scene = Path(scene).expanduser().resolve()
        if not self.scene.is_file():
            raise FileNotFoundError(f"MJCF scene does not exist: {self.scene}")
        self.device = device
        self.dt = float(dt)
        self.enable_contacts = bool(enable_contacts)
        self.contact_gap_m = float(contact_gap_m)
        self.dynamics_storage = dynamics_storage
        self.sparse_linear_solver = sparse_linear_solver
        self.padmm_tolerance = (
            None if padmm_tolerance is None else float(padmm_tolerance)
        )
        self.padmm_max_iterations = (
            None if padmm_max_iterations is None else int(padmm_max_iterations)
        )

        wp.init()
        self.path_resolver = ScenePathResolver(self.scene)
        with wp.ScopedDevice(device):
            builder = SiteRecordingModelBuilder(up_axis=newton.Axis.Z)
            # Newton's builder default is 0.1 m, whereas an MJCF geom without
            # an explicit gap uses MuJoCo's zero-gap default.  Set the fallback
            # before import; explicit per-geom MJCF gaps still take precedence.
            builder.rigid_gap = self.contact_gap_m
            newton.solvers.SolverKamino.register_custom_attributes(builder)
            builder.request_contact_attributes("force")
            builder.add_mjcf(str(self.scene), path_resolver=self.path_resolver)
            self.model = builder.finalize()
            if self.model.world_count != 1:
                raise RuntimeError(
                    f"KaminoOneWorldAdapter requires one world, got {self.model.world_count}"
                )

            config = newton.solvers.SolverKamino.Config.from_model(self.model)
            config.use_fk_solver = True
            config.use_collision_detector = self.enable_contacts
            config.collision_detector.default_gap = self.contact_gap_m
            config.sparse_jacobian = self.dynamics_storage == "sparse"
            config.sparse_dynamics = self.dynamics_storage == "sparse"
            if self.dynamics_storage == "sparse":
                config.dynamics.linear_solver_type = self.sparse_linear_solver
            if self.padmm_tolerance is not None:
                config.padmm.primal_tolerance = self.padmm_tolerance
                config.padmm.dual_tolerance = self.padmm_tolerance
                config.padmm.compl_tolerance = self.padmm_tolerance
            if self.padmm_max_iterations is not None:
                config.padmm.max_iterations = self.padmm_max_iterations
            self.solver = newton.solvers.SolverKamino(self.model, config=config)
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.control = self.model.control()
            # This public Contacts container is diagnostic output only. Kamino's
            # configured unified pipeline still performs collision detection.
            self.contact_export_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.contact_export_pipeline.contacts()

            self.free_joint_q_starts = _free_joint_q_starts(self.model)
            self.control_target_indices = infer_position_target_indices(self.model)
            actuator_count = len(self.model.mujoco.actuator_label)
            if len(self.control_target_indices) != actuator_count:
                raise RuntimeError(
                    "Expected one scalar position target per MuJoCo actuator: "
                    f"{len(self.control_target_indices)} targets, {actuator_count} actuators"
                )

            self.site_records = tuple(
                SiteRecord(
                    name=label.rsplit("/", 1)[-1],
                    label=label,
                    shape_index=shape_index,
                    body_index=body_index,
                )
                for label, shape_index, body_index in builder.recorded_sites
            )
            self.site_shape_indices = wp.array(
                [record.shape_index for record in self.site_records],
                dtype=wp.int32,
                device=device,
            )
            self.site_world = wp.empty(
                len(self.site_records), dtype=wp.transformf, device=device
            )

            self.measured_joint_q = wp.clone(self.model.joint_q)
            self.measured_joint_qd = wp.clone(self.model.joint_qd)
            from_q = newton.solvers.SolverKamino.ResetConfig.FromJointQ(
                self.measured_joint_q
            )
            from_qd = newton.solvers.SolverKamino.ResetConfig.FromJointU(
                self.measured_joint_qd
            )
            self.reset_config = newton.solvers.SolverKamino.ResetConfig(
                body_poses=from_q,
                body_velocities=from_qd,
                base_pose=from_q,
                base_velocity=from_qd,
            )
            self.reset_success = wp.zeros(1, dtype=wp.bool, device=device)
            self._default_targets = self.model.joint_target_q.numpy().copy()
            self._last_action = np.zeros(actuator_count, dtype=np.float64)

        self.set_state(
            newton_qpos_to_mujoco(
                self.model.joint_q.numpy(), self.free_joint_q_starts
            ),
            self.model.joint_qd.numpy(),
        )

    @property
    def nq(self) -> int:
        return int(self.model.joint_coord_count)

    @property
    def nv(self) -> int:
        return int(self.model.joint_dof_count)

    @property
    def nu(self) -> int:
        return len(self.control_target_indices)

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Reset Kamino from a MuJoCo-order measured generalized state."""

        qpos_array = np.asarray(qpos, dtype=np.float64)
        qvel_array = np.asarray(qvel, dtype=np.float64)
        if qpos_array.shape != (self.nq,):
            raise ValueError(f"qpos must have shape {(self.nq,)}, got {qpos_array.shape}")
        if qvel_array.shape != (self.nv,):
            raise ValueError(f"qvel must have shape {(self.nv,)}, got {qvel_array.shape}")
        if not np.isfinite(qvel_array).all():
            raise ValueError("qvel contains NaN or Inf")

        self.measured_joint_q.assign(
            mujoco_qpos_to_newton(qpos_array, self.free_joint_q_starts)
        )
        self.measured_joint_qd.assign(qvel_array.astype(np.float32))
        self.control.clear(self.model)
        self.contacts.clear()
        self.reset_success.zero_()
        self.solver.reset(
            self.state_0,
            config=self.reset_config,
            success_mask=self.reset_success,
        )
        if not bool(self.reset_success.numpy()[0]):
            raise RuntimeError("Kamino FK state reset did not converge")
        self._last_action.fill(0.0)
        self.update_sites_device()

    def set_control(self, action: np.ndarray) -> None:
        """Map a MuJoCo-style actuator vector to Newton position targets."""

        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (self.nu,):
            raise ValueError(f"action must have shape {(self.nu,)}, got {action_array.shape}")
        if not np.isfinite(action_array).all():
            raise ValueError("action contains NaN or Inf")

        targets = self._default_targets.copy()
        targets[np.asarray(self.control_target_indices, dtype=np.int64)] = action_array
        self.control.clear(self.model)
        self.control.joint_target_q.assign(targets)
        self._last_action = action_array.astype(np.float64)

    def _advance_one(self) -> None:
        """Advance one fine physics step without host synchronization."""

        self.state_0.clear_forces()
        self.solver.step(
            self.state_0,
            self.state_1,
            self.control,
            None,
            self.dt,
        )
        if self.enable_contacts:
            self.solver.update_contacts(self.contacts, self.state_0)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def advance(self, *, substeps: int = 1) -> None:
        """Advance while holding the most recently applied control.

        This split mirrors the main project's ``EvalSimulator`` contract:
        ``apply_control(ctrl)`` happens once per controller tick, then the
        simulator advances many fine physics steps with that target held.
        ``step(action, substeps=...)`` remains as the compatibility convenience
        used by the earlier feasibility experiments.
        """

        if substeps < 1:
            raise ValueError("substeps must be positive")
        for _ in range(substeps):
            self._advance_one()
        self.update_sites_device()

    def advance_diagnostic(self, *, substeps: int = 1) -> list[FineStepDiagnostic]:
        """Advance and download status after every fine step.

        This intentionally synchronizes the device repeatedly and must never be
        used for performance measurement.  Its purpose is to prove whether an
        interval's hidden fine steps were finite and converged, rather than
        inferring that from only the last status.
        """

        if substeps < 1:
            raise ValueError("substeps must be positive")
        diagnostics: list[FineStepDiagnostic] = []
        for fine_step in range(1, substeps + 1):
            self._advance_one()
            status = self.solver_status()
            points, pairs = self.contact_diagnostics()
            qpos = self.state_0.joint_q.numpy()
            qvel = self.state_0.joint_qd.numpy()
            diagnostics.append(
                FineStepDiagnostic(
                    fine_step=fine_step,
                    state_finite=bool(
                        np.isfinite(qpos).all() and np.isfinite(qvel).all()
                    ),
                    converged=int(status["converged"]),
                    iterations=int(status["iterations"]),
                    r_p=float(status["r_p"]),
                    r_d=float(status["r_d"]),
                    r_c=float(status["r_c"]),
                    candidate_points=points,
                    candidate_pairs=pairs,
                )
            )
        self.update_sites_device()
        return diagnostics

    def step(self, action: np.ndarray, *, substeps: int = 1) -> None:
        """Advance a fixed number of Kamino physics steps."""

        self.set_control(action)
        self.advance(substeps=substeps)

    def contact_diagnostics(self) -> tuple[int, int]:
        """Return exported candidate-point and unique shape-pair counts.

        Kamino's fixed-capacity contact representation can export positive-gap
        candidates as well as penetrating points.  These counts therefore
        describe the solver workload, not the number of load-bearing contacts.
        """

        if not self.enable_contacts:
            return 0, 0
        point_count = int(self.contacts.rigid_contact_count.numpy()[0])
        if point_count == 0:
            return 0, 0
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:point_count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:point_count]
        pairs = np.stack((np.minimum(shape0, shape1), np.maximum(shape0, shape1)), axis=1)
        unique_pair_count = int(np.unique(pairs, axis=0).shape[0])
        return point_count, unique_pair_count

    def solver_status(self) -> dict[str, float | int]:
        """Return the latest one-world PADMM status as plain Python values."""

        status = self.solver.status.numpy()[0]
        return {
            "converged": int(status["converged"]),
            "iterations": int(status["iterations"]),
            "r_p": float(status["r_p"]),
            "r_d": float(status["r_d"]),
            "r_c": float(status["r_c"]),
        }

    def update_sites_device(self) -> wp.array:
        """Compute all site world transforms without a host round trip."""

        wp.launch(
            _compute_site_world_transforms,
            dim=len(self.site_records),
            inputs=[
                self.site_shape_indices,
                self.model.shape_body,
                self.model.shape_transform,
                self.state_0.body_q,
                self.site_world,
            ],
            device=self.device,
        )
        return self.site_world

    def observation(self) -> EngineObservation:
        """Download a diagnostic observation in the planner's current layout."""

        transforms = self.site_world.numpy()
        site_xpos = transforms[:, :3].astype(np.float64, copy=True)
        site_xmat = np.stack(
            [_quat_xyzw_to_matrix(transform[3:7]) for transform in transforms]
        )
        return EngineObservation(
            qpos=newton_qpos_to_mujoco(
                self.state_0.joint_q.numpy(), self.free_joint_q_starts
            ),
            qvel=self.state_0.joint_qd.numpy().astype(np.float64, copy=True),
            control=self._last_action.copy(),
            site_names=tuple(record.name for record in self.site_records),
            site_xpos=site_xpos,
            site_xmat=site_xmat,
        )
