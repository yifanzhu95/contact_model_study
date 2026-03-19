"""Grasp-robustness check for a Panda arm lifting a passive cube."""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import test_headless.streaming as mjstream

wp = None
cfwarp = None
mjwarp = None
WARP_IMPORT_ERROR = None


MODEL_PATH = "benchmark/franka_cube_grasp/scene.xml"
ARM_DOF = 7
OPEN_GRIPPER = 0.04
CLOSED_GRIPPER = 0.014
LIFTED_THRESHOLD = 0.18


@dataclass(frozen=True)
class Phase:
    name: str
    steps: int
    arm_target: np.ndarray
    finger_target: float
    force_scale: float = 0.0


def _load_keyframe_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if model.nkey == 0:
        return
    key = model.key(0)
    data.qpos[:] = key.qpos
    data.qvel[:] = 0.0
    if model.nu:
        data.ctrl[:] = key.ctrl
    mujoco.mj_forward(model, data)


def _joint_limits(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty(ARM_DOF)
    upper = np.empty(ARM_DOF)
    for i in range(ARM_DOF):
        jnt_id = i
        lower[i], upper[i] = model.jnt_range[jnt_id]
    return lower, upper


def solve_position_ik(
    model: mujoco.MjModel,
    seed_qpos: np.ndarray,
    target_pos: np.ndarray,
    cube_qpos: np.ndarray,
    *,
    site_name: str = "gripper",
    max_steps: int = 250,
    step_scale: float = 0.7,
    damping: float = 1e-4,
    tol: float = 1e-5,
) -> np.ndarray:
    """Damped least-squares IK for the Panda arm only."""
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    lower, upper = _joint_limits(model)
    jacp = np.zeros((3, model.nv))
    qpos = seed_qpos.astype(np.float64).copy()

    for _ in range(max_steps):
        data.qpos[:ARM_DOF] = qpos
        data.qpos[ARM_DOF:ARM_DOF + 2] = OPEN_GRIPPER
        data.qpos[ARM_DOF + 2:] = cube_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        err = target_pos - data.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            break

        jacp.fill(0.0)
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        arm_jac = jacp[:, :ARM_DOF]
        lhs = arm_jac @ arm_jac.T + damping * np.eye(3)
        dq = arm_jac.T @ np.linalg.solve(lhs, err)
        qpos = np.clip(qpos + step_scale * dq, lower, upper)

    return qpos


def build_phase_targets(model: mujoco.MjModel, data: mujoco.MjData) -> list[Phase]:
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_qpos_adr = model.jnt_qposadr[cube_joint_id]
    cube_qpos = data.qpos[cube_qpos_adr:cube_qpos_adr + 7].copy()
    cube_center = cube_qpos[:3]

    seed = data.ctrl[:ARM_DOF].copy()
    approach_q = solve_position_ik(model, seed, cube_center + np.array([0.0, 0.0, 0.14]), cube_qpos)
    pregrasp_q = solve_position_ik(model, approach_q, cube_center + np.array([0.0, 0.0, 0.07]), cube_qpos)
    grasp_q = solve_position_ik(model, pregrasp_q, cube_center + np.array([0.0, 0.0, 0.03]), cube_qpos)
    lift_q = solve_position_ik(model, grasp_q, cube_center + np.array([0.0, 0.0, 0.18]), cube_qpos)

    return [
        Phase("approach", 320, approach_q, OPEN_GRIPPER),
        Phase("pregrasp", 260, pregrasp_q, OPEN_GRIPPER),
        Phase("descend", 360, grasp_q, OPEN_GRIPPER),
        Phase("settle", 220, grasp_q, OPEN_GRIPPER),
        Phase("squeeze", 460, grasp_q, CLOSED_GRIPPER),
        Phase("lift", 420, lift_q, CLOSED_GRIPPER),
        Phase("hold", 560, lift_q, CLOSED_GRIPPER),
        Phase("perturb", 720, lift_q, CLOSED_GRIPPER, force_scale=1.5),
    ]


def _smoothstep(alpha: float) -> float:
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _import_warp_backends() -> tuple[object, object, object]:
    global wp, cfwarp, mjwarp, WARP_IMPORT_ERROR

    if wp is not None and cfwarp is not None and mjwarp is not None:
        return wp, cfwarp, mjwarp

    if WARP_IMPORT_ERROR is not None:
        raise RuntimeError(f"warp backend imports failed earlier: {WARP_IMPORT_ERROR}") from WARP_IMPORT_ERROR

    try:
        import warp as wp_mod
        import comfree_warp as cfwarp_mod
        from comfree_warp import mujoco_warp as mjwarp_mod
    except Exception as exc:  # pragma: no cover - depends on local runtime
        WARP_IMPORT_ERROR = exc
        raise RuntimeError(f"warp backend imports failed: {exc}") from exc

    wp = wp_mod
    cfwarp = cfwarp_mod
    mjwarp = mjwarp_mod
    return wp, cfwarp, mjwarp


def make_backend(
    engine: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    nworld: int,
    nconmax: int,
    njmax: int,
    stiffness: float,
    damping: float,
):
    if engine == "mujoco":
        return None, None, None

    wp_mod, cfwarp_mod, mjwarp_mod = _import_warp_backends()

    wp_mod.init()
    if engine == "mjwarp":
        backend_model = mjwarp_mod.put_model(model)
        backend_data = mjwarp_mod.put_data(model, data, nworld=nworld, nconmax=nconmax, njmax=njmax)
        step_fn = mjwarp_mod.step
    else:
        backend_model = cfwarp_mod.put_model(
            model,
            comfree_stiffness=stiffness,
            comfree_damping=damping,
        )
        backend_data = cfwarp_mod.put_data(model, data, nworld=nworld, nconmax=nconmax, njmax=njmax)
        step_fn = cfwarp_mod.step

    step_fn(backend_model, backend_data)
    step_fn(backend_model, backend_data)
    return backend_model, backend_data, step_fn


def sync_backend_from_mujoco(
    engine: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    backend_data,
) -> None:
    if engine == "mujoco":
        return

    wp_mod, _, _ = _import_warp_backends()
    wp_mod.copy(backend_data.ctrl, wp_mod.array([data.ctrl.astype(np.float32)]))
    wp_mod.copy(backend_data.act, wp_mod.array([data.act.astype(np.float32)]))
    wp_mod.copy(backend_data.xfrc_applied, wp_mod.array([data.xfrc_applied.astype(np.float32)]))
    wp_mod.copy(backend_data.qpos, wp_mod.array([data.qpos.astype(np.float32)]))
    wp_mod.copy(backend_data.qvel, wp_mod.array([data.qvel.astype(np.float32)]))
    wp_mod.copy(backend_data.time, wp_mod.array([data.time], dtype=wp.float32))


def step_backend(
    engine: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    backend_model,
    backend_data,
    step_fn,
) -> None:
    if engine == "mujoco":
        mujoco.mj_step(model, data)
        return

    wp_mod, _, mjwarp_mod = _import_warp_backends()
    sync_backend_from_mujoco(engine, model, data, backend_data)
    step_fn(backend_model, backend_data)
    wp_mod.synchronize()
    mjwarp_mod.get_data_into(data, model, backend_data, world_id=0)


def run_grasp_trial(
    *,
    engine: str,
    steps_scale: float,
    nworld: int,
    nconmax: int,
    njmax: int,
    stiffness: float,
    damping: float,
) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    _load_keyframe_state(model, data)
    phases = build_phase_targets(model, data)
    print(f"timestep: {model.opt.timestep}")

    stream_host = os.getenv("MJSTREAM_HOST", "127.0.0.1")
    stream_port = int(os.getenv("MJSTREAM_PORT", "7000"))
    streamer = mjstream.StreamServer(model_path=MODEL_PATH, host=stream_host, port=stream_port)
    if streamer.enabled:
        streamer.start()
        streamer.send_state(data)

    backend_model, backend_data, step_fn = make_backend(
        engine,
        model,
        data,
        nworld=nworld,
        nconmax=nconmax,
        njmax=njmax,
        stiffness=stiffness,
        damping=damping,
    )

    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_qpos_adr = model.jnt_qposadr[cube_joint_id]
    pedestal_top = 0.12

    phase_schedule: list[Phase] = []
    for phase in phases:
        phase_schedule.append(
            Phase(
                name=phase.name,
                steps=max(1, int(round(phase.steps * steps_scale))),
                arm_target=phase.arm_target,
                finger_target=phase.finger_target,
                force_scale=phase.force_scale,
            )
        )

    min_cube_height = float("inf")
    max_cube_height = float("-inf")
    step_index = 0
    prev_arm_target = data.ctrl[:ARM_DOF].copy()
    prev_finger_target = float(data.ctrl[ARM_DOF])

    try:
        for phase in phase_schedule:
            for local_step in range(phase.steps):
                alpha = _smoothstep((local_step + 1) / phase.steps)
                data.ctrl[:ARM_DOF] = (1.0 - alpha) * prev_arm_target + alpha * phase.arm_target
                data.ctrl[ARM_DOF] = (1.0 - alpha) * prev_finger_target + alpha * phase.finger_target
                data.xfrc_applied[:] = 0.0

                if phase.force_scale > 0.0:
                    t = local_step * model.opt.timestep
                    data.xfrc_applied[hand_body_id, :3] = np.array(
                        [
                            phase.force_scale * math.sin(14.0 * t),
                            0.75 * phase.force_scale * math.cos(9.0 * t),
                            0.0,
                        ]
                    )

                start = time.perf_counter()
                step_backend(engine, model, data, backend_model, backend_data, step_fn)
                elapsed = time.perf_counter() - start

                cube_height = float(data.qpos[cube_qpos_adr + 2])
                min_cube_height = min(min_cube_height, cube_height)
                max_cube_height = max(max_cube_height, cube_height)
                step_index += 1

                if streamer.enabled:
                    streamer.send_state(data)

                if streamer.enabled and elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)

            prev_arm_target = phase.arm_target.copy()
            prev_finger_target = phase.finger_target
    finally:
        if streamer.enabled:
            streamer.stop_connection()

    final_cube_height = float(data.qpos[cube_qpos_adr + 2])
    success = final_cube_height > LIFTED_THRESHOLD and min_cube_height > pedestal_top - 0.005

    return {
        "steps": float(step_index),
        "final_cube_height": final_cube_height,
        "min_cube_height": min_cube_height,
        "max_cube_height": max_cube_height,
        "success": float(success),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=("mujoco", "mjwarp", "comfree"),
        default="comfree",
        help="Simulation backend to run.",
    )
    parser.add_argument("--steps-scale", type=float, default=1.0, help="Scale phase lengths for quicker validation.")
    parser.add_argument("--nworld", type=int, default=1, help="Number of worlds for warp backends.")
    parser.add_argument("--nconmax", type=int, default=256, help="Contacts per world for warp backends.")
    parser.add_argument("--njmax", type=int, default=1024, help="Constraints per world for warp backends.")
    parser.add_argument("--contact-stiffness", type=float, default=0.2, help="comfree_warp contact stiffness.")
    parser.add_argument("--contact-damping", type=float, default=0.001, help="comfree_warp contact damping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_grasp_trial(
        engine=args.engine,
        steps_scale=args.steps_scale,
        nworld=args.nworld,
        nconmax=args.nconmax,
        njmax=args.njmax,
        stiffness=args.contact_stiffness,
        damping=args.contact_damping,
    )

    print(f"engine={args.engine}")
    print(f"steps={int(results['steps'])}")
    print(f"final_cube_height={results['final_cube_height']:.4f}")
    print(f"min_cube_height={results['min_cube_height']:.4f}")
    print(f"max_cube_height={results['max_cube_height']:.4f}")
    print(f"success={bool(results['success'])}")

    if not results["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
