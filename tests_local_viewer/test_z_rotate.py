"""
Torsional Friction Test Script

This script tests torsional friction by simulating a ball with only 1 DOF (rotation around z-axis).
The ball is given an initial angular velocity and the decay process is logged over time.
"""

import mujoco
import mujoco.viewer
import numpy as np
import warp as wp
from comfree_warp import mujoco_warp as mjwarp
import comfree_warp as cfwarp

import time
import os

def log_velocities(d, step_num, log_rows):
    """
    Extract and log angular velocity around z-axis.
    
    Args:
        d: Data structure containing velocity information
        step_num: Current step number
        log_rows: List to append velocity data
    """
    # Get velocity - for ball with hinge joint, qvel has only 1 component (z-axis rotation)
    qvel = d.qvel.numpy().flatten()
    
    # For a hinge joint around z-axis, qvel[0] is the angular velocity
    angular_vel_z = qvel[0]
    
    log_rows.append((step_num, angular_vel_z))
    
    if step_num % 50 == 0:
        print(f"Step {step_num}: angular_vel_z={angular_vel_z:.6f} rad/s")


def run_dynamics_test(num_steps, contact_stiffness=100.0, contact_damping=2.0, timestep=None, engine=2, output_path=None, ground_friction=None, ball_friction=None):
    """
    Run torsional friction test with configurable torsional friction coefficients.
    
    Args:
        num_steps: Number of simulation steps
        contact_stiffness: Contact stiffness for comfree_warp
        contact_damping: Contact damping for comfree_warp
        timestep: Simulation timestep (optional)
        engine: Physics engine (0=MJC, 1=MJWARP, 2=COMFREE_WARP)
        output_path: Path to save velocity log (optional)
        ground_friction: Torsional friction coefficient for ground (modifies friction[1])
        ball_friction: Torsional friction coefficient for ball (modifies friction[1])
    """
    # Configuration
    # MJC = 0; MJWARP = 1; COMFREE_WARP = 2
    nworld = 1
    njmax = 50
    nconmax = 100

    # Model selection
    model_path = "benchmark/test_data/ball_rotation.xml"

    # Load MuJoCo model
    print(f"Loading model from {model_path}...")
    spec = mujoco.MjSpec.from_file(model_path)
    mjm = spec.compile()
    mjd = mujoco.MjData(mjm)
    
    # Modify torsional friction coefficients if provided (friction[1])
    if ground_friction is not None or ball_friction is not None:
        # Iterate through geoms and find by type or position
        for i in range(mjm.ngeom):
            geom_type = mjm.geom_type[i]
            # plane type is 0, sphere type is 2
            if ground_friction is not None and geom_type == 0:  # plane
                friction_tuple = list(mjm.geom_friction[i])
                friction_tuple[1] = ground_friction  # Modify only torsion friction
                mjm.geom_friction[i] = tuple(friction_tuple)
                print(f"Ground (plane geom {i}) torsional friction set to: {ground_friction} (mu preserved at {friction_tuple[0]})")
            elif ball_friction is not None and geom_type == 2:  # sphere
                friction_tuple = list(mjm.geom_friction[i])
                friction_tuple[1] = ball_friction  # Modify only torsion friction
                mjm.geom_friction[i] = tuple(friction_tuple)
                print(f"Ball (sphere geom {i}) torsional friction set to: {ball_friction} (mu preserved at {friction_tuple[0]})")
    
    # Set timestep if specified
    if timestep is not None:
        mjm.opt.timestep = timestep
    
    mujoco.mj_forward(mjm, mjd)

    print(f"Model loaded: {mjm.ngeom} geoms, {mjm.nbody} bodies")
    print(f"Engine: {'MJC' if engine == 0 else 'MJWARP' if engine == 1 else 'COMFREE_WARP'}")
    print(f"Timestep: {mjm.opt.timestep}")
    print("=" * 80)

    # Set initial angular velocity for ball (rotation around z-axis)
    # For a hinge joint, qvel has only 1 component
    initial_angular_vel = 3.0  # rad/s
    mjd.qvel[0] = initial_angular_vel

    print(f"Initial angular velocity (z-axis): {initial_angular_vel} rad/s")

    # Launch native MuJoCo viewer
    viewer = mujoco.viewer.launch_passive(mjm, mjd)

    if engine != 0:
        if engine == 1:
            m = mjwarp.put_model(mjm)
            d = mjwarp.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)

            print("Compiling mjwarp step...")
            mjwarp.step(m, d)
            mjwarp.step(m, d)
            with wp.ScopedCapture() as capture:
                mjwarp.step(m, d)
        else:
            m = cfwarp.put_model(mjm, comfree_stiffness=contact_stiffness, comfree_damping=contact_damping)
            d = cfwarp.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)

            print("Compiling comfree_warp step...")
            cfwarp.step(m, d)
            cfwarp.step(m, d)

            with wp.ScopedCapture() as capture:
                cfwarp.step(m, d)

        graph = capture.graph
        print("Compiled.")
    else:
        print("Running MJC...")

    print("=" * 80)

    time.sleep(2)
    step_counter = 0
    log_rows = []

    try:
        while step_counter < num_steps:
            if engine == 0:
                start = time.time()
                mujoco.mj_step(mjm, mjd)
                elapsed = time.time() - start
                log_velocities(d, step_counter, log_rows)

            else:
                random_ctrl = 0.0 * np.random.randn(*d.ctrl.shape)
                wp.copy(d.ctrl, wp.array(random_ctrl.astype(np.float32)))
                wp.copy(d.act, wp.array([mjd.act.astype(np.float32)]))
                wp.copy(d.xfrc_applied, wp.array([mjd.xfrc_applied.astype(np.float32)]))
                wp.copy(d.qpos, wp.array([mjd.qpos.astype(np.float32)]))
                wp.copy(d.qvel, wp.array([mjd.qvel.astype(np.float32)]))
                wp.copy(d.time, wp.array([mjd.time], dtype=wp.float32))

                start = time.time()
                wp.capture_launch(graph)
                wp.synchronize()
                elapsed = time.time() - start

                log_velocities(d, step_counter, log_rows)
                mjwarp.get_data_into(mjd, mjm, d)

            if elapsed < mjm.opt.timestep:
                time.sleep(mjm.opt.timestep - elapsed)

            # print(f"Step {step_counter} took {(time.time() - start) * 1000:.2f} ms.")

            step_counter += 1

            # Sync viewer
            viewer.sync()

    finally:
        # Close viewer
        viewer.close()

    log_array = np.array(log_rows, dtype=np.float32)
    if output_path is not None:
        np.save(output_path, log_array)
        print(f"Saved velocity log to {output_path} with {len(log_rows)} entries.")

    return log_array


if __name__ == "__main__":
    os.makedirs("logs/z_rotate", exist_ok=True)

    # Torsional friction configurations (friction[1])
    # Only modifying torsional friction, mu is preserved from the model
    torsion_friction_configs = [
        (0.001, 0.001),   # Very low torsional friction
        (0.01, 0.01),     # Low torsional friction
        (0.05, 0.05),     # Medium torsional friction
        (0.1, 0.1),       # High torsional friction
    ]

    # Run comfree_warp tests with different torsional friction configurations
    for ground_torsion, ball_torsion in torsion_friction_configs:
        # output_path = f"logs/z_rotate/angular_vel_gtors{ground_torsion:.3f}_btors{ball_torsion:.3f}.npy"
        output_path = None
        log_arr = run_dynamics_test(
            num_steps=1000,
            contact_stiffness=0.2,
            contact_damping=0.004,
            engine=2,
            output_path=output_path,
            ground_friction=ground_torsion,
            ball_friction=ball_torsion,
        )
        print(
            f"Test complete (ground_torsion={ground_torsion}, ball_torsion={ball_torsion}). "
            f"Logged {len(log_arr)} angular velocity entries.\n"
        )
    
    