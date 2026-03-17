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
    Extract and log linear velocity in x-direction.
    
    Args:
        d: Data structure containing velocity information
        step_num: Current step number
        log_rows: List to append velocity data
    """
    # Get velocity - for rolling cylinder, we care about linear velocity
    qvel = d.qvel.numpy().flatten()
    
    # For a rolling body, qvel has linear velocity in first 3 components
    # and angular velocity in next 3 components
    linear_vel_x = qvel[0]
    linear_vel_y = qvel[1] if len(qvel) > 1 else 0
    linear_vel_z = qvel[2] if len(qvel) > 2 else 0
    
    log_rows.append((step_num, linear_vel_x, linear_vel_y, linear_vel_z))
    
    if step_num % 50 == 0:
        print(f"Step {step_num}: linear_vel_x={linear_vel_x:.6f} m/s")


def run_dynamics_test(num_steps, contact_stiffness=100.0, contact_damping=2.0, timestep=None, engine=2, output_path=None, ground_friction=None, cylinder_friction=None):
    """
    Run rolling friction test with configurable rolling friction coefficients.
    
    Args:
        num_steps: Number of simulation steps
        contact_stiffness: Contact stiffness for comfree_warp
        contact_damping: Contact damping for comfree_warp
        timestep: Simulation timestep (optional)
        engine: Physics engine (0=MJC, 1=MJWARP, 2=COMFREE_WARP)
        output_path: Path to save velocity log (optional)
        ground_friction: Rolling friction coefficient for ground (modifies friction[2])
        cylinder_friction: Rolling friction coefficient for cylinder (modifies friction[2])
    """
    # Configuration
    # MJC = 0; MJWARP = 1; COMFREE_WARP = 2
    nworld = 1
    njmax = 50
    nconmax = 100

    # Model selection
    model_path = "benchmark/test_data/cylinder_rolling.xml"

    # Load MuJoCo model
    print(f"Loading model from {model_path}...")
    spec = mujoco.MjSpec.from_file(model_path)
    mjm = spec.compile()
    mjd = mujoco.MjData(mjm)
    
    # Modify rolling friction coefficients if provided (friction[2])
    if ground_friction is not None or cylinder_friction is not None:
        # Iterate through geoms and find by type
        for i in range(mjm.ngeom):
            geom_type = mjm.geom_type[i]
            # plane type is 0, cylinder type is 3
            if ground_friction is not None and geom_type == 0:  # plane
                friction_tuple = list(mjm.geom_friction[i])
                friction_tuple[2] = ground_friction  # Modify only rolling friction
                mjm.geom_friction[i] = tuple(friction_tuple)
                print(f"Ground (plane geom {i}) rolling friction set to: {ground_friction} (mu preserved at {friction_tuple[0]})")
            elif cylinder_friction is not None and geom_type == 3:  # cylinder
                friction_tuple = list(mjm.geom_friction[i])
                friction_tuple[2] = cylinder_friction  # Modify only rolling friction
                mjm.geom_friction[i] = tuple(friction_tuple)
                print(f"Cylinder (geom {i}) rolling friction set to: {cylinder_friction} (mu preserved at {friction_tuple[0]})")
    
    # Set timestep if specified
    if timestep is not None:
        mjm.opt.timestep = timestep
    
    mujoco.mj_forward(mjm, mjd)

    print(f"Model loaded: {mjm.ngeom} geoms, {mjm.nbody} bodies")
    print(f"Engine: {'MJC' if engine == 0 else 'MJWARP' if engine == 1 else 'COMFREE_WARP'}")
    print(f"Timestep: {mjm.opt.timestep}")
    print("=" * 80)

    # Set initial linear velocity for cylinder (rolling along x-axis)
    # For a free body, qvel has linear velocity in first 3 components
    initial_linear_vel_x = 1.0  # m/s
    mjd.qvel[0] = initial_linear_vel_x
    mjd.qvel[1] = 0.0  # y velocity
    mjd.qvel[2] = 0.0  # z velocity

    print(f"Initial linear velocity (x-axis): {initial_linear_vel_x} m/s")

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
    os.makedirs("logs/rolling", exist_ok=True)

    # Rolling friction configurations (friction[2])
    # Only modifying rolling friction, mu is preserved from the model
    rolling_friction_configs = [
        # (0.0, 0.0),       # No rolling friction (frictionless)
        (0.001, 0.001),   # Very low rolling friction
        (0.01, 0.01),     # Low rolling friction
        (0.05, 0.05),     # Medium rolling friction
        (0.1, 0.1),       # High rolling friction
    ]

    # Run comfree_warp tests with different rolling friction configurations
    for ground_rolling, cylinder_rolling in rolling_friction_configs:
        # output_path = f"logs/rolling/linear_vel_groll{ground_rolling:.3f}_croll{cylinder_rolling:.3f}.npy"
        output_path = None
        log_arr = run_dynamics_test(
            num_steps=1000,
            contact_stiffness=0.2,
            contact_damping=0.004,
            engine=2,
            output_path=output_path,
            ground_friction=ground_rolling,
            cylinder_friction=cylinder_rolling,
        )
        print(
            f"Test complete (ground_rolling={ground_rolling}, cylinder_rolling={cylinder_rolling}). "
            f"Logged {len(log_arr)} linear velocity entries.\n"
        )
    
    