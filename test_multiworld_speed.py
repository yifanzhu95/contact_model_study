import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjwarp
import time
import argparse


def main(engine, nworld, nsteps, model_path):
    """
    Benchmarks multi-world simulation performance for mjwarp.step or mjwarp.step_comfree.

    Args:
        engine (int): The simulation engine to use (1 for mjwarp.step, 2 for mjwarp.step_comfree).
        nworld (int): The number of parallel simulation worlds.
        nsteps (int): The number of simulation steps to run for the benchmark.
        model_path (str): The path to the MuJoCo XML model file.
    """
    engine_name = "MJWARP" if engine == 1 else "COMFREE_WARP"
    print(f"--- Starting Benchmark ---")
    print(f"Engine: {engine_name}")
    print(f"Model: {model_path}")
    print(f"Number of Worlds: {nworld}")
    print(f"Number of Steps: {nsteps}")
    print("--------------------------")

    # Load MuJoCo model
    try:
        mjm = mujoco.MjSpec.from_file(model_path).compile()
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    # Initialize Warp model and data for multi-world simulation
    m = mjwarp.put_model(mjm)
    # Increase constraint buffers for multi-world scenarios
    d = mjwarp.put_data(mjm, mjd, nworld=nworld, nconmax=2000, njmax=2000)

    # Set initial state for all worlds (here, we replicate the default state)
    qpos_init = np.tile(mjd.qpos, (nworld, 1))
    qvel_init = np.tile(mjd.qvel, (nworld, 1))
    wp.copy(d.qpos, wp.array(qpos_init, dtype=wp.float32))
    wp.copy(d.qvel, wp.array(qvel_init, dtype=wp.float32))

    # Select the step function based on the chosen engine
    if engine == 1:
        print("Compiling mjwarp.step...")
        step_func = mjwarp.step
    else:  # engine == 2
        print("Compiling mjwarp.step_comfree...")
        step_func = mjwarp.step_comfree

    # Perform warm-up runs to ensure kernels are compiled
    step_func(m, d)
    step_func(m, d)
    wp.synchronize()

    # Capture the simulation step into a CUDA graph for high-speed execution
    with wp.ScopedCapture() as capture:
        step_func(m, d)

    graph = capture.graph
    print("Compilation complete. Starting benchmark...")

    # --- Run Benchmark ---
    wp.synchronize()
    start_time = time.time()

    for _ in range(nsteps):
        # Launch the captured graph
        wp.capture_launch(graph)

    wp.synchronize()
    end_time = time.time()
    # ---------------------

    elapsed_time = end_time - start_time
    total_steps = nsteps * nworld
    sps = total_steps / elapsed_time

    print("\n--- Benchmark Results ---")
    print(f"Total time for {nsteps} steps: {elapsed_time:.4f} seconds")
    print(f"Aggregate Steps Per Second (SPS): {sps:,.0f}")
    print("-------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark multi-world simulation speed for mjwarp engines.")
    parser.add_argument(
        "--engine", type=int, default=2, choices=[1, 2], help="Engine to test: 1 for mjwarp.step, 2 for mjwarp.step_comfree"
    )
    parser.add_argument("--nworld", type=int, default=128, help="Number of parallel simulation worlds")
    parser.add_argument("--nsteps", type=int, default=1000, help="Number of simulation steps to run")
    parser.add_argument(
        "--model", type=str, default="mujoco_warp/test_data/humanoid/humanoid.xml", help="Path to the MuJoCo XML model"
    )

    args = parser.parse_args()

    main(engine=args.engine, nworld=args.nworld, nsteps=args.nsteps, model_path=args.model)