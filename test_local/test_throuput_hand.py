"""Throughput test with 1024 parallel environments.

Logs per-step total contact count (across all environments) and step time.
"""

import csv
import os
import time
import json
from datetime import datetime

import mujoco
import mujoco.viewer
import numpy as np
import warp as wp
import comfree_warp as cfwarp
from comfree_warp import mujoco_warp as mjwarp

wp.set_device("cuda:0")

def run_hand_test(
    num_steps=1000,
    nworld=1024,
    engine=2,
    model_path="benchmark/test_data/allegro/env_allegro_cube.xml",
    nconmax=64,
    njmax=1000,
    contact_stiffness=0.1,
    contact_damping=0.001,
    qvel_noise_std=1e-3,
):
    # engine: 1 = mjwarp, 2 = comfree_warp
    engine_name = "MJWARP" if engine == 1 else "COMFREE_WARP"
    print(f"Loading model: {model_path}")
    mjm = mujoco.MjSpec.from_file(model_path).compile()
    mjm.opt.ccd_iterations = 50
    mjd = mujoco.MjData(mjm)
    # load keyframe state into data
    if mjm.nkey > 0:
        keyframe = mjm.key(0)
        mjd.qpos[:] = keyframe.qpos
        mjd.qvel[:] = keyframe.qvel
        mjd.ctrl[:] = keyframe.ctrl
    
    ref_ctrl = mjd.ctrl.copy()

    print(f"Engine: {engine_name}")
    print(f"Parallel envs: {nworld}")
    print(f"nconmax/world: {nconmax}, njmax/world: {njmax}")

    # Built-in viewer for visualization (shows world 0 state)
    viewer = mujoco.viewer.launch_passive(mjm, mjd)

    if engine == 1:
        m = mjwarp.put_model(mjm)
        d = mjwarp.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
        step_func = mjwarp.step
    else:
        m = cfwarp.put_model(
            mjm,
            comfree_stiffness=contact_stiffness,
            comfree_damping=contact_damping,
        )
        d = cfwarp.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
        step_func = cfwarp.step

    qpos_init = np.tile(mjd.qpos, (nworld, 1)).astype(np.float32)
    qvel_init = np.tile(mjd.qvel, (nworld, 1)).astype(np.float32)

    # Add tiny random initial velocity noise for per-env randomness.
    if qvel_noise_std > 0.0:
        qvel_init += np.random.normal(0.0, qvel_noise_std, size=qvel_init.shape).astype(np.float32)

    wp.copy(d.qpos, wp.array(qpos_init, dtype=wp.float32))
    wp.copy(d.qvel, wp.array(qvel_init, dtype=wp.float32))

    print("Compiling step graph...")
    step_func(m, d)
    step_func(m, d)
    with wp.ScopedCapture() as capture:
        step_func(m, d)
    graph = capture.graph
    wp.synchronize()
    print("Compilation done.")

    steps_data = []
    try:
        for step in range(num_steps):
            # Update control every 10 steps, uniformly sampled around ref_ctrl
            if step % 20 == 0:
                ctrl_perturbation = np.random.uniform(-0.5, 0.5, size=ref_ctrl.shape)
                new_ctrl = (ref_ctrl + ctrl_perturbation).astype(np.float32)
                ctrl_array = np.tile(new_ctrl, (nworld, 1)).astype(np.float32)
                wp.copy(d.ctrl, wp.array(ctrl_array, dtype=wp.float32))
            
            t0 = time.perf_counter()
            wp.capture_launch(graph)
            wp.synchronize()
            step_t = time.perf_counter() - t0
            steps_data.append(step_t)

            mjwarp.get_data_into(mjd, mjm, d, world_id=0)
            viewer.sync()
    finally:
        viewer.close()

    return np.array(steps_data , dtype=np.float64)


if __name__ == "__main__":
    steps = 1000
    model = "benchmark/test_data/allegro/env_allegro_cube.xml"
    nconmax = 64
    njmax = 1000
    stiffness = 0.1
    damping = 0.001
    
    # nworlds = [256, 512, 1024, 2048, 4096]
    nworlds = [256]
    engines = [1, 2]  # 1 = mjwarp, 2 = comfree_warp
    engine_names = {1: "mjwarp", 2: "comfree_warp"}
    
    results = {}
    
    for engine in engines:
        engine_name = engine_names[engine]
        results[engine_name] = {}
        
        for nworld in nworlds:
            print(f"\n{'='*60}")
            print(f"Testing {engine_name} with nworld={nworld}")
            print(f"{'='*60}")
            
            try:
                dts = run_hand_test(
                    num_steps=steps,
                    nworld=nworld,
                    engine=engine,
                    model_path=model,
                    nconmax=nconmax,
                    njmax=njmax,
                    contact_stiffness=stiffness,
                    contact_damping=damping,
                )
                
                throughput = 1 / dts * nworld
                warmup_steps = 50
                throughput_stats = throughput[warmup_steps:]
                
                stats = {
                    "nworld": nworld,
                    "num_steps": steps,
                    "mean_throughput": float(throughput_stats.mean()),
                    "std_throughput": float(throughput_stats.std()),
                    "min_throughput": float(throughput_stats.min()),
                    "max_throughput": float(throughput_stats.max()),
                    "mean_step_time_ms": float((1 / throughput_stats).mean() * 1000),
                }
                
                results[engine_name][nworld] = stats
                
                print(f"Mean throughput: {stats['mean_throughput']:.2e} steps/sec")
                print(f"Std throughput: {stats['std_throughput']:.2e}")
                print(f"Mean step time: {stats['mean_step_time_ms']:.4f} ms")
                
            except Exception as e:
                print(f"Error: {e}")
                results[engine_name][nworld] = {"error": str(e)}
    
    