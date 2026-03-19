import mujoco
import mujoco.viewer
import numpy as np
import warp as wp
import comfree_warp.mujoco_warp as mjwarp
import comfree_warp as cfwarp

import time


# MJC = 0; MJWARP = 1; COMFREE_WARP = 2
engine = 2
nworld = 1
njmax = 5000
nconmax = 1000

# set device
wp.set_device("cuda:0")

model_path = "benchmark/test_data/primitives.xml"
# model_path = "benchmark/humanoid/n_humanoid.xml"
# model_path = "benchmark/test_data/collision.xml"
# model_path = "benchmark/test_data/flex/floppy.xml"
# model_path = "benchmark/test_data/hfield/hfield.xml"
# model_path = "benchmark/leap/env_leap_cube.xml"

mjm = mujoco.MjSpec.from_file(model_path).compile()
mjm.opt.ccd_iterations = 50
mjd = mujoco.MjData(mjm)
if mjm.nkey > 0:
    keyframe = mjm.key(0)
    mjd.qpos[:] = keyframe.qpos
    mjd.qvel[:] = keyframe.qvel
    mjd.ctrl[:] = keyframe.ctrl
ref_ctrl = mjd.ctrl.copy()
mujoco.mj_forward(mjm, mjd)
print("timestep:", mjm.opt.timestep)

# Launch native MuJoCo viewer
viewer = mujoco.viewer.launch_passive(mjm, mjd)

step_counter = 0

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
        # Check if comfree_stiffness_val is defined, if not, set a default.
        # This makes the stiffness value explicit.
        comfree_stiffness_vec = 0.1 # Default stiffness value
        comfree_damping_vec = 0.001  # Default damping value

        m = cfwarp.put_model(mjm, comfree_stiffness=comfree_stiffness_vec, comfree_damping=comfree_damping_vec)
        d = cfwarp.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)

        print("Compiling comfree_mjwarp step...")
        cfwarp.step(m, d)
        cfwarp.step(m, d)

        with wp.ScopedCapture() as capture:
            cfwarp.step(m, d)

    graph = capture.graph
    print("Compiled.")
else:
    print("Running MJC...")


time.sleep(0.5)

while step_counter<10000:        
    if engine == 0:
        start = time.time()
        mujoco.mj_step(mjm, mjd)
        elapsed = time.time() - start

    else:
        start = time.time()
        wp.capture_launch(graph)
        wp.synchronize()
        elapsed = time.time() - start

        mjwarp.get_data_into(mjd, mjm, d)

    if elapsed < mjm.opt.timestep:
        time.sleep(mjm.opt.timestep - elapsed)

    print(f"Step {step_counter} took {(time.time() - start) * 1000:.2f} ms.")

    # Update viewer with latest data
    viewer.sync()

    step_counter += 1
