import mujoco
import numpy as np
import warp as wp
import comfree_warp.mujoco_warp as mjwarp
import comfree_warp as cfwarp

import time
import os

import streaming as mjstream


#MJC = 0; MJWARP = 1; COMFREE_WARP = 2
engine =2
nworld = 1
njmax = 5000
nconmax = 1000
# wp.clear_kernel_cache()




model_path = "benchmark/test_data/primitives.xml"
# model_path = "benchmark/humanoid/n_humanoid.xml"
# model_path = "benchmark/test_data/collision.xml"
# model_path = "benchmark/test_data/flex/floppy.xml"
# model_path = "benchmark/test_data/hfield/hfield.xml"
# model_path = "benchmark/leap/env_leap_cube.xml"



mjm = mujoco.MjSpec.from_file(model_path).compile()
mjd = mujoco.MjData(mjm)
mujoco.mj_forward(mjm, mjd)
print("timestep:", mjm.opt.timestep)

# viewer = mujoco.viewer.launch_passive(mjm, mjd)

# Remote streaming config; set STREAM_PORT>0 to enable state streaming.
STREAM_HOST = os.getenv("MJSTREAM_HOST", "127.0.0.1")
STREAM_PORT = int(os.getenv("MJSTREAM_PORT", "7000"))  # e.g. export MJSTREAM_PORT=6000
streamer = mjstream.StreamServer(model_path=model_path, host=STREAM_HOST, port=STREAM_PORT)
if streamer.enabled:
    streamer.start()


step_counter = 0


if engine!=0:
    if engine==1:
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
        comfree_stiffness_vec = [0.2, 0]  # Default stiffness value
        comfree_damping_vec = [0.002]  # Default damping value

        m= cfwarp.put_model(mjm, comfree_stiffness=comfree_stiffness_vec, comfree_damping=comfree_damping_vec)
        d= cfwarp.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)

        print("Compiling comfree_mjwarp step...")
        cfwarp.step(m, d)
        cfwarp.step(m, d)

        with wp.ScopedCapture() as capture:
            cfwarp.step(m, d)

        ##### Default stiffness and damping values for compilation
        # m= comfree_warp.put_model(mjm)
        # d= comfree_warp.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
        # comfree_warp.step(m, d)
        # comfree_warp.step(m, d)
        # with wp.ScopedCapture() as capture:
        #     comfree_warp.step(m, d)

    graph = capture.graph
    print("Compiled.")
else:
    print("Running MJC...")

# step_counter = 1000
time.sleep(2)
while step_counter<10000:
    if engine==0:
        start = time.time()
        mujoco.mj_step(mjm, mjd)
        elapsed = time.time() - start

    else:
        random_ctrl = 0.0*np.random.randn(*d.ctrl.shape)  # create random array with same shape
        wp.copy(d.ctrl, wp.array(random_ctrl.astype(np.float32)))
        # wp.copy(d.ctrl, wp.array([mjd.ctrl.astype(np.float32)]))
        wp.copy(d.act, wp.array([mjd.act.astype(np.float32)]))
        wp.copy(d.xfrc_applied, wp.array([mjd.xfrc_applied.astype(np.float32)]))
        wp.copy(d.qpos, wp.array([mjd.qpos.astype(np.float32)]))
        wp.copy(d.qvel, wp.array([mjd.qvel.astype(np.float32)]))
        wp.copy(d.time, wp.array([mjd.time], dtype=wp.float32))
        
        start = time.time()
        wp.capture_launch(graph)
        wp.synchronize()
        elapsed = time.time() - start

        mjwarp.get_data_into(mjd, mjm, d)


    if elapsed < mjm.opt.timestep:
        time.sleep(mjm.opt.timestep - elapsed)

    print(f"Step {step_counter} took {(time.time() - start) * 1000:.2f} ms.")

    step_counter+=1

    # Sync viewer
    if streamer.enabled:
        streamer.send_state(mjd)
    

# Close streamer connection
if streamer.enabled:
    streamer.stop_connection()
