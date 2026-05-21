import time
import mujoco
import mujoco.viewer

# Load a built-in sample model (Humanoid) that contains pre-saved keyframes
model = mujoco.MjModel.from_xml_path("scenes/leap_hand/scene_leap_cube.xml")
data = mujoco.MjData(model)

# Verify the model actually has a 2nd keyframe (index 1)

# Extract the control vector from the 2nd keyframe (index 1)
# model.key_ctrl holds a flattened array of shape (nkey, nact)
target_ctrl = model.key_ctrl[0]

print(f"Successfully loaded control from 2nd keyframe: {target_ctrl}")

# Launch the interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("MuJoCo Viewer is running. Press Ctrl+C in the terminal to exit.")
    
    while viewer.is_running():
        step_start = time.time()

        # 1. Assign the control from the 2nd keyframe
        # Using [:] ensures we copy the values into the existing array memory
        data.ctrl[:] = target_ctrl

        # 2. Step the simulation forward
        mujoco.mj_step(model, data)

        # 3. Print the current states
        print(
            "Ctrl:",data.ctrl
        )
        print(
            "Qpos:",data.qpos
        )
        print(
            "Qvel:",data.qvel
        ) 

        # 4. Sync the viewer
        viewer.sync()

        # Maintain real-time simulation speed
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)