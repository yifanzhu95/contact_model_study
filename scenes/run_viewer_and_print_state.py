import time
import mujoco
import mujoco.viewer
import numpy as np


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

        cam = viewer.cam
        
        # Convert azimuth and elevation to radians
        az = np.radians(cam.azimuth)
        el = np.radians(cam.elevation)
        
        # 1. Right vector (local X axis of the camera)
        right = np.array([
            np.cos(az), 
            np.sin(az), 
            0.0
        ])
        
        # 2. Up vector (local Y axis of the camera)
        up = np.array([
            -np.sin(el) * np.sin(az), 
            np.sin(el) * np.cos(az), 
            np.cos(el)
        ])
        
        # 3. Backwards vector (local Z axis pointing from lookat -> camera position)
        backwards = np.array([
            np.cos(el) * np.sin(az), 
            -np.cos(el) * np.cos(az), 
            np.sin(el)
        ])
        
        # Camera position is lookat PLUS distance along the backwards vector
        pos = cam.lookat + cam.distance * backwards
        
        # Format strings for MuJoCo XML
        pos_str = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
        xyaxis_str = f"{right[0]:.4f} {right[1]:.4f} {right[2]:.4f} {up[0]:.4f} {up[1]:.4f} {up[2]:.4f}"
        print(f'<camera name="exported_cam" pos="{pos_str}" xyaxes="{xyaxis_str}" />')
        # 4. Sync the viewer
        viewer.sync()

        # Maintain real-time simulation speed
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)