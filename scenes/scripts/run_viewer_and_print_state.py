import time
import mujoco
import mujoco.viewer
import numpy as np


# Set to True to initialize qpos/ctrl from _INIT_QPOS / _INIT_CTRL below
# instead of from the model's first keyframe.
_USE_INIT_VECTORS = True

# Set to True to reproduce the MuJoCo eval simulator's model mutations
# (contact_study.sim.mujoco_sim.MujocoSimulator): override the timestep to the
# task's fine eval timestep AND harden every contact via the M1 hard-contact
# preset. With the default soft XML contacts the grasp is stable; flipping this
# on should make the cube fly off exactly like `run_eval_episode --eval_sim
# mujoco`, isolating the mismatch to the eval sim's model config (not the state).
_MIMIC_EVAL_SIM = False

# Eval sim timestep for grasp_reorient (TaskConfig.timestep). The XML declares
# no <option timestep>, so the plain viewer runs at MuJoCo's 0.002 default.
_EVAL_TIMESTEP = 0.001

_INIT_QPOS = np.array([
5.74299632e-01, -5.68530386e-01,  9.13531510e-01,  5.73948062e-01,
 -9.82115272e-03, -8.35144626e-02,  7.03221454e-01,  1.01842742e+00,
  5.88681352e-01,  6.10758737e-01,  9.26063146e-01,  6.10230013e-01,
  7.00238917e-01,  1.45217393e+00,  1.33872725e+00,  8.68901913e-01,
  2.32854961e-02,  3.42861479e-02,  7.92305817e-02,  9.99977455e-01,
 -9.40658784e-04, -4.87316782e-03,  4.52301601e-03
], dtype=np.float64)


_INIT_CTRL = np.array([
0.60184  , -0.568012 ,  0.916951 ,  0.573897 , -0.0191225, -0.0837503,
  0.709056 ,  1.01884  ,  0.61456  ,  0.610365 ,  0.929305 ,  0.610097 ,
  0.69912  ,  1.45882  ,  1.33179  ,  0.8657
], dtype=np.float64)


# Load a built-in sample model (Humanoid) that contains pre-saved keyframes
model = mujoco.MjModel.from_xml_path("scenes/leap/env_leap_eval_cube.xml")#"scenes/leap_hand/scene_leap_cube.xml")

if _MIMIC_EVAL_SIM:
    # Mirror MujocoSimulator.__init__: set the fine eval timestep FIRST (the
    # hard-contact preset derives its solref timeconst from opt.timestep), then
    # harden every contact row toward the near-rigid M1 limit. This mutates the
    # model in place, exactly as the eval sim does before stepping.
    from contact_study.contact_models.api import _apply_hard_contact_preset
    from contact_study.contact_models.config import ContactModelConfig

    model.opt.timestep = _EVAL_TIMESTEP
    _apply_hard_contact_preset(model, ContactModelConfig.M1().mujoco)
    print(
        f"[_MIMIC_EVAL_SIM] hardened contacts + timestep={model.opt.timestep} "
        "(reproducing the MuJoCo eval simulator)"
    )

data = mujoco.MjData(model)

if _USE_INIT_VECTORS:
    # Initialize qpos/ctrl from the hard-coded vectors above.
    data.qpos[:] = _INIT_QPOS
    data.ctrl[:] = _INIT_CTRL
    print(f"Successfully loaded control from _INIT_CTRL: {data.ctrl}")
else:
    # Verify the model actually has a 2nd keyframe (index 1)

    # Extract the control vector from the 2nd keyframe (index 1)
    # model.key_ctrl holds a flattened array of shape (nkey, nact)
    target_ctrl = model.key_ctrl[0]

    print(f"Successfully loaded control from 2nd keyframe: {target_ctrl}")

    # Initialize data to the first keyframe (qpos, qvel, ctrl, etc.)
    mujoco.mj_resetDataKeyframe(model, data, 0)

mujoco.mj_forward(model, data)

# Launch the interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("MuJoCo Viewer is running. Press Ctrl+C in the terminal to exit.")
    
    while viewer.is_running():
        step_start = time.time()

        # Note: data.ctrl is intentionally left alone here so it can be
        # driven live from the viewer's "Control" panel sliders. Mouse-drag
        # perturbations (Ctrl + right-click/left-click a body) are applied
        # below so objects can be pushed/moved around in the viewer too.
        mujoco.mjv_applyPerturbPose(model, data, viewer.perturb, 0)
        mujoco.mjv_applyPerturbForce(model, data, viewer.perturb)

        # Step the simulation forward
        mujoco.mj_step(model, data)

        # 3. Print the current states
        print(
            "Ctrl:", np.array2string(data.ctrl, separator=", ")
        )
        print(
            "Qpos:", np.array2string(data.qpos, separator=", ")
        )
        print(
            "Qvel:", np.array2string(data.qvel, separator=", ")
        )
        for site_id in range(model.nsite):
            site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
            print(f"Site '{site_name}' world pos:", data.site_xpos[site_id])

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