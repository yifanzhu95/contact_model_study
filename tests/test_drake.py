"""Closed-loop cart-pole swing-up: Drake "real" env + GPU-MPPI planner.

Drake (pydrake) runs the high-fidelity "real"/eval environment for the
cart-pole, while all MPPI rollouts and costs are evaluated on the GPU inside a
chosen contact model (M1-M4) via contact_study.planners.mppi.MPPIController.

Pipeline per control step:
  1. Read (cart x, cart x_dot, pole angle, pole angle_rate) from the Drake plant.
  2. Mirror that state into a MuJoCo MjData and run MPPI on the GPU. MPPI is a
     *rate* controller: it optimizes per-step deltas to the cart force and
     returns the first delta (see contact_study/planners/mppi.py:80 and
     experiments/run_episode.py:217).
  3. Integrate the delta into the applied force, hold it constant over the
     control step, and advance Drake.

The MuJoCo planning model is built from scenes/cart_pole.urdf: we load the URDF
with MjSpec, add a force actuator on the cart slider, compile it, and save the
result as MJCF (scenes/cart_pole.xml). Drake keeps using scenes/cart_pole.sdf
(it already exposes one actuator on CartSlider and welds the cart to the world).
The URDF and SDF MUST stay parameter-consistent (cart/pole mass, lengths, axes).

Cost = (pole angle from upright)^2 + (cart distance from x=0)^2.

Run with the `contact_modeling` conda env (it has pydrake + mujoco + warp +
comfree_warp). A CUDA GPU is required (MPPI arrays live on the device). Drake's
VideoWriter needs a display, so run under xvfb:
    xvfb-run -a python tests/test_drake.py
"""

from __future__ import annotations

import os
# Drake's VideoWriter (VTK) does ALL rendering here; we never use MuJoCo's
# renderer. Importing mujoco with MUJOCO_GL=egl/glfw loads a GL stack that
# shadows the system OpenGL and breaks Drake's VTK GLX context under xvfb
# ("Cannot create GLX context" / missing swrast). "disable" stops mujoco from
# loading any GL backend, so Drake renders cleanly. Set before importing mujoco.
os.environ["MUJOCO_GL"] = "disable"

import argparse
from pathlib import Path

import numpy as np
import mujoco
import warp as wp

from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import VideoWriter

from contact_study.contact_models.config import ContactModelConfig
from contact_study.planners.mppi import MPPIController, MPPIConfig
from contact_study.tasks.base import BaseTask, ContactComplexity, TaskSpec

wp.init()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
SCENES   = THIS_DIR.parent / "scenes"
URDF     = SCENES / "cart_pole.urdf"          # source for the MuJoCo model
MJCF_OUT = SCENES / "cart_pole.xml"           # generated MJCF ("save as MJCF")
SDF      = SCENES / "cart_pole.sdf"           # Drake "real" env model

# ---------------------------------------------------------------------------
# Model / cost constants
# ---------------------------------------------------------------------------
TIMESTEP    = 0.02          # MuJoCo planning timestep (s); also the control dt
FORCE_MAX   = 100.0         # cart actuator force limit (N)
UPRIGHT     = float(np.pi)  # pole angle (rad) that points straight up
# Cart-pole has no contacts, so the contact-model axis is moot; M2 (mjwarp soft)
# is the natural default. Swap via --model to exercise M1/M3/M4 on the same task.
MODEL_FACTORIES = {
    "M1": ContactModelConfig.M1,
    "M2": ContactModelConfig.M2,
    "M3": ContactModelConfig.M3,
    "M4": ContactModelConfig.M4,
}


# ---------------------------------------------------------------------------
# Warp cost function — angle-from-upright + cart-distance-from-zero
# ---------------------------------------------------------------------------
@wp.func
def cart_pole_cost_wp(qpos:      wp.array(dtype=float),
                      qvel:      wp.array(dtype=float),
                      ctrl:      wp.array(dtype=float),
                      site_xpos: wp.array(dtype=wp.vec3),
                      site_xmat: wp.array(dtype=wp.mat33),
                      terminal:  bool,
                      goal:      wp.array(dtype=float),
                      indices:   wp.array(dtype=int),
                      weights:   wp.array(dtype=float)) -> float:
    slider_qpos_adr = indices[0]
    hinge_qpos_adr  = indices[1]
    slider_dof_adr  = indices[2]
    hinge_dof_adr   = indices[3]

    x     = qpos[slider_qpos_adr]
    theta = qpos[hinge_qpos_adr]

    # Wrap-safe angle from upright: 0 when the pole points up (theta = +/- pi),
    # +/- pi when it hangs straight down. atan2(sin, cos) maps to (-pi, pi].
    d       = theta - goal[1]
    ang_err = wp.atan2(wp.sin(d), wp.cos(d))
    pos_err = x - goal[0]

    if terminal:
        return weights[2] * ang_err * ang_err + weights[3] * pos_err * pos_err

    cart_vel = qvel[slider_dof_adr]
    pole_vel = qvel[hinge_dof_adr]
    return (
        weights[0] * ang_err * ang_err
        + weights[1] * pos_err * pos_err
        + weights[4] * cart_vel * cart_vel
        + weights[5] * pole_vel * pole_vel
    )


# ---------------------------------------------------------------------------
# Cart-pole task
# ---------------------------------------------------------------------------
class CartPoleTask(BaseTask):
    """Cart-pole swing-up task backed by a URDF-derived MuJoCo model.

    The pole hangs down at hinge angle 0 and is upright at +/- pi (matching
    cart_pole.sdf). The goal is to drive the pole to upright while keeping the
    cart near x = 0.
    """

    # Initial pole angle (rad). 0.0 = hanging straight down (swing-up). Set to
    # ~pi to start near upright (balance-only). Overridable from the CLI.
    init_angle: float = 0.0

    @property
    def spec(self) -> TaskSpec:
        return TaskSpec(
            name               = "cart_pole",
            complexity         = ContactComplexity.LOW,
            xml_path_template  = "cart_pole.xml",
            max_steps          = 1000,
            success_thresholds = {"angle": 0.20, "pos": 0.20, "vel": 1.0},
            # [w_angle_run, w_pos_run, w_angle_term, w_pos_term, w_cart_vel, w_pole_vel]
            cost_weights       = {
                "angle": 100.0, "pos": 75.0,
                "angle_term": 100.0, "pos_term": 100.0,
                "cart_vel": 2.0, "pole_vel": 2.0,
            },
        )

    # --- load: URDF -> add actuator -> compile -> save MJCF -----------------
    def load(self, full_path: str | None = None):
        """Build the MuJoCo model from the URDF and persist it as MJCF."""
        spec = mujoco.MjSpec.from_file(str(URDF))
        spec.option.timestep   = TIMESTEP
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        # Force actuator on the cart slider (force = ctrl; a default general
        # actuator is gain-fixed/bias-none, i.e. a motor).
        act = spec.add_actuator()
        act.name        = "cart_force"
        act.target      = "CartSlider"
        act.trntype     = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype    = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm     = [1.0] + list(act.gainprm[1:])
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange   = [-FORCE_MAX, FORCE_MAX]

        # A pole-tip site keeps the cost kernel's site arrays non-empty and is
        # handy for debugging/visualization. The cost itself reads only qpos.
        tip = spec.body("Pole").add_site()
        tip.name = "pole_tip"
        tip.pos  = [0.0, 0.0, -0.5]

        self._mjm = spec.compile()
        self._mjd = mujoco.MjData(self._mjm)
        MJCF_OUT.write_text(spec.to_xml())

        self.initialize_task()
        return self._mjm, self._mjd

    def initialize_task(self):
        mjm = self.mjm
        slider = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "CartSlider")
        hinge  = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "PolePin")

        # [slider_qposadr, hinge_qposadr, slider_dofadr, hinge_dofadr]
        self.index_vector = np.array([
            mjm.jnt_qposadr[slider],
            mjm.jnt_qposadr[hinge],
            mjm.jnt_dofadr[slider],
            mjm.jnt_dofadr[hinge],
        ], dtype=np.int32)

        # [cart_target_x, upright_angle]
        self.goal_vector = np.array([0.0, UPRIGHT], dtype=np.float32)

        w = self.spec.cost_weights
        weights_list = [
            w["angle"], w["pos"], w["angle_term"], w["pos_term"],
            w["cart_vel"], w["pole_vel"],
        ]

        self.index_vector_wp = wp.array(self.index_vector, dtype=wp.int32,   device="cuda")
        self.goal_vector_wp  = wp.array(self.goal_vector,  dtype=wp.float32, device="cuda")
        self.weights_wp      = wp.array(weights_list,      dtype=wp.float32, device="cuda")

    def get_inital_state(self, rng: np.random.Generator):
        q0 = np.zeros(self.mjm.nq, dtype=np.float64)
        q0[self.index_vector[0]] = 0.0                                  # cart at origin
        q0[self.index_vector[1]] = self.init_angle + rng.uniform(-0.02, 0.02)
        v0 = np.zeros(self.mjm.nv, dtype=np.float64)
        u0 = np.zeros(self.mjm.nu, dtype=np.float64)
        return q0, v0, u0

    @property
    def cost_fn_wp(self) -> wp.func:
        return cart_pole_cost_wp

    def is_success(self, mjd: mujoco.MjData) -> bool:
        x     = mjd.qpos[self.index_vector[0]]
        theta = mjd.qpos[self.index_vector[1]]
        d     = theta - UPRIGHT
        ang_err = np.arctan2(np.sin(d), np.cos(d))
        thr = self.spec.success_thresholds
        return bool(
            abs(ang_err) < thr["angle"]
            and abs(x) < thr["pos"]
            and np.linalg.norm(mjd.qvel) < thr["vel"]
        )


# ---------------------------------------------------------------------------
# Drake "real" environment + closed-loop MPPI driver
# ---------------------------------------------------------------------------
def run_cartpole_drake(
    task:        CartPoleTask,
    cfg:         ContactModelConfig,
    mppi_cfg:    MPPIConfig,
    rng:         np.random.Generator,
    sim_time:    float = 8.0,
    video_path:  str | None = None,
    fps:         float = 30.0,
):
    """Run one closed-loop episode in Drake driven by GPU-MPPI; save a video."""
    mjm = task.mjm
    mjd = task.mjd
    slider_q, hinge_q, slider_v, hinge_v = (int(i) for i in task.index_vector)

    # ---- Drake real environment ------------------------------------------
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)  # continuous
    Parser(plant).AddModels(str(SDF))
    plant.Finalize()

    # Offscreen camera in front of the scene looking toward +y, world +z up.
    R_world_camera = RotationMatrix(np.array([
        [1.0, 0.0,  0.0],
        [0.0, 0.0,  1.0],
        [0.0, -1.0, 0.0],
    ]))
    camera_pose = RigidTransform(R_world_camera, [0.0, -2.5, 0.25])
    video = VideoWriter.AddToBuilder(
        filename=video_path, builder=builder, sensor_pose=camera_pose, fps=fps,
    )

    diagram   = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()
    context       = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    cart = plant.GetJointByName("CartSlider")
    pin  = plant.GetJointByName("PolePin")

    # Initial state (matches task.get_inital_state convention).
    q0, _, _ = task.get_inital_state(rng)
    cart.set_translation(plant_context, float(q0[slider_q]))
    cart.set_translation_rate(plant_context, 0.0)
    pin.set_angle(plant_context, float(q0[hinge_q]))
    pin.set_angular_rate(plant_context, 0.0)

    act_port = plant.get_actuation_input_port()
    u = 0.0                                   # accumulated cart force (N)
    act_port.FixValue(plant_context, np.array([u]))

    # ---- Planner ---------------------------------------------------------
    controller = MPPIController(
        mjm        = mjm,
        cfg        = cfg,
        mppi_cfg   = mppi_cfg,
        cost_fn    = task.cost_fn_wp,
        goals_wp   = task.cost_goal_wp,
        idx_wp     = task.cost_idx_wp,
        weights_wp = task.cost_weights_wp,
        rng        = rng,
    )

    control_dt = mppi_cfg.substeps * mjm.opt.timestep
    n_steps    = int(sim_time / control_dt)
    log = {k: [] for k in ("t", "x", "theta", "ang_err", "u", "cost")}
    steps_to_success = None

    print(f"  control_dt={control_dt*1e3:.1f} ms   n_steps={n_steps}   "
          f"horizon={mppi_cfg.horizon}   n_samples={mppi_cfg.n_samples}")

    for t in range(n_steps):
        # 1. Read Drake state.
        x   = cart.get_translation(plant_context)
        xd  = cart.get_translation_rate(plant_context)
        th  = pin.get_angle(plant_context)
        thd = pin.get_angular_rate(plant_context)

        # 2. Mirror into MjData and plan on the GPU.
        mjd.qpos[slider_q] = x
        mjd.qpos[hinge_q]  = th
        mjd.qvel[slider_v] = xd
        mjd.qvel[hinge_v]  = thd
        mjd.ctrl[:]        = u          # MPPI plans deltas from the current force
        mujoco.mj_forward(mjm, mjd)
        action = controller.plan(mjd)

        # 3. Integrate the force delta, clamp, apply, advance Drake.
        u = float(np.clip(u + action[0], -FORCE_MAX, FORCE_MAX))
        act_port.FixValue(plant_context, np.array([u]))
        simulator.AdvanceTo((t + 1) * control_dt)

        ang_err = float(np.arctan2(np.sin(th - UPRIGHT), np.cos(th - UPRIGHT)))
        w       = task.spec.cost_weights
        cost    = (w["angle"] * ang_err**2 + w["pos"] * x**2
                   + w["cart_vel"] * xd**2 + w["pole_vel"] * thd**2)
        for k, v in zip(log, (t * control_dt, x, th, ang_err, u, cost)):
            log[k].append(v)

        if task.is_success(mjd) and steps_to_success is None:
            steps_to_success = t + 1

        if t % 25 == 0:
            print(f"  step {t:4d}  t={t*control_dt:5.2f}s  x={x:+.3f}  "
                  f"theta={th:+.3f}  ang_err={ang_err:+.3f}  u={u:+7.2f}  cost={cost:.3f}")

    # ---- Save video + results --------------------------------------------
    video.Save()
    print(f"  Saved video -> {video_path}")

    results_path = THIS_DIR / "cart_pole_results.npz"
    np.savez(results_path, **{k: np.asarray(v) for k, v in log.items()},
             success=steps_to_success is not None,
             steps_to_success=steps_to_success if steps_to_success is not None else -1)
    print(f"  Saved results -> {results_path}")

    final_ang = abs(log["ang_err"][-1])
    print(f"  success={steps_to_success is not None}  "
          f"steps_to_success={steps_to_success}  final |ang_err|={final_ang:.3f} rad")
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",       type=str,   default="M2", choices=list(MODEL_FACTORIES))
    p.add_argument("--n_samples",   type=int,   default=512)
    p.add_argument("--horizon",     type=int,   default=100)
    p.add_argument("--temperature", type=float, default=0.025)
    p.add_argument("--noise_sigma", type=float, default=0.50)
    p.add_argument("--delta",       type=float, default=10.0,
                   help="Per-step force-delta clip magnitude (N).")
    p.add_argument("--sim_time",    type=float, default=10.0)
    p.add_argument("--init_angle",  type=float, default=0.0,
                   help="Initial pole angle (rad). 0=down (swing-up), pi=upright (balance).")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--video",       type=str,   default=str(THIS_DIR / "cart_pole.gif"))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    task = CartPoleTask()
    task.init_angle = args.init_angle
    mjm, _ = task.load()
    print(f"  task=cart_pole  model={args.model}  "
          f"nq={mjm.nq} nv={mjm.nv} nu={mjm.nu}  MJCF -> {MJCF_OUT}")

    cfg = MODEL_FACTORIES[args.model]()
    mppi_cfg = MPPIConfig(
        n_samples      = args.n_samples,
        horizon        = args.horizon,
        temperature    = args.temperature,
        noise_sigma    = args.noise_sigma,
        substeps       = 1,
        warm_start     = True,
        use_full_graph = False,
        delta_range    = (-args.delta, args.delta),
        nconmax        = 12,
        njmax          = 24,
        seed           = args.seed,
        debug          = False,   # built-in debug print assumes a free-joint layout
    )

    run_cartpole_drake(
        task       = task,
        cfg        = cfg,
        mppi_cfg   = mppi_cfg,
        rng        = rng,
        sim_time   = args.sim_time,
        video_path = args.video,
    )


if __name__ == "__main__":
    main()
