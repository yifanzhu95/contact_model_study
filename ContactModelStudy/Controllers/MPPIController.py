"""Adapter for the existing MPPI, executing the same finite tasks as RL."""
from dataclasses import asdict
import numpy as np
import torch


class MPPIController:
    def __init__(self, planner, task, rollout_sim):
        self.planner, self.task, self.rollout_sim = planner, task, rollout_sim
        self.last_command_info = {}

    def reset(self, case):
        self.set_goal(case)

    def set_goal(self, case):
        # setGoal refreshes the existing GPU goal buffer in place, including
        # this position; captured graphs keep pointing at the same buffer.
        self.task.params["target_pos"] = np.array(case.goal_position)
        self.task.setGoal(np.array(case.goal_quaternion))
        self.planner.Reset()
        self.last_command_info = {}

    def act(self, env, observation):
        frame = env.read_frame()
        previous = env.current_target
        out = self.planner.Plan(frame.qpos[0].cpu().numpy(), frame.qvel[0].cpu().numpy(),
                                u=previous[0].cpu().numpy())
        if not self.planner.last_plan_ok:
            raise RuntimeError("MPPI returned no valid plan")
        raw = out[0] if isinstance(out, tuple) else out
        raw = torch.as_tensor(raw, dtype=torch.float32, device=env.device).reshape(1, 16)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("Nonfinite MPPI command")
        # Shared execution bounds. Planning uses ctrl_relative with the same
        # delta limit; actuator saturation is also enforced at execution.
        bounded = torch.clamp(raw, previous-env.codec.delta, previous+env.codec.delta)
        applied = torch.clamp(bounded, env.codec.lower, env.codec.upper)
        self.last_command_info = {"raw_command": raw[0].tolist(),
            "projected_components": int((raw != applied).sum()),
            "max_projection": float((raw-applied).abs().max())}
        return applied

    def describe(self):
        return {"kind": "mppi", "implementation": "existing_SamplingBasedPlanners.MPPI",
            "planner": asdict(self.planner.config), "rollout": asdict(self.rollout_sim.config),
            "rollout_class": type(self.rollout_sim).__name__, "num_samples": self.rollout_sim.N,
            "command": "absolute_actuator_target", "execution_projection": "shared_rate_and_actuator_bounds",
            "limitation": "Rollout accumulates deltas; actuator saturation is projected at execution. "
                          "This interface is not an MPC performance qualification.",
            "state_transfer": "canonical_owned_frame_to_host_for_existing_Plan_API", "new_PPO_updates": 0}

    def close(self):
        self.rollout_sim.Close()


def build_mppi_controller(settings, env):
    """Resolve only planner choices here; task clocks and bounds come from env."""
    from ContactModelStudy.SamplingBasedPlanners.MPPI import MPPI, MPPI_Config
    from ContactModelStudy.Tasks.Registry import task_from_environment
    from ContactModelStudy.Utils.ContactModelPresets import GetContactModelSim
    unknown = set(settings)-{"rollout_model", "n_samples", "horizon", "mppi", "rollout_scene"}
    if unknown: raise ValueError(f"Unknown common-driver planner settings: {sorted(unknown)}")
    params = dict(settings.get("mppi", {}))
    reserved = {"control_mode", "delta_range", "plan_on_eval_scene"}
    if reserved & set(params):
        raise ValueError("The finite protocol owns control_mode, delta_range and plan_on_eval_scene")
    for key in ("n_samples", "horizon"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Planner {key} must be an explicit positive integer")
    if settings.get("rollout_model") not in ("M1", "M2", "M3", "M4", "M5"):
        raise ValueError("Planner rollout_model must explicitly name M1-M5")
    cfg = env.config
    params.setdefault("seed", cfg.seed)
    scene = dict(settings.get("rollout_scene", {}))
    if set(scene)-{"role", "hand_acc", "obj_acc"}:
        raise ValueError("rollout_scene accepts role, hand_acc and obj_acc for the same task object")
    task = task_from_environment(cfg, **scene)
    planner_config = MPPI_Config(control_mode="ctrl_relative", delta_range=(-env.codec.delta, env.codec.delta),
                                plan_on_eval_scene=task.role.value == "eval", **params)
    sim = GetContactModelSim(settings["rollout_model"], task.getModelPath(), N=settings["n_samples"],
        timestep=cfg.physics_dt, substeps=cfg.physics_steps, horizon=settings["horizon"])
    try:
        return MPPIController(MPPI(sim, task, planner_config), task, sim)
    except BaseException:
        sim.Close()
        raise
