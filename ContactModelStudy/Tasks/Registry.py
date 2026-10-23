"""Shared LEAP task selection for MPC and RL, using the original task classes."""
from importlib import import_module


TASK_MODULES = {"cube_reorient": "CubeReorient", "duck_reorient": "DuckReorient",
                "ball_reorient": "BallReorient"}


def task_classes():
    return {name: getattr(import_module(f"ContactModelStudy.Tasks.{module}"), module)
            for name, module in TASK_MODULES.items()}


def make_task(name, **options):
    from ContactModelStudy.Tasks.LeapReorient import LeapReorientConfig
    if name not in TASK_MODULES:
        raise ValueError(f"Unknown task {name!r}; choose from {tuple(TASK_MODULES)}")
    module = TASK_MODULES[name]
    cls = getattr(import_module(f"ContactModelStudy.Tasks.{module}"), module)
    task = cls(LeapReorientConfig(**options))
    # A selector is not a promise that an arbitrary scene asset exists.
    from pathlib import Path
    if not Path(task.getModelPath()).is_file():
        raise FileNotFoundError(f"Task scene does not exist: {task.getModelPath()}")
    return task


def task_from_environment(config, **overrides):
    options = dict(role=config.scene_role, timestep=config.physics_dt,
                   seed=config.seed, goal_difficulty=config.goal_difficulty,
                   hand_acc=config.hand_acc, obj_acc=config.obj_acc,
                   cost_weights=config.cost_weights)
    options.update(overrides)
    return make_task(config.task_name, **options)


def available_scenes():
    """List the existing eval/rollout scenes for tasks available to MPC."""
    from ContactModelStudy.Tasks.LeapReorient import SCENES_DIR
    scenes = []
    for name in TASK_MODULES:
        obj = name.removesuffix("_reorient")
        if (SCENES_DIR/f"leap/env_leap_eval_{obj}.xml").is_file():
            scenes.append(dict(task_name=name, scene_role="eval", hand_acc="high", obj_acc="high"))
        for path in sorted((SCENES_DIR/"leap").glob(f"env_leap_rollout_{obj}_*.xml")):
            hand, accuracy = path.stem.removeprefix(f"env_leap_rollout_{obj}_").split("_", 1)
            scenes.append(dict(task_name=name, scene_role="rollout", hand_acc=hand, obj_acc=accuracy))
    return scenes
