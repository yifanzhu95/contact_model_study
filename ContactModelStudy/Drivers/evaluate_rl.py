"""Evaluate a saved actor in explicitly chosen physics; no PPO/training imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ContactModelStudy.Controllers.RLPolicyController import RLPolicyController
from ContactModelStudy.Controllers.EpisodeController import PolicyEpisodeController
from ContactModelStudy.Drivers.ControllerEpisodes import run_controller_cases
from ContactModelStudy.Drivers.EvaluationVideo import add_video_arguments, video_from_arguments
from ContactModelStudy.Drivers.ContinuousGoals import add_goal_arguments, goals_from_arguments
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv
from ContactModelStudy.RL.Provenance import source_runtime
from ContactModelStudy.RL.RunIO import (evaluation_config, validate_policy_environment,
    sha)


def run_evaluation(policy, bank, output, *, environment=None, preset=None, save_steps=False,
                   allow_scene_transfer=False, video=None, goals=None):
    policy, output = Path(policy), Path(output)
    loaded = RLPolicyController(policy)
    config = evaluation_config(loaded.manifest, environment=environment, preset=preset)
    controller = RLPolicyController(policy, device=config.device)
    identity = {"policy_sha256": sha(policy), "case_bank_sha256": bank.sha256,
        "training_model": controller.manifest["config"]["contact_preset"],
        "evaluation_model": (config.physics_backend.removesuffix("_eval")
                             if config.contact_preset == "native" else config.contact_preset)}
    return run_controller_cases(config, bank, output, lambda env: PolicyEpisodeController(controller),
        identity=identity, manifest_extra={"requested_environment": environment,
            "policy_environment": controller.manifest,
            "scene_transfer": {"allowed": allow_scene_transfer,
                               "training_task": controller.manifest["config"].get("task_name", "cube_reorient"),
                               "evaluation_task": config.task_name},
            "runtime": source_runtime(Path(__file__).resolve().parents[2], config.device)},
        validate_environment=lambda current: validate_policy_environment(
            controller.manifest, current, allow_scene_transfer=allow_scene_transfer),
        env_factory=LeapVectorEnv, save_steps=save_steps, video=video, goals=goals)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--case-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-steps", action="store_true", help="Save canonical post-step states and applied commands")
    parser.add_argument("--allow-scene-transfer", action="store_true",
                        help="Explicitly allow another scene/object/cost table; action and observation contracts must match")
    add_video_arguments(parser)
    add_goal_arguments(parser)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--config", type=Path, help="Independent JSON with environment fields")
    choice.add_argument("--contact-preset", choices=("M1", "M2", "M3", "M4", "M5"))
    choice.add_argument("--eval-sim", choices=("mujoco", "pinocchio", "drake"),
                        help="Use MPC's original CPU evaluator with its native settings")
    args = parser.parse_args(argv)
    video = video_from_arguments(args, parser)
    goals = goals_from_arguments(args, parser)
    environment = json.loads(args.config.read_text())["environment"] if args.config else None
    result = run_evaluation(args.policy, CaseBank.load(args.case_bank), args.output,
                            environment=environment, preset=args.contact_preset or args.eval_sim, save_steps=args.save_steps,
                            allow_scene_transfer=args.allow_scene_transfer, video=video, goals=goals)
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=2))
    return result


if __name__ == "__main__": main()
