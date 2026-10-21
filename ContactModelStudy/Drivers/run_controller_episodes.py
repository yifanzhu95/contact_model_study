"""Evaluate either RL or MPPI with the same finite-case driver. Never trains."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from ContactModelStudy.Drivers.ControllerEpisodes import run_controller_cases
from ContactModelStudy.Drivers.evaluate_rl import run_evaluation
from ContactModelStudy.Drivers.EvaluationVideo import add_video_arguments, video_from_arguments
from ContactModelStudy.Drivers.ContinuousGoals import add_goal_arguments, goals_from_arguments
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.Contracts import EnvConfig
from ContactModelStudy.RL.Provenance import source_runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("finite_case_bank",), default="finite_case_bank")
    parser.add_argument("--controller", choices=("rl", "mppi"), required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--planner-config", type=Path)
    parser.add_argument("--config", type=Path, help="Evaluation environment JSON; required for MPPI")
    parser.add_argument("--case-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-steps", action="store_true")
    parser.add_argument("--allow-scene-transfer", action="store_true",
                        help="RL only: allow an explicitly configured compatible scene/object transfer")
    add_video_arguments(parser)
    add_goal_arguments(parser)
    args = parser.parse_args(argv)
    video = video_from_arguments(args, parser)
    goals = goals_from_arguments(args, parser)
    if args.controller == "rl":
        if args.policy is None or args.planner_config is not None:
            parser.error("RL requires --policy and does not accept --planner-config")
    elif args.policy is not None or args.planner_config is None or args.config is None or args.allow_scene_transfer:
        parser.error("MPPI requires --planner-config and --config; no --policy")
    environment = json.loads(args.config.read_text())["environment"] if args.config else None
    bank = CaseBank.load(args.case_bank)
    if args.controller == "rl":
        result = run_evaluation(args.policy, bank, args.output, environment=environment, save_steps=args.save_steps,
                                allow_scene_transfer=args.allow_scene_transfer, video=video, goals=goals)
    else:
        from ContactModelStudy.Controllers.MPPIController import build_mppi_controller
        settings = json.loads(args.planner_config.read_text())
        config = replace(EnvConfig(**environment), num_envs=1)
        result = run_controller_cases(config, bank, args.output,
            lambda env: build_mppi_controller(settings, env), save_steps=args.save_steps, video=video, goals=goals,
            identity={"evaluation_model": config.contact_preset, "case_bank_sha256": bank.sha256},
            manifest_extra={"requested_environment": environment, "requested_planner": settings,
                "runtime": source_runtime(Path(__file__).resolve().parents[2], config.device)})
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=2))
    return result


if __name__ == "__main__": main()
