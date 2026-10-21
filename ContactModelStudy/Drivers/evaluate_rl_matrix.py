"""Evaluate fixed policies across M1-M5 with shared cases and explicit failures."""
import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path

from ContactModelStudy.Drivers.evaluate_rl import run_evaluation
from ContactModelStudy.Drivers.EvaluationVideo import add_video_arguments, video_from_arguments
from ContactModelStudy.Drivers.ContinuousGoals import add_goal_arguments, goals_from_arguments
from ContactModelStudy.Controllers.RLPolicyController import RLPolicyController
from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.RL.RunIO import evaluation_config, validate_policy_environment, save_json, sha, with_contact_model
from ContactModelStudy.RL.Provenance import source_runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, action="append", required=True)
    parser.add_argument("--case-bank", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("M1", "M2", "M3", "M4", "M5", "mujoco", "pinocchio", "drake"),
                        default=["M1", "M2", "M3", "M4", "M5"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Common evaluation environment JSON; each cell selects its own model")
    parser.add_argument("--allow-scene-transfer", action="store_true")
    add_video_arguments(parser)
    add_goal_arguments(parser)
    args = parser.parse_args(argv)
    video = video_from_arguments(args, parser)
    goals = goals_from_arguments(args, parser)
    if len(set(args.models)) != len(args.models): parser.error("Duplicate evaluation models")
    bank = CaseBank.load(args.case_bank)
    policies = [RLPolicyController(p).manifest for p in args.policy]
    environment = json.loads(args.config.read_text())["environment"] if args.config else None
    # Reject different command clocks/observation/reward/scene contracts before
    # starting a matrix. Each cell still validates the actual compiled backend.
    for manifest in policies:
        validate_policy_environment(policies[0], manifest, allow_scene_transfer=args.allow_scene_transfer)
        evaluation_config(policies[0], environment=manifest["config"])
        if environment is not None:
            config = evaluation_config(manifest, environment=environment)
        else:
            config = evaluation_config(manifest)
        bank.require_task(config.task_name)
    args.output.mkdir(parents=True, exist_ok=False)
    cells = [{"policy_id": f"policy_{i:03d}", "policy_path": str(p.resolve()),
        "policy_sha256": sha(p), "training_model": policies[i]["config"]["contact_preset"],
        "evaluation_model": model, "execution_status": "not_executed", "allocated_cases": len(bank.cases)}
        for i, p in enumerate(args.policy) for model in args.models]
    bank.save(args.output/"cases.json")
    save_json(args.output/"manifest.json", {"schema": "leap_evaluation_matrix_v1",
        "runtime": source_runtime(Path(__file__).resolve().parents[2], "per_cell"),
        "case_bank_sha256": bank.sha256, "cells": cells,
        "requested_environment": environment, "allow_scene_transfer": args.allow_scene_transfer,
        "video": asdict(video) if video is not None else None,
        "goal_sequence": asdict(goals) if goals is not None else None,
        "failure_policy": "stop_on_first_fault_no_retry", "new_PPO_updates": 0})

    def publish():
        save_json(args.output/"summary.json", {"case_bank_sha256": bank.sha256, "cells": cells,
            "completed_cells": sum(c["execution_status"] == "completed" for c in cells),
            "failed_cells": sum(c["execution_status"] == "failed" for c in cells),
            "not_executed_cells": sum(c["execution_status"] == "not_executed" for c in cells), "new_PPO_updates": 0})
        columns = ("policy_id", "policy_sha256", "training_model", "evaluation_model", "execution_status", "allocated_cases",
            "executed_cases", "completed_cases", "failed_cases", "successes", "physical_drops", "deadlines",
            "observed_successes_per_allocated", "mean_final_angle_error_deg_completed",
            "mean_position_error_m_completed", "not_executed", "failure_reason",
            "goals_reached_completed", "goals_requested_per_case", "sequence_successes")
        with (args.output/"comparison.csv.pending").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader(); writer.writerows(cells)
        (args.output/"comparison.csv.pending").replace(args.output/"comparison.csv")

    publish()
    for cell in cells:
        output = args.output/cell["policy_id"]/cell["evaluation_model"]
        # The evaluator owns each fresh cell directory; parents are layout only.
        output.parent.mkdir(exist_ok=True)
        try:
            options = {"preset": cell["evaluation_model"]}
            if environment is not None:
                original = policies[int(cell["policy_id"].split("_")[1])]
                config = evaluation_config(original, environment=environment)
                options = {"environment": with_contact_model(config, cell["evaluation_model"]).to_dict()}
            if args.allow_scene_transfer:
                options["allow_scene_transfer"] = True
            if video is not None:
                options['video'] = video
            if goals is not None:
                options['goals'] = goals
            result = run_evaluation(cell["policy_path"], bank, output, **options)
            cell.update({k: v for k, v in result.items() if k != "environment"})
        except (Exception, KeyboardInterrupt) as exc:
            if (output/"summary.json").exists():
                cell.update(json.loads((output/"summary.json").read_text()))
                # A raised call is a failed cell even if an older evaluator
                # wrote a success summary before failing during cleanup.
                cell.update(execution_status="failed", exception=type(exc).__name__, message=str(exc))
                cell.setdefault("failure_reason", "evaluation_exit_failure")
            else:
                cell.update(execution_status="failed", completed_cases=0,
                    executed_cases=0, failed_cases=0, not_executed=len(bank.cases), failure_reason="initialization_failure",
                    exception=type(exc).__name__, message=str(exc))
            publish()
            raise
        publish()
    return cells


if __name__ == "__main__": main()
