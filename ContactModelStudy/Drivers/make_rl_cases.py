"""Create explicit cases for an existing MPC task; never runs physics or training."""
import argparse
import json
from pathlib import Path

from ContactModelStudy.RL.CaseBank import CaseBank
from ContactModelStudy.Tasks.Registry import TASK_MODULES, available_scenes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--task", choices=tuple(TASK_MODULES), default="cube_reorient")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-difficulty", type=int, choices=range(10), default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.list_scenes:
        print(json.dumps(available_scenes(), indent=2))
        return available_scenes()
    if args.output is None or args.count < 1 or args.seed < 0:
        parser.error("Provide --output, positive --count and nonnegative --seed")
    bank = CaseBank.development(count=args.count, seed=args.seed, difficulty=args.goal_difficulty,
                                task_name=args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(bank.json_text()+"\n")
    return {"task_name": args.task, "cases": len(bank.cases), "sha256": bank.sha256}


if __name__ == "__main__":
    main()
