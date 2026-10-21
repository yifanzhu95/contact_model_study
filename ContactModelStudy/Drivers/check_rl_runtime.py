"""Check a local/HPC single-device runtime; never sample actions or train."""
import argparse
import json
from pathlib import Path

from ContactModelStudy.RL.Execution import require_single_process
from ContactModelStudy.RL.Preflight import runtime_preflight
from ContactModelStudy.RL.RunIO import training_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--case-bank', type=Path)
    parser.add_argument('--check-backend', action='store_true', help='Also reset/read at most two environments; zero physics steps')
    parser.add_argument('--output', type=Path, help='Optional fresh JSON report; existing files are never overwritten')
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error('Preflight output already exists')
    try:
        require_single_process()
        config, ppo, bank, metadata = training_inputs(args.config, args.resume, args.case_bank)
        report = runtime_preflight(config, ppo, bank, check_backend=args.check_backend)
        report['resume_contract_checked'] = metadata is not None
        report['weights_optimizer_and_rng_restore_exercised'] = False
    except Exception as exc:
        report = {'execution_status': 'failed', 'exception': type(exc).__name__, 'message': str(exc),
                  'physics_steps': 0, 'training_transitions': 0, 'ppo_updates': 0}
    if args.output:
        with args.output.open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    if report['execution_status'] != 'passed':
        raise SystemExit(2)
    return report


if __name__ == '__main__':
    main()
