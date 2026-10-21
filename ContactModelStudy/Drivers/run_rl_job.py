"""Select one independent bounded run from a job list; default is plan-only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
import time

from ContactModelStudy.Drivers import train_rl
from ContactModelStudy.RL.Execution import BoundaryStop, require_single_process, update_budget
from ContactModelStudy.RL.Preflight import runtime_preflight
from ContactModelStudy.RL.RunIO import save_json, sha, training_inputs


def plan_job(plan_path, output_root, *, job_id=None, job_index=None):
    plan_path, output_root = Path(plan_path).resolve(), Path(output_root).resolve()
    plan = json.loads(plan_path.read_text())
    if plan.get('schema') != 'rl_independent_jobs_v1' or not isinstance(plan.get('jobs'), list) or not plan['jobs']:
        raise ValueError('Expected a nonempty rl_independent_jobs_v1 jobs list')
    jobs, ids = plan['jobs'], []
    allowed = {'id', 'config', 'case_bank', 'resume', 'updates', 'max_transitions',
               'max_seconds', 'stop_margin_seconds', 'checkpoint_interval'}
    for job in jobs:
        if (not isinstance(job, dict) or set(job)-allowed or not isinstance(job.get('id'), str)
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', job['id'])):
            raise ValueError('Jobs require safe unique ids and supported fields only')
        ids.append(job['id'])
    if len(set(ids)) != len(ids):
        raise ValueError('Job ids must be unique')
    if job_id is not None:
        if job_id not in ids:
            raise ValueError('Unknown job id')
        job_index = ids.index(job_id)
    if job_index is None or not 0 <= job_index < len(jobs):
        raise ValueError('job_index must select an existing zero-based job')
    job = jobs[job_index]
    def path(key):
        if key not in job:
            return None
        value = Path(job[key])
        return (value if value.is_absolute() else plan_path.parent/value).resolve()
    config_path, resume_path, bank_path = path('config'), path('resume'), path('case_bank')
    if resume_path is None and bank_path is None:
        raise ValueError('Independent new runs require an explicit case_bank')
    if 'updates' not in job or job.get('max_seconds') is None:
        raise ValueError('Every selected job requires explicit updates and max_seconds budgets')
    config, ppo, bank, metadata = training_inputs(config_path, resume_path, bank_path)
    if job['updates'] == 0 and not resume_path:
        raise ValueError('Zero updates are only for restore-only jobs')
    budget = update_budget(job['updates'], config.num_envs, ppo.steps_per_update, job.get('max_transitions'))
    BoundaryStop(job['max_seconds'], job.get('stop_margin_seconds', 30.))
    interval = job.get('checkpoint_interval', 10)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise ValueError('checkpoint_interval must be a nonnegative integer')
    job_dir = output_root/job['id']
    args = ['--output', str(job_dir/'run'), '--updates', str(job['updates']),
            '--max-seconds', str(job['max_seconds']), '--stop-margin-seconds', str(job.get('stop_margin_seconds', 30.)),
            '--checkpoint-interval', str(interval)]
    for flag, value in (('--config', config_path), ('--resume', resume_path), ('--case-bank', bank_path),
                        ('--max-transitions', job.get('max_transitions'))):
        if value is not None:
            args.extend((flag, str(value)))
    report = {'schema': plan['schema'], 'mode': 'plan_only', 'job_id': job['id'], 'job_index': job_index,
              'job_directory': str(job_dir), 'command': [sys.executable, '-m', 'ContactModelStudy.Drivers.train_rl', *args],
              'training_arguments': args, 'plan_sha256': sha(plan_path), 'environment': config.to_dict(),
              'ppo': asdict(ppo), 'case_bank_sha256': bank.sha256, 'budget': budget,
              'max_seconds': job['max_seconds'], 'stop_margin_seconds': job.get('stop_margin_seconds', 30.),
              'input_sha256': {str(p): sha(p) for p in (config_path, resume_path, bank_path) if p is not None}}
    return report, (config, ppo, bank)


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--job-id')
    selection.add_argument('--job-index', type=int)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Run only the selected job; no scheduler submission or automatic retry')
    args = parser.parse_args(argv)
    require_single_process()
    report, inputs = plan_job(args.plan, args.output_root, job_id=args.job_id, job_index=args.job_index)
    job_dir = Path(report['job_directory'])
    if job_dir.exists():
        raise FileExistsError('Use a fresh job id/output root; completed or failed runs are never overwritten')
    if not args.execute:
        print(json.dumps(report, indent=2, allow_nan=False))
        return report
    job_dir.mkdir(parents=True, exist_ok=False)
    report['mode'] = 'execute_selected_independent_job'
    save_json(job_dir/'job.json', report)
    try:
        preflight = runtime_preflight(*inputs)
        save_json(job_dir/'preflight.json', preflight)
        remaining = report['max_seconds'] - (time.monotonic() - started)
        if remaining <= report['stop_margin_seconds']:
            result = {'execution_status': 'stopped', 'stop_reason': 'wall_time_budget_before_training',
                      'transitions': 0, 'ppo_updates': 0, 'training_started': False}
        else:
            training_args = list(report['training_arguments'])
            training_args[training_args.index('--max-seconds')+1] = str(remaining)
            report['execution_training_arguments'] = training_args
            save_json(job_dir/'job.json', report)
            result = train_rl.main(training_args)
        save_json(job_dir/'job_summary.json', result)
        return result
    except (Exception, KeyboardInterrupt, SystemExit) as exc:
        save_json(job_dir/'job_summary.json', {'execution_status': 'failed',
                  'exception': type(exc).__name__, 'message': str(exc), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
