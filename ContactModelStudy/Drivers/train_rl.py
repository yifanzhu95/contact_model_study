"""Single-device PPO with additional sample/time budgets and reset-based resume."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch

from ContactModelStudy.RL.Contracts import NumericalTransitionError
from ContactModelStudy.RL.Execution import BoundaryStop, require_single_process, update_budget
from ContactModelStudy.RL.LeapVectorEnv import LeapVectorEnv
from ContactModelStudy.RL.Provenance import source_runtime
from ContactModelStudy.RL.RunIO import (save_run_checkpoint, save_json, sha, close_resources, training_inputs)
from ContactModelStudy.RL.SkrlPPO import build_agent, collect_updates, restore_training_checkpoint


COUNTERS = ('completed_ppo_updates', 'vector_steps', 'transitions')


def _require_single_process_launch(parser):
    # Reject before loading checkpoints, creating output paths or importing
    # skrl: its distributed initialization would otherwise wait for all ranks.
    try:
        require_single_process()
    except ValueError as exc:
        parser.error(str(exc))


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Required for a new run; resume can use saved configuration')
    parser.add_argument('--output', type=Path, required=True, help='Fresh run directory, never the source run')
    parser.add_argument('--case-bank', type=Path)
    parser.add_argument('--updates', type=int, default=1, help='Additional PPO updates for THIS invocation; zero only with --resume')
    parser.add_argument('--resume', type=Path, help='Existing training.pt; episodes restart, partial rollouts are discarded')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Save every N completed updates, zero disables periodic saves')
    parser.add_argument('--max-transitions', type=int, help='Additional sample cap for this invocation; whole PPO updates only')
    parser.add_argument('--max-seconds', type=float, help='Soft wall-time budget including setup; checked between updates')
    parser.add_argument('--stop-margin-seconds', type=float, default=30., help='Reserve time for final saving; not a hard deadline guarantee')
    args = parser.parse_args(argv)
    _require_single_process_launch(parser)
    if args.updates < 0 or (args.updates == 0 and not args.resume) or args.checkpoint_interval < 0:
        parser.error('Use positive updates, or --resume --updates 0 for restore-only; checkpoint interval must be nonnegative')
    if args.config is None and args.resume is None: parser.error('A new run requires --config')
    env_config, ppo_config, bank, metadata = training_inputs(args.config, args.resume, args.case_bank)
    try:
        budget = update_budget(args.updates, env_config.num_envs, ppo_config.steps_per_update, args.max_transitions)
        stop = BoundaryStop(args.max_seconds, args.stop_margin_seconds, started=started)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    bank.save(args.output/'cases.json')
    torch.set_num_threads(1); torch.manual_seed(env_config.seed)
    prior = ({k: 0 for k in COUNTERS} if metadata is None else metadata.get('progress', {k: None for k in COUNTERS}))
    progress = dict(prior)
    env, completed_updates, vector_steps, latest = None, 0, 0, None
    cleanup_errors, closing = [], False
    def cleanup():
        nonlocal env
        resource, env = env, None
        cleanup_errors.extend(close_resources((("environment", resource),)))
    latest_progress = None
    outcomes = {k: 0 for k in ('success', 'physical_drop', 'deadline')}
    manifest = {'requested_environment': env_config.to_dict(), 'ppo': asdict(ppo_config),
        'runtime': None,
        'updates_requested': args.updates, 'updates_semantics': 'additional_updates_this_invocation',
        'budget': {**budget, 'max_seconds': args.max_seconds, 'stop_margin_seconds': args.stop_margin_seconds,
                   'time_limit_semantics': 'soft_stop_at_whole_update_boundaries'},
        'checkpoint_interval': args.checkpoint_interval, 'prior_progress': prior,
        'source_checkpoint': str(args.resume.resolve()) if args.resume else None,
        'source_checkpoint_sha256': sha(args.resume) if args.resume else None,
        'source_resume_metadata_sha256': sha(args.resume.with_suffix('.resume.pt')) if args.resume else None}
    save_json(args.output/'manifest.json', manifest)
    try:
        stop.__enter__()
        manifest['runtime'] = source_runtime(Path(__file__).resolve().parents[2], env_config.device)
        save_json(args.output/'manifest.json', manifest)
        env = LeapVectorEnv(env_config, bank)
        manifest['environment'] = env.describe()
        agent, actor = build_agent(env, ppo_config)
        manifest['resolved_skrl_ppo'] = asdict(agent.cfg)
        if args.resume:
            manifest['resume'] = restore_training_checkpoint(agent, env, ppo_config, args.resume)
        save_json(args.output/'manifest.json', manifest)
        initial = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        with ((args.output/'transitions.jsonl').open('w') as stream,
              (args.output/'ppo_updates.jsonl').open('w') as updates_stream):
            def log_step(step, tr, case_ids):
                nonlocal vector_steps
                vector_steps = step+1
                for reason in tr.reasons:
                    if reason in outcomes: outcomes[reason] += 1
                stream.write(json.dumps({'vector_step': step, 'case_ids': case_ids,
                    'domains': getattr(env, 'reset_domains', None),
                    'reward': tr.reward.flatten().tolist(), 'reasons': tr.reasons,
                    'raw_cost': tr.diagnostics['raw_cost'].tolist(),
                    'orientation_cost': tr.diagnostics['quat'].tolist(),
                    'position_error_m': tr.diagnostics['pos'].tolist()}, allow_nan=False)+'\n')
                stream.flush()
            def log_update(index, metrics):
                nonlocal completed_updates
                completed_updates = index
                # A validated update is complete even if the following reset
                # or observation fails. Saved checkpoint progress is separate.
                local = dict(completed_ppo_updates=index, vector_steps=index*ppo_config.steps_per_update,
                             transitions=index*ppo_config.steps_per_update*env.num_envs)
                progress.update({k: prior[k]+local[k] if prior[k] is not None else None for k in COUNTERS})
                updates_stream.write(json.dumps({'update': index,
                    'cumulative_update': prior['completed_ppo_updates']+index if prior['completed_ppo_updates'] is not None else None,
                    'run_seconds': time.monotonic()-started, **metrics}, allow_nan=False)+'\n')
                updates_stream.flush()
            def boundary(index):
                nonlocal latest, latest_progress
                if args.checkpoint_interval and index % args.checkpoint_interval == 0:
                    latest = save_run_checkpoint(agent, actor, env, ppo_config,
                        args.output/'checkpoints'/f'update_{index:08d}', dict(progress))
                    latest_progress = dict(progress)
            if budget['effective_updates']:
                stats = collect_updates(env, agent, updates=budget['effective_updates'], on_transition=log_step,
                                        on_update=log_update, on_boundary=boundary, should_stop=stop.reason)
            else:
                stats = {'vector_steps': 0, 'transitions': 0, 'ppo_updates': 0,
                         'episode_resets': 0, 'completed_episodes': outcomes}
        final = save_run_checkpoint(agent, actor, env, ppo_config, args.output, dict(progress))
        latest, latest_progress = final, dict(progress)
        closing = True
        cleanup()
        if cleanup_errors: raise RuntimeError(cleanup_errors[0]['message'])
        closing = False
        stop_reason = stats.get('stop_reason') or stop.reason()
        if not stop_reason and budget['effective_updates'] < args.updates:
            stop_reason = 'transition_budget'
        elapsed = time.monotonic()-started
        stats.update(actor_parameters_changed=any(not torch.equal(v, initial[k]) for k, v in actor.state_dict().items()),
            execution_status='stopped' if stop_reason else 'completed', stop_reason=stop_reason,
            purpose='restore_only' if args.updates == 0 else 'single_device_PPO_run',
            cumulative_progress=progress, latest_checkpoint=str(final.resolve()),
            prior_progress_known=all(v is not None for v in prior.values()), run_seconds=elapsed,
            end_to_end_transitions_per_second=stats['transitions']/elapsed if elapsed else 0.)
        save_json(args.output/'summary.json', stats)
        print(json.dumps(stats, indent=2)); return stats
    except (Exception, KeyboardInterrupt) as exc:
        cleanup()
        if isinstance(exc, NumericalTransitionError):
            np.savez(args.output/'fault_state.npz', **{k: getattr(exc.frame, k).cpu().numpy()
                for k in ('qpos', 'qvel', 'sites', 'object_twist')})
        save_json(args.output/'summary.json', {'execution_status': 'failed', 'exception': type(exc).__name__,
            'failure_reason': 'cleanup_failure' if closing else 'execution_failure', 'cleanup_errors': cleanup_errors,
            'message': str(exc), 'completed_ppo_updates': completed_updates, 'vector_steps': vector_steps,
            'transitions': vector_steps*env_config.num_envs, 'completed_episodes': outcomes,
            'latest_complete_checkpoint': str(latest.resolve()) if latest else None,
            'latest_checkpoint_progress': latest_progress, 'current_completed_update_progress': progress,
            'source_checkpoint': manifest['source_checkpoint']})
        raise
    finally:
        cleanup()
        stop.__exit__(None, None, None)


if __name__ == '__main__': main()
