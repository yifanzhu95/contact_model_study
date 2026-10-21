"""Optional finite goal sequences for evaluation, separate from PPO episodes."""
from dataclasses import dataclass, replace
import hashlib
import json


@dataclass(frozen=True)
class GoalSequenceConfig:
    goals_per_episode: int = 1
    seed: int = 0

    def __post_init__(self):
        for name in ('goals_per_episode', 'seed'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name} must be an integer')
        if self.goals_per_episode < 1 or self.seed < 0:
            raise ValueError('goals_per_episode must be positive; goal seed must be nonnegative')

    @property
    def enabled(self):
        return self.goals_per_episode > 1


def add_goal_arguments(parser):
    group = parser.add_argument_group('continuous-goal evaluation (off by default)')
    group.add_argument('--goals-per-episode', type=int, default=1,
                       help='Reach this many sequential goals without resetting physics; share the original total time')
    group.add_argument('--goal-seed', type=int,
                       help='Additional goal sequence seed, independent of policy/training RNG (default: 0)')


def goals_from_arguments(args, parser):
    try:
        result = GoalSequenceConfig(args.goals_per_episode, args.goal_seed if args.goal_seed is not None else 0)
        if not result.enabled and args.goal_seed is not None:
            raise ValueError('--goal-seed requires --goals-per-episode greater than 1')
        return result if result.enabled else None
    except ValueError as exc:
        parser.error(str(exc))


def prepare_goal_sequences(config, bank, settings):
    """Precompute all targets so controller behavior cannot change future goals."""
    from ContactModelStudy.Tasks.Registry import task_from_environment
    sampler = task_from_environment(replace(config, seed=settings.seed))
    sequences = []
    for case in bank.cases:
        sequence = [case]
        for _ in range(settings.goals_per_episode - 1):
            quaternion = sampler.sampleNewGoal(current_goal=sequence[-1].goal_quaternion)
            sequence.append(replace(case, goal_quaternion=tuple(quaternion)))
        sequences.append(sequence)
    payload = {'schema': 'leap_goal_sequences_v1', 'case_bank_sha256': bank.sha256,
               'task_name': config.task_name, 'goal_difficulty': config.goal_difficulty,
               'seed': settings.seed, 'goals_per_episode': settings.goals_per_episode,
               'cases': [{'case_id': seq[0].case_id, 'goals': [
                   {'goal_position': list(c.goal_position), 'goal_quaternion_wxyz': list(c.goal_quaternion)}
                   for c in seq]} for seq in sequences]}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    protocol = {'name': 'continuous_goal_sequence_v1', 'goals_per_episode': settings.goals_per_episode,
                'goal_seed': settings.seed, 'goal_sequences_sha256': digest,
                'time_budget': 'shared_episode_steps_no_extension', 'max_control_steps': config.episode_steps,
                'remaining_time_observation': 'whole_sequence', 'success': 'all_requested_goals_reached',
                'minimum_control_steps_per_goal': 1, 'training_updates': 0}
    return sequences, payload, protocol
