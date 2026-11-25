"""Result snapshots and CSV metrics, without file I/O in the control loop."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading

import numpy as np


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def quaternion_angle(error):
    """Radians corresponding to the task's 1 - dot(q, target)^2 metric."""
    return float(2 * np.arcsin(np.sqrt(np.clip(error, 0, 1))))


def timing_stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return dict(count=0, total_s=0.0, mean_s=None, min_s=None, max_s=None)
    return dict(count=int(values.size), total_s=float(values.sum()), mean_s=float(values.mean()),
                min_s=float(values.min()), max_s=float(values.max()))


def snapshot(record):
    # Completed step dictionaries are immutable after publication. Copy the
    # mutable containers so the worker never iterates the live episode list.
    result = dict(record)
    episodes = []
    all_times = []
    for original in record['episodes']:
        episode = dict(original, steps=list(original['steps']))
        times = [step['planning_time'] for step in episode['steps']]
        if episode.get('failed_step', {}).get('planning_time') is not None:
            times.append(episode['failed_step']['planning_time'])
        episode['mppi_time'] = timing_stats(times)
        episode['success'] = episode['finish_reason'] == 'success'
        duration = episode.get('manipulation_wall_time_s')
        episode['actual_control_frequency_hz'] = episode.get('n_steps', len(episode['steps'])) / duration if duration else None
        all_times.extend(times)
        episodes.append(episode)
    completed = [episode for episode in episodes if episode.get('completed', False)]
    successes = sum(episode['success'] for episode in completed)
    result['episodes'] = episodes
    result['summary'] = dict(requested_episodes=record['args']['n_episodes'], started_episodes=len(episodes),
        completed_episodes=len(completed), successful_episodes=successes,
        unsuccessful_episodes=len(completed) - successes,
        aborted_episodes=len(episodes) - len(completed),
        success_rate_completed=successes / len(completed) if completed else None,
        success_rate_requested=successes / record['args']['n_episodes'],
        mppi_time=timing_stats(all_times), wall_time_s=record.get('wall_time_s'),
        success_thresholds=record.get('success_thresholds'))
    return result


STEP_FIELDS = ['episode', 'step', 'action_published', 'timestamp_utc', 'elapsed_s', 'wall_time_s',
               'planning_time_s', 'pacing_wait_s', 'action_published_at_utc', 'action_timestamp_us',
               'position_error_m', 'quaternion_error', 'quaternion_error_rad',
               'object_velocity_norm', 'robot_age_s', 'pose_age_s', 'error']
EPISODE_FIELDS = ['episode', 'finish_reason', 'completed', 'success', 'n_steps',
                  'manipulation_wall_time_s', 'cycle_wall_time_s', 'time_to_success_s',
                  'actual_control_frequency_hz', 'mppi_calls', 'mppi_mean_s', 'mppi_min_s',
                  'mppi_max_s', 'mppi_total_s', 'position_error_m', 'quaternion_error',
                  'quaternion_error_rad', 'object_velocity_norm']


class RunLogger:
    """Coalesce checkpoints; encode incrementally so motor heartbeats can run."""
    def __init__(self, results):
        self.path = Path(results)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pending = queue.Queue(maxsize=1)
        self.error = None
        self.thread = threading.Thread(target=self._work, name='leap-results', daemon=True)
        self.thread.start()

    def checkpoint(self, record):
        item = snapshot(record)
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.pending.put_nowait(item)

    @staticmethod
    def _json(path, record):
        pending = path.with_name(path.name + '.tmp')
        with pending.open('w') as stream:
            # iterencode avoids one large C encoder call holding the GIL while
            # the controller is keeping the hardware's watchdog alive.
            stream.writelines(json.JSONEncoder(indent=2, allow_nan=False, default=str).iterencode(record))
            stream.write('\n')
        pending.replace(path)

    @staticmethod
    def _csv(path, fields, rows):
        pending = path.with_name(path.name + '.tmp')
        with pending.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        pending.replace(path)

    def _save(self, record):
        directory = self.path.parent
        self._json(self.path, record)
        self._json(directory/'summary.json', dict(record['summary'], status=record['status'],
            started_at_utc=record['started_at_utc'], finished_at_utc=record.get('finished_at_utc'),
            args=record['args'], weights=record.get('weights'), rollout_config=record.get('rollout_config'),
            planner_config=record.get('planner_config'), warmup_wall_time_s=record.get('warmup_wall_time_s'),
            joint_reference=record.get('joint_reference'),
            error=record.get('error')))

        def steps():
            for episode in record['episodes']:
                rows = episode['steps']
                if episode.get('failed_step'):
                    rows = [*rows, episode['failed_step']]
                for step in rows:
                    errors, feedback = step.get('goal_errors', {}), step.get('feedback', {})
                    yield dict(episode=episode['episode'] + 1, step=step['step'],
                        action_published=step.get('action_published', False),
                        timestamp_utc=step.get('timestamp_utc'), elapsed_s=step.get('elapsed_s'),
                        wall_time_s=step.get('wall_time_s'), planning_time_s=step.get('planning_time'),
                        pacing_wait_s=step.get('pacing_wait_s'),
                        action_published_at_utc=step.get('action_published_at_utc'),
                        action_timestamp_us=step.get('action_timestamp_us'),
                        position_error_m=errors.get('pos'), quaternion_error=errors.get('quat'),
                        quaternion_error_rad=step.get('quaternion_error_rad'), object_velocity_norm=errors.get('vel'),
                        robot_age_s=feedback.get('robot_age'), pose_age_s=feedback.get('pose_age'), error=step.get('error'))
        self._csv(directory/'steps.csv', STEP_FIELDS, steps())
        rows = []
        for episode in record['episodes']:
            errors, times = episode.get('goal_errors', {}), episode['mppi_time']
            rows.append(dict(episode=episode['episode'] + 1, finish_reason=episode['finish_reason'],
                completed=episode.get('completed', False), success=episode['success'],
                n_steps=episode.get('n_steps', len(episode['steps'])),
                manipulation_wall_time_s=episode.get('manipulation_wall_time_s'),
                cycle_wall_time_s=episode.get('cycle_wall_time_s'), time_to_success_s=episode.get('time_to_success_s'),
                actual_control_frequency_hz=episode['actual_control_frequency_hz'],
                mppi_calls=times['count'], mppi_mean_s=times['mean_s'], mppi_min_s=times['min_s'],
                mppi_max_s=times['max_s'], mppi_total_s=times['total_s'], position_error_m=errors.get('pos'),
                quaternion_error=errors.get('quat'), quaternion_error_rad=episode.get('quaternion_error_rad'),
                object_velocity_norm=errors.get('vel')))
        self._csv(directory/'episodes.csv', EPISODE_FIELDS, rows)

    def _work(self):
        while True:
            item = self.pending.get()
            if item is None:
                return
            try:
                self._save(item)
            except Exception as exc:
                self.error = exc

    def close(self, record):
        # Called after control/feedback cleanup; flushing can now wait for disk.
        self.pending.put(snapshot(record))
        self.pending.put(None)
        self.thread.join()
        if self.error is not None:
            raise OSError(f'Result logging failed: {self.error}') from self.error
