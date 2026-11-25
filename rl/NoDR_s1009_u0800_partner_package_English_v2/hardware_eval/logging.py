"""Asynchronous RL results; full timestamped state and accepted targets in JSON."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading


def utc():
    return datetime.now(timezone.utc).isoformat()


def stats(values):
    return dict(count=len(values), total_s=sum(values), mean_s=sum(values)/len(values) if values else None,
                min_s=min(values) if values else None, max_s=max(values) if values else None)


def snapshot(record):
    result = dict(record)
    episodes = []
    for original in record['episodes']:
        ep = dict(original, steps=list(original['steps']))
        ep['inference_time'] = stats([r['inference_time_s'] for r in ep['steps']])
        ep['success'] = ep.get('success', ep['finish_reason'] == 'success')
        duration = ep.get('manipulation_wall_time_s')
        ep['actual_control_frequency_hz'] = ep['n_steps']/duration if duration else None
        episodes.append(ep)
    result['episodes'] = episodes
    completed = [e for e in episodes if e['completed']]
    successes = sum(e['success'] for e in completed)
    result['summary'] = dict(requested_episodes=record['args']['n_episodes'], started_episodes=len(episodes),
        completed_episodes=len(completed), successful_episodes=successes,
        unsuccessful_episodes=len(completed)-successes, aborted_episodes=len(episodes)-len(completed),
        success_rate_completed=successes/len(completed) if completed else None,
        success_rate_requested=successes/record['args']['n_episodes'],
        inference_time=stats([r['inference_time_s'] for e in episodes for r in e['steps']]),
        wall_time_s=record.get('wall_time_s'), success_thresholds=record['success_thresholds'])
    return result


class RLLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(1)
        self.error = None
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    @staticmethod
    def json(path, data):
        temp = path.with_name(path.name + '.tmp')
        with temp.open('w') as stream:
            stream.writelines(json.JSONEncoder(indent=2, allow_nan=False).iterencode(data))
            stream.write('\n')
        temp.replace(path)

    @staticmethod
    def csv(path, rows, fields):
        temp = path.with_name(path.name + '.tmp')
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        temp.replace(path)

    def save(self, data):
        root = self.path.parent
        self.json(self.path, data)
        self.json(root/'summary.json', dict(data['summary'], status=data['status'],
            started_at_utc=data['started_at_utc'], finished_at_utc=data.get('finished_at_utc'),
            policy=data['policy'], error=data.get('error')))
        steps = []
        episodes = []
        for e in data['episodes']:
            for r in e['steps']:
                steps.append(dict(episode=e['episode']+1, step=r['step'], timestamp_utc=r['timestamp_utc'],
                    elapsed_s=r['elapsed_s'], action_published=r['action_published'],
                    action_timestamp_us=r.get('action_timestamp_us'),
                    action_published_at_utc=r.get('action_published_at_utc'),
                    inference_time_s=r['inference_time_s'], pacing_wait_s=r.get('pacing_wait_s'),
                    cycle_overrun=r.get('cycle_overrun'), position_error_m=r['goal_errors']['pos'],
                    quaternion_error=r['goal_errors']['quat'], quaternion_error_rad=r['quaternion_error_rad'],
                    object_velocity_norm=r['goal_errors']['vel'],
                    robot_timestamp_us=r['feedback']['robot_timestamp'], pose_timestamp_us=r['feedback']['pose_timestamp'],
                    robot_age_s=r['feedback']['robot_age'], pose_age_s=r['feedback']['pose_age'],
                    clipped_joints=r.get('clipped_joints'), target_ack_wait_s=r.get('target_ack_wait_s')))
            episodes.append(dict(episode=e['episode']+1, finish_reason=e['finish_reason'],
                completed=e['completed'], success=e['success'], n_steps=e['n_steps'], goal_sign=e['goal_sign'],
                manipulation_wall_time_s=e.get('manipulation_wall_time_s'),
                actual_control_frequency_hz=e['actual_control_frequency_hz'],
                first_arrival_step=e.get('first_arrival_step'), successful_state_samples=e.get('successful_state_samples', 0),
                final_success=e.get('final_success'), continuous_last150_success=e.get('continuous_last150_success'),
                position_error_m=e.get('goal_errors', {}).get('pos'), quaternion_error=e.get('goal_errors', {}).get('quat'),
                inference_mean_s=e['inference_time']['mean_s']))
        self.csv(root/'steps.csv', steps, list(steps[0]) if steps else
                 ['episode','step','timestamp_utc','inference_time_s','position_error_m','quaternion_error'])
        self.csv(root/'episodes.csv', episodes, list(episodes[0]) if episodes else
                 ['episode','finish_reason','completed','success','n_steps'])

    def checkpoint(self, record):
        if self.error:
            raise OSError(f'RL result logging failed: {self.error}') from self.error
        value = snapshot(record)
        try:
            self.queue.put_nowait(value)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(value)

    def _work(self):
        while True:
            value = self.queue.get()
            if value is None:
                return
            try:
                self.save(value)
            except Exception as exc:
                self.error = exc

    def close(self, record):
        self.queue.put(snapshot(record))
        self.queue.put(None)
        self.thread.join()
        if self.error:
            raise OSError(f'RL result logging failed: {self.error}') from self.error
