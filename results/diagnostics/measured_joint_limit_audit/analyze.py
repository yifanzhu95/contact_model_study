"""Offline historical joint-feedback audit against the new manually captured ranges.

No hardware, camera, tracker or simulator is started. Only actual launcher run
controller.json files are read; copied/merged runs are deduplicated by run ID,
and repeated feedback timestamps are deduplicated within each reported cohort.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EPS = 1e-6


def write_json(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Cohort:
    def __init__(self, names, bounds):
        self.names, self.bounds = names, bounds
        self.seen, self.episodes, self.runs = set(), set(), set()
        self.n = self.duplicates = self.missing_timestamp = self.any = self.all = 0
        self.low, self.high, self.gt1, self.gt5, self.ep_bad = [np.zeros(16, dtype=int) for _ in range(5)]
        self.minimum, self.maximum = np.full(16, np.inf), np.full(16, -np.inf)
        self.max_depth = np.zeros(16)

    def add(self, observations, episode_key, run_id):
        fresh = []
        for stamp, fallback, q in observations:
            key = ('timestamp', stamp) if stamp else ('fallback', fallback)
            if key in self.seen:
                self.duplicates += 1
                continue
            self.seen.add(key)
            self.missing_timestamp += int(not stamp)
            fresh.append(q)
        if not fresh:
            return
        a = np.array(fresh)
        low = a < self.bounds[:, 0]-EPS
        high = a > self.bounds[:, 1]+EPS
        depth = np.maximum(np.maximum(self.bounds[:, 0]-a, a-self.bounds[:, 1]), 0.)
        self.n += len(a)
        self.low += low.sum(axis=0)
        self.high += high.sum(axis=0)
        self.gt1 += (depth > np.deg2rad(1.)).sum(axis=0)
        self.gt5 += (depth > np.deg2rad(5.)).sum(axis=0)
        self.ep_bad += (low | high).any(axis=0)
        self.any += int((low | high).any(axis=1).sum())
        self.all += int((low | high).all(axis=1).sum())
        self.minimum = np.minimum(self.minimum, a.min(axis=0))
        self.maximum = np.maximum(self.maximum, a.max(axis=0))
        self.max_depth = np.maximum(self.max_depth, depth.max(axis=0))
        self.episodes.add(episode_key)
        self.runs.add(run_id)

    def report(self):
        return dict(runs=len(self.runs), episodes=len(self.episodes), unique_feedback_snapshots=self.n,
            duplicate_snapshots_excluded=self.duplicates, snapshots_without_feedback_timestamp=self.missing_timestamp,
            snapshots_any_joint_outside=self.any, any_joint_outside_percent=100*self.any/self.n if self.n else None,
            total_violating_joint_samples=int((self.low+self.high).sum()),
            joints=[dict(joint=name, lower_rad=float(self.bounds[j, 0]), upper_rad=float(self.bounds[j, 1]),
                below=int(self.low[j]), above=int(self.high[j]), outside=int(self.low[j]+self.high[j]),
                outside_percent=float(100*(self.low[j]+self.high[j])/self.n) if self.n else None,
                beyond_1deg=int(self.gt1[j]), beyond_5deg=int(self.gt5[j]),
                measured_min_rad=float(self.minimum[j]) if self.n else None,
                measured_max_rad=float(self.maximum[j]) if self.n else None,
                max_excess_deg=float(np.rad2deg(self.max_depth[j])), episodes_outside=int(self.ep_bad[j]))
                for j, name in enumerate(self.names)])


def main():
    capture_path = ROOT/'scenes/leap/hardware/measured_joint_limits_capture.json'
    capture = json.loads(capture_path.read_text())
    assert capture['xml_status'] == 'all_16_joint_ranges_saved'
    model_path = ROOT/'scenes/leap/hardware/calibrated_hand_measured_limits.xml'
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == capture['output_sha256']
    names = capture['joint_order']
    bounds = np.array([capture['ranges_rad'][j] for j in names])
    cohorts = {name: Cohort(names, bounds) for name in (
        'all_completed', 'all_aborted', 'M2_completed', 'latest_completed_run', 'archived_nov15_10episodes',
        'paced_baseline_24', 'unpaced_baseline_24')}
    base = ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2'
    paced = {'leap_experiment_20261115_210143_507664', 'leap_experiment_20261115_212331_181021'}
    unpaced = {'leap_experiment_20261115_173310_247065', 'leap_experiment_20261115_173430_512823',
        'leap_experiment_20261115_173525_919954', 'leap_experiment_20261115_195200_470971',
        'leap_experiment_20261115_214159_185760'}
    archive_run = 'leap_experiment_20261115_222038_098026'
    candidates, copies, errors = {}, [], []
    for path in (ROOT/'results').rglob('controller.json'):
        if not path.parent.name.startswith('leap_experiment_') or 'code' in path.parts:
            continue
        run_id = path.parent.name
        if run_id in candidates:
            old = candidates[run_id]
            keep, skip = sorted([old, path], key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
            candidates[run_id] = keep
            copies.append(dict(run=run_id, kept=str(keep.relative_to(ROOT)), skipped=str(skip.relative_to(ROOT))))
        else:
            candidates[run_id] = path
    # Run timestamps in the launcher IDs determine chronology, rather than a
    # copied file's mtime. First run with completed measured steps is the latest.
    manifest, per_run, counts = [], [], Counter()
    latest = None
    for run_id, path in sorted(candidates.items(), reverse=True):
        try:
            payload = path.read_bytes()
            d = json.loads(payload)
        except (OSError, ValueError) as exc:
            errors.append(dict(path=str(path.relative_to(ROOT)), error=str(exc)))
            continue
        args = d.get('args', {})
        if d.get('mode') != 'lcm' and not args.get('hardware'):
            counts['non_hardware_runs_excluded'] += 1
            continue
        counts['hardware_runs_read'] += 1
        order = d.get('joint_order', names)
        if order != names:
            errors.append(dict(path=str(path.relative_to(ROOT)), error='joint order does not match capture'))
            continue
        local = Cohort(names, bounds)
        completed_episodes = aborted_episodes = published_steps = 0
        for e_index, episode in enumerate(d.get('episodes', [])):
            completed = bool(episode.get('completed', False))
            observations = []
            for s_index, step in enumerate(episode.get('steps', [])):
                if step.get('action_published') is False:
                    counts['unpublished_steps_excluded'] += 1
                    continue
                q = np.asarray(step.get('q', [])[:16], dtype=float)
                if q.shape != (16,) or not np.isfinite(q).all():
                    counts['invalid_joint_steps_excluded'] += 1
                    continue
                if 'action_published' not in step:
                    counts['legacy_steps_without_publication_flag'] += 1
                stamp = step.get('feedback', {}).get('robot_timestamp')
                stamp = int(stamp) if stamp and float(stamp) > 1e12 else None
                fallback = (run_id, e_index, s_index)
                observations.append((stamp, fallback, q))
            if not observations:
                continue
            key = (run_id, e_index)
            published_steps += len(observations)
            cohorts['all_completed' if completed else 'all_aborted'].add(observations, key, run_id)
            if not completed:
                aborted_episodes += 1
                continue
            completed_episodes += 1
            if latest is None:
                latest = run_id
            local.add(observations, key, run_id)
            if args.get('rollout_model') == 'M2':
                cohorts['M2_completed'].add(observations, key, run_id)
            for group, selected in [('latest_completed_run', run_id == latest),
                ('archived_nov15_10episodes', run_id == archive_run),
                ('paced_baseline_24', run_id in paced), ('unpaced_baseline_24', run_id in unpaced)]:
                if selected:
                    cohorts[group].add(observations, key, run_id)
        if published_steps:
            manifest.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(payload).hexdigest(),
                run=run_id, started_at_utc=d.get('started_at_utc'), model=args.get('rollout_model'),
                completed_episodes=completed_episodes, aborted_episodes=aborted_episodes,
                recorded_action_steps=published_steps, joint_limits_enforced=d.get('joint_limits') is not None))
        if local.n:
            summary = local.report()
            per_run.append(dict(run=run_id, model=args.get('rollout_model'), episodes=completed_episodes,
                unique_feedback_snapshots=local.n, any_joint_outside_percent=summary['any_joint_outside_percent'],
                **{j['joint']+'_outside_percent': j['outside_percent'] for j in summary['joints']}))
        if counts['hardware_runs_read'] % 60 == 0:
            print(f"Read {counts['hardware_runs_read']} hardware runs; {cohorts['all_completed'].n} completed-episode snapshots", flush=True)
    reports = {key: value.report() for key, value in cohorts.items()}
    for group in ('paced_baseline_24', 'unpaced_baseline_24'):
        assert reports[group]['episodes'] == 24, (group, reports[group]['episodes'])
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        limits_source=str(model_path.relative_to(ROOT)), limits_sha256=capture['output_sha256'],
        capture_saved_at=capture['saved_at'], epsilon_rad=EPS,
        denominator='unique pre-planning measured joint feedback snapshots per cohort; repeated feedback timestamps excluded',
        scope='recorded manipulation steps from direct launcher hardware runs; preparation/terminal samples and raw serial telemetry excluded',
        interpretation='Historical exceedances of the newly selected usable ranges; not evidence that the new clamp failed',
        latest_run=latest, counters=dict(counts), skipped_copies=copies, read_errors=errors,
        cohorts=reports, sources=manifest)
    write_json('report.json', report)
    write_csv('joint_exceedances.csv', [dict(cohort=name, snapshots=value['unique_feedback_snapshots'], **joint)
        for name, value in reports.items() for joint in value['joints']])
    write_csv('completed_runs.csv', per_run)
    print(json.dumps({k: {field: value[field] for field in ('runs', 'episodes', 'unique_feedback_snapshots',
        'any_joint_outside_percent')} for k, value in reports.items()}, indent=2))
    for group in ('all_completed', 'latest_completed_run', 'paced_baseline_24', 'archived_nov15_10episodes'):
        print('\n'+group)
        for row in sorted(reports[group]['joints'], key=lambda r: r['outside_percent'] or 0, reverse=True):
            print(f"{row['joint']:8s} {row['below']:7d} below {row['above']:7d} above {row['outside_percent']:7.2f}% max {row['max_excess_deg']:.2f}deg")


if __name__ == '__main__':
    main()
