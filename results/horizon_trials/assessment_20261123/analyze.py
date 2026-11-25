"""Assess recorded horizon trials only. Never constructs hardware or a planner."""
import copy
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / 'run_leap_hardware.py').exists())
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'mpl_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = ROOT / 'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/trial358_shift512_h0256_paced_measured_limits/leap_experiment_20261123_124445_598769'
spec = importlib.util.spec_from_file_location('baseline_analysis', BASE / 'analysis/analyze.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
LABELS = {'high': 'High: 4 steps', 'med': 'Medium: 8 steps', 'low': 'Low: 7 steps'}
COLORS = {'high': '#377eb8', 'med': '#e17826', 'low': '#419c64'}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_write(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def figure(fig, name):
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / 'plots' / f'{name}.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    (OUT / 'plots').mkdir(exist_ok=True)
    baseline = json.loads((BASE / 'controller.json').read_text())
    docs = {'high': baseline}
    inventory, checks = [], {}
    metrics, rows, curves, timings = {}, {}, {}, {}
    for hand in ['med', 'low']:
        paths = sorted((ROOT / 'results/horizon_trials/hardware/cube_reorient' / f'hand-{hand}_obj-high/M2').glob('*/leap_experiment_*/launch.json'))
        merged = None
        for path in paths:
            launch = json.loads(path.read_text())
            entry = dict(run=str(path.parent.relative_to(ROOT)), launch_status=launch['status'], reason=launch.get('reason'))
            controller = path.with_name('controller.json')
            if not controller.exists():
                entry['completed_episodes'] = 0
                inventory.append(entry)
                continue
            doc = json.loads(controller.read_text())
            completed = [e for e in doc['episodes'] if e.get('completed')]
            entry.update(controller_status=doc['status'], completed_episodes=len(completed),
                         controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest())
            if completed:
                entry['metrics'] = helper.collect(doc)[0]
                keys = ['weights', 'hardware_config', 'joint_reference', 'pose_transform', 'planner_config', 'joint_limits', 'success_thresholds', 'control_policy', 'fixed_goal_quat']
                checks[path.parent.name] = {k: doc[k] == baseline[k] for k in keys}
                assert all(checks[path.parent.name].values()), checks[path.parent.name]
                if merged is None:
                    merged = copy.deepcopy(doc)
                    merged['episodes'] = []
                else:
                    assert merged['rollout_config'] == doc['rollout_config']
                    assert merged['scenes']['rollout']['scene_xml'] == doc['scenes']['rollout']['scene_xml']
                for episode in completed:
                    e = copy.deepcopy(episode)
                    e.update(source_episode=e['episode'] + 1, source_run=path.parent.name, episode=len(merged['episodes']))
                    merged['episodes'].append(e)
                status = path.parent / 'bridge_status.json'
                if status.exists():
                    entry['bridge_status'] = json.loads(status.read_text())
                    entry['hold_seconds_after_controller_finish'] = (entry['bridge_status']['updated_at'] - datetime.fromisoformat(doc['finished_at_utc']).timestamp())
            inventory.append(entry)
        assert merged is not None
        docs[hand] = merged
    for hand, doc in docs.items():
        metrics[hand], rows[hand], curves[hand], timings[hand] = helper.collect(doc)
        for row in rows[hand]:
            row['hand_accuracy'] = hand
        csv_write(f'{hand}_episodes.csv', rows[hand])
        csv_write(f'{hand}_joint_clipping.csv', metrics[hand]['clipping']['joints'])

    # A timing projection only: per-plan cost need not scale exactly with horizon.
    projections = []
    for hand, old_h in [('med', 8), ('low', 7)]:
        for h in [4, 5, 6]:
            measured = timings[hand]['planning_ms']
            projected = measured * h / old_h
            overhead = max(0., metrics[hand]['publication_interval_ms']['mean'] - metrics[hand]['planning_ms']['mean'])
            projections.append(dict(hand_accuracy=hand, candidate_steps=h, horizon_s=h * .064,
                method='Proportional planning-time estimate from measured reference horizon; unvalidated',
                projected_planning_mean_ms=float(projected.mean()), projected_planning_p95_ms=float(np.percentile(projected, 95)),
                projected_paced_rate_hz=float(1000 / np.mean(np.maximum(64., projected) + overhead))))
    save('report.json', dict(baseline=str(BASE.relative_to(ROOT)), summaries=metrics, configuration_checks=checks,
        source_inventory=inventory, episodes=rows, horizon_projections=projections,
        counting_policy='Keep every completed episode, including both medium -Y attempts. Exclude runs with no started episodes from episode metrics. Low launcher timeout occurred after completion and does not invalidate its six episodes.',
        timing_definition='Frequency = within-episode publication intervals count / sum(intervals); excludes preparation, operator holds, resets and video padding.',
        limitations=['Geometry and horizon both differ from high baseline; their separate performance effects are not identified.',
            'No continuous GPU-load telemetry; launch-time GPU 2 load was 0%, same physical GPU as baseline.',
            'Clipping is controller-estimated, not exact bridge-written telemetry.',
            'Medium has seven attempts, one repeated -Y; no trial removed to make a balanced six.',
            'Projection assumes proportional planning cost; actual shorter horizons require measurement.']))
    csv_write('horizon_projections.csv', projections)
    names = ['high', 'med', 'low']
    flat = []
    for k in names:
        m = metrics[k]
        flat.append(dict(hand_accuracy=k, horizon_s=docs[k]['args']['time_horizon'], episodes=m['episodes'], successes=m['successes'], drops=m['drops'],
            planning_mean_ms=m['planning_ms']['mean'], planning_p95_ms=m['planning_ms']['p95'], planning_min_ms=m['planning_ms']['min'], planning_max_ms=m['planning_ms']['max'],
            plans_over_64ms=m['planning_over_64ms_count'], plans_over_64ms_percent=m['planning_over_64ms_percent'], pacing_wait_percent=m['paced_wait_percent'],
            publication_hz=m['publication_frequency_hz'], publication_min_hz=m['instantaneous_frequency_hz']['min'], publication_max_hz=m['instantaneous_frequency_hz']['max'],
            delta_clipped_percent=m['clipping']['delta_clipped_percent'], absolute_clipped_percent=m['clipping']['absolute_clipped_percent']))
    csv_write('metrics.csv', flat)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for k in names:
        x = np.sort(timings[k]['planning_ms'])
        axes[0].plot(x, np.arange(1, len(x) + 1) / len(x), label=LABELS[k], color=COLORS[k])
        x = np.sort(timings[k]['frequency'])
        axes[1].plot(x, np.arange(1, len(x) + 1) / len(x), label=LABELS[k], color=COLORS[k])
    axes[0].axvline(64, color='black', ls='--'); axes[0].set(xlabel='Planning time (ms)', ylabel='Fraction of plans', title='Reference horizons exceed the planning budget')
    axes[1].axvline(15.625, color='black', ls='--'); axes[1].set(xlabel='Publication frequency (Hz)', ylabel='Fraction of intervals', title='Actual within-episode control frequency')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    figure(fig, 'timing_distributions')

    rotations = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, k in zip(axes, names):
        values = [metrics[k]['rotations'][r] for r in rotations]
        heights = [v['successes'] / v['episodes'] * 100 for v in values]
        ax.bar(rotations, heights, color=COLORS[k])
        for j, (v, h) in enumerate(zip(values, heights)): ax.text(j, h+3, f"{v['successes']}/{v['episodes']}", ha='center')
        ax.set(ylim=(0, 115), title=LABELS[k], ylabel='Success rate (%)')
    figure(fig, 'success_by_rotation')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, k in enumerate(names):
        clip = metrics[k]['clipping']
        axes[0].bar(np.arange(2) + (j-1)*.25, [clip['delta_clipped_percent'], clip['absolute_clipped_percent']], width=.25, label=LABELS[k], color=COLORS[k])
    axes[0].set(xticks=[0, 1], xticklabels=['±0.2 delta bound', 'Absolute joint bounds'], ylabel='Updates clipped (%)'); axes[0].legend()
    axes[1].bar([LABELS[k] for k in names], [metrics[k]['clipping']['joint_motion_during_planning_max_per_action_deg']['mean'] for k in names], color=[COLORS[k] for k in names])
    axes[1].set(ylabel='Mean maximum-joint movement (degrees)', title='Hand movement between planning and command snapshots')
    figure(fig, 'clipping_and_state_drift')
    print(json.dumps(dict(metrics=flat, projections=projections), indent=2))


if __name__ == '__main__': main()
