"""Reproduce a derived 12-episode aggregate; no hardware/simulation or reencoding."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = next(p for p in OUT.parents if (p/'run_leap_hardware.py').exists())
NAMES = ['leap_experiment_20261123_235834_370757', 'leap_experiment_20261124_000701_785728', 'leap_experiment_20261124_001515_765303', 'leap_experiment_20261124_002536_832464', 'leap_experiment_20261124_002729_098106', 'leap_experiment_20261124_003707_366461']
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'analysis/mpl_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASELINE = ROOT/'results/bo_validation/hardware/cube_reorient/hand-high_obj-high/M2/trial358_shift512_h0256_paced_measured_limits/leap_experiment_20261123_124445_598769'
spec = importlib.util.spec_from_file_location('baseline_analysis', BASELINE/'analysis/analyze.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csv_save(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def link(source, target):
    if target.exists():
        assert target.stat().st_size == source.stat().st_size
        return
    try: os.link(source, target)
    except OSError: shutil.copy2(source, target)


def xml(text):
    tree = ET.fromstring(text)
    for mesh in tree.findall('.//mesh'): mesh.attrib.pop('content_type', None)
    return ET.tostring(tree)


def main():
    for folder in ['analysis', 'video', 'source_runs']: (OUT/folder).mkdir(exist_ok=True)
    docs = [json.loads((BASE/name/'controller.json').read_text()) for name in NAMES]
    keys = ['weights','hardware_config','joint_reference','pose_transform','planner_config','joint_limits',
            'rollout_config','success_thresholds','control_policy','fixed_goal_quat']
    checks = {k: docs[0][k] == docs[1][k] for k in keys}
    checks.update({f'{role}_scene': xml(docs[0]['scenes'][role]['scene_xml']) == xml(docs[1]['scenes'][role]['scene_xml']) for role in ['rollout','eval']})
    assert all(checks.values()), checks
    assert all(d['args']['hand_acc']=='high' and d['args']['time_horizon']==.128 and d['args']['n_samples']==512 for d in docs)
    sources, episodes, videos, per_run = [], [], [], {}
    for index, (name, doc) in enumerate(zip(NAMES, docs), 1):
        source = BASE/name
        destination = OUT/'source_runs'/f'source_{index:02d}'; destination.mkdir(exist_ok=True)
        for file in source.iterdir():
            if file.is_file(): link(file, destination/file.name)
        completed = [e for e in doc['episodes'] if e.get('completed')]
        assert completed and len(completed)==len(doc['episodes'])
        recording = json.loads((source/'video/recording.json').read_text())
        assert recording['status']=='complete' and recording['error'] is None
        for e in completed:
            new = copy.deepcopy(e); old_episode=e['episode']+1; number=len(episodes)+1
            new.update(episode=number-1, source_run=name, source_episode=old_episode)
            episodes.append(new)
            record = copy.deepcopy(next(v for v in recording['episodes'] if v['episode']==old_episode))
            assert record['error'] is None and record['outcome']==e['finish_reason']
            for field in ['video','timestamps']:
                src = source/'video'/record[field]
                suffix = '.mkv' if field=='video' else '.frames.csv'
                record[field] = f'episode_{number:02d}{suffix}'
                link(src, OUT/'video'/record[field])
            record.update(episode=number, source_run=name, source_episode=old_episode)
            videos.append(record)
        sources.append(dict(run=name, controller=str((source/'controller.json').relative_to(ROOT)),
            controller_sha256=hashlib.sha256((source/'controller.json').read_bytes()).hexdigest(),
            completed_episodes=len(completed), launcher=json.loads((source/'launch.json').read_text())['status']))
        per_run[name] = helper.collect(doc)[0]
    assert len(episodes)==12
    aggregate = copy.deepcopy(docs[0]); aggregate.update(episodes=episodes, status='complete',
        started_at_utc=docs[0]['started_at_utc'], finished_at_utc=docs[-1]['finished_at_utc'],
        wall_time_s=sum(d['wall_time_s'] for d in docs),
        aggregation_note='Derived aggregate; do not double-count with source runs. Source files remain unchanged. Wall time sums source run wall times; manipulation time is separately recorded.')
    aggregate['args'].update(n_episodes=12, results=str(OUT/'controller.json'))
    metrics, rows, curves, timing = helper.collect(aggregate)
    assert all(v['episodes']==2 for v in metrics['rotations'].values())
    planning = [s['planning_time'] for e in episodes for s in e['steps'] if s.get('action_published')]
    aggregate['summary'] = dict(requested_episodes=12, started_episodes=12, completed_episodes=12,
        successful_episodes=metrics['successes'], unsuccessful_episodes=12-metrics['successes'], aborted_episodes=0,
        success_rate_completed=metrics['success_rate'], completed_steps=metrics['actions'],
        wall_time_s=aggregate['wall_time_s'], manipulation_wall_time_s=metrics['total_manipulation_s'],
        mppi_time=dict(count=len(planning), total_s=sum(planning), mean_s=float(np.mean(planning)), min_s=min(planning), max_s=max(planning)))
    save(OUT/'controller.json', aggregate); save(OUT/'summary.json', aggregate['summary'])
    csv_save(OUT/'episodes.csv', rows)
    csv_save(OUT/'steps.csv', [dict(episode=e['episode']+1, source_run=e['source_run'], source_episode=e['source_episode'],
        step=s['step'], action_timestamp_us=s['action_timestamp_us'], planning_time_s=s['planning_time'],
        elapsed_s=s['elapsed_s'], position_error_m=s['goal_errors']['pos'], quaternion_error=s['goal_errors']['quat'],
        orientation_error_deg=float(np.degrees(s['quaternion_error_rad']))) for e in episodes for s in e['steps'] if s.get('action_published')])
    save(OUT/'sources.json', dict(sources=sources, equivalence_checks=checks, completed_episodes=12,
        note='Original runs preserved; videos and source files linked without reencoding. Do not sum aggregate and source runs.'))
    save(OUT/'video/recording.json', dict(status='complete', mode='completed_episode_aggregate', episodes=videos,
        reencoding=False, video_duration_s=sum(v['video_duration_s'] for v in videos)))
    save(OUT/'analysis/report.json', dict(metrics=metrics, episodes=rows, per_run=per_run, checks=checks))
    save(OUT/'analysis/curves.json', curves)
    csv_save(OUT/'analysis/joint_clipping.csv', metrics['clipping']['joints'])
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for name, doc in zip(NAMES, docs):
        x=np.sort([s['planning_time']*1000 for e in doc['episodes'] for s in e['steps'] if s.get('action_published')])
        axes[0].plot(x,np.arange(1,len(x)+1)/len(x),label=name.split('_')[3])
    axes[0].axvline(64,color='black',ls='--'); axes[0].set(xlabel='Planning time (ms)',ylabel='Fraction of plans',title='M1 high: 2 steps / 512 samples');axes[0].legend(title='Source run time')
    rotations=list(metrics['rotations']);success=[metrics['rotations'][r]['successes'] for r in rotations]
    axes[1].bar(rotations,success,color='#419c64');axes[1].bar(rotations,2-np.array(success),bottom=success,color='#c47d60');axes[1].set(yticks=[0,1,2],ylabel='Episodes',title='Success (green) / unsuccessful (orange)')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'analysis/timing_and_rotations.{ext}',dpi=170)
    print(json.dumps(dict(out=str(OUT),summary=aggregate['summary'],metrics={k:v for k,v in metrics.items() if k not in ['clipping']},
        latest_episodes=[{k:r[k] for k in ['episode','rotation','outcome','duration_s','final_position_error_mm','final_angle_deg','best_angle_before_below_cost_drop_height_deg']} for r in rows[6:]],clipping={k:v for k,v in metrics['clipping'].items() if k!='joints'}),indent=2))


if __name__=='__main__':main()
