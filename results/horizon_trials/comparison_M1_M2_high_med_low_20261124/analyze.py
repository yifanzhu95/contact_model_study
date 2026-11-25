"""Reproducible analysis of existing recordings. No simulator or hardware imports.

Creates a completed-only low M1 aggregate, then compares the accepted M1/M2
configurations. Original files are preserved; completed low videos are hardlinked.
"""
import copy
import csv
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / 'run_leap_hardware.py').exists())
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'mpl_cache'))
import numpy as np
from scipy.stats import fisher_exact, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = ROOT / 'results/horizon_trials/hardware/cube_reorient'
B = ROOT / 'results/bo_validation/hardware/cube_reorient'
LOW = H / 'hand-low_obj-high/M1/trial358_shift512_h0192_paced'
LOW_OUT = LOW / 'combined_12_completed_20261124_011550_014438'
SOURCES = {
    'M1_high': H / 'hand-high_obj-high/M1/trial358_shift512_h0192_paced/combined_12_completed_20261123_232724_235225',
    'M1_med': H / 'hand-med_obj-high/M1/trial358_shift512_h0192_paced/combined_12_completed_20261124_005310_011258',
    'M1_low': LOW_OUT,
    'M2_high': B / 'hand-high_obj-high/M2/trial358_shift512_h0256_paced_measured_limits/leap_experiment_20261123_124445_598769',
    'M2_med': H / 'hand-med_obj-high/M2/trial358_shift512_h0320_paced/combined_12_completed_20261123_211932_212743',
    'M2_low': H / 'hand-low_obj-high/M2/trial358_shift512_h0320_paced/combined_12_completed_20261123_220437_221201',
}
TWO = H / 'hand-high_obj-high/M1/trial358_shift512_h0128_paced/combined_12_completed_20261123_235834_20261124_003707'
ROTATIONS = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
COLORS = {'M1': '#3977ad', 'M2': '#de8836'}
ACCS = ['high', 'med', 'low']
spec = importlib.util.spec_from_file_location('recorded_analysis', SOURCES['M2_high'] / 'analysis/analyze.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    assert src.stat().st_size == dst.stat().st_size


def xml(text):
    tree = ET.fromstring(text)
    for mesh in tree.findall('.//mesh'):
        mesh.attrib.pop('content_type', None)
    return ET.tostring(tree)


def run_inventory(base):
    """Use video manifests as well as controller checkpoints to retain lost aborts."""
    records = []
    for run in sorted(base.glob('leap_experiment_*')):
        launch_file = run / 'launch.json'
        launch = json.loads(launch_file.read_text()) if launch_file.exists() else {}
        controller = run / 'controller.json'
        doc = json.loads(controller.read_text()) if controller.exists() else {}
        recording = run / 'video/recording.json'
        videos = json.loads(recording.read_text()).get('episodes', []) if recording.exists() else []
        episodes = {e['episode'] + 1: e for e in doc.get('episodes', [])}
        video_map = {v['episode']: v for v in videos}
        log = run / 'controller.log'
        errors = []
        if log.exists():
            errors = [s.strip() for s in log.read_text(errors='replace').splitlines()
                      if 'FeedbackTimingError:' in s or 'Planned state is stale' in s
                      or 'Dynamixel failure:' in s or 'Software failure: Planning feedback' in s]
        for number in sorted(set(episodes) | set(video_map)):
            e = episodes.get(number, {})
            v = video_map.get(number, {})
            completed = bool(e.get('completed') and e.get('finish_reason') in ('success', 'failure', 'timeout'))
            outcome = e.get('finish_reason') if completed else 'aborted'
            records.append(dict(source_run=run.name, source_path=str(run.relative_to(ROOT)),
                source_episode=number, outcome=outcome, completed=completed,
                rotation=e.get('starting_orientation', {}).get('rotation_label'),
                video=str((run / 'video' / v['video']).relative_to(ROOT)) if v.get('video') else None,
                launcher_reason=launch.get('reason'), recorded_error_lines=errors[-3:]))
        if not episodes and not videos:
            records.append(dict(source_run=run.name, source_path=str(run.relative_to(ROOT)),
                source_episode=None, outcome='no_recorded_episode', completed=False,
                rotation=None, video=None, launcher_reason=launch.get('reason'), recorded_error_lines=errors[-3:]))
    return records


def merge_low():
    LOW_OUT.mkdir(exist_ok=True)
    inventory = run_inventory(LOW)
    runs = sorted({r['source_run'] for r in inventory if r['completed']})
    docs = [json.loads((LOW / name / 'controller.json').read_text()) for name in runs]
    keys = ['weights', 'hardware_config', 'joint_reference', 'pose_transform', 'planner_config',
            'joint_limits', 'rollout_config', 'success_thresholds', 'control_policy',
            'fixed_goal_quat', 'temporal_plan_blending', 'effective_ctrl_range', 'failure_z']
    checks = {name: {k: doc.get(k) == docs[0].get(k) for k in keys} for name, doc in zip(runs, docs)}
    for name, doc in zip(runs, docs):
        checks[name].update({role + '_scene': xml(doc['scenes'][role]['scene_xml']) == xml(docs[0]['scenes'][role]['scene_xml']) for role in ['rollout', 'eval']})
        assert all(checks[name].values()), checks[name]
        assert doc['args']['rollout_model'] == 'M1' and doc['args']['hand_acc'] == 'low'
        assert doc['args']['n_samples'] == 512 and doc['args']['time_horizon'] == .192
    episodes, videos, manifest = [], [], []
    for name, doc in zip(runs, docs):
        run = LOW / name
        recording = json.loads((run / 'video/recording.json').read_text())
        assert recording['status'] == 'complete' and recording.get('error') is None
        for filename in ['controller.json', 'launch.json', 'controller.log', 'bridge.log', 'summary.json',
                         'episode_status.json', 'bridge_status.json', 'goal.json', 'steps.csv', 'episodes.csv']:
            p = run / filename
            if p.exists():
                link(p, LOW_OUT / 'source_runs' / name / filename)
        for e in doc['episodes']:
            if not e.get('completed'):
                continue
            assert e['finish_reason'] in ('success', 'failure', 'timeout')
            assert len(e['steps']) == e['n_steps']
            old_number, number = e['episode'] + 1, len(episodes) + 1
            new = copy.deepcopy(e)
            new.update(episode=number - 1, source_run=name, source_episode=old_number)
            episodes.append(new)
            record = copy.deepcopy(next(v for v in recording['episodes'] if v['episode'] == old_number))
            assert record['outcome'] == e['finish_reason'] and record.get('error') is None
            for field in ['video', 'timestamps']:
                suffix = '.mkv' if field == 'video' else '.frames.csv'
                src = run / 'video' / record[field]
                dst = LOW_OUT / 'video' / f'episode_{number:02d}{suffix}'
                link(src, dst)
                record[field] = dst.name
            record.update(episode=number, source_run=name, source_episode=old_number)
            videos.append(record)
        manifest.append(dict(source_run=name, controller=str((run / 'controller.json').relative_to(ROOT)),
            sha256=hashlib.sha256((run / 'controller.json').read_bytes()).hexdigest(),
            completed_episodes=sum(e.get('completed', False) for e in doc['episodes'])))
    assert len(episodes) == 12
    aggregate = copy.deepcopy(docs[0])
    aggregate.update(status='complete', episodes=episodes,
        finished_at_utc=docs[-1].get('finished_at_utc'),
        wall_time_s=sum(d.get('wall_time_s', 0) for d in docs),
        aggregation_note='Derived completed-only dataset. Do not count with raw sources. Run wall time sums saved checkpoint values, which may omit shutdown after an abort; manipulation time is complete.')
    aggregate['args'].update(n_episodes=12, results=str(LOW_OUT / 'controller.json'))
    metrics, rows, curves, timing = helper.collect(aggregate)
    assert all(v['episodes'] == 2 for v in metrics['rotations'].values())
    aggregate['summary'] = {k: metrics[k] for k in ['episodes', 'successes', 'drops', 'timeouts', 'actions', 'total_manipulation_s', 'success_rate']}
    save(LOW_OUT / 'controller.json', aggregate)
    save(LOW_OUT / 'summary.json', aggregate['summary'])
    save(LOW_OUT / 'sources.json', dict(sources=manifest, equivalence_checks=checks,
        excluded_attempts=[r for r in inventory if not r['completed']],
        note='12 distinct completed episodes; four recorded aborts and one permission launch failure remain separate. Source files unchanged.'))
    save(LOW_OUT / 'video/recording.json', dict(status='complete', episodes=videos, reencoding=False))
    csv_save(LOW_OUT / 'episodes.csv', rows)
    csv_save(LOW_OUT / 'analysis/joint_clipping.csv', metrics['clipping']['joints'])
    save(LOW_OUT / 'analysis/report.json', dict(metrics=metrics, episodes=rows, configuration_checks=checks))
    save(LOW_OUT / 'analysis/curves.json', curves)
    return inventory


def wilson(k, n):
    z = 1.95996398454
    p, a = k / n, 1 + z*z/n
    center = (p + z*z/(2*n)) / a
    width = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / a
    return [float(center-width), float(center+width)]


def figure(fig, name):
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / 'plots' / f'{name}.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def relative(path):
    return os.path.relpath(path, OUT)


def main():
    low_inventory = merge_low()
    (OUT / 'plots').mkdir(exist_ok=True)
    docs = {k: json.loads((path / 'controller.json').read_text()) for k, path in SOURCES.items()}
    common_keys = ['weights', 'hardware_config', 'joint_reference', 'pose_transform', 'planner_config',
                   'joint_limits', 'success_thresholds', 'control_policy', 'fixed_goal_quat',
                   'temporal_plan_blending', 'effective_ctrl_range', 'failure_z']
    checks, metrics, episodes, curves, timing = {}, {}, {}, {}, {}
    all_rows, source_manifest = [], {}
    for name, doc in docs.items():
        checks[name] = {k: doc.get(k) == docs['M1_high'].get(k) for k in common_keys}
        assert all(checks[name].values()), checks[name]
        assert doc['args']['pace_control'] and doc['args']['n_samples'] == 512
        model, accuracy = name.split('_')
        assert doc['args']['rollout_model'] == model and doc['args']['hand_acc'] == accuracy
        metrics[name], episodes[name], curves[name], timing[name] = helper.collect(doc)
        m = metrics[name]
        assert m['episodes'] == 12 and all(v['episodes'] == 2 for v in m['rotations'].values())
        m['success_wilson_95'] = wilson(m['successes'], 12)
        for e in episodes[name]:
            e.update(group=name, model=model, hand_accuracy=accuracy,
                horizon_s=doc['args']['time_horizon'],
                video=relative(SOURCES[name] / 'video' / f'episode_{e["episode"]:02d}.mkv'))
            assert (OUT / e['video']).exists()
            all_rows.append(e)
        source_manifest[name] = dict(controller=str((SOURCES[name] / 'controller.json').relative_to(ROOT)),
            sha256=hashlib.sha256((SOURCES[name] / 'controller.json').read_bytes()).hexdigest(),
            horizon_s=doc['args']['time_horizon'], rollout_config=doc['rollout_config'])
    pooled, pooled_rows = {}, {}
    for model in ['M1', 'M2']:
        doc = copy.deepcopy(docs[model + '_high'])
        doc['episodes'] = [e for acc in ACCS for e in docs[model + '_' + acc]['episodes']]
        pooled[model] = helper.collect(doc)[0]
        pooled[model]['success_wilson_95'] = wilson(pooled[model]['successes'], 36)
        pooled_rows[model] = [e for e in all_rows if e['model'] == model]
    two_doc = json.loads((TWO / 'controller.json').read_text())
    two_metrics, two_rows, two_curves, two_timing = helper.collect(two_doc)
    attempts = {name: run_inventory(SOURCES[name].parent) for name in ['M1_high', 'M1_med', 'M1_low']}
    # M2 high source is raw, unlike the other five derived sources.
    for name in ['M2_high', 'M2_med', 'M2_low']:
        attempts[name] = run_inventory(SOURCES[name].parent)
    two_attempts = run_inventory(TWO.parent)
    inventory_rows = [dict(group=k, **row) for k, rows in attempts.items() for row in rows]
    csv_save(OUT / 'attempt_inventory.csv', inventory_rows)
    save(OUT / 'attempt_inventory.json', dict(selected_configurations=attempts, M1_high_two_step=two_attempts))
    fields = {
        'Completed episodes': lambda m: m['episodes'],
        'Successes': lambda m: m['successes'], 'Success rate (%)': lambda m: m['success_rate']*100,
        'Drops': lambda m: m['drops'], 'Timeouts': lambda m: m['timeouts'],
        'Published actions': lambda m: m['actions'],
        'Manipulation wall time (s)': lambda m: m['total_manipulation_s'],
        'Successful duration mean (s)': lambda m: m['successful_duration_s']['mean'],
        'Successful duration median (s)': lambda m: m['successful_duration_s']['median'],
        'Successful endpoint position mean (mm)': lambda m: m['successful_final_pos_mm']['mean'],
        'Successful endpoint angle mean (deg)': lambda m: m['successful_final_angle_deg']['mean'],
        'All endpoint position mean (mm)': lambda m: m['all_final_pos_mm']['mean'],
        'All endpoint angle mean (deg)': lambda m: m['all_final_angle_deg']['mean'],
        'Planning min (ms)': lambda m: m['planning_ms']['min'],
        'Planning mean (ms)': lambda m: m['planning_ms']['mean'],
        'Planning median (ms)': lambda m: m['planning_ms']['median'],
        'Planning p95 (ms)': lambda m: m['planning_ms']['p95'],
        'Planning p99 (ms)': lambda m: m['planning_ms']['p99'],
        'Planning max (ms)': lambda m: m['planning_ms']['max'],
        'Plans over 64 ms (%)': lambda m: m['planning_over_64ms_percent'],
        'Pacing wait (%)': lambda m: m['paced_wait_percent'],
        'Pacing wait total (s)': lambda m: m['paced_wait_total_s'],
        'Publication rate (Hz; inverse mean interval)': lambda m: m['publication_frequency_hz'],
        'Instantaneous frequency min (Hz)': lambda m: m['instantaneous_frequency_hz']['min'],
        'Instantaneous frequency max (Hz)': lambda m: m['instantaneous_frequency_hz']['max'],
        'Instantaneous frequency arithmetic mean (Hz)': lambda m: m['instantaneous_frequency_hz']['mean'],
        'Delta-clipped updates (%)': lambda m: m['clipping']['delta_clipped_percent'],
        'Absolute-clipped updates (%)': lambda m: m['clipping']['absolute_clipped_percent'],
        'Either clipping (%)': lambda m: m['clipping']['either_clipped_percent'],
        'Final targets outside captured absolute limits': lambda m: m['clipping']['final_targets_outside_absolute_limits'],
        'Target-change L2 RMS mean (rad)': lambda m: m['mean_episode_target_change_l2_rms_rad'],
        'Max-joint movement during planning mean (deg)': lambda m: m['clipping']['joint_motion_during_planning_max_per_action_deg']['mean'],
        'Command joint age mean (ms)': lambda m: m['command_joint_age_ms']['mean'],
        'Command pose age mean (ms)': lambda m: m['command_pose_age_ms']['mean'],
        'Initial angle mean (deg)': lambda m: m['initial_angle_deg']['mean'],
        'Initial position error mean (mm)': lambda m: m['initial_position_mm']['mean'],
    }
    table = [dict(metric='Horizon (s)', **{k: d['args']['time_horizon'] for k, d in docs.items()})]
    table.extend(dict(metric=label, **{k: fn(m) for k, m in metrics.items()}) for label, fn in fields.items())
    csv_save(OUT / 'metrics.csv', table)
    csv_save(OUT / 'pooled_metrics.csv', [dict(metric=label, **{k: fn(m) for k, m in pooled.items()}) for label, fn in fields.items()])
    csv_save(OUT / 'episodes.csv', all_rows)
    csv_save(OUT / 'failures.csv', [e for e in all_rows if e['outcome'] != 'success'])
    csv_save(OUT / 'joint_clipping.csv', [dict(group=k, **j) for k, m in metrics.items() for j in m['clipping']['joints']])
    csv_save(OUT / 'rotation_successes.csv', [dict(rotation=r, **{k:f'{m["rotations"][r]["successes"]}/2' for k,m in metrics.items()}) for r in ROTATIONS])
    csv_save(OUT / 'M1_supplemental_horizon_metrics.csv', [dict(metric=label, two_step=fn(two_metrics), three_step=fn(metrics['M1_high'])) for label,fn in fields.items()])
    tests = {acc: float(fisher_exact([[metrics['M1_' + acc]['successes'], 12 - metrics['M1_' + acc]['successes']],
                                    [metrics['M2_' + acc]['successes'], 12 - metrics['M2_' + acc]['successes']]]).pvalue) for acc in ACCS}
    tests['pooled'] = float(fisher_exact([[pooled['M1']['successes'], 36-pooled['M1']['successes']],
                                       [pooled['M2']['successes'], 36-pooled['M2']['successes']]]).pvalue)
    correlations = {}
    for model in ['M1', 'M2']:
        rows = pooled_rows[model]
        correlations[model] = {}
        for field in ['delta_clipped_percent', 'absolute_clipped_percent', 'mean_plan_ms', 'initial_position_error_mm']:
            result = spearmanr([r[field] for r in rows], [int(r['outcome'] == 'success') for r in rows])
            correlations[model][field] = dict(rho=float(result.statistic), p=float(result.pvalue))
    attempt_counts = {k: {outcome: sum(r['outcome']==outcome for r in rows)
                         for outcome in ['success','failure','timeout','aborted','no_recorded_episode']}
                      for k,rows in attempts.items()}
    report = dict(groups=metrics, pooled=pooled, episodes=all_rows, common_configuration_checks=checks,
        source_manifest=source_manifest, fisher_two_sided_p=tests,
        attempt_counts=attempt_counts,
        rollout_preset_differences={k:{f:[docs['M1_high']['rollout_config'].get(f), d['rollout_config'].get(f)]
            for f in set(docs['M1_high']['rollout_config']) | set(d['rollout_config'])
            if docs['M1_high']['rollout_config'].get(f)!=d['rollout_config'].get(f)} for k,d in docs.items()},
        exploratory_episode_success_correlations=correlations,
        supplemental_M1_high_two_step=dict(source=str(TWO.relative_to(ROOT)), metrics=two_metrics),
        limitations=[
            'Independent manual placements, balanced signed rotation categories; not paired physical states or randomized crossover.',
            'M1 uses three horizon steps at all accuracies. M2 uses four at high and five at medium/low. Model, solver/contact preset and horizon effects are confounded.',
            'Trial 358 weights/settings were tuned for M2, not optimized independently for M1. This is a transfer/configuration comparison, not best-possible model performance.',
            'Completed-only outcomes exclude aborts. Separate attempt inventory retains resource/software/motor errors and pre-episode launch failures.',
            'Timing distributions describe published actions in completed episodes; rejected plans and aborted attempts are excluded. Rare tail risk is understated by these distributions.',
            'A drop means the recorded failure condition fired; detailed visual checks exist for selected cases, not every episode.',
            'Success ends at the first qualifying measured state, with no consecutive-step hold. Successful error/duration means condition on different sets of successes.',
            'Clipping/target changes use controller estimates, not exact bridge-written targets. Depth discontinuities are suspected in some traces but not automatically removed.',
            'Pooled rate is inverse mean within-episode publication interval, not mean instantaneous frequency. Manipulation excludes placement, reset, padding and recovery.',
            'Only 12 completed episodes per cell, two per signed rotation. Correlations are exploratory and affected by episode duration and outcome.',
        ])
    save(OUT / 'report.json', report)
    save(OUT / 'sources.json', source_manifest)
    save(OUT / 'curves.json', curves)
    # Main success profile, with intervals indicating small-sample uncertainty.
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for j, model in enumerate(['M1', 'M2']):
        for i, acc in enumerate(ACCS):
            m = metrics[model + '_' + acc]
            x, rate = i + (j-.5)*.34, 100*m['success_rate']
            lo, hi = np.array(m['success_wilson_95'])*100
            ax.bar(x, rate, .32, color=COLORS[model], label=model if i == 0 else None)
            ax.errorbar(x, rate, yerr=[[rate-lo], [hi-rate]], color='black', capsize=4)
            ax.text(x, rate+2, f'{m["successes"]}/12', ha='center')
    ax.set(xticks=range(3), xticklabels=['High', 'Medium', 'Low'], ylim=(0, 108), ylabel='Success rate (%)', title='M1 versus M2 hardware: 95% Wilson intervals')
    ax.legend()
    figure(fig, 'success_rates')
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, model in zip(axes, ['M1', 'M2']):
        values = np.array([[metrics[model+'_'+acc]['rotations'][r]['successes'] for r in ROTATIONS] for acc in ACCS])
        ax.imshow(values, vmin=0, vmax=2, cmap='RdYlGn', aspect='auto')
        for i in range(3):
            for j in range(6): ax.text(j, i, f'{values[i,j]}/2', ha='center', va='center')
        ax.set(xticks=range(6), xticklabels=ROTATIONS, yticks=range(3), yticklabels=['High','Medium','Low'], title=model+' success by starting rotation')
    figure(fig, 'success_by_rotation')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for j, model in enumerate(['M1', 'M2']):
        for i, acc in enumerate(ACCS):
            m=metrics[model+'_'+acc]; x=i+(j-.5)*.34
            axes[0].bar(x, m['drops'], .32, color=COLORS[model], label=model if i==0 else None)
            axes[0].bar(x, m['timeouts'], .32, bottom=m['drops'], color=COLORS[model], alpha=.45, hatch='//')
            axes[1].scatter(x, m['publication_frequency_hz'], color=COLORS[model], s=80, label=model if i==0 else None)
    axes[0].set(ylabel='Unsuccessful episodes / 12', title='Solid: drops; hatched: timeouts')
    axes[1].set(ylabel='Publication rate (Hz)', title='Inverse mean within-episode interval')
    axes[1].axhline(15.625, color='black', ls='--', lw=.8)
    for ax in axes: ax.set(xticks=range(3), xticklabels=['High','Medium','Low']);ax.legend()
    figure(fig, 'outcomes_and_control_rate')
    for name, field, limit, label in [('planning_cdf','planning_ms',64,'Planning time (ms)'), ('frequency_cdf','frequency',15.625,'Instantaneous publication frequency (Hz)')]:
        fig, axes=plt.subplots(1,3,figsize=(14,4))
        for ax, acc in zip(axes, ACCS):
            for model in ['M1','M2']:
                vals=np.sort(timing[model+'_'+acc][field]);ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),label=model,color=COLORS[model])
            ax.axvline(limit,color='black',ls='--',lw=.8);ax.set(title=acc.capitalize(),xlabel=label,ylabel='Cumulative fraction');ax.legend();ax.grid(alpha=.2)
        figure(fig,name)
    fig, axes=plt.subplots(1,3,figsize=(14,4))
    for ax, acc in zip(axes,ACCS):
        for j, model in enumerate(['M1','M2']):
            m=metrics[model+'_'+acc]
            ax.bar(np.arange(3)+(j-.5)*.34,[m['clipping'][f] for f in ['delta_clipped_percent','absolute_clipped_percent','either_clipped_percent']],.32,label=model,color=COLORS[model])
        ax.set(title=acc.capitalize(),xticks=range(3),xticklabels=['±0.2 delta','Joint limits','Either'],ylabel='Updates clipped (%)',ylim=(0,100));ax.legend()
    figure(fig,'hardware_clipping')
    for name, field, label, threshold in [('successful_position_error','final_position_error_mm','Position error at success (mm)',20),('successful_orientation_error','final_angle_deg','Orientation error at success (deg)',np.degrees(2*np.arcsin(.2))),('successful_duration','duration_s','Duration to success (s)',None)]:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,acc in zip(axes,ACCS):
            for j,model in enumerate(['M1','M2']):
                vals=[e[field] for e in episodes[model+'_'+acc] if e['outcome']=='success']
                ax.scatter(j+np.linspace(-.09,.09,len(vals)),vals,color=COLORS[model],alpha=.8)
                ax.plot([j-.17,j+.17],[np.median(vals)]*2,color='black')
            if threshold is not None:ax.axhline(threshold,color='black',ls='--',lw=.8)
            ax.set(title=acc.capitalize(),xticks=[0,1],xticklabels=['M1','M2'],ylabel=label)
        figure(fig,name)
    for field, label, threshold in [('angle_deg','Orientation error (deg)',np.degrees(2*np.arcsin(.2))),('position_mm','Position error (mm)',20)]:
        fig,axes=plt.subplots(4,3,figsize=(13,11))
        for ax,e,c in zip(axes.flat,episodes['M1_low'],curves['M1_low']):
            ax.plot(c['time_s'],c[field],color='#398a60' if e['outcome']=='success' else '#b85343')
            ax.axhline(threshold,color='black',ls='--',lw=.7)
            ax.set(title=f'E{e["episode"]}: {e["rotation"]} {e["outcome"]}',xlabel='Manipulation time (s)',ylabel=label);ax.grid(alpha=.2)
        figure(fig,'M1_low_episode_'+field)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    fails=[(e,c) for e,c in zip(episodes['M1_low'],curves['M1_low']) if e['outcome']=='timeout']
    for (a,b),(e,c) in zip(axes,fails):
        a.plot(c['time_s'],c['angle_deg']);a.axhline(np.degrees(2*np.arcsin(.2)),ls='--',color='black');a.set(title=f'E{e["episode"]} {e["rotation"]}: timeout',ylabel='Orientation error (deg)',xlabel='Time (s)')
        for j,xyz in enumerate('xyz'):b.plot(c['time_s'],np.array(c['position_axes_mm'])[:,j],label=xyz)
        b.axhline(0,color='black',ls='--',lw=.7);b.set(ylabel='Object minus goal (mm)',xlabel='Time (s)');b.legend()
    figure(fig,'M1_low_timeouts')
    # Supplemental horizon tests are deliberately outside the primary 72 episodes.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    variants=[two_metrics,metrics['M1_high']]
    axes[0].bar(['2 steps','3 steps'],[m['successes']/12*100 for m in variants],color=['#8299ad',COLORS['M1']]);axes[0].set(ylabel='Success rate (%)',ylim=(0,100),title='Supplemental M1 high horizon tests')
    axes[1].bar(['2 steps','3 steps'],[m['publication_frequency_hz'] for m in variants],color=['#8299ad',COLORS['M1']]);axes[1].axhline(15.625,color='black',ls='--');axes[1].set(ylabel='Publication rate (Hz)')
    figure(fig,'M1_supplemental_two_vs_three_steps')
    from plot_horizons import make_plot
    make_plot()
    from plot_horizon_timing import make_plots, update_index
    make_plots()
    # Browsable report; all video links point to preserved source/aggregate files.
    body=['<!doctype html><meta charset="utf-8"><title>M1 and M2 hardware comparison</title>',
        '<style>body{font:16px sans-serif;max-width:1450px;margin:30px auto;line-height:1.5}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}img{max-width:100%}</style>',
        '<h1>M1 and M2: high, medium and low hand accuracy</h1>',
        '<p>72 distinct completed hardware episodes, 12 per cell. Supplemental M1 two-step tests and aborted attempts are kept separate. M1: three-step horizon; M2: four steps at high, five at medium/low. Trial 358 shared settings.</p>',
        '<p><a href="FINDINGS.txt">Interpretation</a> · <a href="metrics.csv">Full six-column metrics</a> · <a href="pooled_metrics.csv">Pooled model metrics</a> · <a href="episodes.csv">Episode metrics</a> · <a href="attempt_inventory.csv">Attempt inventory</a> · <a href="report.json">Full report</a></p>',
        '<table><tr><th>Metric</th>'+''.join('<th>'+k+'</th>' for k in docs)+'</tr>']
    for row in table:
        body.append('<tr><td>'+html.escape(row['metric'])+'</td>'+''.join('<td>'+(f'{row[k]:.3f}' if isinstance(row[k],float) else str(row[k]))+'</td>' for k in docs)+'</tr>')
    body.append('</table>')
    for p in sorted((OUT/'plots').glob('*.png')):body.append(f'<h2>{p.stem.replace("_"," ")}</h2><a href="plots/{p.stem}.pdf">PDF</a><br><img src="plots/{p.name}">')
    body.append('<h2>Episodes and videos</h2><table><tr><th>Group</th><th>Episode</th><th>Rotation</th><th>Outcome</th><th>Video</th></tr>')
    for e in all_rows:body.append(f'<tr><td>{e["group"]}</td><td>{e["episode"]}</td><td>{e["rotation"]}</td><td>{e["outcome"]}</td><td><a href="{html.escape(e["video"])}">Video</a></td></tr>')
    body.append('</table><h2>Limitations</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in report['limitations'])+'</ul>')
    (OUT/'index.html').write_text('\n'.join(body))
    update_index()
    (LOW_OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>M1 low: 12 completed episodes</h1><p><a href="'+os.path.relpath(OUT/'index.html',LOW_OUT)+'">Full analysis and comparison</a></p><p><a href="episodes.csv">Episodes</a> · <a href="analysis/report.json">Metrics</a> · <a href="sources.json">Sources and exclusions</a></p>')
    print(json.dumps(dict(output=str(OUT),low_aggregate=str(LOW_OUT),
        metrics={k:{f:m[f] for f in ['successes','drops','timeouts','actions','total_manipulation_s','publication_frequency_hz','planning_ms','planning_over_64ms_percent','paced_wait_percent','rotations']} for k,m in metrics.items()},
        pooled={k:{f:m[f] for f in ['successes','drops','timeouts','publication_frequency_hz','planning_ms','rotations']} for k,m in pooled.items()},tests=tests,
        low_failures=[e for e in episodes['M1_low'] if e['outcome']!='success'],
        inventories={k:{outcome:sum(r['outcome']==outcome for r in rows) for outcome in ['success','failure','timeout','aborted','no_recorded_episode']} for k,rows in attempts.items()}),indent=2))


if __name__ == '__main__':
    main()
