"""Merge the selected M6 episodes and report the explicitly curated hardware comparison."""
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/leap_comparison_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ContactModelStudy.Hardware.ExperimentLogs import configuration_parameters

BASE = ROOT/'results/pilot/hardware/cube_reorient/hand-high_obj-high'
REPORT = BASE/'comparison_report_20261114'
M6 = BASE/'M6/combined_10_completed_20261114_000214_001426'
OLD = BASE/'M6/combined_1_completed_20261114_000214_299972'
NEW = BASE/'M6/leap_experiment_20261114_001426_696295'
MODELS = ['M1', 'M2', 'M3', 'M4', 'M6']
COLORS = dict(success='#26946c', failure='#cf6048', timeout='#d9a130', error='#8d6bb1')
CACHE = {}
HASHES = {}


def load(path):
    return json.loads(path.read_text())


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def csv_read(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def csv_write(path, rows, fields=None):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def controller(path):
    path = path.resolve()
    if path not in CACHE:
        HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        CACHE[path] = load(path)
    return CACHE[path]


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()), total=float(a.sum())) if a.size else None


def qangle(error):
    return float(math.degrees(2*math.asin(math.sqrt(min(1., max(0., error))))))


def main():
    assert not M6.exists() and not REPORT.exists(), 'Outputs already exist; avoid accidental overwrite'
    now = datetime.now(timezone.utc).isoformat()
    REPORT.mkdir()
    for name in ('comparison_index.json', 'comparison.csv'):
        shutil.copy2(BASE/name, REPORT/(name+'.before'))
    index = load(BASE/'comparison_index.json')
    old_sources = load(OLD/'sources.json')
    retained = [s for s in old_sources['sources'] if s['selected_source_episode_numbers']]
    assert len(retained) == 1 and retained[0]['selected_source_episode_numbers'] == [1]
    old_path = Path(retained[0]['controller_json'])
    old = controller(old_path)
    new = controller(NEW/'controller.json')
    assert len(new['episodes']) == 9 and all(e['completed'] for e in new['episodes'])
    old_launch, new_launch = load(old_path.parent/'launch.json'), load(NEW/'launch.json')
    params = [configuration_parameters(d, l) for d, l in ((old, old_launch), (new, new_launch))]
    normalized = copy.deepcopy(params)
    for p in normalized:
        p['tracking'].pop('cuda_visible_devices', None)
    assert normalized[0] == normalized[1], 'M6 configurations differ beyond vision GPU assignment'
    selections = [(old_path, [1]), (NEW/'controller.json', list(range(1, 10)))]
    selected_eps, episode_rows, step_rows, provenance = [], [], [], []
    for path, numbers in selections:
        d, launch = controller(path), load(path.parent/'launch.json')
        provenance.append(dict(run=path.parent.name, controller_json=str(path.resolve()),
            controller_sha256=HASHES[str(path.resolve())], selected_source_episode_numbers=numbers,
            configuration_id=launch['configuration_log']['configuration_id'], source_status=d['status'],
            source_error=d.get('error'), tracking_gpu=launch['config']['tracking_gpu_assignment'],
            mppi_gpu=launch['config']['gpu_assignment']))
        for n in numbers:
            ep = d['episodes'][n-1]
            assert ep['episode'] == n-1 and ep['completed']
            selected_eps.append((path.resolve(), ep))
        renumber = {n: i+1+len(episode_rows) for i, n in enumerate(numbers)}
        for kind, target in (('episodes', episode_rows), ('steps', step_rows)):
            _, rows = csv_read(path.parent/(kind+'.csv'))
            for row in rows:
                number = int(row['episode'])
                if number in renumber:
                    target.append(dict(row, source_run=path.parent.name, source_episode=number,
                        max_robot_age_s=d['args']['max_robot_age'], episode=renumber[number]))
    assert len(selected_eps) == len(episode_rows) == 10
    assert len({(str(path), e['episode']) for path, e in selected_eps}) == 10
    assert len(step_rows) == sum(e['n_steps'] for _, e in selected_eps) == 4895
    assert all(not row['error'] for row in step_rows)
    M6.mkdir()
    for name, rows in (('episodes', episode_rows), ('steps', step_rows)):
        fields = ['source_run', 'source_episode', 'max_robot_age_s']
        fields += [k for k in rows[0] if k not in fields]
        csv_write(M6/(name+'.csv'), rows, fields)
    durations = [ep['manipulation_wall_time_s'] for _, ep in selected_eps]
    plans = [step['planning_time'] for _, ep in selected_eps for step in ep['steps']]
    outcomes = [ep['finish_reason'] for _, ep in selected_eps]
    p = stats(plans)
    dump(M6/'summary.json', dict(record_type='completed_episode_aggregate', source_runs=2,
        completed_episodes=10, successful_episodes=outcomes.count('success'), failure_episodes=outcomes.count('failure'),
        timeout_episodes=outcomes.count('timeout'), success_rate_completed=outcomes.count('success')/10,
        excluded_aborted_episodes=2, completed_steps=len(step_rows), manipulation_wall_time_s=sum(durations),
        mppi_time=dict(count=p['count'], total_s=p['total'], mean_s=p['mean'],
            min_s=float(min(plans)), max_s=p['max']), source_configuration_ids=[s['configuration_id'] for s in provenance],
        scope='One retained success plus nine new completed episodes; both earlier supervision aborts excluded.',
        configuration_notes=['Planning, costs, hardware and task parameters match. Vision/MPPI physical GPU assignments differ between source runs; see sources.json.']))
    dump(M6/'sources.json', dict(created_at_utc=now, sources=provenance, excluded_episodes=old_sources['excluded_episodes'],
        reference_configuration=params[0], source_configurations=params,
        previous_retained_dataset=str(OLD), raw_sources_unchanged=True))
    m6_entry = next(d for d in index['datasets'] if d['model'] == 'M6')
    m6_entry.update(completed_episodes=10, successful_episodes=1, drops=8, timeouts=1, success_rate=.1,
        dataset=str(M6.relative_to(BASE)), summary=str((M6/'summary.json').relative_to(BASE)), evaluated_episodes=10)
    index['completed_episodes'] = sum(d['completed_episodes'] for d in index['datasets'])
    index['evaluated_episodes'] = sum(d['evaluated_episodes'] for d in index['datasets'])
    index['updated_at_utc'] = now
    index['configuration_notes'] = [n for n in index['configuration_notes'] if not n.startswith('M6 currently contains')]
    index['configuration_notes'] += [
        'M6 has ten completed episodes: retained episode 1 of 000214 and episodes 1–9 of 001426. Both earlier supervision aborts are excluded.',
        'M4 has only five unique software-failed attempts, zero completed episodes. No additional five were found; the comparison contains 40 completed manipulations plus five M4 failed attempts.',
        'Timing reflects historical hardware runs on different GPU assignments and loads; this is not a controlled backend-speed benchmark.']
    dump(BASE/'comparison_index.json', index)
    fields, _ = csv_read(REPORT/'comparison.csv.before')
    csv_write(BASE/'comparison.csv', index['datasets'], fields)

    # Resolve the selected datasets to unique raw episodes; never include configuration-log duplicates.
    all_episodes, all_metrics, manifest, configurations = [], [], [], []
    by_model = {}
    for entry in index['datasets']:
        model, path = entry['model'], BASE/entry['dataset']
        rows = []
        if (path/'sources.json').exists():
            sources = load(path/'sources.json')['sources']
            for s in sources:
                raw = Path(s['controller_json'])
                d = controller(raw)
                assert HASHES[str(raw.resolve())] == s['controller_sha256'], 'Source hash changed'
                numbers = s.get('selected_source_episode_numbers', [s.get('original_episode', 1)])
                for n in numbers:
                    rows.append((raw.resolve(), d['episodes'][n-1]))
        else:
            raw = path/'controller.json'
            d = controller(raw)
            rows = [(raw.resolve(), e) for e in d['episodes'] if e['completed']]
        assert len(rows) == entry['evaluated_episodes']
        assert len({(str(path), ep['episode']) for path, ep in rows}) == len(rows)
        per_model_plans, per_model_eps, curve_data = [], [], []
        for i, (raw, ep) in enumerate(rows, 1):
            d = controller(raw)
            assert d['args']['rollout_model'] == model
            if model != 'M4':
                assert ep['completed'] and ep['finish_reason'] in ('success', 'failure', 'timeout')
            else:
                assert not ep['completed'] and ep['finish_reason'] == 'error'
            steps = ep['steps']
            times = [s['planning_time'] for s in steps]
            if model == 'M4' and ep.get('failed_step', {}).get('planning_time') is not None:
                times.append(ep['failed_step']['planning_time'])
            per_model_plans.extend(times)
            errors = ep.get('goal_errors', {})
            start_errors = steps[0]['goal_errors']
            outcome = ep['finish_reason']
            duration = ep['manipulation_wall_time_s']
            row = dict(model=model, episode=i, source_run=raw.parent.name, source_episode=ep['episode']+1,
                completed=ep['completed'], outcome=outcome, steps=ep['n_steps'], duration_s=duration,
                effective_hz=ep['n_steps']/duration, success_time_s=ep.get('time_to_success_s'),
                initial_position_error_mm=start_errors['pos']*1000, initial_orientation_error_deg=qangle(start_errors['quat']),
                final_position_error_mm=errors.get('pos')*1000 if errors else None,
                final_orientation_error_deg=qangle(errors['quat']) if errors else None,
                final_quaternion_error=errors.get('quat'), final_velocity_norm=errors.get('vel'),
                mean_plan_ms=float(np.mean(times)*1000), p95_plan_ms=float(np.percentile(times,95)*1000))
            per_model_eps.append(row)
            all_episodes.append(row)
            manifest.append(dict(model=model, comparison_episode=i, controller_json=str(raw), source_episode=ep['episode']+1,
                controller_sha256=HASHES[str(raw)], inclusion_policy=entry['inclusion_policy']))
            t = [s['elapsed_s'] for s in steps]
            angles = [qangle(s['goal_errors']['quat']) for s in steps]
            pos = [s['goal_errors']['pos']*1000 for s in steps]
            if errors:
                t.append(duration); angles.append(qangle(errors['quat'])); pos.append(errors['pos']*1000)
            curve_data.append(dict(time=t, angle=angles, position=pos, outcome=outcome))
        durations = [e['duration_s'] for e in per_model_eps]
        success_times = [e['success_time_s'] for e in per_model_eps if e['outcome']=='success']
        final_positions = [e['final_position_error_mm'] for e in per_model_eps if e['completed']]
        final_angles = [e['final_orientation_error_deg'] for e in per_model_eps if e['completed']]
        a = np.asarray(per_model_plans)*1000
        args = controller(rows[0][0])['args']
        m = dict(model=model, evaluated_episodes=len(rows), completed_episodes=sum(e['completed'] for e in per_model_eps),
            successes=sum(e['outcome']=='success' for e in per_model_eps), drops=sum(e['outcome']=='failure' for e in per_model_eps),
            timeouts=sum(e['outcome']=='timeout' for e in per_model_eps), planner_failures=sum(e['outcome']=='error' for e in per_model_eps),
            success_rate=sum(e['outcome']=='success' for e in per_model_eps)/len(rows),
            horizon_s=args['time_horizon'], samples=args['n_samples'], noise=args['noise_sigma'], temperature=args['temperature'],
            planning_calls=len(a), mean_plan_ms=float(a.mean()), p95_plan_ms=float(np.percentile(a,95)),
            max_plan_ms=float(a.max()), fraction_plans_over_64ms=float(np.mean(a>64)),
            effective_hz=sum(e['steps'] for e in per_model_eps)/sum(durations),
            mean_episode_duration_s=float(np.mean(durations)), manipulation_wall_time_s=sum(durations),
            mean_success_time_s=float(np.mean(success_times)) if success_times else None,
            median_final_position_error_mm=float(np.median(final_positions)) if final_positions else None,
            median_final_orientation_error_deg=float(np.median(final_angles)) if final_angles else None,
            inclusion_policy=entry['inclusion_policy'])
        all_metrics.append(m)
        by_model[model] = dict(plans=a, episodes=per_model_eps, curves=curve_data)
        for raw in dict.fromkeys(raw for raw, _ in rows):
            d, l = controller(raw), load(raw.parent/'launch.json')
            configurations.append(dict(model=model, source_run=raw.parent.name, arguments=d['args'],
                weights=d['weights'], thresholds=d['success_thresholds'], hardware_config=d['hardware_config'],
                joint_reference=d.get('joint_reference'), rollout_config=d['rollout_config'], planner_config=d['planner_config'],
                tracking_gpu=l['config'].get('tracking_gpu_assignment'), mppi_gpu=l['config'].get('gpu_assignment')))
    assert [m['evaluated_episodes'] for m in all_metrics] == [10, 10, 10, 5, 10]
    assert [m['successes'] for m in all_metrics] == [2, 5, 0, 0, 1]
    assert sum(m['completed_episodes'] for m in all_metrics) == 40
    csv_write(REPORT/'metrics.csv', all_metrics)
    csv_write(REPORT/'episodes.csv', all_episodes)
    dump(REPORT/'metrics.json', dict(created_at_utc=now, models=all_metrics,
        definitions=dict(effective_hz='Total published actions divided by total active manipulation wall time; preparation and user waits excluded.',
            planning_time='All recorded calls in selected completed episodes; M4 additionally includes its failed solve in each of five failed attempts.',
            terminal_errors='Episode goal_errors at normal termination; unavailable for M4. Drop errors describe the fallen cube, not pre-drop tracking accuracy.',
            quaternion_angle='degrees(2*asin(sqrt(clip(1-dot(q,goal)^2,0,1))))',
            success_rate='Successes divided by evaluated episodes; M4 denominator is five software-failed attempts, other models use ten normally completed episodes.'),
        limitations=index['configuration_notes']))
    dump(REPORT/'sources.json', dict(created_at_utc=now, episodes=manifest, configurations=configurations,
        raw_controller_hashes=HASHES))

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    x = np.arange(5)
    fig, axes = plt.subplots(1,2, figsize=(13,4.8), layout='constrained')
    bottom = np.zeros(5)
    for key, label, color in [('successes','Success',COLORS['success']), ('drops','Drop',COLORS['failure']),
                              ('timeouts','Timeout',COLORS['timeout']), ('planner_failures','Planner failure',COLORS['error'])]:
        values = np.array([m[key] for m in all_metrics])
        bars = axes[0].bar(x, values, bottom=bottom, label=label, color=color, width=.62)
        if key == 'planner_failures':
            for bar in bars:bar.set_hatch('///')
        for i,v in enumerate(values):
            if v:axes[0].text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white',fontweight='bold')
        bottom += values
    axes[0].set(xticks=x,xticklabels=MODELS,ylabel='Recorded episodes / attempts',ylim=(0,11.5),title='Outcomes (actual counts)')
    axes[0].legend(loc='upper left',fontsize=8,ncols=2)
    rates = [m['success_rate']*100 for m in all_metrics]
    axes[1].bar(x,rates,color=['#4674a2' if i!=3 else COLORS['error'] for i in x],width=.62)
    for i,m in enumerate(all_metrics):
        axes[1].text(i,rates[i]+2,f"{m['successes']}/{m['evaluated_episodes']}",ha='center')
    axes[1].set(xticks=x,xticklabels=MODELS,ylabel='Observed success rate (%)',ylim=(0,65),title='Success frequency')
    fig.suptitle('Hardware cube reorientation · high hand / object geometry accuracy')
    fig.supxlabel('M4: five software-failed attempts, zero completed manipulations. All other models: ten completed episodes.',fontsize=9)
    fig.savefig(REPORT/'outcomes.png',dpi=180);fig.savefig(REPORT/'outcomes.pdf');plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    axes[0].boxplot([by_model[m]['plans'] for m in MODELS],tick_labels=MODELS,whis=(5,95),showfliers=False)
    axes[0].axhline(64,color='#b22d2d',ls='--',label='Configured 64 ms period')
    axes[0].set(ylabel='MPPI call time (ms)',title='Planning latency: median, IQR and 5–95% whiskers')
    axes[0].legend(fontsize=9)
    for i,m in enumerate(all_metrics):
        hz = [e['effective_hz'] for e in by_model[m['model']]['episodes']]
        axes[1].scatter(i+np.linspace(-.14,.14,len(hz)),hz,color='#888888',alpha=.75,s=28)
        axes[1].scatter(i,m['effective_hz'],color='#285a8c',marker='D',s=65,zorder=3)
        axes[1].annotate(f"{m['effective_hz']:.1f}",(i,m['effective_hz']),xytext=(7,5),textcoords='offset points',fontsize=9)
    axes[1].axhline(15.625,color='#b22d2d',ls='--')
    axes[1].set(xticks=x,xticklabels=MODELS,ylabel='Published actions / active second (Hz)',title='Observed control rate')
    axes[1].legend(handles=[Line2D([],[],color='#888888',marker='o',ls='',label='Individual episode / attempt'),
        Line2D([],[],color='#285a8c',marker='D',ls='',label='Total steps / total active duration')],fontsize=8)
    fig.suptitle('Historical hardware timing · GPU load and assignments differed between runs')
    fig.supxlabel('M4 timings include failed attempts and failed solves. Preparation, warmup and user waits excluded.',fontsize=9)
    fig.savefig(REPORT/'timing.png',dpi=180);fig.savefig(REPORT/'timing.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for i,model in enumerate(MODELS):
        episodes=by_model[model]['episodes']
        if model=='M4':
            for ax in axes:ax.text(i,.5,'No completed\nepisodes',ha='center',va='center',transform=ax.get_xaxis_transform(),fontsize=9)
            continue
        for j,e in enumerate(episodes):
            color=COLORS[e['outcome']]
            for ax,key in zip(axes,['final_position_error_mm','final_orientation_error_deg']):
                ax.scatter(i+(j-4.5)*.028,e[key],s=35,color=color,alpha=.85)
        for ax,key in zip(axes,['final_position_error_mm','final_orientation_error_deg']):
            ax.hlines(np.median([e[key] for e in episodes]),i-.26,i+.26,color='black',lw=2)
    for ax in axes:ax.set(xticks=x,xticklabels=MODELS);ax.grid(axis='y',alpha=.18)
    axes[0].axhline(20,color='#555555',ls='--');axes[1].axhline(qangle(.04),color='#555555',ls='--')
    axes[0].set(ylabel='Final position error (mm)',title='Position error at episode termination')
    axes[1].set(ylabel='Final orientation error (degrees)',title='Orientation error at episode termination')
    fig.legend(handles=[Line2D([],[],color=COLORS[k],marker='o',ls='',label=label) for k,label in
        [('success','Success'),('failure','Drop'),('timeout','Timeout')]],loc='outside lower center',ncol=3)
    fig.suptitle('Completed episodes only · black bars: medians · dashed lines: individual success thresholds')
    fig.savefig(REPORT/'terminal_errors.png',dpi=180);fig.savefig(REPORT/'terminal_errors.pdf');plt.close(fig)

    fig,axes=plt.subplots(2,4,figsize=(16,7),layout='constrained',sharey='row')
    for column,model in enumerate(['M1','M2','M3','M6']):
        for curve in by_model[model]['curves']:
            for row,key in enumerate(['angle','position']):
                axes[row,column].plot(curve['time'],curve[key],color=COLORS[curve['outcome']],lw=1,alpha=.65)
        axes[0,column].set_title(model)
        axes[0,column].axhline(qangle(.04),color='#333333',ls='--',lw=1)
        axes[1,column].axhline(20,color='#333333',ls='--',lw=1)
        axes[1,column].set_xlabel('Manipulation time (s)')
        for row in (0,1):axes[row,column].grid(alpha=.15)
    axes[0,0].set_ylabel('Orientation error (degrees)');axes[1,0].set_ylabel('Position error (mm)')
    fig.suptitle('Error trajectories for all 40 completed episodes · outcome colors match the outcome chart')
    fig.supxlabel('Each line ends when its episode terminates. M4 omitted: zero completed episodes. No interpolation across terminated episodes.',fontsize=9)
    fig.savefig(REPORT/'convergence.png',dpi=170);fig.savefig(REPORT/'convergence.pdf');plt.close(fig)

    header = '| Model | Evaluated / completed | Success | Drops | Timeout | Planner failure | Mean / P95 MPPI (ms) | Effective Hz | Mean success time (s) |'
    table = [header, '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in all_metrics:
        st=f"{m['mean_success_time_s']:.2f}" if m['mean_success_time_s'] is not None else '—'
        table.append(f"| {m['model']} | {m['evaluated_episodes']} / {m['completed_episodes']} | {m['successes']} ({m['success_rate']:.0%}) | {m['drops']} | {m['timeouts']} | {m['planner_failures']} | {m['mean_plan_ms']:.1f} / {m['p95_plan_ms']:.1f} | {m['effective_hz']:.2f} | {st} |")
    text = '\n'.join(table)+'\n\n'
    text += 'M1, M2, M3 and M6 each have ten completed episodes. M4 has five distinct software-failed attempts and no completed episodes; no additional five M4 episodes were found. M4 is scored as failed under the earlier explicit selection policy and is not equivalent to a completed ten-episode dataset.\n\n'
    text += 'M6 merges the retained success from 000214 with all nine completed episodes from 001426: one success, eight drops and one timeout. The two previous supervision aborts remain excluded. All original logs and older derived reports are preserved.\n\n'
    text += 'Timing is weighted over recorded calls; effective Hz is total published actions divided by total active manipulation duration. Preparation, compilation and user waits are excluded. M4 includes its five failed calls. Historical GPU assignments/load changed, so these timings do not isolate contact-model computation costs.\n\n'
    text += 'Final error statistics use the terminal feedback sample from normally completed episodes, including drops. No terminal error is fabricated for M4. Success requires position <20 mm, quaternion metric <0.04 (about 23.1 degrees) and object velocity norm <10 at the same check; there is no consecutive-step dwell requirement.\n\n'
    text += 'All selected configurations use 512 samples, one iteration, control period 0.064 s, noise 0.05 and temperature 0.1. Horizons: M1 0.192 s, M2 0.32 s, M3 0.384 s, M4 0.256 s, M6 0.384 s. Full per-source configurations and hashes are in sources.json. Freshness limits and GPU placements differ between some sources.\n\n'
    text += '\n\n'.join(f'![{label}]({name}.png)' for name,label in [('outcomes','Episode outcomes'),('timing','Planning and control timing'),('terminal_errors','Final tracking errors'),('convergence','Error trajectories')])+'\n'
    (REPORT/'report.md').write_text(text)
    for path,digest in HASHES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, 'Raw source modified'
    dump(REPORT/'validation.json',dict(unique_selected_episodes=len(manifest), completed_episodes=40,
        planner_failed_attempts=5, m6_completed=10, m6_excluded_prior_aborts=2,
        expected_counts_verified=True, unique_raw_source_hashes_verified=len(HASHES), raw_sources_unchanged=True))
    print(text.split('M1, M2, M3')[0])
    print('Report:', REPORT)


if __name__ == '__main__':
    main()
