"""Analyze saved hardware telemetry only; do not import a physics engine."""
import csv
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', '/tmp/leap_mass70g_review_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
OLD = ROOT / 'results/friction/hardware/cube_reorient/hand-high_obj-high/M2'
NEW = ROOT / 'results/friction_mass70g/hardware/cube_reorient/hand-high_obj-high/M2'
CONDITIONS = [
    ('Original baseline', .14, 1., [ROOT / 'results/pilot/hardware/cube_reorient/hand-high_obj-high/M2/leap_experiment_20261113_213615_713571']),
    ('Recent baseline', .14, 1., [OLD / 'leap_experiment_20261114_155247_800807']),
    ('Uniform 0.5', .14, .5, [OLD / 'leap_experiment_20261114_153625_636084']),
    ('Plastic 0.4', .14, .4, [OLD / 'leap_experiment_20261114_170114_416393']),
    ('Plastic 0.45, 140g model', .14, .45, [OLD / 'leap_experiment_20261114_171723_473986']),
    ('Plastic 0.45, 70g model', .07, .45, [NEW / ('leap_experiment_' + n) for n in
        ('20261114_174618_647297', '20261114_174819_206109', '20261114_180042_549924')]),
]


def angle(errors):
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(errors, 0, 1))))


def scene_without_mass(xml):
    root = ET.fromstring(xml)
    root.find('.//body[@name="obj"]/geom[@name="obj"]').set('mass', 'IGNORED')
    # The XML writer sometimes spells out the MIME type inferred from .obj.
    # Preserve every mesh's name/file/scale and normalize only that default.
    for mesh in root.findall('./asset/mesh'):
        if mesh.get('file', '').endswith('.obj') and mesh.get('content_type') == 'model/obj':
            del mesh.attrib['content_type']
    return ET.tostring(root)


def main():
    metrics, rows, hashes, series, checks = [], [], {}, [], []
    reference = json.loads((CONDITIONS[4][3][0] / 'controller.json').read_text())
    match_keys = ['weights', 'hardware_config', 'joint_reference', 'planner_config',
        'rollout_config', 'pose_transform', 'success_thresholds', 'effective_ctrl_range',
        'task_contact_params', 'failure_z', 'goal_policy', 'execution', 'action_rate_cost']
    for label, mass, plastic, directories in CONDITIONS:
        group_rows, times, masks = [], [], []
        for directory in directories:
            path = directory / 'controller.json'
            raw = path.read_bytes()
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
            d = json.loads(raw)
            if mass == .07:
                matching = {k: d[k] == reference[k] for k in match_keys}
                assert all(matching.values()), matching
                scenes_match = {role: scene_without_mass(d['scenes'][role]['scene_xml']) ==
                    scene_without_mass(reference['scenes'][role]['scene_xml']) for role in ('rollout', 'eval')}
                assert all(scenes_match.values()), scenes_match
                checks.append(dict(run=directory.name, identical_configuration_fields=matching,
                    scene_xml_identical_except_cube_mass_and_inferred_obj_mime=scenes_match,
                    run_status=d['status'], run_error=d.get('error')))
            for role in ('rollout', 'eval'):
                root = ET.fromstring(d['scenes'][role]['scene_xml'])
                assert float(root.find('.//body[@name="obj"]/geom[@name="obj"]').get('mass')) == mass
            down = np.asarray(d['effective_ctrl_range']['lower'])
            up = np.asarray(d['effective_ctrl_range']['upper'])
            for e in d['episodes']:
                if not e.get('completed'):
                    continue
                steps = e['steps']
                q = np.asarray([s['q'] for s in steps] + [e['q_end']])
                err = [s['goal_errors'] for s in steps] + [e['goal_errors']]
                a = angle([s['quat'] for s in err])
                pos = np.asarray([s['pos'] for s in err])
                t = np.asarray([s['elapsed_s'] for s in steps] + [e['manipulation_wall_time_s']])
                best = int(np.argmin(a))
                valid = pos < d['success_thresholds']['pos']
                in_orientation = np.asarray([s['quat'] for s in err]) < d['success_thresholds']['quat']
                delta = np.asarray([s['action_delta'] for s in steps])
                hit = (delta < -down - 1e-6) | (delta > up + 1e-6)
                times.extend(s['planning_time'] for s in steps)
                masks.extend(hit)
                w, x, y, z = q[0, 19:23] / np.linalg.norm(q[0, 19:23])
                rz = np.asarray([2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2*(x*x + y*y)])
                axis = int(np.argmax(np.abs(rz)))
                row = dict(condition=label, model_mass_kg=mass, plastic_friction=plastic,
                    run=directory.name, episode=e['episode']+1, outcome=e['finish_reason'],
                    initial_up_face=('+' if rz[axis]>0 else '-')+'XYZ'[axis],
                    steps=e['n_steps'], duration_s=e['manipulation_wall_time_s'],
                    initial_orientation_error_deg=float(a[0]), best_orientation_error_deg=float(a[best]),
                    time_at_best_s=float(t[best]), position_error_at_best_mm=float(pos[best]*1000),
                    best_orientation_with_position_valid_deg=float(a[valid].min()) if valid.any() else None,
                    final_orientation_error_deg=float(a[-1]), final_position_error_mm=float(pos[-1]*1000),
                    final_object_z_m=float(q[-1,18]), both_pose_thresholds_met=bool(np.any(valid & in_orientation)),
                    planning_mean_ms=float(np.mean([s['planning_time'] for s in steps])*1000),
                    clipped_update_percent=float(hit.any(axis=1).mean()*100),
                    clipped_joint_component_percent=float(hit.mean()*100))
                rows.append(row)
                group_rows.append(row)
                if mass == .07:
                    series.append((row, t, a, pos*1000))
        counts = {name: sum(r['outcome'] == name for r in group_rows) for name in ('success', 'failure', 'timeout')}
        metrics.append(dict(condition=label, model_mass_g=mass*1000,
            effective_tip_friction=.5, effective_plastic_friction=plastic,
            completed=len(group_rows), successes=counts['success'], drops=counts['failure'], timeouts=counts['timeout'],
            median_best_orientation_error_deg=float(np.median([r['best_orientation_error_deg'] for r in group_rows])),
            median_best_orientation_with_position_valid_deg=float(np.median([r['best_orientation_with_position_valid_deg'] for r in group_rows if r['best_orientation_with_position_valid_deg'] is not None])),
            planning_mean_ms=float(np.mean(times)*1000), planning_p95_ms=float(np.percentile(times,95)*1000),
            effective_hz=len(masks)/sum(r['duration_s'] for r in group_rows),
            clipped_update_percent=float(np.asarray(masks).any(axis=1).mean()*100),
            clipped_joint_component_percent=float(np.asarray(masks).mean()*100)))
    assert len(series) == 6
    assert [sum(r['run']==p.name for r in rows if r['model_mass_kg']==.07) for p in CONDITIONS[-1][3]] == [1,4,1]
    assert len(rows) == 32
    for name, data in [('metrics', metrics), ('episodes', rows)]:
        with (OUT / (name+'.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    threshold = float(angle(reference['success_thresholds']['quat']))
    fig, axes = plt.subplots(1, 2, figsize=(13,5), layout='constrained')
    x = np.arange(len(metrics)); bottom = np.zeros(len(metrics))
    for key, label, color in [('successes','Success','#29936d'), ('drops','Logged drop','#ce614d'), ('timeouts','Timeout','#d5a132')]:
        values = np.asarray([m[key]/m['completed']*100 for m in metrics])
        axes[0].bar(x, values, bottom=bottom, label=label, color=color)
        for i, value in enumerate(values):
            if value:
                axes[0].text(i,bottom[i]+value/2,str(metrics[i][key]),ha='center',va='center',color='white')
        bottom += values
    labels = ['Original\n140g / 1.0\nn=10','Recent\n140g / 1.0\nn=6','140g / 0.5\nn=2','140g / 0.4\nn=6','140g / 0.45\nn=2','70g / 0.45\nn=6']
    axes[0].set(xticks=x, xticklabels=labels, ylabel='Episode outcomes (%)', ylim=(0,112), title='Model mass / plastic friction; counts inside bars')
    axes[0].tick_params(axis='x', labelsize=8)
    axes[0].legend(ncol=3, fontsize=8, loc='upper center')
    for i, m in enumerate(metrics):
        values = [r['best_orientation_with_position_valid_deg'] for r in rows if r['condition']==m['condition']]
        axes[1].scatter(np.full(len(values),i),values,alpha=.6,s=25)
    axes[1].plot(x,[m['median_best_orientation_with_position_valid_deg'] for m in metrics], 'k_',ms=22,label='Median')
    axes[1].axhline(threshold,color='gray',ls='--',label='Orientation success threshold')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Degrees',title='Best orientation while position error <20 mm')
    axes[1].tick_params(axis='x',labelsize=8)
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(axis='y',alpha=.2)
    fig.suptitle('Recorded M2 hardware episodes; actual cube mass 70 g, internal model mass varies')
    fig.savefig(OUT/'comparison.png',dpi=170)
    fig.savefig(OUT/'comparison.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(13,7),layout='constrained',sharey=True)
    for i, (row, t, a, pos) in enumerate(series):
        ax=axes.flat[i]
        ax.plot(t,a,label='Orientation error',color='#3179b5')
        ax.axhline(threshold,color='gray',ls='--')
        ax.scatter(t[pos<20],a[pos<20],s=3,color='#29936d',label='Position within 20 mm',zorder=3)
        ax.set(title=f"E{i+1} ({row['initial_up_face']}): {row['outcome']}",xlabel='Seconds',ylabel='Orientation error (deg)',ylim=(0,185))
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Six completed episodes with corrected 70 g model and plastic friction 0.45')
    fig.savefig(OUT/'corrected_mass_episodes.png',dpi=170)
    fig.savefig(OUT/'corrected_mass_episodes.pdf')
    plt.close(fig)
    result=dict(metrics=metrics,corrected_mass_episodes=[s[0] for s in series],
        configuration_comparisons=checks,source_sha256=hashes,
        old_vs_corrected_mass_note='All cubes on hardware were reported as 70 g; old internal models used 140 g. Historical data remain valid outcomes of a controller with a mass mismatch, not trials with a physically heavier cube.',
        recommendation=dict(cube=[.3,.005,.0001],fingertips=[.5,.005,.0001],palm=[.4,.005,.0001],other_fingers=[.4,.005,.0001],
            model_mass_kg=.07,status='Unvalidated next candidate, not identified physical friction.',
            rationale='Correcting mass did not produce success at plastic 0.45. Compare plastic 0.4 against 0.45 at fixed corrected mass; 0.4 is the adjacent previously tried value with reorientation successes under the old mass-mismatched model. No evidence identifies new fingertip/torsional/rolling coefficients.'),
        limitations=['No logged contact forces/identities or simulated predicted object trajectory. Cannot uniquely estimate friction or attribute observed drops to a particular contact.',
            'Small unequal samples and manually varying starting poses; descriptive comparisons only.',
            'No 70 g-model baseline at plastic 1.0 or 0.4 exists in this selected data.',
            'Final hardware target clipping affects almost every update and is a separate planning/execution discrepancy.'],
        simulations_run=False,hardware_run=False)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for path, digest in hashes.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
