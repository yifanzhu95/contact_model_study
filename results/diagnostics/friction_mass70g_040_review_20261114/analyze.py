"""Review recorded hardware logs only; never import or run a simulator."""
import csv
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', '/tmp/leap_friction040_review_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
PREVIOUS = ROOT / 'results/diagnostics/friction_mass70g_review_20261114/analysis.json'
RUN = ROOT / 'results/friction_mass70g/hardware/cube_reorient/hand-high_obj-high/M2/leap_experiment_20261114_182513_421747/controller.json'
REFERENCE = RUN.parent.parent / 'leap_experiment_20261114_180042_549924/controller.json'


def angle(values):
    return np.degrees(2*np.arcsin(np.sqrt(np.clip(values, 0, 1))))


def normalized_scene(xml):
    root = ET.fromstring(xml)
    for mesh in root.findall('./asset/mesh'):
        if mesh.get('file', '').endswith('.obj') and mesh.get('content_type') == 'model/obj':
            del mesh.attrib['content_type']
    return root


def main():
    previous = json.loads(PREVIOUS.read_text())
    hashes = dict(previous['source_sha256'])
    for path, digest in hashes.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
    raw = RUN.read_bytes()
    hashes[str(RUN.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    d = json.loads(raw)
    reference = json.loads(REFERENCE.read_text())
    keys = ['weights', 'hardware_config', 'joint_reference', 'planner_config',
        'rollout_config', 'pose_transform', 'success_thresholds', 'effective_ctrl_range',
        'task_contact_params', 'failure_z', 'goal_policy', 'execution', 'action_rate_cost']
    matched = {k: d[k] == reference[k] for k in keys}
    assert all(matched.values()), matched
    scene_checks = {}
    for role in ('rollout', 'eval'):
        a, b = [normalized_scene(x['scenes'][role]['scene_xml']) for x in (reference, d)]
        assert b.find('.//body[@name="obj"]/geom[@name="obj"]').get('mass') == '0.07'
        changes = []
        old, new = a.findall('.//geom'), b.findall('.//geom')
        assert len(old) == len(new)
        for before, after in zip(old, new):
            if before.get('friction') != after.get('friction'):
                assert before.get('name') == after.get('name')
                assert before.get('friction') == '0.45' and after.get('friction') == '0.4'
                assert after.get('name').startswith(('palm_', 'if_', 'mf_', 'rf_', 'th_'))
                changes.append(after.get('name'))
            before.attrib.pop('friction', None)
            after.attrib.pop('friction', None)
        assert ET.tostring(a) == ET.tostring(b)
        scene_checks[role] = dict(other_scene_fields_identical=True, changed_friction_geoms=changes)
    label = 'Plastic 0.4, 70g model'
    rows, series, times, masks, clocks = [], [], [], [], []
    down, up = [np.asarray(d['effective_ctrl_range'][k]) for k in ('lower', 'upper')]
    episodes = [e for e in d['episodes'] if e.get('completed')]
    assert len(episodes) == 6
    for e in episodes:
        steps = e['steps']
        q = np.asarray([s['q'] for s in steps] + [e['q_end']])
        errors = [s['goal_errors'] for s in steps] + [e['goal_errors']]
        t = np.asarray([s['elapsed_s'] for s in steps] + [e['manipulation_wall_time_s']])
        a = angle([s['quat'] for s in errors])
        pos = np.asarray([s['pos'] for s in errors])
        valid = pos < .02
        best = int(np.argmin(a))
        delta = np.asarray([s['action_delta'] for s in steps])
        assert np.isfinite(delta).all() and np.isfinite(q).all()
        hits = (delta < -down-1e-6) | (delta > up+1e-6)
        times.extend(s['planning_time'] for s in steps)
        masks.extend(hits)
        w, x, y, z = q[0, 19:23] / np.linalg.norm(q[0, 19:23])
        rz = np.asarray([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
        axis = int(np.argmax(np.abs(rz)))
        row = dict(condition=label, model_mass_kg=.07, plastic_friction=.4,
            run=RUN.parent.name, episode=e['episode']+1, outcome=e['finish_reason'],
            initial_up_face=('+' if rz[axis]>0 else '-')+'XYZ'[axis],
            initial_object_xyz=q[0,16:19].tolist(), initial_object_quat_wxyz=q[0,19:23].tolist(),
            steps=e['n_steps'], duration_s=e['manipulation_wall_time_s'],
            initial_orientation_error_deg=float(a[0]), best_orientation_error_deg=float(a[best]),
            position_error_at_best_mm=float(pos[best]*1000),
            best_orientation_with_position_valid_deg=float(a[valid].min()),
            final_orientation_error_deg=float(a[-1]), final_position_error_mm=float(pos[-1]*1000),
            planning_mean_ms=float(np.mean([s['planning_time'] for s in steps])*1000),
            clipped_update_percent=float(hits.any(axis=1).mean()*100),
            clipped_joint_component_percent=float(hits.mean()*100))
        rows.append(row)
        series.append((t, a, pos))
    counts = {key: sum(e['finish_reason']==key for e in episodes) for key in ('success','failure','timeout')}
    latest = dict(condition=label, model_mass_g=70., effective_tip_friction=.5, effective_plastic_friction=.4,
        completed=6, successes=counts['success'], drops=counts['failure'], timeouts=counts['timeout'],
        median_best_orientation_error_deg=float(np.median([r['best_orientation_error_deg'] for r in rows])),
        median_best_orientation_with_position_valid_deg=float(np.median([r['best_orientation_with_position_valid_deg'] for r in rows])),
        planning_mean_ms=float(np.mean(times)*1000), planning_p95_ms=float(np.percentile(times,95)*1000),
        effective_hz=len(masks)/sum(e['manipulation_wall_time_s'] for e in episodes),
        clipped_update_percent=float(np.asarray(masks).any(axis=1).mean()*100),
        clipped_joint_component_percent=float(np.asarray(masks).mean()*100))
    metrics = previous['metrics'] + [latest]
    old_rows = previous['corrected_mass_episodes']
    assert [r['initial_up_face'] for r in rows] == [r['initial_up_face'] for r in old_rows]
    paired = [dict(episode=i+1, face=a['initial_up_face'],
        old_outcome=a['outcome'], new_outcome=b['outcome'],
        old_best_valid_deg=a['best_orientation_with_position_valid_deg'],
        new_best_valid_deg=b['best_orientation_with_position_valid_deg']) for i,(a,b) in enumerate(zip(old_rows,rows))]
    with (OUT/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(latest));writer.writeheader();writer.writerows(metrics)
    with (OUT/'latest_episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (OUT/'corresponding_faces.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    threshold = float(angle(d['success_thresholds']['quat']))
    fig, axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    groups = [metrics[-2],latest]
    bottom=np.zeros(2)
    for key, text, color in [('successes','Success','#29936d'),('drops','Logged drop','#ce614d'),('timeouts','Timeout','#d5a132')]:
        values=np.asarray([m[key] for m in groups])
        axes[0].bar([0,1],values,bottom=bottom,label=text,color=color)
        for i,value in enumerate(values):
            if value:axes[0].text(i,bottom[i]+value/2,str(value),ha='center',va='center',color='white')
        bottom+=values
    axes[0].set(xticks=[0,1],xticklabels=['Plastic 0.45','Plastic 0.40'],ylabel='Completed episodes',ylim=(0,7.5),title='Both use corrected 70 g internal model')
    axes[0].legend(ncol=3,fontsize=8)
    for row in paired:
        axes[1].plot([0,1],[row['old_best_valid_deg'],row['new_best_valid_deg']],'-o',label=f"E{row['episode']}: {row['face']}")
    axes[1].axhline(threshold,color='gray',ls='--',label='Orientation threshold')
    axes[1].set(xticks=[0,1],xticklabels=['Plastic 0.45','Plastic 0.40'],ylabel='Orientation error (degrees)',title='Best orientation with position error <20 mm',xlim=(-.15,1.15))
    axes[1].legend(fontsize=8,ncol=2)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('M2 hardware logs only; fingertip friction 0.5; six episodes per setting')
    fig.supxlabel('Corresponding faces are descriptive; placements, tracking and timing are not controlled pairs.',fontsize=8)
    for ext in ('png','pdf'):fig.savefig(OUT/('comparison.'+ext),dpi=170)
    plt.close(fig)
    fig, axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained',sharey=True)
    for ax,row,(t,a,pos) in zip(axes.flat,rows,series):
        ax.plot(t,a,color='#3179b5')
        ax.scatter(t[pos<.02],a[pos<.02],s=3,color='#29936d')
        ax.axhline(threshold,color='gray',ls='--')
        ax.set(title=f"E{row['episode']} ({row['initial_up_face']}): {row['outcome']}",xlabel='Seconds',ylabel='Orientation error (degrees)',ylim=(0,185))
        ax.grid(alpha=.2)
    fig.suptitle('Latest 70 g / plastic 0.4 run; green points have position error <20 mm')
    for ext in ('png','pdf'):fig.savefig(OUT/('latest_episodes.'+ext),dpi=170)
    plt.close(fig)
    result=dict(metrics=metrics,latest_episodes=rows,corresponding_faces=paired,
        matching_configuration_fields=matched,scene_comparison=scene_checks,source_sha256=hashes,
        recommendation=dict(model_mass_kg=.07,cube=[.3,.005,.0001],fingertips=[.5,.005,.0001],
            palm=[.4,.005,.0001],other_fingers=[.4,.005,.0001],
            status='Retain as best observed 70 g-model candidate among tested 0.4/0.45 settings; not an identified physical coefficient.',
            rationale='Two successes and two drops at 0.4 versus zero successes and three drops at 0.45; four of six corresponding-face episodes improved best orientation while position was valid. No data below 0.4, no independent fingertip/palm/finger variation, and small manually placed samples.'),
        simulation_runs=False,hardware_runs=False)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for path,digest in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    print(json.dumps(dict(latest=latest,episodes=rows,paired=paired),indent=2))


if __name__=='__main__':
    main()
