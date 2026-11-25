"""Compare recorded friction trials only; never instantiate a physics engine."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/leap_friction_refinement_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
FRICTION=ROOT/'results/friction/hardware/cube_reorient/hand-high_obj-high/M2'
RUNS=[
    ('Original comparison',ROOT/'results/pilot/hardware/cube_reorient/hand-high_obj-high/M2/leap_experiment_20261113_213615_713571',1.0),
    ('Recent baseline',FRICTION/'leap_experiment_20261114_155247_800807',1.0),
    ('Uniform 0.5',FRICTION/'leap_experiment_20261114_153625_636084',.5),
    ('Updated 0.4',FRICTION/'leap_experiment_20261114_170114_416393',.4),
]


def main():
    metrics=[];rows=[];loaded={};hashes={}
    for label,path,plastic in RUNS:
        raw=path/'controller.json'
        hashes[str(raw)]=hashlib.sha256(raw.read_bytes()).hexdigest()
        d=json.loads(raw.read_text());loaded[label]=d
        episodes=[e for e in d['episodes'] if e.get('completed')]
        times=[];clips=[];durations=[];best_angles=[];success_times=[]
        for e in episodes:
            steps=e['steps'];q=np.array([s['q'] for s in steps]);angles=np.degrees(2*np.arcsin(np.sqrt(np.clip(
                [s['goal_errors']['quat'] for s in steps]+[e['goal_errors']['quat']],0,1))))
            delta=np.array([s['action_delta'] for s in steps])
            lo,hi=np.array(d['effective_ctrl_range']['lower']),np.array(d['effective_ctrl_range']['upper'])
            hit=(delta < -lo-1e-6)|(delta > hi+1e-6)
            times.extend(s['planning_time'] for s in steps);clips.extend(hit)
            durations.append(e['manipulation_wall_time_s']);best_angles.append(float(angles.min()))
            if e['finish_reason']=='success':success_times.append(e['manipulation_wall_time_s'])
            w,x,y,z=q[0,19:23]/np.linalg.norm(q[0,19:23])
            rz=np.array([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)])
            axis=int(np.argmax(np.abs(rz)))
            rows.append(dict(condition=label,episode=e['episode']+1,outcome=e['finish_reason'],
                steps=e['n_steps'],duration_s=e['manipulation_wall_time_s'],
                initial_up_face=('+' if rz[axis]>0 else '-')+'XYZ'[axis],
                initial_orientation_error_deg=float(angles[0]),best_orientation_error_deg=float(angles.min()),
                final_orientation_error_deg=float(angles[-1]),final_position_error_mm=e['goal_errors']['pos']*1000))
        mask=np.array(clips)
        outcomes=[e['finish_reason'] for e in episodes]
        metrics.append(dict(condition=label,run=str(path),effective_tip_friction=.5,effective_plastic_friction=plastic,
            completed=len(episodes),successes=outcomes.count('success'),drops=outcomes.count('failure'),
            timeouts=outcomes.count('timeout'),success_rate=outcomes.count('success')/len(episodes),
            median_best_orientation_error_deg=float(np.median(best_angles)),
            mean_success_time_s=float(np.mean(success_times)) if success_times else None,
            planning_mean_ms=float(np.mean(times)*1000),effective_hz=len(mask)/sum(durations),
            clipped_update_percent=float(mask.any(axis=1).mean()*100),
            clipped_joint_component_percent=float(mask.mean()*100)))
    a,b=loaded['Recent baseline'],loaded['Updated 0.4']
    keys=['weights','hardware_config','joint_reference','planner_config','rollout_config',
          'success_thresholds','pose_transform','task_contact_params']
    assert all(a.get(k)==b.get(k) for k in keys)
    assert all(x['goal']==y['goal'] and x['target_pos']==y['target_pos'] for x,y in zip(a['episodes'],b['episodes']))
    baseline=[r for r in rows if r['condition']=='Recent baseline']
    updated=[r for r in rows if r['condition']=='Updated 0.4']
    assert [r['initial_up_face'] for r in baseline]==[r['initial_up_face'] for r in updated]
    assert all(u['best_orientation_error_deg'] < r['best_orientation_error_deg'] for r,u in zip(baseline,updated))
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    x=np.arange(4);bottom=np.zeros(4)
    for key,label,color in [('successes','Success','#29936d'),('drops','Logged drop','#ce614d'),('timeouts','Timeout','#d5a132')]:
        vals=np.array([m[key] for m in metrics]);axes[0].bar(x,vals,bottom=bottom,color=color,label=label)
        for i,v in enumerate(vals):
            if v:axes[0].text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white',fontweight='bold')
        bottom+=vals
    axes[0].set(xticks=x,xticklabels=['Original\nplastic 1.0','Recent\nplastic 1.0','Plastic 0.5','Plastic 0.4'],ylabel='Completed episodes',ylim=(0,12),title='Hardware outcomes; unequal sample counts')
    axes[0].legend(fontsize=8,ncol=3,loc='upper center')
    for i,(r,u) in enumerate(zip(baseline,updated)):
        axes[1].plot([1,0.4],[r['best_orientation_error_deg'],u['best_orientation_error_deg']],'-o',label=f"E{i+1}: {r['initial_up_face']}")
    axes[1].axhline(np.degrees(2*np.arcsin(.2)),color='black',ls='--',lw=1,label='Orientation threshold')
    axes[1].set(xticks=[1,.4],xticklabels=['Recent baseline\nplastic 1.0','Updated\nplastic 0.4'],xlim=(1.12,.28),ylabel='Best achieved orientation error (degrees)',title='Corresponding initial cube orientations')
    axes[1].legend(fontsize=8,ncol=2)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Recorded M2 hardware data only · fingertip coefficient held at 0.5')
    fig.supxlabel('Episode pairing is descriptive: placements and control timing differed. No simulation or hardware tests performed.',fontsize=9)
    fig.savefig(OUT/'comparison.png',dpi=170);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    for name,data in [('metrics',metrics),('episodes',rows)]:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    report=dict(metrics=metrics,matching_configuration_fields=keys,source_hashes=hashes,
        simulation_tests_run=False,hardware_tests_run=False,
        interpretation='Lower plastic friction is consistent with improved reorientation in the recent six-versus-six comparison. Four drops remain in both. The original ten-episode baseline achieved 5/10 successes, so updated friction is not established as superior overall.',
        refinement=dict(best_recent_tested_effective_friction=dict(fingertips=.5,palm=.4,other_fingers=.4),
            next_unvalidated_candidate=dict(cube=.3,fingertips=.5,palm=.45,other_fingers=.45),
            rationale='Probe the interval between 0.4 (better reorientation, four drops in six) and 0.5 (two retained but stalled episodes). This midpoint is a search choice, not an identified physical estimate. Keep fingertip coefficient fixed and palm/finger coefficients tied because those effects have not been identified independently.'),
        limitations=['No contact identities/forces or predicted object trajectories.',
            'Small, unequal trial counts; different initial placements, tracking sessions and effective control rates.',
            'Near-universal final-target clipping remains a separate planning/execution mismatch.'])
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
