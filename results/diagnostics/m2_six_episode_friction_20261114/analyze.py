"""Logged-data analysis only. No simulator, planner, camera, LCM or hardware imports."""
import csv
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR','/tmp/leap_friction_analysis_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
RUN=ROOT/'results/friction/hardware/cube_reorient/hand-high_obj-high/M2/leap_experiment_20261114_155247_800807'
PREVIOUS=RUN.parent/'leap_experiment_20261114_153625_636084'


def angle(error):
    return np.degrees(2*np.arcsin(np.sqrt(np.clip(error,0,1))))


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))


def friction_in_saved_xml(text, name):
    root=ET.fromstring(text)
    defaults={}
    def walk(node, inherited):
        attrs=dict(inherited)
        geom=node.find('geom')
        if geom is not None:attrs.update(geom.attrib)
        defaults[node.get('class','main')]=attrs
        for child in node.findall('default'):walk(child,attrs)
    walk(root.find('default'),{})
    geom=root.find(f".//worldbody//geom[@name='{name}']")
    attrs={**defaults.get('main',{}),**defaults.get(geom.get('class','main'),{}),**geom.attrib}
    result=np.array([1.,.005,.0001])
    values=[float(x) for x in attrs.get('friction','').split()]
    result[:len(values)]=values
    return result.tolist()


def main():
    source=RUN/'controller.json'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    data=json.loads(source.read_text())
    prior=json.loads((PREVIOUS/'controller.json').read_text())
    assert len(data['episodes'])==6 and all(e['completed'] for e in data['episodes'])
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    rows=[];all_delta=[];all_plans=[];all_clip=[];robot_ages=[];pose_ages=[];target_changes=[]
    colors=plt.get_cmap('tab10').colors
    down,up=np.array(data['effective_ctrl_range']['lower']),np.array(data['effective_ctrl_range']['upper'])
    for e in data['episodes']:
        steps=e['steps'];q=np.array([s['q'] for s in steps]);t=np.array([s['elapsed_s'] for s in steps])
        end=e['goal_errors'];times=np.r_[t,e['manipulation_wall_time_s']]
        angles=angle([s['goal_errors']['quat'] for s in steps]+[end['quat']])
        positions=np.array([s['goal_errors']['pos'] for s in steps]+[end['pos']])*1000
        heights=np.r_[q[:,18],e['q_end'][18]]*1000
        delta=np.array([s['action_delta'] for s in steps])
        clipped=(delta < -down-1e-6)|(delta > up+1e-6)
        plans=np.array([s['planning_time'] for s in steps])*1000
        changes=np.abs(np.diff(np.array([s['u'] for s in steps]),axis=0))
        best=int(np.argmin(angles));outcome=e['finish_reason']
        row=dict(episode=e['episode']+1,outcome=outcome,steps=e['n_steps'],duration_s=e['manipulation_wall_time_s'],
            initial_position_m=q[0,16:19].tolist(),initial_quaternion_wxyz=q[0,19:23].tolist(),
            initial_orientation_error_deg=float(angles[0]),best_orientation_error_deg=float(angles[best]),
            best_orientation_time_s=float(times[best]),position_error_at_best_orientation_mm=float(positions[best]),
            final_orientation_error_deg=float(angles[-1]),final_position_error_mm=float(positions[-1]),
            final_cube_z_mm=float(heights[-1]),clipped_updates=int(clipped.any(axis=1).sum()),
            clipped_update_percent=float(clipped.any(axis=1).mean()*100),
            clipped_joint_component_percent=float(clipped.mean()*100),
            mean_plan_ms=float(plans.mean()),p95_plan_ms=float(np.percentile(plans,95)),
            effective_hz=e['n_steps']/e['manipulation_wall_time_s'])
        rows.append(row)
        all_delta.extend(delta);all_plans.extend(plans);all_clip.extend(clipped);target_changes.extend(changes.ravel())
        robot_ages.extend(s['feedback']['robot_age']*1000 for s in steps)
        pose_ages.extend(s['feedback']['pose_age']*1000 for s in steps)
        color=colors[e['episode']]
        label=f"E{e['episode']+1}: {outcome}"
        for ax,y in zip(axes.flat[:3],[angles,positions,heights]):
            ax.plot(times,y,color=color,lw=1.2,label=label)
    axes[0,0].axhline(angle(.04),color='black',ls='--',lw=1)
    axes[0,0].set(title='Orientation',xlabel='Manipulation time (s)',ylabel='Orientation error (degrees)')
    axes[0,0].legend(fontsize=8,ncols=2)
    axes[0,1].axhline(20,color='black',ls='--',lw=1)
    axes[0,1].set(title='Position retention',xlabel='Manipulation time (s)',ylabel='Position error (mm)')
    axes[1,0].axhline(65,color='#a87519',ls='--',lw=1,label='Fallen-cost height')
    axes[1,0].axhline(0,color='#b22d2d',ls='--',lw=1,label='Drop termination height')
    axes[1,0].set(title='Tracked cube center height',xlabel='Manipulation time (s)',ylabel='World Z (mm)')
    axes[1,0].legend(fontsize=8)
    axes[1,1].bar(np.arange(1,7),[r['clipped_update_percent'] for r in rows],color=colors[:6])
    axes[1,1].set(title='Final-target clipping at ±0.1 rad',xlabel='Episode',ylabel='Updates with ≥1 joint clipped (%)',ylim=(0,105),xticks=np.arange(1,7))
    for ax in axes.flat:ax.grid(alpha=.18)
    fig.suptitle('M2 · six completed hardware episodes · logged data only, no simulation replay')
    fig.savefig(OUT/'analysis.png',dpi=170);fig.savefig(OUT/'analysis.pdf');plt.close(fig)
    friction={role:{geom:friction_in_saved_xml(scene['scene_xml'],geom) for geom in
        ('obj','if_tip','mf_tip','rf_tip','th_tip','palm_collision_1','if_px_collision')} for role,scene in data['scenes'].items()}
    previous_friction={geom:friction_in_saved_xml(prior['scenes']['rollout']['scene_xml'],geom) for geom in
        ('obj','if_tip','palm_collision_1','if_px_collision')}
    clipped=np.array(all_clip);duration=sum(r['duration_s'] for r in rows)
    summary=dict(run=str(RUN),episodes=6,successes=1,drops=4,timeouts=1,aborted_episodes=0,
        success_rate=1/6,published_updates=len(clipped),clipped_updates=int(clipped.any(axis=1).sum()),
        clipped_update_percent=float(clipped.any(axis=1).mean()*100),
        clipped_joint_component_percent=float(clipped.mean()*100),planning_ms=stats(all_plans),
        effective_hz=len(clipped)/duration,manipulation_duration_s=duration,
        robot_age_ms=stats(robot_ages),pose_age_ms=stats(pose_ages),
        target_component_change_rad=stats(target_changes),
        saved_scene_friction=friction,previous_two_episode_scene_friction=previous_friction,
        previous_two_episode_outcomes=[e['finish_reason'] for e in prior['episodes']],
        previous_run=str(PREVIOUS),source_sha256=digest,simulation_tests_run=False,
        recommendation=dict(status='Unvalidated engineering hypothesis; not identified parameters',
            cube=[.3,.005,.0001],fingertip=[.5,.005,.0001],palm=[.4,.005,.0001],finger=[.4,.005,.0001],
            expected_effective_sliding=dict(cube_fingertip=.5,cube_palm=.4,cube_other_finger=.4)),
        interpretation_limits=[
            'No predicted object rollouts, contact identities or measured contact forces were logged; prediction residuals and individual pair friction cannot be identified from these records.',
            'Different initial placements and historical GPU loads prevent a controlled causal comparison with the earlier two episodes or ten-episode baseline.',
            'Near-universal final-target clipping is an additional planning/execution mismatch independent of friction.',
            'Suggested values preserve the tested fingertip coefficient and reduce the plastic coefficient; their exact magnitudes are hypotheses, not estimates.'])
    (OUT/'analysis.json').write_text(json.dumps(dict(summary=summary,episodes=rows),indent=2)+'\n')
    scalar_rows=[{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]
    with (OUT/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(scalar_rows[0]));writer.writeheader();writer.writerows(scalar_rows)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
